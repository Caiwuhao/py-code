import numpy as np
from scipy.integrate import quad
from scipy.linalg import svd
from types import SimpleNamespace
from unittest import SkipTest
from unittest.mock import patch

from opln_pso.model import (
    build_geometry,
    cumulative_erf_target,
    domain_table,
    evaluate_jsa_from_pmf,
    jsa_wavelength_grid,
    material_mismatch,
    pmf_from_duties,
    qute_3500_geometry,
)
from opln_pso.optimizer import (
    PMFObjective, PSOConfig, _load_backend, optimize_duty_cycles,
    polish_duty_cycles_lbfgsb,
)


def test_qute_geometry():
    geometry = qute_3500_geometry()
    assert np.isclose(geometry.qpm_period_um, 15.497)
    assert np.isclose(geometry.calculated_qpm_period_um, 15.50037043769857)
    assert geometry.number_periods == 1935


def test_periodic_peak_is_two_over_pi():
    geometry = qute_3500_geometry()
    duties = np.full(geometry.number_periods, 0.5)
    peak = pmf_from_duties(
        duties, geometry.qpm_period_um, [2.0 * np.pi / geometry.qpm_period_um]
    )[0]
    assert np.isclose(abs(peak), 2.0 / np.pi, rtol=2e-4)


def test_pmf_matches_independent_domain_integral():
    period = 15.497
    duties = np.array([0.17, 0.61, 0.43, 0.82, 0.29])
    mismatch = np.array([0.39, 0.405, 0.42])
    actual = pmf_from_duties(duties, period, mismatch)
    domains = domain_table(duties, period)
    expected = []
    for value in mismatch:
        integral = np.sum(
            domains[:, 1]
            * (
                np.exp(-1j * value * domains[:, 2])
                - np.exp(-1j * value * domains[:, 3])
            )
            / (1j * value)
        )
        expected.append(integral / (duties.size * period))
    assert np.allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_erf_target_boundaries():
    geometry = qute_3500_geometry()
    values = cumulative_erf_target(
        [0.0, geometry.crystal_length_um],
        geometry.crystal_length_um,
        geometry.target_sigma_z_um,
    )
    assert np.allclose(values, [0.0, 1.0], atol=1e-12)


def test_minimum_domain_constraint():
    geometry = qute_3500_geometry(min_domain_um=0.5)
    duties = np.linspace(geometry.duty_min, geometry.duty_max, geometry.number_periods)
    assert domain_table(duties, geometry.qpm_period_um)[:, 4].min() >= 0.5 - 1e-12


def test_objective_wavelength_slice_center():
    geometry = qute_3500_geometry()
    objective = PMFObjective(geometry, q_points=101, device="cpu")
    assert np.isclose(objective.signal_wavelengths_um[0], 3.47)
    assert np.isclose(objective.signal_wavelengths_um[-1], 3.53)
    assert np.isclose(objective.q_rad_um[50], 0.0, atol=1e-14)


def test_jsa_axis_order():
    geometry = qute_3500_geometry()
    signal, idler, signal_grid, idler_grid = jsa_wavelength_grid(geometry, 40, 60)
    assert np.allclose(signal_grid[0, :], signal)
    assert np.allclose(idler_grid[:, 0], idler)


def test_ideal_gaussian_purity_exceeds_paper():
    geometry = build_geometry(
        pump_wavelength_um=1.75,
        signal_wavelength_um=3.5,
        pump_intensity_fwhm_nm=3.76,
        crystal_length_mm=30,
        target_sigma_z_mm=6,
        min_domain_um=0,
        qpm_period_um=15.497,
    )
    _, _, signal, idler = jsa_wavelength_grid(geometry, 80, 60)
    pump = signal * idler / (signal + idler) * 1e-3
    mismatch = material_mismatch(pump, signal * 1e-3, idler * 1e-3)
    center = float(material_mismatch(1.75, 3.5, 3.5))
    ideal = np.exp(-0.5 * ((mismatch - center) * 6000) ** 2)
    result = evaluate_jsa_from_pmf(ideal, 3.76, signal, idler)
    assert result["purity"] > 0.999


def test_optional_polish_reduces_objective():
    geometry = build_geometry(
        pump_wavelength_um=1.75,
        signal_wavelength_um=3.5,
        crystal_length_mm=0.2,
        target_sigma_z_mm=0.04,
        min_domain_um=0,
        qpm_period_um=15.497,
    )
    objective = PMFObjective(geometry, q_points=31, device="cpu")
    initial = np.full(geometry.number_periods, 0.5)
    result = polish_duty_cycles_lbfgsb(objective, initial, 10)
    assert result["cost"] < float(objective(initial)[0])


def test_exact_pmf_matches_numerical_quadrature_and_zero_limit():
    duties = np.array([0.17, 0.61, 0.43, 0.82, 0.29])
    period = 15.497
    mismatch = np.array([0.0, 1e-12, -0.405, 0.39, 0.405, 0.42])
    domains = domain_table(duties, period)
    independent = []
    for k in mismatch:
        real, imag = 0.0, 0.0
        for _, sign, left, right, _ in domains:
            real += sign * quad(lambda z: np.cos(k*z), left, right, epsabs=1e-11)[0]
            imag -= sign * quad(lambda z: np.sin(k*z), left, right, epsabs=1e-11)[0]
        independent.append((real + 1j*imag) / (len(duties)*period))
    actual = pmf_from_duties(duties, period, mismatch)
    assert np.allclose(actual, independent, rtol=1e-11, atol=1e-12)
    assert np.isclose(actual[0], np.mean(2*duties - 1))


def test_batched_objective_uses_identical_pmf_to_plotting():
    geometry = build_geometry(crystal_length_mm=0.3, target_sigma_z_mm=0.06)
    objective = PMFObjective(geometry, q_points=101, device="cpu")
    rng = np.random.RandomState(8)
    duties = rng.uniform(geometry.duty_min, geometry.duty_max, (3, geometry.number_periods))
    actual = objective.pmf_batch(duties)
    for index, row in enumerate(duties):
        expected = pmf_from_duties(row, geometry.qpm_period_um, objective.mismatch_rad_um)
        assert np.allclose(actual[index], expected, rtol=1e-10, atol=1e-12)
    intensity = np.abs(actual)**2
    expected_cost = np.mean((intensity / intensity.max(axis=1, keepdims=True)
                            - objective.target_intensity)**2, axis=1)
    assert np.allclose(objective(duties), expected_cost, rtol=1e-12)


def test_purity_matches_py_jsi_spdc_equations():
    # Independent transcription of calculate_purity(), without loading its
    # unrelated SFG files or executing that script's top-level plots.
    signal, idler = np.meshgrid(np.linspace(3470, 3530, 40), np.linspace(3470, 3530, 40))
    c = 299.792458
    ws, wi = 2*np.pi*c/signal, 2*np.pi*c/idler
    wp0 = 2*np.pi*c*(1/np.mean(signal) + 1/np.mean(idler))
    lp0 = 2*np.pi*c/wp0
    sigma_p = np.sqrt(2)*(2*np.pi*c*3.76/lp0**2)/2.355
    rng = np.random.RandomState(9)
    complex_pmf = rng.normal(size=signal.shape) + 1j*rng.normal(size=signal.shape)
    original_jsa = np.exp(-(ws+wi-wp0)**2/(2*sigma_p**2))*np.abs(complex_pmf)
    original_jsa /= np.sqrt(np.sum(np.abs(original_jsa)**2))
    original_purity = np.sum(svd(original_jsa, compute_uv=False)**4)
    actual = evaluate_jsa_from_pmf(complex_pmf, 3.76, signal, idler)
    assert np.allclose(actual["jsa"], original_jsa, atol=2e-12)
    assert np.isclose(actual["purity"], original_purity, atol=1e-12)


def test_cuda_requested_does_not_silently_fall_back():
    fake = SimpleNamespace(cuda=SimpleNamespace(runtime=SimpleNamespace(getDeviceCount=lambda: 0)))
    with patch.dict("sys.modules", {"cupy": fake}):
        try:
            _load_backend("cuda")
        except RuntimeError:
            pass
        else:
            raise AssertionError("explicit CUDA must fail if no GPU exists")
        assert _load_backend("auto")[1] == "cpu"


def test_pso_constraints_and_monotone_history():
    geometry = build_geometry(crystal_length_mm=0.3, target_sigma_z_mm=0.06)
    objective = PMFObjective(geometry, q_points=31, device="cpu")
    result = optimize_duty_cycles(objective, PSOConfig(particles=8, iterations=10, cycles=2))
    assert len(result["history"]) == 21
    assert np.all(np.diff(result["history"]) <= 1e-15)
    assert np.all(result["duties"] >= geometry.duty_min)
    assert np.all(result["duties"] <= geometry.duty_max)
    assert np.isclose(result["cost"], objective(result["duties"])[0])


def test_main_defaults_are_read_from_child_modules():
    import inspect
    import run_pso
    from opln_pso.model import JSAConfig
    with patch("sys.argv", ["run_pso.py"]):
        args = run_pso.parse_args()
    assert args.pulse_fs == inspect.signature(build_geometry).parameters["pump_intensity_fwhm_fs"].default
    assert args.device == PSOConfig().device
    assert args.jsa_grid == JSAConfig().grid_size


def test_optional_gpu_matches_cpu():
    try:
        import cupy as cp
        if cp.cuda.runtime.getDeviceCount() < 1:
            raise SkipTest("No CUDA GPU available")
    except ImportError:
        raise SkipTest("CuPy not installed; GPU numerical test not run")
    geometry = build_geometry(crystal_length_mm=0.3, target_sigma_z_mm=0.06)
    cpu = PMFObjective(geometry, q_points=101, device="cpu")
    gpu = PMFObjective(geometry, q_points=101, device="cuda")
    duties = np.random.RandomState(11).uniform(geometry.duty_min, geometry.duty_max,
                                             (4, geometry.number_periods))
    assert np.allclose(cp.asnumpy(gpu.pmf_batch(duties)), cpu.pmf_batch(duties), atol=1e-11)
    assert np.allclose(cp.asnumpy(gpu(duties)), cpu(duties), atol=1e-11)
    cp.cuda.get_current_stream().synchronize()


if __name__ == "__main__":
    passed = skipped = 0
    for name, function in sorted(list(globals().items())):
        if name.startswith("test_") and callable(function):
            try:
                function()
            except SkipTest as exc:
                skipped += 1
                print(f"SKIP {name}: {exc}")
            else:
                passed += 1
                print(f"PASS {name}")
    print(f"{passed} passed, {skipped} skipped")
