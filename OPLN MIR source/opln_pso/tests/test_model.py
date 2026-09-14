import numpy as np

from opln_pso.model import (
    build_geometry,
    cumulative_erf_target,
    domain_table,
    pmf_from_duties,
)


def test_3120_nm_geometry():
    geometry = build_geometry()
    assert np.isclose(geometry.qpm_period_um, 15.399410673983054, rtol=1e-10)
    assert geometry.number_periods == 102
    assert np.isclose(geometry.duty_min, 0.5 / geometry.qpm_period_um)


def test_periodic_peak_is_two_over_pi():
    geometry = build_geometry()
    duties = np.full(geometry.number_periods, 0.5)
    k0 = 2.0 * np.pi / geometry.qpm_period_um
    peak = pmf_from_duties(duties, geometry.qpm_period_um, [k0])[0]
    assert np.isclose(abs(peak), 2.0 / np.pi, rtol=2e-4)


def test_erf_target_boundaries():
    geometry = build_geometry()
    values = cumulative_erf_target(
        [0.0, geometry.crystal_length_um],
        geometry.crystal_length_um,
        geometry.target_sigma_z_um,
    )
    assert np.allclose(values, [0.0, 1.0], atol=1e-12)


def test_minimum_domain_constraint_table():
    geometry = build_geometry(min_domain_um=0.5)
    duties = np.linspace(geometry.duty_min, geometry.duty_max, geometry.number_periods)
    domains = domain_table(duties, geometry.qpm_period_um)
    assert domains[:, 4].min() >= 0.5 - 1e-12


def test_measured_pump_bandwidth_override():
    geometry = build_geometry(pump_intensity_fwhm_nm=43.0)
    assert np.isclose(geometry.pump_intensity_fwhm_nm, 43.0)
    assert geometry.number_periods == 141
