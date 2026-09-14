# Python PSO design for 1560 nm -> 3120 nm OPLN

This is a first Python port and modernization of the MATLAB code printed in
Appendix C of the thesis. It uses the same LiNbO3 Sellmeier coefficients and
the same type-II `y -> y + z` configuration, while adding a fabrication bound
on every physical domain.

## Main conclusions about the target

The MATLAB listing uses two different Gaussian/error-function objects:

1. The initial duty-cycle array is erf-shaped. Algebraically, lines 27-32 of
   `ML.m` reduce to `A(x) = 0.5 + 0.45 erf[(x-0.5)/0.45]`.
2. The actual PSO cost does **not** use an erf target. `arrayloop.m` returns
   normalized PMF intensity, `effcalc.m` returns a Gaussian spectrum, and
   `cost.m` calculates their squared difference. Spectral phase is discarded.

The O-band/Graffiti-style construction uses a cumulative target in real space.
Starting from the ideal spectral field

`phi_I(delta_k) = exp[-(delta_k-delta_k0)^2 sigma_z^2 / 2]`,

its spatial nonlinearity is Gaussian, and its cumulative amplitude along the
crystal is an erf. Thus the Gaussian spectrum and cumulative erf are Fourier-
dual descriptions of the same desired device, not competing choices.

Recommended division of labor:

- Duty-cycle PSO with exact broadband PMF evaluation: fit the Gaussian
  spectral field. Use `complex_amplitude` when spectral phase matters, or
  `legacy_intensity` for a close comparison with the old MATLAB objective.
- Binary `+1/-1` orientation design: use the cumulative erf target in a greedy,
  dynamic-programming, or mixed-integer stage.
- A later hybrid optimizer can first select orientations with the cumulative
  erf and then fine-tune allowed domain walls/duty cycles against the complex
  spectral Gaussian.

An erf can therefore be used in the next optimizer, but it should describe the
cumulative spatial amplitude. It should not simply replace the spectral
Gaussian inside the existing frequency-domain cost.

## 3120 nm default geometry

With 1560 nm pump, degenerate 3120 nm output, the thesis Sellmeier equation,
and `y -> y + z` polarization:

- first-order QPM period: about 15.3994 um;
- 500 nm minimum domain: duty cycle is bounded to about
  `[0.03247, 0.96753]`;
- for a 60 fs Gaussian pump, first-order bandwidth matching gives
  `sigma_z` about 0.313 mm;
- using `L/sigma_z = 5` gives about 1.57 mm, or 102 periods.

This is deliberately different from the old 30 mm example. A 30 mm device is
matched to a pump duration near 1.15 ps (roughly 3 nm intensity FWHM at
1560 nm), not to an unmodified 60 fs pulse. Use the measured pump spectrum in
the final design.

The 60 fs default assumes a transform-limited Gaussian pulse. For a sech-squared
or non-ideal laser, pass the measured spectral intensity FWHM directly, e.g.
`--pump-fwhm-nm 43`; this overrides the duration-based conversion.

## Run

```bash
python run_pso.py --iterations 200 --particles 48
```

The default is the phase-aware, symmetric, wide-range objective. To reproduce
the central limitation of the old intensity-only cost, use:

```bash
python run_pso.py \
  --objective legacy_intensity \
  --no-symmetry \
  --q-span-sigma 4.5 \
  --q-points 401 \
  --output output/legacy_intensity
```

Force a 30 mm crystal while retaining a 500 nm minimum domain:

```bash
python run_pso.py --crystal-length-mm 30 --min-domain-um 0.5
```

Important outputs are `summary.json`, `duty_cycles.csv`, `domains.csv`,
`optimization_overview.png`, `jsa_comparison.png`, and `result_arrays.npz`.
The summary reports optimized purity for several integration windows by
default. Purity is not meaningful without this range (or an equivalent
experimental filter specification); change it with, for example,
`--purity-spans 4.5,6,8,10`.

## GPU

The objective evaluates all particles as a batch and chunks the spectral axis.
If a compatible CuPy package and NVIDIA CUDA device are installed, run:

```bash
python run_pso.py --device cuda --wavelength-chunk 128
```

GPU acceleration is most useful for the 30 mm (~1948-variable) case. The
60 fs automatic design has only ~102 variables and is often acceptable on a
CPU. The CUDA path is optional; the CPU path has no CuPy dependency.

## Current limitations

- The Sellmeier coefficients are copied exactly from the old code and do not
  yet include temperature or MgO concentration.
- The pump is an ideal transform-limited Gaussian; importing the measured
  laser spectrum is the next practical improvement.
- This version optimizes duty cycles only. The mixed-integer orientation stage
  is intentionally left for version 2.
- The default numerical settings should be converged with several
  random seeds before a fabrication file is accepted.

The objective range matters. A narrow fit can look excellent around the main
lobe while leaving farther PMF structure that lowers the two-dimensional JSA
purity. The default `+-15 sigma` range is intentional.
