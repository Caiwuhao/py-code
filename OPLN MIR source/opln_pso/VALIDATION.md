# First-version validation

All results below use the thesis Sellmeier coefficients, `y -> y + z`, a
1560 nm Gaussian pump with 60 fs intensity FWHM, degenerate 3120 nm output,
and a 500 nm minimum physical domain.

## Geometry checks

- QPM period: 15.3994107 um
- Gaussian spatial width from first-order bandwidth matching: 0.313225 mm
- Implemented length after rounding to 102 complete periods: 1.570740 mm
- Allowed duty-cycle interval: 0.0324688 to 0.9675312
- Finite-length ideal Gaussian relative peak efficiency: 24.381% of a
  conventional 50% duty PPLN of the same length

## PSO repeatability

Phase-aware, symmetric, `+-15 sigma` objective; 48 particles and 200
iterations:

| Seed | Best cost | Schmidt purity (4.5 sigma window) | Relative peak efficiency | Minimum domain |
|---:|---:|---:|---:|---:|
| 1 | 0.002507 | 0.99536 | 24.26% | 0.500 um |
| 7 | 0.004499 | 0.99706 | 23.34% | 0.500 um |
| 19 | 0.003125 | 0.99511 | 23.46% | 0.500 um |

The precise duty array changes with the seed, as expected for PSO, while the
purity and brightness remain close. A fabrication design should still be
selected after more seeds and a robustness calculation for domain-wall error.

The purity values above are not unfiltered, infinite-range values. For seed 7,
expanding the half-width of each JSA frequency axis from 4.5 to 6 and 8 pump
sigma changes the result from 0.99707 to 0.99159 and 0.96197, respectively.
The 4.5-sigma plot window corresponds approximately to 2.60-3.94 um here.
Therefore, the final quoted purity must use the intended bandpass filter,
detector response, and validity range of the material model.

## Why the old intensity target is retained only as a reference

For the same seed, 48 particles, and 200 iterations, the narrow-range legacy
objective reached a normalized spectral-intensity cost of 6.06e-7. However,
its JSA purity in the same 4.5-sigma window was 0.94213 and its relative peak
efficiency was only 6.86%. A visually excellent central intensity fit does not establish
that the complex PMF or the full JSA is optimal.

The recommended objective explicitly includes complex spectral amplitude,
absolute peak amplitude, symmetry, and a wider mismatch interval. The erf
function remains available as the mathematically consistent cumulative spatial
target for the future binary-orientation stage.

## Execution checks

- Syntax and four deterministic physics/constraint tests passed on Python 3.
- Three formal optimization seeds completed on the NumPy CPU backend.
- The optional CuPy/CUDA path is implemented but could not be executed in the
  present environment because no CUDA device or CuPy installation was
  available.
