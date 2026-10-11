# MultipleScatteringCalculations: notes for Claude

## Numerical pitfalls (measured; do not reintroduce)

### Gauss-Legendre weights: do not use `numpy.polynomial.legendre.leggauss` or `scipy.special.roots_legendre`

Their weights carry relative errors that grow with n. Against 30-digit mpmath (11 October 2026):

| n  | numpy `leggauss` | scipy `roots_legendre` |
|----|------------------|------------------------|
| 12 | 2.5e-15          | 5.5e-15                |
| 32 | 4.2e-15          | 4.5e-14                |
| 40 | up to 7e-13 (small end weights) | 2.2e-14 |
| 48 | 6.2e-14          | 5.3e-14                |

The nodes are accurate (about 6e-17); the weights are not.  Any integral built on these rules is limited to a
few times 1e-15, or worse at larger n, and the error does not converge as n grows. Found when the interface
image blocks (`cubic_scattering/interface_image.py`) stalled at 2-7e-15 against an extended-precision
reference in every basis.

**Use `cubic_scattering.gauss_rules.gauss_legendre(n)` / `gauss_interval(n, lo, hi)`.** It reads
`scripts/data/gauss_legendre.json` (n = 1-100, 30 digits; produced and checked by
`scripts/derive_gauss_legendre.py`: Newton's method at 40 digits, weights 2 / ((1 - x^2) P_n'(x)^2)) and
computes other n the same way on demand. Weights are then correct to about 1e-16.

About 90 files in `cubic_scattering/` and `scripts/` still call `leggauss`. They have not all been audited;
results quoted at the 1e-15 level from them (Paper 2's quadrature routes among them) may be limited by the rule.

### Sum tensor-product rules one axis at a time

A single `np.einsum("ijk,ia,jb,kc->abc", ...)` over an n^3 Gauss grid accumulates every term in one running
sum: about 2e-15 lost at n = 32. Three successive contractions (`np.tensordot`, or matrix products) lose about
2e-16. The same holds for 2-D face sums (`"ab,ra,rb->r"`): contract one axis, then the other.
See `interface_image._contract3`.

### Plane-wave mode sums at large lateral wavenumber

In the up/down mode representation (`cubic_scattering/stratified_march.py`) the P and SV terms cancel as
(c q / omega)^4 at large q: the reflected kernel is wrong by 6e-7 at q h = 20 (omega h / beta = 0.17) and the
static-subtracted remainder is rounding noise above q h ~ 100 in double precision. Use 40-digit arithmetic for
reference computations at large q (`scripts/measure_interface_static_subtraction.py`), or keep cutoffs low.

### SymPy: symbols with and without assumptions are different symbols

`sp.sympify("z")` makes `Symbol("z")`, which is not `Symbol("z", real=True)`: differentiating one with
respect to the other gives 0 silently. Parse stored expressions with
`sp.sympify(text, locals={"z": z, ...})` using the same symbol objects.

## Workflow

- Development branch for Claude sessions: as given by the session; the user asks "fast forward to main" to
  publish: `git fetch origin main`, check `git merge-base --is-ancestor origin/main HEAD`, then
  `git push origin HEAD:main`.
- Mathematica derivations live in `Mathematica/*.wl` with Python twins in `scripts/derive_*.py`; the user runs
  the `.wl` locally in a notebook (`Get[SystemDialogInput["FileOpen"]]`) and uploads the `.json` it writes.
- Papers are in `LatexPDFs/<Name>/`, built with `latexmk -lualatex`. They are journal submissions, not lab
  reports: state results; code paths only under Data availability.
