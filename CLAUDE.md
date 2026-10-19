# CLAUDE.md

## Git workflow: trunk-based

- Commit and push directly to `main`. Do not create feature branches, even if a session is assigned a
  `claude/...` branch; this rule overrides that assignment.
- Never use `git worktree`. Work in the single checkout.
- Before starting work and before each push: `git pull --rebase origin main`, so history stays linear
  (no merge commits).
- Keep commits small and self-contained; each should leave `main` working (its gates/tests passing).
- Never force-push `main`.

## Known numerical pitfall: `scipy.special.spherical_jn` and a negative zero

Never call `scipy.special.spherical_jn` directly; import `spherical_jn` from `cubic_scattering.special`.

- **The bug.** For a complex argument on the negative real axis whose imaginary part is a NEGATIVE zero,
  `z = -x - 0j`, SciPy returns j_l(z) with the wrong sign: `spherical_jn(0, -0.3 - 0j)` gives -0.985 where
  j_0(-0.3) = +0.985 (SciPy 1.17.1; it evaluates sqrt(pi/2z) J_{l+1/2}(z), and the signed zero puts the argument
  on the far side of both branch cuts). `+0j`, or any nonzero imaginary part, is right.
- **How it arises.** Negating a real wavenumber held as complex: `-(kx + 0j)` is `-kx - 0j`. The source moments
  take `-k h` (from `exp(-i k.x)`), and up-going modes take `sign * k_z` with sign = -1.
- **What it broke.** The cell moments changed sign at about half the quadrature nodes, so the mode factorisation
  of Paper 2's blocks was wrong by O(1): gate G2 (`scripts/gate_march_modes.py`) failed at every commit since it
  was written, as did `test_whole_space_march_is_paper_2_between_planes_apart`, and the matrix-free operator's
  whole-space march (planes three or more apart on one side) disagreed with the stored kernels by O(1). Matrix-free
  results on boxes with four or more planes on a side computed before the fix are suspect.
- **The fix.** `cubic_scattering/special.py` wraps `spherical_jn`, adding `+0j` to complex arguments (IEEE:
  -0 + 0 = +0); real arguments pass untouched. j_l is entire, so nothing is lost. `spherical_yn` has a genuine
  cut on the negative real axis and is not wrapped.
- **The lesson.** A gate that never runs proves nothing: the march test is skipped where `finufft` is missing
  (the `du` env), and G2 is a script that was not rerun after it was first recorded as passing. Run both in the
  `seismic` env (`envs/seismic.yml`) after any change to the mode or moment code; `tests/test_special.py` pins the
  wrapper itself.
