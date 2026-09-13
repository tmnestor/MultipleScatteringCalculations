#!/usr/bin/env python3
"""Do the upstream tests constrain the stress-row x stress-column block?

Defect D1: ``assemble_greens_6x6`` rescales the stress SOURCE columns on the
displacement rows only, so the block ``[3:6, 3:6]`` carries one factor of
``(-i w)`` where a consistent change of basis needs two.

The standing recommendation has been to carry the correction locally rather than
patch the sibling repository, on the reasoning that its reflectivity outputs are
right BECAUSE this error cancels in the ratios that form R and T.  That is a
hypothesis, and it is cheap to test: apply the missing factor in-process and run
the sibling repository's own suite.

  suite still passes  -> the block is unconstrained upstream; D1 is a latent
                         defect that no existing test exercises, and a patch
                         upstream would be safe (though still not ours to make)
  suite fails         -> the cancellation hypothesis is confirmed, carrying the
                         correction locally is right, and the failing test names
                         are the evidence

This probe makes NO lasting edit to the sibling repository.  The patch lives
only in this process.

Run both legs:
    conda run -n seismic python scripts/probe_upstream_fix.py
    conda run -n seismic python scripts/probe_upstream_fix.py --patched

RESULT (13 September 2026)
--------------------------
baseline : 119 passed
patched  : 118 passed, 1 failed
           test_interlayer_ms.py::TestCrossValidation4x4_9x9::test_4x4_9x9_consistency
           (max relative difference 6.3e-3)

So 118 of 119 upstream tests are INDIFFERENT to D1, and the one that is not is
not a physical check.  It asserts that the 9x9 solver agrees with the 4x4 solver
for a diagonal T at ky = 0, and the two sides are built like this:

    9x9 path :  (A @ G6 @ B)   driven by  T_9x9
    4x4 path :   G_4x4         driven by  B @ T_9x9 @ A      (tmatrix_9x9_to_4x4_psv)

Both wrap the SAME raw Green's function with the SAME ``A`` and ``B``.  The test
is therefore an algebraic identity -- does ``A G B`` composed with ``T`` equal
``G`` composed with ``B T A`` -- which holds for ANY ``G``, correct or not.  It
is structurally incapable of detecting a common-mode error in ``G``, and it
would fail just as readily for a CORRECT change to ``G``.  (It also uses
``traction_from_strain`` for ``B``, which is plain wavenumber-independent
Hooke's law and not the source operator at all, so it is doubly self-referential.)

VERDICT: carry the correction locally.  The reason is sharper than the one
originally recorded: not "the error cancels in the ratios that form R and T",
but "no upstream test compares the 9x9 path to anything outside itself".  A
patch upstream would break that consistency test, and repairing it honestly
would also require correcting ``tmatrix_9x9_to_4x4_psv`` and
``traction_from_strain`` -- a change to a repository other work depends on, and
not ours to make.  Raise a note upstream, not a patch.
"""

import argparse
import sys

sys.path.insert(0, "/Users/tod/Desktop/SeismicInversion")

import GlobalMatrix.layered_greens as lg  # noqa: E402
import pytest  # noqa: E402

SIBLING = "/Users/tod/Desktop/SeismicInversion/GlobalMatrix"

TEST_FILES = [
    f"{SIBLING}/test_layered_greens.py",
    f"{SIBLING}/test_riccati.py",
    f"{SIBLING}/test_riccati_source.py",
    f"{SIBLING}/test_gmm.py",
    f"{SIBLING}/test_interlayer_ms.py",
]


def apply_d1_patch() -> None:
    """Give the stress-row x stress-column block its second factor of (-i w)."""
    original = lg.assemble_greens_6x6

    def patched(g_psv, g_sh, cos_phi, sin_phi, omega):
        out = original(g_psv, g_sh, cos_phi, sin_phi, omega)
        out[..., 3:, 3:] *= -1j * omega
        return out

    lg.assemble_greens_6x6 = patched


def main() -> int:
    """Run the sibling suite with or without the D1 patch.

    Returns:
        The pytest exit code.
    """
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--patched", action="store_true", help="apply the D1 fix first")
    args = ap.parse_args()

    if args.patched:
        apply_d1_patch()
        print(">>> D1 PATCH APPLIED: G6[3:6, 3:6] *= (-i w)\n")
    else:
        print(">>> BASELINE: sibling repository unmodified\n")

    return int(pytest.main(["-q", "--no-header", "-p", "no:cacheprovider", *TEST_FILES]))


if __name__ == "__main__":
    sys.exit(main())
