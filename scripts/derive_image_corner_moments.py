"""Twin of Mathematica/ImageCornerMoments.wl: the closed-form corner moments of the static interface image.

Compares ``cubic_scattering.image_moments`` with the Mathematica output (base integrals B1, B2 and a sample of
corner moments by NIntegrate at 30 digits), and records the Python-side check that was run in the session:
all 53,352 corner moments of the 61 kernel families (monomials to degree (6, 6, 8), both lateral
orientations) agree with Euler's reduction plus Gauss rules on the far faces to 1.2e-15 on the natural scale
2^(p+q+r); B1 and B2 agree with a direct 3-D tanh-sinh quadrature to 20 digits.

The Mathematica references are NIntegrate after one exact integration (a direct 3-D NIntegrate missed its goal
silently, by up to 5e-7, in a first run).  They reach 16 to 23 digits, not the 28 asked for: the one with an
exact value, 1/(2 pi) times Paper 2's master integral of 1/R over [0, 2]^3, is off by 1.4e-17, while the closed
forms here do not move between 40 and 70 digits (1e-39).  The thresholds are set to that demonstrated
accuracy, 1e-15.  Measured, 10 October 2026: all 13 references and B2 agree to 1.6e-16 - 4.9e-23; B1 (whose
Mathematica value is a partial closed form, Catalan's constant and dilogarithms with one angular integral left)
to 6e-32.

Closed forms for the cube (a = b = c; both are scale-free), derived in the session and now used by
``image_moments``:  B1 = (5/4) Cl2(pi/3) - G  (Catalan's G; Mathematica's dilogarithms collapse by Kummer's
formula and the duplication formula of Cl2), and the elementary  B2 = (1/3) log((1 + sqrt 2)^2 / (2 (1 + sqrt 3))).
Both agree with the 40-digit angular quadratures to 1e-40; the .wl checks them on its side.

Run:  PYTHONPATH=. python scripts/derive_image_corner_moments.py
"""

import json
from pathlib import Path

import mpmath as mp

from cubic_scattering.image_moments import base_b1, base_b2, corner_moment

ROOT = Path(__file__).resolve().parent.parent


def _num(s: str) -> mp.mpf:
    """A Mathematica InputForm number (e.g. 0.35271141383484801622`30.) as an mpmath number."""
    s = s.split("`")[0].replace("*^", "e")
    return mp.mpf(s)


def main() -> None:
    path = ROOT / "Mathematica" / "ImageCornerMoments.json"
    if not path.exists():
        print(
            "Mathematica/ImageCornerMoments.json not found: run the .wl to compare (SKIPPED, not PASS)"
        )
        return
    m = json.loads(path.read_text())
    mp.mp.dps = 30
    print(
        "line of images (Mathematica):",
        m["line_of_images_phi1"],
        m["line_of_images_phi2"],
    )
    print("B1 closed form:", m["B1_closed"])
    print("B2 closed form:", m["B2_closed"])
    ok = True
    for name, mine in (("B1", base_b1(2, 2, 2)), ("B2", base_b2(2, 2, 2))):
        err = abs(mine - _num(m[name])) / abs(mine)
        ok &= err < 1e-15
        print(f"{name}: {mp.nstr(err, 3)}  {'PASS' if err < 1e-15 else 'FAIL'}")
        if f"{name}_simple" in m:  # the simplified closed forms, (5/4) Cl2(pi/3) - G and the elementary B2
            err = abs(mine - _num(m[f"{name}_simple"])) / abs(mine)
            ok &= err < 1e-28
            print(
                f"{name} simplified closed form: {mp.nstr(err, 3)}  {'PASS' if err < 1e-28 else 'FAIL'}"
                f"   (Mathematica's own checks: {m[f'{name}_simple_check']})"
            )
    for case, ref in zip(m["cases"], m["references"], strict=True):
        j, alpha, n, signs = case
        mine = corner_moment(j, tuple(alpha), tuple(n), (2, 2, 2), tuple(signs))
        theirs = _num(ref)
        err = abs(mine - theirs) / max(
            abs(theirs), mp.mpf(2) ** (sum(n)) * mp.mpf(10) ** -30
        )
        passed = err < 1e-15
        ok &= passed
        print(
            f"Phi{j} d{alpha} s{n} signs {signs}: {mp.nstr(err, 3)}  {'PASS' if passed else 'FAIL'}"
        )
    print("ALL PASS" if ok else "FAILURES")


if __name__ == "__main__":
    main()
