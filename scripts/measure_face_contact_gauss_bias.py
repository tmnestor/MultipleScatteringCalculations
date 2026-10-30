#!/usr/bin/env python3
"""The face-contact bias of a tensor Gauss rule: refinement settles on a value three times too large.

Two unit cubes in face contact, R = (1, 0, 0), and the static strain kernel d2G: the entry S00 of the
volume-averaged face block, int int d2G_0000(R + x - x') dx dx' over both cubes. A double-cube tensor Gauss rule
of n points per axis uses the same lateral nodes in both cubes, so the separation w = R + x - x' has w_perp = 0
exactly at n^2 node pairs: it samples the singular ray of the 1/w^3 kernel, with weight that does not vanish as
n grows. The rule's value settles under refinement, each step smaller than the last, and so looks converged,
but to a limit about three times the true value (``face_s_rederivation.biased_tensor_gauss``).

The true value comes from two routes that use no singular quadrature and share nothing with the rule or with
each other (``face_s_rederivation``): the subdivision fixed point (exact homogeneity of the static kernel, and
smooth Gauss integrals of separated half-cubes only) and the dyadic-shell correlation quadrature.

Writes scripts/data/face_contact_gauss_bias.json.

Run:  python scripts/measure_face_contact_gauss_bias.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import face_s_rederivation as red  # noqa: E402

OUT = Path(__file__).resolve().parent / "data" / "face_contact_gauss_bias.json"
LADDER = range(4, 17, 2)


def main() -> int:
    exact_sub = float(red.subdivision_tables(ng=10)["face"][0, 0, 0, 0])
    exact_dy = float(red.dyadic_contact_table(ng=12)[0, 0, 0, 0])
    print(f"S00 at face contact: subdivision {exact_sub:.10e}, dyadic {exact_dy:.10e}")
    print("   n   tensor Gauss        / true    step")
    rows: list[dict] = []
    prev: float | None = None
    for n in LADDER:
        g = float(red.biased_tensor_gauss(n))
        step: float | None = None if prev is None else g - prev
        rows.append({"n": n, "gauss": g, "ratio": g / exact_sub, "step": step})
        print(f"  {n:2d}   {g:.6e}    {g / exact_sub:6.3f}   {'' if step is None else f'{step:+.3e}'}")
        prev = g
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps({"exact": {"subdivision": exact_sub, "dyadic": exact_dy}, "rows": rows}, indent=1) + "\n"
    )
    print(f"wrote {OUT.relative_to(Path(__file__).resolve().parent.parent)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
