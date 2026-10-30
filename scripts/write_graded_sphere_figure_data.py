#!/usr/bin/env python3
"""The data of the Error paper's convergence figure, panel (d): the graded sphere, three voxels.

Writes ``LatexPDFs/PolynomialVoxelError/figures/data_graded_sphere_ka{0.5,1.0}.json``, which
``plot_convergence_orders.py`` reads:
  the mean-only Galerkin voxel   from the runs recorded with the archived hierarchy paper
                                 (``pilot_graded_voxel_sphere.py --arms=g0fft``, 4 to 24 cells across);
  the collocation voxel          the uniform-strain closure exact at every frequency (``point_cell_self``),
                                 contrast at the cell centre, cells whose centre is inside the sphere;
  the collocation voxel, cell mean  the same closure with the cell mean of the contrast on every cell that
                                 overlaps the sphere,
both collocation ladders from ``scripts/data/equal_degree_sphere.jsonl`` (``measure_equal_degree_sphere.py``,
the smoothstep body). Until 31 October 2026 the collocation curves were the package's cube T-matrix, whose
far-field form-factor corrections depart from the closure; its values are kept in the archived files.

Run:  python scripts/write_graded_sphere_figure_data.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ARCHIVE = ROOT / "LatexPDFs" / "archive" / "CartesianMultipoleHierarchyArchive" / "figures"
FIG = ROOT / "LatexPDFs" / "PolynomialVoxelError" / "figures"
ROWS = ROOT / "scripts" / "data" / "equal_degree_sphere.jsonl"


def ladder(rows: list[dict], ka: float, arm: str) -> dict:
    """One arm's ladder on the smoothstep body at k_S a = ka, in order of n; a rerun replaces a row."""
    latest = {r["n"]: r for r in rows if r["body"] == "smoothstep" and r["ka_s"] == ka and r["arm"] == arm}
    if not latest:
        raise ValueError(
            f"{ROWS}: no {arm} rows on the smoothstep sphere at k_S a = {ka}; run "
            f"python scripts/measure_equal_degree_sphere.py --body=smoothstep --ka={ka} --arms={arm} 4 6 8 12 16 24 32"
        )
    ns = sorted(latest)
    return {"n_sub": ns, "error_vs_exact": [latest[n]["error"] for n in ns]}


def main() -> int:
    rows = [json.loads(line) for line in ROWS.read_text().splitlines() if line.strip()]
    for tag, ka in (("0.5", 0.5), ("1.0", 1.0)):
        data = json.loads((ARCHIVE / f"data_graded_sphere_ka{tag}.json").read_text())
        data["collocation"] = {
            "voxel": "collocation voxel, uniform-strain closure exact at every frequency: contrast at the cell "
            "centre, cells kept if the centre is inside, far field at 5e8 radii",
            "script": f"scripts/measure_equal_degree_sphere.py --body=smoothstep --ka={ka} --arms=exact_centre",
            **ladder(rows, ka, "exact_centre"),
        }
        data["collocation_cell_mean"] = {
            "voxel": "the same collocation voxel with the cell mean of the contrast and every overlapping cell kept",
            "script": f"scripts/measure_equal_degree_sphere.py --body=smoothstep --ka={ka} --arms=exact_mean",
            **ladder(rows, ka, "exact_mean"),
        }
        out = FIG / f"data_graded_sphere_ka{tag}.json"
        out.write_text(json.dumps(data, indent=2) + "\n")
        print(f"wrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
