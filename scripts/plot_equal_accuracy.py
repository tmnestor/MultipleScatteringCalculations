#!/usr/bin/env python3
"""The equal-accuracy figure of the Error paper: error against unknowns for the four arms on the graded sphere.

Writes the coordinates of Fig. fig:equal_accuracy (``LatexPDFs/PolynomialVoxelError/figures/fig_equal_accuracy.tex``,
pgfplots) from the measured ladders of ``scripts/compare_point_galerkin.py``: one whitespace table per frequency and
arm, ``figures/data_eqacc_ka<K>_<arm>.dat`` with the columns ``n unknowns error wall``, K = 05 for k_S a = 0.5 and 4
for k_S a = 4. A rerun of a grid replaces the earlier row (``equal_accuracy.load``). It then prints the table of
unknowns and wall time at the paper's target errors, each entry with its status (measured or interpolated, by the
rules of ``scripts/equal_accuracy.py``) or the reason it is not given, and the error each arm's ladder ends at.

Run:  python scripts/plot_equal_accuracy.py      (then lualatex fig_equal_accuracy.tex in the figures directory)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from equal_accuracy import ROWS, equal_accuracy_table, load  # noqa: E402

FIG = Path(__file__).resolve().parent.parent / "LatexPDFs" / "PolynomialVoxelError" / "figures"
BODY = "graded"
#: the exact classical point cell with the cell mean of the contrast (the headline: the Galerkin cell's
#: representation of the medium), the same cell with the centre value, and the two fourth-order cells
ARMS = ("point_exact_mean", "point_exact", "galerkin", "hierarchy")
#: the frequencies of the figure's two panels, with the tag each one's file names carry
PANELS = {0.5: "05", 4.0: "4"}
#: the target errors of the paper's table, per frequency
TARGETS = {0.5: (1e-3, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6), 4.0: (1e-3, 3e-4, 1e-4, 1e-5)}


def data_name(ka: float, arm: str) -> str:
    """The file name of one panel's ladder for one arm."""
    return f"data_eqacc_ka{PANELS[ka]}_{arm}.dat"


def write_ladders(rows_path: Path, out_dir: Path) -> list[Path]:
    """Write one table per frequency and arm, rows in order of n; returns the files written.

    Raises:
        ValueError: a panel's arm has no rows (the figure would silently lose a curve).
    """
    rows = [r for r in load(rows_path) if r["body"] == BODY]
    written = []
    for ka in PANELS:
        for arm in ARMS:
            ladder = sorted((r for r in rows if r["ka_s"] == ka and r["arm"] == arm), key=lambda r: r["n"])
            if not ladder:
                raise ValueError(
                    f"{rows_path}: no {arm} rows on the {BODY} sphere at k_S a = {ka}; run "
                    f"python scripts/compare_point_galerkin.py --body={BODY} --ka={ka} --arm={arm} 4 6 8 12 16"
                )
            lines = ["n unknowns error wall"]
            lines += [f"{r['n']} {r['unknowns']} {r['error']:.6e} {r['wall_s']:.4f}" for r in ladder]
            path = out_dir / data_name(ka, arm)
            path.write_text("\n".join(lines) + "\n")
            written.append(path)
    return written


def report(rows_path: Path) -> None:
    """Print the paper's table: unknowns and wall time at each target, and where each ladder ends."""
    rows = [r for r in load(rows_path) if r["body"] == BODY]
    for ka, targets in TARGETS.items():
        print(f"{BODY} sphere, k_S a = {ka}")
        for arm in ARMS:
            ladder = sorted((r for r in rows if r["ka_s"] == ka and r["arm"] == arm), key=lambda r: r["n"])
            first, last = ladder[0], ladder[-1]
            print(
                f"  {arm:10s} n = {first['n']} to {last['n']}: error {first['error']:.2e} "
                f"({first['unknowns']:,} unknowns) to {last['error']:.2e} ({last['unknowns']:,})"
            )
        for t in equal_accuracy_table(rows_path, BODY, ka, targets):
            if t["unknowns"] is None:
                print(f"  {t['target']:.0e}  {t['arm']:10s}  not reached")
            else:
                print(
                    f"  {t['target']:.0e}  {t['arm']:10s}  {t['unknowns']:12,.0f}  {t['wall_s']:7.1f} s  {t['status']}"
                )


def main() -> int:
    for path in write_ladders(ROWS, FIG):
        print(f"wrote {path.relative_to(FIG.parent.parent.parent)}")
    report(ROWS)
    return 0


if __name__ == "__main__":
    sys.exit(main())
