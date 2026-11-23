#!/usr/bin/env python3
"""Degree against cell size: at equal unknowns, which degree of cell has the smaller error?

Reads the uniform grids stored by ``measure_two_term_fft.py`` for cells of degree 0, 1 and 2 (9, 36 and
90 unknowns a cell) on the three radial bodies (graded shell, halo, sharp sphere) at k_S a = 0.5, 2 and
4, and answers three questions from them. Nothing is solved here.

1. THE LAW AT EACH DEGREE. For each body, frequency and degree: the constant C = (error beyond first
   order) / (relative projection error E of the medium at that degree), its smallest and largest value
   over the grids, and the fitted orders of the error and of E in the cell size (the slope of the
   logarithm against the logarithm of cells across, between the two finest grids that are above the
   floor of the reference, 1e-7 of the peak).

2. THE CHOICE AT EQUAL UNKNOWNS. At each of a list of budgets of unknowns the solved error of each
   degree, interpolated in the logarithms between its grids (never extrapolated), and the degree that
   wins. Beside it the same choice made WITHOUT the fine solves: predicted error = dT1 + C E, with dT1
   and E from no solve at all and C from the one coarsest grid of that degree (4 cells across, under a
   second to solve). The record says whether the predicted winner is the solved winner and how far the
   predicted errors are from the solved ones.

3. THE COST OF A TARGET. For target errors 1e-3, 1e-4 and 1e-5 of the peak: the unknowns and the seconds
   (kernel stage plus solve, on this machine) each degree needs, by the same interpolation.

Run:
    conda run -n seismic python scripts/analyse_degree_vs_size.py
Writes ``scripts/analyse_degree_vs_size.json`` and prints the tables.
"""

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BODIES = (("graded shell", "0.75", ""), ("halo", "0.2", "_halo"), ("sharp sphere", "1", "_sharp"))
KAS = ("0.5", "2", "4")
BUDGETS = (10_000, 30_000, 100_000, 300_000, 1_000_000)
TARGETS = (1e-3, 1e-4, 1e-5)
FLOOR = 1e-7


def load(ka: str, core: str, p: int, tag: str) -> list[dict] | None:
    path = HERE / f"measure_two_term_fft_ka{ka}_core{core}_p{p}{tag}.json"
    if not path.exists():
        return None
    return sorted(json.loads(path.read_text())["rows"], key=lambda r: r["n"])


def at(x: float, xs: np.ndarray, ys: np.ndarray) -> float | None:
    """ys at x by interpolation in the logarithms; None outside the solved range."""
    if not xs[0] <= x <= xs[-1]:
        return None
    return float(np.exp(np.interp(np.log(x), np.log(xs), np.log(ys))))


def first_crossing(target: float, errs: np.ndarray, vals: np.ndarray) -> float | None:
    """vals where the error first falls to the target, from the coarse end; None if it never does."""
    if errs[0] <= target:
        return float(vals[0])
    for i in range(len(errs) - 1):
        if errs[i] >= target >= errs[i + 1]:
            t = (np.log(errs[i]) - np.log(target)) / (np.log(errs[i]) - np.log(errs[i + 1]))
            return float(np.exp(np.log(vals[i]) + t * (np.log(vals[i + 1]) - np.log(vals[i]))))
    return None


def order(rows: list[dict], key: str) -> float | None:
    good = [r for r in rows if r["error"] > FLOOR and r[key] > 0]
    if len(good) < 2:
        return None
    a, b = good[-2], good[-1]
    return float(-np.log(b[key] / a[key]) / np.log(b["n"] / a["n"]))


def main() -> None:
    record = []
    for body, core, tag in BODIES:
        for ka in KAS:
            degrees = {p: rows for p in (0, 1, 2) if (rows := load(ka, core, p, tag))}
            entry: dict = {"body": body, "ka_s": float(ka), "law": {}, "budgets": [], "targets": []}
            print(f"=== {body}, k_S a = {ka}")
            for p, rows in degrees.items():
                c = np.array([r["rest"] / r["E"] for r in rows if r["error"] > FLOOR])
                entry["law"][p] = {
                    "grids": [r["n"] for r in rows],
                    "unknowns": [r["unknowns"] for r in rows],
                    "error": [r["error"] for r in rows],
                    "E": [r["E"] for r in rows],
                    "C_min": float(c.min()),
                    "C_max": float(c.max()),
                    "C_coarsest": rows[0]["rest"] / rows[0]["E"],
                    "order_error": order(rows, "error"),
                    "order_E": order(rows, "E"),
                }
                law = entry["law"][p]
                print(
                    f"  degree {p}: C {law['C_min']:.4f} to {law['C_max']:.4f} (coarsest {law['C_coarsest']:.4f}), "
                    f"order of error {law['order_error']}, of E {law['order_E']}, finest {rows[-1]['n']} across "
                    f"{rows[-1]['unknowns']} unknowns error {rows[-1]['error']:.2e}"
                )
            for budget in BUDGETS:
                solved, predicted = {}, {}
                for p, rows in degrees.items():
                    u = np.array([r["unknowns"] for r in rows], dtype=float)
                    c0 = rows[0]["rest"] / rows[0]["E"]
                    e = at(budget, u, np.array([r["error"] for r in rows]))
                    if e is None:
                        continue
                    solved[p] = e
                    predicted[p] = at(budget, u, np.array([r["dT1"] + c0 * r["E"] for r in rows]))
                if len(solved) < 2:
                    continue
                win_s = min(solved, key=solved.get)
                win_p = min(predicted, key=predicted.get)
                entry["budgets"].append(
                    {
                        "unknowns": budget,
                        "solved": solved,
                        "predicted": predicted,
                        "winner_solved": win_s,
                        "winner_predicted": win_p,
                    }
                )
                print(
                    f"  {budget:8d} unknowns: "
                    + "; ".join(f"p{p} {solved[p]:.2e} (pred {predicted[p]:.2e})" for p in solved)
                    + f" | winner p{win_s}, predicted p{win_p}"
                )
            for target in TARGETS:
                need = {}
                for p, rows in degrees.items():
                    errs = np.array([r["error"] for r in rows])
                    u = first_crossing(target, errs, np.array([r["unknowns"] for r in rows], dtype=float))
                    s = first_crossing(
                        target, errs, np.array([max(r["kernel_s"] + r["solve_s"], 1e-3) for r in rows])
                    )
                    if u is not None:
                        need[p] = {"unknowns": u, "seconds": s}
                entry["targets"].append({"error": target, "need": need})
                print(
                    f"  error {target:.0e}: "
                    + "; ".join(
                        f"p{p} {v['unknowns']:.3g} unknowns {v['seconds']:.3g} s" for p, v in need.items()
                    )
                )
            record.append(entry)
    out = HERE / "analyse_degree_vs_size.json"
    out.write_text(json.dumps(record, indent=2) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
