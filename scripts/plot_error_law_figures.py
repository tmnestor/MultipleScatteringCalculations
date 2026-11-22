#!/usr/bin/env python3
"""The coordinates of the Error paper's two figures on the three-dimensional error law.

Fig. fig:twotermlaw (``LatexPDFs/PolynomialVoxelError/figures/fig_two_term_law.tex``): the two-term
separation against the number of cells across the body.
  data_twoterm_ratio_p<p>.dat           n, error / E, rest / E at k_S a = 0.5 (panel a)
  data_twoterm_freq_ka<ka>_p<p>.dat     n, rest / (nu E) at each frequency (panels b and c)

Fig. fig:factorfreq (``.../fig_factor_frequency.tex``): the factor F of the second term against the
scattering angle, the scheme and the prediction, a panel for each frequency.
  data_factor_p<p>_ka<ka>_<P|S>.dat     angle (degrees), F of the scheme, F predicted, F static

The numbers are read from the result files the measurement scripts wrote beside themselves
(``measure_two_term_fft_*.json``, ``measure_t2_factor_fft_*.json``, ``measure_t2_factor_dynamic_*.json``,
``measure_t2_factor_operators_*.json``); nothing is computed here but ratios and, for degree-one cells,
the second-order extrapolation of the prediction from its two finest grids. For degree-one cells the
column ``static`` is the prediction at the lowest frequency, k_S a = 0.125.

Run:  python scripts/plot_error_law_figures.py     (then lualatex each figure in the figures directory)
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
FIGS = ROOT / "LatexPDFs/PolynomialVoxelError/figures"
FREQUENCIES = ("0.5", "1", "2", "4")
FACTOR_FREQUENCIES = ("0.125", "1", "2", "4")
RIGHT_ANGLE = 4  # the index of the angle pi / 2, where the outgoing S wave of a stiffness contrast vanishes


def load(name: str) -> dict:
    path = SCRIPTS / name
    if not path.exists():
        raise SystemExit(
            f"plot_error_law_figures: missing result file {path}. Fix: run the measurement script named in "
            "the file's stem with the arguments its name records."
        )
    return json.loads(path.read_text())


def write(name: str, header: str, rows: list[tuple | None]) -> None:
    """Rows of numbers; a row given as None becomes a blank line, at which the figure breaks its curve."""
    lines = [header] + ["" if row is None else " ".join(f"{v:.6g}" for v in row) for row in rows]
    (FIGS / name).write_text("\n".join(lines) + "\n")


def two_term() -> None:
    for p in (0, 1):
        d = load(f"measure_two_term_fft_ka0.5_core0.75_p{p}.json")
        write(
            f"data_twoterm_ratio_p{p}.dat",
            "n error_over_E rest_over_E",
            [(r["n"], r["error"] / r["E"], r["rest"] / r["E"]) for r in d["rows"]],
        )
        for ka in FREQUENCIES:
            d = load(f"measure_two_term_fft_ka{ka}_core0.75_p{p}.json")
            nu = d["nonlinear_fraction"]
            write(
                f"data_twoterm_freq_ka{ka}_p{p}.dat",
                "n rest_over_nuE",
                [(r["n"], r["rest"] / (r["E"] * nu)) for r in d["rows"]],
            )


def predicted_degree_one(ka: str) -> dict[str, list[float]]:
    """The degree-one prediction at zero cell size: its two finest fine grids, extrapolated at second
    order, the order of what the fine grid still approximates (the field scattered once is held as cell
    means). At k_S a = 2, where three grids were run, the three share that limit to 0.005."""
    have = [
        n
        for n in (32, 48, 64)
        if (SCRIPTS / f"measure_t2_factor_operators_ka{ka}_core0.5_p1_fine{n}.json").exists()
    ]
    if len(have) < 2:
        raise SystemExit(
            f"plot_error_law_figures: fewer than two fine grids stored for degree-one cells at k_S a = {ka}. "
            f"Fix: run scripts/measure_t2_factor_operators.py {ka} 0.5 1 <N> for N = 32 and 48."
        )
    n0, n1 = have[-2], have[-1]
    a, b = (load(f"measure_t2_factor_operators_ka{ka}_core0.5_p1_fine{n}.json") for n in (n0, n1))
    gain = 1.0 / ((n1 / n0) ** 2 - 1.0)
    return {
        lab: [
            y["F_predicted"] + (y["F_predicted"] - x["F_predicted"]) * gain
            for x, y in zip(a["outgoing_" + lab], b["outgoing_" + lab], strict=True)
        ]
        for lab in "PS"
    }


def factor() -> None:
    for p in (0, 1):
        for ka in FACTOR_FREQUENCIES:
            scheme = load(f"measure_t2_factor_fft_ka{ka}_core0.5_p{p}.json")
            row = scheme["rows"][-1]
            theta = np.degrees(scheme["theta"])
            if p == 0:
                d = load(f"measure_t2_factor_dynamic_ka{ka}_core0.5_fine48.json")
                pred = {lab: [r["F_predicted"] for r in d["outgoing_" + lab]] for lab in "PS"}
                static = {lab: [r["F_static"] for r in d["outgoing_" + lab]] for lab in "PS"}
            else:
                pred = predicted_degree_one(ka)
                low = predicted_degree_one("0.125")
                static = {lab: low[lab] for lab in "PS"}
            for lab in "PS":
                rows = [
                    None
                    if (lab == "S" and j == RIGHT_ANGLE)
                    else (theta[j], row["F_" + lab][j], pred[lab][j], static[lab][j])
                    for j in range(len(theta))
                ]
                write(f"data_factor_p{p}_ka{ka}_{lab}.dat", "angle scheme predicted static", rows)
            print(f"factor: p = {p}, k_S a = {ka}: scheme on {row['n']} cells")


def main() -> int:
    two_term()
    factor()
    print(f"wrote the coordinate files to {FIGS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
