#!/usr/bin/env python3
"""Emit \\input-able LaTeX booktabs fragments from the evidence CSVs."""

import csv
from collections.abc import Callable
from pathlib import Path

EV = Path(__file__).resolve().parent
TAB = EV / "tables"


def _table(
    csv_name: str,
    cols: list[str],
    headers: list[str],
    caption: str,
    label: str,
    row_filter: Callable[[dict[str, str]], bool] | None = None,
    col_formats: dict[str, Callable[[str], str]] | None = None,
) -> Path:
    """Write a booktabs tabular fragment to ``TAB/<label>.tex``.

    Args:
        csv_name: Filename of the source CSV relative to ``EV``.
        cols: Column keys to include (in order).
        headers: LaTeX header strings (same length as ``cols``).
        caption: Caption text — written as a leading ``% caption:`` comment.
        label: Output filename stem (also used as the LaTeX label).
        row_filter: Optional callable; rows for which it returns ``False``
            are excluded.  ``None`` keeps all rows.
        col_formats: Optional mapping from column key to a format callable.
            Missing keys use ``str`` (identity).

    Returns:
        Path to the written ``.tex`` fragment.
    """
    if col_formats is None:
        col_formats = {}

    rows = list(csv.DictReader((EV / csv_name).open()))
    if row_filter is not None:
        rows = [r for r in rows if row_filter(r)]

    col_spec = "l" * len(cols)
    lines: list[str] = [
        f"% caption: {caption}",
        r"\begin{tabular}{" + col_spec + "}",
        r"\toprule",
        " & ".join(headers) + r" \\",
        r"\midrule",
    ]
    for r in rows:
        fmt_values = [col_formats.get(c, str)(r[c]) for c in cols]
        lines.append(" & ".join(fmt_values) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]

    TAB.mkdir(parents=True, exist_ok=True)
    out = TAB / f"{label}.tex"
    out.write_text("\n".join(lines) + "\n")
    return out


def _fmt_sci(v: str) -> str:
    """Format a numeric string to 2-digit scientific notation."""
    return f"{float(v):.2e}"


def main() -> None:
    """Generate all LaTeX booktabs table fragments into ``paper/evidence/tables/``."""
    # 1. Cost-accuracy table — P-incidence only (SV/SH far field undefined)
    _table(
        csv_name="cost_accuracy.csv",
        cols=["contrast", "ka", "pol", "rep", "l2", "linf"],
        headers=["Contrast", r"$ka$", "Pol.", "Rep.", r"$L_2$", r"$L_\infty$"],
        caption=(
            "Point-scatterer representation accuracy vs.\\ the exact finite-size T9"
            " far field for P-incidence (SV/SH single-site far field is undefined"
            " for all point representations)."
        ),
        label="tab_cost_accuracy",
        row_filter=lambda r: r["status"] == "ok",
        col_formats={"l2": _fmt_sci, "linf": _fmt_sci},
    )

    # 2. Form-factor / Kennett convergence table
    _table(
        csv_name="formfactor_kennett.csv",
        cols=["a", "ka_S", "rel_err"],
        headers=[r"$a$ (m)", r"$ka_S$", r"Rel.\ error"],
        caption="Volume-averaged T9 slab vs.\\ Kennett reflectivity: relative error.",
        label="tab_formfactor",
    )


if __name__ == "__main__":
    main()
