#!/usr/bin/env python3
"""Emit \\input-able LaTeX booktabs fragments from the evidence CSVs."""

import csv
from collections.abc import Callable
from pathlib import Path

EV = Path(__file__).resolve().parent
TAB = EV / "tables"


def _tex_escape(s: str) -> str:
    """Escape LaTeX-special characters in a cell string.

    Args:
        s: Raw cell value string.

    Returns:
        String safe for use in LaTeX text mode.
    """
    return (
        s.replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("%", r"\%")
        .replace("&", r"\&")
        .replace("#", r"\#")
    )


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
        fmt_values = [
            _tex_escape(col_formats[c](r[c])) if c in col_formats else _tex_escape(r[c])
            for c in cols
        ]
        lines.append(" & ".join(fmt_values) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]

    TAB.mkdir(parents=True, exist_ok=True)
    out = TAB / f"{label}.tex"
    out.write_text("\n".join(lines) + "\n")
    return out


def _fmt_sci(v: str) -> str:
    """Format a numeric string to 2-digit scientific notation."""
    return f"{float(v):.2e}"


def dump_pgfplots() -> None:
    """Write per-representation pgfplots .dat files from ``cost_accuracy.csv``.

    Filters to ``contrast == "moderate"``, ``pol == "P"``, ``status == "ok"``
    and writes one whitespace-separated file per representation into
    ``paper/evidence/tables/``:

    - ``error_vs_ka_born.dat``
    - ``error_vs_ka_eshelby.dat``
    - ``error_vs_ka_t9.dat``

    Each file has a header line ``ka l2`` followed by rows sorted by ka ascending.
    """
    rows = list(csv.DictReader((EV / "cost_accuracy.csv").open()))
    filtered = [
        r
        for r in rows
        if r["contrast"] == "moderate" and r["pol"] == "P" and r["status"] == "ok"
    ]

    reps = ("born", "eshelby", "t9")
    TAB.mkdir(parents=True, exist_ok=True)
    for rep in reps:
        rep_rows = sorted(
            [r for r in filtered if r["rep"] == rep], key=lambda r: float(r["ka"])
        )
        out = TAB / f"error_vs_ka_{rep}.dat"
        lines = ["ka l2"]
        for r in rep_rows:
            lines.append(f"{r['ka']} {r['l2']}")
        out.write_text("\n".join(lines) + "\n")


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

    # 3. pgfplots .dat files for F6 (error vs ka per representation)
    dump_pgfplots()


if __name__ == "__main__":
    main()
