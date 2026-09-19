#!/usr/bin/env python3
"""Do the write-up's claimed check tallies match what the gates actually report?

WHY THIS EXISTS
---------------
``FirstOrderContrastOperator.tex`` states, in the caption of its summary table,
how many checks stand behind each object -- "the T-matrix assembly 16/16", and
so on.  Those numbers are written by hand; the gates are not.  On 2026-09-19 a
check-by-check audit found the caption claiming 14/14 for a gate that had
reported 16 for hours, three rows filed under the wrong object, and two rows
describing checks that no longer existed.  None of it touched a result.  All of
it made the paper's own account of its evidence wrong.

This closes that loop mechanically.  It is deliberately narrow: it checks the
CAPTION's explicit "N/N" claims, which are unambiguous, and not the per-row
counts in the table body, which are editorial groupings and would produce false
alarms.

It is not a physics gate.  It asserts nothing about the formulation; it only
refuses to let the paper misreport how much evidence it has.

Run:  conda run -n seismic python scripts/check_paper_tallies.py
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEX = ROOT / "LatexPDFs" / "FirstOrderContrastOperator" / "FirstOrderContrastOperator.tex"

#: Phrase in the caption -> gate script that must report that tally.  The
#: symbolic scripts (25/25, 44/44) are Mathematica and are not run here.
CLAIMS = {
    "T$-matrix assembly": "gate_first_order_tmatrix",
    "Schwinger settlement": "gate_first_order_schwinger",
    "layer comparison": "gate_first_order_layer_vs_kennett",
    "thesis spectral representation": "gate_thesis_spectral",
    "slab comparison": "gate_first_order_slab_vs_kennett",
    "static moments": "gate_first_order_static_moments",
    "Riccati blocks": "gate_first_order_riccati_blocks",
    "the impedance": "gate_first_order_impedance",
    "its march": "gate_first_order_impedance_march",
    "layered $\\boldsymbol\\Gamma$": "gate_first_order_layered_gamma",
    "lateral impedance march": "gate_first_order_lateral_impedance",
}

TALLY = re.compile(r"(\d+)/(\d+) checks passed|(\d+) passed, (\d+) failed")


def claimed(caption: str, phrase: str) -> tuple[int, int] | None:
    """The N/N the caption states just after `phrase`.

    Args:
        caption: The caption text.
        phrase: The phrase naming the gate.

    Returns:
        (numerator, denominator), or None if the phrase is absent.
    """
    i = caption.find(phrase)
    if i < 0:
        return None
    m = re.search(r"\$(\d+)/(\d+)\$", caption[i : i + 160])
    return (int(m.group(1)), int(m.group(2))) if m else None


def actual(gate: str) -> tuple[int, int] | None:
    """What the gate reports when run.

    Args:
        gate: Script stem in ``scripts/``.

    Returns:
        (passed, total), or None if it could not be parsed.
    """
    out = subprocess.run(  # noqa: S603
        ["conda", "run", "-n", "seismic", "python", f"scripts/{gate}.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    ).stdout
    hits = list(TALLY.finditer(out))
    if not hits:
        return None
    m = hits[-1]  # the summary line, not an intermediate mention
    if m.group(1):
        return int(m.group(1)), int(m.group(2))
    return int(m.group(3)), int(m.group(3)) + int(m.group(4))


def main() -> int:
    """Compare every caption tally against its gate.

    Returns:
        0 if all agree, 1 otherwise.
    """
    text = TEX.read_text()
    try:
        i = text.index("All checks pass")
        j = text.index("\\toprule", i)
    except ValueError:
        print(
            "Could not find the summary caption.\n"
            f"  Where: {TEX}\n"
            '  Expected: a caption beginning "All checks pass", followed by the\n'
            "            table's \\toprule -- the summary table is a longtable,\n"
            "            so its caption precedes the header rows.\n"
            "  Fix:   restore that caption, or update the anchors here if the\n"
            "         table's structure has deliberately changed."
        )
        return 1
    # Collapse whitespace: LaTeX wraps the caption, and a phrase split across a
    # line break would otherwise read as absent -- which is a false alarm, and a
    # checker that cries wolf is worse than none.
    caption = " ".join(text[i:j].split())

    print("=" * 74)
    print("  The write-up's claimed tallies, against the gates")
    print("=" * 74)
    print(f"    {'gate':<40} {'claimed':>9} {'actual':>9}")
    bad = 0
    for phrase, gate in CLAIMS.items():
        cl, ac = claimed(caption, phrase), actual(gate)
        if cl is None:
            print(f"    {gate:<40} {'ABSENT':>9} {'--':>9}  <- not claimed at all")
            bad += 1
            continue
        if ac is None:
            print(f"    {gate:<40} {f'{cl[0]}/{cl[1]}':>9} {'UNPARSED':>9}")
            bad += 1
            continue
        ok = cl == ac and ac[0] == ac[1]
        note = "" if ok else "  <- MISMATCH"
        print(f"    {gate:<40} {f'{cl[0]}/{cl[1]}':>9} {f'{ac[0]}/{ac[1]}':>9}{note}")
        bad += 0 if ok else 1

    print("")
    if bad:
        print(f"  {bad} claim(s) wrong.  Fix the caption, or the gate, before circulating.")
    else:
        print(f"  all {len(CLAIMS)} claims match, and every gate is green")
    print("=" * 74)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
