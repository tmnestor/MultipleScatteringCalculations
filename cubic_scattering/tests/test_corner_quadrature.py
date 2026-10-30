"""Adaptive cubature at the corner of the octant (Mathematica/CubeMomentCornerQuadrature.wl), read from its JSON.

The script integrates every index class of the grades (1,3), (0,4) and (1,5) with three settings and compares each
with its closed form. These tests pin what the papers cite: Duffy coordinates are clean and agree below 1e-12 on all
33 classes; plain GlobalAdaptive warns NIntegrate::eincr on every class of grade (1,3) and on no other, while still
within the check's 1e-8; raising MaxErrorIncreases is tried only where plain warns; and no message that is switched
off by default (General::newsym) is recorded as issued.
"""

import json
from pathlib import Path

import pytest

DATA = Path(__file__).resolve().parents[2] / "Mathematica" / "CubeMoment_corner_quadrature.json"


@pytest.fixture(scope="module")
def rows() -> list[dict]:
    if not DATA.exists():
        pytest.fail(f"{DATA} is missing: run wolframscript -file Mathematica/CubeMomentCornerQuadrature.wl")
    return json.loads(DATA.read_text())


def _by(rows: list[dict], method: str) -> list[dict]:
    return [r for r in rows if r["method"] == method]


def test_every_class_integrated(rows: list[dict]) -> None:
    for method in ("plain", "duffy"):
        grades = [tuple(r["grade"]) for r in _by(rows, method)]
        assert grades.count((1, 3)) == 9 and grades.count((0, 4)) == 6 and grades.count((1, 5)) == 18


def test_duffy_clean_and_exact(rows: list[dict]) -> None:
    for r in _by(rows, "duffy"):
        assert r["messages"] == [] and r["relative_error"] < 1e-12, r


def test_plain_warns_on_grade_one_three_only(rows: list[dict]) -> None:
    for r in _by(rows, "plain"):
        warned = "NIntegrate::eincr" in r["messages"]
        assert warned == (tuple(r["grade"]) == (1, 3)), r
        assert r["relative_error"] < 1e-8, r


def test_maxerr_only_where_plain_warns(rows: list[dict]) -> None:
    warned = {(tuple(r["d"]), tuple(r["w"])) for r in _by(rows, "plain") if r["messages"]}
    assert {(tuple(r["d"]), tuple(r["w"])) for r in _by(rows, "maxerr")} == warned


def test_no_switched_off_message_recorded(rows: list[dict]) -> None:
    assert all("General::newsym" not in r["messages"] for r in rows)
