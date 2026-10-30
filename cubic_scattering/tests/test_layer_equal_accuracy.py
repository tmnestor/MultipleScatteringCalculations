"""The layer's Galerkin-against-hierarchy comparison (Mathematica/ContinuumLimit_EqualAccuracy.wl), read from its JSON.

The script runs both schemes at 40 and 60 digits; these tests check what it wrote: each scheme at its predicted
order, the unknowns a cell from the solvers' definitions, and every equal-accuracy entry either measured,
interpolated or refused with a reason, never extrapolated.
"""

import json
from pathlib import Path

import pytest

DATA = Path(__file__).resolve().parents[2] / "Mathematica" / "ContinuumLimit_equal_accuracy.json"
ORDERS = {("galerkin", 0): 2, ("galerkin", 1): 4, ("galerkin", 2): 6, ("galerkin", 3): 8,
          ("hierarchy", 1): 2, ("hierarchy", 3): 4, ("hierarchy", 5): 6, ("hierarchy", 7): 8}  # fmt: skip


@pytest.fixture(scope="module")
def data() -> dict:
    if not DATA.exists():
        pytest.fail(
            f"{DATA} is missing: run wolframscript -file Mathematica/ContinuumLimit_EqualAccuracy.wl"
        )
    return json.loads(DATA.read_text())


def _key(entry: dict) -> tuple[str, int]:
    return (entry["scheme"], entry["degree"] if entry["scheme"] == "galerkin" else entry["q"])


def test_every_scheme_at_its_order(data: dict) -> None:
    orders = {_key(o): o for o in data["orders"]}
    assert set(orders) == set(ORDERS)
    for key, want in ORDERS.items():
        assert orders[key]["order"] == want
        assert abs(orders[key]["measured_order"] - want) < 0.15, (key, orders[key]["measured_order"])


def test_unknowns_a_cell_from_the_solvers(data: dict) -> None:
    for o in data["orders"]:
        expected = 2 * (o["degree"] + 1) if o["scheme"] == "galerkin" else o["q"] + 1
        assert o["perCell"] == expected


def test_equal_accuracy_entries_never_extrapolated(data: dict) -> None:
    assert data["equal_accuracy"]
    for e in data["equal_accuracy"]:
        for arm in ("galerkin", "hierarchy"):
            status = e[f"{arm}_status"]
            assert status in ("measured", "interpolated", "outside the ladder", "not monotone")
            if status in ("measured", "interpolated"):
                assert e[f"{arm}_cells"] > 0 and e[f"{arm}_unknowns"] > 0
            else:
                assert e[f"{arm}_cells"] is None and e[f"{arm}_unknowns"] is None
