"""The face-contact bias of a tensor Gauss rule, as measured by scripts/measure_face_contact_gauss_bias.py.

Two cubes in face contact: the double-cube tensor Gauss rule shares one lateral node set between the cubes, so it
samples the singular ray of the 1/w^3 kernel with O(1) weight. Refined, its value settles (each step smaller than
the last), but to a limit about three times the true one. These tests read the measurement's JSON and recompute
its cheapest row.
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "scripts" / "data" / "face_contact_gauss_bias.json"
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="module")
def data() -> dict:
    if not DATA.exists():
        pytest.fail(f"{DATA} is missing: run python scripts/measure_face_contact_gauss_bias.py")
    return json.loads(DATA.read_text())


def test_two_bias_free_routes_agree(data: dict) -> None:
    exact = data["exact"]
    assert abs(exact["subdivision"] - exact["dyadic"]) <= 1e-6 * abs(exact["subdivision"])


def test_refinement_settles_on_a_biased_limit(data: dict) -> None:
    rows = data["rows"]
    assert [r["n"] for r in rows] == list(range(4, 17, 2))
    steps = [abs(r["step"]) for r in rows[1:]]
    assert all(b < a for a, b in zip(steps, steps[1:], strict=False)), steps  # looks converged
    assert rows[-1]["gauss"] / data["exact"]["subdivision"] > 3.0  # but three times too large


def test_cheapest_row_recomputed(data: dict) -> None:
    import face_s_rederivation as red

    assert red.biased_tensor_gauss(4) == pytest.approx(data["rows"][0]["gauss"], rel=1e-13)
