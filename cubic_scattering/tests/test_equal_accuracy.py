"""Cost at equal error, read off a measured ladder: interpolation only, never extrapolation."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import equal_accuracy as ea  # noqa: E402


def ladder(arm: str = "galerkin") -> list[dict]:
    """error = 3 n^-4 and unknowns = 36 n^3: unknowns = 36 (3 / error)^(3/4) exactly, a straight line in log-log."""
    return [
        {"body": "graded", "ka_s": 0.5, "arm": arm, "n": n, "error": 3.0 * n**-4.0, "unknowns": 36 * n**3,
         "wall_s": 0.01 * n**3}
        for n in range(4, 41, 4)
    ]  # fmt: skip


def test_power_law_recovered() -> None:
    value, status = ea.cost_at_error(ladder(), 3e-4)  # n = 10, between the rows at 8 and 12
    assert status == "interpolated"
    assert abs(value - 36_000.0) <= 1e-12 * 36_000.0


def test_a_rows_own_error_is_measured() -> None:
    value, status = ea.cost_at_error(ladder(), 3.0 * 8.0**-4)
    assert status == "measured" and abs(value - 36 * 512) <= 1e-9 * 36 * 512


def test_wall_time_by_key() -> None:
    value, status = ea.cost_at_error(ladder(), 3e-4, key="wall_s")
    assert status == "interpolated" and abs(value - 10.0) <= 1e-12 * 10.0


def test_outside_range_refused() -> None:
    with pytest.raises(ValueError, match="extend the ladder") as err:
        ea.cost_at_error(ladder(), 1e-9)
    message = str(err.value)
    assert "galerkin" in message and "graded" in message and "0.5" in message
    assert f"{3.0 * 40**-4:.3g}" in message  # the smallest error the ladder reaches


def test_non_monotone_bracket_refused() -> None:
    rows = ladder()
    rows[3]["error"], rows[4]["error"] = (
        rows[4]["error"],
        rows[3]["error"] * 1.5,
    )  # errors rise from n = 16 to 20
    with pytest.raises(ValueError, match="does not fall"):
        ea.cost_at_error(rows, 0.5 * (rows[3]["error"] + rows[4]["error"]))


def test_equal_accuracy_table(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    # the point ladder stops at n = 8 (error 7.3e-4), short of the target 3e-4
    rows = ladder("galerkin") + ladder("point")[:2] + [
        {**r, "body": "sharp"} for r in ladder("galerkin")
    ]  # fmt: skip
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    table = ea.equal_accuracy_table(path, "graded", 0.5, (3e-4,))
    by_arm = {t["arm"]: t for t in table}
    assert set(by_arm) == {"galerkin", "point"}
    assert by_arm["galerkin"]["status"] == "interpolated"
    assert abs(by_arm["galerkin"]["unknowns"] - 36_000.0) <= 1e-9 * 36_000.0
    assert by_arm["point"]["unknowns"] is None and "extend the ladder" in by_arm["point"]["reason"]
