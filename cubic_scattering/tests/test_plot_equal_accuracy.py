"""The equal-accuracy figure's coordinates, and the numbers the Error paper quotes from the same ladders."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import equal_accuracy as ea  # noqa: E402
import plot_equal_accuracy as pea  # noqa: E402


def _ladder(ka: float, arm: str) -> list[dict]:
    rows = [r for r in ea.load(ea.ROWS) if r["body"] == "graded" and r["ka_s"] == ka and r["arm"] == arm]
    return sorted(rows, key=lambda r: r["n"])


def test_one_table_per_panel_and_arm(tmp_path: Path) -> None:
    written = pea.write_ladders(ea.ROWS, tmp_path)
    assert sorted(p.name for p in written) == sorted(
        pea.data_name(ka, arm) for ka in pea.PANELS for arm in pea.ARMS
    )
    for ka in pea.PANELS:
        for arm in pea.ARMS:
            lines = (tmp_path / pea.data_name(ka, arm)).read_text().splitlines()
            assert lines[0] == "n unknowns error wall"
            ladder = _ladder(ka, arm)
            assert len(lines) - 1 == len(ladder) >= 5
            for line, r in zip(lines[1:], ladder, strict=True):
                n, unknowns, error, _ = line.split()
                assert int(n) == r["n"] and int(unknowns) == r["unknowns"]
                assert abs(float(error) / r["error"] - 1) < 1e-6


def test_committed_tables_are_current(tmp_path: Path) -> None:
    """The figure's committed coordinates are the ones the ladders give now."""
    pea.write_ladders(ea.ROWS, tmp_path)
    for path in tmp_path.iterdir():
        assert (pea.FIG / path.name).read_text() == path.read_text(), (
            f"rerun scripts/plot_equal_accuracy.py ({path.name})"
        )


def test_missing_arm_refused(tmp_path: Path) -> None:
    rows = [r for r in ea.load(ea.ROWS) if not (r["arm"] == "hierarchy" and r["ka_s"] == 4.0)]
    src = tmp_path / "rows.jsonl"
    src.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    with pytest.raises(
        ValueError, match="no hierarchy rows .* k_S a = 4.0; run python scripts/compare_point_galerkin.py"
    ):
        pea.write_ladders(src, tmp_path)


#: (k_S a, target, arm) -> (unknowns, status) as the paper's table prints them; None where it says not reached
PAPER = {
    (0.5, 1e-3, "point"): (5_632, "interpolated"),
    (0.5, 1e-3, "galerkin"): None,
    (0.5, 1e-4, "point"): (69_610, "interpolated"),
    (0.5, 1e-4, "galerkin"): (10_610, "interpolated"),
    (0.5, 1e-4, "hierarchy"): (14_691, "interpolated"),
    (0.5, 3e-5, "point"): (421_452, "interpolated"),
    (0.5, 3e-5, "galerkin"): (26_187, "interpolated"),
    (0.5, 3e-5, "hierarchy"): (35_893, "interpolated"),
    (0.5, 1e-5, "point"): None,
    (0.5, 1e-5, "galerkin"): (60_305, "interpolated"),
    (0.5, 1e-5, "hierarchy"): (82_459, "interpolated"),
    (0.5, 1e-6, "galerkin"): (317_376, "measured"),
    (0.5, 1e-6, "hierarchy"): (437_847, "interpolated"),
    (4.0, 1e-3, "point"): (933_316, "interpolated"),
    (4.0, 1e-3, "galerkin"): (38_204, "interpolated"),
    (4.0, 1e-3, "hierarchy"): (59_983, "interpolated"),
    (4.0, 3e-4, "point"): None,
    (4.0, 3e-4, "galerkin"): (88_004, "interpolated"),
    (4.0, 3e-4, "hierarchy"): (134_023, "interpolated"),
    (4.0, 1e-4, "galerkin"): (192_727, "interpolated"),
    (4.0, 1e-4, "hierarchy"): (280_060, "interpolated"),
    (4.0, 1e-5, "galerkin"): (999_775, "interpolated"),
    (4.0, 1e-5, "hierarchy"): None,
    # the classical point cell done exactly: the headline
    (0.5, 1e-3, "point_exact"): (3_264, "interpolated"),
    (0.5, 1e-4, "point_exact"): (11_247, "interpolated"),
    (0.5, 3e-5, "point_exact"): (23_893, "interpolated"),
    (0.5, 1e-5, "point_exact"): (98_637, "interpolated"),
    (0.5, 3e-6, "point_exact"): (679_751, "interpolated"),
    (0.5, 3e-6, "galerkin"): (143_734, "interpolated"),
    (0.5, 3e-6, "hierarchy"): (195_051, "interpolated"),
    (0.5, 1e-6, "point_exact"): None,
    (4.0, 1e-3, "point_exact"): (26_878, "interpolated"),
    (4.0, 3e-4, "point_exact"): (155_304, "measured"),
    (4.0, 1e-4, "point_exact"): (803_610, "interpolated"),
    (4.0, 1e-5, "point_exact"): None,
}


@pytest.mark.parametrize("key", sorted(PAPER, key=str))
def test_paper_table_entries(key: tuple[float, float, str]) -> None:
    ka, target, arm = key
    try:
        got = ea.cost_at_error(_ladder(ka, arm), target, "unknowns")
    except ValueError:
        got = None
    want = PAPER[key]
    if want is None:
        assert got is None
    else:
        assert got is not None and round(got[0]) == want[0] and got[1] == want[1]
