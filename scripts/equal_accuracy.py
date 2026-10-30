#!/usr/bin/env python3
"""The cost of each arm at equal error, read off the measured ladders of ``compare_point_galerkin.py``.

A ladder is one arm's rows on one body at one frequency: grid sizes n with their error and cost (unknowns, wall
time). The cost at a target error is read by linear interpolation of log(cost) against log(error) between the
two consecutive rows whose errors bracket the target. That is exact for a power law, error ~ n^-p with cost ~
n^3, which is what a converging scheme follows between neighbouring grids; and a row within 2% of the target is
reported as measured. Nothing is ever extrapolated: a target outside the ladder's errors is refused, with the
range the ladder reaches, and so is a target the ladder crosses more than once or across a pair of rows whose
error does not fall (the sharp sphere's staircase makes its errors rise and fall with the grid, and a value
read between such rows means nothing).

Run:  python scripts/equal_accuracy.py --body=graded --ka=0.5 1e-3 1e-4 1e-5 1e-6
      (rows from scripts/data/point_vs_galerkin.jsonl; --path=FILE for another)
"""

import json
import math
import sys
from pathlib import Path

ROWS = Path(__file__).resolve().parent / "data" / "point_vs_galerkin.jsonl"
MEASURED_WITHIN = 0.02  # a row this close to the target (relative) is the measurement itself


def _where(rows: list[dict]) -> str:
    r = rows[0]
    return f"the {r['arm']} arm on the {r['body']} sphere at k_S a = {r['ka_s']}"


def cost_at_error(rows: list[dict], target: float, key: str = "unknowns") -> tuple[float, str]:
    """The cost ``key`` of one arm's ladder at error ``target``: ``(value, "measured" | "interpolated")``.

    Args:
        rows: one arm's rows on one body at one frequency, each with ``n``, ``error`` and ``key``.
        target: the error at which the cost is wanted.
        key: the cost read, ``"unknowns"`` or ``"wall_s"``.

    Raises:
        ValueError: the target lies outside the ladder's errors, or the ladder crosses it more than once or
            across a pair of rows whose error does not fall.
    """
    ladder = sorted(rows, key=lambda r: r["n"])
    errors = [r["error"] for r in ladder]
    lo, hi = min(errors), max(errors)
    if not lo <= target <= hi:
        raise ValueError(
            f"{_where(ladder)}: the target error {target:.3g} is outside the errors its ladder reaches, "
            f"{lo:.3g} to {hi:.3g} (n = {ladder[0]['n']} to {ladder[-1]['n']}); "
            "extend the ladder (finer grids for a smaller error, coarser for a larger) rather than extrapolate"
        )
    for r in ladder:
        if abs(r["error"] / target - 1.0) <= MEASURED_WITHIN:
            return float(r[key]), "measured"
    brackets = [
        (a, b) for a, b in zip(ladder, ladder[1:], strict=False) if min(a["error"], b["error"]) <= target <= max(a["error"], b["error"])
    ]  # fmt: skip
    rising = [(a, b) for a, b in brackets if b["error"] >= a["error"]]
    if rising or len(brackets) != 1:
        a, b = (rising or brackets)[0]
        raise ValueError(
            f"{_where(ladder)}: the error does not fall monotonically through the target {target:.3g} "
            f"(it crosses it {len(brackets)} times; from n = {a['n']} to {b['n']} it goes "
            f"{a['error']:.3g} to {b['error']:.3g}); the cost at that error cannot be read off this ladder: "
            "report its rows as they are"
        )
    (a, b) = brackets[0]
    t = (math.log(target) - math.log(a["error"])) / (math.log(b["error"]) - math.log(a["error"]))
    return math.exp(math.log(a[key]) + t * (math.log(b[key]) - math.log(a[key]))), "interpolated"


def load(path: Path) -> list[dict]:
    """Every row of a JSONL file; a rerun of the same arm, body, frequency and n replaces the earlier row."""
    latest: dict[tuple, dict] = {}
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            latest[(r["body"], r["ka_s"], r["arm"], r["n"])] = r
    return list(latest.values())


def equal_accuracy_table(path: Path, body: str, ka: float, targets: tuple[float, ...]) -> list[dict]:
    """For each target error and arm: unknowns and wall time at that error, with the status or the reason not.

    Returns:
        Rows ``{"target", "arm", "unknowns", "wall_s", "status", "reason"}``; an arm whose ladder cannot give a
        target has ``unknowns = wall_s = status = None`` and the refusal in ``reason``.
    """
    rows = [r for r in load(path) if r["body"] == body and r["ka_s"] == ka]
    arms = sorted({r["arm"] for r in rows})
    table = []
    for target in targets:
        for arm in arms:
            ladder = [r for r in rows if r["arm"] == arm]
            try:
                unknowns, status = cost_at_error(ladder, target, "unknowns")
                wall, _ = cost_at_error(ladder, target, "wall_s")
                table.append(
                    {
                        "target": target,
                        "arm": arm,
                        "unknowns": unknowns,
                        "wall_s": wall,
                        "status": status,
                        "reason": "",
                    }
                )
            except ValueError as err:
                table.append(
                    {
                        "target": target,
                        "arm": arm,
                        "unknowns": None,
                        "wall_s": None,
                        "status": None,
                        "reason": str(err),
                    }
                )
    return table


def main() -> int:
    body: str | None = None
    ka: float | None = None
    path = ROWS
    targets: list[float] = []
    for a in sys.argv[1:]:
        if a.startswith("--body="):
            body = a.split("=", 1)[1]
        elif a.startswith("--ka="):
            ka = float(a.split("=", 1)[1])
        elif a.startswith("--path="):
            path = Path(a.split("=", 1)[1])
        else:
            targets.append(float(a))
    if body is None or ka is None or not targets:
        print(__doc__)
        return 2
    print(f"{body} sphere, k_S a = {ka}: cost at equal error ({path.name})")
    print("  target     arm          unknowns      wall      status")
    for t in equal_accuracy_table(path, body, ka, tuple(targets)):
        if t["unknowns"] is None:
            print(
                f"  {t['target']:.0e}  {t['arm']:10s}  {'-':>12s}  {'-':>8s}    not reached: {t['reason']}"
            )
        else:
            print(
                f"  {t['target']:.0e}  {t['arm']:10s}  {t['unknowns']:12,.0f}  {t['wall_s']:7.1f} s  {t['status']}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
