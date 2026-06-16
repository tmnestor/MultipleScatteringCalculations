import csv as _csv

from run_evidence import (  # noqa: E402
    capture,
    evidence_formfactor,
    evidence_optical_theorem,
    evidence_radiation_need,
    evidence_radiation_reaction,
    evidence_t27_intervoxel,
)


def test_capture_writes_log(tmp_path):
    log = tmp_path / "echo.log"
    rc, out = capture(["python", "-c", "print('hello-evidence')"], log)
    assert rc == 0
    assert "hello-evidence" in out
    assert log.read_text().strip() == "hello-evidence"


def test_formfactor_evidence_anchor():
    """Smoke test: single-layer Foldy-Lax vs Kennett rel-err stays below 5%; min < 1%.

    Validated values (omega=150 rad/s, moderate contrast, a in [2.0, 0.25]):
        rel_err ~ 8.8e-3 → 3.1e-3  (0.88% → 0.31% as cell size shrinks)
    History note: the originally expected range was ~1.1% → 0.31%; measured values are
    slightly lower at the coarsest mesh (0.88% vs 1.1%) — physics unchanged, just a
    different omega/contrast point in the script.
    """
    csv_path = evidence_formfactor()
    rows = list(_csv.DictReader(csv_path.open()))
    errs = [float(r["rel_err"]) for r in rows]
    assert all(e < 0.05 for e in errs), f"rel-err exceeded 5%: {errs}"
    assert min(errs) < 0.01, f"best rel-err not sub-1%: {min(errs)}"


def test_radiation_reaction_gates_pass():
    csv_path = evidence_radiation_reaction()
    rows = list(_csv.DictReader(csv_path.open()))
    assert all(r["passed"] == "True" for r in rows), (
        "a radiation-reaction Mie gate failed"
    )


def test_optical_theorem_gate_passes():
    csv_path = evidence_optical_theorem()
    rows = list(_csv.DictReader(csv_path.open()))
    assert all(r["passed"] == "True" for r in rows), "optical-theorem gate failed"


def test_t27_intervoxel_negligible():
    """Smoke test: T₂₇ quadratic inter-voxel coupling stays ≤0.1% across all cases.

    The d(i,ii) column (PRIMARY: quadratic channels OFF in P, M-orthogonalised)
    measures the fractional observable change from zeroing the 18 quadratic
    coupling channels.  Historically max ~0.07% (SV edge ka=0.5).
    """
    csv_path = evidence_t27_intervoxel()
    rows = list(_csv.DictReader(csv_path.open()))
    val = float(rows[0]["value_percent"])
    assert val <= 0.1, f"inter-voxel coupling not negligible: {val:.4f}%"


def test_radiation_need_runs():
    """Smoke test: radiation-part-need probe runs and produces data rows.

    The script measures |Im/Re| of the inter-voxel propagator G/C/S blocks
    (Measurement A) and the slab-R_PP propagator contribution (Measurement B).
    The CSV captures all lines with >=3 floats — must be non-empty.
    """
    csv_path = evidence_radiation_need()
    assert csv_path.exists() and csv_path.stat().st_size > 0
