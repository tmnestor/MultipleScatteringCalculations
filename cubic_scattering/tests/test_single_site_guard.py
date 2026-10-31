"""The single site of a lattice cell: chosen once in numerics.yml, and never bypassed.

``compute_cube_tmatrix`` carries far-field form-factor corrections derived for a lone cube; inside a lattice
they cost a factor of 7 to 11 in error against the uniform-strain closure (Error paper, Section 11). Every
solver that couples cells therefore takes its cell T-matrix from ``point_cell_self.lattice_cube_tmatrix``,
and the direct calls that remain are the lone-cube uses named below. A new direct call fails here until
it is named, with the reason, or routed through ``lattice_cube_tmatrix``.
"""

import ast
from pathlib import Path

import numpy as np
import pytest

from cubic_scattering import point_cell_self
from cubic_scattering.effective_contrasts import MaterialContrast, ReferenceMedium
from cubic_scattering.point_cell_self import (
    exact_cube_tmatrix,
    exact_point_tmatrix_9x9,
    lattice_cube_tmatrix,
    lattice_single_site,
    self_block_9x9,
)
from cubic_scattering.resonance_tmatrix import _sub_cell_tmatrix_9x9

PACKAGE = Path(point_cell_self.__file__).parent
REF = ReferenceMedium(alpha=5000.0, beta=3000.0, rho=2500.0)
CONTRAST = MaterialContrast(Dlambda=2e9, Dmu=1e9, Drho=100.0)

#: (module, enclosing function) -> why the lone-cube T-matrix is right there
ALLOWED = {
    ("cube_eshelby.py", "compute_cube_born_tmatrix"): "one cube against the exact sphere",
    ("cube_eshelby.py", "compute_cube_eshelby_factors"): "one cube against the exact sphere",
    (
        "resonance_tmatrix.py",
        "compute_resonance_tmatrix",
    ): "the full cube, whose internals cube_eshelby reads",
    (
        "sphere_scattering_fft_gpu.py",
        "solve_point_plane_wave.cell_tmatrix",
    ): "single_site='package', explicit",
    ("point_cell_self.py", "lattice_cube_tmatrix"): "the configured choice itself",
}


def _direct_calls() -> set[tuple[str, str]]:
    """(module, dotted enclosing function) of every call of compute_cube_tmatrix outside the tests."""
    found = set()
    for py in sorted(PACKAGE.rglob("*.py")):
        if "tests" in py.relative_to(PACKAGE).parts:
            continue
        tree = ast.parse(py.read_text())
        funcs = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            f = node.func
            name = f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else None
            if name != "compute_cube_tmatrix":
                continue
            enclosing = sorted(
                (fn for fn in funcs if fn.lineno <= node.lineno <= (fn.end_lineno or fn.lineno)),
                key=lambda fn: fn.lineno,
            )
            found.add((str(py.relative_to(PACKAGE)), ".".join(fn.name for fn in enclosing) or "<module>"))
    return found


def test_no_lattice_solver_calls_the_lone_cube_tmatrix_directly() -> None:
    unexpected = sorted(_direct_calls() - set(ALLOWED))
    assert not unexpected, (
        f"compute_cube_tmatrix is called directly in {unexpected}.\n"
        "  Where: those functions, under cubic_scattering/\n"
        "  Valid: a cell inside a lattice takes\n"
        "         point_cell_self.lattice_cube_tmatrix(omega, a, ref, contrast),\n"
        "         which follows single_site.lattice in cubic_scattering/numerics.yml\n"
        "  Fix:   route the call through lattice_cube_tmatrix, or, for a lone cube, add it to ALLOWED in\n"
        "         cubic_scattering/tests/test_single_site_guard.py with the reason."
    )


def test_every_allowed_call_still_exists() -> None:
    """A stale entry would let a future call in the same place through unexamined."""
    assert set(ALLOWED) <= _direct_calls()


def test_the_configured_single_site_is_the_exact_closure() -> None:
    assert lattice_single_site() == "exact"


@pytest.mark.parametrize("omega", [300.0, 1500.0 + 20.0j])
def test_the_drop_in_is_the_exact_closure(omega: complex) -> None:
    a = 0.5
    got = _sub_cell_tmatrix_9x9(exact_cube_tmatrix(omega, a, REF, CONTRAST), omega, a)
    want = exact_point_tmatrix_9x9(omega, 2 * a, CONTRAST, self_block_9x9(omega, REF, 2 * a))
    assert np.max(np.abs(got - want)) <= 1e-13 * np.max(np.abs(want))
    via_config = _sub_cell_tmatrix_9x9(lattice_cube_tmatrix(omega, a, REF, CONTRAST), omega, a)
    assert np.array_equal(via_config, got)


def test_the_drop_in_exposes_only_the_effective_contrasts() -> None:
    res = exact_cube_tmatrix(300.0, 0.5, REF, CONTRAST)
    for name in (
        "Gamma0",
        "Ac",
        "Bc",
        "Cc",
        "T1c",
        "T2c",
        "T3c",
        "amp_u",
        "amp_theta",
        "amp_e_off",
        "amp_e_diag",
    ):
        assert np.isnan(getattr(res, name)), name
    for name in ("Drho_star", "Dlambda_star", "Dmu_star_off", "Dmu_star_diag"):
        assert np.isfinite(getattr(res, name)), name


@pytest.mark.parametrize(
    ("text", "problem"),
    [
        ("cell_average: {}\n", "is missing"),
        ("single_site:\n  lattice: sphere\n", "is 'sphere'; it must be one of"),
    ],
)
def test_a_bad_setting_is_refused_with_its_remedy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, text: str, problem: str
) -> None:
    cfg = tmp_path / "numerics.yml"
    cfg.write_text(text)
    monkeypatch.setattr(point_cell_self, "NUMERICS_YAML", cfg)
    lattice_single_site.cache_clear()
    try:
        with pytest.raises(ValueError) as err:
            lattice_single_site()
        msg = str(err.value)
        assert problem in msg
        assert str(cfg) in msg and "single_site.lattice" in msg  # where
        assert "lattice: exact" in msg  # what it should look like
        assert "Fix:" in msg  # how to recover
    finally:
        lattice_single_site.cache_clear()


def test_an_attenuating_medium_keeps_its_complex_speeds() -> None:
    """The self term is cached by medium; complex (attenuating) speeds must not be cast to real on the way."""
    lossy = ReferenceMedium(alpha=5000.0 * (1 - 0.01j), beta=3000.0 * (1 - 0.02j), rho=2500.0)
    omega, a = 300.0, 0.5
    got = _sub_cell_tmatrix_9x9(exact_cube_tmatrix(omega, a, lossy, CONTRAST), omega, a)
    want = exact_point_tmatrix_9x9(omega, 2 * a, CONTRAST, self_block_9x9(omega, lossy, 2 * a))
    lossless = _sub_cell_tmatrix_9x9(exact_cube_tmatrix(omega, a, REF, CONTRAST), omega, a)
    assert np.max(np.abs(got - want)) <= 1e-13 * np.max(np.abs(want))
    assert np.max(np.abs(got - lossless)) > 1e-6 * np.max(np.abs(lossless))
