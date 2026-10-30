"""The point, Galerkin and gradient-hierarchy arms of the comparison solve one graded sphere, measured one way.

The hierarchy builds its body in ``measure_graded_sphere_gradient_hierarchy`` and takes its medium from
``measure_ball_gradient_hierarchy``; the comparison takes its profile from ``pilot_graded_sphere_vs_exact`` and
its medium from ``gate_sphere_cell_average_vs_mie``. The constants are written separately in those modules, so
these tests are what keep them one body: a different core, profile, medium, contrast or observation set would
make every equal-accuracy comparison between the arms void.
"""

import dataclasses
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

import compare_point_galerkin as cmp  # noqa: E402
import gate_sphere_cell_average_vs_mie as gate  # noqa: E402
import measure_ball_gradient_hierarchy as hier  # noqa: E402
import measure_graded_sphere_gradient_hierarchy as gs  # noqa: E402


def test_graded_bodies_identical() -> None:
    assert gs.PROFILE == "smoothstep"
    assert (gs.RADIUS, gs.CORE) == (cmp.RADIUS, cmp.CORE)
    rng = np.random.default_rng(20261030)
    # points inside, across the shell and outside
    pts = rng.uniform(-1.05, 1.05, size=(10_000, 3)) * cmp.RADIUS
    hierarchy = gs.smoothstep(np.linalg.norm(pts, axis=1))
    # graded_profile is a BatchProfile: calling it takes one point, .batch takes (M, 3)
    comparison = np.asarray(cmp.graded_profile.batch(pts), dtype=float)
    # the same polynomial, written 10x^3 - 15x^4 + 6x^5 in one module and x^3 (10 - 15x + 6x^2) in the other,
    # so the two differ by rounding (2.7e-15 measured); another core, profile or shell would differ at O(1)
    assert np.max(np.abs(hierarchy - comparison)) < 1e-14


def test_medium_and_contrast_identical() -> None:
    assert dataclasses.asdict(hier.REF) == dataclasses.asdict(gate.REF)
    assert dataclasses.asdict(hier.CONTRAST) == dataclasses.asdict(gate.CONTRAST)


def test_observation_set_identical() -> None:
    assert gs.R_FAR == cmp.R_MULT * cmp.RADIUS
    assert np.array_equal(gs.THETA, cmp.THETA)
    assert np.array_equal(gs.K_HAT, cmp.K_HAT) and np.array_equal(gs.POL, cmp.POL)


# --- the hierarchy arm ---------------------------------------------------------------------------------

ROW_KEYS = {
    "body", "ka_s", "arm", "n", "tol", "cells", "unknowns", "error", "setup_s", "gmres_s", "products",
    "readout_s", "wall_s", "peak_GB", "threads", "near_store",
}  # fmt: skip


def test_hierarchy_arm_row() -> None:
    """The arm's row has every key, 60 unknowns a voxel at q = 3, and the standalone script's error."""
    import measure_graded_sphere_gradient_hierarchy_fft as hfft
    from crosscheck_graded_sphere import graded_mie_result

    from cubic_scattering.sphere_scattering import mie_scattered_displacement

    row = cmp.one("graded", 0.5, "hierarchy", 4, 1e-10)
    assert set(row) == ROW_KEYS
    assert row["arm"] == "hierarchy" and row["unknowns"] == 60 * row["cells"]

    omega = 0.5 * hier.REF.beta / gs.RADIUS
    side, centres, coefs, asm, sol, _ = hfft.solve_fft(4, 3, omega, hier.CONTRAST, tol=1e-10)
    obs = gate.obs_points(gs.R_FAR, gs.THETA)
    n_max = max(8, int(np.ceil(0.5 + 4 * 0.5 ** (1 / 3) + 6)))
    exact = mie_scattered_displacement(
        graded_mie_result(omega, gs.RADIUS, gs.CORE, hier.REF, hier.CONTRAST, n_max),
        obs,
    )
    got = gs.far_field(side, centres, coefs, asm, sol, omega, hier.CONTRAST, obs)
    standalone = float(np.max(np.abs(got - exact)) / np.max(np.abs(exact)))
    assert abs(row["error"] - standalone) <= 1e-12 * standalone


def test_hierarchy_arm_refuses_sharp() -> None:
    """The hierarchy projects a smooth contrast; on the sharp sphere it is not defined, so it refuses."""
    import pytest

    with pytest.raises(ValueError, match="graded"):
        cmp.one("sharp", 0.5, "hierarchy", 4, 1e-10)
