"""The closed-form touching blocks against the 40-digit corner and edge reference.

Mathematica/GradedVoxel_CornerReference.wl integrates the static Galerkin double integral of cells touching
at a corner or an edge directly at 40 digits (absolutely convergent there: no distribution, no moved
derivative), sharing no integration with the package. The previous closed forms (origin-anchored master
integrals assembled in double precision) were wrong by 2.3e-13 (linear field, corner) and 1.7e-11
(quadratic field, corner) against it; the stable evaluation (Legendre moments away from the singular point,
Duffy pyramids at it) is at round-off, as quadrature is.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from cubic_scattering.effective_contrasts import ReferenceMedium
from cubic_scattering.graded_voxel import blocks

REF = ReferenceMedium(alpha=5000.0, beta=3000.0, rho=2500.0)
DATA = Path(__file__).resolve().parents[2] / "Mathematica"
OFFSETS = {"corner": (1, 1, 1), "edge": (1, 1, 0)}


def _reference(cell: str, name: str, n_test: int, n_source: int) -> np.ndarray:
    data = json.loads((DATA / f"GradedVoxel_corner_reference_{cell}.json").read_text())
    return np.array(data[name]["block"], dtype=float).reshape(n_test, n_source, 9, 9)


def _static_series_block(offset, n_source: int, n_test: int) -> np.ndarray:
    # at k_S h = 1e-12 the dynamic part of the real block is below 1e-23
    return blocks.near_block_series(offset, 1.0, 1e-12 * REF.beta, REF, n_source, n_test).real


@pytest.mark.slow  # 3.3 min and 31 s (measured 2026-10-05)
@pytest.mark.parametrize("name", ["corner", "edge"])
def test_linear_field_blocks_match_the_reference_to_round_off(name):
    want = _reference("linear", name, 4, 10)
    got = _static_series_block(OFFSETS[name], 10, 4)
    assert np.linalg.norm(got - want) / np.linalg.norm(want) < 5e-15


@pytest.mark.slow  # about 10 min: the one-off sparse solves of the quadratic field
def test_quadratic_field_corner_block_matches_the_reference_to_round_off():
    want = _reference("quadratic", "corner", 10, 35)
    got = _static_series_block(OFFSETS["corner"], 35, 10)
    assert np.linalg.norm(got - want) / np.linalg.norm(want) < 1e-14
