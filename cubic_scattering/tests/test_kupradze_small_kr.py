"""The derivative-ladder propagators at small k r: no cancellation between the P and S scalars.

``kupradze_derivatives`` builds G from the scalars g_P and g_S, and every derivative order of
d_i d_j (g_S - g_P) cancels its leading terms between the two modes. Formed as a difference of two
separately evaluated tensors it loses digits as eps / (k r)^2 (8.7e-7 relative at k_S r = 1e-4). The
difference must be evaluated as one function, by its own series near the origin.

References:
- the point block: the 50-digit evaluation of ``test_greens_small_kr``;
- the cell-averaged pair block: the same Gauss rule applied to ``kernel_9x9``, which shares no code with
  the derivative ladder and holds round-off at every k r.
"""

import numpy as np
import pytest

from cubic_scattering.cell_averaged_lattice import _cell_nodes
from cubic_scattering.cell_averaged_pair import auto_n_gauss, averaged_pair_block_9x9, clear_block_cache
from cubic_scattering.effective_contrasts import ReferenceMedium
from cubic_scattering.graded_voxel.kernel import kernel_9x9
from cubic_scattering.kupradze_derivatives import propagator_block_9x9_kupradze
from cubic_scattering.resonance_tmatrix import _voigt_contract
from cubic_scattering.tests.test_greens_small_kr import (
    DIRECTION,
    ETAS,
    KR_VALUES,
    OMEGA,
    REF,
    RTOL,
    _reference_tensors,
)


def _rel(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


@pytest.mark.parametrize("eta", ETAS)
@pytest.mark.parametrize("kr", KR_VALUES)
def test_kupradze_point_block_holds_round_off(kr: float, eta: float) -> None:
    omega = OMEGA * (1.0 + 1j * eta) if eta else OMEGA
    x = DIRECTION * (kr / abs(omega / REF.beta))
    G, Gd, Gdd = _reference_tensors(x, omega)
    C, H, S = _voigt_contract(Gd, Gdd)
    ref_block = np.block([[G, C], [H, S]])
    got = propagator_block_9x9_kupradze(x, omega, REF)
    assert _rel(got, ref_block) < RTOL, f"|k_S| r = {kr}, eta = {eta}: {_rel(got, ref_block):.2e}"


@pytest.mark.parametrize("kd", [1e-3, 1e-2, 0.3])
@pytest.mark.parametrize("offset", [(1, 0, 0), (1, 1, 0), (2, 1, 1)])
def test_averaged_pair_block_holds_round_off(kd: float, offset: tuple[int, int, int]) -> None:
    ref = ReferenceMedium(alpha=5000.0, beta=3000.0, rho=2500.0)
    omega = 2.0 * np.pi * 10.0
    d = kd / (omega / ref.beta)
    s = np.array(offset, dtype=float) * d
    clear_block_cache()
    got = averaged_pair_block_9x9(s, omega, ref, d)
    nodes, wts = _cell_nodes(0.5 * d, auto_n_gauss(s, d))
    u = np.stack(np.meshgrid(nodes, nodes, nodes, indexing="ij"), -1).reshape(-1, 3)
    w = np.einsum("i,j,k->ijk", wts, wts, wts).ravel()
    expected = np.einsum("n,nab->ab", w, kernel_9x9(s - u, omega, ref))
    assert _rel(got, expected) < 1e-12, f"k d = {kd}, offset {offset}: {_rel(got, expected):.2e}"
