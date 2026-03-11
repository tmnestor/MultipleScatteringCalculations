"""Tests for the horizontal Green's tensor and 9×9 propagator.

Validates:
  1. exact_propagator_9x9 vs resonance_tmatrix._propagator_block_9x9
  2. FFT 9×9 vs exact spatial at several (Δx, Δy) points
  3. ky residue 9×9 vs exact for Δx=0
  4. Propagator 9×9 symmetry: G symmetric, H = C^T
  5. D4h 9×9 transformations against direct evaluation
  6. Lattice matvec 9×9 vs direct summation
"""

from __future__ import annotations

import numpy as np
import pytest

from cubic_scattering import ReferenceMedium
from cubic_scattering.horizontal_greens import (
    exact_greens,
    exact_propagator_9x9,
    horizontal_greens_fft,
    horizontal_greens_fft_9x9,
    horizontal_greens_ky_residue,
    horizontal_greens_ky_residue_9x9,
)
from cubic_scattering.lattice_greens import (
    LatticeGreens,
    _apply_refl_x,
    _apply_refl_y,
    _apply_rot90,
    _matvec_direct,
)
from cubic_scattering.resonance_tmatrix import _propagator_block_9x9

# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture()
def ref():
    return ReferenceMedium(alpha=5.0, beta=3.0, rho=3.0)


@pytest.fixture()
def omega():
    return 2 * np.pi * (1 + 0.03j)


# ── Test 1: exact_propagator_9x9 vs resonance_tmatrix ─────────────


def test_exact_propagator_9x9_vs_resonance(ref, omega):
    """exact_propagator_9x9 must agree exactly with _propagator_block_9x9."""
    test_points = [
        (0.8, 0.5, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.3, 0.7, 0.2),
    ]
    for x, y, z in test_points:
        P1 = exact_propagator_9x9(x, y, z, omega, ref)
        # exact_propagator_9x9 uses seismological r_vec = [z, x, y]
        P2 = _propagator_block_9x9(np.array([z, x, y]), omega, ref)
        err = np.linalg.norm(P1 - P2) / max(np.linalg.norm(P2), 1e-30)
        assert err < 1e-14, f"Mismatch at ({x},{y},{z}): err={err:.2e}"


# ── Test 2: FFT 9×9 vs exact ─────────────────────────────────────


def test_fft_9x9_vs_exact(ref, omega):
    """FFT 9×9 at several Δy values must agree with exact propagator."""
    dx_abs = 1.0
    P_fft, y_grid = horizontal_greens_fft_9x9(
        dx_abs,
        Nky=512,
        ky_max=15,
        kz_max=15,
        Nkz=256,
        omega=omega,
        rho=ref.rho,
        alpha=ref.alpha,
        beta=ref.beta,
    )
    for dy_target in [0.0, 0.5, 1.0]:
        iy = np.argmin(np.abs(y_grid - dy_target))
        yv = y_grid[iy]
        P_ex = exact_propagator_9x9(dx_abs, yv, 0.0, omega, ref)
        P_ff = P_fft[:, :, iy]
        err = np.linalg.norm(P_ff - P_ex) / np.linalg.norm(P_ex)
        assert err < 5e-3, f"FFT 9x9 at dy={yv:.3f}: err={err:.4e}"


def test_fft_9x9_G_block_matches_3x3(ref, omega):
    """The G block (3×3) of the 9×9 FFT must match the standalone 3×3 FFT.

    The 3×3 FFT uses (x=0,y=1,z=2) ordering while the 9×9 G block uses
    seismological (z=0,x=1,y=2). Apply permutation P: G_9x9 = P @ G_3x3 @ P^T.
    """
    dx_abs = 1.0
    params = dict(
        Nky=256,
        ky_max=12,
        kz_max=12,
        Nkz=128,
        omega=omega,
        rho=ref.rho,
        alpha=ref.alpha,
        beta=ref.beta,
    )
    P_9, _ = horizontal_greens_fft_9x9(dx_abs, **params)
    G_3, _ = horizontal_greens_fft(dx_abs, **params)
    # Permute 3×3 from (x,y,z) to (z,x,y): P = [[0,0,1],[1,0,0],[0,1,0]]
    perm = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=float)
    G_3_sei = np.einsum("ia,jb,abk->ijk", perm, perm, G_3)
    err = np.linalg.norm(P_9[:3, :3, :] - G_3_sei) / np.linalg.norm(G_3_sei)
    assert err < 1e-14, f"G block mismatch: {err:.2e}"


# ── Test 3: ky residue 9×9 vs exact (Δx=0) ──────────────────────


def test_ky_residue_9x9_vs_exact(ref, omega):
    """ky residue 9×9 at Δx=0 must agree with exact propagator."""
    for dy in [0.5, 1.0]:
        P_ky = horizontal_greens_ky_residue_9x9(
            dy,
            kx_max=15,
            Nkx=256,
            kz_max=15,
            Nkz=256,
            omega=omega,
            rho=ref.rho,
            alpha=ref.alpha,
            beta=ref.beta,
        )
        P_ex = exact_propagator_9x9(0.0, dy, 0.0, omega, ref)
        err = np.linalg.norm(P_ky - P_ex) / np.linalg.norm(P_ex)
        assert err < 0.1, f"ky residue 9x9 at dy={dy}: err={err:.4e}"


def test_ky_residue_9x9_G_block_matches_3x3(ref, omega):
    """G block of ky-residue 9×9 must match standalone 3×3 ky-residue.

    Applies (x,y,z)→(z,x,y) permutation to the 3×3 result.
    """
    dy = 0.8
    params = dict(
        kx_max=15,
        Nkx=256,
        kz_max=15,
        Nkz=256,
        omega=omega,
        rho=ref.rho,
        alpha=ref.alpha,
        beta=ref.beta,
    )
    P_9 = horizontal_greens_ky_residue_9x9(dy, **params)
    G_3 = horizontal_greens_ky_residue(dy, **params)
    perm = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=float)
    G_3_sei = perm @ G_3 @ perm.T
    err = np.linalg.norm(P_9[:3, :3] - G_3_sei) / np.linalg.norm(G_3_sei)
    assert err < 1e-14, f"G block mismatch: {err:.2e}"


# ── Test 4: Propagator 9×9 symmetry ──────────────────────────────


def test_propagator_9x9_symmetry(ref, omega):
    """G block is symmetric, H = C^T (reciprocity)."""
    test_points = [
        (0.8, 0.5, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.3, 0.7, 0.2),
    ]
    for x, y, z in test_points:
        P = exact_propagator_9x9(x, y, z, omega, ref)
        G_block = P[:3, :3]
        C_block = P[:3, 3:]
        H_block = P[3:, :3]

        err_sym = np.linalg.norm(G_block - G_block.T) / np.linalg.norm(G_block)
        assert err_sym < 1e-14, f"G not symmetric at ({x},{y},{z}): {err_sym:.2e}"

        # Engineering convention: H_αj = W_αα C_jα  ⟹  H = W @ C^T
        W = np.diag([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
        err_recip = np.linalg.norm(H_block - W @ C_block.T) / np.linalg.norm(H_block)
        assert err_recip < 1e-14, f"H != W@C^T at ({x},{y},{z}): {err_recip:.2e}"


# ── Test 5: D4h 9×9 transformations ──────────────────────────────


def test_d4h_9x9_transformations(ref, omega):
    """D4h orbit produces correct 9×9 tensors."""
    x, y = 0.3, 0.2
    P0 = exact_propagator_9x9(x, y, 0.0, omega, ref)

    # x-reflection: (-x, y, 0)
    P_rx = exact_propagator_9x9(-x, y, 0.0, omega, ref)
    err = np.linalg.norm(P_rx - _apply_refl_x(P0)) / np.linalg.norm(P_rx)
    assert err < 1e-14, f"x-reflection error: {err:.2e}"

    # y-reflection: (x, -y, 0)
    P_ry = exact_propagator_9x9(x, -y, 0.0, omega, ref)
    err = np.linalg.norm(P_ry - _apply_refl_y(P0)) / np.linalg.norm(P_ry)
    assert err < 1e-14, f"y-reflection error: {err:.2e}"

    # x↔y swap: (y, x, 0)
    P_rot = exact_propagator_9x9(y, x, 0.0, omega, ref)
    err = np.linalg.norm(P_rot - _apply_rot90(P0)) / np.linalg.norm(P_rot)
    assert err < 1e-14, f"rotation error: {err:.2e}"

    # Combined: (-x, -y, 0) via both reflections
    P_rxy = exact_propagator_9x9(-x, -y, 0.0, omega, ref)
    P_pred = _apply_refl_x(_apply_refl_y(P0))
    err = np.linalg.norm(P_rxy - P_pred) / np.linalg.norm(P_rxy)
    assert err < 1e-14, f"xy-reflection error: {err:.2e}"


# ── Test 6: Lattice matvec 9×9 ───────────────────────────────────


def test_lattice_matvec_9x9():
    """FFT-accelerated matvec matches direct matvec for small M."""
    M = 4
    lg = LatticeGreens(0.1, M, 2 * np.pi, 3.0, 5.0, 3.0, 0.03)
    G_arr = lg.compute_spatial(block_size=9)

    rng = np.random.default_rng(42)
    u = rng.standard_normal((M, M, 9)) + 1j * rng.standard_normal((M, M, 9))

    v_fft = lg.matvec(u)
    v_dir = _matvec_direct(G_arr, u, M)

    err = np.linalg.norm(v_fft - v_dir) / np.linalg.norm(v_dir)
    assert err < 1e-12, f"Matvec 9x9 error: {err:.2e}"


def test_lattice_matvec_3x3_backward_compat():
    """3×3 matvec still works correctly."""
    M = 4
    lg = LatticeGreens(0.1, M, 2 * np.pi, 3.0, 5.0, 3.0, 0.03)
    G_arr = lg.compute_spatial(block_size=3)

    rng = np.random.default_rng(42)
    u = rng.standard_normal((M, M, 3)) + 1j * rng.standard_normal((M, M, 3))

    v_fft = lg.matvec(u)
    v_dir = _matvec_direct(G_arr, u, M)

    err = np.linalg.norm(v_fft - v_dir) / np.linalg.norm(v_dir)
    assert err < 1e-12, f"Matvec 3x3 error: {err:.2e}"


# ── Test: exact_greens with ReferenceMedium kwarg ─────────────────


def test_exact_greens_ref_kwarg(ref, omega):
    """exact_greens with ref= must match raw-parameter version."""
    G1 = exact_greens(0.8, 0.5, 0.0, omega, ref=ref)
    G2 = exact_greens(0.8, 0.5, 0.0, omega, rho=ref.rho, alpha=ref.alpha, beta=ref.beta)
    err = np.linalg.norm(G1 - G2) / np.linalg.norm(G2)
    assert err < 1e-14


# ── Test: Spectral 9×9 via horizontal residues ──────────────────


def test_spectral_9x9_vs_spatial():
    """Horizontal-residue spectral 9×9 matches spatial 9×9."""
    M = 4
    lg = LatticeGreens(0.1, M, 2 * np.pi, 3.0, 5.0, 3.0, 0.03)
    G_spatial = lg.compute_spatial(block_size=9)
    G_spectral = lg.compute_spectral(block_size=9, Nky=512, Nkz=512)
    S = 2 * M - 1
    for n1 in range(S):
        for n2 in range(S):
            if n1 == M - 1 and n2 == M - 1:
                continue
            norm = np.linalg.norm(G_spatial[n1, n2])
            if norm < 1e-30:
                continue
            err = np.linalg.norm(G_spectral[n1, n2] - G_spatial[n1, n2]) / norm
            assert err < 2e-3, f"({n1},{n2}): err={err:.2e}"


def test_hybrid_9x9_vs_spatial():
    """Hybrid 9×9 near-field should be exact where spatial is used."""
    M = 4
    d = 0.1
    lg = LatticeGreens(d, M, 2 * np.pi, 3.0, 5.0, 3.0, 0.03)
    G_spatial = lg.compute_spatial(block_size=9)
    lg._G_spectral = None
    # r_cut=1.5d covers only the nearest neighbours (r=d)
    G_hybrid = lg.compute_hybrid(block_size=9, r_cut=1.5 * d, Nky=512, Nkz=512)
    S = 2 * M - 1
    for n1 in range(S):
        for n2 in range(S):
            if n1 == M - 1 and n2 == M - 1:
                continue
            norm = np.linalg.norm(G_spatial[n1, n2])
            if norm < 1e-30:
                continue
            dn1 = n1 - (M - 1)
            dn2 = n2 - (M - 1)
            r = np.sqrt((dn1 * d) ** 2 + (dn2 * d) ** 2)
            if r <= 1.5 * d:
                # Near-field: exact spatial → machine precision
                err = np.linalg.norm(G_hybrid[n1, n2] - G_spatial[n1, n2]) / norm
                assert err < 1e-12, f"near ({n1},{n2}): err={err:.2e}"
            else:
                # Far-field: spectral → same tolerance as pure spectral
                err = np.linalg.norm(G_hybrid[n1, n2] - G_spatial[n1, n2]) / norm
                assert err < 2e-3, f"far ({n1},{n2}): err={err:.2e}"


def test_fcc_9x9_routes_to_spectral():
    """compute_fcc(block_size=9) should produce same result as compute_spectral(block_size=9)."""
    M = 3
    lg = LatticeGreens(0.1, M, 2 * np.pi, 3.0, 5.0, 3.0, 0.03)
    G_fcc = lg.compute_fcc(block_size=9)
    lg._G_spectral = None
    G_spec = lg.compute_spectral(block_size=9)
    err = np.linalg.norm(G_fcc - G_spec) / max(np.linalg.norm(G_spec), 1e-30)
    assert err < 1e-14, f"fcc vs spectral routing: err={err:.2e}"
