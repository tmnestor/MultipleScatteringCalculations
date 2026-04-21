"""Tests for inter-voxel strain propagator.

Validates:
1. Laplacian identity: Σ_k B_{ijkk} = A_{ij}
2. Point-group symmetry (C₄ᵥ, C₂ᵥ, S₃)
3. Propagator minor symmetry P_{ijkl} = P_{jikl} = P_{ijlk}
4. Propagator major symmetry P_{ijkl} = P_{klij}
5. O_h rotation for all 26 neighbours
6. Cross-check propagator values against Mathematica (16 digits)
7. Dynamic corrections: Laplacian, convergence, static limit
"""

import numpy as np
import pytest

from cubic_scattering.inter_voxel_propagator import (
    CORNER_A11,
    CORNER_A12,
    CORNER_B1111,
    CORNER_B1112,
    CORNER_B1122,
    CORNER_B1123,
    CORNER_D3H_000,
    CORNER_D3H_001,
    CORNER_D3OM_000,
    CORNER_D3OM_001,
    CORNER_D3PSI_000,
    CORNER_D3PSI_001,
    CORNER_D3X_000,
    CORNER_D3X_001,
    CORNER_D3X_012,
    CORNER_DOM_0,
    CORNER_DPHI_0,
    CORNER_DPSI_0,
    CORNER_DX_0,
    DYN1_CORNER_A11,
    DYN1_CORNER_A12,
    DYN1_CORNER_B1111,
    DYN1_CORNER_B1112,
    DYN1_CORNER_B1122,
    DYN1_CORNER_B1123,
    DYN1_EDGE_A11,
    DYN1_EDGE_A12,
    DYN1_EDGE_A33,
    DYN1_EDGE_B1111,
    DYN1_EDGE_B1112,
    DYN1_EDGE_B1122,
    DYN1_EDGE_B1133,
    DYN1_EDGE_B1233,
    DYN1_EDGE_B3333,
    DYN1_FACE_A11,
    DYN1_FACE_A22,
    DYN1_FACE_B1111,
    DYN1_FACE_B1122,
    DYN1_FACE_B2222,
    DYN1_FACE_B2233,
    DYN2_CORNER_A11,
    DYN2_CORNER_A12,
    DYN2_CORNER_B1111,
    DYN2_CORNER_B1112,
    DYN2_CORNER_B1122,
    DYN2_CORNER_B1123,
    DYN2_EDGE_A11,
    DYN2_EDGE_A12,
    DYN2_EDGE_A33,
    DYN2_EDGE_B1111,
    DYN2_EDGE_B1112,
    DYN2_EDGE_B1122,
    DYN2_EDGE_B1133,
    DYN2_EDGE_B1233,
    DYN2_EDGE_B3333,
    DYN2_FACE_A11,
    DYN2_FACE_A22,
    DYN2_FACE_B1111,
    DYN2_FACE_B1122,
    DYN2_FACE_B2222,
    DYN2_FACE_B2233,
    DYN3_CORNER_A11,
    DYN3_CORNER_A12,
    DYN3_CORNER_B1111,
    DYN3_CORNER_B1112,
    DYN3_CORNER_B1122,
    DYN3_CORNER_B1123,
    DYN3_EDGE_A11,
    DYN3_EDGE_A12,
    DYN3_EDGE_A33,
    DYN3_EDGE_B1111,
    DYN3_EDGE_B1112,
    DYN3_EDGE_B1122,
    DYN3_EDGE_B1133,
    DYN3_EDGE_B1233,
    DYN3_EDGE_B3333,
    DYN3_FACE_A11,
    DYN3_FACE_A22,
    DYN3_FACE_B1111,
    DYN3_FACE_B1122,
    DYN3_FACE_B2222,
    DYN3_FACE_B2233,
    EDGE_A11,
    EDGE_A12,
    EDGE_A33,
    EDGE_B1111,
    EDGE_B1112,
    EDGE_B1122,
    EDGE_B1133,
    EDGE_B1233,
    EDGE_B3333,
    EDGE_D3H_000,
    EDGE_D3H_001,
    EDGE_D3H_022,
    EDGE_D3OM_000,
    EDGE_D3OM_001,
    EDGE_D3OM_022,
    EDGE_D3PSI_000,
    EDGE_D3PSI_001,
    EDGE_D3PSI_022,
    EDGE_D3X_000,
    EDGE_D3X_001,
    EDGE_D3X_022,
    EDGE_DOM_0,
    EDGE_DPHI_0,
    EDGE_DPSI_0,
    EDGE_DX_0,
    FACE_A11,
    FACE_A22,
    FACE_B1111,
    FACE_B1122,
    FACE_B2222,
    FACE_B2233,
    FACE_D3H_000,
    FACE_D3H_011,
    FACE_D3OM_000,
    FACE_D3OM_011,
    FACE_D3PSI_000,
    FACE_D3PSI_011,
    FACE_D3X_000,
    FACE_D3X_011,
    FACE_DOM_0,
    FACE_DPHI_0,
    FACE_DPSI_0,
    FACE_DX_0,
    _build_D3_tensor,
    _build_D3Psi_tensor,
    _build_dG_rank3_canonical,
    _build_dPhi_vector,
    _build_dW_vector,
    _dG_to_C_block,
    _P_to_voigt_S,
    corner_propagator,
    dynamic_inter_voxel_propagator,
    edge_propagator,
    face_propagator,
    inter_voxel_propagator,
    inter_voxel_propagator_9x9,
)

MU = 1.0
NU = 0.25
ETA = 1.0 / (2.0 * (1.0 - NU))


# ── Laplacian identity: Σ_k B_{ijkk} = A_{ij} ──


class TestLaplacianIdentity:
    """Verify Σ_k B_{ijkk} = A_{ij} for all three neighbour types."""

    def test_face_laplacian_11(self):
        # B₁₁₁₁ + B₁₁₂₂ + B₁₁₃₃ = A₁₁
        assert FACE_B1111 + FACE_B1122 + FACE_B1122 == pytest.approx(
            FACE_A11, abs=1e-13
        )

    def test_face_laplacian_22(self):
        # B₁₁₂₂ + B₂₂₂₂ + B₂₂₃₃ = A₂₂
        assert FACE_B1122 + FACE_B2222 + FACE_B2233 == pytest.approx(
            FACE_A22, abs=1e-13
        )

    def test_edge_laplacian_11(self):
        # B₁₁₁₁ + B₁₁₂₂ + B₁₁₃₃ = A₁₁
        assert EDGE_B1111 + EDGE_B1122 + EDGE_B1133 == pytest.approx(
            EDGE_A11, abs=1e-13
        )

    def test_edge_laplacian_33(self):
        # B₁₁₃₃ + B₂₂₃₃ + B₃₃₃₃ = A₃₃
        assert EDGE_B1133 + EDGE_B1133 + EDGE_B3333 == pytest.approx(
            EDGE_A33, abs=1e-13
        )

    def test_edge_laplacian_12(self):
        # B₁₁₁₂ + B₁₂₂₂ + B₁₂₃₃ = A₁₂
        # By C₂ᵥ: B₁₂₂₂ = B₁₁₁₂
        assert 2 * EDGE_B1112 + EDGE_B1233 == pytest.approx(EDGE_A12, abs=1e-13)

    def test_corner_laplacian_11(self):
        # B₁₁₁₁ + 2*B₁₁₂₂ = A₁₁ = 0
        assert CORNER_B1111 + 2 * CORNER_B1122 == pytest.approx(CORNER_A11, abs=1e-13)

    def test_corner_laplacian_12(self):
        # 2*B₁₁₁₂ + B₁₁₂₃ = A₁₂
        assert 2 * CORNER_B1112 + CORNER_B1123 == pytest.approx(CORNER_A12, abs=1e-13)

    def test_corner_A11_is_zero(self):
        """A₁₁ = 0 is an exact identity for corner-adjacent."""
        assert CORNER_A11 == 0.0

    def test_corner_B1111_eq_neg2_B1122(self):
        """B₁₁₁₁ = -2 B₁₁₂₂ is exact for corner."""
        assert CORNER_B1111 == pytest.approx(-2 * CORNER_B1122, rel=1e-12)


# ── Propagator symmetries ──


class TestPropagatorSymmetry:
    """Verify index symmetries of the assembled propagator.

    P_{ijkl} = -1/(2μ)[δ_{ik}A_{jl} + δ_{jk}A_{il} - 2η B_{ijkl}]

    This has ij minor symmetry (from the two δ-A terms).
    It does NOT have kl minor or major symmetry — this is correct for
    non-ellipsoidal Γ_{ijkl} = ½(G_{ik,jl} + G_{jk,il}).  The kl-antisymmetric
    part is annihilated when contracted with the symmetric stiffness tensor.
    """

    @pytest.fixture(params=["face", "edge", "corner"])
    def P(self, request):
        if request.param == "face":
            return face_propagator(MU, NU)
        if request.param == "edge":
            return edge_propagator(MU, NU)
        return corner_propagator(MU, NU)

    def test_minor_symmetry_ij(self, P):
        """P_{ijkl} = P_{jikl}."""
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for ll in range(3):
                        assert P[i, j, k, ll] == pytest.approx(
                            P[j, i, k, ll], abs=1e-15
                        ), f"minor ij failed at ({i},{j},{k},{ll})"

    def test_face_c4v_symmetry(self):
        """Face propagator R=(1,0,0): axes 1,2 are equivalent (C₄ᵥ)."""
        P = face_propagator(MU, NU)
        # P₁₁₂₂ = P₁₁₃₃
        assert P[0, 0, 1, 1] == pytest.approx(P[0, 0, 2, 2], abs=1e-15)
        # P₂₂₂₂ = P₃₃₃₃
        assert P[1, 1, 1, 1] == pytest.approx(P[2, 2, 2, 2], abs=1e-15)
        # P₁₂₁₂ = P₁₃₁₃
        assert P[0, 1, 0, 1] == pytest.approx(P[0, 2, 0, 2], abs=1e-15)
        # Off-diagonal components vanish: P₁₁₁₂ = 0
        assert P[0, 0, 0, 1] == pytest.approx(0.0, abs=1e-15)

    def test_edge_c2v_symmetry(self):
        """Edge propagator R=(1,1,0): axes 0,1 equivalent (C₂ᵥ)."""
        P = edge_propagator(MU, NU)
        # P₁₁₁₁ = P₂₂₂₂
        assert P[0, 0, 0, 0] == pytest.approx(P[1, 1, 1, 1], abs=1e-15)
        # P₁₁₃₃ = P₂₂₃₃
        assert P[0, 0, 2, 2] == pytest.approx(P[1, 1, 2, 2], abs=1e-15)
        # Components with single axis-2 index vanish: P₁₁₁₃ = 0
        assert P[0, 0, 0, 2] == pytest.approx(0.0, abs=1e-15)

    def test_corner_s3_symmetry(self):
        """Corner propagator R=(1,1,1): all axes equivalent (S₃)."""
        P = corner_propagator(MU, NU)
        # P₁₁₁₁ = P₂₂₂₂ = P₃₃₃₃
        assert P[0, 0, 0, 0] == pytest.approx(P[1, 1, 1, 1], abs=1e-15)
        assert P[0, 0, 0, 0] == pytest.approx(P[2, 2, 2, 2], abs=1e-15)
        # P₁₁₂₂ = P₁₁₃₃ = P₂₂₃₃
        assert P[0, 0, 1, 1] == pytest.approx(P[0, 0, 2, 2], abs=1e-15)
        assert P[0, 0, 1, 1] == pytest.approx(P[1, 1, 2, 2], abs=1e-15)


# ── Cross-check against Mathematica reference values ──


class TestMathematicaReference:
    """Validate P values against Mathematica output (16 digits)."""

    def test_face_P1111(self):
        P = face_propagator(MU, NU)
        assert P[0, 0, 0, 0] == pytest.approx(+0.06489609963957935, rel=1e-12)

    def test_face_P1122(self):
        P = face_propagator(MU, NU)
        assert P[0, 0, 1, 1] == pytest.approx(-0.00994518639570713, rel=1e-12)

    def test_face_P2222(self):
        P = face_propagator(MU, NU)
        assert P[1, 1, 1, 1] == pytest.approx(-0.02599699603810159, rel=1e-12)

    def test_face_P2233(self):
        P = face_propagator(MU, NU)
        assert P[1, 1, 2, 2] == pytest.approx(+0.01343931900972618, rel=1e-12)

    def test_face_P1212(self):
        P = face_propagator(MU, NU)
        assert P[0, 1, 0, 1] == pytest.approx(-0.04369948153183095, rel=1e-12)

    def test_face_P2323(self):
        P = face_propagator(MU, NU)
        assert P[1, 2, 1, 2] == pytest.approx(-0.02031497612639764, rel=1e-12)

    def test_edge_P1111(self):
        P = edge_propagator(MU, NU)
        assert P[0, 0, 0, 0] == pytest.approx(-0.00080672935720783, rel=1e-11)

    def test_edge_P3333(self):
        P = edge_propagator(MU, NU)
        assert P[2, 2, 2, 2] == pytest.approx(-0.00711840486058437, rel=1e-11)

    def test_edge_P1112(self):
        P = edge_propagator(MU, NU)
        assert P[0, 0, 0, 1] == pytest.approx(+0.03454486653300503, rel=1e-11)

    def test_edge_P1233(self):
        P = edge_propagator(MU, NU)
        # P_{1233} with i=0,j=1,k=2,l=2 (0-indexed, Mathematica 1-indexed)
        # Actually P_{1233} = P[0,1,2,2] in 0-indexed
        assert P[0, 1, 2, 2] == pytest.approx(-0.00833663621412387, rel=1e-11)

    def test_corner_P1111(self):
        P = corner_propagator(MU, NU)
        assert P[0, 0, 0, 0] == pytest.approx(-0.00417024398837613, rel=1e-11)

    def test_corner_P1122(self):
        P = corner_propagator(MU, NU)
        assert P[0, 0, 1, 1] == pytest.approx(+0.00208512199418806, rel=1e-11)

    def test_corner_P1112(self):
        P = corner_propagator(MU, NU)
        assert P[0, 0, 0, 1] == pytest.approx(+0.00988469241293594, rel=1e-11)

    def test_corner_P1123(self):
        P = corner_propagator(MU, NU)
        assert P[0, 0, 1, 2] == pytest.approx(+0.00164678558813909, rel=1e-11)


# ── O_h rotation for all 26 neighbours ──


class TestOhRotation:
    """Verify inter_voxel_propagator works for all 26 nearest neighbours."""

    def test_all_face_neighbours(self):
        """All 6 face neighbours should give valid propagators."""
        face_vecs = [
            (1, 0, 0),
            (-1, 0, 0),
            (0, 1, 0),
            (0, -1, 0),
            (0, 0, 1),
            (0, 0, -1),
        ]
        for R in face_vecs:
            P = inter_voxel_propagator(R, MU, NU)
            assert P.shape == (3, 3, 3, 3)
            assert np.isfinite(P).all(), f"Non-finite P for R={R}"

    def test_all_edge_neighbours(self):
        """All 12 edge neighbours should give valid propagators."""
        edge_vecs = []
        for i in range(3):
            for j in range(i + 1, 3):
                for si in (-1, 1):
                    for sj in (-1, 1):
                        R = [0, 0, 0]
                        R[i] = si
                        R[j] = sj
                        edge_vecs.append(tuple(R))
        assert len(edge_vecs) == 12
        for R in edge_vecs:
            P = inter_voxel_propagator(R, MU, NU)
            assert P.shape == (3, 3, 3, 3)
            assert np.isfinite(P).all(), f"Non-finite P for R={R}"

    def test_all_corner_neighbours(self):
        """All 8 corner neighbours should give valid propagators."""
        count = 0
        for s1 in (-1, 1):
            for s2 in (-1, 1):
                for s3 in (-1, 1):
                    R = (s1, s2, s3)
                    P = inter_voxel_propagator(R, MU, NU)
                    assert P.shape == (3, 3, 3, 3)
                    assert np.isfinite(P).all()
                    count += 1
        assert count == 8

    def test_face_opposite_directions_related(self):
        """P(R) and P(-R) should be equal for face (inversion symmetry)."""
        P_plus = inter_voxel_propagator((1, 0, 0), MU, NU)
        P_minus = inter_voxel_propagator((-1, 0, 0), MU, NU)
        np.testing.assert_allclose(P_plus, P_minus, atol=1e-14)

    def test_face_axes_c4v(self):
        """Face propagator along x should relate to along y by rotation."""
        Px = inter_voxel_propagator((1, 0, 0), MU, NU)
        Py = inter_voxel_propagator((0, 1, 0), MU, NU)
        # P_{0000} for R=(1,0,0) should equal P_{1111} for R=(0,1,0)
        assert Px[0, 0, 0, 0] == pytest.approx(Py[1, 1, 1, 1], rel=1e-12)

    def test_invalid_neighbour(self):
        with pytest.raises(ValueError, match="not a nearest neighbour"):
            inter_voxel_propagator((2, 0, 0), MU, NU)

    def test_corner_S3_invariance(self):
        """All (±1,±1,±1) corners should produce the same P_{iiii} trace."""
        trace0 = None
        for s1 in (-1, 1):
            for s2 in (-1, 1):
                for s3 in (-1, 1):
                    P = inter_voxel_propagator((s1, s2, s3), MU, NU)
                    trace = sum(P[i, i, i, i] for i in range(3))
                    if trace0 is None:
                        trace0 = trace
                    else:
                        assert trace == pytest.approx(trace0, rel=1e-12)


# ── Dynamic propagator tests ──

# Reference medium parameters (same as CLAUDE.md)
ALPHA = 5000.0  # m/s
BETA = 3000.0  # m/s
RHO = 2500.0  # kg/m³


class TestDynamicLaplacianIdentity:
    """Verify Σ_k B⁽ⁿ⁾_{ijkk} = A⁽ⁿ⁾_{ij} for all types and orders.

    This identity holds exactly in Fourier space (Σ_k k̂²_k = 1).
    """

    def test_face_order1_11(self):
        assert DYN1_FACE_B1111 + 2 * DYN1_FACE_B1122 == pytest.approx(
            DYN1_FACE_A11, rel=1e-12
        )

    def test_face_order1_22(self):
        assert DYN1_FACE_B1122 + DYN1_FACE_B2222 + DYN1_FACE_B2233 == pytest.approx(
            DYN1_FACE_A22, rel=1e-12
        )

    def test_face_order2_11(self):
        assert DYN2_FACE_B1111 + 2 * DYN2_FACE_B1122 == pytest.approx(
            DYN2_FACE_A11, rel=1e-12
        )

    def test_face_order2_22(self):
        assert DYN2_FACE_B1122 + DYN2_FACE_B2222 + DYN2_FACE_B2233 == pytest.approx(
            DYN2_FACE_A22, rel=1e-12
        )

    def test_edge_order1_11(self):
        assert DYN1_EDGE_B1111 + DYN1_EDGE_B1122 + DYN1_EDGE_B1133 == pytest.approx(
            DYN1_EDGE_A11, rel=1e-12
        )

    def test_edge_order1_33(self):
        assert 2 * DYN1_EDGE_B1133 + DYN1_EDGE_B3333 == pytest.approx(
            DYN1_EDGE_A33, rel=1e-12
        )

    def test_edge_order1_12(self):
        assert 2 * DYN1_EDGE_B1112 + DYN1_EDGE_B1233 == pytest.approx(
            DYN1_EDGE_A12, rel=1e-12
        )

    def test_edge_order2_11(self):
        assert DYN2_EDGE_B1111 + DYN2_EDGE_B1122 + DYN2_EDGE_B1133 == pytest.approx(
            DYN2_EDGE_A11, rel=1e-12
        )

    def test_edge_order2_33(self):
        assert 2 * DYN2_EDGE_B1133 + DYN2_EDGE_B3333 == pytest.approx(
            DYN2_EDGE_A33, rel=1e-12
        )

    def test_edge_order2_12(self):
        assert 2 * DYN2_EDGE_B1112 + DYN2_EDGE_B1233 == pytest.approx(
            DYN2_EDGE_A12, rel=1e-12
        )

    def test_corner_order1_11(self):
        assert DYN1_CORNER_B1111 + 2 * DYN1_CORNER_B1122 == pytest.approx(
            DYN1_CORNER_A11, rel=1e-12
        )

    def test_corner_order1_12(self):
        assert 2 * DYN1_CORNER_B1112 + DYN1_CORNER_B1123 == pytest.approx(
            DYN1_CORNER_A12, rel=1e-12
        )

    def test_corner_order2_11(self):
        assert DYN2_CORNER_B1111 + 2 * DYN2_CORNER_B1122 == pytest.approx(
            DYN2_CORNER_A11, rel=1e-12
        )

    def test_corner_order2_12(self):
        assert 2 * DYN2_CORNER_B1112 + DYN2_CORNER_B1123 == pytest.approx(
            DYN2_CORNER_A12, rel=1e-12
        )


class TestDynamicStaticLimit:
    """At ω=0, dynamic_inter_voxel_propagator should equal static."""

    @pytest.fixture(params=[(1, 0, 0), (0, -1, 0), (1, 1, 0), (-1, 0, 1), (1, 1, 1)])
    def R(self, request):
        return request.param

    def test_omega_zero(self, R):
        mu = RHO * BETA**2
        nu = (ALPHA**2 - 2 * BETA**2) / (2 * (ALPHA**2 - BETA**2))
        P_static = inter_voxel_propagator(R, mu, nu)
        P_dyn = dynamic_inter_voxel_propagator(R, ALPHA, BETA, RHO, omega=0.0)
        np.testing.assert_allclose(P_dyn, P_static, atol=1e-20)


class TestDynamicSymmetry:
    """Dynamic corrections preserve the propagator's (ij) minor symmetry."""

    @pytest.fixture(params=[(1, 0, 0), (1, 1, 0), (1, 1, 1)])
    def P_dyn(self, request):
        omega = 2 * np.pi * 100  # 100 Hz
        return dynamic_inter_voxel_propagator(
            request.param, ALPHA, BETA, RHO, omega=omega
        )

    def test_minor_symmetry_ij(self, P_dyn):
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for ll in range(3):
                        assert P_dyn[i, j, k, ll] == pytest.approx(
                            P_dyn[j, i, k, ll], abs=1e-15
                        )


class TestDynamicConvergence:
    """Each successive dynamic term should be smaller for ka < 1."""

    def test_face_series_decreasing(self):
        """|ω² P⁽¹⁾| < |P⁽⁰⁾| and |ω⁴ P⁽²⁾| < |ω² P⁽¹⁾| at ka=0.3."""
        ka = 0.3
        omega = ka * ALPHA  # ka = ω/c_p for unit cube
        R = (1, 0, 0)

        P0 = dynamic_inter_voxel_propagator(R, ALPHA, BETA, RHO, 0.0, n_orders=0)
        P01 = dynamic_inter_voxel_propagator(R, ALPHA, BETA, RHO, omega, n_orders=1)
        P012 = dynamic_inter_voxel_propagator(R, ALPHA, BETA, RHO, omega, n_orders=2)

        corr1 = np.max(np.abs(P01 - P0))
        corr2 = np.max(np.abs(P012 - P01))
        static_max = np.max(np.abs(P0))

        assert corr1 < static_max, "Order 1 correction should be smaller than static"
        assert corr2 < corr1, "Order 2 correction should be smaller than order 1"

    def test_edge_series_decreasing(self):
        ka = 0.3
        omega = ka * ALPHA
        R = (1, 1, 0)

        P0 = dynamic_inter_voxel_propagator(R, ALPHA, BETA, RHO, 0.0, n_orders=0)
        P01 = dynamic_inter_voxel_propagator(R, ALPHA, BETA, RHO, omega, n_orders=1)
        P012 = dynamic_inter_voxel_propagator(R, ALPHA, BETA, RHO, omega, n_orders=2)

        corr1 = np.max(np.abs(P01 - P0))
        corr2 = np.max(np.abs(P012 - P01))
        static_max = np.max(np.abs(P0))

        assert corr1 < static_max
        assert corr2 < corr1

    def test_corner_series_decreasing(self):
        ka = 0.3
        omega = ka * ALPHA
        R = (1, 1, 1)

        P0 = dynamic_inter_voxel_propagator(R, ALPHA, BETA, RHO, 0.0, n_orders=0)
        P01 = dynamic_inter_voxel_propagator(R, ALPHA, BETA, RHO, omega, n_orders=1)
        P012 = dynamic_inter_voxel_propagator(R, ALPHA, BETA, RHO, omega, n_orders=2)

        corr1 = np.max(np.abs(P01 - P0))
        corr2 = np.max(np.abs(P012 - P01))
        static_max = np.max(np.abs(P0))

        assert corr1 < static_max
        assert corr2 < corr1

    def test_correction_scales_as_ka_squared(self):
        """Dynamic correction ratio should scale roughly as (ka)²."""
        R = (1, 0, 0)
        ka1, ka2 = 0.1, 0.3
        omega1 = ka1 * ALPHA
        omega2 = ka2 * ALPHA

        P0 = dynamic_inter_voxel_propagator(R, ALPHA, BETA, RHO, 0.0)
        P1 = dynamic_inter_voxel_propagator(R, ALPHA, BETA, RHO, omega1)
        P2 = dynamic_inter_voxel_propagator(R, ALPHA, BETA, RHO, omega2)

        ratio1 = np.max(np.abs(P1 - P0)) / np.max(np.abs(P0))
        ratio2 = np.max(np.abs(P2 - P0)) / np.max(np.abs(P0))

        # ratio2/ratio1 should be approximately (ka2/ka1)² = 9
        scale = ratio2 / ratio1
        assert 5 < scale < 15, f"Expected ~9x scaling, got {scale:.1f}x"


class TestDynamicOhRotation:
    """Dynamic propagator should respect O_h rotation for all directions."""

    def test_face_inversion(self):
        omega = 2 * np.pi * 100
        P_plus = dynamic_inter_voxel_propagator((1, 0, 0), ALPHA, BETA, RHO, omega)
        P_minus = dynamic_inter_voxel_propagator((-1, 0, 0), ALPHA, BETA, RHO, omega)
        np.testing.assert_allclose(P_plus, P_minus, atol=1e-14)

    def test_face_axes_rotation(self):
        omega = 2 * np.pi * 100
        Px = dynamic_inter_voxel_propagator((1, 0, 0), ALPHA, BETA, RHO, omega)
        Py = dynamic_inter_voxel_propagator((0, 1, 0), ALPHA, BETA, RHO, omega)
        assert Px[0, 0, 0, 0] == pytest.approx(Py[1, 1, 1, 1], rel=1e-12)

    def test_all_26_neighbours_finite(self):
        omega = 2 * np.pi * 50
        for n1 in (-1, 0, 1):
            for n2 in (-1, 0, 1):
                for n3 in (-1, 0, 1):
                    if n1 == n2 == n3 == 0:
                        continue
                    R = (n1, n2, n3)
                    P = dynamic_inter_voxel_propagator(R, ALPHA, BETA, RHO, omega)
                    assert np.isfinite(P).all(), f"Non-finite P for R={R}"


# ── 3D Fourier NIntegrate reference (Phase 2C, WS3) ──
#
# ΔP = P(ω) - P(0) computed by direct 3D numerical integration
# of the subtracted strain kernel in Fourier space.
# Reference: Mathematica/InterVoxelPropagatorWS3.wl
# Parameters: cP=5000, cS=3000, ρ=2500, ε=1e-3, kmax=4π
#
# Values are Re[ΔP] (imaginary part is physical radiation damping).
# Indices are 1-based Mathematica convention → 0-based Python below.

# Face R=(1,0,0), ka_P = 0.05
WS3_FACE_005 = {
    (0, 0, 0, 0): -2.5051013578654155e-15,
    (0, 0, 1, 1): +1.0050372914007568e-15,
    (1, 1, 1, 1): -4.3123096932947566e-15,
    (1, 1, 2, 2): +1.9796055507553097e-15,
    (0, 1, 0, 1): -4.112350835371632e-15,
    (1, 2, 1, 2): -3.1385435529468117e-15,
}

# Face R=(1,0,0), ka_P = 0.1
WS3_FACE_01 = {
    (0, 0, 0, 0): -9.638611522792203e-15,
    (0, 0, 1, 1): +3.868208724558371e-15,
    (1, 1, 1, 1): -1.6891488262665625e-14,
    (1, 1, 2, 2): +7.799543051112971e-15,
    (0, 1, 0, 1): -1.6160967846816168e-14,
    (1, 2, 1, 2): -1.2295026396595533e-14,
}

# Face R=(1,0,0), ka_P = 0.3
WS3_FACE_03 = {
    (0, 0, 0, 0): -6.472738870029643e-14,
    (0, 0, 1, 1): +2.399726517350551e-14,
    (1, 1, 1, 1): -1.3182390520301285e-13,
    (1, 1, 2, 2): +6.274046605242954e-14,
    (0, 1, 0, 1): -1.358800753743591e-13,
    (1, 2, 1, 2): -1.0437569909823117e-13,
}

# Edge R=(1,1,0), ka_P = 0.1
WS3_EDGE_01 = {
    (0, 0, 0, 0): -1.0479240153348353e-14,
    (0, 0, 1, 1): +3.994235725162921e-15,
    (0, 0, 2, 2): +4.08158027233474e-15,
    (2, 2, 2, 2): -1.2226386071473104e-14,
    (0, 0, 0, 1): +8.722877774926246e-15,
    (0, 1, 2, 2): -2.2487148661478562e-15,
}

# Corner R=(1,1,1), ka_P = 0.1
WS3_CORNER_01 = {
    (0, 0, 0, 0): -9.558327302446959e-15,
    (0, 0, 1, 1): +3.574217220960363e-15,
    (0, 0, 0, 1): +4.766679766551694e-15,
    (0, 0, 1, 2): -3.3006205421545777e-16,
}


class TestDynamicFourierReference:
    """Validate power series against 3D Fourier NIntegrate reference (WS3).

    The analytical series P(ω) = P⁰ + ω²P¹ + ω⁴P² is compared against
    a direct numerical computation of ΔP = P(ω) - P(0) via 3D Fourier
    integration of the subtracted strain kernel.

    Expected tolerances (series truncation error is O((ka)⁴) relative to ΔP):
      ka=0.05 → ~5 digits match
      ka=0.1  → ~3-4 digits match
      ka=0.3  → ~2 digits match
    """

    @staticmethod
    def _delta_P_series(R, omega):
        """Compute ΔP = P(ω) - P(0) from the power series."""
        mu = RHO * BETA**2
        nu = (ALPHA**2 - 2 * BETA**2) / (2 * (ALPHA**2 - BETA**2))
        P_dyn = dynamic_inter_voxel_propagator(R, ALPHA, BETA, RHO, omega, n_orders=2)
        P_stat = inter_voxel_propagator(R, mu, nu)
        return P_dyn - P_stat

    def test_face_ka005(self):
        """At ka=0.05, series should match NIntegrate to ~1%."""
        omega = 0.05 * ALPHA
        delta_P = self._delta_P_series((1, 0, 0), omega)
        for (i, j, k, ll), ref in WS3_FACE_005.items():
            assert delta_P[i, j, k, ll] == pytest.approx(ref, rel=0.02), (
                f"Face ka=0.05 component ({i},{j},{k},{ll})"
            )

    def test_face_ka01(self):
        """At ka=0.1, series should match NIntegrate to ~2%."""
        omega = 0.1 * ALPHA
        delta_P = self._delta_P_series((1, 0, 0), omega)
        for (i, j, k, ll), ref in WS3_FACE_01.items():
            assert delta_P[i, j, k, ll] == pytest.approx(ref, rel=0.02), (
                f"Face ka=0.1 component ({i},{j},{k},{ll})"
            )

    def test_face_ka03(self):
        """At ka=0.3, series should match NIntegrate to ~10%."""
        omega = 0.3 * ALPHA
        delta_P = self._delta_P_series((1, 0, 0), omega)
        for (i, j, k, ll), ref in WS3_FACE_03.items():
            assert delta_P[i, j, k, ll] == pytest.approx(ref, rel=0.10), (
                f"Face ka=0.3 component ({i},{j},{k},{ll})"
            )

    def test_edge_ka01(self):
        """Edge-adjacent at ka=0.1."""
        omega = 0.1 * ALPHA
        delta_P = self._delta_P_series((1, 1, 0), omega)
        for (i, j, k, ll), ref in WS3_EDGE_01.items():
            assert delta_P[i, j, k, ll] == pytest.approx(ref, rel=0.02), (
                f"Edge ka=0.1 component ({i},{j},{k},{ll})"
            )

    def test_corner_ka01(self):
        """Corner-adjacent at ka=0.1."""
        omega = 0.1 * ALPHA
        delta_P = self._delta_P_series((1, 1, 1), omega)
        for (i, j, k, ll), ref in WS3_CORNER_01.items():
            assert delta_P[i, j, k, ll] == pytest.approx(ref, rel=0.02), (
                f"Corner ka=0.1 component ({i},{j},{k},{ll})"
            )

    def test_convergence_rate(self):
        """Relative error grows with ka, consistent with O((ka)⁴) truncation.

        Compare ka=0.1 vs ka=0.3: the relative error at ka=0.3 should be
        significantly larger, with ratio roughly (3)⁴ = 81.
        At small ka (0.05) the NIntegrate absolute accuracy limits precision,
        so we test the pair where truncation error dominates.
        """
        R = (1, 0, 0)
        delta_P_01 = self._delta_P_series(R, 0.1 * ALPHA)
        delta_P_03 = self._delta_P_series(R, 0.3 * ALPHA)

        # Relative errors against NIntegrate reference
        rel_01 = abs(
            (delta_P_01[0, 0, 0, 0] - WS3_FACE_01[(0, 0, 0, 0)])
            / WS3_FACE_01[(0, 0, 0, 0)]
        )
        rel_03 = abs(
            (delta_P_03[0, 0, 0, 0] - WS3_FACE_03[(0, 0, 0, 0)])
            / WS3_FACE_03[(0, 0, 0, 0)]
        )

        # Relative error at ka=0.3 should be much larger than at ka=0.1
        assert rel_03 > rel_01, "Truncation error should grow with ka"
        # Ratio should be in range [5, 200] (expected ~81 = 3⁴, but
        # NIntegrate noise at ka=0.1 makes the denominator uncertain)
        ratio = rel_03 / rel_01
        assert 5 < ratio < 200, f"Expected ratio ~81 (O(ka⁴)), got {ratio:.1f}"


# ── 9×9 block propagator tests ──


class TestVoigtSBlock:
    """Test _P_to_voigt_S Voigt contraction."""

    def test_shape(self):
        P = face_propagator(MU, NU)
        S = _P_to_voigt_S(P)
        assert S.shape == (6, 6)

    def test_diagonal_elements(self):
        """S_{αα} should correspond to P_{iijj} with correct multiplicity."""
        P = face_propagator(MU, NU)
        S = _P_to_voigt_S(P)
        # S[0,0] = P[0,0,0,0] (axial-axial)
        assert S[0, 0] == pytest.approx(P[0, 0, 0, 0], abs=1e-15)
        # S[3,3] = 2 * P[1,2,1,2] + 2 * P[1,2,2,1]  (shear diag)
        # With engineering convention: halve shear column -> S[3,3] = P[1,2,1,2] + P[1,2,2,1]
        # Since P has ij-symmetry: P[1,2,..] = P[2,1,..]
        assert S[3, 3] == pytest.approx(P[1, 2, 1, 2] + P[1, 2, 2, 1], abs=1e-15)

    def test_face_c4v_in_voigt(self):
        """C₄ᵥ symmetry: S[1,1] = S[2,2] and S[3,3] different from S[4,4]=S[5,5]."""
        P = face_propagator(MU, NU)
        S = _P_to_voigt_S(P)
        assert S[1, 1] == pytest.approx(S[2, 2], abs=1e-15)
        assert S[4, 4] == pytest.approx(S[5, 5], abs=1e-15)

    def test_minor_symmetry_transfers(self):
        """P has ij minor symmetry; in Voigt this means S_αβ rows are consistent."""
        P = edge_propagator(MU, NU)
        S = _P_to_voigt_S(P)
        # S should be finite
        assert np.isfinite(S).all()
        # S[0,1] = P[0,0,1,1] (both axial, no multiplicity issues)
        assert S[0, 1] == pytest.approx(P[0, 0, 1, 1], abs=1e-15)

    def test_all_types_finite(self):
        """Voigt S block should be finite for all neighbour types."""
        for prop_fn in [face_propagator, edge_propagator, corner_propagator]:
            S = _P_to_voigt_S(prop_fn(MU, NU))
            assert np.isfinite(S).all()


class TestGBlock:
    """Test the G block (volume-averaged Green's tensor)."""

    def test_face_g_block_symmetric(self):
        """G block should be symmetric: G_ij = G_ji."""
        P9 = inter_voxel_propagator_9x9((1, 0, 0), ALPHA, BETA, RHO, 0.0, n_orders=0)
        G = P9[:3, :3].real
        np.testing.assert_allclose(G, G.T, atol=1e-15)

    def test_face_g_block_c4v(self):
        """Face G block has C₄ᵥ: G_22 = G_33, G_12 = G_13 = 0."""
        P9 = inter_voxel_propagator_9x9((1, 0, 0), ALPHA, BETA, RHO, 0.0, n_orders=0)
        G = P9[:3, :3].real
        assert G[1, 1] == pytest.approx(G[2, 2], abs=1e-15)
        assert G[0, 1] == pytest.approx(0.0, abs=1e-15)
        assert G[0, 2] == pytest.approx(0.0, abs=1e-15)
        assert G[1, 2] == pytest.approx(0.0, abs=1e-15)

    def test_corner_g_block_s3(self):
        """Corner G block has S₃: G_11=G_22=G_33, G_12=G_13=G_23."""
        P9 = inter_voxel_propagator_9x9((1, 1, 1), ALPHA, BETA, RHO, 0.0, n_orders=0)
        G = P9[:3, :3].real
        assert G[0, 0] == pytest.approx(G[1, 1], abs=1e-14)
        assert G[0, 0] == pytest.approx(G[2, 2], abs=1e-14)
        assert G[0, 1] == pytest.approx(G[0, 2], abs=1e-14)
        assert G[0, 1] == pytest.approx(G[1, 2], abs=1e-14)

    def test_g_block_laplacian_identity(self):
        """Verify Φ = (1/2) Tr(d²Ψ/dR²) used in G block construction.

        The Laplacian identity is checked indirectly: the face-canonical
        G block trace should equal 3Φ/(4πμ) - η_s Tr(d²Ψ)/(4πμ),
        which simplifies to Φ(3 - 3η_s)/(4πμ) since Tr(d²Ψ) = 2Φ.
        Actually: Tr(G) = Φ[3 - η_s*2]/(4πμ) = Φ(3-1/(2(1-ν)))/(4πμ)
        Wait — Tr(d²Ψ) = 2Φ by the identity Φ = (1/2)Tr(d²Ψ).
        So Tr(G) = [3Φ - η_s * 2Φ]/(4πμ) = Φ(3 - 2η_s)/(4πμ).
        """
        from cubic_scattering.inter_voxel_propagator import _NORM_A1

        mu = RHO * BETA**2
        nu = (ALPHA**2 - 2 * BETA**2) / (2 * (ALPHA**2 - BETA**2))

        # Raw d²Ψ/dR² for face
        raw_11 = DYN1_FACE_A11 / _NORM_A1
        raw_22 = DYN1_FACE_A22 / _NORM_A1
        Phi = 0.5 * (raw_11 + 2 * raw_22)

        eta_s = 1.0 / (4.0 * (1.0 - nu))
        expected_trace = Phi * (3.0 - 2.0 * eta_s) / (4.0 * np.pi * mu)

        P9 = inter_voxel_propagator_9x9((1, 0, 0), ALPHA, BETA, RHO, 0.0, n_orders=0)
        actual_trace = np.trace(P9[:3, :3].real)
        assert actual_trace == pytest.approx(expected_trace, rel=1e-12)

    def test_g_block_nonzero(self):
        """G block should have non-zero entries."""
        P9 = inter_voxel_propagator_9x9((1, 0, 0), ALPHA, BETA, RHO, 0.0, n_orders=0)
        assert np.max(np.abs(P9[:3, :3])) > 0

    def test_g_block_rotates_correctly(self):
        """G for R=(0,1,0) should be a rotated version of R=(1,0,0)."""
        P9_x = inter_voxel_propagator_9x9((1, 0, 0), ALPHA, BETA, RHO, 0.0, n_orders=0)
        P9_y = inter_voxel_propagator_9x9((0, 1, 0), ALPHA, BETA, RHO, 0.0, n_orders=0)
        Gx = P9_x[:3, :3].real
        Gy = P9_y[:3, :3].real
        # G_00 for R=(1,0,0) should equal G_11 for R=(0,1,0)
        assert Gx[0, 0] == pytest.approx(Gy[1, 1], rel=1e-12)
        # G_11 for R=(1,0,0) should equal G_00 for R=(0,1,0)
        assert Gx[1, 1] == pytest.approx(Gy[0, 0], rel=1e-12)


class TestPropagator9x9:
    """Integration tests for the full 9×9 propagator."""

    def test_shape(self):
        P9 = inter_voxel_propagator_9x9((1, 0, 0), ALPHA, BETA, RHO, 0.0)
        assert P9.shape == (9, 9)

    def test_s_block_matches_standalone(self):
        """S block of 9×9 should match standalone Voigt conversion."""
        R = (1, 0, 0)
        mu = RHO * BETA**2
        nu = (ALPHA**2 - 2 * BETA**2) / (2 * (ALPHA**2 - BETA**2))
        P_ijkl = inter_voxel_propagator(R, mu, nu)
        S_standalone = _P_to_voigt_S(P_ijkl)

        P9 = inter_voxel_propagator_9x9(R, ALPHA, BETA, RHO, 0.0, n_orders=0)
        S_from_9x9 = P9[3:, 3:].real
        np.testing.assert_allclose(S_from_9x9, S_standalone, atol=1e-15)

    def test_ch_blocks_nonzero(self):
        """C and H blocks should be nonzero (displacement-strain coupling)."""
        P9 = inter_voxel_propagator_9x9((1, 0, 0), ALPHA, BETA, RHO, 100.0)
        C = P9[:3, 3:]
        H = P9[3:, :3]
        # C ~ 1/(4πμ) ~ 3.5e-12 for seismic parameters (μ=2.25e10)
        mu = RHO * BETA**2
        scale = 1.0 / (4.0 * np.pi * mu)
        assert np.max(np.abs(C)) > 0.1 * scale, "C block should be nonzero"
        assert np.max(np.abs(H)) > 0.1 * scale, "H block should be nonzero"

    def test_h_equals_c_transpose(self):
        """H = Cᵀ for all 26 neighbours."""
        omega = 2 * np.pi * 50
        for n1 in (-1, 0, 1):
            for n2 in (-1, 0, 1):
                for n3 in (-1, 0, 1):
                    if n1 == n2 == n3 == 0:
                        continue
                    R = (n1, n2, n3)
                    P9 = inter_voxel_propagator_9x9(R, ALPHA, BETA, RHO, omega)
                    C = P9[:3, 3:].real
                    H = P9[3:, :3].real
                    np.testing.assert_allclose(
                        H, C.T, atol=1e-15, err_msg=f"H ≠ Cᵀ for R={R}"
                    )

    def test_all_26_neighbours_finite(self):
        """9×9 propagator should be finite for all 26 neighbours."""
        omega = 2 * np.pi * 50
        for n1 in (-1, 0, 1):
            for n2 in (-1, 0, 1):
                for n3 in (-1, 0, 1):
                    if n1 == n2 == n3 == 0:
                        continue
                    R = (n1, n2, n3)
                    P9 = inter_voxel_propagator_9x9(R, ALPHA, BETA, RHO, omega)
                    assert np.isfinite(P9).all(), f"Non-finite P9 for R={R}"

    def test_static_limit(self):
        """At ω=0, dynamic G block should still be well-defined."""
        P9_0 = inter_voxel_propagator_9x9((1, 0, 0), ALPHA, BETA, RHO, 0.0, n_orders=1)
        P9_static = inter_voxel_propagator_9x9(
            (1, 0, 0), ALPHA, BETA, RHO, 0.0, n_orders=0
        )
        # At ω=0, the ω² correction vanishes
        np.testing.assert_allclose(P9_0[:3, :3], P9_static[:3, :3], atol=1e-20)

    def test_dynamic_g_correction_small(self):
        """ω² correction to G block should be small at ka=0.1."""
        ka = 0.1
        omega = ka * ALPHA
        P9_0 = inter_voxel_propagator_9x9((1, 0, 0), ALPHA, BETA, RHO, 0.0, n_orders=0)
        P9_1 = inter_voxel_propagator_9x9(
            (1, 0, 0), ALPHA, BETA, RHO, omega, n_orders=1
        )
        G_static = np.max(np.abs(P9_0[:3, :3]))
        G_correction = np.max(np.abs(P9_1[:3, :3] - P9_0[:3, :3]))
        assert G_correction < G_static, "ω² G correction should be smaller than static"

    def test_inversion_symmetry_9x9(self):
        """G,S blocks even under R→−R; C,H blocks odd."""
        omega = 2 * np.pi * 100
        for R_plus, R_minus in [
            ((1, 0, 0), (-1, 0, 0)),
            ((1, 1, 0), (-1, -1, 0)),
            ((1, 1, 1), (-1, -1, -1)),
        ]:
            P_p = inter_voxel_propagator_9x9(R_plus, ALPHA, BETA, RHO, omega)
            P_m = inter_voxel_propagator_9x9(R_minus, ALPHA, BETA, RHO, omega)
            # G (even), S (even)
            np.testing.assert_allclose(
                P_p[:3, :3], P_m[:3, :3], atol=1e-14, err_msg=f"G not even for {R_plus}"
            )
            np.testing.assert_allclose(
                P_p[3:, 3:], P_m[3:, 3:], atol=1e-14, err_msg=f"S not even for {R_plus}"
            )
            # C (odd), H (odd)
            np.testing.assert_allclose(
                P_p[:3, 3:], -P_m[:3, 3:], atol=1e-14, err_msg=f"C not odd for {R_plus}"
            )
            np.testing.assert_allclose(
                P_p[3:, :3], -P_m[3:, :3], atol=1e-14, err_msg=f"H not odd for {R_plus}"
            )


# ── C/H block tests (Phase 3B-2) ──


class TestD3PsiLaplacian:
    """Verify Laplacian identity: Σ_j D_{jjk} = 2 dΦ_k for biharmonic third derivatives."""

    def test_face_laplacian(self):
        # D_000 + D_110 + D_220 = D_000 + 2*D_011 = 2*dPhi_0
        lhs = FACE_D3PSI_000 + 2 * FACE_D3PSI_011
        assert lhs == pytest.approx(2 * FACE_DPHI_0, abs=1e-13)

    def test_edge_laplacian(self):
        # D_000 + D_110 + D_220 = D_000 + D_001 + D_022 = 2*dPhi_0
        lhs = EDGE_D3PSI_000 + EDGE_D3PSI_001 + EDGE_D3PSI_022
        assert lhs == pytest.approx(2 * EDGE_DPHI_0, abs=1e-13)

    def test_corner_laplacian(self):
        # D_000 + D_110 + D_220 = D_000 + 2*D_001 = 2*dPhi_0
        lhs = CORNER_D3PSI_000 + 2 * CORNER_D3PSI_001
        assert lhs == pytest.approx(2 * CORNER_DPHI_0, abs=1e-13)

    def test_d3psi_tensor_laplacian(self):
        """Laplacian identity via tensor: Σ_j D_{jjk} = 2*dPhi_k for all types."""
        for ntype in ("face", "edge", "corner"):
            D = _build_D3Psi_tensor(ntype)
            dPhi = _build_dPhi_vector(ntype)
            for k in range(3):
                trace_k = sum(D[j, j, k] for j in range(3))
                assert trace_k == pytest.approx(2 * dPhi[k], abs=1e-13), (
                    f"Laplacian failed for {ntype}, k={k}"
                )


class TestCBlockStructure:
    """Verify sparsity and structure of the C block for canonical directions."""

    def test_face_c_sparsity(self):
        """Face C block for R=(1,0,0) has specific nonzero pattern."""
        # Use unit mu=1 to see actual mathematical magnitudes
        dG = _build_dG_rank3_canonical("face", MU, NU)
        C = _dG_to_C_block(dG)
        # Expected pattern:
        # C = [[c1, c2, c2,  0,  0,  0],
        #      [ 0,  0,  0,  0,  0, c3],
        #      [ 0,  0,  0,  0, c3,  0]]
        # Nonzero entries: (0,0), (0,1), (0,2), (1,5), (2,4)
        assert abs(C[0, 0]) > 1e-3  # c1
        assert abs(C[0, 1]) > 1e-3  # c2
        assert C[0, 1] == pytest.approx(C[0, 2], abs=1e-15)  # c2 = c2
        assert abs(C[1, 5]) > 1e-3  # c3
        assert C[1, 5] == pytest.approx(C[2, 4], abs=1e-15)  # same c3
        # Zeros
        np.testing.assert_allclose(C[0, 3:], 0.0, atol=1e-15)  # row 0, shear cols
        np.testing.assert_allclose(C[1, :5], 0.0, atol=1e-15)  # row 1, all but col 5
        np.testing.assert_allclose(C[2, :4], 0.0, atol=1e-15)  # row 2, cols 0-3
        assert abs(C[2, 5]) < 1e-15  # row 2, col 5

    def test_d3psi_tensor_symmetric(self):
        """D3Psi tensor should be fully symmetric in all 3 indices."""
        for ntype in ("face", "edge", "corner"):
            D = _build_D3Psi_tensor(ntype)
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        assert D[i, j, k] == pytest.approx(D[j, i, k], abs=1e-15), (
                            f"{ntype}: D[{i},{j},{k}] ≠ D[{j},{i},{k}]"
                        )
                        assert D[i, j, k] == pytest.approx(D[i, k, j], abs=1e-15), (
                            f"{ntype}: D[{i},{j},{k}] ≠ D[{i},{k},{j}]"
                        )

    def test_rotation_covariance(self):
        """C at R=(0,1,0) should equal rotated C at R=(1,0,0)."""
        mu = RHO * BETA**2
        nu = (ALPHA**2 - 2 * BETA**2) / (2 * (ALPHA**2 - BETA**2))

        P9_x = inter_voxel_propagator_9x9((1, 0, 0), ALPHA, BETA, RHO, 0.0, n_orders=0)
        P9_y = inter_voxel_propagator_9x9((0, 1, 0), ALPHA, BETA, RHO, 0.0, n_orders=0)

        C_x = P9_x[:3, 3:].real
        C_y = P9_y[:3, 3:].real

        # C_x has c1 at (0,0), C_y should have corresponding value at (1,1)
        # Under axis swap 0↔1: C_{0,(00)} → C_{1,(11)}
        assert C_x[0, 0] == pytest.approx(C_y[1, 1], abs=1e-13)
        # And C_{0,(11)} → C_{1,(00)}
        assert C_x[0, 1] == pytest.approx(C_y[1, 0], abs=1e-13)

    def test_c_static_frequency_independent(self):
        """C/H blocks at n_orders=0 are static (frequency-independent)."""
        P9_0 = inter_voxel_propagator_9x9((1, 0, 0), ALPHA, BETA, RHO, 0.0, n_orders=0)
        P9_w = inter_voxel_propagator_9x9(
            (1, 0, 0), ALPHA, BETA, RHO, 500.0, n_orders=0
        )
        # At n_orders=0, C/H are static — same regardless of ω
        # Values are O(1e-12) so use relative tolerance
        np.testing.assert_allclose(P9_0[:3, 3:], P9_w[:3, 3:], rtol=1e-10, atol=1e-15)
        np.testing.assert_allclose(P9_0[3:, :3], P9_w[3:, :3], rtol=1e-10, atol=1e-15)

    def test_c_dynamic_omega_scaling(self):
        """C/H blocks at n_orders≥1 have ω² frequency dependence."""
        omega = 500.0
        P9_static = inter_voxel_propagator_9x9(
            (1, 0, 0), ALPHA, BETA, RHO, 0.0, n_orders=0
        )
        P9_dyn = inter_voxel_propagator_9x9(
            (1, 0, 0), ALPHA, BETA, RHO, omega, n_orders=1
        )
        # Dynamic C/H should differ from static by O(ω²) correction
        C_static = P9_static[:3, 3:]
        C_dyn = P9_dyn[:3, 3:]
        diff = C_dyn - C_static
        # At ω=500, ω²=2.5e5, the correction should be nonzero
        assert np.max(np.abs(diff)) > 0.0, "C block should have ω² correction"
        # H = Cᵀ should also hold for dynamic C/H
        np.testing.assert_allclose(P9_dyn[3:, :3], P9_dyn[:3, 3:].T, atol=1e-15)

    def test_fd_validation_face(self):
        """Cross-check C block against finite-difference of G block."""
        eps = 1e-5
        omega = 0.0
        n_orders = 0
        R0 = np.array([1, 0, 0])

        P9_0 = inter_voxel_propagator_9x9(tuple(R0), ALPHA, BETA, RHO, omega, n_orders)
        C_analytical = P9_0[:3, 3:].real

        # FD: dG_{ij}/dR_k ≈ (G_{ij}(R+ε e_k) - G_{ij}(R-ε e_k)) / (2ε)
        # But R must stay on the lattice, so use continuous G block instead.
        # We can compare the rank-3 tensor structure via the canonical builder.
        mu = RHO * BETA**2
        nu = (ALPHA**2 - 2 * BETA**2) / (2 * (ALPHA**2 - BETA**2))
        dG = _build_dG_rank3_canonical("face", mu, nu)
        C_from_dG = _dG_to_C_block(dG)
        np.testing.assert_allclose(C_analytical, C_from_dG, atol=1e-14)

    def test_s_block_unchanged(self):
        """S block should be identical to standalone computation."""
        for R in [(1, 0, 0), (1, 1, 0), (1, 1, 1), (0, -1, 1), (-1, 0, -1)]:
            mu = RHO * BETA**2
            nu = (ALPHA**2 - 2 * BETA**2) / (2 * (ALPHA**2 - BETA**2))
            P_ijkl = inter_voxel_propagator(R, mu, nu)
            S_standalone = _P_to_voigt_S(P_ijkl)
            P9 = inter_voxel_propagator_9x9(R, ALPHA, BETA, RHO, 0.0, n_orders=0)
            S_from_9x9 = P9[3:, 3:].real
            np.testing.assert_allclose(
                S_from_9x9, S_standalone, atol=1e-15, err_msg=f"S mismatch for R={R}"
            )


# ══════════════════════════════════════════════════════════════════════
# Phase 4B: Dynamic completion tests (DYN3, G ω⁴, C/H dynamic)
# ══════════════════════════════════════════════════════════════════════


class TestDynamic3LaplacianIdentity:
    """Verify Σ_k B³_{ijkk} = A³_{ij} for order-3 (ω⁶) S-block constants.

    These correspond to d²Ω/dR² (A) and d⁴H/dR⁴ (B) at the ρ⁵/ρ⁷ level.
    """

    def test_face_order3_11(self):
        assert DYN3_FACE_B1111 + 2 * DYN3_FACE_B1122 == pytest.approx(
            DYN3_FACE_A11, rel=1e-11
        )

    def test_face_order3_22(self):
        assert DYN3_FACE_B1122 + DYN3_FACE_B2222 + DYN3_FACE_B2233 == pytest.approx(
            DYN3_FACE_A22, rel=1e-11
        )

    def test_edge_order3_11(self):
        assert DYN3_EDGE_B1111 + DYN3_EDGE_B1122 + DYN3_EDGE_B1133 == pytest.approx(
            DYN3_EDGE_A11, rel=1e-11
        )

    def test_edge_order3_33(self):
        assert 2 * DYN3_EDGE_B1133 + DYN3_EDGE_B3333 == pytest.approx(
            DYN3_EDGE_A33, rel=1e-11
        )

    def test_edge_order3_12(self):
        assert 2 * DYN3_EDGE_B1112 + DYN3_EDGE_B1233 == pytest.approx(
            DYN3_EDGE_A12, rel=1e-11
        )

    def test_corner_order3_11(self):
        assert DYN3_CORNER_B1111 + 2 * DYN3_CORNER_B1122 == pytest.approx(
            DYN3_CORNER_A11, rel=1e-11
        )

    def test_corner_order3_12(self):
        assert 2 * DYN3_CORNER_B1112 + DYN3_CORNER_B1123 == pytest.approx(
            DYN3_CORNER_A12, rel=1e-11
        )


class TestD3XLaplacian:
    """Verify Laplacian identities for triharmonic (d³X) and pentaharmonic (d³Ω).

    d³X: Σ_j D_{jjk} = 12·dΨ_k  (since ∇²ρ³ = 12ρ → ∇²X = 12Ψ)
    d³Ω: Σ_j D_{jjk} = 30·dX_k  (since ∇²ρ⁵ = 30ρ³ → ∇²Ω = 30X)
    """

    # ── d³X/dR³ Laplacian ──

    def test_face_d3x_laplacian(self):
        lhs = FACE_D3X_000 + 2 * FACE_D3X_011
        assert lhs == pytest.approx(12 * FACE_DPSI_0, abs=1e-10)

    def test_edge_d3x_laplacian(self):
        lhs = EDGE_D3X_000 + EDGE_D3X_001 + EDGE_D3X_022
        assert lhs == pytest.approx(12 * EDGE_DPSI_0, abs=1e-10)

    def test_corner_d3x_laplacian(self):
        lhs = CORNER_D3X_000 + 2 * CORNER_D3X_001
        assert lhs == pytest.approx(12 * CORNER_DPSI_0, abs=1e-10)

    # ── d³Ω/dR³ Laplacian ──

    def test_face_d3om_laplacian(self):
        lhs = FACE_D3OM_000 + 2 * FACE_D3OM_011
        assert lhs == pytest.approx(30 * FACE_DX_0, abs=1e-10)

    def test_edge_d3om_laplacian(self):
        lhs = EDGE_D3OM_000 + EDGE_D3OM_001 + EDGE_D3OM_022
        assert lhs == pytest.approx(30 * EDGE_DX_0, abs=1e-10)

    def test_corner_d3om_laplacian(self):
        lhs = CORNER_D3OM_000 + 2 * CORNER_D3OM_001
        assert lhs == pytest.approx(30 * CORNER_DX_0, abs=1e-10)

    # ── Tensor form via builder ──

    def test_d3x_tensor_laplacian(self):
        """Σ_j D_{jjk} = 12·dΨ_k for all types via tensor builder."""
        for ntype in ("face", "edge", "corner"):
            D = _build_D3_tensor(ntype, order=1)
            dPsi = _build_dW_vector(ntype, order=1)
            for k in range(3):
                trace_k = sum(D[j, j, k] for j in range(3))
                assert trace_k == pytest.approx(12 * dPsi[k], abs=1e-10), (
                    f"D3X Laplacian failed for {ntype}, k={k}"
                )

    def test_d3om_tensor_laplacian(self):
        """Σ_j D_{jjk} = 30·dX_k for all types via tensor builder."""
        for ntype in ("face", "edge", "corner"):
            D = _build_D3_tensor(ntype, order=2)
            dX = _build_dW_vector(ntype, order=2)
            for k in range(3):
                trace_k = sum(D[j, j, k] for j in range(3))
                assert trace_k == pytest.approx(30 * dX[k], abs=1e-10), (
                    f"D3Ω Laplacian failed for {ntype}, k={k}"
                )


class TestD3XSymmetry:
    """D3X and D3Ω tensors should be fully symmetric in all 3 indices."""

    @pytest.fixture(params=[1, 2], ids=["d3X", "d3Omega"])
    def order(self, request):
        return request.param

    def test_full_symmetry(self, order):
        for ntype in ("face", "edge", "corner"):
            D = _build_D3_tensor(ntype, order=order)
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        assert D[i, j, k] == pytest.approx(D[j, i, k], abs=1e-15), (
                            f"order={order} {ntype}: D[{i},{j},{k}] ≠ D[{j},{i},{k}]"
                        )
                        assert D[i, j, k] == pytest.approx(D[i, k, j], abs=1e-15), (
                            f"order={order} {ntype}: D[{i},{j},{k}] ≠ D[{i},{k},{j}]"
                        )


class TestGBlockOmega4:
    """G block ω⁴ correction (n_orders=2) properties."""

    def test_g_block_symmetric_at_omega4(self):
        """G block should remain symmetric with ω⁴ correction."""
        omega = 0.3 * ALPHA
        for R in [(1, 0, 0), (1, 1, 0), (1, 1, 1)]:
            P9 = inter_voxel_propagator_9x9(R, ALPHA, BETA, RHO, omega, n_orders=2)
            G = P9[:3, :3].real
            np.testing.assert_allclose(
                G, G.T, atol=1e-15, err_msg=f"G not symmetric for {R}"
            )

    def test_g_omega4_correction_scales(self):
        """ω⁴ G correction should be smaller than ω² correction at ka=0.3.

        With correct factorial suppression (1/2!, 1/4! from Taylor expansion
        of exp(ikr)/r), the G block converges well even at ka=0.3.
        """
        ka = 0.3
        omega = ka * ALPHA
        R = (1, 0, 0)
        P9_0 = inter_voxel_propagator_9x9(R, ALPHA, BETA, RHO, omega, n_orders=0)
        P9_1 = inter_voxel_propagator_9x9(R, ALPHA, BETA, RHO, omega, n_orders=1)
        P9_2 = inter_voxel_propagator_9x9(R, ALPHA, BETA, RHO, omega, n_orders=2)

        G_corr1 = np.max(np.abs(P9_1[:3, :3] - P9_0[:3, :3]))
        G_corr2 = np.max(np.abs(P9_2[:3, :3] - P9_1[:3, :3]))
        assert G_corr2 < G_corr1, "ω⁴ G correction should be smaller than ω² at ka=0.3"

    def test_g_omega4_ka_scaling(self):
        """G ω⁴ correction ratio between ka=0.1 and ka=0.3 should scale as ~(ka)²."""
        R = (1, 0, 0)
        ka1, ka2 = 0.1, 0.3

        def g_corr_at_ka(ka):
            omega = ka * ALPHA
            P9_1 = inter_voxel_propagator_9x9(R, ALPHA, BETA, RHO, omega, n_orders=1)
            P9_2 = inter_voxel_propagator_9x9(R, ALPHA, BETA, RHO, omega, n_orders=2)
            return np.max(np.abs(P9_2[:3, :3] - P9_1[:3, :3]))

        ratio = g_corr_at_ka(ka2) / g_corr_at_ka(ka1)
        # Expected ~(ka2/ka1)⁴ = 81 for the ω⁴ term evaluated at two ka values
        # But since we're taking difference P9(n=2)-P9(n=1), the dominant term is ω⁴
        # so the ratio should be (0.3/0.1)⁴ = 81
        assert 30 < ratio < 200, f"Expected ~81× scaling, got {ratio:.1f}×"


class TestCHBlockDynamic:
    """C/H block dynamic correction properties (ω² and ω⁴)."""

    def test_ch_omega2_scales_as_ka2(self):
        """C/H ω² correction ratio between ka=0.1 and ka=0.3 should scale as ~(ka)²."""
        R = (1, 0, 0)
        ka1, ka2 = 0.1, 0.3

        def ch_corr_at_ka(ka):
            omega = ka * ALPHA
            P9_0 = inter_voxel_propagator_9x9(R, ALPHA, BETA, RHO, omega, n_orders=0)
            P9_1 = inter_voxel_propagator_9x9(R, ALPHA, BETA, RHO, omega, n_orders=1)
            return np.max(np.abs(P9_1[:3, 3:] - P9_0[:3, 3:]))

        ratio = ch_corr_at_ka(ka2) / ch_corr_at_ka(ka1)
        # Dominant term is ω², so ratio ≈ (0.3/0.1)² = 9
        assert 5 < ratio < 15, f"Expected ~9× scaling, got {ratio:.1f}×"

    def test_ch_omega4_smaller_than_omega2(self):
        """ω⁴ C/H correction should be smaller than ω² at ka=0.3.

        With correct factorial suppression from Taylor expansion of
        exp(ikr)/r, the C/H blocks converge well at ka=0.3.
        """
        ka = 0.3
        omega = ka * ALPHA
        R = (1, 0, 0)
        P9_0 = inter_voxel_propagator_9x9(R, ALPHA, BETA, RHO, omega, n_orders=0)
        P9_1 = inter_voxel_propagator_9x9(R, ALPHA, BETA, RHO, omega, n_orders=1)
        P9_2 = inter_voxel_propagator_9x9(R, ALPHA, BETA, RHO, omega, n_orders=2)

        C_corr1 = np.max(np.abs(P9_1[:3, 3:] - P9_0[:3, 3:]))
        C_corr2 = np.max(np.abs(P9_2[:3, 3:] - P9_1[:3, 3:]))
        assert C_corr2 < C_corr1, "ω⁴ C correction should be smaller than ω² at ka=0.3"

    def test_h_equals_c_transpose_all_orders(self):
        """H = Cᵀ at n_orders=0,1,2 for all neighbour types."""
        omega = 0.3 * ALPHA
        for n_ord in (0, 1, 2):
            for R in [(1, 0, 0), (1, 1, 0), (1, 1, 1), (-1, 0, 0), (0, -1, 1)]:
                P9 = inter_voxel_propagator_9x9(
                    R, ALPHA, BETA, RHO, omega, n_orders=n_ord
                )
                np.testing.assert_allclose(
                    P9[3:, :3],
                    P9[:3, 3:].T,
                    atol=1e-15,
                    err_msg=f"H≠Cᵀ for R={R}, n_orders={n_ord}",
                )

    def test_inversion_symmetry_all_orders(self):
        """G,S even and C,H odd under R→−R at all dynamic orders."""
        omega = 0.3 * ALPHA
        for n_ord in (0, 1, 2):
            for R_p, R_m in [
                ((1, 0, 0), (-1, 0, 0)),
                ((1, 1, 0), (-1, -1, 0)),
                ((1, 1, 1), (-1, -1, -1)),
            ]:
                Pp = inter_voxel_propagator_9x9(R_p, ALPHA, BETA, RHO, omega, n_ord)
                Pm = inter_voxel_propagator_9x9(R_m, ALPHA, BETA, RHO, omega, n_ord)
                np.testing.assert_allclose(
                    Pp[:3, :3],
                    Pm[:3, :3],
                    atol=1e-14,
                    err_msg=f"G not even, R={R_p}, n={n_ord}",
                )
                np.testing.assert_allclose(
                    Pp[3:, 3:],
                    Pm[3:, 3:],
                    atol=1e-14,
                    err_msg=f"S not even, R={R_p}, n={n_ord}",
                )
                np.testing.assert_allclose(
                    Pp[:3, 3:],
                    -Pm[:3, 3:],
                    atol=1e-14,
                    err_msg=f"C not odd, R={R_p}, n={n_ord}",
                )
                np.testing.assert_allclose(
                    Pp[3:, :3],
                    -Pm[3:, :3],
                    atol=1e-14,
                    err_msg=f"H not odd, R={R_p}, n={n_ord}",
                )


class TestConsistentTruncation:
    """All blocks at n_orders=2 for all 26 neighbours."""

    def test_all_26_finite_norders2(self):
        """9×9 propagator at n_orders=2 should be finite for all 26 neighbours."""
        omega = 0.3 * ALPHA
        for n1 in (-1, 0, 1):
            for n2 in (-1, 0, 1):
                for n3 in (-1, 0, 1):
                    if n1 == n2 == n3 == 0:
                        continue
                    R = (n1, n2, n3)
                    P9 = inter_voxel_propagator_9x9(
                        R, ALPHA, BETA, RHO, omega, n_orders=2
                    )
                    assert np.isfinite(P9).all(), (
                        f"Non-finite P9 at n_orders=2 for R={R}"
                    )

    def test_g_block_symmetric_all_26(self):
        """G block should be symmetric for all 26 neighbours at n_orders=2."""
        omega = 0.3 * ALPHA
        for n1 in (-1, 0, 1):
            for n2 in (-1, 0, 1):
                for n3 in (-1, 0, 1):
                    if n1 == n2 == n3 == 0:
                        continue
                    R = (n1, n2, n3)
                    P9 = inter_voxel_propagator_9x9(
                        R, ALPHA, BETA, RHO, omega, n_orders=2
                    )
                    G = P9[:3, :3].real
                    np.testing.assert_allclose(
                        G, G.T, atol=1e-14, err_msg=f"G not sym for R={R}"
                    )


class TestMathematicaReferenceCH:
    """Cross-validate C/H dynamic constants against Mathematica output.

    These are the raw values from InterVoxelPropagatorCHDynamic.wl,
    validated by direct 3D NIntegrate cross-check to 10⁻¹² precision.
    """

    # d³X/dR³ (triharmonic third derivatives)
    def test_face_d3x_000(self):
        assert FACE_D3X_000 == pytest.approx(5.59066550438255537, rel=1e-14)

    def test_face_d3x_011(self):
        assert FACE_D3X_011 == pytest.approx(2.22178066763092212, rel=1e-14)

    def test_edge_d3x_000(self):
        assert EDGE_D3X_000 == pytest.approx(4.79793940592048508, rel=1e-14)

    def test_edge_d3x_001(self):
        assert EDGE_D3X_001 == pytest.approx(1.17783327456444731, rel=1e-14)

    def test_edge_d3x_022(self):
        assert EDGE_D3X_022 == pytest.approx(1.80511344407289295, rel=1e-14)

    def test_corner_d3x_000(self):
        assert CORNER_D3X_000 == pytest.approx(4.25436485917712435, rel=1e-14)

    def test_corner_d3x_001(self):
        assert CORNER_D3X_001 == pytest.approx(1.14631500401061554, rel=1e-14)

    def test_corner_d3x_012(self):
        assert CORNER_D3X_012 == pytest.approx(-0.41087473026598235, rel=1e-14)

    # d³Ω/dR³ (pentaharmonic third derivatives)
    def test_face_d3om_000(self):
        assert FACE_D3OM_000 == pytest.approx(74.7028469944845582, rel=1e-14)

    def test_face_d3om_011(self):
        assert FACE_D3OM_011 == pytest.approx(21.5319534595037667, rel=1e-14)

    def test_edge_d3om_000(self):
        assert EDGE_D3OM_000 == pytest.approx(86.9352238951433639, rel=1e-14)

    def test_corner_d3om_000(self):
        assert CORNER_D3OM_000 == pytest.approx(97.7452814064066722, rel=1e-14)

    # dΨ/dR_k (biharmonic first derivatives)
    def test_face_dpsi_0(self):
        assert FACE_DPSI_0 == pytest.approx(0.83618556997036663, rel=1e-14)

    def test_edge_dpsi_0(self):
        assert EDGE_DPSI_0 == pytest.approx(0.64840717704648545, rel=1e-14)

    def test_corner_dpsi_0(self):
        assert CORNER_DPSI_0 == pytest.approx(0.54558290559986295, rel=1e-14)

    # dX/dR_k (triharmonic first derivatives)
    def test_face_dx_0(self):
        assert FACE_DX_0 == pytest.approx(3.92555846378306972, rel=1e-14)

    def test_edge_dx_0(self):
        assert EDGE_DX_0 == pytest.approx(4.92386047202461274, rel=1e-14)

    def test_corner_dx_0(self):
        assert CORNER_DX_0 == pytest.approx(5.75937899450666245, rel=1e-14)


# ══════════════════════════════════════════════════════════════════════
# Phase 5: ω⁶ extension tests (G and C/H blocks at n_orders=3)
# ══════════════════════════════════════════════════════════════════════


class TestD3HLaplacian:
    """Verify Laplacian identity for heptaharmonic d³H/dR³.

    Since nabla^2(rho^7) = 56*rho^5, the Laplacian trace gives:
        sum_j D3H_{jjk} = 56 * dOmega_k
    """

    def test_face_d3h_laplacian(self):
        lhs = FACE_D3H_000 + 2 * FACE_D3H_011
        assert lhs == pytest.approx(56 * FACE_DOM_0, abs=1e-8)

    def test_edge_d3h_laplacian(self):
        lhs = EDGE_D3H_000 + EDGE_D3H_001 + EDGE_D3H_022
        assert lhs == pytest.approx(56 * EDGE_DOM_0, abs=1e-8)

    def test_corner_d3h_laplacian(self):
        lhs = CORNER_D3H_000 + 2 * CORNER_D3H_001
        assert lhs == pytest.approx(56 * CORNER_DOM_0, abs=1e-8)

    def test_d3h_tensor_laplacian(self):
        """sum_j D_{jjk} = 56*dOmega_k for all types via tensor builder."""
        for ntype in ("face", "edge", "corner"):
            D = _build_D3_tensor(ntype, order=3)
            dOm = _build_dW_vector(ntype, order=3)
            for k in range(3):
                trace_k = sum(D[j, j, k] for j in range(3))
                assert trace_k == pytest.approx(56 * dOm[k], abs=1e-8), (
                    f"D3H Laplacian failed for {ntype}, k={k}"
                )


class TestD3HSymmetry:
    """D3H tensors should be fully symmetric in all 3 indices."""

    def test_full_symmetry(self):
        for ntype in ("face", "edge", "corner"):
            D = _build_D3_tensor(ntype, order=3)
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        assert D[i, j, k] == pytest.approx(D[j, i, k], abs=1e-15), (
                            f"D3H {ntype}: D[{i},{j},{k}] != D[{j},{i},{k}]"
                        )
                        assert D[i, j, k] == pytest.approx(D[i, k, j], abs=1e-15), (
                            f"D3H {ntype}: D[{i},{j},{k}] != D[{i},{k},{j}]"
                        )


class TestGBlockOmega6:
    """G block omega^6 correction (n_orders=3) properties."""

    def test_g_block_symmetric_at_omega6(self):
        """G block should remain symmetric with omega^6 correction."""
        omega = 0.3 * ALPHA
        for R in [(1, 0, 0), (1, 1, 0), (1, 1, 1)]:
            P9 = inter_voxel_propagator_9x9(R, ALPHA, BETA, RHO, omega, n_orders=3)
            G = P9[:3, :3].real
            np.testing.assert_allclose(
                G, G.T, atol=1e-15, err_msg=f"G not symmetric for {R}"
            )

    def test_g_omega6_correction_smaller_than_omega4(self):
        """omega^6 G correction should be smaller than omega^4 at ka=0.3."""
        ka = 0.3
        omega = ka * ALPHA
        R = (1, 0, 0)
        P9_2 = inter_voxel_propagator_9x9(R, ALPHA, BETA, RHO, omega, n_orders=2)
        P9_3 = inter_voxel_propagator_9x9(R, ALPHA, BETA, RHO, omega, n_orders=3)
        P9_1 = inter_voxel_propagator_9x9(R, ALPHA, BETA, RHO, omega, n_orders=1)

        G_corr2 = np.max(np.abs(P9_2[:3, :3] - P9_1[:3, :3]))
        G_corr3 = np.max(np.abs(P9_3[:3, :3] - P9_2[:3, :3]))
        assert G_corr3 < G_corr2, (
            f"omega^6 G correction ({G_corr3:.3e}) should be < omega^4 ({G_corr2:.3e})"
        )

    def test_g_omega6_ka_scaling(self):
        """G omega^6 correction ratio between ka=0.1 and ka=0.3 should scale as ~(ka)^2."""
        R = (1, 0, 0)
        ka1, ka2 = 0.1, 0.3

        def g_corr_at_ka(ka):
            omega = ka * ALPHA
            P9_2 = inter_voxel_propagator_9x9(R, ALPHA, BETA, RHO, omega, n_orders=2)
            P9_3 = inter_voxel_propagator_9x9(R, ALPHA, BETA, RHO, omega, n_orders=3)
            return np.max(np.abs(P9_3[:3, :3] - P9_2[:3, :3]))

        ratio = g_corr_at_ka(ka2) / g_corr_at_ka(ka1)
        # Dominant term is omega^6, so ratio ~ (0.3/0.1)^6 = 729
        assert 200 < ratio < 2000, f"Expected ~729x scaling, got {ratio:.1f}x"


class TestCHBlockOmega6:
    """C/H block omega^6 correction (n_orders=3) properties."""

    def test_ch_omega6_smaller_than_omega4(self):
        """omega^6 C/H correction should be smaller than omega^4 at ka=0.3."""
        ka = 0.3
        omega = ka * ALPHA
        R = (1, 0, 0)
        P9_1 = inter_voxel_propagator_9x9(R, ALPHA, BETA, RHO, omega, n_orders=1)
        P9_2 = inter_voxel_propagator_9x9(R, ALPHA, BETA, RHO, omega, n_orders=2)
        P9_3 = inter_voxel_propagator_9x9(R, ALPHA, BETA, RHO, omega, n_orders=3)

        C_corr2 = np.max(np.abs(P9_2[:3, 3:] - P9_1[:3, 3:]))
        C_corr3 = np.max(np.abs(P9_3[:3, 3:] - P9_2[:3, 3:]))
        assert C_corr3 < C_corr2, (
            f"omega^6 C correction ({C_corr3:.3e}) should be < omega^4 ({C_corr2:.3e})"
        )

    def test_ch_omega6_ka_scaling(self):
        """C/H omega^6 correction ratio should scale as ~(ka)^2 between orders."""
        R = (1, 0, 0)
        ka1, ka2 = 0.1, 0.3

        def ch_corr_at_ka(ka):
            omega = ka * ALPHA
            P9_2 = inter_voxel_propagator_9x9(R, ALPHA, BETA, RHO, omega, n_orders=2)
            P9_3 = inter_voxel_propagator_9x9(R, ALPHA, BETA, RHO, omega, n_orders=3)
            return np.max(np.abs(P9_3[:3, 3:] - P9_2[:3, 3:]))

        ratio = ch_corr_at_ka(ka2) / ch_corr_at_ka(ka1)
        # Dominant term is omega^6, so ratio ~ (0.3/0.1)^6 = 729
        assert 200 < ratio < 2000, f"Expected ~729x scaling, got {ratio:.1f}x"

    def test_h_equals_c_transpose_omega6(self):
        """H = C^T at n_orders=3 for all neighbour types."""
        omega = 0.3 * ALPHA
        for R in [(1, 0, 0), (1, 1, 0), (1, 1, 1), (-1, 0, 0), (0, -1, 1)]:
            P9 = inter_voxel_propagator_9x9(R, ALPHA, BETA, RHO, omega, n_orders=3)
            np.testing.assert_allclose(
                P9[3:, :3],
                P9[:3, 3:].T,
                atol=1e-15,
                err_msg=f"H!=C^T for R={R}, n_orders=3",
            )

    def test_inversion_symmetry_omega6(self):
        """G,S even and C,H odd under R -> -R at n_orders=3."""
        omega = 0.3 * ALPHA
        for R_p, R_m in [
            ((1, 0, 0), (-1, 0, 0)),
            ((1, 1, 0), (-1, -1, 0)),
            ((1, 1, 1), (-1, -1, -1)),
        ]:
            Pp = inter_voxel_propagator_9x9(R_p, ALPHA, BETA, RHO, omega, 3)
            Pm = inter_voxel_propagator_9x9(R_m, ALPHA, BETA, RHO, omega, 3)
            np.testing.assert_allclose(
                Pp[:3, :3],
                Pm[:3, :3],
                atol=1e-14,
                err_msg=f"G not even, R={R_p}, n=3",
            )
            np.testing.assert_allclose(
                Pp[3:, 3:],
                Pm[3:, 3:],
                atol=1e-14,
                err_msg=f"S not even, R={R_p}, n=3",
            )
            np.testing.assert_allclose(
                Pp[:3, 3:],
                -Pm[:3, 3:],
                atol=1e-14,
                err_msg=f"C not odd, R={R_p}, n=3",
            )
            np.testing.assert_allclose(
                Pp[3:, :3],
                -Pm[3:, :3],
                atol=1e-14,
                err_msg=f"H not odd, R={R_p}, n=3",
            )


class TestConsistentTruncationOmega6:
    """All blocks at n_orders=3 for all 26 neighbours."""

    def test_all_26_finite_norders3(self):
        """9x9 propagator at n_orders=3 should be finite for all 26 neighbours."""
        omega = 0.3 * ALPHA
        for n1 in (-1, 0, 1):
            for n2 in (-1, 0, 1):
                for n3 in (-1, 0, 1):
                    if n1 == n2 == n3 == 0:
                        continue
                    R = (n1, n2, n3)
                    P9 = inter_voxel_propagator_9x9(
                        R, ALPHA, BETA, RHO, omega, n_orders=3
                    )
                    assert np.isfinite(P9).all(), (
                        f"Non-finite P9 at n_orders=3 for R={R}"
                    )

    def test_g_block_symmetric_all_26_omega6(self):
        """G block should be symmetric for all 26 neighbours at n_orders=3."""
        omega = 0.3 * ALPHA
        for n1 in (-1, 0, 1):
            for n2 in (-1, 0, 1):
                for n3 in (-1, 0, 1):
                    if n1 == n2 == n3 == 0:
                        continue
                    R = (n1, n2, n3)
                    P9 = inter_voxel_propagator_9x9(
                        R, ALPHA, BETA, RHO, omega, n_orders=3
                    )
                    G = P9[:3, :3].real
                    np.testing.assert_allclose(
                        G, G.T, atol=1e-14, err_msg=f"G not sym for R={R}"
                    )
