"""Inter-voxel propagator for cubic lattice nearest neighbours.

Provides:
1. **Strain propagator** P_{ijkl}(R) — the (3,3,3,3) tensor coupling
   volume-averaged strain to stress source (existing).
2. **9×9 block propagator** coupling displacement (3) and Voigt strain (6)
   DOF between adjacent cubes (Phase 3B).

The strain propagator is assembled from Newton (A_{jl}) and biharmonic
(B_{ijkl}) potential derivatives:

    P_{ijkl} = -1/(2μ) [δ_{ik} A_{jl} + δ_{jk} A_{il} - 2η B_{ijkl}]

The 9×9 propagator has block structure [[G, C], [H, S]] where:
- S (6×6): Voigt contraction of P_{ijkl}
- G (3×3): volume-averaged Green's tensor <G_{ij}>
- C (3×6), H (6×3): displacement-strain coupling from dG/dR

All values computed analytically in Mathematica via delta-function collapse
and validated against finite-difference cross-checks (10⁻⁸ or better).

Reference scripts:
    Mathematica/InterVoxelPropagator.wl       (face, Phase 1A)
    Mathematica/InterVoxelPropagatorEdge.wl    (edge, Phase 1B)
    Mathematica/InterVoxelPropagatorCorner.wl  (corner, Phase 1C)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

# ──────────────────────────────────────────────────────────────────────
# Newton potential derivatives A_{jl} = -1/(4π) ∂²Φ/∂R_j∂R_l
# Biharmonic derivatives B_{ijkl} = -1/(8π) ∂⁴Ψ/���R_i��R_j∂R_k∂R_l
#
# All for unit cube a=1 with μ=1, ν=1/4 normalization stripped out.
# Indexed as Voigt-like but stored per independent component.
# ──��────────────────────────────────��──────────────────────────────────

# === FACE-ADJACENT  R = (a, 0, 0)  C₄ᵥ symmetry ===
# A: A₁₁, A₂₂=A₃₃.  Off-diag A₁₂=A₁₃=A₂₃=0.
FACE_A11 = -0.13501718054449527
FACE_A22 = +0.06750859027224763  # = A₃₃
# B: B₁₁₁₁, B₁₁₂₂=B₁₁₃₃, B₂₂₂��=B₃₃₃₃, B₂₂₃₃
FACE_B1111 = -0.10518162135737388
FACE_B1122 = -0.01491777959356069  # = B₁₁₃₃
FACE_B2222 = +0.06226739135121906  # = B��₃₃₃
FACE_B2233 = +0.02015897851458926

# === EDGE-ADJACENT  R = (a, a, 0)  C₂ᵥ symmetry ===
# A: A₁₁=A₂₂, A₃₃, A₁₂. A₁₃=A₂₃=0.
EDGE_A11 = -0.01378576204834812  # = A₂₂
EDGE_A33 = +0.02757152409669624
EDGE_A12 = -0.04556482263891465
# B: B₁₁₁₁=B₂₂₂₂, B₁₁₂₂, B₁₁₃₃=B₂₂₃₃, B₃₃₃₃, B₁₁₁₂=B₁₂₂₂, B₁₂₃₃
EDGE_B1111 = -0.02188873710833393  # = B₂₂₂₂
EDGE_B1122 = +0.00965705243872159
EDGE_B1133 = -0.00155407737873578  # = B₂₂₃₃
EDGE_B3333 = +0.03067967885416780
EDGE_B1112 = -0.01652993415886442  # = B₁₂₂₂
EDGE_B1233 = -0.01250495432118580

# === CORNER-ADJACENT  R = (a, a, a)  S₃ symmetry ===
# A: A₁₁=A₂₂=A₃₃=0 (exact), A₁₂=A₁₃=A₂₃.
CORNER_A11 = 0.0  # exact by B₁₁₁₁ = -2B₁₁₂₂
CORNER_A12 = -0.01606212781050823
# B: B₁₁₁₁=B₂₂₂₂=B₃₃₃₃, B₁₁₂₂=B��₁₃₃=B₂₂₃���, B₁₁₁₂ (6 equiv), B₁₁₂₃
CORNER_B1111 = -0.00625536598256419
CORNER_B1122 = +0.00312768299128210  # = -B₁₁₁₁/2
CORNER_B1112 = -0.00926615309635844  # 6 equivalent by S₃
CORNER_B1123 = +0.00247017838220864

# ──────────────────────────────────────────────────────────────────────
# BIHARMONIC THIRD DERIVATIVES d³Ψ/dR_i dR_j dR_k
# For C/H displacement-strain coupling blocks (Phase 3B-2).
# Computed via delta-function collapse in InterVoxelPropagatorFirstDeriv.wl.
# Laplacian identity Σ_j D_{jjk} = 2 dΦ_k verified to 18+ digits.
# ──────────────────────────────────────────────────────────────────────

# === FACE  R = (a,0,0)  C₄ᵥ: 2 independent values ===
FACE_D3PSI_000 = -0.79543834473319394  # d³Ψ/dR₀³
FACE_D3PSI_011 = -0.52826208819206832  # d³Ψ/(dR₀ dR₁²) = d³Ψ/(dR₀ dR₂²)

# === EDGE  R = (a,a,0)  C₂ᵥ: 3 independent values ===
EDGE_D3PSI_000 = -0.50186397481765351  # = D₁₁₁ by C₂ᵥ
EDGE_D3PSI_001 = +0.04882866652718464  # = D₀₁₁ by C₂ᵥ
EDGE_D3PSI_022 = -0.26277092923552829  # = D₁₂₂ by C₂ᵥ

# === CORNER  R = (a,a,a)  S₃: 3 independent values ===
CORNER_D3PSI_000 = -0.34647399020181943  # = D₁₁₁ = D₂₂₂ by S₃
CORNER_D3PSI_001 = -0.02187587552538497  # all mixed-pair by S₃
CORNER_D3PSI_012 = +0.14258930467014646  # all-different

# dΦ/dR_k (Newton first derivatives, from Laplacian identity dΦ_k = ½ Σ_j D_{jjk})
FACE_DPHI_0 = -0.92598126055866529  # dΦ_1 = dΦ_2 = 0 by mirror symmetry
EDGE_DPHI_0 = -0.35790311876299858  # = dΦ_1 by C₂ᵥ; dΦ_2 = 0
CORNER_DPHI_0 = -0.19511287062629469  # = dΦ_1 = dΦ_2 by S₃

# ──────────────────────────────────────────────────────────────────────
# TRIHARMONIC THIRD DERIVATIVES d³X/dR_i dR_j dR_k  (kernel ρ³)
# For ω² correction to C/H blocks (Phase 2E).
# Computed via delta-prime/delta/step collapse in
# InterVoxelPropagatorCHDynamic.wl. Validated by direct NIntegrate.
# ──────────────────────────────────────────────────────────────────────

# === FACE  R = (a,0,0) ===
FACE_D3X_000 = 5.59066550438255537  # d³X/dR₀³
FACE_D3X_011 = 2.22178066763092212  # d³X/(dR₀ dR₁²) = d³X/(dR₀ dR₂²)

# === EDGE  R = (a,a,0) ===
EDGE_D3X_000 = 4.79793940592048508  # = D₁₁₁ by C₂ᵥ
EDGE_D3X_001 = 1.17783327456444731  # = D₀₁₁ by C₂ᵥ
EDGE_D3X_022 = 1.80511344407289295  # = D₁₂₂ by C₂ᵥ

# === CORNER  R = (a,a,a) ===
CORNER_D3X_000 = 4.25436485917712435  # = D₁₁₁ = D₂₂₂ by S₃
CORNER_D3X_001 = 1.14631500401061554  # all mixed-pair by S₃
CORNER_D3X_012 = -0.41087473026598235  # all-different

# dΨ/dR_k (biharmonic first derivative, from Laplacian: dΨ_k = (1/12) Σ_j d³X_{jjk})
FACE_DPSI_0 = 0.83618556997036663  # dΨ_1 = dΨ_2 = 0 by mirror symmetry
EDGE_DPSI_0 = 0.64840717704648545  # = dΨ_1 by C₂ᵥ; dΨ_2 = 0
CORNER_DPSI_0 = 0.54558290559986295  # = dΨ_1 = dΨ_2 by S₃

# ──────────────────────────────────────────────────────────────────────
# PENTAHARMONIC THIRD DERIVATIVES d³Ω/dR_i dR_j dR_k  (kernel ρ⁵)
# For ω⁴ correction to C/H blocks (Phase 2E).
# ──────────────────────────────────────────────────────────────────────

# === FACE  R = (a,0,0) ===
FACE_D3OM_000 = 74.7028469944845582  # d³Ω/dR₀³
FACE_D3OM_011 = 21.5319534595037667  # d³Ω/(dR₀ dR₁²) = d³Ω/(dR₀ dR₂²)

# === EDGE  R = (a,a,0) ===
EDGE_D3OM_000 = 86.9352238951433639  # = D₁₁₁ by C₂ᵥ
EDGE_D3OM_001 = 34.6277658601723239  # = D₀₁₁ by C₂ᵥ
EDGE_D3OM_022 = 26.1528244054226945  # = D₁₂₂ by C₂ᵥ

# === CORNER  R = (a,a,a) ===
CORNER_D3OM_000 = 97.7452814064066722  # = D₁₁₁ = D₂₂₂ by S₃
CORNER_D3OM_001 = 37.5180442143966006  # all mixed-pair by S₃
CORNER_D3OM_012 = 7.42106273126978393  # all-different

# dX/dR_k (triharmonic first derivative, from Laplacian: dX_k = (1/30) Σ_j d³Ω_{jjk})
FACE_DX_0 = 3.92555846378306972  # dX_1 = dX_2 = 0 by mirror symmetry
EDGE_DX_0 = 4.92386047202461274  # = dX_1 by C₂ᵥ; dX_2 = 0
CORNER_DX_0 = 5.75937899450666245  # = dX_1 = dX_2 by S₃

# ──────────────────────────────────────────────────────────────────────
# HEPTAHARMONIC THIRD DERIVATIVES d³H/dR_i dR_j dR_k  (kernel ρ⁷)
# For ω⁶ correction to C/H blocks (Phase 5).
# Computed via delta-prime/delta/step collapse in
# InterVoxelPropagatorOmega6.wl. Validated by Laplacian identity.
# ──────────────────────────────────────────────────────────────────────

# === FACE  R = (a,0,0) ===
FACE_D3H_000 = 520.366848816268754  # d³H/dR₀³
FACE_D3H_011 = 119.391877525300900  # d³H/(dR₀ dR₁²) = d³H/(dR₀ dR₂²)

# === EDGE  R = (a,a,0) ===
EDGE_D3H_000 = 818.181285566568028  # = D₁₁₁ by C₂ᵥ
EDGE_D3H_001 = 401.751494351498964  # = D₀₁₁ by C₂ᵥ
EDGE_D3H_022 = 208.558979721460787  # = D₁₂₂ by C₂ᵥ

# === CORNER  R = (a,a,a) ===
CORNER_D3H_000 = 1154.46537928822656  # = D₁₁₁ = D₂₂₂ by S₃
CORNER_D3H_001 = 531.429829090769075  # all mixed-pair by S₃
CORNER_D3H_012 = 219.478429989378284  # all-different

# dΩ/dR_k (pentaharmonic first derivative, from Laplacian: dΩ_k = (1/56) Σ_j d³H_{jjk})
FACE_DOM_0 = 13.5562607833369742  # dΩ_1 = dΩ_2 = 0 by mirror symmetry
EDGE_DOM_0 = 25.5087814221344246  # = dΩ_1 by C₂ᵥ; dΩ_2 = 0
CORNER_DOM_0 = 39.5950899548172270  # = dΩ_1 = dΩ_2 by S₃

# ──────────────────────────────────────────────────────────────────────
# DYNAMIC CORRECTIONS: P(ω) = P⁽⁰⁾ + ω²P⁽¹⁾ + ω⁴P⁽²⁾
#
# From the Fourier expansion of the elastodynamic Green's tensor
# (docs/inter_voxel_propagator_plan.tex §3), each order has the SAME
# tensorial structure as the static propagator:
#
#   P⁽ⁿ⁾ = -1/(2ρcₛ²ⁿ⁺²) [δ_{ik}A⁽ⁿ⁾_{jl} + δ_{jk}A⁽ⁿ⁾_{il} - 2ηₙ B⁽ⁿ⁾]
#
# with ηₙ = 1 - (cₛ/cₚ)^{2n+2}.
#
# Normalised A⁽ⁿ⁾, B⁽ⁿ⁾ from the Fourier potential hierarchy:
#   n=1: A⁽¹⁾ = ∂²Ψ/(8π ∂R²),    B⁽¹⁾ = ∂⁴X/(96π ∂R⁴)
#   n=2: A⁽²⁾ = -∂²X/(96π ∂R²),   B⁽²⁾ = -∂⁴Ω/(2880π ∂R⁴)
#
# Raw derivatives from Mathematica/InterVoxelPropagatorDynamic.wl.
# Laplacian identity Σ_k B⁽ⁿ⁾_{ijkk} = A⁽ⁿ⁾_{ij} verified to 13+ digits.
# ──────────────────────────────────────────────────────────────────────

_NORM_A1 = 1.0 / (8.0 * np.pi)  # 1/(8π) for order 1 A
_NORM_B1 = 1.0 / (96.0 * np.pi)  # 1/(96π) for order 1 B
_NORM_A2 = -1.0 / (96.0 * np.pi)  # -1/(96π) for order 2 A
_NORM_B2 = -1.0 / (2880.0 * np.pi)  # -1/(2880π) for order 2 B
_NORM_A3 = 1.0 / (2880.0 * np.pi)  # 1/(2880π) for order 3 A
_NORM_B3 = 1.0 / (161280.0 * np.pi)  # 1/(161280π) for order 3 B  [= NORM_A3/56]
_NORM_A4 = -1.0 / (
    161280.0 * np.pi
)  # -1/(161280π) for order 4 A (H kernel) [= -NORM_B3]

# === FACE DYN ORDER 1: Ψ(ρ)→A, X(ρ³)→B  ===
DYN1_FACE_A11 = 0.30434003593387251229 * _NORM_A1
DYN1_FACE_A22 = 0.82871516563404197555 * _NORM_A1  # = A₃₃
DYN1_FACE_B1111 = 1.41327701454403944664 * _NORM_B1
DYN1_FACE_B1122 = 1.11940170833121535043 * _NORM_B1  # = B₁₁₃₃
DYN1_FACE_B2222 = 6.61390374095420502775 * _NORM_B1  # = B₃₃₃₃
DYN1_FACE_B2233 = 2.21127653832308332846 * _NORM_B1

# === FACE DYN ORDER 2: X(ρ³)→A, Ω(ρ⁵)→B ===
DYN2_FACE_A11 = 6.14687033491410334021 * _NORM_A2
DYN2_FACE_A22 = 3.93095155409960859065 * _NORM_A2  # = A₃₃
DYN2_FACE_B1111 = 121.10629615723429259 * _NORM_B2
DYN2_FACE_B1122 = 31.64990694509440380732 * _NORM_B2  # = B₁₁₃₃
DYN2_FACE_B2222 = 64.72963275084629337 * _NORM_B2  # = B₃₃₃₃
DYN2_FACE_B2233 = 21.54900692704756054521 * _NORM_B2

# === EDGE DYN ORDER 1 ===
DYN1_EDGE_A11 = 0.38529898485200793724 * _NORM_A1  # = A₂₂
DYN1_EDGE_A33 = 0.64639228402098784773 * _NORM_A1
DYN1_EDGE_A12 = -0.26788042772784072223 * _NORM_A1
DYN1_EDGE_B1111 = 2.27247134973510232851 * _NORM_B1  # = B₂₂₂₂
DYN1_EDGE_B1122 = 1.16949686073991690189 * _NORM_B1
DYN1_EDGE_B1133 = 1.18161960774907601651 * _NORM_B1  # = B₂₂₃₃
DYN1_EDGE_B3333 = 5.39346819275370213980 * _NORM_B1
DYN1_EDGE_B1112 = -1.29181426541309712008 * _NORM_B1  # = B₁₂₂₂
DYN1_EDGE_B1233 = -0.63093660190789442655 * _NORM_B1

# === EDGE DYN ORDER 2 ===
DYN2_EDGE_A11 = 6.72804661239907577216 * _NORM_A2  # = A₂₂
DYN2_EDGE_A33 = 4.92607224313657099253 * _NORM_A2
DYN2_EDGE_A12 = 1.80873224417671787438 * _NORM_A2
DYN2_EDGE_B1111 = 126.69450210248536477560 * _NORM_B2  # = B₂₂₂₂
DYN2_EDGE_B1122 = 40.52235834689523028490 * _NORM_B2
DYN2_EDGE_B1133 = 34.62453792259167810431 * _NORM_B2  # = B₂₂₃₃
DYN2_EDGE_B3333 = 78.53309144891377356740 * _NORM_B2
DYN2_EDGE_B1112 = 22.88723612738029927492 * _NORM_B2  # = B₁₂₂₂
DYN2_EDGE_B1233 = 8.48749507054093768171 * _NORM_B2

# === CORNER DYN ORDER 1 ===
DYN1_CORNER_A11 = 0.38586466785236013460 * _NORM_A1  # = A₂₂ = A₃₃
DYN1_CORNER_A12 = -0.16121187591918290103 * _NORM_A1  # = A₁₃ = A₂₃
DYN1_CORNER_B1111 = 2.53304140303690244565 * _NORM_B1  # = B₂₂₂₂ = B₃₃₃₃
DYN1_CORNER_B1122 = 1.04866730559570958475 * _NORM_B1  # = B₁₁₃₃ = B₂₂₃₃
DYN1_CORNER_B1112 = -0.91956804085889468740 * _NORM_B1  # 6 equiv by S₃
DYN1_CORNER_B1123 = -0.09540642931240543758 * _NORM_B1

# === CORNER DYN ORDER 2 ===
DYN2_CORNER_A11 = 7.31149202498253105099 * _NORM_A2  # = A₂₂ = A₃₃
DYN2_CORNER_A12 = 1.55428124313434973873 * _NORM_A2  # = A₁₃ = A₂₃
DYN2_CORNER_B1111 = 133.00383894991481736747 * _NORM_B2  # = B₂₂₂₂ = B₃₃₃₃
DYN2_CORNER_B1122 = 43.17046089978055708110 * _NORM_B2  # = B₁₁₃₃ = B₂₂₃₃
DYN2_CORNER_B1112 = 20.48871726047062015519 * _NORM_B2  # 6 equiv by S₃
DYN2_CORNER_B1123 = 5.65100277308925185145 * _NORM_B2

# === DYN ORDER 3: Ω(ρ⁵)→A, H(ρ⁷)→B  ===
# From Mathematica/InterVoxelPropagatorOmegaHessian.wl
# d²Ω/dR² → A⁽³⁾,  d⁴H/dR⁴ → B⁽³⁾
# Laplacian identity verified: Σ_k B_{jjkk} = 56·A_{jj} (10⁻¹² precision)

# === FACE DYN ORDER 3 ===
DYN3_FACE_A11 = 35.09972390156100239 * _NORM_A3
DYN3_FACE_A22 = 13.54175081054303704 * _NORM_A3  # = A₃₃
DYN3_FACE_B1111 = 1401.02708938036651 * _NORM_B3
DYN3_FACE_B1122 = 282.27872455352481 * _NORM_B3  # = B₁₁₃₃
DYN3_FACE_B2222 = 356.74707004533833 * _NORM_B3  # = B₃₃₃₃
DYN3_FACE_B2233 = 119.31225079154694 * _NORM_B3

# === EDGE DYN ORDER 3 ===
DYN3_EDGE_A11 = 51.67064499234048322 * _NORM_A3  # = A₂₂
DYN3_EDGE_A33 = 25.50019876734907121 * _NORM_A3
DYN3_EDGE_A12 = 26.14480437842729518 * _NORM_A3
DYN3_EDGE_B1111 = 1840.74246384076366 * _NORM_B3  # = B₂₂₂₂
DYN3_EDGE_B1122 = 651.07099896048098 * _NORM_B3
DYN3_EDGE_B1133 = 401.74265676982242 * _NORM_B3  # = B₂₂₃₃
DYN3_EDGE_B3333 = 624.52581743190315 * _NORM_B3
DYN3_EDGE_B1112 = 635.51239625893890 * _NORM_B3  # = B₁₂₂₂
DYN3_EDGE_B1233 = 193.08425267405073 * _NORM_B3

# === CORNER DYN ORDER 3 ===
DYN3_CORNER_A11 = 69.71171346052524425 * _NORM_A3  # = A₂₂ = A₃₃
DYN3_CORNER_A12 = 30.10535779745780000 * _NORM_A3  # = A₁₃ = A₂₃
DYN3_CORNER_B1111 = 2302.44036080125068 * _NORM_B3  # = B₂₂₂₂ = B₃₃₃₃
DYN3_CORNER_B1122 = 800.70779649408150 * _NORM_B3  # = B₁₁₃₃ = B₂₂₃₃
DYN3_CORNER_B1112 = 708.32238983951127 * _NORM_B3  # 6 equiv by S₃
DYN3_CORNER_B1123 = 269.25525697861426 * _NORM_B3

# === DYN ORDER 4: H(ρ⁷)→A  (A-channel only, for G block ω⁶) ===
# From Mathematica/InterVoxelPropagatorOmega6.wl
# d²H/dR² → A⁽⁴⁾  (no B-tensor needed — S block already has ω⁶)
# Laplacian trace validated: Tr(d²H) = 56×Ω₀₀ to 10⁻⁹ precision

# === FACE DYN ORDER 4 ===
DYN4_FACE_A11 = 166.040235936049047 * _NORM_A4
DYN4_FACE_A22 = 47.1604212749853438 * _NORM_A4  # = A₃₃

# === EDGE DYN ORDER 4 ===
DYN4_EDGE_A11 = 336.612687195747214 * _NORM_A4  # = A₂₂
DYN4_EDGE_A33 = 128.501096007485975 * _NORM_A4
DYN4_EDGE_A12 = 208.609329750912862 * _NORM_A4

# === CORNER DYN ORDER 4 ===
DYN4_CORNER_A11 = 569.489574041776214 * _NORM_A4  # = A₂₂ = A₃₃
DYN4_CORNER_A12 = 311.882584477024042 * _NORM_A4  # = A₁₃ = A₂₃


def _build_A_matrix(
    a_diag: tuple[float, float, float], a_offdiag: tuple[float, float, float]
) -> NDArray:
    """Build 3x3 symmetric A_{jl} matrix."""
    A = np.zeros((3, 3))
    A[0, 0], A[1, 1], A[2, 2] = a_diag
    A[0, 1] = A[1, 0] = a_offdiag[0]  # A₁₂
    A[0, 2] = A[2, 0] = a_offdiag[1]  # A₁₃
    A[1, 2] = A[2, 1] = a_offdiag[2]  # A₂₃
    return A


def _build_B_tensor(b_dict: dict[tuple[int, int, int, int], float]) -> NDArray:
    """Build 3x3x3x3 B_{ijkl} tensor from independent components.

    B is the fourth derivative of a scalar potential, so it has full S₄
    permutation symmetry: B_{ijkl} = B_{σ(ijkl)} for any permutation σ.
    """
    from itertools import permutations

    B = np.zeros((3, 3, 3, 3))
    for (i, j, k, l), val in b_dict.items():
        for perm in set(permutations((i, j, k, l))):
            B[perm] = val
    return B


def _assemble_P(A: NDArray, B: NDArray, mu: float, nu: float) -> NDArray:
    """Assemble propagator P_{ijkl} = -1/(2μ)[δ_{ik}A_{jl} + δ_{jk}A_{il} - 2η B_{ijkl}]."""
    eta = 1.0 / (2.0 * (1.0 - nu))
    delta = np.eye(3)
    P = np.zeros((3, 3, 3, 3))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for ll in range(3):
                    P[i, j, k, ll] = (
                        -1.0
                        / (2.0 * mu)
                        * (
                            delta[i, k] * A[j, ll]
                            + delta[j, k] * A[i, ll]
                            - 2.0 * eta * B[i, j, k, ll]
                        )
                    )
    return P


def face_propagator(mu: float, nu: float) -> NDArray:
    """Static propagator for face-adjacent cubes R=(a,0,0), C₄ᵥ symmetry.

    Returns:
        P: shape (3,3,3,3) tensor P_{ijkl} for R along axis 0.
    """
    A = _build_A_matrix(
        a_diag=(FACE_A11, FACE_A22, FACE_A22),
        a_offdiag=(0.0, 0.0, 0.0),
    )
    b_components = {
        (0, 0, 0, 0): FACE_B1111,
        (0, 0, 1, 1): FACE_B1122,
        (0, 0, 2, 2): FACE_B1122,
        (1, 1, 1, 1): FACE_B2222,
        (2, 2, 2, 2): FACE_B2222,
        (1, 1, 2, 2): FACE_B2233,
    }
    B = _build_B_tensor(b_components)
    return _assemble_P(A, B, mu, nu)


def edge_propagator(mu: float, nu: float) -> NDArray:
    """Static propagator for edge-adjacent cubes R=(a,a,0), C₂ᵥ symmetry.

    Returns:
        P: shape (3,3,3,3) tensor P_{ijkl} for R along (1,1,0)/√2.
    """
    A = _build_A_matrix(
        a_diag=(EDGE_A11, EDGE_A11, EDGE_A33),
        a_offdiag=(EDGE_A12, 0.0, 0.0),
    )
    b_components = {
        (0, 0, 0, 0): EDGE_B1111,
        (1, 1, 1, 1): EDGE_B1111,
        (0, 0, 1, 1): EDGE_B1122,
        (0, 0, 2, 2): EDGE_B1133,
        (1, 1, 2, 2): EDGE_B1133,
        (2, 2, 2, 2): EDGE_B3333,
        (0, 0, 0, 1): EDGE_B1112,
        (0, 1, 1, 1): EDGE_B1112,
        (0, 1, 2, 2): EDGE_B1233,
    }
    B = _build_B_tensor(b_components)
    return _assemble_P(A, B, mu, nu)


def corner_propagator(mu: float, nu: float) -> NDArray:
    """Static propagator for corner-adjacent cubes R=(a,a,a), S₃ symmetry.

    Returns:
        P: shape (3,3,3,3) tensor P_{ijkl} for R along (1,1,1)/√3.
    """
    A = _build_A_matrix(
        a_diag=(CORNER_A11, CORNER_A11, CORNER_A11),
        a_offdiag=(CORNER_A12, CORNER_A12, CORNER_A12),
    )
    b_components = {
        (0, 0, 0, 0): CORNER_B1111,
        (1, 1, 1, 1): CORNER_B1111,
        (2, 2, 2, 2): CORNER_B1111,
        (0, 0, 1, 1): CORNER_B1122,
        (0, 0, 2, 2): CORNER_B1122,
        (1, 1, 2, 2): CORNER_B1122,
        (0, 0, 0, 1): CORNER_B1112,
        (0, 0, 0, 2): CORNER_B1112,
        (0, 1, 1, 1): CORNER_B1112,
        (1, 1, 1, 2): CORNER_B1112,
        (0, 2, 2, 2): CORNER_B1112,
        (1, 2, 2, 2): CORNER_B1112,
        (0, 0, 1, 2): CORNER_B1123,
        (0, 1, 0, 2): CORNER_B1123,
        (0, 1, 2, 2): CORNER_B1123,
    }
    B = _build_B_tensor(b_components)
    return _assemble_P(A, B, mu, nu)


# ──────────────────────────────────────────────────────────────────────
# Dynamic propagator builders (canonical directions only)
# ──────────────────────────────────────────────────────────────────────


def _face_propagator_dyn(order: int, rho: float, alpha: float, beta: float) -> NDArray:
    """Dynamic correction P⁽ⁿ⁾ for face-adjacent cubes R=(a,0,0)."""
    cs, cp = beta, alpha
    mu_eff = rho * cs ** (2 * order + 2)
    eta_n = 1.0 - (cs / cp) ** (2 * order + 2)
    nu_eff = 1.0 - 1.0 / (2.0 * eta_n)

    if order == 1:
        A = _build_A_matrix(
            a_diag=(DYN1_FACE_A11, DYN1_FACE_A22, DYN1_FACE_A22),
            a_offdiag=(0.0, 0.0, 0.0),
        )
        b_dict = {
            (0, 0, 0, 0): DYN1_FACE_B1111,
            (0, 0, 1, 1): DYN1_FACE_B1122,
            (0, 0, 2, 2): DYN1_FACE_B1122,
            (1, 1, 1, 1): DYN1_FACE_B2222,
            (2, 2, 2, 2): DYN1_FACE_B2222,
            (1, 1, 2, 2): DYN1_FACE_B2233,
        }
    elif order == 2:
        A = _build_A_matrix(
            a_diag=(DYN2_FACE_A11, DYN2_FACE_A22, DYN2_FACE_A22),
            a_offdiag=(0.0, 0.0, 0.0),
        )
        b_dict = {
            (0, 0, 0, 0): DYN2_FACE_B1111,
            (0, 0, 1, 1): DYN2_FACE_B1122,
            (0, 0, 2, 2): DYN2_FACE_B1122,
            (1, 1, 1, 1): DYN2_FACE_B2222,
            (2, 2, 2, 2): DYN2_FACE_B2222,
            (1, 1, 2, 2): DYN2_FACE_B2233,
        }
    elif order == 3:
        A = _build_A_matrix(
            a_diag=(DYN3_FACE_A11, DYN3_FACE_A22, DYN3_FACE_A22),
            a_offdiag=(0.0, 0.0, 0.0),
        )
        b_dict = {
            (0, 0, 0, 0): DYN3_FACE_B1111,
            (0, 0, 1, 1): DYN3_FACE_B1122,
            (0, 0, 2, 2): DYN3_FACE_B1122,
            (1, 1, 1, 1): DYN3_FACE_B2222,
            (2, 2, 2, 2): DYN3_FACE_B2222,
            (1, 1, 2, 2): DYN3_FACE_B2233,
        }
    else:
        msg = f"Dynamic order {order} not implemented (only 1, 2, and 3)"
        raise ValueError(msg)
    return _assemble_P(A, _build_B_tensor(b_dict), mu_eff, nu_eff)


def _edge_propagator_dyn(order: int, rho: float, alpha: float, beta: float) -> NDArray:
    """Dynamic correction P⁽ⁿ⁾ for edge-adjacent cubes R=(a,a,0)."""
    cs, cp = beta, alpha
    mu_eff = rho * cs ** (2 * order + 2)
    eta_n = 1.0 - (cs / cp) ** (2 * order + 2)
    nu_eff = 1.0 - 1.0 / (2.0 * eta_n)

    if order == 1:
        A = _build_A_matrix(
            a_diag=(DYN1_EDGE_A11, DYN1_EDGE_A11, DYN1_EDGE_A33),
            a_offdiag=(DYN1_EDGE_A12, 0.0, 0.0),
        )
        b_dict = {
            (0, 0, 0, 0): DYN1_EDGE_B1111,
            (1, 1, 1, 1): DYN1_EDGE_B1111,
            (0, 0, 1, 1): DYN1_EDGE_B1122,
            (0, 0, 2, 2): DYN1_EDGE_B1133,
            (1, 1, 2, 2): DYN1_EDGE_B1133,
            (2, 2, 2, 2): DYN1_EDGE_B3333,
            (0, 0, 0, 1): DYN1_EDGE_B1112,
            (0, 1, 1, 1): DYN1_EDGE_B1112,
            (0, 1, 2, 2): DYN1_EDGE_B1233,
        }
    elif order == 2:
        A = _build_A_matrix(
            a_diag=(DYN2_EDGE_A11, DYN2_EDGE_A11, DYN2_EDGE_A33),
            a_offdiag=(DYN2_EDGE_A12, 0.0, 0.0),
        )
        b_dict = {
            (0, 0, 0, 0): DYN2_EDGE_B1111,
            (1, 1, 1, 1): DYN2_EDGE_B1111,
            (0, 0, 1, 1): DYN2_EDGE_B1122,
            (0, 0, 2, 2): DYN2_EDGE_B1133,
            (1, 1, 2, 2): DYN2_EDGE_B1133,
            (2, 2, 2, 2): DYN2_EDGE_B3333,
            (0, 0, 0, 1): DYN2_EDGE_B1112,
            (0, 1, 1, 1): DYN2_EDGE_B1112,
            (0, 1, 2, 2): DYN2_EDGE_B1233,
        }
    elif order == 3:
        A = _build_A_matrix(
            a_diag=(DYN3_EDGE_A11, DYN3_EDGE_A11, DYN3_EDGE_A33),
            a_offdiag=(DYN3_EDGE_A12, 0.0, 0.0),
        )
        b_dict = {
            (0, 0, 0, 0): DYN3_EDGE_B1111,
            (1, 1, 1, 1): DYN3_EDGE_B1111,
            (0, 0, 1, 1): DYN3_EDGE_B1122,
            (0, 0, 2, 2): DYN3_EDGE_B1133,
            (1, 1, 2, 2): DYN3_EDGE_B1133,
            (2, 2, 2, 2): DYN3_EDGE_B3333,
            (0, 0, 0, 1): DYN3_EDGE_B1112,
            (0, 1, 1, 1): DYN3_EDGE_B1112,
            (0, 1, 2, 2): DYN3_EDGE_B1233,
        }
    else:
        msg = f"Dynamic order {order} not implemented (only 1, 2, and 3)"
        raise ValueError(msg)
    return _assemble_P(A, _build_B_tensor(b_dict), mu_eff, nu_eff)


def _corner_propagator_dyn(
    order: int, rho: float, alpha: float, beta: float
) -> NDArray:
    """Dynamic correction P⁽ⁿ⁾ for corner-adjacent cubes R=(a,a,a)."""
    cs, cp = beta, alpha
    mu_eff = rho * cs ** (2 * order + 2)
    eta_n = 1.0 - (cs / cp) ** (2 * order + 2)
    nu_eff = 1.0 - 1.0 / (2.0 * eta_n)

    if order == 1:
        A = _build_A_matrix(
            a_diag=(DYN1_CORNER_A11, DYN1_CORNER_A11, DYN1_CORNER_A11),
            a_offdiag=(DYN1_CORNER_A12, DYN1_CORNER_A12, DYN1_CORNER_A12),
        )
        b_dict = {
            (0, 0, 0, 0): DYN1_CORNER_B1111,
            (1, 1, 1, 1): DYN1_CORNER_B1111,
            (2, 2, 2, 2): DYN1_CORNER_B1111,
            (0, 0, 1, 1): DYN1_CORNER_B1122,
            (0, 0, 2, 2): DYN1_CORNER_B1122,
            (1, 1, 2, 2): DYN1_CORNER_B1122,
            (0, 0, 0, 1): DYN1_CORNER_B1112,
            (0, 0, 0, 2): DYN1_CORNER_B1112,
            (0, 1, 1, 1): DYN1_CORNER_B1112,
            (1, 1, 1, 2): DYN1_CORNER_B1112,
            (0, 2, 2, 2): DYN1_CORNER_B1112,
            (1, 2, 2, 2): DYN1_CORNER_B1112,
            (0, 0, 1, 2): DYN1_CORNER_B1123,
            (0, 1, 0, 2): DYN1_CORNER_B1123,
            (0, 1, 2, 2): DYN1_CORNER_B1123,
        }
    elif order == 2:
        A = _build_A_matrix(
            a_diag=(DYN2_CORNER_A11, DYN2_CORNER_A11, DYN2_CORNER_A11),
            a_offdiag=(DYN2_CORNER_A12, DYN2_CORNER_A12, DYN2_CORNER_A12),
        )
        b_dict = {
            (0, 0, 0, 0): DYN2_CORNER_B1111,
            (1, 1, 1, 1): DYN2_CORNER_B1111,
            (2, 2, 2, 2): DYN2_CORNER_B1111,
            (0, 0, 1, 1): DYN2_CORNER_B1122,
            (0, 0, 2, 2): DYN2_CORNER_B1122,
            (1, 1, 2, 2): DYN2_CORNER_B1122,
            (0, 0, 0, 1): DYN2_CORNER_B1112,
            (0, 0, 0, 2): DYN2_CORNER_B1112,
            (0, 1, 1, 1): DYN2_CORNER_B1112,
            (1, 1, 1, 2): DYN2_CORNER_B1112,
            (0, 2, 2, 2): DYN2_CORNER_B1112,
            (1, 2, 2, 2): DYN2_CORNER_B1112,
            (0, 0, 1, 2): DYN2_CORNER_B1123,
            (0, 1, 0, 2): DYN2_CORNER_B1123,
            (0, 1, 2, 2): DYN2_CORNER_B1123,
        }
    elif order == 3:
        A = _build_A_matrix(
            a_diag=(DYN3_CORNER_A11, DYN3_CORNER_A11, DYN3_CORNER_A11),
            a_offdiag=(DYN3_CORNER_A12, DYN3_CORNER_A12, DYN3_CORNER_A12),
        )
        b_dict = {
            (0, 0, 0, 0): DYN3_CORNER_B1111,
            (1, 1, 1, 1): DYN3_CORNER_B1111,
            (2, 2, 2, 2): DYN3_CORNER_B1111,
            (0, 0, 1, 1): DYN3_CORNER_B1122,
            (0, 0, 2, 2): DYN3_CORNER_B1122,
            (1, 1, 2, 2): DYN3_CORNER_B1122,
            (0, 0, 0, 1): DYN3_CORNER_B1112,
            (0, 0, 0, 2): DYN3_CORNER_B1112,
            (0, 1, 1, 1): DYN3_CORNER_B1112,
            (1, 1, 1, 2): DYN3_CORNER_B1112,
            (0, 2, 2, 2): DYN3_CORNER_B1112,
            (1, 2, 2, 2): DYN3_CORNER_B1112,
            (0, 0, 1, 2): DYN3_CORNER_B1123,
            (0, 1, 0, 2): DYN3_CORNER_B1123,
            (0, 1, 2, 2): DYN3_CORNER_B1123,
        }
    else:
        msg = f"Dynamic order {order} not implemented (only 1, 2, and 3)"
        raise ValueError(msg)
    return _assemble_P(A, _build_B_tensor(b_dict), mu_eff, nu_eff)


def _rotate_tensor4(P: NDArray, R: NDArray) -> NDArray:
    """Rotate rank-4 tensor: P'_{ijkl} = R_{ia} R_{jb} R_{kc} R_{ld} P_{abcd}."""
    return np.einsum("ia,jb,kc,ld,abcd->ijkl", R, R, R, R, P)


def _rotation_to_align(target: NDArray) -> NDArray:
    """Rotation matrix that maps the canonical direction to target.

    For face: canonical = (1,0,0) -> rotation maps (1,0,0) to target/|target|.
    """
    target = np.asarray(target, dtype=float)
    t_norm = target / np.linalg.norm(target)

    # Build rotation from canonical basis to align axis-0 with target
    e0 = t_norm
    # Choose e1 perpendicular to e0
    if abs(e0[2]) < 0.9:
        e1 = np.cross(e0, np.array([0.0, 0.0, 1.0]))
    else:
        e1 = np.cross(e0, np.array([1.0, 0.0, 0.0]))
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(e0, e1)
    return np.column_stack([e0, e1, e2])  # columns = new basis


def inter_voxel_propagator(
    R_lattice: tuple[int, int, int], mu: float, nu: float
) -> NDArray:
    """Static inter-voxel strain propagator for nearest-neighbour cubes.

    Args:
        R_lattice: integer lattice vector (n1, n2, n3) of the neighbour.
            Must be one of the 26 nearest neighbours (face/edge/corner).
        mu: shear modulus of the reference medium.
        nu: Poisson's ratio of the reference medium.

    Returns:
        P: shape (3,3,3,3) propagator tensor P_{ijkl}(R).
    """
    n = np.array(R_lattice, dtype=int)
    n_abs = np.sort(np.abs(n))[::-1]  # descending sorted absolute values

    if np.array_equal(n_abs, [1, 0, 0]):
        # Face-adjacent: 6 neighbours
        P_canon = face_propagator(mu, nu)
    elif np.array_equal(n_abs, [1, 1, 0]):
        # Edge-adjacent: 12 neighbours
        P_canon = edge_propagator(mu, nu)
    elif np.array_equal(n_abs, [1, 1, 1]):
        # Corner-adjacent: 8 neighbours
        P_canon = corner_propagator(mu, nu)
    else:
        msg = f"R_lattice={R_lattice} is not a nearest neighbour"
        raise ValueError(msg)

    # Apply O_h rotation to map canonical direction to actual R
    # For face: canonical is along axis 0 (R=(1,0,0))
    # For edge: canonical is (1,1,0)
    # For corner: canonical is (1,1,1)
    R_canon = np.array(
        [1, 0, 0]
        if np.array_equal(n_abs, [1, 0, 0])
        else [1, 1, 0]
        if np.array_equal(n_abs, [1, 1, 0])
        else [1, 1, 1],
        dtype=float,
    )

    # Build the signed target direction
    R_target = np.array(R_lattice, dtype=float)

    # Find the O_h transformation: signed permutation matrix
    # that maps R_canon (with appropriate signs) to R_target
    perm_matrix = _oh_permutation(R_canon, R_target)
    if perm_matrix is not None:
        return _rotate_tensor4(P_canon, perm_matrix)

    # Fallback: general rotation
    rot = _rotation_to_align(R_target)
    return _rotate_tensor4(P_canon, rot)


def _oh_permutation(source: NDArray, target: NDArray) -> NDArray | None:
    """Find signed permutation matrix P such that P @ source = target.

    Returns None if no such matrix exists (should not happen for O_h).
    """
    s = source.copy()
    t = target.copy()

    P = np.zeros((3, 3))
    used = [False, False, False]

    for i in range(3):
        for j in range(3):
            if not used[j] and abs(abs(t[i]) - abs(s[j])) < 1e-10 and abs(s[j]) > 1e-10:
                P[i, j] = np.sign(t[i]) * np.sign(s[j])
                used[j] = True
                break
            if not used[j] and abs(s[j]) < 1e-10 and abs(t[i]) < 1e-10:
                P[i, j] = 1.0
                used[j] = True
                break

    # Verify
    if np.allclose(P @ s, t) and abs(np.linalg.det(P)) > 0.5:
        return P
    return None


# ──────────────────────────────────────────────────────────────────────
# Frequency-dependent propagator (static + dynamic corrections)
# ──────────────────────────────────────────────────────────────────────


def _dynamic_correction(
    R_lattice: tuple[int, int, int],
    order: int,
    rho: float,
    alpha: float,
    beta: float,
) -> NDArray:
    """Compute P⁽ⁿ⁾ for a given neighbour with O_h rotation applied."""
    n = np.array(R_lattice, dtype=int)
    n_abs = np.sort(np.abs(n))[::-1]

    if np.array_equal(n_abs, [1, 0, 0]):
        P_canon = _face_propagator_dyn(order, rho, alpha, beta)
    elif np.array_equal(n_abs, [1, 1, 0]):
        P_canon = _edge_propagator_dyn(order, rho, alpha, beta)
    elif np.array_equal(n_abs, [1, 1, 1]):
        P_canon = _corner_propagator_dyn(order, rho, alpha, beta)
    else:
        msg = f"R_lattice={R_lattice} is not a nearest neighbour"
        raise ValueError(msg)

    R_canon = np.array(
        [1, 0, 0]
        if np.array_equal(n_abs, [1, 0, 0])
        else [1, 1, 0]
        if np.array_equal(n_abs, [1, 1, 0])
        else [1, 1, 1],
        dtype=float,
    )
    R_target = np.array(R_lattice, dtype=float)
    perm_matrix = _oh_permutation(R_canon, R_target)
    if perm_matrix is not None:
        return _rotate_tensor4(P_canon, perm_matrix)
    rot = _rotation_to_align(R_target)
    return _rotate_tensor4(P_canon, rot)


def dynamic_inter_voxel_propagator(
    R_lattice: tuple[int, int, int],
    alpha: float,
    beta: float,
    rho: float,
    omega: float,
    n_orders: int = 3,
) -> NDArray:
    """Frequency-dependent inter-voxel propagator P(ω) = Σₙ ω²ⁿ P⁽ⁿ⁾.

    Computes the analytical power series for the volume-averaged strain
    propagator between nearest-neighbour cubic voxels, valid for ka < π.

    Args:
        R_lattice: integer lattice vector (face/edge/corner neighbour).
        alpha: P-wave velocity of reference medium (m/s).
        beta: S-wave velocity of reference medium (m/s).
        rho: density of reference medium (kg/m³).
        omega: angular frequency (rad/s).
        n_orders: dynamic correction orders (0=static, 1=+ω², 2=+ω⁴, 3=+ω⁶).

    Returns:
        P: shape (3,3,3,3) propagator tensor P_{ijkl}(R, ω).
    """
    mu = rho * beta**2
    nu = (alpha**2 - 2.0 * beta**2) / (2.0 * (alpha**2 - beta**2))

    P = inter_voxel_propagator(R_lattice, mu, nu)

    for n in range(1, n_orders + 1):
        P = P + omega ** (2 * n) * _dynamic_correction(R_lattice, n, rho, alpha, beta)

    return P


# ──────────────────────────────────────────────────────────────────────
# 9×9 block propagator: [[G, C], [H, S]]
# ──────────────────────────────────────────────────────────────────────

# Voigt index pairs: (0,0),(1,1),(2,2),(1,2),(0,2),(0,1)
_VOIGT_PAIRS = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]


def _P_to_voigt_S(P: NDArray) -> NDArray:
    """Convert (3,3,3,3) strain propagator to (6,6) Voigt S block.

    The Voigt contraction maps P_{ijkl} to S_{αβ} where α=(ij) and β=(kl)
    with the standard ordering (11,22,33,23,13,12). The engineering-strain
    convention introduces factors of 2 for shear indices.

    S_{αβ} = mult_α × [P_{ijkl} + (1-δ_{kl}) P_{ijlk}] × (1 if β<3 else 1/2)

    where mult_α = 2 if α is shear (α≥3), else 1.
    """
    S = np.zeros((6, 6), dtype=P.dtype)
    for alpha, (p, q) in enumerate(_VOIGT_PAIRS):
        mult_pq = 2 if p != q else 1
        for beta, (m, n) in enumerate(_VOIGT_PAIRS):
            val = mult_pq * P[p, q, m, n]
            if m != n:
                val += mult_pq * P[p, q, n, m]
            S[alpha, beta] = val
    # Engineering strain convention: halve shear columns
    for beta in range(3, 6):
        S[:, beta] *= 0.5
    return S


def _rotate_matrix3(M: NDArray, R: NDArray) -> NDArray:
    """Rotate 3×3 matrix: M' = R M Rᵀ."""
    return R @ M @ R.T


def _rotate_voigt6(S: NDArray, R: NDArray) -> NDArray:
    """Rotate 6×6 Voigt matrix via rank-4 round-trip.

    Convert S to (3,3,3,3), rotate, convert back. Correct but not fast
    — fine for 26 neighbours computed once.
    """
    # Voigt -> tensor
    P = np.zeros((3, 3, 3, 3), dtype=S.dtype)
    for alpha, (i, j) in enumerate(_VOIGT_PAIRS):
        for beta, (k, l) in enumerate(_VOIGT_PAIRS):
            # Undo the Voigt factors
            val = S[alpha, beta]
            if beta >= 3:
                val *= 2.0  # undo shear column halving
            mult_ij = 2 if i != j else 1
            if k != l:
                # S_αβ was formed as mult_ij * (P_ijkl + P_ijlk)
                # For symmetric P, P_ijkl = P_ijlk, so val = mult_ij * 2 * P_ijkl
                P[i, j, k, l] = val / (mult_ij * 2.0)
                P[i, j, l, k] = val / (mult_ij * 2.0)
            else:
                P[i, j, k, l] = val / mult_ij
            if i != j:
                P[j, i, k, l] = P[i, j, k, l]
                if k != l:
                    P[j, i, l, k] = P[i, j, l, k]
    # Rotate
    P_rot = _rotate_tensor4(P, R)
    # Tensor -> Voigt
    return _P_to_voigt_S(P_rot)


# ── G block: volume-averaged Green's tensor ──
#
# The static volume-averaged Green's tensor is:
#   <G_ij>^(0) = (1/(4πμ)) [δ_ij Φ₀₀ - η_s d²Ψ₀₀/dR_i dR_j]
#
# where η_s = 1/(4(1-ν)), and the potentials relate via Laplacian identities:
#   Φ₀₀ = (1/2) Σ_k d²Ψ₀₀/dR_k²    (Newton = ½ ∇² biharmonic)
#   Ψ₀₀ = (1/12) Σ_k d²X₀₀/dR_k²   (biharmonic = 1/12 ∇² triharmonic)
#
# The raw d²Ψ/dR² values are obtained from the DYN1 A constants:
#   DYN1_A_{ij} = raw_d²Ψ/dR_i dR_j × _NORM_A1   where _NORM_A1 = 1/(8π)
# So:  raw_d²Ψ/dR² = DYN1_A / _NORM_A1
#
# Dynamic order ω²:
#   <G_ij>^(1) = coefficients × [δ_ij Ψ₀₀ - η₁ d²X₀₀/dR_i dR_j]
#   where Ψ₀₀ = (1/12) Tr(d²X₀₀/dR²), values from DYN2_A.
#
# All normalisation factors:
#   d²Ψ/dR² = DYN1_A / (1/(8π))   = DYN1_A × 8π
#   d²X/dR² = DYN2_A / (-1/(96π)) = DYN2_A × (-96π)
#
# Green's tensor at order n uses potential W_{n+1}:
#   <G_ij>^(n) = 1/(ρ c_s^{2n+2}) × [c_n δ_ij W_{n+1} - d_n d²W_{n+2}/dR_i dR_j]
#
# For the isotropic + deviatoric decomposition:
#   c_0 = 1/(4πμ),  d_0 = 1/(16πμ(1-ν))
#   c_1 = see formula below,  d_1 = ...


def _get_raw_d2W(
    neighbour_type: str, order: int
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Get raw d²W/dR² values (diag, offdiag) for a given potential order.

    order=0: d²Ψ/dR² from DYN1_A (raw = stored / _NORM_A1)
    order=1: d²X/dR² from DYN2_A (raw = stored / _NORM_A2)
    order=2: d²Ω/dR² from DYN3_A (raw = stored / _NORM_A3)
    order=3: d²H/dR² from DYN4_A (raw = stored / _NORM_A4)
    """
    if order == 0:
        norm = _NORM_A1
        if neighbour_type == "face":
            diag = (DYN1_FACE_A11 / norm, DYN1_FACE_A22 / norm, DYN1_FACE_A22 / norm)
            offdiag = (0.0, 0.0, 0.0)
        elif neighbour_type == "edge":
            diag = (DYN1_EDGE_A11 / norm, DYN1_EDGE_A11 / norm, DYN1_EDGE_A33 / norm)
            offdiag = (DYN1_EDGE_A12 / norm, 0.0, 0.0)
        else:
            diag = (DYN1_CORNER_A11 / norm,) * 3
            offdiag = (DYN1_CORNER_A12 / norm,) * 3
    elif order == 1:
        norm = _NORM_A2
        if neighbour_type == "face":
            diag = (DYN2_FACE_A11 / norm, DYN2_FACE_A22 / norm, DYN2_FACE_A22 / norm)
            offdiag = (0.0, 0.0, 0.0)
        elif neighbour_type == "edge":
            diag = (DYN2_EDGE_A11 / norm, DYN2_EDGE_A11 / norm, DYN2_EDGE_A33 / norm)
            offdiag = (DYN2_EDGE_A12 / norm, 0.0, 0.0)
        else:
            diag = (DYN2_CORNER_A11 / norm,) * 3
            offdiag = (DYN2_CORNER_A12 / norm,) * 3
    elif order == 2:
        norm = _NORM_A3
        if neighbour_type == "face":
            diag = (DYN3_FACE_A11 / norm, DYN3_FACE_A22 / norm, DYN3_FACE_A22 / norm)
            offdiag = (0.0, 0.0, 0.0)
        elif neighbour_type == "edge":
            diag = (DYN3_EDGE_A11 / norm, DYN3_EDGE_A11 / norm, DYN3_EDGE_A33 / norm)
            offdiag = (DYN3_EDGE_A12 / norm, 0.0, 0.0)
        else:
            diag = (DYN3_CORNER_A11 / norm,) * 3
            offdiag = (DYN3_CORNER_A12 / norm,) * 3
    elif order == 3:
        norm = _NORM_A4
        if neighbour_type == "face":
            diag = (DYN4_FACE_A11 / norm, DYN4_FACE_A22 / norm, DYN4_FACE_A22 / norm)
            offdiag = (0.0, 0.0, 0.0)
        elif neighbour_type == "edge":
            diag = (DYN4_EDGE_A11 / norm, DYN4_EDGE_A11 / norm, DYN4_EDGE_A33 / norm)
            offdiag = (DYN4_EDGE_A12 / norm, 0.0, 0.0)
        else:
            diag = (DYN4_CORNER_A11 / norm,) * 3
            offdiag = (DYN4_CORNER_A12 / norm,) * 3
    else:
        msg = f"Potential order {order} not available (only 0, 1, 2, and 3)"
        raise ValueError(msg)
    return diag, offdiag


def _build_G_block_canonical(
    neighbour_type: str,
    mu: float,
    nu: float,
    rho: float,
    alpha: float,
    beta: float,
    omega: float,
    n_orders: int,
) -> NDArray:
    """Build 3×3 volume-averaged Green's tensor for canonical direction.

    Static (n=0):
      <G_ij>^(0) = 1/(4πμ) [δ_ij Φ - 1/(4(1-ν)) d²Ψ/dR_i dR_j]
      where Φ = (1/2) Tr(d²Ψ/dR²)

    Dynamic (n=1, ω² correction):
      <G_ij>^(1) = 1/(4πρc_s⁴) [δ_ij Ψ - η₁/(4) d²X/dR_i dR_j]
      where Ψ = (1/12) Tr(d²X/dR²), η₁ = 1 - (c_s/c_p)⁴

    The factor 1/(4π) comes from G = Φδ/(4πμ) - d²Ψ/(16πμ(1-ν)dR²).
    """
    cs, cp = beta, alpha
    delta = np.eye(3)
    G = np.zeros((3, 3))

    # Static term (n=0): uses d²Ψ/dR² from DYN1_A (order=0)
    d2Psi_diag, d2Psi_offdiag = _get_raw_d2W(neighbour_type, order=0)
    d2Psi = _build_A_matrix(d2Psi_diag, d2Psi_offdiag)
    Phi = 0.5 * np.trace(d2Psi)  # Laplacian identity: Φ = ½ ∇²Ψ
    eta_s = 1.0 / (4.0 * (1.0 - nu))
    G = (1.0 / (4.0 * np.pi * mu)) * (delta * Phi - eta_s * d2Psi)

    if n_orders >= 1:
        # ω² correction (n=1): uses d²X/dR² from DYN2_A (order=1)
        d2X_diag, d2X_offdiag = _get_raw_d2W(neighbour_type, order=1)
        d2X = _build_A_matrix(d2X_diag, d2X_offdiag)
        Psi_val = (1.0 / 12.0) * np.trace(d2X)  # Ψ = (1/12) ∇²X

        # Coefficient from Taylor expansion of exp(ik_S r)/r:
        #   G^(1) = (-1)^1 ω² / (2! × 4πρc_S⁴) × [δΨ - η₁/12 × d²X]
        # where η₁/12 = η₁/((2·1+1)(2·1+2)) is the deviatoric suppression
        eta_1 = 1.0 - (cs / cp) ** 4
        coeff_1 = -(omega**2) / (8.0 * np.pi * rho * cs**4)
        G += coeff_1 * (delta * Psi_val - (eta_1 / 12.0) * d2X)

    if n_orders >= 2:
        # ω⁴ correction (n=2): uses d²Ω/dR² from DYN3_A (order=2)
        d2Omega_diag, d2Omega_offdiag = _get_raw_d2W(neighbour_type, order=2)
        d2Omega = _build_A_matrix(d2Omega_diag, d2Omega_offdiag)
        X_val = (1.0 / 30.0) * np.trace(d2Omega)  # X = (1/30) ∇²Ω

        # Coefficient from Taylor expansion:
        #   G^(2) = (-1)^2 ω⁴ / (4! × 4πρc_S⁶) × [δX - η₂/30 × d²Ω]
        # where η₂/30 = η₂/((2·2+1)(2·2+2)) is the deviatoric suppression
        eta_2 = 1.0 - (cs / cp) ** 6
        coeff_2 = omega**4 / (96.0 * np.pi * rho * cs**6)
        G += coeff_2 * (delta * X_val - (eta_2 / 30.0) * d2Omega)

    if n_orders >= 3:
        # ω⁶ correction (n=3): uses d²H/dR² from DYN4_A (order=3)
        d2H_diag, d2H_offdiag = _get_raw_d2W(neighbour_type, order=3)
        d2H = _build_A_matrix(d2H_diag, d2H_offdiag)
        Omega_val = (1.0 / 56.0) * np.trace(d2H)  # Ω = (1/56) ∇²H

        # Coefficient from Taylor expansion:
        #   G^(3) = (-1)^3 ω⁶ / (6! × 4πρc_S⁸) × [δΩ - η₃/56 × d²H]
        # where η₃/56 = η₃/((2·3+1)(2·3+2)) is the deviatoric suppression
        eta_3 = 1.0 - (cs / cp) ** 8
        coeff_3 = -(omega**6) / (2880.0 * np.pi * rho * cs**8)
        G += coeff_3 * (delta * Omega_val - (eta_3 / 56.0) * d2H)

    return G


def _classify_neighbour(R_lattice: tuple[int, int, int]) -> str:
    """Classify lattice vector as face/edge/corner."""
    n_abs = np.sort(np.abs(R_lattice))[::-1]
    if np.array_equal(n_abs, [1, 0, 0]):
        return "face"
    if np.array_equal(n_abs, [1, 1, 0]):
        return "edge"
    if np.array_equal(n_abs, [1, 1, 1]):
        return "corner"
    msg = f"R_lattice={R_lattice} is not a nearest neighbour"
    raise ValueError(msg)


def _get_oh_perm(R_lattice: tuple[int, int, int]) -> NDArray:
    """Get the O_h permutation matrix for a lattice vector."""
    n_abs = np.sort(np.abs(R_lattice))[::-1]
    R_canon = np.array(
        [1, 0, 0]
        if np.array_equal(n_abs, [1, 0, 0])
        else [1, 1, 0]
        if np.array_equal(n_abs, [1, 1, 0])
        else [1, 1, 1],
        dtype=float,
    )
    R_target = np.array(R_lattice, dtype=float)
    perm = _oh_permutation(R_canon, R_target)
    if perm is not None:
        return perm
    return _rotation_to_align(R_target)


def _build_D3_tensor(neighbour_type: str, order: int = 0) -> NDArray:
    """Build (3,3,3) fully symmetric third-derivative tensor.

    order=0: d³Ψ/dR³ (biharmonic, kernel ρ)   — static
    order=1: d³X/dR³  (triharmonic, kernel ρ³) — ω² correction
    order=2: d³Ω/dR³  (pentaharmonic, kernel ρ⁵) — ω⁴ correction
    order=3: d³H/dR³  (heptaharmonic, kernel ρ⁷) — ω⁶ correction
    """
    from itertools import permutations

    # Select constants by order
    if order == 0:
        f000, f011 = FACE_D3PSI_000, FACE_D3PSI_011
        e000, e001, e022 = EDGE_D3PSI_000, EDGE_D3PSI_001, EDGE_D3PSI_022
        c000, c001, c012 = CORNER_D3PSI_000, CORNER_D3PSI_001, CORNER_D3PSI_012
    elif order == 1:
        f000, f011 = FACE_D3X_000, FACE_D3X_011
        e000, e001, e022 = EDGE_D3X_000, EDGE_D3X_001, EDGE_D3X_022
        c000, c001, c012 = CORNER_D3X_000, CORNER_D3X_001, CORNER_D3X_012
    elif order == 2:
        f000, f011 = FACE_D3OM_000, FACE_D3OM_011
        e000, e001, e022 = EDGE_D3OM_000, EDGE_D3OM_001, EDGE_D3OM_022
        c000, c001, c012 = CORNER_D3OM_000, CORNER_D3OM_001, CORNER_D3OM_012
    elif order == 3:
        f000, f011 = FACE_D3H_000, FACE_D3H_011
        e000, e001, e022 = EDGE_D3H_000, EDGE_D3H_001, EDGE_D3H_022
        c000, c001, c012 = CORNER_D3H_000, CORNER_D3H_001, CORNER_D3H_012
    else:
        msg = f"D3 tensor order {order} not available (only 0, 1, 2, and 3)"
        raise ValueError(msg)

    D = np.zeros((3, 3, 3))
    if neighbour_type == "face":
        entries: dict[tuple[int, int, int], float] = {
            (0, 0, 0): f000,
            (0, 1, 1): f011,
            (0, 2, 2): f011,
        }
    elif neighbour_type == "edge":
        entries = {
            (0, 0, 0): e000,
            (1, 1, 1): e000,
            (0, 0, 1): e001,
            (0, 1, 1): e001,
            (0, 2, 2): e022,
            (1, 2, 2): e022,
        }
    else:  # corner
        entries = {
            (0, 0, 0): c000,
            (1, 1, 1): c000,
            (2, 2, 2): c000,
            (0, 0, 1): c001,
            (0, 0, 2): c001,
            (0, 1, 1): c001,
            (0, 2, 2): c001,
            (1, 1, 2): c001,
            (1, 2, 2): c001,
            (0, 1, 2): c012,
        }
    for (i, j, k), val in entries.items():
        for perm in set(permutations((i, j, k))):
            D[perm] = val
    return D


# Keep old name as alias for backward compatibility in tests
_build_D3Psi_tensor = _build_D3_tensor


def _build_dW_vector(neighbour_type: str, order: int = 0) -> NDArray:
    """Build (3,) first-derivative vector for the isotropic piece.

    order=0: dΦ/dR_k  (Newton, from Laplacian of d³Ψ)
    order=1: dΨ/dR_k  (biharmonic, from Laplacian of d³X)
    order=2: dX/dR_k  (triharmonic, from Laplacian of d³Ω)
    order=3: dΩ/dR_k  (pentaharmonic, from Laplacian of d³H)
    """
    if order == 0:
        f0, e0, c0 = FACE_DPHI_0, EDGE_DPHI_0, CORNER_DPHI_0
    elif order == 1:
        f0, e0, c0 = FACE_DPSI_0, EDGE_DPSI_0, CORNER_DPSI_0
    elif order == 2:
        f0, e0, c0 = FACE_DX_0, EDGE_DX_0, CORNER_DX_0
    elif order == 3:
        f0, e0, c0 = FACE_DOM_0, EDGE_DOM_0, CORNER_DOM_0
    else:
        msg = f"dW vector order {order} not available (only 0, 1, 2, and 3)"
        raise ValueError(msg)

    if neighbour_type == "face":
        return np.array([f0, 0.0, 0.0])
    if neighbour_type == "edge":
        return np.array([e0, e0, 0.0])
    return np.array([c0, c0, c0])


# Keep old name as alias
_build_dPhi_vector = _build_dW_vector


def _build_dG_rank3_canonical(
    neighbour_type: str,
    mu: float,
    nu: float,
    rho: float = 0.0,
    alpha: float = 0.0,
    beta: float = 0.0,
    omega: float = 0.0,
    n_orders: int = 0,
) -> NDArray:
    """Build rank-3 tensor dG_{ijk} = d<G_ij>/dR_k for canonical direction.

    Static (n=0):
      dG^(0)_{ijk} = (1/(4πμ)) [δ_{ij} dΦ_k − η_s d³Ψ_{ijk}]
      where η_s = 1/(4(1−ν))

    Dynamic (n=1, ω²):
      dG^(1)_{ijk} = ω²/(4πρc_s⁴) [δ_{ij} dΨ_k − (η₁/4) d³X_{ijk}]
      where η₁ = 1 − (c_s/c_p)⁴

    Dynamic (n=2, ω⁴):
      dG^(2)_{ijk} = ω⁴/(4πρc_s⁶) [δ_{ij} dX_k − (η₂/4) d³Ω_{ijk}]
      where η₂ = 1 − (c_s/c_p)⁶
    """
    cs, cp = beta, alpha
    eta_s = 1.0 / (4.0 * (1.0 - nu))
    D3Psi = _build_D3_tensor(neighbour_type, order=0)
    dPhi = _build_dW_vector(neighbour_type, order=0)

    delta = np.eye(3)
    dG = np.zeros((3, 3, 3))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                dG[i, j, k] = (1.0 / (4.0 * np.pi * mu)) * (
                    delta[i, j] * dPhi[k] - eta_s * D3Psi[i, j, k]
                )

    if n_orders >= 1 and omega != 0.0:
        # ω² correction: d³X/dR³ (triharmonic) + dΨ/dR (biharmonic)
        D3X = _build_D3_tensor(neighbour_type, order=1)
        dPsi = _build_dW_vector(neighbour_type, order=1)
        eta_1 = 1.0 - (cs / cp) ** 4
        coeff_1 = -(omega**2) / (8.0 * np.pi * rho * cs**4)
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    dG[i, j, k] += coeff_1 * (
                        delta[i, j] * dPsi[k] - (eta_1 / 12.0) * D3X[i, j, k]
                    )

    if n_orders >= 2 and omega != 0.0:
        # ω⁴ correction: d³Ω/dR³ (pentaharmonic) + dX/dR (triharmonic)
        D3Om = _build_D3_tensor(neighbour_type, order=2)
        dX = _build_dW_vector(neighbour_type, order=2)
        eta_2 = 1.0 - (cs / cp) ** 6
        coeff_2 = omega**4 / (96.0 * np.pi * rho * cs**6)
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    dG[i, j, k] += coeff_2 * (
                        delta[i, j] * dX[k] - (eta_2 / 30.0) * D3Om[i, j, k]
                    )

    if n_orders >= 3 and omega != 0.0:
        # ω⁶ correction: d³H/dR³ (heptaharmonic) + dΩ/dR (pentaharmonic)
        D3H = _build_D3_tensor(neighbour_type, order=3)
        dOm = _build_dW_vector(neighbour_type, order=3)
        eta_3 = 1.0 - (cs / cp) ** 8
        coeff_3 = -(omega**6) / (2880.0 * np.pi * rho * cs**8)
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    dG[i, j, k] += coeff_3 * (
                        delta[i, j] * dOm[k] - (eta_3 / 56.0) * D3H[i, j, k]
                    )

    return dG


def _dG_to_C_block(dG: NDArray) -> NDArray:
    """Contract rank-3 dG_{ijk} to (3,6) C block via Voigt mapping.

    C_{i, α=(mm)} = dG[i, m, m]
    C_{i, α=(mn)} = ½(dG[i, m, n] + dG[i, n, m])   (shear, engineering convention)

    Maps strain source at B to displacement response at A.
    """
    C = np.zeros((3, 6), dtype=dG.dtype)
    for alpha, (m, n) in enumerate(_VOIGT_PAIRS):
        for i in range(3):
            if m == n:
                C[i, alpha] = dG[i, m, m]
            else:
                C[i, alpha] = 0.5 * (dG[i, m, n] + dG[i, n, m])
    return C


def _rotate_tensor3(T: NDArray, R: NDArray) -> NDArray:
    """Rotate rank-3 tensor: T'_{ijk} = R_{ia} R_{jb} R_{kc} T_{abc}."""
    return np.einsum("ia,jb,kc,abc->ijk", R, R, R, T)


def inter_voxel_propagator_9x9(
    R_lattice: tuple[int, int, int],
    alpha: float,
    beta: float,
    rho: float,
    omega: float,
    n_orders: int = 2,
) -> NDArray:
    """9×9 inter-voxel propagator coupling displacement + Voigt strain.

    Returns the block matrix [[G, C], [H, S]] where:
    - G (3×3): volume-averaged Green's tensor <G_ij>
    - S (6×6): Voigt contraction of the strain propagator P_{ijkl}
    - C (3×6): displacement-strain coupling (d<G_ij>/dR_k Voigt-contracted)
    - H (6×3): strain-displacement coupling (H = Cᵀ)

    Args:
        R_lattice: integer lattice vector (face/edge/corner neighbour).
        alpha: P-wave velocity of reference medium (m/s).
        beta: S-wave velocity of reference medium (m/s).
        rho: density of reference medium (kg/m³).
        omega: angular frequency (rad/s).
        n_orders: dynamic correction orders (0=static, 1=+ω², 2=+ω⁴, 3=+ω⁶).
            All blocks (G, S, C/H) support n_orders ≤ 3.


    Returns:
        P9: shape (9, 9) complex array.
    """
    mu = rho * beta**2
    nu = (alpha**2 - 2.0 * beta**2) / (2.0 * (alpha**2 - beta**2))
    ntype = _classify_neighbour(R_lattice)
    perm = _get_oh_perm(R_lattice)

    # S block: Voigt contraction of the (3,3,3,3) dynamic propagator
    s_orders = min(n_orders, 3)
    P_ijkl = dynamic_inter_voxel_propagator(
        R_lattice, alpha, beta, rho, omega, s_orders
    )
    S = _P_to_voigt_S(P_ijkl)

    # G block: volume-averaged Green's tensor (canonical then rotate)
    g_orders = min(n_orders, 3)
    G_canon = _build_G_block_canonical(ntype, mu, nu, rho, alpha, beta, omega, g_orders)
    G = _rotate_matrix3(G_canon, perm)

    # C, H blocks: displacement-strain coupling from dG/dR (dynamic)
    ch_orders = min(n_orders, 3)
    dG_canon = _build_dG_rank3_canonical(
        ntype, mu, nu, rho, alpha, beta, omega, ch_orders
    )
    dG_rot = _rotate_tensor3(dG_canon, perm)
    C = _dG_to_C_block(dG_rot)
    H = C.T  # H = Cᵀ (verified analytically, no sign flip)

    P9 = np.zeros((9, 9), dtype=complex)
    P9[:3, :3] = G
    P9[:3, 3:] = C
    P9[3:, :3] = H
    P9[3:, 3:] = S
    return P9
