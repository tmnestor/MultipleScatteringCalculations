# The two-scale formulation of the Galerkin projection (Paper 3, Part III)

Status: derivation, with its numerical check in `scripts/derive_two_scale_galerkin.py`.

## 1. Two nested Galerkin problems

A cell of half-width H divides into eight children of half-width h = H/2. Let

- V_H be the state functions that are polynomials of degree p on every coarse cell,
- V_h the same on every child.

Then V_H ⊂ V_h. A polynomial of degree p on a parent is a polynomial of degree p on each child:
Q_a(ξ_parent) = Σ_b C_d[a,b] Q_b(ξ_child), with ξ_parent = ξ_child/2 + s_d and s_d ∈ {±1/2}³
(`octree.field_reexpansion`). The prolongation S maps coarse coefficients to fine ones exactly:
(S x)_{d,b} = Σ_a x_a C_d[a,b].

On each grid the Galerkin system tests the volume-integral equation with that grid's own functions. The
contrast is projected to degree r on that grid's cells: Δ_H = Π_H Δ and Δ_h = Π_h Δ. In matrices:

    A_H x_H = b_H,   A_H = M_H − K_H E_H,
    A_h y_h = b_h,   A_h = M_h − K_h E_h.

Nesting gives S^T M_h S = M_H and S^T b_h = b_H exactly, and S^T K_h (·) S = K_H (·) exactly, because the
coupling integrals are linear in the functions. But S^T K_h E_h S ≠ K_H E_H: the fine operator applies the
fine contrast Δ_h to the coarse field.

## 2. The residual of the prolonged coarse solution

Let ψ_H = S x_H, the coarse solution as a fine function, and let

    r_h = b_h − A_h S x_H.

Then the coarse-to-fine error e = y_h − S x_H satisfies, exactly,

    A_h e = r_h.                                                           (1)

Split the test space as V_h = V_H ⊕ W_H, with W_H the multiwavelets of each parent: degree p on the
children, orthogonal to V_H. The dual projector onto V_H is P = M_h S M_H⁻¹ S^T. Write r_med = P r_h and
r_fld = (I − P) r_h.

**The medium part.** Using A_H x_H = b_H,

    S^T r_h = b_H − S^T A_h S x_H = K_H (Ẽ − E_H) x_H,                      (2)

with Ẽ the coarse field multiplied by the fine contrast. So r_med is driven only by the contrast's
detail δΔ = Δ_h − Δ_H. It vanishes when the medium is held exactly at the coarse scale.

**The field part.** For w ∈ W_H, ⟨w, ψ_H⟩ = 0, so

    (I − P) r_h ↔ ⟨w, ψ⁰ + 𝒦 Δ_h ψ_H⟩.                                    (3)

This is the multiwavelet detail of the field implied by the coarse solution: the part that the coarse cells
cannot hold. At first order in the contrast it is the detail of the incident wave.

## 3. The far field

The far field is a linear functional of the source, F = R(Δψ). Then, exactly,

    F_h − F_H = R Δ_h A_h⁻¹ r_med + R Δ_h A_h⁻¹ r_fld + R(δΔ ψ_H).         (4)

The three terms are the medium's detail propagated by the solve, the field's detail propagated by the
solve, and the medium's detail radiating with the coarse field. With the adjoint z_h,
A_h^T z_h = (R E_h)^T, the first two are z_h^T r_med and z_h^T r_fld, sums over cells. That makes (4) a
dual-weighted residual: each cell's share of the error is explicit.

## 4. What is to be checked

1. Identity (1) and the split (4), to the tolerance of the solves.
2. That (2) holds: S^T r_h equals K_H (Ẽ − E_H) x_H, which vanishes for a contrast uniform at the coarse
   scale.
3. Saturation: at fourth order, F_H − F_exact ≈ (16/15)(F_H − F_h). The two-scale difference then
   measures the coarse model's error.
4. The sizes of the three terms of (4) on the graded sphere, against the two terms of the octree paper's
   law (Born error, and nonlinear fraction × medium projection error).
5. A local estimate of e, without the fine solve (block-diagonal inverse of A_h per child, or per
   parent), as the refinement indicator.
