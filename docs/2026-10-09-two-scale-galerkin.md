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

## 5. Results (9 October; `scripts/derive_two_scale_galerkin.py`, logs in `scratch/two_scale/`)

Degree-one cells (p = r = 1), graded sphere of Paper 2 (core a/10, smoothstep shell), k_S a = 0.5.

**The identities hold.** y_h = S x_H + e_med + e_fld holds to 1e-13–3e-13 at 4→8, 6→12 and 8→16 cells
across. The far-field split (4) holds to 5e-14. In a medium with no detail (a contrast constant over the
bounding cube), S^T r_h = 7e-13: the medium part of the residual vanishes, as (2) says.

**Saturation.** F_H − F_exact equals F_H − F_h to 7.5%, 7.2% and 6.8% at 4, 6 and 8 cells, as fourth-order
convergence predicts (coarse errors 2.9e-4, 6.4e-5, 2.2e-5; apparent orders 3.75, 3.79, 3.88).

**The orders in the contrast.** The relative sizes of the three terms of (4), against |F_h|:

| contrast | \|F_h − F_H\| | R(δΔ ψ_H) | field (solve) | medium (solve) |
|---|---|---|---|---|
| 1     | 2.7e-4 | 1.7e-5 | 2.5e-4 | 4.1e-5 |
| 0.1   | 2.8e-5 | 2.0e-5 | 2.6e-5 | 4.2e-6 |
| 0.01  | 1.8e-5 | 2.0e-5 | 2.8e-6 | 4.2e-7 |

- **First order (the Born error):** the medium's detail radiating with the coarse field, R(δΔ ψ_H). For
  degree-one cells the incident wave's own detail adds almost nothing at first order.
- **Second order, two parts:** the medium's detail propagated by the solve (r_med), and the detail of the
  field scattered inside the body (the part of r_fld beyond first order). At full contrast the second
  dominates, 2.5e-4 against 4e-5. The octree paper's second term, ν × projection error × F, lumped them;
  here each has its own residual.

**Estimates without a fine solve.**
- Block-diagonal inverse per child: F_h − F_H to 0.18–0.29% on the graded sphere, but 59% off in the
  uniform medium.
- Two-level estimate, ê = L r + S A_H⁻¹ S^T (r − A_h L r): to 0.03–0.14% on the graded sphere, 0.005% at
  weak contrast, and 0.65% in the uniform medium. It needs one coarse solve and one fine matvec.

**The indicator, cell by cell.** Each coarse cell's share of F_h − F_H is its children's error sources plus
its own medium-detail radiation. The shares sum to F_h − F_H to 2e-10. The two-level shares are off by
6e-4 to 8e-4 of the total (3e-6 at weak contrast). The top 20% of cells chosen by the two-level indicator
hold exactly the largest possible share of the error: 0.558 and 0.673 at full contrast, 0.567 and 0.597 at
0.01. The same number of cells chosen by the medium's detail alone hold 0.415, 0.556, 0.413 and 0.395.

## 6. Next

- The octree. The formulation holds for any tree, refining every leaf into its eight children. But the
  fine operator cannot be assembled densely for trees of useful size: 8× the leaves, ~10⁵ unknowns. A fine
  matvec is needed. One option is the FFT on the finest uniform grid with leaves as unions of finest
  cells. Another is to apply the residual leaf by leaf, from the coarse solution's prolongation and the
  equal-cell blocks.
- Part III text: the derivation (sections 1–3), the checks and the table above, the indicator.

## 7. The octree (9 October; `scripts/pilot_octree_two_scale_refinement.py`, logs in `scratch/two_scale/`)

**The finest-grid operator.** Leaf sources re-expand onto the finest cells (half-width a/32), are convolved
by FFT, and are tested back with each leaf's polynomials. On a uniform tree it reproduces the dense octree
solve to 5e-13 (error 7.2499e-3, as stored). The one-off cost is 100–130 s.

**Refinement on the localised feature** (degree one, k_S a = 1, leaves ≥ a/32), against the octree paper's
stored trees:

| leaves | two-scale, tolerance per leaf | two-scale, Dörfler θ = 0.5 | two-term (paper) | medium alone (paper) | uniform (paper) |
|---|---|---|---|---|---|
| ~120–180 | 120: 6.8e-4 | 134: 5.4e-4 | — | 176: 2.47e-4 | 184: 2.73e-3 |
| ~220–260 | 260: 1.677e-4 | 218: 2.9e-4 | 260: 1.677e-4 | — | — |
| ~340 | 344: 1.658e-4 | 337: 1.74e-4 | 336: 1.658e-4 | 344: 1.96e-4 | — |
| ~450–530 | — | 533: 1.19e-4 | 452: 1.14e-4 | — | 408: 6.8e-4 |

- The two-scale indicator with a tolerance per leaf builds the two-term rule's trees. Dörfler marking reaches
  the same band. Neither improves on the two-term rule on this body.
- The two-scale difference, which needs no reference, is 73–97% of the true error on the trees above 100
  leaves.
- Cost: each step needs a coarse solve, a fine matvec and a second coarse solve, 4–5 min here. The two-term
  rule needs no solve.
