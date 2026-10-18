# Route B: the plane-averaged factorisation on the plane-wave modes (19 October)

Follows `docs/2026-10-15-interface-solver-scaling-plan.md`, S6c (DtNSweep, TrueSweep) and route A
(`LeadingSweep`, PR #3).

## Why

`LeadingSweep` (route A) solves the true system's leading systems over all the planes above; on the
matrix-free operator it takes the fewest outer products (6 on 4 x 3 x 3, against the march's 8 and 6) but not
the least time (204 and 165 s, against 141 and 65 s), because each inner product of a leading solve is a full
matrix-free product, preconditioned only cell by cell. `TrueSweep` on the stored kernels preconditions the
same solves with the plane-averaged factorisation (Algorithm 5.1 of the thesis for the medium in which each
plane's contrast is its lateral mean), but builds it from the stored kernels: a dense L x L block system at
every lateral bin, O(L^2) stored kernels per pair, O(L^3) work per bin. At a realistic size (scaling plan S7,
64 x 64 x 8) the stored kernels are tens of GB, which is why the matrix-free operator exists.

Route B builds the same factorisation from what the matrix-free operator already has, without a kernel per
pair of planes, and in O(L) per bin.

## The structure it uses

At a lateral wavenumber q, the averaged depth system is A(q) = M - K(q) E_bar, L x L blocks. The coupling K
splits as the matrix-free operator splits it:

- planes up to s = `stored_planes` apart: the stored lateral kernels, `kernels[dz]` per side, already on the
  box's padded lattice and independent of L;
- planes more than s apart, same side: the whole-space modes, sum_m u_m(z_r) p_m^{|dz|} v_m(z_s) with
  p_m = e^{i k_z,m 2h} (`stratified_apply.WholeSpaceModes`): SEMISEPARABLE in depth, three modes per direction;
- the interface part D: the mode factors of `stratified_apply.ModeFactors`, a rank-3 map per k through R_d or
  T_d plus the window's static pseudo-modes, also separable in the two depths;
- the corrections: finite stencils (`MatrixFreeOperator.stencils`), local in depth.

So A(q) is block-banded (half-width s, plus the local corrections) plus a low-rank semiseparable part, and its
block LDU can be computed by a recursion in depth that carries the semiseparable part's state (the
amplitudes of the modes) rather than every pair: the Riccati / Dirichlet-to-Neumann recursion of Algorithm 5.1,
O(L) per bin.

## The open question, measured first

The stored version's averaged system is the DFT, on the padded grid (2 n - 1)^2, of the kernels TRUNCATED to the
box's offsets. Mode evaluations at the bins give the kernel summed over the INFINITE lattice (Poisson). For a
propagating, slowly decaying kernel the two differ, perhaps substantially on small boxes. Either is an
approximation for a preconditioner; the gap decides which to use, and how the outer count changes.

## Milestones (each gated, each with the thesis kept in sync)

1. **M1: the averaged system from the operator's pieces.** At the padded lattice's bins: the stored kernels as
   they are, the corrections' stencils transformed, the whole-space modes beyond s, D from its mode factors.
   Gate: against `_averaged_system` on the stored kernels, piece by piece; record the truncation/periodisation
   gap, and the outer count of the averaged sweep with each.
2. **M2: the recursion in depth.** The block LDU of A(q) by a forward recursion carrying the modes' state and a
   back substitution, O(L) per bin. Gate: against the dense LDU of M1 at every bin, to round-off.
3. **M3: the preconditioner.** The leading solves of `LeadingSweep` (and windowed `TrueSweep`) on the
   matrix-free operator, preconditioned by M2. Gate: the outer and inner counts and the time against the march
   and block-Jacobi on the S6 boxes; the exact inverse with tight solves.
