(* ============================================================== *)
(* SmallSphereScattering.m                                        *)
(* Self-consistent evaluation of Equation 8 (Lippmann-Schwinger)  *)
(* for a small sphere of radius a with constant isotropic         *)
(* contrast (\[CapitalDelta]\[Lambda], \[CapitalDelta]\[Mu], \[CapitalDelta]\[Rho])                              *)
(*                                                                *)
(* From: Shekhar et al. (2023)                                    *)
(* ============================================================== *)

ClearAll["SmallSphere`*"];

(* Load the Green's function package *)
Get[FileNameJoin[{DirectoryName[$InputFileName /. $InputFileName ->
  NotebookDirectory[]], "ElasticGreensFunction.m"}]];

Print["====================================================="];
Print["  Small Sphere Scattering - Self-Consistent Solution  "];
Print["====================================================="];

(* ============================================================== *)
(* PART 1: ANGULAR INTEGRAL IDENTITIES                             *)
(* ============================================================== *)
(* These are exact results for integrals of direction cosines      *)
(* over the unit sphere \[CapitalOmega]:                                          *)
(*   \[Integral] d\[CapitalOmega] = 4\[Pi]                                                 *)
(*   \[Integral] d\[CapitalOmega] n_i = 0                                             *)
(*   \[Integral] d\[CapitalOmega] n_i n_j = (4\[Pi]/3) \[Delta]_{ij}                            *)
(*   \[Integral] d\[CapitalOmega] n_i n_j n_k = 0                                    *)
(*   \[Integral] d\[CapitalOmega] n_i n_j n_k n_l = (4\[Pi]/15)(\[Delta]_{ij}\[Delta]_{kl}+\[Delta]_{ik}\[Delta]_{jl}+\[Delta]_{il}\[Delta]_{jk}) *)
(* ============================================================== *)

Print["\n--- Part 1: Angular Integral Identities ---"];

(* Kronecker delta *)
\[Delta]f[i_, j_] := KroneckerDelta[i, j];

(* Angular average of n_i n_j *)
angAvg2[i_, j_] := (4 Pi/3) \[Delta]f[i, j];

(* Angular average of n_i n_j n_k n_l *)
angAvg4[i_, j_, k_, l_] := (4 Pi/15) (
  \[Delta]f[i, j] \[Delta]f[k, l] +
  \[Delta]f[i, k] \[Delta]f[j, l] +
  \[Delta]f[i, l] \[Delta]f[j, k]
);

(* Verify: Tr(angAvg2) = \[Integral] d\[CapitalOmega] n_i n_i = 4\[Pi] *)
Print["Check: Sum angAvg2[i,i] = ", Sum[angAvg2[i, i], {i, 3}],
  "  (should be 4\[Pi])"];

(* ============================================================== *)
(* PART 2: GREEN'S TENSOR DECOMPOSITION                            *)
(* ============================================================== *)
(* The Green's tensor has the isotropic form:                      *)
(*   G_{ij}(x) = f(r) \[Delta]_{ij} + g(r) n_i n_j                     *)
(* where n_i = x_i/r is the unit direction vector.                 *)
(*                                                                 *)
(* Identifying from Appendix A:                                    *)
(*   G^{NF}_{ij} = -(C/r)(\[Delta]_{ij} - 3 n_i n_j)                   *)
(*   G^P_{ij}    = (X(r)/r) n_i n_j                               *)
(*   G^S_{ij}    = (V(r)/r)(\[Delta]_{ij} - n_i n_j)                    *)
(*                                                                 *)
(* So: f(r) = -C/r + V(r)/r                                       *)
(*     g(r) = 3C/r - X(r)/r - V(r)/r                              *)
(*     (with Nota Bene: NF has no dependence on \[Omega])                *)
(* ============================================================== *)

Print["\n--- Part 2: Radial Functions ---"];

(* The radial coefficient functions *)
(* C_NF = (1/(8\[Pi]\[Rho]\[Beta]^2))(1 - \[Beta]^2/\[Alpha]^2) *)
cNFval[\[Alpha]_, \[Beta]_, \[Rho]_] := 1/(8 Pi \[Rho] \[Beta]^2) (1 - \[Beta]^2/\[Alpha]^2);

(* f(r) and g(r) for the full Green's tensor *)
fRad[r_, \[Omega]_, \[Alpha]_, \[Beta]_, \[Rho]_] :=
  -cNFval[\[Alpha], \[Beta], \[Rho]]/r +
  Exp[I \[Omega] r/\[Beta]]/(4 Pi \[Rho] \[Beta]^2 r);

gRad[r_, \[Omega]_, \[Alpha]_, \[Beta]_, \[Rho]_] :=
  3 cNFval[\[Alpha], \[Beta], \[Rho]]/r +
  Exp[I \[Omega] r/\[Alpha]]/(4 Pi \[Rho] \[Alpha]^2 r) -
  Exp[I \[Omega] r/\[Beta]]/(4 Pi \[Rho] \[Beta]^2 r);

(* Verify decomposition: G_{ij} = f(r)\[Delta]_{ij} + g(r) n_i n_j *)
Print["Verifying G_{ij} decomposition..."];

(* ============================================================== *)
(* PART 3: VOLUME INTEGRAL OF G_{ij} OVER SPHERE                  *)
(* ============================================================== *)
(* \[Integral]_{sphere} G_{ij}(x') d^3x' = \[Integral]_0^a dr' r'^2 \[Integral]_{S^2} [f(r')\[Delta]_{ij} + g(r') n_i n_j] d\[CapitalOmega] *)
(*                        = \[Delta]_{ij} \[Integral]_0^a r'^2 [4\[Pi] f(r') + (4\[Pi]/3) g(r')] dr' *)
(*                        = \[Delta]_{ij} \[CapitalGamma]_0                                    *)
(* ============================================================== *)

Print["\n--- Part 3: Volume Integral of G_{ij} ---"];

(* The scalar integral \[CapitalGamma]_0 such that \[Integral] G_{ij} d^3x' = \[Delta]_{ij} \[CapitalGamma]_0 *)
\[CapitalGamma]0Integrand[r_, \[Omega]_, \[Alpha]_, \[Beta]_, \[Rho]_] :=
  r^2 (4 Pi fRad[r, \[Omega], \[Alpha], \[Beta], \[Rho]] +
       (4 Pi/3) gRad[r, \[Omega], \[Alpha], \[Beta], \[Rho]]);

(* Simplify the integrand *)
\[CapitalGamma]0IntegrandSimp = FullSimplify[
  \[CapitalGamma]0Integrand[r, \[Omega], \[Alpha], \[Beta], \[Rho]],
  Assumptions -> {r > 0, \[Omega] > 0, \[Alpha] > 0, \[Beta] > 0, \[Rho] > 0, \[Alpha] > \[Beta]}
];
Print["\[CapitalGamma]_0 integrand = ", \[CapitalGamma]0IntegrandSimp];

(* Perform the radial integral exactly *)
\[CapitalGamma]0Exact = Integrate[
  \[CapitalGamma]0Integrand[r, \[Omega], \[Alpha], \[Beta], \[Rho]], {r, 0, a},
  Assumptions -> {a > 0, \[Omega] > 0, \[Alpha] > 0, \[Beta] > 0, \[Rho] > 0, \[Alpha] > \[Beta]}
];
\[CapitalGamma]0Exact = FullSimplify[\[CapitalGamma]0Exact];
Print["\[CapitalGamma]_0 (exact) = ", \[CapitalGamma]0Exact];

(* Small sphere approximation: expand for small \[Omega]*a/\[Alpha] and \[Omega]*a/\[Beta] *)
(* i.e. ka << 1 where k = \[Omega]/\[Alpha] or \[Omega]/\[Beta] *)
\[CapitalGamma]0Small = Normal[Series[\[CapitalGamma]0Exact /. {a -> \[Epsilon] a0}, {\[Epsilon], 0, 4}]] /.
  {\[Epsilon] -> 1, a0 -> a};
\[CapitalGamma]0Small = FullSimplify[\[CapitalGamma]0Small];
Print["\[CapitalGamma]_0 (small sphere, O(a^4)) = ", \[CapitalGamma]0Small];

(* Leading order: \[CapitalGamma]_0 \[TildeTilde] a^2/(6\[Rho])(1/\[Alpha]^2 + 2/\[Beta]^2) * volume_normalization *)
Print["\nSo \[Integral] G_{ij} d^3x' = \[Delta]_{ij} \[CapitalGamma]_0"];

(* ============================================================== *)
(* PART 4: VOLUME INTEGRAL OF G_{ij,k} OVER SPHERE                *)
(* ============================================================== *)
(* By parity (odd integrand over symmetric domain):                *)
(*   \[Integral]_{sphere} G_{ij,k}(x') d^3x' = 0                          *)
(* This means \[Integral] H_{ijk} d^3x' = 0                               *)
(* The stiffness contrast does NOT contribute at the center!       *)
(* ============================================================== *)

Print["\n--- Part 4: Volume Integral of G_{ij,k} (vanishes by symmetry) ---"];
Print["\[Integral] G_{ij,k} d^3x' = 0  (odd integrand over symmetric domain)"];
Print["\[Integral] H_{ijk} d^3x' = 0"];
Print["=> Stiffness contrast does not affect displacement at sphere center."];
Print["=> We need the STRAIN equation for self-consistency."];

(* ============================================================== *)
(* PART 5: VOLUME INTEGRAL OF G_{ij,kl} OVER SPHERE               *)
(* ============================================================== *)
(* The second derivative of G is needed for the strain equation.   *)
(* By isotropy of the sphere:                                      *)
(*   \[Integral] G_{ij,kl} d^3x' = A \[Delta]_{ij}\[Delta]_{kl} + B (\[Delta]_{ik}\[Delta]_{jl} + \[Delta]_{il}\[Delta]_{jk}) *)
(* where A, B are scalar integrals.                                *)
(*                                                                 *)
(* The derivative G_{ij,k} = f'(r)(x_k/r)\[Delta]_{ij} + ...            *)
(* Taking another derivative and integrating requires careful      *)
(* treatment of the angular structure.                             *)
(* ============================================================== *)

Print["\n--- Part 5: Second Derivative Integral (for strain equation) ---"];

(* G_{ij}(x) = f(r)\[Delta]_{ij} + g(r) n_i n_j where n_i = x_i/r *)
(*
   G_{ij,k} = [f'(r) \[Delta]_{ij} + g'(r) n_i n_j] n_k
              + g(r)/r [\[Delta]_{ik} n_j + \[Delta]_{jk} n_i - 2 n_i n_j n_k]

   G_{ij,kl} involves f''(r), g''(r), f'(r)/r, g'(r)/r, g(r)/r^2
   with angular structures: n_k n_l, \[Delta]_{kl}, n_i n_j n_k n_l, etc.
*)

(* Rather than expand all terms, we use the two scalar invariants: *)
(* Trace over ij: G_{ii,kl} -> get one relation *)
(* Trace over ik: G_{ij,jl} -> get another relation *)
(* These give us A and B. *)

(* First, compute G_{ii}(x) = 3f(r) + g(r) (trace of G) *)
(* and G_{ii,kl} integrated over sphere *)

(* Define the two key radial functions for the trace: *)
(* trG(r) = 3f(r) + g(r) *)
trG[r_, \[Omega]_, \[Alpha]_, \[Beta]_, \[Rho]_] :=
  3 fRad[r, \[Omega], \[Alpha], \[Beta], \[Rho]] + gRad[r, \[Omega], \[Alpha], \[Beta], \[Rho]];

(* For the full second derivative integral, we work term by term. *)
(* G_{ij,kl} has the general isotropic 4th-rank structure when    *)
(* integrated over a sphere:                                       *)
(*   I_{ijkl} = A \[Delta]_{ij}\[Delta]_{kl} + B(\[Delta]_{ik}\[Delta]_{jl} + \[Delta]_{il}\[Delta]_{jk})         *)
(*                                                                 *)
(* We determine A and B from two independent contractions:         *)
(*   I_{iikl} = (3A + 2B)\[Delta]_{kl}                                  *)
(*   I_{ijjl} = (A + 4B)\[Delta]_{il}                                   *)
(* ============================================================== *)

(* ---- Contraction 1: I_{iikl} = \[Integral] G_{ii,kl} d^3x' ---- *)
(* G_{ii}(x) = 3f(r) + g(r) is a scalar function of r only *)
(* G_{ii,k} = [3f'(r) + g'(r)] n_k *)
(* G_{ii,kl} = [3f''(r) + g''(r)] n_k n_l + [3f'(r) + g'(r)]/r (\[Delta]_{kl} - n_k n_l) *)

(* Integrate over sphere: *)
(* \[Integral] G_{ii,kl} d^3x' = \[Delta]_{kl} \[Integral]_0^a r^2 { (4\[Pi]/3)[3f'' + g''] + (4\[Pi] - 4\[Pi]/3)[3f' + g']/r } dr *)
(*                   = \[Delta]_{kl} \[Integral]_0^a r^2 { (4\[Pi]/3)(3f'' + g'') + (8\[Pi]/3)(3f' + g')/r } dr *)

(* Define h(r) = 3f(r) + g(r) = trG *)
(* Then G_{ii,kl} integrand coefficient of \[Delta]_{kl}: *)
trace1Integrand[r_, \[Omega]_, \[Alpha]_, \[Beta]_, \[Rho]_] := Module[
  {h, hp, hpp},
  h = trG[r, \[Omega], \[Alpha], \[Beta], \[Rho]];
  hp = D[trG[rr, \[Omega], \[Alpha], \[Beta], \[Rho]], rr] /. rr -> r;
  hpp = D[trG[rr, \[Omega], \[Alpha], \[Beta], \[Rho]], {rr, 2}] /. rr -> r;
  r^2 ((4 Pi/3) hpp + (8 Pi/3) hp/r)
];

\[CapitalGamma]1 = Integrate[
  trace1Integrand[r, \[Omega], \[Alpha], \[Beta], \[Rho]], {r, 0, a},
  Assumptions -> {a > 0, \[Omega] > 0, \[Alpha] > 0, \[Beta] > 0, \[Rho] > 0, \[Alpha] > \[Beta]}
];
\[CapitalGamma]1 = FullSimplify[\[CapitalGamma]1];
Print["I_{iikl} = \[Delta]_{kl} * \[CapitalGamma]_1"];
Print["\[CapitalGamma]_1 = ", \[CapitalGamma]1];

(* ---- Contraction 2: I_{ijjl} = \[Integral] G_{ij,jl} d^3x' ---- *)
(* This requires summing over j in G_{ij,jl}. *)
(* G_{ij,j} = divergence w.r.t. second index of derivative *)
(* Using G_{ij} = f(r)\[Delta]_{ij} + g(r) n_i n_j *)
(* G_{ij,j} = f'(r) n_i + g'(r) n_i + g(r)(2 n_i)/r = [f'(r) + g'(r) + 2g(r)/r] n_i *)
(* Wait, let me be more careful. G_{ij,j} means \[PartialD]G_{ij}/\[PartialD]x_j summed over j *)

(* G_{ij,j} = f'(r) n_j \[Delta]_{ij} + g'(r) n_j n_i n_j + g(r)/r[\[Delta]_{ij}n_j + \[Delta]_{jj}n_i - 2 n_i n_j n_j] *)
(* = f'(r) n_i + g'(r) n_i + g(r)/r[n_i + 3 n_i - 2 n_i] *)
(* Wait, \[Delta]_{jj} = 3, n_j n_j = 1 *)
(* = f'(r) n_i + g'(r) n_i + g(r)/r[n_i + 3 n_i - 2 n_i] *)
(* = [f'(r) + g'(r) + 2g(r)/r] n_i *)

(* Then G_{ij,jl} = \[PartialD]/\[PartialD]x_l of the above *)
(* = [f'' + g'' + 2g'/r - 2g/r^2] n_i n_l + [f' + g' + 2g/r]/r (\[Delta]_{il} - n_i n_l) *)

trace2Integrand[r_, \[Omega]_, \[Alpha]_, \[Beta]_, \[Rho]_] := Module[
  {fp, fpp, gp, gpp, gVal, q, qp},
  fp = D[fRad[rr, \[Omega], \[Alpha], \[Beta], \[Rho]], rr] /. rr -> r;
  fpp = D[fRad[rr, \[Omega], \[Alpha], \[Beta], \[Rho]], {rr, 2}] /. rr -> r;
  gVal = gRad[r, \[Omega], \[Alpha], \[Beta], \[Rho]];
  gp = D[gRad[rr, \[Omega], \[Alpha], \[Beta], \[Rho]], rr] /. rr -> r;
  gpp = D[gRad[rr, \[Omega], \[Alpha], \[Beta], \[Rho]], {rr, 2}] /. rr -> r;
  q = fp + gp + 2 gVal/r;
  qp = fpp + gpp + 2 gp/r - 2 gVal/r^2;
  (* Integral of G_{ij,jl} over sphere:  *)
  (* = \[Delta]_{il} \[Integral] r^2 { (4\[Pi]/3) qp + (8\[Pi]/3) q/r } dr *)
  r^2 ((4 Pi/3) qp + (8 Pi/3) q/r)
];

\[CapitalGamma]2 = Integrate[
  trace2Integrand[r, \[Omega], \[Alpha], \[Beta], \[Rho]], {r, 0, a},
  Assumptions -> {a > 0, \[Omega] > 0, \[Alpha] > 0, \[Beta] > 0, \[Rho] > 0, \[Alpha] > \[Beta]}
];
\[CapitalGamma]2 = FullSimplify[\[CapitalGamma]2];
Print["I_{ijjl} = \[Delta]_{il} * \[CapitalGamma]_2"];
Print["\[CapitalGamma]_2 = ", \[CapitalGamma]2];

(* Solve for A and B: *)
(* 3A + 2B = \[CapitalGamma]_1 *)
(* A + 4B = \[CapitalGamma]_2 *)
ABsol = Solve[{3 AA + 2 BB == \[CapitalGamma]1, AA + 4 BB == \[CapitalGamma]2}, {AA, BB}];
Aval = AA /. ABsol[[1]];
Bval = BB /. ABsol[[1]];
Aval = FullSimplify[Aval];
Bval = FullSimplify[Bval];
Print["\nA = ", Aval];
Print["B = ", Bval];
Print["\n\[Integral] G_{ij,kl} d^3x' = A \[Delta]_{ij}\[Delta]_{kl} + B(\[Delta]_{ik}\[Delta]_{jl} + \[Delta]_{il}\[Delta]_{jk})"];

(* Small sphere expansion *)
AvalSmall = Normal[Series[Aval /. {a -> \[Epsilon] a0}, {\[Epsilon], 0, 3}]] /. {\[Epsilon] -> 1, a0 -> a};
BvalSmall = Normal[Series[Bval /. {a -> \[Epsilon] a0}, {\[Epsilon], 0, 3}]] /. {\[Epsilon] -> 1, a0 -> a};
AvalSmall = FullSimplify[AvalSmall];
BvalSmall = FullSimplify[BvalSmall];
Print["\nSmall sphere approximation:"];
Print["A \[TildeTilde] ", AvalSmall];
Print["B \[TildeTilde] ", BvalSmall];

(* ============================================================== *)
(* PART 6: SELF-CONSISTENT EQUATIONS                               *)
(* ============================================================== *)
(* For constant field inside the small sphere:                     *)
(*   u_i, \[Epsilon]_{mn} are the (unknown) interior values.              *)
(*   u^(0)_i, \[Epsilon]^(0)_{mn} are the background (incident) values.   *)
(*                                                                 *)
(* Isotropic contrast:                                             *)
(*   \[CapitalDelta]c_{jklm} = \[CapitalDelta]\[Lambda] \[Delta]_{jk}\[Delta]_{lm} + \[CapitalDelta]\[Mu](\[Delta]_{jl}\[Delta]_{km} + \[Delta]_{jm}\[Delta]_{kl})    *)
(*                                                                 *)
(* DISPLACEMENT equation (Eq. 8 at sphere center):                 *)
(*   u_i = u^(0)_i + 0 (H integral vanishes)                     *)
(*         + \[Omega]^2 \[CapitalDelta]\[Rho] \[CapitalGamma]_0 u_i                                      *)
(*   => u_i (1 - \[Omega]^2 \[CapitalDelta]\[Rho] \[CapitalGamma]_0) = u^(0)_i                            *)
(*                                                                 *)
(* STRAIN equation (symmetric gradient of Eq. 8):                  *)
(*   \[Epsilon]_{mn} = \[Epsilon]^(0)_{mn}                                          *)
(*     + \[CapitalDelta]c_{jklp} \[Epsilon]_{lp} * (1/2)(\[Integral] G_{mj,kn} + G_{nj,km} d^3x')  *)
(*     + \[Omega]^2 \[CapitalDelta]\[Rho] u_j * (1/2)(\[Integral] G_{mj,n} + G_{nj,m} d^3x')       *)
(*                                                                 *)
(* The last term vanishes (odd integrand).                         *)
(* The second term uses the I_{ijkl} integral computed above.      *)
(* ============================================================== *)

Print["\n--- Part 6: Self-Consistent Equations ---"];

(* --- Displacement equation --- *)
Print["\n** Displacement equation **"];
Print["u_i (1 - \[Omega]^2 \[CapitalDelta]\[Rho] \[CapitalGamma]_0) = u^(0)_i"];
Print[""];
Print["=> u_i = u^(0)_i / (1 - \[Omega]^2 \[CapitalDelta]\[Rho] \[CapitalGamma]_0)"];

(* Define the displacement amplification factor *)
ampU[\[Omega]_, \[CapitalDelta]\[Rho]val_, \[CapitalGamma]0val_] := 1/(1 - \[Omega]^2 \[CapitalDelta]\[Rho]val \[CapitalGamma]0val);
Print["Displacement amplification factor = ", ampU[\[Omega], \[CapitalDelta]\[Rho], \[CapitalGamma]0]];

(* --- Strain equation --- *)
(* The integral \[Integral] (1/2)(G_{mj,kn} + G_{nj,km}) d^3x' contracted with \[CapitalDelta]c_{jklp}\[Epsilon]_{lp} *)
(* We need: (1/2)(I_{mjkn} + I_{njkm}) where I_{ijkl} = A \[Delta]_{ij}\[Delta]_{kl} + B(\[Delta]_{ik}\[Delta]_{jl} + \[Delta]_{il}\[Delta]_{jk}) *)

(* Symmetrized integral tensor: *)
(* S_{mnjk} = (1/2)(I_{mjkn} + I_{njkm}) *)
(* = (1/2){ A\[Delta]_{mj}\[Delta]_{kn} + B(\[Delta]_{mk}\[Delta]_{jn} + \[Delta]_{mn}\[Delta]_{jk}) *)
(*         + A\[Delta]_{nj}\[Delta]_{km} + B(\[Delta]_{nk}\[Delta]_{jm} + \[Delta]_{nm}\[Delta]_{jk}) } *)
(* = (A/2)(\[Delta]_{mj}\[Delta]_{kn} + \[Delta]_{nj}\[Delta]_{km}) + B \[Delta]_{mn}\[Delta]_{jk} + (B/2)(\[Delta]_{mk}\[Delta]_{jn} + \[Delta]_{nk}\[Delta]_{jm}) *)

Print["\n** Strain equation **"];
Print["\[Epsilon]_{mn} = \[Epsilon]^(0)_{mn} + S_{mnjk} \[CapitalDelta]c_{jklp} \[Epsilon]_{lp}"];
Print["where S_{mnjk} = (1/2)(I_{mjkn} + I_{njkm})"];

(* Contract S_{mnjk} with \[CapitalDelta]c_{jklp} = \[CapitalDelta]\[Lambda] \[Delta]_{jk}\[Delta]_{lp} + \[CapitalDelta]\[Mu](\[Delta]_{jl}\[Delta]_{kp} + \[Delta]_{jp}\[Delta]_{kl}) *)
(* Result: T_{mnlp} = S_{mnjk} \[CapitalDelta]c_{jklp} *)
(* We compute this contraction symbolically *)

Print["\nComputing T_{mnlp} = S_{mnjk} * \[CapitalDelta]c_{jklp} ..."];

(* Build T tensor symbolically using index notation *)
(* S_{mnjk} *)
sTensor[m_, n_, j_, k_, Acoeff_, Bcoeff_] :=
  (Acoeff/2) (\[Delta]f[m, j] \[Delta]f[k, n] + \[Delta]f[n, j] \[Delta]f[k, m]) +
  Bcoeff \[Delta]f[m, n] \[Delta]f[j, k] +
  (Bcoeff/2) (\[Delta]f[m, k] \[Delta]f[j, n] + \[Delta]f[n, k] \[Delta]f[j, m]);

(* \[CapitalDelta]c_{jklp} isotropic *)
\[CapitalDelta]cTensor[j_, k_, l_, p_, \[CapitalDelta]\[Lambda]_, \[CapitalDelta]\[Mu]_] :=
  \[CapitalDelta]\[Lambda] \[Delta]f[j, k] \[Delta]f[l, p] + \[CapitalDelta]\[Mu] (\[Delta]f[j, l] \[Delta]f[k, p] + \[Delta]f[j, p] \[Delta]f[k, l]);

(* Contract: T_{mnlp} = Sum_{j,k} S_{mnjk} \[CapitalDelta]c_{jklp} *)
tTensor[m_, n_, l_, p_, Acoeff_, Bcoeff_, \[CapitalDelta]\[Lambda]_, \[CapitalDelta]\[Mu]_] :=
  Sum[sTensor[m, n, j, k, Acoeff, Bcoeff] \[CapitalDelta]cTensor[j, k, l, p, \[CapitalDelta]\[Lambda], \[CapitalDelta]\[Mu]],
    {j, 3}, {k, 3}];

(* Evaluate for specific indices to identify the structure *)
(* T_{mnlp} must have the isotropic form: *)
(* T_{mnlp} = T1 \[Delta]_{mn}\[Delta]_{lp} + T2 (\[Delta]_{ml}\[Delta]_{np} + \[Delta]_{mp}\[Delta]_{nl}) *)

T1111 = Simplify[tTensor[1, 1, 1, 1, AA, BB, \[CapitalDelta]\[Lambda], \[CapitalDelta]\[Mu]]];
T1122 = Simplify[tTensor[1, 1, 2, 2, AA, BB, \[CapitalDelta]\[Lambda], \[CapitalDelta]\[Mu]]];
T1212 = Simplify[tTensor[1, 2, 1, 2, AA, BB, \[CapitalDelta]\[Lambda], \[CapitalDelta]\[Mu]]];

Print["T_{1111} = ", T1111];
Print["T_{1122} = ", T1122];
Print["T_{1212} = ", T1212];

(* From isotropic form: T1111 = T1 + 2*T2, T1122 = T1, T1212 = T2 *)
T1coeff = T1122;
T2coeff = T1212;
Print["\nT1 (volumetric coupling) = ", Simplify[T1coeff]];
Print["T2 (shear coupling) = ", Simplify[T2coeff]];

(* Verification *)
Print["Check: T1 + 2*T2 = T1111? ",
  Simplify[T1coeff + 2 T2coeff - T1111] === 0];

(* ============================================================== *)
(* PART 7: SOLVE THE SELF-CONSISTENT SYSTEM                        *)
(* ============================================================== *)
(* The strain equation in isotropic decomposition:                 *)
(*   \[Theta] = \[Theta]^(0) + (3T1 + 2T2)\[Theta]    (dilatation)                *)
(*   e_{mn} = e^(0)_{mn} + 2 T2 e_{mn}  (deviatoric strain)      *)
(* where \[Theta] = \[Epsilon]_{ii} and e_{mn} = \[Epsilon]_{mn} - (1/3)\[Theta]\[Delta]_{mn}            *)
(* ============================================================== *)

Print["\n--- Part 7: Solution of Self-Consistent System ---"];

(* Dilatation: \[Theta](1 - 3T1 - 2T2) = \[Theta]^(0) *)
(* Deviatoric: e_{mn}(1 - 2T2) = e^(0)_{mn} *)

(* The strain equation: \[Epsilon]_{mn} = \[Epsilon]^(0)_{mn} + T_{mnlp} \[Epsilon]_{lp} *)
(* = \[Epsilon]^(0)_{mn} + T1 \[Delta]_{mn} \[Theta] + T2 (\[Epsilon]_{mn} + \[Epsilon]_{nm}) *)
(* = \[Epsilon]^(0)_{mn} + T1 \[Delta]_{mn} \[Theta] + 2 T2 \[Epsilon]_{mn}  (using symmetry of \[Epsilon]) *)

(* Taking trace: \[Theta] = \[Theta]^(0) + 3 T1 \[Theta] + 2 T2 \[Theta] *)
(* => \[Theta] (1 - 3T1 - 2T2) = \[Theta]^(0) *)

ampTheta = 1/(1 - 3 T1coeff - 2 T2coeff);
ampTheta = Simplify[ampTheta];
Print["Dilatation amplification: \[Theta] = \[Theta]^(0) / (1 - 3T1 - 2T2)"];
Print["  = \[Theta]^(0) * ", ampTheta];

(* Deviatoric part: e_{mn} = e^(0)_{mn} + T1 * 0 + 2 T2 e_{mn} *)
(* (trace of deviatoric is zero, so T1 term vanishes) *)
ampDev = 1/(1 - 2 T2coeff);
ampDev = Simplify[ampDev];
Print["Deviatoric amplification: e_{mn} = e^(0)_{mn} / (1 - 2T2)"];
Print["  = e^(0)_{mn} * ", ampDev];

(* Full strain solution: *)
(* \[Epsilon]_{mn} = (1/3)\[Theta] \[Delta]_{mn} + e_{mn} *)
(*       = (1/3) ampTheta \[Theta]^(0) \[Delta]_{mn} + ampDev e^(0)_{mn} *)
(*       = (1/3) ampTheta \[Theta]^(0) \[Delta]_{mn} + ampDev (\[Epsilon]^(0)_{mn} - (1/3)\[Theta]^(0) \[Delta]_{mn}) *)
(*       = ampDev \[Epsilon]^(0)_{mn} + (1/3)(ampTheta - ampDev) \[Theta]^(0) \[Delta]_{mn} *)

Print["\nFull interior strain:"];
Print["\[Epsilon]_{mn} = ", Simplify[ampDev], " \[Epsilon]^(0)_{mn} + (1/3)(",
  Simplify[ampTheta - ampDev], ") \[Theta]^(0) \[Delta]_{mn}"];

(* Full displacement solution *)
Print["\nFull interior displacement:"];
Print["u_i = u^(0)_i / (1 - \[Omega]^2 \[CapitalDelta]\[Rho] \[CapitalGamma]_0)"];

(* ============================================================== *)
(* PART 8: SCATTERED FIELD AT EXTERIOR POINT                       *)
(* ============================================================== *)
(* For an observation point x far from the sphere:                 *)
(* u^{scat}_i(x) = V_{sphere} * [                                *)
(*   H_{ijk}(x, x_s) \[CapitalDelta]c_{jklm} \[Epsilon]_{lm}                       *)
(*   + \[Omega]^2 G_{ij}(x, x_s) \[CapitalDelta]\[Rho] u_j ]                           *)
(* where V_{sphere} = (4/3)\[Pi] a^3                                    *)
(* and \[Epsilon]_{lm}, u_j are the self-consistent interior values.       *)
(* ============================================================== *)

Print["\n--- Part 8: Scattered Field at Exterior Point ---"];
Print[""];
Print["For observation point x outside the sphere (|x - x_s| >> a):"];
Print[""];
Print["u^{scat}_i(x) = V * { H_{ijk}(x, x_s) [\[CapitalDelta]\[Lambda] \[Theta] \[Delta]_{jk} + 2\[CapitalDelta]\[Mu] \[Epsilon]_{jk}]"];
Print["                     + \[Omega]^2 G_{ij}(x, x_s) \[CapitalDelta]\[Rho] u_j }"];
Print[""];
Print["where V = (4/3)\[Pi] a^3"];
Print["and u_j, \[Epsilon]_{jk}, \[Theta] are the self-consistent interior values above."];

(* Substituting the self-consistent solutions: *)
Print["\nSubstituting self-consistent interior fields:"];
Print[""];
Print["u^{scat}_i(x) = V * {"];
Print["  H_{ijk}(x,x_s) [\[CapitalDelta]\[Lambda] ampTheta \[Theta]^(0) \[Delta]_{jk} + 2\[CapitalDelta]\[Mu] (ampDev \[Epsilon]^(0)_{jk} + (1/3)(ampTheta-ampDev)\[Theta]^(0)\[Delta]_{jk})]"];
Print["  + \[Omega]^2 G_{ij}(x,x_s) \[CapitalDelta]\[Rho] ampU u^(0)_j"];
Print["}"];

(* ============================================================== *)
(* Simplify the stiffness term:                                    *)
(* \[CapitalDelta]c_{jklm}\[Epsilon]_{lm} = \[CapitalDelta]\[Lambda] \[Theta] \[Delta]_{jk} + 2\[CapitalDelta]\[Mu] \[Epsilon]_{jk}                     *)
(* = \[CapitalDelta]\[Lambda] ampTheta \[Theta]^(0) \[Delta]_{jk} + 2\[CapitalDelta]\[Mu] [ampDev \[Epsilon]^(0)_{jk} +       *)
(*   (1/3)(ampTheta - ampDev) \[Theta]^(0) \[Delta]_{jk}]                 *)
(* = [\[CapitalDelta]\[Lambda] ampTheta + (2/3)\[CapitalDelta]\[Mu](ampTheta - ampDev)] \[Theta]^(0) \[Delta]_{jk}  *)
(*   + 2\[CapitalDelta]\[Mu] ampDev \[Epsilon]^(0)_{jk}                                  *)
(* ============================================================== *)

Print["\n--- Effective scattering coefficients ---"];
(* Define effective \[CapitalDelta]\[Lambda]* and \[CapitalDelta]\[Mu]* *)
\[CapitalDelta]\[Lambda]star = Simplify[\[CapitalDelta]\[Lambda] ampTheta + (2/3) \[CapitalDelta]\[Mu] (ampTheta - ampDev)] /.
  {AA -> Aval, BB -> Bval};
\[CapitalDelta]\[Mu]star = Simplify[\[CapitalDelta]\[Mu] ampDev] /. {AA -> Aval, BB -> Bval};

Print["\[CapitalDelta]\[Lambda]* (effective volumetric contrast) = ", Simplify[\[CapitalDelta]\[Lambda]star]];
Print["\[CapitalDelta]\[Mu]* (effective shear contrast)      = ", Simplify[\[CapitalDelta]\[Mu]star]];
Print[""];
Print["Scattered field:"];
Print["u^{scat}_i = V * { H_{ijk}(x,x_s)[\[CapitalDelta]\[Lambda]* \[Theta]^(0) \[Delta]_{jk} + 2 \[CapitalDelta]\[Mu]* \[Epsilon]^(0)_{jk}]"];
Print["              + \[Omega]^2 G_{ij}(x,x_s) \[CapitalDelta]\[Rho]* u^(0)_j }"];
Print[""];
Print["where \[CapitalDelta]\[Rho]* = \[CapitalDelta]\[Rho] / (1 - \[Omega]^2 \[CapitalDelta]\[Rho] \[CapitalGamma]_0)"];

(* ============================================================== *)
(* PART 9: NUMERICAL EXAMPLE                                       *)
(* ============================================================== *)

Print["\n--- Part 9: Numerical Example ---"];
Print["Background: Vp=5000 m/s, Vs=3000 m/s, \[Rho]=2500 kg/m^3"];
Print["Contrast: \[CapitalDelta]\[Lambda]=2 GPa, \[CapitalDelta]\[Mu]=1 GPa, \[CapitalDelta]\[Rho]=100 kg/m^3"];
Print["Sphere radius a=10 m, frequency f=10 Hz"];

(* Background Lam\[EAcute] parameters from Vp, Vs, \[Rho] *)
\[Alpha]num = 5000.0;
\[Beta]num = 3000.0;
\[Rho]num = 2500.0;
\[Omega]num = 2 Pi 10.0;
anum = 10.0;

\[Lambda]bg = \[Rho]num (\[Alpha]num^2 - 2 \[Beta]num^2);
\[Mu]bg = \[Rho]num \[Beta]num^2;
Print["\[Lambda]_bg = ", \[Lambda]bg/10^9, " GPa"];
Print["\[Mu]_bg = ", \[Mu]bg/10^9, " GPa"];

(* Evaluate \[CapitalGamma]_0, A, B numerically *)
\[CapitalGamma]0num = \[CapitalGamma]0Exact /. {a -> anum, \[Omega] -> \[Omega]num, \[Alpha] -> \[Alpha]num, \[Beta] -> \[Beta]num, \[Rho] -> \[Rho]num};
\[CapitalGamma]0num = N[\[CapitalGamma]0num];
Print["\[CapitalGamma]_0 = ", \[CapitalGamma]0num];

Anum = Aval /. {a -> anum, \[Omega] -> \[Omega]num, \[Alpha] -> \[Alpha]num, \[Beta] -> \[Beta]num, \[Rho] -> \[Rho]num,
  \[CapitalDelta]\[Lambda] -> 2.0*10^9, \[CapitalDelta]\[Mu] -> 1.0*10^9};
Bnum = Bval /. {a -> anum, \[Omega] -> \[Omega]num, \[Alpha] -> \[Alpha]num, \[Beta] -> \[Beta]num, \[Rho] -> \[Rho]num,
  \[CapitalDelta]\[Lambda] -> 2.0*10^9, \[CapitalDelta]\[Mu] -> 1.0*10^9};
Print["A = ", N[Anum]];
Print["B = ", N[Bnum]];

(* Amplification factors *)
T1num = T1coeff /. {AA -> N[Anum], BB -> N[Bnum], \[CapitalDelta]\[Lambda] -> 2.0*10^9, \[CapitalDelta]\[Mu] -> 1.0*10^9};
T2num = T2coeff /. {AA -> N[Anum], BB -> N[Bnum], \[CapitalDelta]\[Lambda] -> 2.0*10^9, \[CapitalDelta]\[Mu] -> 1.0*10^9};
Print["T1 = ", N[T1num]];
Print["T2 = ", N[T2num]];

ampUnum = 1/(1 - \[Omega]num^2 * 100.0 * \[CapitalGamma]0num);
ampThetaNum = N[1/(1 - 3 T1num - 2 T2num)];
ampDevNum = N[1/(1 - 2 T2num)];

Print["\nAmplification factors:"];
Print["  Displacement: ", ampUnum];
Print["  Dilatation:   ", ampThetaNum];
Print["  Deviatoric:   ", ampDevNum];

Print["\n====================================================="];
Print["  Computation Complete                                 "];
Print["====================================================="];
