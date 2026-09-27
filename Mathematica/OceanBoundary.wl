#!/usr/bin/env wolframscript
(* ============================================================================
   OceanBoundary.wl  --  the layered propagator's layer 0 is an OCEAN.

   WHY.  In a model meant to be uniform, the package's layered 9x9 returned the S
   response multiplied by 1 + e^{i phi} -- an extra wave of EXACTLY unit amplitude
   whose path is twice the distance from the shallower plane up to interface 0 --
   while P agreed at normal incidence and drifted away from it as the lateral
   wavenumber grew.  The hypothesis: the layered solver treats layer 0 as a FLUID
   whatever beta[0] says, so interface 0 is a fluid-solid boundary.  This notebook
   derives that model's response exactly and independently, and compares.

   THE MODEL.  A uniform solid (alpha, beta, rho) for z > 0 below a fluid half-space
   (the same alpha and rho, no shear) for z < 0; z down.  A point force at depth
   z_s > 0, at lateral wavenumber (k_x, 0).  The field in the solid is the whole-
   space response plus downgoing reflected P, SV and SH; in the fluid, an upgoing
   transmitted P.  At z = 0: u_z and sigma_zz continuous, sigma_xz = sigma_yz = 0.

   Spectral whole-space kernel (notebook 2a, validated against the package):
       G^_ij = (i / 2 rho w^2) [ (kS^2 d_ij - kS_i kS_j) e^{i gS |dz|}/gS
                                + kP_i kP_j e^{i gP |dz|}/gP ],
       kW = (gW sign dz, k_x, 0) in (z, x, y), Im gW >= 0.
   Every plane wave a e^{i k.x} has d_j -> i k_j.

   CHECKS (against scripts/ocean_boundary_reference.py's JSON):
     [1] SH: the fluid carries no shear, so SH reflects with coefficient +1 at every
         k_x -- an image source;
     [2] normal incidence: the S factor is 1 + e^{2 i kS min(z, z_s)}, the measured
         "1 + e^{i phi}"; P is transparent (matched impedance);
     [3] oblique P-SV: all 27 force-column entries (u and strain rows), 20 cases
         including evanescent k_x, against the package;
     [4] the reflected P amplitude against k_x: why P drifts with obliquity.
   Coordinates (z, x, y); time e^{-i w t}; SI units.
   ============================================================================ *)

ref = Import["/Users/tod/Desktop/MultipleScatteringCalculations/Mathematica/OceanBoundary_reference.json", "RawJSON"];
cplx[{re_, im_}] := re + I im;
mat[g_] := Map[cplx, g, {2}];
om = ref["omega"]; rho = ref["rho"]; h = ref["h"];
al = cplx[ref["alpha_c"]]; be = cplx[ref["beta_c"]]; alf = cplx[ref["alpha_fluid_c"]];
kP = om/al; kS = om/be; kPf = om/alf;
mu = rho be^2; lam = rho al^2 - 2 mu; lamf = rho alf^2;
sci[x_] := ToString[NumberForm[N[x], 3, NumberFormat -> (If[#3 == "", #1, Row[{#1, "e", #3}]] &)], OutputForm];
pass[b_] := If[TrueQ[b], "PASS", "FAIL"];
relMax[a_, b_] := Max[Abs[Flatten[a - b]]]/Max[Abs[Flatten[b]]];
gam[k_, kx_] := Module[{g = Sqrt[k^2 - kx^2]}, If[Im[g] < 0, -g, g]];

voigt = {{1, 1}, {2, 2}, {3, 3}, {2, 3}, {1, 3}, {1, 2}};  (* e_zz, e_xx, e_yy, 2e_xy, 2e_zy, 2e_zx *)
eng = {1, 1, 1, 2, 2, 2};
(* the 9 rows of a displacement vector u with gradient d_j u_i = grad[[i, j]] *)
rows9[u_, grad_] := Join[u, Table[eng[[v]] (1/2) (grad[[voigt[[v, 1]], voigt[[v, 2]]]] +
       grad[[voigt[[v, 2]], voigt[[v, 1]]]]), {v, 6}]];

(* whole-space plane-wave parts of force column j at receiver depth z from source depth zs:
   a list of {amplitude vector, wavevector} *)
wsWaves[kx_, z_, zs_, j_] := Module[{s = If[z - zs > 0, 1, -1], out = {}},
   Do[Module[{k = w[[1]], g, kv, amp, col},
      g = gam[k, kx]; kv = {g s, kx, 0};
      amp = I/(2 rho om^2) Exp[I g Abs[z - zs]]/g;
      col = If[w[[2]] == "P", amp kv kv[[j]], amp (k^2 UnitVector[3, j] - kv kv[[j]])];
      AppendTo[out, {col, kv}]],
    {w, {{kP, "P"}, {kS, "S"}}}];
   out];
(* the field of a list of plane waves, evaluated at their own point: 9 rows *)
field9[waves_] := Total[rows9[#[[1]], I Outer[Times, #[[1]], #[[2]]]] & /@ waves];

(* stresses sigma_zz, sigma_xz, sigma_yz of plane waves; fluid when mu = 0 *)
stressZ[waves_, lm_, m_] := Total[Module[{a = #[[1]], kv = #[[2]], grad},
      grad = I Outer[Times, a, kv];
      {lm Tr[grad] + 2 m grad[[1, 1]], m (grad[[1, 2]] + grad[[2, 1]]), m (grad[[1, 3]] + grad[[3, 1]])}] & /@ waves];

(* solve the interface for force column j; returns the 9-row response at depth z *)
oceanColumn[kx_, z_, zs_, j_] := Module[
  {gP = gam[kP, kx], gS = gam[kS, kx], gPf = gam[kPf, kx], inc, rp, rs, rh, tf, pw, sol, refl, uzInc, sInc},
  inc = wsWaves[kx, 0, zs, j];                          (* upgoing at the interface *)
  (* unknown waves at z = 0: reflected P, SV, SH (downgoing, solid) and transmitted P (upgoing, fluid) *)
  pw[a_] := {{a[[1]] {gP, kx, 0}, {gP, kx, 0}}, {a[[2]] {kx, -gS, 0}, {gS, kx, 0}},
     {a[[3]] {0, 0, 1}, {gS, kx, 0}}};
  uzInc = Total[#[[1]][[1]] & /@ inc]; sInc = stressZ[inc, lam, mu];
  (* unknowns in units of 1/(rho w^2), and stresses divided by mu: the SI system is otherwise
     scaled over ~20 orders and Solve's row reduction reports it as badly conditioned *)
  Module[{sc = 1/(rho om^2), x1, x2, x3, x4, s},
   s = First@Solve[{
       rho om^2 (uzInc + sc (x1 gP + x2 kx)) == rho om^2 sc x4 (-gPf),                  (* u_z *)
       (sInc[[1]] + stressZ[pw[sc {x1, x2, x3}], lam, mu][[1]])/mu ==
        stressZ[{{sc x4 {-gPf, kx, 0}, {-gPf, kx, 0}}}, lamf, 0][[1]]/mu,               (* sigma_zz *)
       (sInc[[2]] + stressZ[pw[sc {x1, x2, x3}], lam, mu][[2]])/mu == 0,                 (* sigma_xz *)
       (sInc[[3]] + stressZ[pw[sc {x1, x2, x3}], lam, mu][[3]])/mu == 0},                (* sigma_yz *)
      {x1, x2, x3, x4}];
   sol = {rp -> sc x1, rs -> sc x2, rh -> sc x3, tf -> sc x4} /. s];
  refl = pw[{rp, rs, rh} /. sol];
  (* propagate the reflected waves to depth z: phase e^{i g z} *)
  refl = MapThread[{#1[[1]] Exp[I #2 z], #1[[2]]} &, {refl, {gP, gS, gS}}];
  {field9[wsWaves[kx, z, zs, j]] + field9[refl], rp /. sol, rs /. sol, rh /. sol}];

Print["==== OceanBoundary :: the layered propagator below a fluid layer 0 ===="];
Print["  alpha = ", N[al], ", beta = ", N[be], ", rho = ", rho, "; fluid alpha = ", N[alf], "; k_S = ", N[Re[kS]]];

(* ---------------------------------------------------------------------------
   [1] SH: image source with coefficient +1, every k_x
   --------------------------------------------------------------------------- *)
res1 = Table[
   Module[{z = c["rcv"] h, zs = c["src"] h, kx = c["kx"], gS, direct, image, pk},
    gS = gam[kS, kx];
    direct = I/(2 mu gS) Exp[I gS Abs[z - zs]];
    image = I/(2 mu gS) Exp[I gS (z + zs)];
    pk = mat[c["G"]][[3, 3]];
    Abs[pk - (direct + image)]/Abs[direct]],
   {c, ref["cases"]}];
Print["  [1] SH u_y/f_y = direct + IMAGE (coefficient +1), 20 cases: worst ", sci[Max[res1]], " -> ",
  pass[Max[res1] < 10^-6]];

(* ---------------------------------------------------------------------------
   [2] normal incidence: S factor 1 + e^{2 i kS min(z, zs)}; P transparent
   --------------------------------------------------------------------------- *)
res2 = Table[
   Module[{z = c["rcv"] h, zs = c["src"] h, g = mat[c["G"]], sF, pF},
    sF = g[[2, 2]]/(I/(2 mu kS) Exp[I kS Abs[z - zs]]);
    pF = g[[1, 1]]/(I/(2 rho al^2 kP) Exp[I kP Abs[z - zs]]);
    {c["src"], c["rcv"], sF, 1 + Exp[2 I kS Min[z, zs]], pF}],
   {c, Select[ref["cases"], #["kx"] < 10^-3 &]}];
Do[Print["      src ", r[[1]], " rcv ", r[[2]], ": package S factor ", sci[r[[3]]], "   1 + e^{2i kS min(z,zs)} ",
   sci[r[[4]]], "   P factor ", sci[r[[5]]]], {r, res2}];
worst2 = Max[Abs[res2[[All, 3]] - res2[[All, 4]]]];
Print["  [2] normal incidence, S factor predicted: worst ", sci[worst2], " -> ", pass[worst2 < 10^-5]];

(* ---------------------------------------------------------------------------
   [3] oblique P-SV (and SH): all force columns, u and strain rows
   --------------------------------------------------------------------------- *)
res3 = Table[
   Module[{z = c["rcv"] h, zs = c["src"] h, kx = c["kx"], g = mat[c["G"]], mine},
    mine = Transpose[Table[oceanColumn[kx, z, zs, j][[1]], {j, 3}]];
    {c["src"], c["rcv"], kx/Re[kS], relMax[mine, g[[All, 1 ;; 3]]],
     relMax[Transpose[Table[field9[wsWaves[kx, z, zs, j]], {j, 3}]], g[[All, 1 ;; 3]]]}],
   {c, ref["cases"]}];
Print["  [3] force columns (27 entries), ocean model vs package   [whole space alone vs package]:"];
Do[Print["      src ", r[[1]], " rcv ", r[[2]], "  k/kS = ", sci[r[[3]]], ":  ", sci[r[[4]]],
   "   [", sci[r[[5]]], "]"], {r, res3}];
worst3 = Max[res3[[All, 4]]];
Print["      worst ", sci[worst3], " -> ", pass[worst3 < 10^-6]];

(* ---------------------------------------------------------------------------
   [4] the reflected P amplitude for a vertical force, against k_x
   --------------------------------------------------------------------------- *)
Print["  [4] reflected P / reflected SV for a vertical force at 5 h, against k_x/k_S:"];
Do[Module[{o = oceanColumn[f Re[kS], 5 h, 5 h, 1]},
   Print["      ", sci[f], ":  |R_P| ", sci[Abs[o[[2]]]], "   |R_SV| ", sci[Abs[o[[3]]]]]],
  {f, {10^-6, 0.05, 0.3, 0.7}}];
Print["      matched impedance: no P reflection at normal incidence, growing with obliquity through"];
Print["      conversion at the traction-free shear boundary."];

Print["==== OceanBoundary: ",
  If[Max[res1] < 10^-6 && worst2 < 10^-5 && worst3 < 10^-6, "ALL CHECKS PASS", "CHECKS FAILED"], " ===="];
