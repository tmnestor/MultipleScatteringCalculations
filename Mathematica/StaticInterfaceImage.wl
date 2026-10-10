(* ::Package:: *)
(* StaticInterfaceImage.wl -- the elastostatic image of a point force at a welded interface,
   in the lateral-wavenumber domain.

   Plan: docs/2026-10-10-stratified-reference-legendre-cells-3d.md, option 1 (static images in closed form).

   WHAT.  Medium A (lam_A, mu_A) fills z < 0, medium B (lam_B, mu_B) fills z > 0, welded at z = 0.  A unit
   point force in A at depth zp < 0.  In the frame whose x-axis lies along the lateral wavevector (q, 0),
   every field is  f(z) Exp[I q x],  and the static Navier equations have the solutions
   (a + b z) Exp[+-q z]  (a double eigenvalue).  The field in A is Kelvin's plus a REFLECTED part regular
   as z -> -Infinity; the field in B is TRANSMITTED, regular as z -> +Infinity; u and the traction on
   z-planes are continuous at z = 0.

   The reflected part, GR[q; z, zp] (3 x 3, rows u_k, columns the force direction, index order z, x, y),
   is the large-q (equivalently the omega -> 0) limit of the dynamic reflected kernel of
   cubic_scattering.stratified_march.reflected_spectrum.  Subtracting it leaves a remainder whose
   lateral-wavenumber integral converges for cells touching the interface; the static part itself is to be
   integrated over the cells in closed form.

   OUTPUT.  Mathematica/StaticInterfaceImage.json: the nine entries as InputForm strings, and their values
   at fixed test points for the Python twin
   (scripts/derive_static_interface_image.py) to compare against.

   Run:  wolframscript -file Mathematica/StaticInterfaceImage.wl
*)

ClearAll["Global`*"];
$Assumptions = q > 0 && lamA > 0 && muA > 0 && lamB > 0 && muB > 0 && zp < 0;

(* stresses of u = {ux, uy, uz}[z] Exp[I q x]; no y dependence in the frame *)
stress[u_, lam_, mu_] := Module[{ux = u[[1]], uy = u[[2]], uz = u[[3]]},
  <|"zz" -> (lam + 2 mu) D[uz, z] + lam I q ux,
    "xz" -> mu (D[ux, z] + I q uz),
    "yz" -> mu D[uy, z],
    "xx" -> lam D[uz, z] + (lam + 2 mu) I q ux,
    "xy" -> mu I q uy|>];
navier[u_, lam_, mu_] := Module[{s = stress[u, lam, mu]},
  {I q s["xx"] + D[s["xz"], z], I q s["xy"] + D[s["yz"], z], I q s["xz"] + D[s["zz"], z]}];
traction[u_, lam_, mu_] := Module[{s = stress[u, lam, mu]}, {s["xz"], s["yz"], s["zz"]}];

(* the general solution ~ Exp[s q z]: two P-SV solutions and one SH, as {ux, uy, uz} *)
basis[lam_, mu_, s_] := Module[{a, b, c, d, u, eqs, sol},
  u = {(a + b z) Exp[s q z], 0, (c + d z) Exp[s q z]};
  eqs = Simplify[navier[u, lam, mu] Exp[-s q z]];
  sol = First@Solve[Flatten[CoefficientList[#, z] & /@ eqs] == 0, {c, d}];
  Join[{u /. sol /. {a -> 1, b -> 0}, u /. sol /. {a -> 0, b -> 1}}, {{0, Exp[s q z], 0}}]];

combo[bs_, cs_] := cs . bs;

reflected[f_] := Module[{F, cb, ca, cr, ct, below, above, kel, uK, uR, uT, eqs, sol},
  F = UnitVector[3, f];  (* force direction in the order x, y, z *)
  cb = Array[cbb, 3]; ca = Array[caa, 3]; cr = Array[crr, 3]; ct = Array[ctt, 3];
  below = combo[basis[lamA, muA, -1], cb]; above = combo[basis[lamA, muA, 1], ca];
  eqs = Join[(below - above /. z -> zp),
    (traction[below, lamA, muA] - traction[above, lamA, muA] /. z -> zp) + F];
  kel = First@Solve[eqs == 0, Join[cb, ca]];
  uK = below /. kel;
  uR = combo[basis[lamA, muA, 1], cr]; uT = combo[basis[lamB, muB, -1], ct];
  eqs = Join[(uK + uR - uT /. z -> 0),
    (traction[uK, lamA, muA] + traction[uR, lamA, muA] - traction[uT, lamB, muB] /. z -> 0)];
  sol = First@Solve[eqs == 0, Join[cr, ct]];
  Simplify[uR /. sol]];

(* columns by force direction x, y, z; rows ux, uy, uz.  Reorder both to the package order z, x, y. *)
gr = Transpose[reflected /@ {1, 2, 3}];
perm = {3, 1, 2};
GR = gr[[perm, perm]];

(* checks *)
shExpected = (muA - muB) Exp[q (z + zp)]/(2 muA q (muA + muB));
Print["SH entry equals the scalar image: ", Simplify[GR[[3, 3]] - shExpected] === 0];
Print["no reflection for identical media: ", Simplify[GR /. {lamB -> lamA, muB -> muA}] === ConstantArray[0, {3, 3}]];

(* test points: the Python twin evaluates the same entries *)
tests = {{2.3, -0.4, -0.9, 1.7, 1.1, 2.6, 1.9}, {11.0, -0.05, -0.12, 0.6, 0.9, 1.4, 2.2}};
vals = Table[N[GR /. Thread[{q, z, zp, lamA, muA, lamB, muB} -> t], 20], {t, tests}];
Export[FileNameJoin[{DirectoryName[$InputFileName], "StaticInterfaceImage.json"}],
  <|"order" -> "rows u_(z,x,y), columns force (z,x,y); frame x along the lateral wavevector",
    "entries" -> Map[ToString[#, InputForm] &, GR, {2}],
    "test_args" -> {"q", "z", "zp", "lamA", "muA", "lamB", "muB"},
    "tests" -> tests,
    "values_re" -> Re[vals], "values_im" -> Im[vals]|>, "JSON"];
Print["written StaticInterfaceImage.json"];
