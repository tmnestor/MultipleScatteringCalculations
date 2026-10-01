#!/usr/bin/env wolframscript
(* ============================================================================
   ContinuumLimit_LayerDefect.wl -- the projection defect of the layer's profile,
   and the long-wave identity that makes it the error of the second-order term.

   THE LAYER.  0 < z < D, contrast  s(z) = 1 + Sin[2 Pi z/D]/2  times a constant,
   cut into n cells of half-width h = D/(2n).  Pi_q is the L2 projection onto the
   Legendre polynomials of degree <= q in each cell.

   THE DEFECT.   D_q = Sum_cells Int (s - Pi_q s)^2 / Int_0^D s^2 .

   THE IDENTITY.  In the long-wave limit the strain-strain kernel of the layer is
   local,  S(z - z') -> -(1/M) delta(z - z'),  and the incident strain is uniform
   across the layer.  The once-scattered strain is then -(dM/M) s(z) e0 exactly.
   A cell holds the field to degree p and the contrast to degree r, so the scheme
   holds  Pi_p (Pi_r s) = Pi_q s,  q = Min[p, r],  and the second-order term is

        exact   proportional to  Int s . s            = |s|^2
        scheme  proportional to  Int (Pi_r s)(Pi_q s) = |Pi_q s|^2 ,

   the last because Pi_q s lies in the range of Pi_r.  The relative error of the
   second-order term is  1 - |Pi_q s|^2 / |s|^2 = D_q .

   CHECKS: (a) <Pi_r s, Pi_q s> = |Pi_q s|^2 for q = Min[p, r], all nine (p, r),
           exactly;  (b) |s|^2 - |Pi_q s|^2 = Int (s - Pi_q s)^2, exactly;
           (c) the defect falls by 4^(q+1) per halving of the cells.
   EXPORT: Mathematica/ContinuumLimit_layer_defect.json, D_q for n = 2..32, read
           by scripts/measure_layer_bases.py.
   ============================================================================ *)

dL = 2;  (* the layer thickness of the Python scripts, in metres *)
s[z_] := 1 + Sin[2 Pi z/dL]/2;
oks = {};
check[name_, ok_] := (AppendTo[oks, ok]; Print[If[ok, "PASS  ", "FAIL  "], name]);

(* projection coefficients of s on LegendreP[c, (z - zc)/h] in the cell centred at zc *)
coef[c_, zc_, h_] := (2 c + 1)/(2 h) Integrate[s[zc + t] LegendreP[c, t/h], {t, -h, h}];
proj[q_, zc_, h_][t_] := Sum[coef[c, zc, h] LegendreP[c, t/h], {c, 0, q}];
cells[n_] := Table[{(j - 1/2) dL/n, dL/(2 n)}, {j, 1, n}];

(* a sum over the cells: f is called with each cell's centre and half-width *)
sumCells[f_, n_] := Total[f @@@ cells[n]];
inner[f_, g_, n_] := sumCells[Function[{zc, h}, Integrate[f[zc, h][t] g[zc, h][t], {t, -h, h}]], n];
norm2 = Integrate[s[z]^2, {z, 0, dL}];
projNorm2[q_, n_] := Simplify[sumCells[Function[{zc, h}, Integrate[proj[q, zc, h][t]^2, {t, -h, h}]], n]];
defect[q_, n_] := Simplify[1 - projNorm2[q, n]/norm2];

(* (a) the scheme's pairing, for every (p, r), at n = 4 *)
pairing = Table[
   With[{q = Min[p, r]},
    Simplify[inner[Function[{zc, h}, proj[r, zc, h]], Function[{zc, h}, proj[q, zc, h]], 4] - projNorm2[q, 4]]],
   {p, 0, 2}, {r, 0, 2}];
check["(a) <Pi_r s, Pi_min(p,r) s> = |Pi_min(p,r) s|^2 for all nine (p, r), n = 4", AllTrue[Flatten[pairing], # === 0 &]];

(* (b) Pythagoras, at n = 4 *)
direct[q_, n_] := Simplify[
   sumCells[Function[{zc, h}, Integrate[(s[zc + t] - proj[q, zc, h][t])^2, {t, -h, h}]], n]/norm2];
check["(b) 1 - |Pi_q s|^2/|s|^2 = Int (s - Pi_q s)^2 / |s|^2, q = 0, 1, 2, n = 4",
  AllTrue[Table[Simplify[defect[q, 4] - direct[q, 4]], {q, 0, 2}], # === 0 &]];

(* (c) the rate, and the export *)
ns = {2, 4, 8, 16, 32};
tab = Table[N[defect[q, n], 30], {q, 0, 2}, {n, ns}];
rates = Table[tab[[q + 1, -2]]/tab[[q + 1, -1]], {q, 0, 2}];
Print["defects (rows q = 0, 1, 2; columns n = ", ns, "):"]; Print[N[tab, 6] // MatrixForm];
Print["ratio n = 16 to 32: ", N[rates, 6], "  (4^(q+1) = ", {4, 16, 64}, ")"];
check["(c) the defect falls by 4^(q+1) per halving, to 3%", AllTrue[Range[0, 2], Abs[rates[[# + 1]]/4^(# + 1) - 1] < 0.03 &]];

Export[FileNameJoin[{DirectoryName[$InputFileName], "ContinuumLimit_layer_defect.json"}],
  <|"n" -> ns, "defect" -> Map[ToString[#, InputForm] &, tab, {2}]|>, "JSON"];
Print[Count[oks, True], "/", Length[oks], " checks passed"];
If[! And @@ oks, Exit[1]];
