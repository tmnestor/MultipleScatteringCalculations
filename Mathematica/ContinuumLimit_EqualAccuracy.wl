(* ============================================================================
   ContinuumLimit_EqualAccuracy.wl
   The Galerkin voxel against the gradient hierarchy at equal accuracy, on the exact layer.

   QUESTION.  The Galerkin voxel of degree p (Legendre moments of the field and its strain, tested
   against the same polynomials) converges on the layer at order 2p + 2.  The gradient hierarchy (the
   field a Taylor polynomial about each cell centre to the q-th gradient, the equation and its first q
   derivatives imposed at the centre: point matching) converges, with both contrasts, at order 2 for
   q = 1 and 2, 4 for q = 3 and 4, 6 for q = 5 and 6, 8 for q = 7.  At the same order, how many cells,
   and how many unknowns, does each need for a given error?

   MODEL.  The layer of ContinuumLimit_SecondMoment.wl and ContinuumLimit_GradientHierarchy.wl, which
   are one model (checked below): 2 m thick, alpha = 5000 m/s, rho = 2500 kg/m^3, contrasts
   Delta rho = 100 kg/m^3 and Delta M = Delta lambda + 2 Delta mu = 4e9 Pa, omega = 300 rad/s,
   P at normal incidence, reflection and transmission read at z = -1 and 4.  Each scheme is run by
   its own script's solver, loaded into a context of its own so that their names cannot collide:
   gal`solveLayerP[om, n, p + 1] (2 (p + 1) unknowns a cell, 40 digits) and
   hier`solveHier[om, n, q, drho, dm] (q + 1 unknowns a cell, 60 digits).  The hierarchy's closed-form
   cell integrals are defined there for monomials to degree 6; the degree-7 one that q = 7 needs is
   added here by the same formula.

   MEASURE.  The error of a run is the larger of |discrete/exact - 1| in reflection and transmission.
   The cells needed for a target error are read off each ladder by linear interpolation of log n
   against log(error) between the two grids that bracket it (exact for a power law); a target outside
   a ladder, or one the ladder crosses more than once, is refused, never extrapolated (the rules of
   scripts/equal_accuracy.py).  Unknowns = cells x unknowns a cell.

   Writes ContinuumLimit_equal_accuracy.json.
   ============================================================================ *)

(* both scripts are loaded first, each in its own context and silently: they define clashing globals *)
eqaBase = DirectoryName[$InputFileName];
Block[{Print}, Begin["hier`"]; Get[FileNameJoin[{eqaBase, "ContinuumLimit_GradientHierarchy.wl"}]]; End[]];
Block[{Print}, Begin["gal`"]; Get[FileNameJoin[{eqaBase, "ContinuumLimit_SecondMoment.wl"}]]; End[]];

eqaPass[b_] := If[TrueQ[b], "PASS", "****FAIL****"];
eqaSci[x_] := If[N[x] == 0, "0",
   Module[{e = Floor[Log10[Abs[N[x]]]]}, ToString[NumberForm[N[x]/10^e, {3, 2}]] <> "e" <> ToString[e]]];
eqaOks = {};
eqaChk[b_] := (AppendTo[eqaOks, TrueQ[b]]; eqaPass[b]);
Print["==== ContinuumLimit_EqualAccuracy :: the Galerkin voxel against the gradient hierarchy on the layer ===="];

(* the degree-7 cell integral, by the hierarchy script's own formula *)
Do[hier`fInt[7, s] = Function[{kk, lo, hi}, Evaluate[
     Integrate[Exp[-I s kq xq] xq^7/7!, {xq, loq, hiq}] /. {kq -> kk, loq -> lo, hiq -> hi}]], {s, {1, -1}}];

(* ---------------------------------------------------------------------------
   [1] one model
   --------------------------------------------------------------------------- *)
eqaOm = 300;
same = {hier`al == gal`al, hier`rho == gal`rho, hier`dLayer == gal`dLayer, hier`dRho0 == gal`dRho,
   hier`dM0 == gal`dM, hier`zR == gal`zObsR, hier`zT == gal`zObsT};
exH = hier`exactScat[eqaOm, hier`dRho0, hier`dM0];
exG = N[gal`exactScat[eqaOm], 40];
(* the incident waves differ in amplitude and phase reference, which cancels in |R / T| and in every relative error *)
ratioDiff = Abs[Abs[exH[[1]]/exH[[2]]] - Abs[exG[[1]]/exG[[2]]]]/Abs[exG[[1]]/exG[[2]]];
Print["  [1] the two scripts' layers are one model (constants equal; |R/T| of their exact solutions to ",
  eqaSci[ratioDiff], "): ", eqaChk[And @@ same && ratioDiff < 10^-30]];

(* ---------------------------------------------------------------------------
   [2] the ladders
   --------------------------------------------------------------------------- *)
grids = {1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64};
(* a ladder stops once its error passes 1e-25, well above the 40-digit floor of the Galerkin chain: at
   omega = 300 the eighth-order schemes are already near 1e-15 on one cell *)
floor = 10^-25;
runGal[p_, n_] := Max[Abs[gal`solveLayerP[eqaOm, n, p + 1]/exG - 1]];
runHier[q_, n_] := Max[Abs[hier`solveHier[eqaOm, n, q, hier`dRho0, hier`dM0]/exH - 1]];
ladderOf[run_] := Module[{rows = {}, e},
   Do[e = run[n]; AppendTo[rows, <|"n" -> n, "error" -> N[e, 20]|>]; If[e < floor, Break[]], {n, grids}];
   rows];
(* the apparent order: the mean over the two finest pairs of the ladder (at least three grids; Missing if fewer) *)
orderOf[rows_] := Module[{r = rows, o},
   If[Length[r] < 3, Return[Missing["fewer than three grids"]]];
   o = Table[Log[r[[i, "error"]]/r[[i + 1, "error"]]]/Log[r[[i + 1, "n"]]/r[[i, "n"]]], {i, Length[r] - 1}];
   N[Mean[Take[o, -2]]]];

schemes = {
   <|"scheme" -> "galerkin", "degree" -> 0, "order" -> 2, "perCell" -> 2, "run" -> (runGal[0, #] &)|>,
   <|"scheme" -> "galerkin", "degree" -> 1, "order" -> 4, "perCell" -> 4, "run" -> (runGal[1, #] &)|>,
   <|"scheme" -> "galerkin", "degree" -> 2, "order" -> 6, "perCell" -> 6, "run" -> (runGal[2, #] &)|>,
   <|"scheme" -> "galerkin", "degree" -> 3, "order" -> 8, "perCell" -> 8, "run" -> (runGal[3, #] &)|>,
   <|"scheme" -> "hierarchy", "q" -> 1, "order" -> 2, "perCell" -> 2, "run" -> (runHier[1, #] &)|>,
   <|"scheme" -> "hierarchy", "q" -> 3, "order" -> 4, "perCell" -> 4, "run" -> (runHier[3, #] &)|>,
   <|"scheme" -> "hierarchy", "q" -> 5, "order" -> 6, "perCell" -> 6, "run" -> (runHier[5, #] &)|>,
   <|"scheme" -> "hierarchy", "q" -> 7, "order" -> 8, "perCell" -> 8, "run" -> (runHier[7, #] &)|>};
(* unknowns a cell, from the solvers' own definitions: 2 nb for the Galerkin chain (field and strain, nb Legendre
   functions each), q + 1 Taylor coefficients for the hierarchy *)
schemes = Table[Module[{sc = s0}, sc["perCell"] = If[sc["scheme"] == "galerkin", 2 (sc["degree"] + 1), sc["q"] + 1]; sc],
   {s0, schemes}];

Print["  [2] ladders, error = max over R and T of |discrete/exact - 1|:"];
results = Table[Module[{rows, ord, ok, t0 = AbsoluteTime[]},
    rows = ladderOf[sc["run"]];
    ord = orderOf[rows];
    ok = ! MissingQ[ord] && Abs[ord - sc["order"]] < 0.15;
    Print["    ", sc["scheme"], " ", If[sc["scheme"] == "galerkin", "p = " <> ToString[sc["degree"]], "q = " <> ToString[sc["q"]]],
     " (", sc["perCell"], " unknowns a cell):  ",
     StringRiffle[("n " <> ToString[#n] <> " " <> eqaSci[#error]) & /@ rows, ", "],
     "   order ", NumberForm[ord, 3], " (predicted ", sc["order"], ") ", eqaChk[ok], "   ", Round[AbsoluteTime[] - t0], " s"];
    Join[KeyDrop[sc, "run"], <|"rows" -> rows, "measured_order" -> ord|>]],
   {sc, schemes}];

(* ---------------------------------------------------------------------------
   [3] cells and unknowns at equal error
   --------------------------------------------------------------------------- *)
(* the targets of an order: every decade inside the error range that both of its ladders reach *)
targetsOf[g_, h_] := Module[{lo = Max[Min[#error & /@ g["rows"]], Min[#error & /@ h["rows"]]],
    hi = Min[Max[#error & /@ g["rows"]], Max[#error & /@ h["rows"]]]},
   If[lo > hi, {}, 10^Range[Ceiling[Log10[N[lo]]], Floor[Log10[N[hi]]]]]];
cellsAt[rows_, target_] := Module[{lad = SortBy[rows, #n &], errs, br},
   errs = #error & /@ lad;
   If[target < Min[errs] || target > Max[errs], Return[Missing["outside the ladder"]]];
   Do[If[Abs[r["error"]/target - 1] <= 1/50, Return[{r["n"], "measured"}, Module]], {r, lad}];
   br = Select[Partition[lad, 2, 1], Min[#[[1, "error"]], #[[2, "error"]]] <= target <= Max[#[[1, "error"]], #[[2, "error"]]] &];
   If[Length[br] != 1 || br[[1, 2, "error"]] >= br[[1, 1, "error"]], Return[Missing["not monotone"]]];
   With[{a = br[[1, 1]], b = br[[1, 2]]},
    {N[Exp[Log[a["n"]] + (Log[target] - Log[a["error"]]) (Log[b["n"]] - Log[a["n"]])/(Log[b["error"]] - Log[a["error"]])], 6],
     "interpolated"}]];

Print["  [3] at equal error (cells; unknowns = cells x unknowns a cell):"];
equal = Flatten[Table[Module[{g = SelectFirst[results, #scheme == "galerkin" && #order == ord &],
      h = SelectFirst[results, #scheme == "hierarchy" && #order == ord &]},
     Table[Module[{cg, ch, rec},
     cg = cellsAt[g["rows"], t]; ch = cellsAt[h["rows"], t];
     rec = <|"order" -> ord, "target" -> N[t],
       "galerkin_cells" -> If[MissingQ[cg], Null, cg[[1]]], "galerkin_status" -> If[MissingQ[cg], First[cg], cg[[2]]],
       "galerkin_unknowns" -> If[MissingQ[cg], Null, cg[[1]] g["perCell"]],
       "hierarchy_cells" -> If[MissingQ[ch], Null, ch[[1]]], "hierarchy_status" -> If[MissingQ[ch], First[ch], ch[[2]]],
       "hierarchy_unknowns" -> If[MissingQ[ch], Null, ch[[1]] h["perCell"]]|>;
     Print["    order ", ord, ", error ", eqaSci[t], ":  Galerkin ",
      If[MissingQ[cg], First[cg], ToString[NumberForm[cg[[1]], 3]] <> " cells (" <> cg[[2]] <> ")"],
      "   hierarchy ", If[MissingQ[ch], First[ch], ToString[NumberForm[ch[[1]], 3]] <> " cells (" <> ch[[2]] <> ")"]];
     rec], {t, targetsOf[g, h]}]], {ord, {2, 4, 6, 8}}], 1];

Export[FileNameJoin[{eqaBase, "ContinuumLimit_equal_accuracy.json"}],
  <|"model" -> <|"omega" -> eqaOm, "layer_m" -> gal`dLayer, "alpha" -> gal`al, "rho" -> gal`rho,
     "drho" -> gal`dRho, "dM" -> gal`dM, "zR" -> gal`zObsR, "zT" -> gal`zObsT|>,
   "rows" -> Flatten[Table[Map[Join[KeyDrop[sc, {"rows", "measured_order"}], #] &, sc["rows"]], {sc, results}], 1],
   "orders" -> (KeyDrop[#, "rows"] & /@ results),
   "equal_accuracy" -> equal|>, "JSON"];
Print["  exported ContinuumLimit_equal_accuracy.json"];
Print[If[And @@ eqaOks, "ALL PASS", "SOME FAILED"]];
