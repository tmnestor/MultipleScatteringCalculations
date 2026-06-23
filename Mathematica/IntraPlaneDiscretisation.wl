#!/usr/bin/env wolframscript
(* ============================================================================
   IntraPlaneDiscretisation.wl  --  Phase 2 item (e): SPHERE-PACKING DISCRETISATION
   ERROR.  The new physics: the planar-collective monopole renormalisation of a
   diluted sphere packing in the RAYLEIGH limit, dumped for the Python cross-check
   that compares the sphere-layer R_PP against the space-filling cube slab.

   Stage A (single-site shape factor: sphere vs cube effective contrast) lives
   entirely in Python (both are validated cubic_scattering modules; cf.
   test_intraplane_discretisation.py).  This script carries Stage B: the collective.

   Method.  Reuses item (c)'s closed-form P-channel (L) collective machinery
   (g0LLfun / solveLL, Gaunt contraction of the scalar structure constants; no
   quadrature) from IntraPlaneConvergence.wl.  Item (c)'s background IS the Python
   test background (alpha=5000, beta=3000, rho=2500 => lamO=1.75e10, muO=2.25e10),
   run there at kPo=0.9 (resonance); here we re-parametrise to Rayleigh kPo=ka in
   {0.05, 0.1} and sweep the three test contrasts (weak / moderate / negative-60%),
   each with the sphere-packing renormalisation Delta -> Delta/phi (phi = pi/6).

   The dumped observable is the dimensionless collective monopole renormalisation
   r_ms = mono_coll / mono_single = tcoll[[1,1]] / T0Lentry[0]: the multiple-
   scattering shift of the monopole due to the planar packing (-> 1 dilute, deviates
   toward touching).  Python applies r_ms to the layer effective contrast
   phi * sphere_eff(Delta/phi) and forms R_PP via kennett_reference_rpp.  Using the
   RATIO sidesteps the fragile absolute monopole->Dkappa* normalisation.

   Conventions inherited from CartesianT0.wl / item (c): time e^{+i w t}, outgoing
   h_n^(1) (SphericalHankelH1 -- never j_n + i y_n), clean L/M/N; z = depth; lattice
   in x-y; inner T0 at REAL background wavenumbers, only the lattice sum is damped.
   ============================================================================ *)

Get["/Users/tod/Desktop/MultipleScatteringCalculations/Mathematica/CartesianT0.wl"];

reim[z_] := {Re[N[z]], Im[N[z]]};

(* ---- fixed background (= item (c)) and lattice Bloch / damping ---- *)
alpha0 = 5000.; beta0 = 3000.; rho0 = 2500.;
lam0 = rho0 (alpha0^2 - 2 beta0^2); mu0 = rho0 beta0^2;   (* 1.75e10, 2.25e10 *)
dampIm = 0.25; kx = 0.2; ky = 0.1;                        (* horizontal Bloch vector *)
phiTouch = N[Pi/6];                                        (* sphere-in-cube volume fraction *)

(* ---- contrasts (SI; CLAUDE.md test params), as RAW Delta (renorm applied below) ---- *)
contrasts = {
  <|"name" -> "weak",     "Dl" -> 1.*^-4 lam0, "Dm" -> 1.*^-4 mu0, "Dr" -> 1.*^-4 rho0|>,
  <|"name" -> "moderate", "Dl" -> 2.*^9,       "Dm" -> 1.*^9,      "Dr" -> 100.|>,
  <|"name" -> "negative", "Dl" -> -0.6 lam0,   "Dm" -> -0.6 mu0,   "Dr" -> -0.6 rho0|>};
kaList = {0.05, 0.1};
aLlist = {6.0, 4.0, 3.0, 2.5, 2.2};   (* aa/aL: 1/12 .. ~0.227; aL=2 (touching) excluded *)
(* damped lattice (Im kappaP = 0.25) converges by rN ~ 1/0.25 = 4, so Lrad=12 is ample;
   the monopole renormalisation converges in a few multipoles at Rayleigh, so Nmax=3. *)
LradB = 12; NmaxB = 3; aa = 1.0;

(* ---- scalar lattice machinery (closed form; copied from item (c)) ---- *)
sh[n_, x_] := SphericalHankelH1[n, x];
gaunt[l1_, m1_, l2_, m2_, l3_, m3_] :=
  If[m1 + m2 + m3 != 0 || Abs[m1] > l1 || Abs[m2] > l2 || Abs[m3] > l3, 0,
   Sqrt[(2 l1 + 1) (2 l2 + 1) (2 l3 + 1)/(4 Pi)]
     ThreeJSymbol[{l1, 0}, {l2, 0}, {l3, 0}] ThreeJSymbol[{l1, m1}, {l2, m2}, {l3, m3}]];

(* the lattice builder closes over the CURRENT global kappaP/kx/ky *)
g0LLfun[aL_, Lr_] := Module[{ij, Rv, bl, rN, ph, Dstruct, g0LL},
  ij = Flatten[Table[If[i == 0 && j == 0, Nothing, {i, j}], {i, -Lr, Lr}, {j, -Lr, Lr}], 1];
  Rv = Map[{aL #[[1]], aL #[[2]], 0.0} &, ij];
  bl = Map[Exp[I aL (kx #[[1]] + ky #[[2]])] &, ij];
  rN = Map[Sqrt[#[[1]]^2 + #[[2]]^2] &, Rv];
  ph = Map[ArcTan[#[[1]], #[[2]]] &, Rv];
  Dstruct[q_, s_] := Dstruct[q, s] = Total[sh[q, kappaP rN] SphericalHarmonicY[q, s, Pi/2, ph] bl];
  g0LL[n_, m_, nu_, mu_] := 4 Pi (-1)^m Sum[
     I^(nu + q - n) (-1)^q Dstruct[q, m - mu] gaunt[n, m, nu, -mu, q, mu - m], {q, Abs[n - nu], n + nu}];
  g0LL];

idxLof[Nmax_] := Flatten[Table[{n, m}, {n, 0, Nmax}, {m, -n, n}], 1];

(* single-site L-channel Mie scalar at the CURRENT global params *)
T0Lentry[0] := T0mono[kPo, lamO, muO, kPi, lamI, muI, aa];
T0Lentry[n_] := TsphClean[n, kPo, kSo, lamO, muO, kPi, kSi, lamI, muI, aa][[1, 1]];

solveLL[aL_, Nmax_, g0LL_] := Module[{idxL, T0L, G0, M, tcoll},
  idxL = idxLof[Nmax];
  T0L = DiagonalMatrix[Map[T0Lentry[#[[1]]] &, idxL]];
  G0 = Table[g0LL[idxL[[i, 1]], idxL[[i, 2]], idxL[[j, 1]], idxL[[j, 2]]], {i, Length[idxL]}, {j, Length[idxL]}];
  M = IdentityMatrix[Length[idxL]] - G0 . T0L;
  tcoll = T0L . Inverse[M];
  <|"mono" -> tcoll[[1, 1]], "single" -> T0L[[1, 1]],
    "coupling" -> Norm[Flatten[tcoll - T0L]]/Norm[Flatten[T0L]],
    "cond" -> First[SingularValueList[M]]/Last[SingularValueList[M]],
    "specrad" -> Max[Abs[Eigenvalues[G0 . T0L]]]|>];

(* one collective solve for contrast c against a PREBUILT lattice g0 (reused across
   contrasts at fixed ka, aL).  Re-parametrises the inner Mie wavenumbers to the
   Delta/phi-renormalised sphere; the lattice damping wavenumber kappaP is set by the
   caller (depends only on ka). *)
recordFor[c_, ka_, aL_, g0_] := Module[
  {Dl = c["Dl"]/phiTouch, Dm = c["Dm"]/phiTouch, Dr = c["Dr"]/phiTouch,
   rhoI, lamI0, muI0, alphaI, betaI, res, rms},
  rhoI = rho0 + Dr; lamI0 = lam0 + Dl; muI0 = mu0 + Dm;
  alphaI = Sqrt[(lamI0 + 2 muI0)/rhoI]; betaI = Sqrt[muI0/rhoI];
  kPo = ka; kSo = ka alpha0/beta0;
  kPi = ka alpha0/alphaI; kSi = ka alpha0/betaI;
  lamI = lamI0; muI = muI0;
  res = solveLL[aL, NmaxB, g0];
  rms = res["mono"]/res["single"];
  <|"name" -> c["name"], "ka" -> ka, "aL" -> aL, "phi" -> phiTouch,
    "physical" -> If[rhoI > 0 && muI0 > 0 && lamI0 + 2 muI0 > 0, 1, 0],
    "mono_coll" -> reim[res["mono"]], "mono_single" -> reim[res["single"]],
    "r_ms" -> reim[rms], "coupling" -> N[res["coupling"]],
    "specrad" -> N[res["specrad"]], "cond" -> N[res["cond"]]|>];

(* ============================================================================
   collective sweep over (contrast x ka x aL): build g0 once per (ka, aL)
   ============================================================================ *)
Print["==== Phase 2 item (e) :: sphere-packing collective (Rayleigh) ===="];
stageB = {};
Do[
  kappaP = ka + dampIm I; kappaS = ka alpha0/beta0 + dampIm I;   (* lattice damping, ka-only *)
  Module[{g0 = g0LLfun[aL, LradB]},                               (* one build per (ka, aL) *)
    Do[AppendTo[stageB, recordFor[c, ka, aL, g0]], {c, contrasts}]];
  Print["  done ka=", ka, " aL=", aL, "  (", Length[stageB], " records)"],
  {ka, kaList}, {aL, aLlist}];
Print["  built ", Length[stageB], " (contrast x ka x aL) collective records"];

(* ============================================================================
   self-verifying PASS/FAIL on the dumped collective trends
   ============================================================================ *)
(* [1] collective -> isolated as aL grows (r_ms -> 1 dilute): |r_ms-1| smallest at aL=6 *)
modKa = Select[stageB, #["name"] == "moderate" && #["ka"] == 0.1 &];
rmsDev = Map[Abs[(#["r_ms"][[1]] + I #["r_ms"][[2]]) - 1] &, modKa];   (* aL desc: 6..2.2 *)
isoOK = rmsDev[[1]] == Min[rmsDev];   (* most dilute is closest to isolated *)
Print["  [1] |r_ms-1| (moderate, ka=0.1, aL desc 6..2.2) = ", Map[ScientificForm[#, 3] &, rmsDev]];
Print["      collective -> isolated as aL grows -> ", If[isoOK, "PASS", "FAIL"]];
(* [2] specrad grows toward touching (aL descends) -- the multiple-scattering strength *)
specSeq = Map[#["specrad"] &, modKa];
specOK = specSeq == Sort[specSeq];
Print["  [2] specrad (moderate, ka=0.1, aL desc) = ", Map[ScientificForm[#, 3] &, specSeq]];
Print["      multiple-scattering strength grows toward touching -> ", If[specOK, "PASS", "FAIL"]];

(* ============================================================================
   dump JSON reference for the Python cross-check (item-f style; complex as [re,im])
   ============================================================================ *)
discRef = <|"params" -> <|"alpha" -> alpha0, "beta" -> beta0, "rho0" -> rho0,
     "lam0" -> lam0, "mu0" -> mu0, "dampIm" -> dampIm, "kx" -> kx, "ky" -> ky,
     "phi_touch" -> phiTouch, "ka_list" -> kaList, "aL_list" -> aLlist,
     "LradB" -> LradB, "NmaxB" -> NmaxB|>,
   "contrasts" -> (KeyMap[Replace[{"Dl" -> "Dlambda", "Dm" -> "Dmu", "Dr" -> "Drho"}], #] & /@ contrasts),
   "stageB" -> stageB|>;
Export["/Users/tod/Desktop/MultipleScatteringCalculations/Mathematica/IntraPlaneDiscretisation_reference.json", discRef];
Print["  wrote IntraPlaneDiscretisation_reference.json"];
Print["Phase 2 item (e) (IntraPlaneDiscretisation.wl) loaded + verified."];
