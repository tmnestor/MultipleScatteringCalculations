#!/usr/bin/env wolframscript
(* Generalised deliverable-notebook generator for the Phase-2 .wl files.  Splits each
   executable .wl into Title/Text/Section/Input cells on the long-'=' banner edges (the
   same robust rule as makeIntraPlaneNotebook.wl) and writes the .nb twin.  Generation is
   a faithful, deterministic cell-wrapping of the .wl source; the .wl scripts are already
   self-verified (and Python-cross-checked), so round-trip is spot-checked separately. *)

dir = "/Users/tod/Desktop/MultipleScatteringCalculations/Mathematica/";
bannerEdge[l_] := StringContainsQ[l, Repeated["=", {10, Infinity}]];
cleanText[c_] := StringTrim[StringReplace[c,
    {"(*" -> "", "*)" -> "", Repeated["=", {3, Infinity}] -> "", Repeated["-", {4, Infinity}] -> ""}]];

makeTwin[wlfile_, title_, intro_] := Module[
   {wlpath, nbpath, body, lines, segs = {}, codeBuf = {}, banBuf = {}, inB = False,
    firstBanner = True, emit, toCell, cells},
  wlpath = dir <> wlfile; nbpath = dir <> StringReplace[wlfile, ".wl" -> ".nb"];
  body = StringReplace[Import[wlpath, "Text"], StartOfString ~~ "#!/usr/bin/env wolframscript" ~~ "\n" -> ""];
  lines = StringSplit[body, "\n"];
  emit[] := (If[codeBuf =!= {} && StringTrim[StringRiffle[codeBuf, "\n"]] =!= "",
      AppendTo[segs, {"code", StringTrim[StringRiffle[codeBuf, "\n"]]}]]; codeBuf = {});
  Do[With[{l = lines[[i]]},
    If[bannerEdge[l],
     If[! inB, emit[]; inB = True; banBuf = {l},
      AppendTo[banBuf, l]; AppendTo[segs, {"banner", StringRiffle[banBuf, "\n"]}]; inB = False; banBuf = {}],
     If[inB, AppendTo[banBuf, l], AppendTo[codeBuf, l]]]], {i, Length[lines]}];
  emit[];
  toCell[{"banner", c_}] := If[firstBanner, firstBanner = False; Cell[cleanText[c], "Text"], Cell[cleanText[c], "Section"]];
  toCell[{"code", c_}] := Cell[c, "Input"];
  cells = Join[{Cell[title, "Title"], Cell[intro, "Text"]}, toCell /@ segs];
  Export[nbpath, Notebook[cells]];
  Print["wrote ", nbpath, " : ", Length[cells], " cells (", Count[cells, Cell[_, "Input", ___]],
     " Input, ", Count[cells, Cell[_, "Section", ___]], " Section)"];
];

makeTwin["IntraPlaneFoldyLax.wl", "Phase 2 TB1 : Single-Site T0 + Foldy-Lax Scaffold",
   "Assembles the single-site full-wave Mie T0 in the L/M/N multipole basis from the verified "
   <> "CartesianT0.wl (spheroidal L-N 2x2, toroidal M, n=0 monopole), the collective "
   <> "T_coll = T0 (I - G0 T0)^{-1}, the single-voxel limit, and the monopole-channel collective "
   <> "solve with the Phase-1 scalar G0.  Executable twin: IntraPlaneFoldyLax.wl."];

makeTwin["IntraPlaneVectorTranslation.wl", "Phase 2 TB2 : Vector Translation as an Explicit L/M/N Matrix",
   "Extracts the Phase-0 elastic vector translation as an explicit matrix W^{c'c}_{nu mu,n m}(d): "
   <> "L->L via beta^P, and M,N->M,N by sign-safe projection onto the orthogonal P/B/C vector "
   <> "spherical harmonics.  Executable twin: IntraPlaneVectorTranslation.wl."];

makeTwin["IntraPlaneTwoBody.wl", "Phase 2 (a) : Two-Voxel Direct Foldy-Lax",
   "Two-voxel direct Foldy-Lax with the pairwise vector translation W(d).  Verifies the field "
   <> "reconstruction of W, the isolated limit, the direct Neumann/Born multiple-scattering series "
   <> "against the matrix inverse, the closed-form monopole two-body solve, and the fixed point.  "
   <> "Executable twin: IntraPlaneTwoBody.wl."];

makeTwin["IntraPlaneVectorLattice.wl", "Phase 2 (b) : Lattice-Summed Multi-Channel Vector G0(k_par)",
   "The planar Bloch sum of the pairwise vector translation: L->L closed form (Phase-1 scalar G0 at "
   <> "kappa_P), M,N by damped direct lattice sum.  Verified by vector field reconstruction against "
   <> "a direct damped Bloch sum, geometric convergence, and the collective solve.  Executable twin: "
   <> "IntraPlaneVectorLattice.wl.  Independently cross-checked in Python "
   <> "(cubic_scattering/tests/test_intraplane_collective.py)."];

makeTwin["IntraPlaneCollectiveReciprocity.wl", "Phase 2 (d) : Collective Reciprocity (Symplectic Metric)",
   "Reciprocity of the collective Foldy-Lax operator via the symplectic-J reconciliation.  A single "
   <> "symplectic channel metric D = diag((kS/kP)^{3/2}, I sqrt(n(n+1)), sqrt(n(n+1))) makes BOTH T0 "
   <> "and G0 sigma-symmetric, so T_coll is reciprocal.  Executable twin: "
   <> "IntraPlaneCollectiveReciprocity.wl.  Cross-checked in Python."];

makeTwin["IntraPlaneConvergence.wl", "Phase 2 (c) : Convergence in Multipole Order + Packing Density",
   "Convergence of the planar collective Foldy-Lax solve T_coll = T0 (I - G0 T0)^{-1} in (i) multipole "
   <> "order n and (ii) packing density.  [A] the single-site Mie spectrum ||T0(n)||_F decays "
   <> "super-exponentially (why n-truncation converges); [B] the closed-form P-channel collective "
   <> "monopole converges to <1e-5 relative at every density, while the coupling, spectral radius and "
   <> "conditioning grow as the spheres approach touching (aL -> 2 aa) -- the translation-theorem "
   <> "region-of-validity boundary; [C] one full-vector build confirms the elastic L/M/N collective "
   <> "tracks the same convergence above the lattice floor.  Executable twin: IntraPlaneConvergence.wl.  "
   <> "Independently cross-checked in Python (cubic_scattering/tests/test_intraplane_convergence.py)."];

makeTwin["IntraPlaneDiscretisation.wl", "Phase 2 (e) : Sphere-Packing Discretisation Error",
   "The sphere-packing collective monopole in the Rayleigh limit: a re-parametrisation of the item (c) "
   <> "closed-form collective to Rayleigh ka and the three test contrasts, each Delta->Delta/phi "
   <> "renormalised (phi = pi/6).  Dumps the dimensionless collective renormalisation "
   <> "r_ms = mono_coll/mono_single, the spectral radius and the conditioning over (contrast x ka x aL). "
   <> "Python (cubic_scattering/tests/test_intraplane_discretisation.py) maps r_ms onto the layer "
   <> "effective contrast and compares the sphere-layer R_PP against the space-filling cube slab: "
   <> "Delta->Delta/phi collapses the ~48% dilution error to the irreducible shape + nonlinear-mixing "
   <> "residual (0.4%/4% weak/moderate); the collective is negligible at Rayleigh; the -60% contrast "
   <> "lies beyond the renorm validity floor.  Executable twin: IntraPlaneDiscretisation.wl."];

makeTwin["IntraPlaneRT.wl", "Phase 3a : Layer R/T(p) Projection (thesis symplectic energy normalisation)",
   "Projects the Phase-2 spherical collective T_coll(k_par) onto the PhD thesis Section 3.1 "
   <> "eps-normalised P/SV/SH plane-wave eigenvectors (Eqs. Peigen/SVeigen/SHeigen + epsdef), giving the "
   <> "layer R/T(p) operator (Rd, Ru, Td, Tu 2x2 P-SV + SH scalar) across normal / sub-critical / "
   <> "post-critical slowness.  The incident IS the eps-eigenvector; the scattered field is projected onto "
   <> "the eps-eigenvectors (no slab D, no post-hoc factors).  The full symplectic reciprocity holds at "
   <> "every p: Rd=-Rd^T, Ru=-Ru^T (quadrature ~1e-8) and Tu=Sig.Td.Sig, Sig=diag(1,-1) (exact 1e-19).  "
   <> "Energy balance |R|^2+|T|^2=1 is deferred to Phase 3b (needs the undamped G0).  Executable twin: "
   <> "IntraPlaneRT.wl.  Python cross-check: cubic_scattering/tests/test_intraplane_rt.py."];

makeTwin["IntraPlaneKambe.wl", "Phase 3b cycle 1 : Undamped Multipole Structure Constants D[q,s]",
   "Computes the undamped (kappa real) planar lattice multipole structure constants D[q,s] by "
   <> "multipole-projecting the validated undamped scalar Ewald field onto regular multipoles on a "
   <> "small sphere, instead of per-q analytic Kambe forms.  Extends the Phase-1 TB2 scalar Ewald to a "
   <> "general-z field point (general-z reciprocal half by Poisson summation, reducing to the TB2 z=0 "
   <> "form), subtracts the R=0 self-term, and projects the regular field G = i kappa Sum Dbar[q,s] "
   <> "j_q(kappa r) Y_q^s.  Gates: eta-independence (kappa real), agreement with the damped direct 3D "
   <> "sum, the projection method anchored to the damped direct structure constant, undamped "
   <> "eta-independence, and undamped G0 reciprocity (exact).  Executable twin: IntraPlaneKambe.wl.  "
   <> "Python cross-check: cubic_scattering/tests/test_intraplane_kambe.py."];

makeTwin["IntraPlaneKambeVector.wl", "Phase 3b cycle 2 : Undamped Vector G0(k_par)",
   "Builds the undamped (kappa real) planar vector coupling G0^vec(k_par) in the L/M/N basis by "
   <> "contracting the cycle-1 scalar structure constants D[q,s]: L->L via scalar-Gaunt . D(kappa_P), "
   <> "and M,N via Sum_q coeff_q . D[q,m-mu](kappa_S), where coeff_q are extracted numerically from "
   <> "the validated single-pair vector translation W^{c'c}(d) by angular projection over source "
   <> "directions (no literature transcription).  All gates PASS: coeff reconstruction ~1e-8, M/N "
   <> "contraction == direct damped sum 1.6e-12, L-block == direct beta^P sum 5.5e-15, collective "
   <> "iso-limit=0 / coupling=0.021 / finite, undamped L-block reciprocity = 0 (exact), undamped "
   <> "G0^vec eta-independence ~1.2e-14.  nDim = 25 (Nmax=2).  Executable twin: "
   <> "IntraPlaneKambeVector.wl.  Python cross-check: "
   <> "cubic_scattering/tests/test_intraplane_kambe_vector.py."];

Print["Phase-2/3 notebook twins generated."];
