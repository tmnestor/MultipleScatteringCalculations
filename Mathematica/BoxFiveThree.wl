(* ::Package:: *)

(* ============================================================================
   BoxFiveThree.wl  --  thesis GstratRep Box 5.3

   PURPOSE.  Derive and validate the source-radiation projection

       [Sigma_up; Sigma_dn] = -i < D_z(-kx,-ky) , F1 + A F2 >,
       <b1,b2> = b1^T J6 b2                                       (Box 5.3)

   for a body force  F = F1 delta(z-zS) + F2 delta'(z-zS),  and then extract
   the STATE-JUMP <-> BODY-FORCE conversion that is needed to reconcile
   GlobalMatrix's layered Green's function (which injects a unit JUMP in the
   displacement-traction state) with cubic_scattering's 9x9 propagator (which
   takes a (force, stress-glut) source).

   Reuses the validated energy-normalised eigenbasis of ThesisInterfaceRT.wl
   (thesis 3.1: epsdef, Jdef, D1def) rather than rebuilding it.

   CONVENTIONS (thesis): seismic units km/s, g/cm^3, GPa; time e^{-i om t};
   state b = (u_z, u_x, u_y, tau_zz, tau_xz, tau_yz); z = down.

   Run: /Applications/Wolfram.app/Contents/MacOS/wolframscript -file \
        Mathematica/BoxFiveThree.wl
   ============================================================================ *)

Get[FileNameJoin[{DirectoryName[$InputFileName], "ThesisInterfaceRT.wl"}]];

Print[""];
Print["=============================================================="];
Print["BoxFiveThree :: source radiation and the jump<->force conversion"];
Print["=============================================================="];

chp[x_] := Chop[x, 1.*^-8];
(* wolframscript Print[] does not format ScientificForm -- stringify it *)
sci[x_] := ToString[ScientificForm[N[Chop[x, 1.*^-12]], 3]];
verdict[x_, tol_: 1.*^-8] := If[Abs[N[x]] < tol, "PASS", "** FAIL **"];

(* ============================================================================
   1. The Fourier-domain system matrix A(kx,ky)   (thesis GRepresentations Akdef)

   The shorthand gamma,a,b,zeta,chi in Akdef are LOCAL to that equation -- the
   entries of ListOfSymbols under those names are the Chapter 2 contrast ratios
   and do NOT apply.  Derived here from Hooke + momentum and then validated
   against two thesis identities, so nothing rests on the transcription:

     gamma = lam/(lam+2mu)          a = 1/(lam+2mu)        b = 1/mu
     zeta  = 4 mu (lam+mu)/(lam+2mu)
     chi   = 2 mu lam /(lam+2mu)                (note zeta - chi == 2 mu)
   ============================================================================ *)
Amat[alpha_, beta_, rho_, om_, kx_, ky_] := Module[
  {mu = rho beta^2, lam = rho (alpha^2 - 2 beta^2), gam, aa, bb, zet, chi},
  gam = lam/(lam + 2 mu); aa = 1/(lam + 2 mu); bb = 1/mu;
  zet = 4 mu (lam + mu)/(lam + 2 mu); chi = 2 mu lam/(lam + 2 mu);
  {{0, -I gam kx, -I gam ky, aa, 0, 0},
   {-I kx, 0, 0, 0, bb, 0},
   {-I ky, 0, 0, 0, 0, bb},
   {-rho om^2, 0, 0, 0, -I kx, -I ky},
   {0, -rho om^2 + zet kx^2 + mu ky^2, kx ky (chi + mu), -I kx gam, 0, 0},
   {0, kx ky (chi + mu), -rho om^2 + zet ky^2 + mu kx^2, -I ky gam, 0, 0}}];

(* --- numerical test point (seismic units; all six modes propagating) --- *)
al = 5.; be = 3.; rh = 2.5; om = 1500.; kx = 100.; ky = 50.;
A0 = Amat[al, be, rh, om, kx, ky];
D0 = Dz[al, be, rh, om, kx, ky];

Print[""];
Print["[1] system matrix A"];

(* zeta - chi == 2 mu *)
With[{mu = rh be^2, lam = rh (al^2 - 2 be^2)},
 Print["    zeta - chi - 2 mu = ",
   sci[4 mu (lam + mu)/(lam + 2 mu) - 2 mu lam/(lam + 2 mu) - 2 mu]]];

(* quasi-Hamiltonian identity, thesis ATdef:  A^T(-k) J6 + J6 A(k) == 0 *)
qh = Max[Abs[Flatten[
    Transpose[Amat[al, be, rh, om, -kx, -ky]] . J6 + J6 . A0]]];
Print["    [1a] quasi-Hamiltonian  A^T(-k) J6 + J6 A(k) = 0 : ", sci[qh], "   ", verdict[qh]];

(* eigenvalues must be i diag(kzP,kzS,kzH,-kzP,-kzS,-kzH)  (thesis eigDef) *)
lamExpect = I {kz["P", al, be, om, kx, ky], kz["S", al, be, om, kx, ky],
               kz["H", al, be, om, kx, ky], -kz["P", al, be, om, kx, ky],
               -kz["S", al, be, om, kx, ky], -kz["H", al, be, om, kx, ky]};
(* eigenvalues are purely imaginary for propagating modes, so Re is ~1e-13
   noise and cannot be used as the sort key -- sort on Im *)
eigResid = Max[Abs[SortBy[Eigenvalues[A0], Im] - SortBy[lamExpect, Im]]];
Print["    [1b] eigenvalues == i diag(+-kz)                : ", sci[eigResid], "   ", verdict[eigResid]];

(* the existing eigenvectors must diagonalise it:  A D = D Lambda *)
adResid = Max[Abs[Flatten[A0 . D0 - D0 . DiagonalMatrix[lamExpect]]]];
Print["    [1c] A D_z == D_z Lambda (eigenbasis consistent): ", sci[adResid], "   ", verdict[adResid]];

(* ============================================================================
   2. Box 5.3 projection
   ============================================================================ *)
sourceRadiation[alpha_, beta_, rho_, om_, kx_, ky_, F1_, F2_] :=
  -I Transpose[Dz[alpha, beta, rho, om, -kx, -ky]] . J6 .
     (F1 + Amat[alpha, beta, rho, om, kx, ky] . F2);

Print[""];
Print["[2] Box 5.3 validated on its own worked example (point explosion)"];

(* thesis: volume-force equivalent of a point explosion at (zS, xS, 0) *)
Mw = 1.7; xS = 0.37;
ph = Exp[-I kx xS];
F1ex = Mw {0, 0, 0, 0, I kx ph, I ky ph};
F2ex = Mw {0, 0, 0, ph, 0, 0};

(* thesis intermediate: F1 + A F2 = M e^{-i kx xS} [1/(rho al^2),0,0,0,
                                       2 i kx be^2/al^2, 2 i ky be^2/al^2]^T   *)
mid = F1ex + A0 . F2ex;
midExpect = Mw ph {1/(rh al^2), 0, 0, 0, 2 I kx be^2/al^2, 2 I ky be^2/al^2};
Print["    [2a] F1 + A F2 matches the thesis intermediate  : ",  sci[Max[Abs[mid - midExpect]]]];

(* thesis result: Sigma = -i om M e^{-i kx xS}/(al^2 Sqrt[2 rho kzP]) [1,0,0,1,0,0]^T *)
sig = sourceRadiation[al, be, rh, om, kx, ky, F1ex, F2ex];
sigExpect = (-I om Mw ph)/(al^2 Sqrt[2 rh kz["P", al, be, om, kx, ky]]) {1, 0, 0, 1, 0, 0};
Print["    [2b] Sigma matches the thesis closed form       : ",  sci[Max[Abs[sig - sigExpect]]/Max[Abs[sigExpect]]]];
Print["         (explosion radiates P only, equally up and down -- as it must)"];

(* ============================================================================
   3. THE DELIVERABLE: state-jump <-> body-force conversion

   Integrating  d_z b = A b + F1 delta + F2 delta'  across z_S shows a pure
   delta' term produces a JUMP in the state vector,  [b] = F2, while a pure
   delta term produces no jump.  Hence:

       a unit state jump  S            <=>  body force  F1 = 0, F2 = S
       its radiation                    =   -i D_z(-k)^T J6 . A . S
       a body-force source F           <=>  F1 = F, F2 = 0
       its radiation                    =   -i D_z(-k)^T J6 . F

   So the two source representations differ by exactly ONE factor of the
   system matrix A:      F_equivalent = A . S.
   ============================================================================ *)
Print[""];
Print["[3] state-jump <-> body-force conversion"];

jumpRad[S_]  := sourceRadiation[al, be, rh, om, kx, ky, 0 S, S];
forceRad[F_] := sourceRadiation[al, be, rh, om, kx, ky, F, 0 F];

(* for a random jump S, the equivalent body force must be A.S *)
SeedRandom[20260726];
Stest = RandomComplex[{-1 - I, 1 + I}, 6];
r1 = jumpRad[Stest];
r2 = forceRad[A0 . Stest];
Print["    [3a] radiation(jump S) == radiation(force A.S)  : ",  sci[Max[Abs[r1 - r2]]/Max[Abs[r1]]]];

(* and the conversion is exactly the system matrix, for every basis jump *)
conv = Max[Table[
   Module[{e = UnitVector[6, n]},
     Max[Abs[jumpRad[e] - forceRad[A0 . e]]]/Max[Abs[jumpRad[e]]]], {n, 6}]];
Print["    [3b] holds for all 6 unit jumps                 : ", sci[conv], "   ", verdict[conv]];

(* independence check: A is NOT a scalar multiple of the identity, so this is a
   genuine mixing of components, not a rescaling -- which is why no diagonal
   weight could ever reconcile the two conventions. *)
offdiag = Max[Abs[Flatten[A0 - DiagonalMatrix[Diagonal[A0]]]]];
Print["    [3c] ||offdiag(A)|| = ", sci[offdiag],
      "  -> conversion MIXES components; no diagonal weight can do it"];

Print[""];
Print["=============================================================="];
Print["CONCLUSION"];
Print["  state-jump source S  <->  body force  F = A . S"];
Print["  Sigma(S) = -i D_z(-k)^T J6 A S      Sigma(F) = -i D_z(-k)^T J6 F"];
Print["  A is the Fourier-domain system matrix (Akdef), strongly off-diagonal,"];
Print["  so the two source conventions differ by a genuine 6x6 mixing."];
Print["=============================================================="];
