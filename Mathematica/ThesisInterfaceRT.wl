#!/usr/bin/env wolframscript
(* ============================================================================
   ThesisInterfaceRT.wl  --  PhD thesis Section 3.1 energy-normalised
   displacement-traction eigenbasis D_z and canonical symplectic J6 for two
   half-spaces, plus the plane-interface R/T scaffold.

   Faithful to GRepresentations.tex (Eqs. Peigen / SVeigen / SHeigen + epsdef +
   the symplectic identity dinv2).  Self-contained; no Get[].

   CONVENTIONS (thesis Section 3.1):
     - state vector  b = (u_z, u_x, u_y, t_z, t_x, t_y):
       3 displacement components, then 3 traction-on-the-z-plane components.
     - "+" = downgoing, "-" = upgoing; column order of D_z is (+P,+S,+H, -P,-S,-H).
     - TIME convention  e^{-i w t}  (the THESIS convention, GRepresentations l.29 -
       NOT the project's e^{+i w t} used by CartesianT0 / the Phase-3 code).
     - traction = physical stress on the +z plane (sigma_zz, sigma_zx, sigma_zy);
       lambda = rho(alpha^2 - 2 beta^2), mu = rho beta^2.
   ============================================================================ *)

(* ============================================================================
   1. Kinematics: velocity per mode, K-hat, vertical wavenumber, energy-norm eps
   ============================================================================ *)
vel[c_, alpha_, beta_] := Switch[c, "P", alpha, "S", beta, "H", beta];
Khat[c_, alpha_, beta_, om_, ky_] := Sqrt[(om/vel[c, alpha, beta])^2 - ky^2];      (* Kdef    *)
kz[c_, alpha_, beta_, om_, kx_, ky_] := Sqrt[Khat[c, alpha, beta, om, ky]^2 - kx^2]; (* kzcDef  *)
epsP[rho_, om_, kzP_] := 1/Sqrt[2 rho om^2 kzP];                                     (* epsdef  *)
epsS[beta_, rho_, om_, KS_, kzS_] := om/(beta KS Sqrt[2 rho om^2 kzS]);
epsH[rho_, om_, KH_, kzH_] := 1/(KH Sqrt[2 rho om^2 kzH]);

(* ============================================================================
   2. The three energy-normalised eigenvectors (s = +1 down, s = -1 up).
      Transcribed from Peigen / SVeigen / SHeigen; the thesis "+-/-+" reduce to
      the factor (-s) and "+/-" to (+s) with the downgoing (+) top sign.
   ============================================================================ *)
eigP[s_, alpha_, beta_, rho_, om_, kx_, ky_] := Module[
   {b2 = beta^2, kzc = kz["P", alpha, beta, om, kx, ky], eP},
   eP = epsP[rho, om, kzc];
   eP {s I kzc, I kx, I ky,
      rho (2 b2 kx^2 + 2 b2 ky^2 - om^2),
      -s 2 rho b2 kx kzc,
      -s 2 rho b2 ky kzc}];

eigS[s_, alpha_, beta_, rho_, om_, kx_, ky_] := Module[
   {b2 = beta^2, KS = Khat["S", alpha, beta, om, ky], kzc = kz["S", alpha, beta, om, kx, ky], eS},
   eS = epsS[beta, rho, om, KS, kzc];
   eS {I kx, -s I kzc, 0,
      -s 2 rho b2 kx kzc,
      rho (om^2 - 2 b2 kx^2 - b2 ky^2),
      -rho b2 kx ky}];

eigH[s_, alpha_, beta_, rho_, om_, kx_, ky_] := Module[
   {b2 = beta^2, KH = Khat["H", alpha, beta, om, ky], kzc = kz["H", alpha, beta, om, kx, ky], eH},
   eH = epsH[rho, om, KH, kzc];
   eH {-s ky kzc, -kx ky, KH^2,
      2 I ky rho (b2 kx^2 + b2 ky^2 - om^2),
      -s 2 I rho b2 kx ky kzc,
      s I kzc rho (om^2 - 2 b2 ky^2)}];

(* ============================================================================
   3. Eigen-matrix D_z = [ +P +S +H  -P -S -H ] (columns) and symplectic J6
   ============================================================================ *)
Dz[alpha_, beta_, rho_, om_, kx_, ky_] := Transpose[{
    eigP[ 1, alpha, beta, rho, om, kx, ky], eigS[ 1, alpha, beta, rho, om, kx, ky], eigH[ 1, alpha, beta, rho, om, kx, ky],
    eigP[-1, alpha, beta, rho, om, kx, ky], eigS[-1, alpha, beta, rho, om, kx, ky], eigH[-1, alpha, beta, rho, om, kx, ky]}];

Id3 = IdentityMatrix[3]; Zero3 = ConstantArray[0, {3, 3}];
J6 = ArrayFlatten[{{Zero3, Id3}, {-Id3, Zero3}}];                                   (* Jdef *)

(* symplectic inverse, Eq. D1def:  D_z^{-1}(k) = -i J6 D_z^T(-k) J6 *)
DzInv[alpha_, beta_, rho_, om_, kx_, ky_] :=
  -I J6 . Transpose[Dz[alpha, beta, rho, om, -kx, -ky]] . J6;

(* ============================================================================
   4. SELF-CHECKS.  Numeric instantiation: two propagating half-spaces, all six
      modes real (sub-critical) in BOTH media so every column is a true plane wave.
   ============================================================================ *)
chop[x_] := Chop[x, 1.*^-9];
(* SEISMIC UNITS: velocity km/s, density g/cm^3 -> moduli rho v^2 in GPa.  In these units D_z is
   well-conditioned (cond(D_z) = rho*omega*v ~ 1.9e4) and every check below holds at MACHINE
   precision.  In SI (m/s, kg/m^3, Pa) the SAME cond is ~1.9e10 -- a pure UNITS artifact of
   stacking displacement (length) over traction (stress), NOT a defect of the representation
   (in nondimensional units cond = 1.66).  The thesis never inverts D_z naively anyway: it uses
   the symplectic inverse D1def, and the symplectic identity holds at 1e-16 in every unit system. *)
(* half-space 1 (above) and half-space 2 (below) *)
a1 = 5.; b1 = 3.; r1 = 2.5;            (* km/s, km/s, g/cm^3 *)
a2 = 5.5; b2v = 3.3; r2 = 2.7;
om0 = 1500.; kx0 = 100.; ky0 = 50.;    (* rad/s, rad/km; kx,ky < om/alpha = 300 -> all modes propagate *)

Print["==== ThesisInterfaceRT :: half-space (u,t) eigenbasis + J6 ===="];
Print["  media: HS1 {a,b,r}=", {a1, b1, r1}, "  HS2=", {a2, b2v, r2}, "  (om,kx,ky)=", {om0, kx0, ky0}];

(* --- [1] energy-normalisation / symplectic identity  (J6 D_z(-k))^T D_z(k) == i J6  (dinv2) --- *)
sympResid[al_, be_, rh_] := Module[{Dk = Dz[al, be, rh, om0, kx0, ky0], Dmk = Dz[al, be, rh, om0, -kx0, -ky0]},
   Max[Abs[Flatten[Transpose[J6 . Dmk] . Dk - I J6]]]];
s1 = sympResid[a1, b1, r1]; s2 = sympResid[a2, b2v, r2];
Print["  [1] symplectic identity (J6 D(-k))^T D(k)=i J6 : HS1 = ", ScientificForm[s1, 3],
   ", HS2 = ", ScientificForm[s2, 3], " -> ", If[Max[s1, s2] < 1.*^-7, "PASS", "FAIL"]];

(* --- [1b] inverse consistency: D_z DzInv == I6 --- *)
invResid[al_, be_, rh_] := Max[Abs[Flatten[
    Dz[al, be, rh, om0, kx0, ky0] . DzInv[al, be, rh, om0, kx0, ky0] - IdentityMatrix[6]]]];
Print["  [1b] D_z . D_z^{-1} == I6 : HS1 = ", ScientificForm[invResid[a1, b1, r1], 3],
   ", HS2 = ", ScientificForm[invResid[a2, b2v, r2], 3], " -> ",
   If[Max[invResid[a1, b1, r1], invResid[a2, b2v, r2]] < 1.*^-7, "PASS", "FAIL"]];

(* --- [2] traction consistency: rows 4-6 reproduced from rows 1-3 via Hooke's law
        for a plane wave with d/dx=i kx, d/dy=i ky, d/dz = i*(s kz_c).  A-matrix-free. --- *)
modeList = {{"P", 1}, {"S", 1}, {"H", 1}, {"P", -1}, {"S", -1}, {"H", -1}};
tractionFromU[uvec_, lam_, mu_, kx_, ky_, kzsig_] := Module[{ux = uvec[[2]], uy = uvec[[3]], uz = uvec[[1]]},
   {lam (I kx ux + I ky uy + I kzsig uz) + 2 mu (I kzsig uz),  (* sigma_zz *)
    mu (I kzsig ux + I kx uz),                                  (* sigma_zx *)
    mu (I kzsig uy + I ky uz)}];                                (* sigma_zy *)
hookeResid[al_, be_, rh_] := Module[{lam = rh (al^2 - 2 be^2), mu = rh be^2, D = Dz[al, be, rh, om0, kx0, ky0], res = {}},
   Do[With[{c = modeList[[j, 1]], s = modeList[[j, 2]], col = D[[All, j]]},
      With[{kzsig = s kz[c, al, be, om0, kx0, ky0], upred = col[[1 ;; 3]], tact = col[[4 ;; 6]]},
       AppendTo[res, Max[Abs[tractionFromU[upred, lam, mu, kx0, ky0, kzsig] - tact]]]]], {j, 6}];
   Max[res]];
Print["  [2] traction == Hooke(displacement) : HS1 = ", ScientificForm[hookeResid[a1, b1, r1], 3],
   ", HS2 = ", ScientificForm[hookeResid[a2, b2v, r2], 3], " -> ",
   If[Max[hookeResid[a1, b1, r1], hookeResid[a2, b2v, r2]] < 1.*^-7, "PASS", "FAIL"]];

(* ============================================================================
   5. INTERFACE R/T via the SYMPLECTIC INVERSE: one 3x3 inverse, no 6x6 solve.
      Continuity of the full 6-vector b across z = 0:
        D1 . {a_inc(down); a_refl(up)} = D2 . {a_trans(down); 0(up)}.
      Form the interface map  Q = D1^{-1} D2  with D1^{-1} from the symplectic
      identity D1def (Q = -i J6 D1^T(-k) J6 . D2 -- NO 6x6 elimination), then
      partition Q into 3x3 blocks [[Q11,Q12],[Q21,Q22]].  Since a2 has no upgoing
      part, (a_inc; a_refl) = Q.(a_trans; 0) gives
        a_inc  = Q11 . a_trans      a_refl = Q21 . a_trans
      =>  T = Q11^{-1}   (the ONLY inverse, 3x3),   R = Q21 . T.
      R (down-in -> up-out), T (down-in -> down-out); columns = incident mode P,S,H.
   ============================================================================ *)
Qmat = DzInv[a1, b1, r1, om0, kx0, ky0] . Dz[a2, b2v, r2, om0, kx0, ky0];  (* D1^{-1} via J, no solve *)
Q11 = Qmat[[1 ;; 3, 1 ;; 3]]; Q21 = Qmat[[4 ;; 6, 1 ;; 3]];
Tmat = Inverse[Q11];          (* 3x3 inverse -- the only one *)
Rmat = Q21 . Tmat;
Print["  --- interface R/T (energy-normalised eigenbasis, mode order P,S,H) ---"];
Print["  R (down-in -> up-out) ="]; Print["    ", MatrixForm[chop[Rmat]]];
Print["  T (down-in -> down-out) ="]; Print["    ", MatrixForm[chop[Tmat]]];

(* --- [3] energy/flux conservation: with unit-flux (energy-normalised) modes the
        per-incident-column power balance is  Sum|R|^2 + Sum|T|^2 = 1. --- *)
colPower = Table[Sum[Abs[Rmat[[r, inc]]]^2, {r, 3}] + Sum[Abs[Tmat[[t, inc]]]^2, {t, 3}], {inc, 3}];
Print["  [3] per-incident power |R|^2+|T|^2 (P,S,H) = ", chop[colPower],
   "  max|.-1| = ", ScientificForm[Max[Abs[colPower - 1.]], 3], " -> ",
   If[Max[Abs[colPower - 1.]] < 1.*^-7, "PASS", "FAIL"]];

(* --- [3b] regression: the 3x3-via-J R/T equals the old 6x6 linear solve --- *)
Mref = Join[Dz[a1, b1, r1, om0, kx0, ky0][[All, 4 ;; 6]], -Dz[a2, b2v, r2, om0, kx0, ky0][[All, 1 ;; 3]], 2];
Xref = LinearSolve[Mref, -Dz[a1, b1, r1, om0, kx0, ky0][[All, 1 ;; 3]]];
rtDev = Max[Abs[Flatten[{Rmat - Xref[[1 ;; 3]], Tmat - Xref[[4 ;; 6]]}]]];
Print["  [3b] 3x3-via-J R/T == 6x6 solve : ", ScientificForm[rtDev, 3],
   " -> ", If[rtDev < 1.*^-7, "PASS", "FAIL"]];

Print["ThesisInterfaceRT.wl loaded."];
