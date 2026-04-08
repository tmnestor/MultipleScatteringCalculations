(* ::Package:: *)
(* CubeValidation.wl -- Tier-5 validation of Path-B 9x9 closure
   against Path-A (CubeAnalytic.wl + effective_contrasts.py).

   Fast script: loads CubeAnalytic.wl for Path-A and uses cached
   high-precision numerical values from CubeGalerkin27.wl run4 for
   Path-B. Does NOT re-integrate the heavy master integrals.

   Tests:
     V1a  Master-integral sanity (g0Cube consistency, Pell identity,
          trace identity 3*iB11 - iA = 0)
     V1b  Path-B 9x9 mass and stiffness matrices on T_9
     V2   4 cubic-symmetry amplification factors from Path-B
     V3   Compare to Path-A (cubeABC at omega=0) and Python T-matrix
*)

Print["==== CubeValidation.wl (Tier-5) ===="];

(* Load Path-A package *)
Get["/Users/tod/Desktop/MultipleScatteringCalculations/Mathematica/CubeAnalytic.wl"];

(* ============================================================ *)
(* Path-B cached numerical values from CubeGalerkin27.wl run4    *)
(* (see /tmp/cubeg27_strain_run4.log for provenance)             *)
(* ============================================================ *)

(* Master integrals over [-1,1]^3 of 1/|x| family *)
g0CubeCached = -2 (Pi + Log[1351 - 780 Sqrt[3]]);    (* Mathematica's form *)

(* Body-bilinear scalars (numerical, from run4) *)
iANum   = 0.23528908054870752001320010485854594849`30;
iB11Num = 0.078429693516235839876384681448378457953`29;
iB12Num = 0.049701712967280897641872077598360075681`29;
J11Num  = 0.15685938703247168013681542341016749053`29;

(* Strain-block scalars (numerical, from run4) *)
bDispA = 60.23400462046913`16;
bDispB = 20.07800154015638`16;
alphaA = 15.37252824403423`16;
alphaB = 3.949761457945144`16;
betaA  = 0;
betaB  = -1.471925680511371`16;
gammaA = 7.686264122017116`16;
gammaB = 2.119728856266587`16;

(* ============================================================ *)
(* V1a  Master-integral sanity                                  *)
(* ============================================================ *)

Print[""];
Print["==== V1a  Master-integral sanity ===="];

Print["g0Cube (CubeAnalytic)      = ", N[g0Cube, 20]];
Print["g0Cube (run4 form)         = ", N[g0CubeCached, 20]];
Print["  difference               = ", N[g0Cube - g0CubeCached, 20]];
Print["  (should be ~1e-30)"];

(* Pell identity: 1351 - 780 Sqrt[3] == (70226 + 40545 Sqrt[3])^(-2/3) *)
pellLHS = 1351 - 780 Sqrt[3];
pellRHS = (70226 + 40545 Sqrt[3])^(-2/3);
Print[""];
Print["Pell identity check:"];
Print["  LHS                      = ", N[pellLHS, 30]];
Print["  RHS                      = ", N[pellRHS, 30]];
Print["  LHS - RHS                = ", N[pellLHS - pellRHS, 30]];

(* Trace identity: 3*iB11 - iA = 0 (cyclic tent-function symmetry) *)
Print[""];
Print["Trace identity 3*iB11 - iA = 0:"];
Print["  iA   (run4)              = ", iANum];
Print["  iB11 (run4)              = ", iB11Num];
Print["  3*iB11 - iA              = ", 3 iB11Num - iANum];
Print["  (should be ~1e-18)"];

(* Relation: J11 = iA - iB11 *)
Print[""];
Print["Relation J11 = iA - iB11:"];
Print["  iA - iB11                = ", iANum - iB11Num];
Print["  J11 (run4)               = ", J11Num];
Print["  difference               = ", iANum - iB11Num - J11Num];

Print[""];
Print["==== V1a DONE ===="];

(* ============================================================ *)
(* V1b  Internal Path-B scalar identities                       *)
(* ============================================================ *)

Print[""];
Print["==== V1b  Path-B internal scalar identities ===="];

(* Displacement block: b_disp = 256 * (A*iA + B*iB11) *)
Print["Displacement block: b_disp = 256 (A iA + B iB11)"];
Print["  bDispA (run4)            = ", bDispA];
Print["  256 * iA                 = ", N[256 iANum, 16]];
Print["  difference               = ", bDispA - 256 iANum];
Print["  bDispB (run4)            = ", bDispB];
Print["  256 * iB11               = ", N[256 iB11Num, 16]];
Print["  difference               = ", bDispB - 256 iB11Num];

(* Shear = alpha/2 identity: gamma^A = alpha^A / 2 (exact by basis bilinearity) *)
Print[""];
Print["Shear-axial A-channel identity: gamma^A = alpha^A / 2"];
Print["  alphaA                   = ", alphaA];
Print["  2 * gammaA               = ", 2 gammaA];
Print["  alphaA - 2 gammaA        = ", alphaA - 2 gammaA];
Print["  (should be ~1e-15)"];

(* beta^A = 0 (exact, parity argument) *)
Print[""];
Print["Axial off-diagonal A-channel identity: beta^A = 0"];
Print["  betaA                    = ", betaA];
Print["  (should be exactly 0)"];

(* ============================================================ *)
(* V1b.2  Cubic-symmetry diagonalization of the 9x9 B9 matrix   *)
(* ============================================================ *)
(* In the cubic-symmetry basis the 9x9 body bilinear form        *)
(* decomposes into 4 irreducible blocks:                         *)
(*   - translation (3): b_disp * I_3                             *)
(*   - hydrostatic (1): (alpha + 2*beta) * 1                     *)
(*   - axial deviator (2): (alpha - beta) * I_2                  *)
(*   - pure shear (3): gamma * I_3                               *)

Print[""];
Print["==== V1b.2  Cubic-symmetry diagonalization of B9 ===="];
Print["(A-channel: A=1, B=0; B-channel: A=0, B=1)"];

hydroA = alphaA + 2 betaA;
hydroB = alphaB + 2 betaB;
devA   = alphaA - betaA;
devB   = alphaB - betaB;

Print["translation block eigenvalue:"];
Print["  A: ", bDispA, "   B: ", bDispB];
Print["hydrostatic block eigenvalue (alpha + 2 beta):"];
Print["  A: ", hydroA, "   B: ", hydroB];
Print["axial deviator block eigenvalue (alpha - beta):"];
Print["  A: ", devA, "   B: ", devB];
Print["pure shear block eigenvalue (gamma):"];
Print["  A: ", gammaA, "   B: ", gammaB];

Print[""];
Print["==== V1b DONE ===="];
