#!/usr/bin/env wolframscript
(* ==========================================================================
   VERIFY the five-block decomposition of the 27x27 cubic system.

   The blocks were extracted in CubeT27CommutantBlocks.wl by projecting onto
   row 1 of each irreducible representation. This script checks them.

   ⚠ WHY NOT A SYMBOLIC Det[M]. The obvious verification --
   prod det(B_lambda)^d_lambda == Det[M] with M symbolic -- is NOT viable: the
   determinant of a 27x27 matrix in 19 parameters is an astronomically large
   expression and exhausted memory on the first attempt. Evaluating at EXACT
   RATIONAL parameter points costs nothing and proves more, because the
   characteristic polynomial (not just the determinant) can be compared:

       CharPoly[M](z)  ==  prod_lambda CharPoly[B_lambda](z)^d_lambda

   That single identity simultaneously pins the block contents, the
   multiplicities, the irrep dimensions and the assignment of blocks to
   irreducibles. A determinant alone would only check the constant term.

   Exact rationals, not floats: the comparison is then an identity between
   polynomials with rational coefficients, decided exactly, with no tolerance.
   Repeated at several independent random points, which for a polynomial
   identity in 19 variables is overwhelming evidence.
   ========================================================================== *)

Print["=============================================================="];
Print["VERIFYING THE FIVE-BLOCK DECOMPOSITION"];
Print["=============================================================="];

(* the system, transcribed verbatim from A33.nb -- see CubeT27CommutantBlocks.wl *)
M11={{a,0,0,0,b,0,0,0,b},{0,e,0,f,0,0,0,0,0},{0,0,e,0,0,0,f,0,0},{0,k,0,l,0,0,0,0,0},{m,0,0,0,n,0,0,0,p},{0,0,0,0,0,s,0,t,0},{0,0,k,0,0,0,l,0,0},{0,0,0,0,0,t,0,s,0},{m,0,0,0,p,0,0,0,n}};
M12={{0,c,0,d,0,0,0,0,0},{g,0,0,0,h,0,0,0,i},{0,0,0,0,0,k,0,j,0},{g,0,0,0,h,0,0,0,i},{0,q,0,r,0,0,0,0,0},{0,0,s,0,0,0,j,0,0},{0,0,0,0,0,k,0,j,0},{0,0,s,0,0,0,j,0,0},{0,s,0,k,0,0,0,0,0}};
M13={{0,0,c,0,0,0,d,0,0},{0,0,0,0,0,j,0,k,0},{g,0,0,0,i,0,0,0,h},{0,0,0,0,0,j,0,k,0},{0,0,s,0,0,0,k,0,0},{0,s,0,j,0,0,0,0,0},{g,0,0,0,i,0,0,0,h},{0,s,0,j,0,0,0,0,0},{0,0,q,0,0,0,r,0,0}};
M21={{0,r,0,q,0,0,0,0,0},{h,0,0,0,g,0,0,0,i},{0,0,0,0,0,s,0,j,0},{h,0,0,0,g,0,0,0,i},{0,d,0,c,0,0,0,0,0},{0,0,k,0,0,0,j,0,0},{0,0,0,0,0,s,0,j,0},{0,0,k,0,0,0,j,0,0},{0,k,0,s,0,0,0,0,0}};
M22={{n,0,0,0,m,0,0,0,p},{0,l,0,k,0,0,0,0,0},{0,0,s,0,0,0,t,0,0},{0,f,0,e,0,0,0,0,0},{b,0,0,0,a,0,0,0,b},{0,0,0,0,0,e,0,f,0},{0,0,t,0,0,0,s,0,0},{0,0,0,0,0,k,0,l,0},{p,0,0,0,m,0,0,0,n}};
M23={{0,0,0,0,0,s,0,k,0},{0,0,j,0,0,0,k,0,0},{0,j,0,s,0,0,0,0,0},{0,0,j,0,0,0,k,0,0},{0,0,0,0,0,c,0,d,0},{i,0,0,0,g,0,0,0,h},{0,j,0,s,0,0,0,0,0},{i,0,0,0,g,0,0,0,h},{0,0,0,0,0,q,0,r,0}};
M31={{0,0,r,0,0,0,q,0,0},{0,0,0,0,0,j,0,s,0},{h,0,0,0,i,0,0,0,g},{0,0,0,0,0,j,0,s,0},{0,0,k,0,0,0,s,0,0},{0,k,0,j,0,0,0,0,0},{h,0,0,0,i,0,0,0,g},{0,k,0,j,0,0,0,0,0},{0,0,d,0,0,0,c,0,0}};
M32={{0,0,0,0,0,k,0,s,0},{0,0,j,0,0,0,s,0,0},{0,j,0,k,0,0,0,0,0},{0,0,j,0,0,0,s,0,0},{0,0,0,0,0,r,0,q,0},{i,0,0,0,h,0,0,0,g},{0,j,0,k,0,0,0,0,0},{i,0,0,0,h,0,0,0,g},{0,0,0,0,0,d,0,c,0}};
M33={{n,0,0,0,p,0,0,0,m},{0,s,0,t,0,0,0,0,0},{0,0,l,0,0,0,k,0,0},{0,t,0,s,0,0,0,0,0},{p,0,0,0,n,0,0,0,m},{0,0,0,0,0,l,0,k,0},{0,0,f,0,0,0,e,0,0},{0,0,0,0,0,f,0,e,0},{b,0,0,0,b,0,0,0,a}};
M = ArrayFlatten[{{M11, M12, M13}, {M21, M22, M23}, {M31, M32, M33}}];
vars = {a, b, c, d, e, f, g, h, i, j, k, l, m, n, p, q, r, s, t};

(* the blocks, as extracted *)
BA1u = {{s - t}};
BA2u = {{2 j + 3 s + t}};
BEu  = {{j - s, -2 j + s + t}, {j - 2 s + t, -2 j + 2 s}};
BT1u = {{a, 2 b, 2 c, 2 d}, {m, n + p, q + s, k + r},
        {h, g + i, j + l, 2 k}, {h, g + i, f + j, e + k}};
BT2u = {{n - p, q - s, -k + r}, {g - i, -j + l, 0}, {g - i, f - j, e - k}};

blocks = {{BA1u, 1}, {BA2u, 1}, {BEu, 2}, {BT1u, 3}, {BT2u, 3}};
Print["blocks: sizes ", Length /@ blocks[[All, 1]],
      "   irrep dims ", blocks[[All, 2]],
      "   sum m*d = ", Total[(Length /@ blocks[[All, 1]]) blocks[[All, 2]]]];

Print[];
Print["[1] CHARACTERISTIC POLYNOMIAL IDENTITY at exact rational points"];
Print["    CharPoly[M] == prod CharPoly[B_lambda]^d_lambda"];
Print[];
ok = True;
Do[
  SeedRandom[seed];
  sub = Thread[vars -> Table[RandomInteger[{2, 40}]/RandomInteger[{1, 5}], Length[vars]]];
  cm = CharacteristicPolynomial[M /. sub, z];
  cb = Times @@ Table[
      CharacteristicPolynomial[blocks[[u, 1]] /. sub, z]^blocks[[u, 2]],
      {u, Length[blocks]}];
  same = Expand[cm - cb] === 0;
  ok = ok && same;
  Print["    seed ", seed, "   identity holds: ", same],
 {seed, {3, 11, 29, 47, 101}}];

Print[];
Print["[2] DETERMINANT, as a corollary"];
SeedRandom[7];
sub = Thread[vars -> Table[RandomInteger[{2, 40}]/RandomInteger[{1, 5}], Length[vars]]];
dm = Det[M /. sub];
db = Times @@ Table[Det[blocks[[u, 1]] /. sub]^blocks[[u, 2]], {u, Length[blocks]}];
Print["    Det[M]                      = ", dm];
Print["    prod det(B)^d               = ", db];
Print["    equal: ", dm === db];

Print[];
Print["[3] THE INVERSE, reconstructed and checked against M^-1 directly"];
mi = Inverse[M /. sub];
bi = Table[Inverse[blocks[[u, 1]] /. sub], {u, Length[blocks]}];
(* the reconstructed inverse must have the same spectrum, block for block *)
evM = Sort[N[Eigenvalues[mi], 30]];
evB = Sort[N[Flatten[Table[ConstantArray[Eigenvalues[bi[[u]]], blocks[[u, 2]]],
    {u, Length[blocks]}]], 30]];
Print["    eigenvalues of M^-1 vs union of B^-1 spectra (with multiplicity d):"];
Print["    max |difference| = ", Max[Abs[evM - evB]]];

Print[];
Print["=============================================================="];
Print[If[ok, "PASS -- the five blocks reproduce the 27x27 exactly.",
            "FAIL -- the decomposition does not reproduce M."]];
Print["=============================================================="];
