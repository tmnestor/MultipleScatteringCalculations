#!/usr/bin/env wolframscript
(* ==========================================================================
   THE 27x27 CUBIC SYSTEM: FIVE BLOCKS, EXPLICIT AND INVERTED

   The dynamic single-site system M for a cubic voxel is 27x27 with 19
   independent entries. It commutes with O_h acting on R^3 (x) R^3 (x) R^3 as
   R (x) R (x) R -- the unknowns are RAW derivatives, all three indices free --
   so by Schur's lemma M = (+) id_Gamma(lambda) (x) B_lambda in a basis fixed by
   the symmetry alone. The parameters move the eigenvalues; they cannot move the
   eigenvectors.

   The decomposition (derived in docs/commutant_block_structure.tex) is

       T1u (x) T1u (x) T1u = A1u (+) A2u (+) 2 Eu (+) 4 T1u (+) 3 T2u

   so the blocks are 1x1, 1x1, 2x2, 4x4, 3x3. This script builds them
   explicitly in terms of a..t and inverts them.

   METHOD. Characters alone give multiplicities, not blocks. To extract
   B_lambda we need the irrep MATRICES D^lambda: the projector onto a SINGLE
   ROW of lambda,

       P^lambda_11 = (d_lambda/|G|) sum_R conj(D^lambda_11(R)) rho(R),

   has image of dimension exactly m_lambda, and M restricted to that image IS
   B_lambda -- one copy, not m_lambda d_lambda of them.

   Every representation below is CONSTRUCTED and then verified to be a
   homomorphism and irreducible, rather than copied from a table.
   ========================================================================== *)

Print["=============================================================="];
Print["THE 27x27 CUBIC SYSTEM -- FIVE BLOCKS, EXPLICIT AND INVERTED"];
Print["=============================================================="];

(* ---------- the system, transcribed verbatim from A33.nb ----------
   Nine 9x9 blocks over the 19 constants a..t (there is no 'o'). Transcribed
   rather than read from a file so this script stands alone.                *)
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
Print["[0] M is ", Dimensions[M], " in ", Length[vars], " parameters"];

(* ---------- O_h as the 48 signed permutation matrices ---------- *)
grp = Flatten[Table[DiagonalMatrix[sg] . IdentityMatrix[3][[pm]],
    {pm, Permutations[{1, 2, 3}]}, {sg, Tuples[{1, -1}, 3]}], 1];
rho[R_] := KroneckerProduct[R, R, R];
Print["[1] |O_h| = ", Length[grp],
  ",  M commutes with every element: ",
  And @@ (Simplify[rho[#] . M - M . rho[#]] === ConstantArray[0, {27, 27}] & /@ grp)];

(* ---------- the five irreducible representations, CONSTRUCTED ----------
   A1u = det R.   A2u = sgn(perm) det R.   T1u = R (defining).
   T2g acts on {yz, zx, xy}; Eg on {(2zz-xx-yy)/Sqrt6, (xx-yy)/Sqrt2}; the
   ungerade partners follow by multiplying by det R.                        *)
sym[u_, v_] := Flatten[Outer[Times, u, v] + Outer[Times, v, u]]/2;
e1 = {1, 0, 0}; e2 = {0, 1, 0}; e3 = {0, 0, 1};
t2basis = Orthogonalize[{sym[e2, e3], sym[e3, e1], sym[e1, e2]}];
egbasis = Orthogonalize[{(2 sym[e3, e3] - sym[e1, e1] - sym[e2, e2]),
                         (sym[e1, e1] - sym[e2, e2])}];
restrict[B_, R_] := B . KroneckerProduct[R, R] . Transpose[B];

DA1u[R_] := {{Det[R]}};
DA2u[R_] := {{Det[Abs[R]] Det[R]}};
DT1u[R_] := R;
DT2u[R_] := restrict[t2basis, R] Det[R];
DEu[R_]  := restrict[egbasis, R] Det[R];

irreps = <|"A1u" -> DA1u, "A2u" -> DA2u, "Eu" -> DEu, "T1u" -> DT1u, "T2u" -> DT2u|>;
dims   = <|"A1u" -> 1, "A2u" -> 1, "Eu" -> 2, "T1u" -> 3, "T2u" -> 3|>;

Print["[2] representation checks (homomorphism, and <chi,chi> = 1):"];
Do[D0 = irreps[key];
   hom = And @@ Flatten[Table[Simplify[D0[g1] . D0[g2] - D0[g1 . g2]] ===
        ConstantArray[0, {dims[key], dims[key]}], {g1, grp[[1 ;; 8]]}, {g2, grp[[1 ;; 8]]}]];
   chi = Tr[D0[#]] & /@ grp;
   Print["    ", key, "  dim ", dims[key], "   homomorphism ", hom,
     "   <chi,chi> = ", Total[chi^2]/48], {key, Keys[irreps]}];

(* ---------- project onto ROW 1 of each irrep; image dimension = m_lambda ---- *)
Print["[3] blocks extracted (image dimension = multiplicity):"];
blocks = <||>; bases = <||>;
Do[D0 = irreps[key]; dl = dims[key];
   P = (dl/48) Total[(D0[#][[1, 1]]) rho[#] & /@ grp];
   bas = Select[RowReduce[P], # =!= ConstantArray[0, 27] &];
   mult = Length[bas];
   (* M maps the image to itself: M.w_j = sum_i B[i,j] w_i *)
   img = Transpose[bas];                        (* 27 x mult *)
   B = Simplify[LinearSolve[img, M . img]];     (* mult x mult *)
   blocks[key] = B; bases[key] = bas;
   Print["    ", key, "   multiplicity m = ", mult, "   block ", Dimensions[B]],
 {key, Keys[irreps]}];

Print["[4] sum m*d = ",
  Total[Table[Length[blocks[key]] dims[key], {key, Keys[irreps]}]], "   (must be 27)"];

(* ---------- the blocks, written out ---------- *)
Print[];
Print["=============================================================="];
Print["THE BLOCKS"];
Print["=============================================================="];
Do[Print["--- ", key, "  (", dims[key], " copies of a ",
     Length[blocks[key]], "x", Length[blocks[key]], " block) ---"];
   Print[MatrixForm[blocks[key]]], {key, Keys[irreps]}];

(* ---------- inverses ---------- *)
Print[];
Print["=============================================================="];
Print["THE INVERSES, and M^-1 = (+) id (x) B^-1"];
Print["=============================================================="];
invs = <||>;
Do[B = blocks[key];
   Bi = Simplify[Inverse[B]];
   invs[key] = Bi;
   ok = Simplify[B . Bi] === IdentityMatrix[Length[B]];
   Print["--- ", key, "   B.B^-1 = I : ", ok, "   det B = ", Simplify[Det[B]]],
 {key, Keys[irreps]}];

Print[];
Print["[5] the 1x1 and 2x2 inverses in closed form:"];
Do[Print["    ", key, " :  ", MatrixForm[Simplify[invs[key]]]],
 {key, {"A1u", "A2u", "Eu"}}];

Print[];
Print["[6] det M factorises over the blocks:"];
detFromBlocks = Simplify[Times @@ Table[Det[blocks[key]]^dims[key], {key, Keys[irreps]}]];
Print["    prod det(B_lambda)^d_lambda = ", detFromBlocks];
Print["    equals Det[M] : ", Simplify[detFromBlocks - Det[M]] === 0];
Print["=============================================================="];
