(* ExtractUngeradeBlocks.wl -- Extract ungerade Bbody A/B channel
   values from CubeT27AssembleResults.wl for hardcoding into Python.

   FIXES the Missing[KeyAbsent, 0] bug: beta_7 (paper label 7) was
   skipped in CubeT6ScalarValues_HighPrec_Pell.wl.  We patch it here
   using the numerical value from CubeT6QuadQuad.wl.

   Usage:
     /Applications/Wolfram.app/Contents/MacOS/wolframscript \
       -file Mathematica/ExtractUngeradeBlocks.wl
*)

Print["==== ExtractUngeradeBlocks.wl ===="];
Print[];

$here = DirectoryName[$InputFileName];

(* Load numerical quad-quad for beta_7 value *)
Get[FileNameJoin[{$here, "CubeT6QuadQuad.wl"}]];
beta7val = B27quadB[[2, 17]];
Print["beta_7 (from CubeT6QuadQuad.wl) = ", NumberForm[beta7val, 16]];

(* Load symbolic results *)
Print["Loading CubeT27AssembleResults.wl ..."];
t0 = AbsoluteTime[];
Get[FileNameJoin[{$here, "CubeT27AssembleResults.wl"}]];
Print["  loaded in ", Round[AbsoluteTime[] - t0, 0.1], " s"];
Print[];

(* Patch: replace Missing[KeyAbsent, 0] -> beta7val in BbodySym *)
Print["Patching Missing[KeyAbsent, 0] -> beta_7 in BbodySym ..."];
nMissing = Count[Flatten[BbodySym], _Missing, Infinity];
Print["  Missing entries before patch: ", nMissing];
BbodySym = BbodySym /. Missing["KeyAbsent", 0] -> beta7val;
nMissing2 = Count[Flatten[BbodySym], _Missing, Infinity];
Print["  Missing entries after patch: ", nMissing2];
Assert[nMissing2 === 0];

(* Also patch irrepData Bblock *)
Do[
  Module[{Bb},
    Bb = irrepData[irrep, "Bblock"];
    Bb = Bb /. Missing["KeyAbsent", 0] -> beta7val;
    irrepData[irrep] = Append[irrepData[irrep], "Bblock" -> Bb];
  ],
  {irrep, Keys[irrepData]}];
Print["  irrepData Bblocks patched."];
Print[];

(* ---- Evaluate and print per-irrep blocks ---- *)
ungeradeIrreps = {"T1u", "T2u", "A2u", "Eu"};

Do[
  Module[{Bb, BbA, BbB, Mb, m},
    m = irrepData[irrep, "m"];
    Mb = irrepData[irrep, "Mblock"];
    Bb = irrepData[irrep, "Bblock"] /. atomRules;

    BbA = N[Bb /. {Aelas -> 1, Belas -> 0}, 20];
    BbB = N[Bb /. {Aelas -> 0, Belas -> 1}, 20];

    Print["---- ", irrep, " (", m, "x", m, ") ----"];
    Print["  Mass block (exact rational):"];
    If[m == 1,
      Print["    M = ", Mb],
      Do[Print["    M[", i, "] = ", Mb[[i]]], {i, m}]
    ];
    Print[];
    Print["  Bbody A-channel (Aelas=1, Belas=0):"];
    If[m == 1,
      Print["    Bbody_A = ", N[BbA, 16]],
      Do[Print["    row ", i, " = ",
        Map[Function[x, NumberForm[N[x, 16], 16]], BbA[[i]]]], {i, m}]
    ];
    Print[];
    Print["  Bbody B-channel (Aelas=0, Belas=1):"];
    If[m == 1,
      Print["    Bbody_B = ", N[BbB, 16]],
      Do[Print["    row ", i, " = ",
        Map[Function[x, NumberForm[N[x, 16], 16]], BbB[[i]]]], {i, m}]
    ];
    Print[];

    (* BelBlock *)
    If[KeyExistsQ[irrepData[irrep], "BelBlock"],
      Module[{BelBlk = irrepData[irrep, "BelBlock"]},
        Print["  BelBlock (symbolic):"];
        If[m == 1,
          Print["    Bel = ", Simplify[BelBlk]],
          Do[Print["    Bel[", i, "] = ", Simplify[BelBlk[[i]]]], {i, m}]
        ];
        Print[];
      ]
    ];
  ],
  {irrep, ungeradeIrreps}];

(* ---- Gerade cross-check ---- *)
Print["==== Gerade body blocks (cross-check) ===="];
Do[
  Module[{Bb, BbA, BbB, Mb},
    Mb = irrepData[irrep, "Mblock"];
    Bb = irrepData[irrep, "Bblock"] /. atomRules;
    BbA = N[Bb /. {Aelas -> 1, Belas -> 0}, 20];
    BbB = N[Bb /. {Aelas -> 0, Belas -> 1}, 20];
    Print["  ", irrep, ": M=", Mb[[1,1]],
      "  Bbody_A=", NumberForm[BbA[[1,1]], 16],
      "  Bbody_B=", NumberForm[BbB[[1,1]], 16],
      "  ev_A=", NumberForm[BbA[[1,1]]/Mb[[1,1]], 16],
      "  ev_B=", NumberForm[BbB[[1,1]]/Mb[[1,1]], 16]];
  ],
  {irrep, {"A1g", "Eg", "T2g"}}];

(* ---- Python-ready format ---- *)
Print[];
Print["==== PYTHON COPY-PASTE FORMAT ===="];
Print[];

printPyMatrix[name_, mat_] := Module[{m = Length[mat]},
  Print["    ", name, " = np.array(["];
  Do[
    Print["        [",
      StringRiffle[Map[ToString[CForm[N[#, 16]]] &, mat[[i]]], ", "],
      "],"],
    {i, m}];
  Print["    ])"];
];

printPyScalar[name_, val_] :=
  Print["    ", name, " = ", ToString[CForm[N[val, 16]]]];

Do[
  Module[{Bb, BbA, BbB, Mb, m},
    m = irrepData[irrep, "m"];
    Mb = irrepData[irrep, "Mblock"];
    Bb = irrepData[irrep, "Bblock"] /. atomRules;
    BbA = N[Bb /. {Aelas -> 1, Belas -> 0}, 20];
    BbB = N[Bb /. {Aelas -> 0, Belas -> 1}, 20];

    Print["    # ", irrep, " (", m, "x", m, ")"];
    If[m == 1,
      (* Scalar irrep — mass, bbody are 1x1 matrices, extract scalar *)
      printPyScalar["M_" <> irrep, Mb[[1, 1]]];
      printPyScalar["Bbody_A_" <> irrep, BbA[[1, 1]]];
      printPyScalar["Bbody_B_" <> irrep, BbB[[1, 1]]];
      ,
      (* Matrix irrep *)
      printPyMatrix["M_" <> irrep, Mb];
      printPyMatrix["Bbody_A_" <> irrep, BbA];
      printPyMatrix["Bbody_B_" <> irrep, BbB];
    ];
    Print[];
  ],
  {irrep, ungeradeIrreps}];

Print["==== done ===="];
