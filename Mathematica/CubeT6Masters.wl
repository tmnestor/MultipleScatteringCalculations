(* ::Package:: *)
(* CubeT6Masters.wl -- Cached closed forms of the new master integrals
   required by the 27-component Path-B closure (Tier-6).

   These values were computed by Mathematica's Integrate[] in
   CubeT6Probe.wl and verified against NIntegrate to >= 8 digits
   (the degree-8 even-parity ones are harder for NIntegrate due to
   the 1/|r| singularity, but the symbolic closed forms are exact).

   Usage:
     Get["/Users/tod/.../Mathematica/CubeT6Masters.wl"];
     (* Now Mp6m220[], Mp6m400[], ... are defined *)

   All values are Mp[p,q,r] = int_{[0,1]^3} x^p y^q z^r / |r| dV.
*)

(* ----- quad-linear block (rank-6, degree 6) ----- *)

(* Mp[2,2,2] *)
mp222 = (
    1008 Sqrt[3] - 42 Pi - 672 ArcCoth[Sqrt[3]]
  + 522 Log[2] + 441 Log[-1 + Sqrt[3]] - 1395 Log[1 + Sqrt[3]]
  - 84 Log[2 + Sqrt[3]] - 90 Log[-19 + 11 Sqrt[3]]
  + 35 Log[26 + 15 Sqrt[3]]) / 10080;

(* Mp[4,2,0] *)
mp420 = (
   -1536 Sqrt[3] + 544 Pi + 11808 ArcCoth[Sqrt[3]]
  - 45 Log[2] + 90 Log[189750626 - 109552575 Sqrt[3]]
  + 1470 Log[1351 - 780 Sqrt[3]] - 770 Log[26 - 15 Sqrt[3]]
  + 441 Log[-1 + Sqrt[3]] - 351 Log[1 + Sqrt[3]]
  + 1008 Log[2 + Sqrt[3]] + 1440 Log[1351 + 780 Sqrt[3]]) / 161280;

(* Mp[6,0,0] *)
mp600 = (
    1536 Sqrt[3] - 320 Pi + 1824 ArcCoth[Sqrt[3]]
  + 861 Log[2] - 110 Log[189750626 - 109552575 Sqrt[3]]
  - 302 Log[1351 - 780 Sqrt[3]] + 14 Log[26 - 15 Sqrt[3]]
  + 3375 Log[-1 + Sqrt[3]] - 5097 Log[1 + Sqrt[3]]
  + 224 Log[26 + 15 Sqrt[3]]) / 21504;

(* ----- quad-quad block (rank-8, degree 8) ----- *)

(* Mp[4,2,2] *)
mp422 = (
   -512 Pi - 51090 ArcCoth[Sqrt[3]] - 61437 Log[2]
  - 200 Log[26 - 15 Sqrt[3]]
  + 6 (9728 Sqrt[3] + 16170 Log[-1 + Sqrt[3]] + 4909 Log[1 + Sqrt[3]]
       + 2400 Log[2 + Sqrt[3]] - 600 Log[-19 + 11 Sqrt[3]])
  ) / 3225600;

(* Mp[4,4,0] *)
mp440 = (
   -176128 Pi - 1340973 Log[2]
  + 36000 Log[189750626 - 109552575 Sqrt[3]]
  - 63000 Log[1351 - 780 Sqrt[3]]
  + 271200 Log[26 - 15 Sqrt[3]]
  - 451395 Log[-1 + Sqrt[3]] + 3334941 Log[1 + Sqrt[3]]
  - 128 (11136 Sqrt[3] + 1575 Log[-19 + 11 Sqrt[3]]
         - 1925 Log[1351 + 780 Sqrt[3]])
  ) / 25804800;

(* Mp[6,2,0] *)
mp620 = (
    2162688 Sqrt[3] + 237568 Pi - 112107 Log[2]
  - 74400 Log[189750626 - 109552575 Sqrt[3]]
  + 416920 Log[1351 - 780 Sqrt[3]]
  - 341280 Log[26 - 15 Sqrt[3]]
  + 2281755 Log[-1 + Sqrt[3]] - 2057541 Log[1 + Sqrt[3]]
  ) / 30965760;

(* Mp[8,0,0] *)
mp800 = (
   -22272 Sqrt[3] - 5632 Pi + 19890 ArcCoth[Sqrt[3]]
  + 3708 Log[2] + 2400 Log[189750626 - 109552575 Sqrt[3]]
  - 5120 Log[1351 - 780 Sqrt[3]] + 2800 Log[26 - 15 Sqrt[3]]
  - 70035 Log[-1 + Sqrt[3]] + 62619 Log[1 + Sqrt[3]]
  ) / 276480;

(* Numerical values for quick verification (20 digits) *)
mp222Num = 0.02841932167444858666331231865612579692`20.;
mp420Num = 0.05367758667590844931171787307731876719`20.;
mp600Num = 0.12400223475238657546382753406975904768`20.;
mp422Num = 0.01641938167007140719500603834668888915`20.;
mp440Num = 0.03088317871181584481033140024904519664`20.;
mp620Num = 0.03741471164087810752421247908849027780`20.;
mp800Num = 0.09462291403763880507014615349410955936`20.;

Print["CubeT6Masters.wl loaded: mp222, mp420, mp600, mp422, mp440, mp620, mp800"];
