(* ::Package:: *)
(* CubeAnalytic.wl -- Analytical cube T-matrix: Gamma0, A, B, C *)
(* Static Eshelby + smooth Taylor-moment radiation corrections *)
(* Matches cubic_scattering/effective_contrasts.py exactly *)

(* ============================================================ *)
(* Geometric constant: g0 = Integral[1/|x|, {x in [-1,1]^3}]   *)
(* Pell-stable form: 70226^2 - 3*40545^2 = 1                   *)
(* ============================================================ *)
g0Cube = (4/3) Log[70226 + 40545 Sqrt[3]] - 2 Pi;

(* Surface integral constants for cube Eshelby *)
j1Cube = 2 Pi/3;
j2Cube = -2/9 (Sqrt[3] - Pi);
k1Cube = 2/9 (2 Sqrt[3] + Pi);

(* ============================================================ *)
(* Taylor coefficients of the smooth Green's tensor              *)
(* G^s_{ij}(x) = delta_{ij} Phi(r^2) + x_i x_j Psi(r^2)       *)
(* Phi(u) = Sum[phi_n u^n], Psi(u) = Sum[psi_n u^n]            *)
(* ============================================================ *)
cubeTaylorCoefficients[omega_, alpha_, beta_, rho_, nTaylor_: 8] :=
  Module[{phi, psi, ikb, ika, cf, cg},
    ikb = I omega/beta;
    ika = I omega/alpha;
    cf = 1/(4 Pi rho beta^2);
    cg = 1/(4 Pi rho);
    phi = Table[
      cf ikb^(2 n + 1)/(2 n + 1)!,
      {n, 0, nTaylor - 1}];
    psi = Table[
      cg (ika^(2 n + 3)/alpha^2 - ikb^(2 n + 3)/beta^2)/(2 n + 3)!,
      {n, 0, nTaylor - 1}];
    {phi, psi}
  ];

(* ============================================================ *)
(* Monomial moments of the cube [-a,a]^3                        *)
(* via trinomial expansion u^m = (x1^2+x2^2+x3^2)^m            *)
(* Returns {S0, S1, S2, S11} each of length nMax+1              *)
(* ============================================================ *)
cubeMoments[a_, nMax_] :=
  Module[{mu, s0, s1, s2, s11, c, r},
    mu = Table[2. a^(2 k + 1)/(2 k + 1), {k, 0, nMax + 2}];
    s0 = Table[0., {nMax + 1}];
    s1 = Table[0., {nMax + 1}];
    s2 = Table[0., {nMax + 1}];
    s11 = Table[0., {nMax + 1}];
    Do[
      Do[
        Do[
          r = m - p - q;
          If[r >= 0,
            c = m!/(p! q! r!);
            s0[[m + 1]] += c mu[[p + 1]] mu[[q + 1]] mu[[r + 1]];
            s1[[m + 1]] += c mu[[p + 2]] mu[[q + 1]] mu[[r + 1]];
            s2[[m + 1]] += c mu[[p + 3]] mu[[q + 1]] mu[[r + 1]];
            s11[[m + 1]] += c mu[[p + 2]] mu[[q + 2]] mu[[r + 1]]
          ],
        {q, 0, m - p}],
      {p, 0, m}],
    {m, 0, nMax}];
    {s0, s1, s2, s11}
  ];

(* ============================================================ *)
(* Gamma0 = Integral[G_{11}(x), {x in cube}]                    *)
(* = static (g0) + smooth (Taylor moments)                      *)
(* ============================================================ *)
cubeGamma0[omega_, a_, alpha_, beta_, rho_, nTaylor_: 8] :=
  Module[{a0k, b0k, stat, phi, psi, s0, s1, s2, s11},
    a0k = (alpha^2 + beta^2)/(8 Pi rho alpha^2 beta^2);
    b0k = (alpha^2 - beta^2)/(8 Pi rho alpha^2 beta^2);
    stat = a^2 (a0k + b0k/3) g0Cube;
    {phi, psi} = cubeTaylorCoefficients[omega, alpha, beta, rho, nTaylor];
    {s0, s1, s2, s11} = cubeMoments[a, nTaylor - 1];
    N[stat + Total[phi s0] + Total[psi s1]]
  ];

(* ============================================================ *)
(* A, B, C = static Eshelby + smooth radiation corrections       *)
(* Returns {Atot, Btot, Ctot}                                   *)
(* ============================================================ *)
cubeABC[omega_, a_, alpha_, beta_, rho_, nTaylor_: 8] :=
  Module[{a0k, b0k, As, Bs, Cs, phi, psi, s0, s1, s2, s11,
          Asm, Bsm, Csm, nt = nTaylor},
    a0k = (alpha^2 + beta^2)/(8 Pi rho alpha^2 beta^2);
    b0k = (alpha^2 - beta^2)/(8 Pi rho alpha^2 beta^2);
    (* Static Eshelby depolarization *)
    As = 2 (-a0k j1Cube - 3 b0k j2Cube);
    Bs = 2 b0k (j1Cube - 3 j2Cube);
    Cs = 6 b0k (3 j2Cube - k1Cube);
    (* Smooth Taylor-moment corrections *)
    {phi, psi} = cubeTaylorCoefficients[omega, alpha, beta, rho, nt];
    {s0, s1, s2, s11} = cubeMoments[a, nt - 1];
    (* A_smooth *)
    Asm = If[nt > 2,
      Sum[4 nn (nn - 1) phi[[nn + 1]] s1[[nn - 1]], {nn, 2, nt - 1}] +
      Sum[4 nn (nn - 1) psi[[nn + 1]] s11[[nn - 1]], {nn, 2, nt - 1}], 0];
    If[nt > 1,
      Asm += Sum[2 nn phi[[nn + 1]] s0[[nn]], {nn, 1, nt - 1}] +
             Sum[2 nn psi[[nn + 1]] s1[[nn]], {nn, 1, nt - 1}]];
    (* B_smooth *)
    Bsm = Total[psi s0];
    If[nt > 1,
      Bsm += Sum[4 nn psi[[nn + 1]] s1[[nn]], {nn, 1, nt - 1}]];
    If[nt > 2,
      Bsm += Sum[4 nn (nn - 1) psi[[nn + 1]] s11[[nn - 1]], {nn, 2, nt - 1}]];
    (* C_smooth: depends only on psi *)
    Csm = If[nt > 2,
      Sum[4 nn (nn - 1) psi[[nn + 1]] (s2[[nn - 1]] - 3 s11[[nn - 1]]),
        {nn, 2, nt - 1}], 0];
    N[{As + Asm, Bs + Bsm, Cs + Csm}]
  ];

(* ============================================================ *)
(* Far-field P-wave amplitude (asymptotic Green's tensor)       *)
(* T-matrix force = +V w^2 Drho* u has opposite sign to the    *)
(* Lippmann-Schwinger body force -w^2 drho u, so we negate:    *)
(* f_P = -(rhat.f - ikP sig_RR) / (4 pi rho alpha^2)          *)
(* Returns complex f_P(theta) directly, no finite-R artifacts   *)
(* ============================================================ *)
cubeFarFieldP[theta_, kP_, alpha_, rho_,
              centres_, sources_, voigtPairs_, voigtWeight_] :=
  Module[{rhat, QP = 0. + 0. I, fEff, sigEff, phase, sigRR},
    rhat = {Cos[theta], Sin[theta], 0.};
    Do[
      fEff = sources[[kk, 1 ;; 3]];
      sigEff = sources[[kk, 4 ;; 9]];
      phase = Exp[-I kP (rhat . centres[[kk]])];
      sigRR = Sum[
        voigtWeight[[J1]] rhat[[voigtPairs[[J1, 1]]]]
          rhat[[voigtPairs[[J1, 2]]]] sigEff[[J1]],
        {J1, 6}];
      QP += phase (rhat . fEff - I kP sigRR),
    {kk, Length[centres]}];
    -QP / (4 Pi rho alpha^2)
  ];

Print["CubeAnalytic.wl loaded: cubeGamma0, cubeABC, cubeMoments, cubeFarFieldP"];
