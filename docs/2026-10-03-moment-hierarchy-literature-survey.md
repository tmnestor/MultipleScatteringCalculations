# Literature survey: the moment hierarchy for the voxel (Section 6 and Section 9 of ContinuumLimit.tex)

Date: 3 October 2026. Scope: prior work that anticipates or closely relates to the single-site
hierarchy of Section 6 (Taylor polynomial of degree l about the centre; Lippmann-Schwinger equation
and its derivatives imposed at the centre; distributional moments g[D;W] of the Green's tensor; parity
split; closed-form error; check against the exact sphere) and to its lattice form in Section 9.

Reading levels used below: **FULL** = full text read (PDF in `reference_papers/`); **SUMMARY** = a
full-text paper by other authors that describes the work in detail was read, the work itself was not;
**ABSTRACT** = only the publisher abstract (via the Crossref record) was read; **TITLE** = only
bibliographic metadata was verified. Every DOI below was resolved against Crossref on 3 October 2026.

## Headline

The construction is **not new in its static, stiffness-only form**. It is the Taylor-based
equivalent inclusion method (EIM-T) of Moschovidis and Mura (1975): polynomial eigenstrain about the
centre of each inhomogeneity, consistency (Lippmann-Schwinger) equation Taylor-expanded about the
centre, coefficients of every monomial equated, which is exactly "the equation and its derivatives to
order l imposed at the centre". The coefficients are derivatives at the centre of polynomially weighted
integrals of the Green operator over the body, i.e. static counterparts of g[D;W]. It has been applied
**to a cuboid** (Johnson, Earmme and Lee 1980, Taylor expansion about the origin, edge and corner
fluctuations reported) and **to polyhedra** (Wu, Zhang and Yin 2021, expansion at the centroid to
quadratic eigenstrain, vertex discrepancy reported; this paper is already cited but its EIM content is
not acknowledged). The **dynamic** EIM with polynomial eigenstrains and an additional density
("eigenforce") unknown was formulated by Fu and Mura (1983) for ellipsoids, and its uniform truncation
was compared with the exact sphere (Sheu and Fu 1983, as reported in Fu 1983). Brisard, Dormieux and Sab
(2014) describe EIM-T as a strong-form (Taylor) discretisation of the Lippmann-Schwinger equation and
replace it by a Galerkin form, for exactly the reason Section 9 gives for the Legendre voxel's advantage
on a body with corners.

What survives, to our knowledge, is listed in section (c) below: the elastodynamic hierarchy to
arbitrary degree for a non-ellipsoidal body with density and stiffness together, its moments as
distributions in closed form on four constants including the even-power dynamic terms, the exact
even/odd split and the parity-pair order rule with its closed-form error factor, the degree-by-degree
validation against the exact sphere, and the lattice (voxel) version of the gradient hierarchy.

Two further statements in Section 6 are special cases of known results: the dilatational channel and the
isotropic shear average of the cube equal the sphere's for **any** inclusion shape (the traces S_iijj and
S_ijij of the Eshelby tensor are shape independent; Rodin 1996, already cited, and Zheng, Zhao and Du
2006).

## (a) Table of works

Columns: reference; DOI or URL; read; what it does; relation to the hierarchy; cite?

### 1. Equivalent inclusion method with polynomial eigenstrains (static)

| Reference | DOI / URL | Read | What it does | Relation | Cite? |
|---|---|---|---|---|---|
| Moschovidis, Z. A. and Mura, T. (1975) Two-ellipsoidal inhomogeneities by the equivalent inclusion method. J. Appl. Mech. 42(4), 847-852 | 10.1115/1.3423718 | ABSTRACT + SUMMARY (Brisard et al. 2014, Sec. 2.2, full text) | Two ellipsoidal inhomogeneities; applied strain a polynomial of degree M; eigenstrain a polynomial about each centre; the consistency equation, non-polynomial because of the other inhomogeneity, is Taylor-expanded about each centre and the monomials are equated (Brisard et al. Eqs. 12-17) | **Same idea**, static, stiffness only, ellipsoids (for which the self term of a polynomial eigenstrain is exactly polynomial, so the Taylor step acts only on the interaction) | **MUST** |
| Johnson, W. C., Earmme, Y. Y. and Lee, J. K. (1980) Approximation of the strain field associated with an inhomogeneous precipitate. Part 1: Theory. J. Appl. Mech. 47(4), 775-780 | 10.1115/1.3153789 | ABSTRACT | Two methods for a precipitate of arbitrary shape: an integral equation solved by "a Taylor series expansion about the origin" (and a variant expanded about the field point); and the Moschovidis-Mura equivalency with a polynomial equivalent eigenstrain. Isotropic, static | **Same idea** for a non-ellipsoidal body, static, stiffness only | **MUST** |
| Johnson, W. C., Earmme, Y. Y. and Lee, J. K. (1980) Part 2: The cuboidal inhomogeneity. J. Appl. Mech. 47(4), 781-788 | 10.1115/1.3153790 | ABSTRACT | Both methods applied to a **cuboidal** inhomogeneity; good agreement for moderate shear-modulus mismatch; "fluctuations ... near the cube edge and corner" for large mismatch; error "due to the cutoff of higher-order terms in the Taylor series expansion" | **Same idea on a cube**, static. Directly anticipates the edge/corner floor of Table `tab:hiercube` | **MUST** |
| Wu, C., Zhang, L. and Yin, H. (2021) Elastic solution of a polyhedral particle with a polynomial eigenstrain and particle discretization. J. Appl. Mech. 88(12), 121001 (already cited as `WuZhangYin2021`) | 10.1115/1.4051869 | FULL (accepted manuscript, NSF PAR 10282844; OCR of a font-encoded PDF) | Closed-form polynomial (linear, quadratic) Eshelby tensors for polyhedra; **EIM with the eigenstrain expanded at the centroid and the stress equivalence imposed at the centroid** (Sec. 5.2: "the stress equivalence is satisfied at the centroid (0,0,0) of the tetrahedron, and the accuracy decreases with the distance from the centroid"); vertex singularities make the centroid Taylor series inaccurate, which motivates a tetrahedral discretisation | **Same idea**, static, stiffness only, polyhedra, to quadratic eigenstrain. Its intro cites Zhou et al. 2011 and 2016 for cubic elements with uniform eigenstrain and equivalence "set up at the centers" | **MUST (already cited; description must be widened)** |
| Wu, C. and Yin, H. (2021) Elastic solution of a polygon-shaped inclusion with a polynomial eigenstrain. J. Appl. Mech. 88(6), 061002 | 10.1115/1.4050279 | ABSTRACT (PDF downloaded, font-encoded) | 2-D counterpart; polynomial Eshelby tensors by Hadamard regularisation and contour integrals; EIM with higher-order eigenstrain terms | Same idea, 2-D, static | optional |
| Brisard, S., Dormieux, L. and Sab, K. (2014) A variational form of the equivalent inclusion method for numerical homogenization. Int. J. Solids Struct. 51(3-4), 716-728 | 10.1016/j.ijsolstr.2013.10.037 ; HAL hal-00922779 | FULL | States that EIM "can be seen as the discretization of the Lippmann-Schwinger equation with piecewise polynomials"; that Moschovidis-Mura "discretized the strong form ... through Taylor expansions"; that for EIM-T invertibility is not guaranteed and increasing the degree "does not necessarily improve" accuracy (citing Rodin and Hwang 1991, Fond et al. 2001); proposes a Galerkin (weak-form) EIM with proven convergence in degree and Hashin-Shtrikman bounds | **Partial overlap, and the key framing**: the hierarchy is EIM-T with the field rather than the eigenstrain as unknown; the Legendre voxel is the Galerkin side of the same dichotomy | **MUST** |
| Rodin, G. J. and Hwang, Y.-L. (1991) On the problem of linear elasticity for an infinite region containing a finite number of non-intersecting spherical inhomogeneities. Int. J. Solids Struct. 27(2), 145-159 | 10.1016/0020-7683(91)90225-5 | TITLE (content from Brisard et al.) | Reported, per Brisard et al., that raising the polynomial degree in EIM-T need not improve accuracy | Caveat on monotone improvement | recommended |
| Fond, C., Riccardi, A., Schirrer, R. and Montheillet, F. (2001) Mechanical interaction between spherical inhomogeneities: an assessment of a method based on the equivalent inclusion. Eur. J. Mech. A/Solids 20(1), 59-75 | 10.1016/S0997-7538(00)01118-9 | TITLE (content from Brisard et al.) | As above, an assessment of EIM-T, with examples where a higher degree lowers accuracy (per Brisard et al.) | Caveat | optional |
| Asaro, R. J. and Barnett, D. M. (1975) The non-uniform transformation strain problem for an anisotropic ellipsoidal inclusion. J. Mech. Phys. Solids 23(1), 77-83 | 10.1016/0022-5096(75)90012-5 | TITLE (content from secondary sources) | Polynomial eigenstrain in an anisotropic ellipsoid gives a polynomial strain of the same degree (polynomial conservation) | Background; explains why EIM-T is exact in its self term for ellipsoids and not for a cube | optional |
| Mura, T. and Kinoshita, N. (1978) The polynomial eigenstrain problem for an anisotropic ellipsoidal inclusion. Phys. Status Solidi A 48(2), 447-450 | 10.1002/pssa.2210480222 | TITLE | Same topic | Background | optional |
| Rahman, M. (2002) The isotropic ellipsoidal inclusion with a polynomial distribution of eigenstrain. J. Appl. Mech. 69(5), 593-601 | 10.1115/1.1491270 | ABSTRACT | Explicit interior polynomial strain for polynomial eigenstrain in an isotropic ellipsoid | Background | optional |
| Mura, T. (1987) Micromechanics of Defects in Solids, 2nd edn, Ch. 4 "Ellipsoidal inhomogeneities", pp. 177-239 (already cited as `Mura1987`) | 10.1007/978-94-009-3489-4_4 | TITLE (chapter title and pages from Crossref) | The textbook account of the EIM, including polynomial applied strains and the Moschovidis-Mura method (per Wu et al. 2021, who cite Mura and Moschovidis-Mura for the centroid expansion) | Same idea, textbook | cite the chapter where the EIM-T is mentioned |
| Zhou, K., Keer, L. M. and Wang, Q. J. (2011) Semi-analytic solution for multiple interacting three-dimensional inhomogeneous inclusions of arbitrary shape in an infinite space. Int. J. Numer. Meth. Eng. 87(7), 617-638 | 10.1002/nme.3117 | ABSTRACT + SUMMARY (Wu et al. 2021) | Inhomogeneities cut into **cuboidal elements**, each a cuboidal inclusion with uniform unknown eigenstrain; equivalence at element centres; closed-form cuboid solutions summed by **FFT**; validated against Eshelby's ellipsoid and FEM | **Static counterpart of the uniform-strain voxel scheme** (the l = 1 lattice of Section 9 and the collocation closure) | **MUST** (Section 9) |
| Zhou, Q. et al. (2016) Numerical EIM with 3D FFT for the contact with a smooth or rough surface involving complicated and distributed inhomogeneities. Tribol. Int. 93, 91-103 | not resolved here (cited by Wu et al. 2021) | TITLE | Same method in a half space | Same as above | optional |
| Nakasone, Y., Nishiyama, H. and Nojiri, T. (2000) Numerical equivalent inclusion method: a new computational method for analyzing stress fields in and around inclusions of various shapes. Mater. Sci. Eng. A 285(1-2), 229-238 | 10.1016/S0921-5093(00)00637-7 | TITLE | Numerical EIM with element shape functions, arbitrary shapes | Different (FE-type basis) | optional |
| Shodja, H. M., Rad, I. Z. and Soheilifard, R. (2003) Interacting cracks and ellipsoidal inhomogeneities by the equivalent inclusion method. J. Mech. Phys. Solids 51(5), 945-960 | 10.1016/S0022-5096(02)00106-0 | TITLE (content from Brisard et al.) | Point collocation in place of Taylor expansion | Variant | optional |
| Benedikt, B., Lewis, M. and Rangaswamy, P. (2006) On elastic interactions between spherical inclusions by the equivalent inclusion method. Comput. Mater. Sci. 37(3), 380-392 | 10.1016/j.commatsci.2005.10.002 | TITLE (content from Brisard et al.) | Taylor expansions at points of interest rather than at the centres | Variant | optional |

### 1b. Dynamic equivalent inclusion method

| Reference | DOI / URL | Read | What it does | Relation | Cite? |
|---|---|---|---|---|---|
| Fu, L. S. and Mura, T. (1983) The determination of the elastodynamic fields of an ellipsoidal inhomogeneity. J. Appl. Mech. 50(2), 390-396 | 10.1115/1.3167050 | ABSTRACT + SUMMARY (Fu 1983, NASA CR-3705, full text) | Dynamic EIM for an ellipsoid under plane time-harmonic waves; **two types of eigenstrain** (the second, for the density mismatch, later called "eigenforces"); "expanding the eigenstrains and applied strains in the polynomial form in the position vector r and satisfying the equivalence conditions at every point"; cross sections given for uniform eigenstrains | **Same idea, dynamic, density and stiffness together, ellipsoid**. In the worked cases only the uniform truncation (l = 1) is carried out, with the conditions evaluated at the centre [0] and the frequency series kept to fourth order in ka (NASA CR-3705, Eqs. 20-26) | **MUST** |
| Fu, L. S. and Mura, T. (1982) Volume integrals of ellipsoids associated with the inhomogeneous Helmholtz equation. Wave Motion 4(2), 141-149 | 10.1016/0165-2125(82)90030-0 | TITLE + SUMMARY (NASA CR-3705) | Evaluation of the dynamic volume integrals the 1983 method needs, inside and outside the ellipsoid | Dynamic counterpart of the moments, for an ellipsoid | recommended |
| Fu, L. S. (1983) Scatter of elastic waves by a thin flat elliptical inhomogeneity. NASA Contractor Report 3705 | https://ntrs.nasa.gov/citations/19830021454 | FULL | Applies Fu-Mura with uniform eigenstrains and eigenforces; notes that the coupling terms f_mij[0], F_mij[0] "vanish automatically" so the force (3x3) and strain (6x6) systems uncouple; reports that Sheu and Fu found agreement with the exact sphere "up to ka about two" for uniform eigenstrains | The l = 1 parity split for an ellipsoid, and a uniform-truncation check against the exact sphere | optional (cite Fu and Mura 1983 instead) |
| Sheu, Y. C. and Fu, L. S. (1983) The transmission/scattering of elastic waves by a simple inhomogeneity: a comparison of theories. Review of Progress in QNDE 2, 557-565 | 10.1007/978-1-4613-3706-5_34 | TITLE (content from NASA CR-3705) | Uniform dynamic EIM for a sphere against Ying and Truell | Prior check of the l = 1 closure against the exact sphere | recommended, after reading |
| Hao, L., Zhong, W.-F. and Li, G.-F. (1985) On the method of equivalent inclusions in elastodynamics and the scattering fields of two ellipsoidal inhomogeneities. Appl. Math. Mech. 6(6), 511-521 | 10.1007/BF01876391 | TITLE | Dynamic EIM for two ellipsoids | Probably dynamic Moschovidis-Mura; unread | read before deciding |
| Shodja, H. M. and Delfani, M. R. (2009) 3D elastodynamic fields of non-uniformly coated obstacles: notion of eigenstress and eigenbody-force fields. Mech. Mater. 41(9), 989-999 | 10.1016/j.mechmat.2009.05.005 | TITLE + search summary | Revised dynamic EIM (eigenstress and eigenbody-force fields) with the fields expanded in spherical wave functions | Different basis (spherical wave functions, not a Taylor polynomial) | optional |
| Wheeler, P. and Mura, T. (1973) Dynamic equivalence of composite material and eigenstrain problems. J. Appl. Mech. 40(2), 498-502 | 10.1115/1.3423012 | ABSTRACT | Variational conditions for dynamic equivalence of a periodic composite and eigenstrain/body-force problem | Origin of the dynamic EIM; periodic | optional |
| Mikata and Nemat-Nasser (1990); Michelitsch, Gao and Levin (2003); Wang, Michelitsch, Gao and Levin (2005) (all already cited) | 10.1115/1.2897650 ; 10.1098/rspa.2002.1054 ; 10.1016/j.ijsolstr.2004.06.042 | ABSTRACT (Mikata); TITLE + search summary (others) | Dynamic Eshelby tensors: sphere (closed form), ellipsoid (non-uniform inside), various shapes including cubic and prismatic | Dynamic inclusion problem, uniform eigenstrain; no hierarchy | already cited; descriptions fair |

### 2. Low-frequency scattering hierarchies

| Reference | DOI / URL | Read | What it does | Relation | Cite? |
|---|---|---|---|---|---|
| Stevenson, A. F. (1953) Solution of electromagnetic scattering problems as power series in the ratio (dimension of scatterer)/wavelength. J. Appl. Phys. 24(9), 1134-1142 | 10.1063/1.1721461 | ABSTRACT | Interior and scattered fields as power series in ka, each term a potential problem for the exact body | **Different**: an exact expansion in frequency for the exact shape; the hierarchy truncates the field polynomial in space and is approximate in shape at every order | recommended |
| Dassios, G. and Kleinman, R. (2000) Low Frequency Scattering. Oxford University Press, Oxford | 10.1093/oso/9780198536789.001.0001 | ABSTRACT (book and chapter abstracts) | Unified low-frequency (Rayleigh-Stevenson) theory for acoustics, electromagnetics and elasticity; elastic part credited to Dassios, Kiriaki, Polyzos | Different (as Stevenson), the reference treatment | recommended. Note: the publisher record dates the book 9 December 1999; it is normally cited as 2000 |
| Dassios, G. and Kiriaki, K. (1984) The low-frequency theory of elastic wave scattering. Q. Appl. Math. 42(2), 225-248 | 10.1090/qam/745101 | ABSTRACT | Rigid scatterer and cavity; iterative sequence of potential problems | Different; impenetrable bodies | optional |
| Kiriaki, K. (1989) Low-frequency expansions for a penetrable ellipsoidal scatterer in an elastic medium. J. Eng. Math. 23(4), 295-314 | 10.1007/BF00128904 | TITLE | Penetrable ellipsoid, low-frequency expansions | Different; ellipsoid | optional |
| Datta, S. K. (1977) Diffraction of plane elastic waves by ellipsoidal inclusions. J. Acoust. Soc. Am. 61(6), 1432-1437 ; and A self-consistent approach to multiple scattering by elastic ellipsoidal inclusions. J. Appl. Mech. 44(4), 657-662 | 10.1121/1.381458 ; 10.1115/1.3424153 | ABSTRACT | Matched asymptotic expansions; scattered field to O(ka^3) for an ellipsoidal inclusion | Different method, same order of accuracy as the l = 1 closure | optional |
| Gubernatis, J. E., Krumhansl, J. A. and Thomson, R. M. (1979) Interpretation of elastic-wave scattering theory ...: the long-wavelength limit. J. Appl. Phys. 50(5), 3338-3345 | 10.1063/1.326376 | ABSTRACT | Long-wavelength f-vector of volume flaws, "exactly determinable"; compared with Born | The quasi-static (Eshelby-type) closure in ultrasonics | recommended |
| Korneev, V. A. and Johnson, L. R. (1993) Scattering of elastic waves by a spherical inclusion. I and II. Geophys. J. Int. 115(1), 230-250 and 251-263 | 10.1111/j.1365-246X.1993.tb05601.x ; 10.1111/j.1365-246X.1993.tb05602.x | TITLE | Exact sphere solution and the limits of asymptotic (Born, Rayleigh) solutions | An exact-sphere reference used in seismology | optional |
| Margerin, L. (2011) Mean-field T-matrix approach to elastic wave scattering by small and point-like objects. Waves Random Complex Media 21(4), 628-644 | 10.1080/17455030.2011.613418 | ABSTRACT (from search summary) | Single-inclusion elastic T-matrix with local stress and momentum replaced by volume averages, summing the full series; optical theorem; compared with exact spheres | **Partial overlap** with the l = 1 closure (volume averages rather than centre values) | recommended |

### 3. Elastic T-matrix / volume-integral methods (existing citations checked)

| Reference | Read | Check of the present wording | Verdict |
|---|---|---|---|
| Kanaun and Levin (2008), book | TITLE | "self-consistent or effective-field closure" | fair |
| Kanaun and Levin (2013), Wave Motion 50, 687-707 | TITLE + secondary descriptions | "discretise the inclusion problem with Gaussian approximating functions whose matrix elements are analytic" | consistent with Kanaun's own descriptions of the method; abstract not obtained |
| Jakobsen (2012), Stud. Geophys. Geod. 56, 1-20 | TITLE | Line 263 reads as if both Jakobsen papers are anisotropic elastic. The 2012 paper is **in the acoustic approximation** (its title) | **reword**: "the first in the acoustic approximation, the second in arbitrary anisotropic elastic media with a nine-component displacement-strain state" (the latter verified from the 2020 abstract) |
| Touhei (2011); Yang et al. (2008) | TITLE | as worded | plausible from titles; abstracts not obtained |
| Gubernatis, Domany and Krumhansl (1977), J. Appl. Phys. 48, 2804-2811 | ABSTRACT | Cited (line 621) for the "self-consistent or effective-field closure". Its abstract describes only the integral equation, scattered amplitudes, cross sections and an optical theorem | **not supported by the abstract**: either check the full text or move this citation to the integral equation only and cite Gubernatis, Krumhansl and Thomson (1979) for the long-wavelength closure |
| Lee and Johnson (1978), Phys. Status Solidi A 46, 267-272 | ABSTRACT (Michigan Tech repository) | Grouped (line 613) with works giving the cube's field "in closed form". The abstract says the field of a cuboid in an **anisotropic** matrix "is calculated" from anisotropic Green's functions | **reword or check**: probably numerical, not closed form |
| Nozaki and Taya (2001) | ABSTRACT | "closed form" for polyhedra with uniform eigenstrain | fair ("exact solutions"); its finding that the centre stress of a polyhedron differs from the sphere's except for the dodecahedron and icosahedron is relevant to Eq. `eq:shearsplit` |
| Chiu (1977) | ABSTRACT | closed form for a cuboid | fair ("integrated in closed form") |
| Wang, Michelitsch, Gao and Levin (2005) | TITLE + search summary | "formulated for shapes other than the ellipsoid, cubes among them" | fair at the level read |
| Yurkin and Smunev (2023) | FULL | "integrated analytically over a cuboid" | add "to an error of O((kd)^4)", which they state |
| Mikata and Nemat-Nasser (1990); Michelitsch, Gao and Levin (2003) | ABSTRACT / TITLE | "dynamic generalisation known for spheres and ellipsoids" | fair; add Fu and Mura (1983) as the earliest dynamic EIM |
| Waterman (1976) | ABSTRACT | "represents a scatterer in spherical multipoles" | fair |
| Mura (1987) | TITLE | cited for the ellipsoid only | add Ch. 4 for the EIM with polynomial eigenstrains |
| Line 602-603 "the classical closed form of the cube's depolarisation" | | uncited | cite Chiu (1977) and/or Nozaki and Taya (2001) there |

### 4. Electromagnetic analogues and terminology

| Reference | DOI / URL | Read | What it does | Relation | Cite? |
|---|---|---|---|---|---|
| Yaghjian, A. D. (1980) Electric dyadic Green's functions in the source region. Proc. IEEE 68(2), 248-263 | 10.1109/PROC.1980.11620 | TITLE | Source dyadic (depolarisation dyadic) depending on the shape of the exclusion; cube value 1/3 | EM counterpart of the delta term and the shape dependence of the principal value | recommended at Eq. `eq:sumrule` |
| Harrington, R. F. (1993) Field Computation by Moment Methods. IEEE Press reprint (first published 1968, Macmillan) | 10.1109/9780470544631 | TITLE | Method of moments: expansion functions and testing (weighting) functions | **Terminology**. In MoM terms the hierarchy is a MoM with Taylor monomials as expansion functions and the delta function and its derivatives at the centre as testing functions. "Moments" in the paper means integrals of the Green's tensor against monomials, a different use | recommended (one sentence) |
| Livesay, D. E. and Chen, K.-M. (1974) Electromagnetic fields induced inside arbitrarily shaped biological bodies. IEEE Trans. Microw. Theory Tech. 22(12), 1273-1280 | 10.1109/TMTT.1974.1128475 | TITLE | Classic pulse-basis, point-matching (centre collocation) volume integral method on cubic cells | EM uniform-field voxel collocation | optional |
| Lemaire, T. (1997) Coupled-multipole formulation for the treatment of electromagnetic scattering by a small dielectric particle of arbitrary shape. JOSA A 14(2), 470 | 10.1364/JOSAA.14.000470 | TITLE | Coupled-dipole method extended to electric quadrupoles per cell (per a search summary) | Possible EM analogue of the l = 2 hierarchy as a voxel scheme | **read before deciding**; could be close |
| Smunev, D. A., Chaumet, P. C. and Yurkin, M. A. (2015) Rectangular dipoles in the discrete dipole approximation. JQSRT 156, 67-79 | 10.1016/j.jqsrt.2015.01.019 | TITLE | Cuboid voxels and the integrated Green tensor | Background | optional |
| Chaumet, P. C. (2022) The discrete dipole approximation: a review. Mathematics 10(17), 3049 | 10.3390/math10173049 | FULL (downloaded; searched, not read end to end) | Review; no Taylor or higher-multipole voxel scheme found by text search | Background | optional |
| Ammari, H. and Kang, H. (2007) Polarization and Moment Tensors. Applied Mathematical Sciences 162, Springer, New York | 10.1007/978-0-387-71566-7 | TITLE + chapter titles (Crossref) | Generalised polarisation tensors and **elastic moment tensors** M_{alpha beta}, a hierarchy graded by two multi-indices, defined for any shape through layer potentials; used for asymptotic expansions and imaging | **Same graded object, exact rather than truncated**: the static multipole response of an arbitrary inclusion; the hierarchy's static limit approximates it. Also the main prior use of "moment tensor" for inclusions | **MUST** (terminology and relation) |

### 5. Closed-form integrals of the Green's tensor with polynomial weights over cubes and polyhedra

| Reference | DOI / URL | Read | What it does | Relation | Cite? |
|---|---|---|---|---|---|
| Hackbusch, W. (2002) Direct integration of the Newton potential over cubes. Computing 68(3), 193-216 (MPI Leipzig preprint 68, 2001) | 10.1007/s00607-001-1443-8 | FULL (preprint) | Explicit sixfold integrals of monomial-weighted 1/r over two axis-parallel bricks (Galerkin), and the threefold collocation integrals "in the same way" | **Prior art** for the static scalar moments E[-1; ; W] (no derivatives); our engine adds derivatives as distributions, the r kernel and the dynamic series | **MUST** |
| Waldvogel, J. (1976) The Newtonian potential of a homogeneous cube. Z. Angew. Math. Phys. 27(6), 867-871 ; (1979) The Newtonian potential of homogeneous polyhedra. ZAMP 30(2), 388-398 | 10.1007/BF01595137 ; 10.1007/BF01601950 | TITLE | Closed-form potential of a cube and of polyhedra | Origin of the constants log(2 + sqrt3), pi, sqrt3 in cube potentials | recommended |
| Ren, Z. et al. (2020) Recursive analytical formulae of gravitational fields and gradient tensors for polyhedral bodies with polynomial density contrasts of arbitrary non-negative integer orders. Surv. Geophys. 41(4), 695-722 | 10.1007/s10712-020-09587-4 | ABSTRACT (search summary) | Singularity-free closed forms of the potential, field and gradient tensor (up to two derivatives of 1/r) of polyhedra with polynomial density of any order | **Prior art** for static moments with D <= 2 and any weight, on polyhedra | recommended |
| D'Urso, M. G. and Trotta, S. (2017) Gravity anomaly of polyhedral bodies having a polynomial density contrast. Surv. Geophys. 38(4), 781-832 | 10.1007/s10712-017-9411-9 | TITLE | As above | As above | optional |
| Nagy, D., Papp, G. and Benedek, J. (2000) The gravitational potential and its derivatives for the prism. J. Geodesy 74, 552-560 | 10.1007/s001900000116 | TITLE | Closed-form prism (cuboid) potential and derivatives | Static D <= 2 moments of a cuboid | optional |
| Wilton, D. R. et al. (1984) Potential integrals for uniform and linear source distributions on polygonal and polyhedral domains. IEEE TAP 32(3), 276-281 ; Graglia, R. D. (1987) Static and dynamic potential integrals for linearly varying source distributions .... IEEE TAP 35(6), 662-669 | 10.1109/TAP.1984.1143304 ; 10.1109/TAP.1987.1144160 | TITLE | Closed-form 1/R and R integrals with constant and linear sources on polygons and polyhedra; dynamic case by singularity extraction | CEM counterpart of the kernels r^-1 and r and of the dynamic series | recommended (one citation) |
| Zheng, Q.-S., Zhao, Z.-H. and Du, D.-X. (2006) Irreducible structure, symmetry and average of Eshelby's tensor fields in isotropic elasticity. J. Mech. Phys. Solids 54(2), 368-383 (corrigendum: JMPS 58 (2010) 103-104, 10.1016/j.jmps.2009.11.006) | 10.1016/j.jmps.2005.08.012 | TITLE + secondary summaries | The isotropic part of the Eshelby tensor field of an inclusion of arbitrary shape is uniform and equal to the sphere's; with Rodin (1996), the traces S_iijj and S_ijij are shape independent | **Undermines** the presentation of Eq. `eq:A1g` and the 3:2 average in Eq. `eq:shearsplit` as cube facts: both are consequences of a general shape-independence. The split itself (values of S_off and S_diag) remains the cube's | **MUST** |
| Zou, W.-N., He, Q.-C., Huang, M.-J. and Zheng, Q.-S. (2010) Eshelby's problem of non-elliptical inclusions. J. Mech. Phys. Solids 58(3), 346-372 | 10.1016/j.jmps.2009.11.008 | TITLE | Non-elliptical inclusions; shape-independent invariants (per secondary sources) | As above | optional |

### 6. "Moment hierarchy" / "gradient hierarchy" as terms

Searches for "moment hierarchy" and "gradient hierarchy" (and "hierarchy of gradients") combined with
inclusion, scattering, T-matrix, Eshelby and Lippmann-Schwinger returned no use of either term for this
construction. The nearest established terms are "polynomial eigenstrain" / "equivalent inclusion method"
(mechanics), "generalised polarisation tensors" / "elastic moment tensors" (Ammari and Kang), "influence
pseudotensors" (Brisard et al. for the Moschovidis-Mura coefficients), and "method of moments"
(Harrington, a different meaning). The paper should say once that its "moments" are the polynomial
Eshelby tensors of the EIM literature generalised to elastodynamics and to the density channel, and are
not moments in Harrington's sense.

## (b) Draft sentences for the must-cite works

British spelling; to be adapted to the surrounding text.

1. Section 6 opening (replacing or following the sentence on WuZhangYin2021, around line 615):

   "The construction below is, in its static form, the equivalent inclusion method with polynomial
   eigenstrains of \citet{Moschovidis1975}: the eigenstrain of each inhomogeneity is expanded about its
   centre, the consistency equation is expanded in a Taylor series there, and the coefficients of each
   monomial are equated, which is to impose the Lippmann--Schwinger equation and its derivatives at the
   centre \citep[see][\S2.2]{Brisard2014}. \citet{Johnson1980a,Johnson1980b} applied the same expansion
   about the origin to a cuboidal inhomogeneity, and \citet{WuZhangYin2021} to polyhedra with the
   eigenstrain to second degree; both report that the expansion about the centre fails to follow the
   field near the edges, corners or vertices. \citet{FuMura1983} extended the method to time-harmonic
   waves for an ellipsoid, with a second unknown for the density mismatch, and evaluated it with uniform
   eigenstrains. What is added here is the elastodynamic hierarchy for a body that is not an ellipsoid,
   to any degree, with the moments as distributions in closed form, and the structure and error of its
   truncations."

2. Section 6, "The hierarchy at any degree", after Eq. `eq:hierarchy`:

   "With the field rather than the eigenstrain as unknown, the moments $\mathsf g[D;W]$ are the
   polynomial Eshelby tensors of the equivalent inclusion method \citep[Ch.~4]{Mura1987}, which
   \citet{Brisard2014} call influence pseudotensors, extended to the elastodynamic kernel and to the
   density channel. For an ellipsoid the static self term of a polynomial field is itself a polynomial of
   the same degree, so the expansion about the centre is exact in the self term and approximate only in
   the interaction; for a cube it is approximate in both."

3. Section 6, after "Two independent systems":

   "At first degree the separation is implicit in the dynamic equivalent inclusion method for an
   ellipsoid, where the force and strain systems uncouple at the centre \citep{FuMura1983}; here it holds
   at every degree for any centrosymmetric body, and \S\ref{sec:blocks} shows what it implies for the
   order of convergence."

4. Section 6, "The cube against the sphere" (Eq. `eq:shearsplit`):

   "Both equalities are instances of a general property: the traces $S_{iijj}$ and $S_{ijij}$ of the
   Eshelby tensor do not depend on the shape of the inclusion \citep{Rodin1996}, and its isotropic part
   is the sphere's for an inclusion of any shape \citep{Zheng2006}. What is particular to the cube is the
   cubic part, the split into $S_{\text{off}}$ and $S_{\text{diag}}$."

5. Section 6, "The moments as distributions", where closed forms are introduced:

   "Closed forms for monomial-weighted integrals of $1/r$ over bricks are known from \citet{Hackbusch2002},
   for the Galerkin and the collocation integrals, and closed forms for the potential and its first two
   derivatives over polyhedra with polynomial density from the gravity literature \citep{Ren2020}. The
   engine used here differs in taking derivatives of any order as distributions, in the kernel $r$ as well
   as $1/r$, and in the even-power terms of the dynamic series."

6. Section 6, "What a closed-form single-site T-matrix gives", paragraph "Nothing is fitted" (and the
   abstract's "derived entirely from the physics"):

   "The equivalent inclusion method shares this property \citep{Moschovidis1975,FuMura1983}; what is
   new is that every coefficient is in closed form for the cube, at finite frequency, and that the
   error of each truncation is known."

7. Section 6, "It improves block by block" (caveat):

   "In the Taylor form of the equivalent inclusion method a higher degree does not always improve the
   solution for interacting inhomogeneities, and the linear system is not guaranteed to be invertible
   \citep{Brisard2014}; for the single sphere and cube here every degree to the third improves on the one
   below it, and the systems are well conditioned."

8. Section 9, "The scheme" and "A body with corners":

   "With $\ell=1$ and no density contrast the scheme is the static numerical equivalent inclusion method
   on cuboidal elements with uniform eigenstrain and equivalence at the element centres, summed by the
   fast Fourier transform \citep{Zhou2011}. The advantage of the Legendre voxel on a body with corners is
   the advantage of a Galerkin over a Taylor discretisation of the Lippmann--Schwinger equation, which
   \citet{Brisard2014} established for the static equivalent inclusion method."

9. Terminology (first use of "moment", Section 6):

   "The moments here are integrals of the Green's tensor against monomials over the body; they are not
   moments in the sense of the method of moments \citep{Harrington1993}, of which the hierarchy is the
   instance with Taylor monomials as expansion functions and the delta function and its derivatives at
   the centre as testing functions. The exact static counterparts, for any shape, are the generalised
   polarisation and elastic moment tensors of \citet{AmmariKang2007}."

## (c) What appears not to have been done before, and what undermines a claim

To our knowledge, after this search:

1. **The elastodynamic Taylor (centre-collocation) hierarchy for a non-ellipsoidal body to arbitrary
   degree, with density and stiffness contrasts together.** Static versions for the cuboid (Johnson,
   Earmme and Lee 1980) and polyhedra (Wu, Zhang and Yin 2021) and a dynamic version for the ellipsoid
   (Fu and Mura 1983, evaluated with uniform eigenstrains) exist; none of the works found combines
   finite frequency, a non-ellipsoidal shape and degrees above one or two.
2. **The moments as distributions in closed form on the four constants 1, sqrt3, pi, log(2 - sqrt3),
   through degree 5 and up to six derivatives, for the kernels r^-1 and r, gated by the Laplacian sum
   rules and cross-checked by an independent ball-plus-remainder route.** Closed forms with polynomial
   weights exist for 1/r over bricks without derivatives (Hackbusch 2002) and for up to two derivatives
   over polyhedra (Ren et al. 2020); the general-derivative, distributional, dynamic set was not found.
   The constants themselves are classical for cube potentials (Waldvogel 1976).
3. **The exact even/odd separation at every degree for any centrosymmetric body, read as the
   "parity-pair" order rule (density gains at even l, stiffness at odd l), with the closed-form error
   factor E_d.** Only the l = 1 decoupling for an ellipsoid was found (Fu 1983 report).
4. **Degree-by-degree validation against the exact elastic sphere**, including the closed-form (ka)^2
   departure of the l = 1 closure with exact radiation damping (Eq. `eq:spheredynamic`). A comparison of
   the uniform dynamic EIM with the exact sphere exists (Sheu and Fu 1983, unread; reported as good to
   ka about 2) and Margerin (2011) compares a volume-averaged closure with exact spheres; neither gives
   the departure in closed form or tests higher degrees.
5. **The gradient hierarchy as an elastodynamic voxel scheme (Section 9)**, with a closed-form self block
   and FFT convolution. The static l = 1 version with uniform eigenstrains on cuboids and FFT exists
   (Zhou, Keer and Wang 2011).

What undermines or must qualify a claim:

- The sentence "identifies the uniform-strain closure as one truncation of an exact hierarchy" (line
  630-631) and the abstract's "the single site is the first member of a hierarchy in the gradients of the
  field at the voxel's centre" describe the structure of the Moschovidis-Mura EIM-T. They must be
  attributed, not presented as new.
- "Nothing is fitted" / "derived entirely from the physics" is equally true of the EIM; reword as in (b)6.
- "It improves block by block": not true in general for EIM-T (Rodin and Hwang 1991; Fond et al. 2001,
  per Brisard et al. 2014). The paper's evidence is for a single sphere and a single cube; say so.
- The cube's dilatational channel equalling the sphere's and the isotropic shear average equalling the
  sphere's are shape-independent facts (Rodin 1996; Zheng, Zhao and Du 2006), not properties of the cube.
- The cube floor of Table `tab:hiercube` (edge and corner concentrations defeat any finite Taylor degree)
  was reported for the static cuboid by Johnson, Earmme and Lee (1980) and for polyhedra by Wu, Zhang and
  Yin (2021); cite them there.
- Section 9's conclusion, that a projection follows a non-polynomial field better than an expansion
  about a point, is Brisard et al.'s Galerkin-versus-Taylor result in the static EIM; cite it.
- Not closed: Lemaire (1997, coupled multipoles per cell) and Hao, Zhong and Li (1985, dynamic EIM for
  two ellipsoids) were found by title only and could be closer than they look. Johnson, Earmme and Lee
  (1980) and Fu and Mura (1983) are known from abstracts and a report only; the degree reached in
  Johnson et al. and whether Fu and Mura carried out any non-uniform dynamic case should be checked in
  the full texts before the wording is final.

Limits of the survey: about forty searches; full texts read for Brisard et al. (2014), Wu, Zhang and Yin
(2021), Fu (1983, NASA CR-3705), Hackbusch (2001 preprint) and Yurkin and Smunev (2023); the rest from
publisher abstracts or from descriptions in those full texts, as marked.

## (d) BibTeX for the must-cite and recommended works

Every field below was taken from the Crossref record of the DOI (3 October 2026), except where noted.
Keys are suggestions.

```bibtex
@article{Moschovidis1975,
  author  = {Moschovidis, Z. A. and Mura, T.},
  title   = {Two-ellipsoidal inhomogeneities by the equivalent inclusion method},
  journal = {J. Appl. Mech.},
  volume  = {42},
  number  = {4},
  pages   = {847--852},
  year    = {1975},
  doi     = {10.1115/1.3423718}
}

@article{Johnson1980a,
  author  = {Johnson, W. C. and Earmme, Y. Y. and Lee, J. K.},
  title   = {Approximation of the strain field associated with an inhomogeneous precipitate. {P}art 1: {T}heory},
  journal = {J. Appl. Mech.},
  volume  = {47},
  number  = {4},
  pages   = {775--780},
  year    = {1980},
  doi     = {10.1115/1.3153789}
}

@article{Johnson1980b,
  author  = {Johnson, W. C. and Earmme, Y. Y. and Lee, J. K.},
  title   = {Approximation of the strain field associated with an inhomogeneous precipitate. {P}art 2: {T}he cuboidal inhomogeneity},
  journal = {J. Appl. Mech.},
  volume  = {47},
  number  = {4},
  pages   = {781--788},
  year    = {1980},
  doi     = {10.1115/1.3153790}
}

@article{Brisard2014,
  author  = {Brisard, S. and Dormieux, L. and Sab, K.},
  title   = {A variational form of the equivalent inclusion method for numerical homogenization},
  journal = {Int. J. Solids Struct.},
  volume  = {51},
  number  = {3--4},
  pages   = {716--728},
  year    = {2014},
  doi     = {10.1016/j.ijsolstr.2013.10.037}
}

@article{FuMura1983,
  author  = {Fu, L. S. and Mura, T.},
  title   = {The determination of the elastodynamic fields of an ellipsoidal inhomogeneity},
  journal = {J. Appl. Mech.},
  volume  = {50},
  number  = {2},
  pages   = {390--396},
  year    = {1983},
  doi     = {10.1115/1.3167050}
}

@article{FuMura1982,
  author  = {Fu, L. S. and Mura, T.},
  title   = {Volume integrals of ellipsoids associated with the inhomogeneous {H}elmholtz equation},
  journal = {Wave Motion},
  volume  = {4},
  number  = {2},
  pages   = {141--149},
  year    = {1982},
  doi     = {10.1016/0165-2125(82)90030-0}
}

@article{Zhou2011,
  author  = {Zhou, K. and Keer, L. M. and Wang, Q. J.},
  title   = {Semi-analytic solution for multiple interacting three-dimensional inhomogeneous inclusions of arbitrary shape in an infinite space},
  journal = {Int. J. Numer. Meth. Eng.},
  volume  = {87},
  number  = {7},
  pages   = {617--638},
  year    = {2011},
  doi     = {10.1002/nme.3117}
}

@article{Zheng2006,
  author  = {Zheng, Q.-S. and Zhao, Z.-H. and Du, D.-X.},
  title   = {Irreducible structure, symmetry and average of {E}shelby's tensor fields in isotropic elasticity},
  journal = {J. Mech. Phys. Solids},
  volume  = {54},
  number  = {2},
  pages   = {368--383},
  year    = {2006},
  doi     = {10.1016/j.jmps.2005.08.012}
}

@book{AmmariKang2007,
  author    = {Ammari, H. and Kang, H.},
  title     = {Polarization and Moment Tensors: With Applications to Inverse Problems and Effective Medium Theory},
  series    = {Applied Mathematical Sciences},
  volume    = {162},
  publisher = {Springer},
  address   = {New York},
  year      = {2007},
  doi       = {10.1007/978-0-387-71566-7},
  note      = {Series volume number from library catalogue records, not from Crossref}
}

@article{Hackbusch2002,
  author  = {Hackbusch, W.},
  title   = {Direct integration of the {N}ewton potential over cubes},
  journal = {Computing},
  volume  = {68},
  number  = {3},
  pages   = {193--216},
  year    = {2002},
  doi     = {10.1007/s00607-001-1443-8}
}

@article{Ren2020,
  author  = {Ren, Z. and Chen, C. and Zhong, Y. and Chen, H. and Kalscheuer, T. and Maurer, H. and Tang, J. and Hu, X.},
  title   = {Recursive analytical formulae of gravitational fields and gradient tensors for polyhedral bodies with polynomial density contrasts of arbitrary non-negative integer orders},
  journal = {Surv. Geophys.},
  volume  = {41},
  number  = {4},
  pages   = {695--722},
  year    = {2020},
  doi     = {10.1007/s10712-020-09587-4}
}

@article{Waldvogel1976,
  author  = {Waldvogel, J.},
  title   = {The {N}ewtonian potential of a homogeneous cube},
  journal = {Z. Angew. Math. Phys.},
  volume  = {27},
  number  = {6},
  pages   = {867--871},
  year    = {1976},
  doi     = {10.1007/BF01595137}
}

@article{Stevenson1953,
  author  = {Stevenson, A. F.},
  title   = {Solution of electromagnetic scattering problems as power series in the ratio (dimension of scatterer)/wavelength},
  journal = {J. Appl. Phys.},
  volume  = {24},
  number  = {9},
  pages   = {1134--1142},
  year    = {1953},
  doi     = {10.1063/1.1721461}
}

@book{DassiosKleinman2000,
  author    = {Dassios, G. and Kleinman, R.},
  title     = {Low Frequency Scattering},
  publisher = {Oxford University Press},
  address   = {Oxford},
  year      = {2000},
  doi       = {10.1093/oso/9780198536789.001.0001},
  note      = {The publisher record dates publication 9 December 1999}
}

@article{Gubernatis1979,
  author  = {Gubernatis, J. E. and Krumhansl, J. A. and Thomson, R. M.},
  title   = {Interpretation of elastic-wave scattering theory for analysis and design of flaw-characterization experiments: {T}he long-wavelength limit},
  journal = {J. Appl. Phys.},
  volume  = {50},
  number  = {5},
  pages   = {3338--3345},
  year    = {1979},
  doi     = {10.1063/1.326376}
}

@article{Margerin2011,
  author  = {Margerin, L.},
  title   = {Mean-field {T}-matrix approach to elastic wave scattering by small and point-like objects},
  journal = {Waves Random Complex Media},
  volume  = {21},
  number  = {4},
  pages   = {628--644},
  year    = {2011},
  doi     = {10.1080/17455030.2011.613418}
}

@article{Yaghjian1980,
  author  = {Yaghjian, A. D.},
  title   = {Electric dyadic {G}reen's functions in the source region},
  journal = {Proc. IEEE},
  volume  = {68},
  number  = {2},
  pages   = {248--263},
  year    = {1980},
  doi     = {10.1109/PROC.1980.11620}
}

@book{Harrington1993,
  author    = {Harrington, R. F.},
  title     = {Field Computation by Moment Methods},
  publisher = {IEEE Press},
  address   = {New York},
  year      = {1993},
  doi       = {10.1109/9780470544631},
  note      = {Reprint of the 1968 edition (Macmillan, New York); publisher city from the IEEE reprint, not from Crossref}
}

@article{Graglia1987,
  author  = {Graglia, R. D.},
  title   = {Static and dynamic potential integrals for linearly varying source distributions in two- and three-dimensional problems},
  journal = {IEEE Trans. Antennas Propag.},
  volume  = {35},
  number  = {6},
  pages   = {662--669},
  year    = {1987},
  doi     = {10.1109/TAP.1987.1144160}
}

@article{RodinHwang1991,
  author  = {Rodin, G. J. and Hwang, Y.-L.},
  title   = {On the problem of linear elasticity for an infinite region containing a finite number of non-intersecting spherical inhomogeneities},
  journal = {Int. J. Solids Struct.},
  volume  = {27},
  number  = {2},
  pages   = {145--159},
  year    = {1991},
  doi     = {10.1016/0020-7683(91)90225-5}
}

@incollection{SheuFu1983,
  author    = {Sheu, Y. C. and Fu, L. S.},
  title     = {The transmission/scattering of elastic waves by a simple inhomogeneity: a comparison of theories},
  booktitle = {Review of Progress in Quantitative Nondestructive Evaluation},
  volume    = {2},
  pages     = {557--565},
  publisher = {Springer},
  address   = {Boston, MA},
  year      = {1983},
  doi       = {10.1007/978-1-4613-3706-5_34},
  note      = {Volume number from Fu (1983, NASA CR-3705); not read}
}
```

Already in `references.bib` and to be kept: `WuZhangYin2021`, `Rodin1996`, `Mura1987`, `Nozaki2001`,
`Chiu1977`, `WangMichelitsch2005`, `Mikata1990`, `Michelitsch2003`, `YurkinSmunev2023`.

## PDFs downloaded to `reference_papers/` (git-ignored)

- `Brisard_Dormieux_Sab_2014_Variational_EIM_numerical_homogenization_HAL00922779.pdf` (author version, HAL)
- `Fu_1983_NASA_CR-3705_Scatter_elastic_waves_thin_flat_elliptical_inhomogeneity.pdf` (NASA NTRS)
- `Wu_Zhang_Yin_2021_Polyhedral_particle_polynomial_eigenstrain_JAM_accepted.pdf` (NSF PAR; text layer is font-encoded, read by OCR)
- `Wu_Yin_2021_Polygon_inclusion_polynomial_eigenstrain_JAM.pdf` (NSF PAR; font-encoded, not read)
- `Chaumet_2022_DDA_review_Mathematics_10_3049.pdf` (author copy)

Not obtained (paywalled): Moschovidis and Mura 1975; Johnson, Earmme and Lee 1980 (both parts); Fu and
Mura 1982, 1983; Sheu and Fu 1983; Zhou et al. 2011; Zheng et al. 2006; Ammari and Kang 2007; Lemaire
1997; Hao et al. 1985; Margerin 2011 (a HAL copy exists, hal-00677313, but the server refused automated
access).
