# Literature survey for the octree paper (Paper 4)

- **Date:** 2 October 2026
- **Question:** which parts of the octree paper already exist in the literature, and which appear to be new?
- **Scope and limits:** about a dozen web searches and the abstracts or opening pages of the papers
  below. Only the Hackbusch preprint, Georgakis et al. and Malhotra and Biros were opened beyond the
  abstract. Absence from this survey is not proof of novelty. PDFs are in `reference_papers/` (not
  tracked).

## 1. What already exists

| Topic | Work | What it does | Bearing on our papers |
|---|---|---|---|
| Adaptive tree of polynomial cells for the Lippmann–Schwinger equation, 2-D | Ambikasaran, Borges, Imbert-Gerard and Greengard (2016), arXiv:1505.07157 | Quad tree, p×p Chebyshev grid per leaf, fast direct solver, O(N^3/2). A leaf is split by TWO criteria: at least M points per wavelength, and the contrast and right-hand side resolved to a tolerance by the leaf's polynomial | The tree, the polynomial leaf and a medium criterion plus a wavelength criterion are prior art for the scalar problem. No a priori link between the tolerance and the error of the solution is stated in the abstract or the refinement section |
| Same, accelerated direct solver | Gopal and Martinsson (2022), arXiv:2007.12718 | High-order direct solver in the plane, O(N^3/2) build, O(N log N) solve | Prior art for fast solvers on such discretisations |
| Volume potentials on adaptive octrees, 3-D | Malhotra and Biros (2016), PVFMM; Malhotra's thesis; earlier Ethridge and Greengard (2001), Langston, Greengard and Zorin (2011) | Octree, Chebyshev polynomials of TOTAL degree at most q per leaf, refinement where the tail coefficients exceed a tolerance, kernel-independent fast multipole method, 2:1 balanced tree, precomputed near interactions; Poisson, Stokes, low-frequency Helmholtz; 1.8e10 unknowns | The octree of polynomial cells with a fast multipole product exists and scales. The kernel-independent method could in principle take the elastic kernel. Variable coefficients are solved as an integral equation with an iterative solver |
| Piecewise-smooth media | Anderson, Burbano-Gallegos, Faria and Pérez-Arancibia (2026), arXiv:2608.17615; Anderson, Bonnet, Faria and Pérez-Arancibia (2022), arXiv:2209.03844 | High-order Nyström solver in 2-D on grids fitted to the interfaces of discontinuity; states that jumps of the medium across interfaces defeat high order unless the grid follows them | Agrees with our finding that a jump inside a cell undoes the gain; prior art for that statement in the scalar case |
| Polynomial voxels, Galerkin, uniform grid | Georgakis, Giannakopoulos, Litsarev and Polimeridis (2020), IEEE TAP, arXiv:1902.02196 | Electromagnetic volume integral equation, discontinuous piecewise LINEAR basis on voxels, Galerkin, FFT product; motivated by p-refinement where the grid cannot be refined; compares with piecewise constant | Prior art for the linear voxel with an FFT solver (electromagnetic). The companion paper must cite it |
| Closed-form integrals over boxes | Hackbusch (2001), MPI Leipzig preprint 68 | Explicit expressions for the SIXFOLD integrals of monomials times 1/r over two axis-parallel bricks, by antiderivatives and recursions, including identical, face-, edge- and vertex-touching boxes; program included | Direct prior art for the static master integrals of the moments paper (Paper 2). Ours adds the elastic tensor kernels (derivatives of r and 1/r), the series in the wavenumber and the reduction to faces and edges; Paper 2 must cite and distinguish it |
| Analytical integration of the Green's tensor over a cube | Yurkin and Smunev (2023) | Integrated Green's tensor of the discrete dipole approximation in closed form | Bears on Papers 1 and 2 |
| Analysis of the discrete dipole approximation | Yurkin, Maltsev and Hoekstra (2006), parts I and II; Costabel, Dauge and Nedaiasl (2023), arXiv:2302.13159; Costabel (2023), arXiv:2303.13693 | Error bounds quadratic in the cell size with a linear shape term, and extrapolation; the method as a finite section of a block Toeplitz matrix, stable for most but not all parameters; convergence O(h^2) in the interior with a boundary layer | Rigorous counterparts of Paper 1's closed-form constants. Paper 1 cites Yurkin; the Costabel papers should be added |
| Polynomial bases on trees as wavelets | Alpert (1993); Alpert, Beylkin, Coifman and Rokhlin (1993) | Multiwavelets built from Legendre polynomials on dyadic intervals; integral operators sparse in them; O(n log^2 n) for second-kind equations | The Legendre polynomials on the cells of a tree are Alpert's scaling functions. Prior art for the wavelet reading of the octree |
| Elastic volume integral equation in seismology | Shekhar, Jakobsen, Iversen, Berre and Radu (2023), arXiv:2301.12836; Jakobsen and co-workers on renormalised scattering series | Elastic and anisotropic Lippmann–Schwinger equation on a uniform grid, FFT, matrix-free iterative solve | The nearest work in our own field: uniform grid, low order, no error law, no adaptivity |

## 1b. Second pass: higher-order and adaptive volume methods in electromagnetics; elastic fast multipole methods

| Topic | Work | What it does | Bearing on our papers |
|---|---|---|---|
| SEPARATE orders for the medium and the field | Chobanyan, Ilić and Notaroš (2015), Radio Science 50, 406; the finite-element precedent Ilić et al. (2009) | Volume integral equation, Galerkin, large curved hexahedra. The permittivity inside an element is a Lagrange polynomial of orders (Mu, Mv, Mw), "entirely independent" of the current-expansion orders (Nu, Nv, Nw) and of the geometrical orders; all three may differ from element to element. States that inhomogeneous elements pay off only with higher-order fields in large elements. Dense direct solver, no error law | DIRECT prior art for Paper 1's framing that a cell carries two bases, one for the medium and one for the field, chosen independently. What is ours is the law: the order min(2p+2, 2r+2), the projection-error equality and the closed-form constants. Papers 1, 3 and 4 must cite it |
| Higher-order large-domain volume elements, hp-refinement | Chobanyan, Ilić and Notaroš (2013), IEEE TAP 61, 6051; Notaroš (2008) review | Hierarchical polynomial bases of arbitrary order on hexahedra of up to two wavelengths; element sizes and orders mixed in one model, "enabling hp-refinement" | Prior art for large cells with rich fields in electromagnetics; no tree, no error indicator |
| Error estimation and adaptive refinement for integral equations | Harmon, Key and Notaroš (adjoint-based, goal-oriented, surface equations and finite elements); an adaptive finite-element method for the electromagnetic volume integral equation with a posteriori estimates (J. Comput. Phys. 2022) | A POSTERIORI estimates: the error is estimated from a computed solution, then the mesh is refined and the problem solved again | Ours is a priori: known before any solve. This is the distinction to state |
| Fast multipole for volume integral equations, electromagnetic | Multilevel fast multipole algorithm applied to volume equations with low-order bases on tetrahedra (several groups) | O(N log N) products for volume unknowns | The fast product for volume equations is routine in electromagnetics |
| Elastic fast multipole methods | Fujiwara (2000), Geophys. J. Int. 140, 198; Chaillat, Bonnet and Semblat (2008); Tong and Chew (multilevel algorithm with separate trees for P and S); Chebyshev kernel-interpolation methods; Chaillat, Desiderio and Ciarlet (2017), hierarchical matrices | All for BOUNDARY integral equations: surfaces, topography, basins, inclusions with homogeneous interiors. Up to about 1e6 unknowns. The two wavenumbers are handled by two trees or by a kernel-independent scheme | The elastic kernels have been put through fast multipole and hierarchical-matrix methods many times, on surfaces. This is the machinery a volume solver would reuse |
| Kernel-independent fast multipole | Ying, Biros and Zorin (2004) | Needs only kernel evaluations; tested on the Navier (elastostatic) kernels | With Malhotra and Biros (2016) it gives a route to an elastic volume fast multipole method without new expansions |
| Elastic volume integral equations | Kanaun and Levin (Gaussian approximating functions); Shekhar, Jakobsen et al. (2023, FFT, uniform grid); Lai and Zhang (2022) and Zhang, He and Lai (2026) for many particles | Volume or multi-particle elastic scattering with FFT or particle fast multipole acceleration | No search result showed a fast multipole method for an elastic VOLUME integral equation with polynomial cells on a tree |
| The competing route | Wang, de Hoop, Xia and Li (2012) | Time-harmonic elastic waves in 3-D anisotropic media by finite differences and a structured multifrontal direct solver | The differential-equation competitor for the series goal |

## 2. What was not found

- An error LAW for a volume-integral scheme: the statement that the error beyond first order equals the
  nonlinear fraction of the response times the relative projection error of the medium, with a measured
  constant. The adaptive solvers above refine to a tolerance on the polynomial tail of the contrast; none
  of the pages read connects that tolerance to the error of the scattered field.
- The first-order error computed without a solve, and the closed-form factor of a leaf in spherical
  Bessel functions, with its reduction to the layer's constants.
- The matched-degree rule min(2p+2, 2r+2). Separate bases for the medium and the field exist
  (Chobanyan, Ilić and Notaroš 2015); the rule that relates them, and the demonstration that a constant medium with a richer field is worse on a smooth medium and far better on
  a blocky one.
- An elastic volume-integral scheme with polynomial cells, on a tree or not.

The general theory behind the law is standard: a Galerkin solution of a second-kind equation is
quasi-optimal, its error bounded by the best approximation of the solution. Our law is a sharpened,
quantitative instance (an equality with a known constant, for the part of the field that copies the
medium). It should be presented as that, not as a new kind of result.

## 3. Consequences

1. The octree, the polynomial leaves, refinement by resolution of the medium and of the wave, and a
   scalable fast multipole solver are established for scalar problems. The octree paper must not claim
   them. Its contribution is the error known in advance, and the elastic case.
2. The scale question has a known answer: the volume fast multipole method on a balanced octree. This is
   engineering with a published route and an open-source library, not an open research problem.
3. Paper 2 (moments) has direct prior art in Hackbusch (2001) for the scalar static kernel and must be
   positioned against it.
4. To be read in full before any claim is written: Ambikasaran et al. (2016) for any error estimate;
   Malhotra's thesis for the variable-coefficient Helmholtz solver and its error control; Langston,
   Greengard and Zorin (2011); Costabel, Dauge and Nedaiasl (2023).
5. Paper 1 must cite Chobanyan, Ilić and Notaroš (2015) where it introduces the two bases, and say
   that the independence of the two orders is theirs and the law relating them is ours.
6. An elastic volume fast multipole method appears to be open, but only as an assembly of published
   parts: the volume method on an octree and the elastic kernels in boundary methods. It would be
   engineering, with modest novelty.
7. Not obtained (paywalled or blocked): Chaillat, Bonnet and Semblat (2008); Fujiwara (2000); Tong and
   Chew; Langston, Greengard and Zorin (2011); the 2022 adaptive finite-element paper.
