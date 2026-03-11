# How AI-Inspired Architecture Changes Are Making Volume Integral Equations Computationally Feasible

## 1. Introduction: The VIE Promise and Its Historical Barrier

Volume integral equation (VIE) methods model scattering by discretizing the
**entire volume** of the heterogeneity, not just its surface. The scattered field
is expressed as a convolution of the Green's function with the material contrast
over the scatterer volume:

$$\mathbf{u}^{\text{scat}}(\mathbf{x}) = \int_V G(\mathbf{x}, \mathbf{x}') \, \delta C(\mathbf{x}') \, \mathbf{u}(\mathbf{x}') \, dV'$$

This formulation is preferred by physically oriented scientists because it:

- Directly encodes the **physics of wave-matter interaction** (contrast sources)
- Naturally handles **arbitrary heterogeneity** without re-meshing
- Provides **exact radiation conditions** at infinity (no absorbing boundaries)
- Yields the **T-matrix** and effective medium properties by construction

However, VIE methods have been impractical for large-scale problems due to
their computational demands:

| Operation | Scaling | Memory |
|---|---|---|
| Dense matrix fill | O(N^2) | O(N^2) |
| Direct solve (LU) | O(N^3) | O(N^2) |
| Matrix-vector product | O(N^2) per iteration | O(N^2) |
| FFT-accelerated matvec | O(N log N) per iteration | O(N) |

where N is the number of volume elements. For a 3D problem on a grid of side L
voxels, N ~ L^3 --- even with FFT acceleration, N grows rapidly.

Three converging developments --- all driven by the AI revolution --- are now
removing these barriers.

---

## 2. GPU Hardware: The AI Dividend for Dense Linear Algebra

### 2.1 The FLOPS Explosion

AI training demands dense matrix-matrix multiplication (GEMM) at enormous
scale. This drove GPU architectures to deliver extraordinary floating-point
throughput:

| GPU (Year) | FP64 TFLOPS | HBM Capacity | Memory Bandwidth |
|---|---|---|---|
| Tesla K80 (2014) | 1.9 | 24 GB GDDR5 | 480 GB/s |
| Tesla P100 (2016) | 4.7 | 16 GB HBM2 | 732 GB/s |
| Tesla V100 (2017) | 7.8 | 32 GB HBM2 | 900 GB/s |
| A100 (2020) | 9.7 | 80 GB HBM2e | 2.0 TB/s |
| H100 (2022) | 60 (Tensor) | 80 GB HBM3 | 3.35 TB/s |
| H200 (2024) | 60 (Tensor) | 141 GB HBM3e | 4.8 TB/s |
| Roadmap (Rubin) | -- | up to 1 TB HBM4e | -- |

**Reference:** [NVIDIA Hopper Architecture In-Depth](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)

A VIE problem with N = 50,000 unknowns requires ~20 GB for the dense matrix in
FP64 --- this now fits comfortably on a single GPU. Problems with N = 200,000
require ~300 GB, feasible on a 4-GPU H200 node.

### 2.2 Tensor Cores and Mixed-Precision Iterative Refinement

NVIDIA Tensor Cores were designed for AI matrix operations in FP16/TF32, but a
breakthrough by Haidar et al. (2018, 2020) showed they can accelerate
**scientific FP64 solves** via mixed-precision iterative refinement:

1. Factor the matrix **A** in low precision (FP16 or TF32) using Tensor Cores
2. Solve the correction equation: A c_i = b - A x_i
3. Update: x_{i+1} = x_i + c_i
4. Repeat until backward error reaches FP64 tolerance

This achieves **4--5x speedup** and **5x energy efficiency** over pure FP64,
with no loss of numerical accuracy.

**References:**
- Haidar, A., et al. "Harnessing GPU Tensor Cores for Fast FP16 Arithmetic to Speed up Mixed-Precision Iterative Refinement Solvers," *SC'18*. [(PDF)](https://www.netlib.org/utk/people/JackDongarra/PAPERS/haidar_fp16_sc18.pdf)
- Haidar, A., et al. "Mixed-precision iterative refinement using tensor cores on GPUs to accelerate solution of linear systems," *Proc. R. Soc. A*, 476, 2020. [(DOI)](https://royalsocietypublishing.org/doi/10.1098/rspa.2020.0110)
- [Using Tensor Cores for Mixed-Precision Scientific Computing (NVIDIA Blog)](https://developer.nvidia.com/blog/tensor-cores-mixed-precision-scientific-computing/)

These developments are released in the **MAGMA** and **CUSOLVER** libraries,
making them immediately accessible to VIE codes.

### 2.3 TPU Pods for Extreme-Scale Dense Algebra

Google demonstrated that TPU pods (2048 TPU v3 cores) can multiply two matrices
of size **1,000,000 x 1,000,000 in 2 minutes** --- a scale that makes even
brute-force VIE thinkable for very large problems.

**Reference:** [Tensor Processing Units as Scientific Supercomputers (UC Berkeley)](https://physics.berkeley.edu/news/tensor-processing-units-tpus-scientific-supercomputers)

---

## 3. FFT-Accelerated VIE on GPUs: O(N log N) on Massively Parallel Hardware

### 3.1 The CG-FFT Framework

For **translation-invariant** Green's functions on regular grids, the
matrix-vector product in the VIE is a **discrete convolution**. By the
convolution theorem, it can be evaluated via the FFT:

1. **Embed** the Green's tensor in a doubled grid (to avoid circular convolution artifacts)
2. **Pre-compute** the FFT of the Green's tensor (once)
3. Each matrix-vector product in the iterative solver becomes:
   - Forward FFT of the current field estimate
   - Pointwise multiply with pre-computed Green's tensor spectrum
   - Inverse FFT

This reduces the cost from O(N^2) to **O(N log N)** per iteration, and memory
from O(N^2) to **O(N)**.

**Reference:** Bleszynski, M., E. Bleszynski, and T. Jaroszewicz. "A CG-FFT
approach to the solution of a stress-velocity formulation of three-dimensional
elastic scattering problems," *J. Comput. Phys.*, 227(12), 5597--5614, 2008.
[(DOI)](https://www.sciencedirect.com/science/article/abs/pii/S0021999108004117)

### 3.2 cuFFT: GPU-Native FFT

NVIDIA's cuFFT library provides GPU-accelerated FFT that adds another 10--100x
on top of algorithmic acceleration:

- **cuFFTDx** (device extensions) provides >2x speedup over host-call cuFFT for
  convolution workloads
- Multi-GPU FFT support for problems exceeding single-GPU memory
- Native complex-to-complex 3D transforms perfectly matched to VIE convolutions

**Reference:** [cuFFT Documentation (NVIDIA)](https://docs.nvidia.com/cuda/cufft/)

### 3.3 IE-FFT for Non-Uniform Geometries

For VIE formulations that do not naturally live on a regular grid, the IE-FFT
algorithm projects near-field and far-field interactions onto a uniform auxiliary
grid, recovering FFT acceleration even for non-conformal meshes.

**Reference:** Seo, S. M. and J.-F. Lee. "IE-FFT Algorithm for a Nonconformal
Volume Integral Equation for Electromagnetic Scattering from Dielectric
Objects," *IEEE Trans. Antennas Propag.*, 56(10), 3261--3266, 2008.
[(IEEE Xplore)](https://ieeexplore.ieee.org/document/4526967/)

---

## 4. Gabor Frames: A Phase-Space Bridge Between Spatial and Spectral Domains

### 4.1 Why Gabor Frames Matter for VIE

The standard FFT-accelerated VIE assumes a uniform background or at most a
layered medium. Real heterogeneous media --- geological formations, biological
tissue, metamaterials --- have spatially varying properties that interact with
waves **locally**. This is where Gabor frames offer a fundamental advantage.

A **Gabor frame** decomposes a function into a set of basis elements that are
simultaneously localized in **both space and wavenumber** (the phase-space
lattice):

$$f(\mathbf{x}) = \sum_{m,n} c_{m,n} \, g(\mathbf{x} - m\Delta x) \, e^{i n \Delta k \cdot \mathbf{x}}$$

where g is a Gaussian window function and the lattice {m Delta x, n Delta k} is
**overcomplete** (more basis functions than strictly necessary). This
overcompleteness is not a bug --- it is the key feature:

- **Stable coefficients**: The dual window is smooth and localized (unlike the
  Gabor basis itself in the critically-sampled case)
- **Sparse representation**: Smooth heterogeneity concentrates energy in a small
  number of Gabor coefficients
- **Natural multi-scale**: Different scales of heterogeneity are captured at
  different lattice points
- **Fourier transform = transposition**: Converting between spatial and spectral
  domains reduces to a **reindexing of coefficients**, not a full FFT ---
  achieving O(N) domain transformation

### 4.2 The Eindhoven Spatial-Spectral VIE Program

A systematic program at Eindhoven University of Technology (van Beurden, Dilz,
Eijsvogel, and collaborators) has developed the Gabor-frame VIE into a mature
computational tool:

#### Foundational 2D formulation (2017)

Dilz and van Beurden introduced the Gabor-frame discretization for the 2D TM
scattering VIE in layered media, showing that the spectral-domain Green's
function (with its poles and branch cuts) could be represented on a **complex
integration manifold** that avoids all singularities.

**References:**
- Dilz, R. J., M. G. M. M. van Kraaij, and M. C. van Beurden. "2D TM
  scattering problem for finite dielectric objects in a dielectric stratified
  medium employing Gabor frames in a domain integral equation," *J. Opt. Soc.
  Am. A*, 34(8), 1315--1321, 2017.
  [(Optica)](https://opg.optica.org/josaa/abstract.cfm?uri=josaa-34-8-1315)
- Dilz, R. J. and M. C. van Beurden. "A domain integral equation approach for
  simulating two dimensional transverse electric scattering in a layered medium
  with a Gabor frame discretization," *J. Comput. Phys.*, 345, 528--542, 2017.
  [(ScienceDirect)](https://www.sciencedirect.com/science/article/pii/S0021999117304096)

#### 3D extension (2018)

The method was generalized to full 3D vector electromagnetic scattering with the
**Li factorization rules** --- a critical innovation that correctly handles the
discontinuity of both permittivity and the electric field at material
interfaces, which otherwise causes poor spectral convergence.

The key idea: replace the direct permittivity-field product (both discontinuous
at the same location) with an auxiliary field **F** constructed via
normal-vector fields, so that only products of one continuous and one
discontinuous quantity appear. This satisfies the Li rules and restores fast
convergence.

Results on three test cases showed O(N log N) scaling with relative errors of
4x10^-5 (small cube), 2.8x10^-3 (cylinder), and 2.5x10^-3 (finite grating),
validated against commercial FEM (JCMWave).

**Reference:** Dilz, R. J., M. G. M. M. van Kraaij, and M. C. van Beurden. "A
3D spatial spectral integral equation method for electromagnetic scattering from
finite objects in a layered medium," *Opt. Quantum Electron.*, 50, 206, 2018.
[(PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC6445559/)

#### Fast and accurate Gabor coefficient computation (2022)

Eijsvogel et al. presented two semi-analytical methods for computing Gabor
coefficients of discontinuous 3D scatterers, achieving **super-exponential
convergence** and drastically reducing pre-processing time (from 4058s to 181s).

**References:**
- Eijsvogel, S., L. Sun, F. Sepehripour, R. J. Dilz, and M. C. van Beurden.
  "Describing discontinuous finite 3D scattering objects in Gabor coefficients:
  fast and accurate methods," *J. Opt. Soc. Am. A*, 39(1), 86--97, 2022.
  [(Optica)](https://opg.optica.org/josaa/abstract.cfm?uri=josaa-39-1-86)
- Eijsvogel, S., R. J. Dilz, and M. C. van Beurden. "Computing Gabor
  Coefficients for a Scattering Problem: Super Exponential Converging Accuracy
  and a More Localized Representation," *ICEAA 2022*, Cape Town, 2022.
  [(IEEE Xplore)](https://ieeexplore.ieee.org/document/9899938)

#### Parallel 3D VIE (2023)

The Gabor-frame VIE was parallelized by decomposing the matrix-vector product
into independent components. Scattering problems involving volumes up to
**1300 cubic wavelengths** with numerous finite scatterers were solved with
significant wall-clock time reduction using shared-memory OpenMP parallelism.

**Reference:** Eijsvogel, S., R. J. Dilz, and M. C. van Beurden. "A Parallel
3D Spatial Spectral Volume Integral Equation Method for Electromagnetic
Scattering from Finite Scatterers," *PIER B*, 102, 1--17, 2023.
[(PIER)](https://www.jpier.org/issues/volume.html?paper=23060708)

#### Inverse scattering extension (2024)

The parametrized spatial-spectral VIE was applied to **phaseless inverse
scattering** in the soft X-ray regime, demonstrating the framework's utility
beyond forward modeling.

**Reference:** van Beurden, M. C. et al. "Phaseless inverse scattering with a
parametrized spatial spectral volume integral equation for finite scatterers in
the soft x-ray regime," *J. Opt. Soc. Am. A*, 41(11), 2076, 2024.
[(Optica)](https://opg.optica.org/josaa/abstract.cfm?uri=josaa-41-11-2076)

### 4.3 Gabor Frames in Seismic Wave Propagation

Gabor frames and Gaussian beam summation have a parallel and longer history in
seismic wave propagation, where they address precisely the same physical
challenge: representing wave fields in **heterogeneous media** where ray theory
breaks down.

#### Theoretical foundations (2003)

Lugara and Letrou established the frame-theoretic foundation for Gaussian beam
summation, showing that the overcomplete frame removes the instability and
nonlocality of Gabor expansion coefficients. The overcompleteness parameter
becomes a **design parameter** controlling phase-space locality.

**Reference:** Lugara, D. and C. Letrou. "Frame-based Gaussian beam summation
method: Theory and applications," *Radio Science*, 38(2), 2003.
[(Wiley)](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2001RS002593)

#### Seismic applications (2008--2016)

- **Nowack (2008)** applied frame-based Gaussian beam summation to seismic head
  waves, showing that the overcomplete representation naturally generates both
  wide-angle reflections and head waves.
  [(PDF, Purdue GMIG)](https://gmig.science.purdue.edu/pdfs/2008/08-05.pdf)

- **Nowack (2011)** extended this to dynamically focused Gaussian beams for
  seismic imaging.
  [(Wiley)](https://onlinelibrary.wiley.com/doi/10.1155/2011/316581)

- **Huang, Sun, and Sun (2016)** derived a Born modeling formula using Gaussian
  beam summation for the Green's function in heterogeneous media, resolving
  caustics, shadow zones, and multi-pathing that defeat geometric ray theory.
  [(ScienceDirect)](https://www.sciencedirect.com/science/article/abs/pii/S0926985116301604)

#### Gaussian-windowed MoM for surface integrals (2016)

Shlivinski projected surface integral equations onto an overcomplete Gabor-type
frame set with Gaussian windows. On the homogeneous side, this renders the
scattered field as a **phase-space summation of Gaussian beams**; on the
waveguide side, it describes local coupling to waveguide modes.

**Reference:** Shlivinski, A. "Gaussian-windowed frame based method of moments
formulation of surface-integral-equation for extended apertures," *J. Comput.
Phys.*, 315, 2016.
[(ScienceDirect)](https://sciencedirect.com/science/article/pii/S0021999115008475)

### 4.4 Why Gabor Frames Are Natural for Heterogeneity

The deep reason Gabor frames excel for heterogeneous media is **sparsity in
phase space**. A heterogeneous medium has structure at multiple scales:

- **Large-scale** trends (background velocity gradient) concentrate energy at
  low spatial frequencies
- **Small-scale** fluctuations (grain boundaries, fractures) contribute at high
  wavenumbers but are spatially localized
- **Interfaces** are localized in space but broad in wavenumber

A Gabor frame captures all three regimes simultaneously. The coefficient
c_{m,n} is large only when there is energy at position m*Delta_x AND
wavenumber n*Delta_k. For typical geological or material heterogeneity, the
Gabor representation is **far sparser** than either a purely spatial (voxel)
or purely spectral (plane-wave) representation.

This sparsity translates directly to computational savings: fewer significant
coefficients means smaller effective matrices, faster convergence of iterative
solvers, and more efficient parallelization.

---

## 5. AI-Native Software: Neural Surrogates and Physics-Constrained Solvers

### 5.1 Physics-Embedded Neural Networks for VIE

The conjugate gradient iteration for solving the VIE can be **unrolled** into a
deep neural network, where the Green's function acts as an explicit physics
operator and the FFT accelerates volume integrations within each layer.

**Reference:** Guo, R., T. Lin, and X. Yang. "Physics Embedded Deep Neural
Network for Solving Volume Integral Equation: 2-D Case," *IEEE Trans. Antennas
Propag.*, 69(10), 2021.
[(IEEE Xplore)](https://ieeexplore.ieee.org/document/9397371)

### 5.2 VIE-NN: Neural Networks Constrained by VIE Physics (2025)

The most recent development uses the VIE (via discrete dipole approximation) as
a **physics constraint** to train a neural network. The network weights are
optimized to satisfy the VIE instead of solving the linear system iteratively
--- eliminating the need to pre-compute training data pairs.

**Reference:** "Acceleration of Solving Volume Integral Equations through
Neural-Network-Assisted Approach," *PIER*, 183, 2025.
[(PIER)](https://www.jpier.org/ac_api/download.php?id=25012103)

### 5.3 Physics-Informed Neural Operators (PINO)

Neural operators learn the **solution map** (Green's function -> scattered
field) over a family of scatterer configurations. Once trained, inference is
near-instantaneous, making them ideal for:

- **Inverse problems** (thousands of forward solves needed)
- **Uncertainty quantification** (Monte Carlo over random media)
- **Real-time monitoring** applications

Recent advances include **Separable Physics-Informed DeepONet** (100x training
speedup for 4D problems) and **VINO** (Variational Physics-Informed Neural
Operator) that reduces dependence on paired training data.

**References:**
- Li, Z. et al. "Physics-Informed Neural Operator for Learning Partial
  Differential Equations," *J. Data Science*, 2024.
  [(ACM)](https://dl.acm.org/doi/10.1145/3648506)
- Mandl, L. et al. "Separable Physics-Informed DeepONet," *arXiv*, 2024.
  [(arXiv)](https://arxiv.org/html/2511.04576v1)
- "Variational Physics-informed Neural Operator (VINO) for solving partial
  differential equations," *CMAME*, 2025.
  [(ScienceDirect)](https://www.sciencedirect.com/science/article/pii/S004578252500057X)

### 5.4 Neural-Network Modified Contrast Operators (2025)

Coarse VIE discretizations introduce errors in the contrast operator. Trainable
neural networks can learn a **modified contrast operator** that achieves the
accuracy of fine discretization at the computational cost of coarse
discretization.

**Reference:** *PIER*, 2025.
[(PIER)](https://www.jpier.org/issues/volume.html?paper=25082502)

---

## 6. Synthesis: The Convergence

The following table maps VIE bottlenecks to AI-era solutions:

| VIE Bottleneck | AI-Era Solution | Speedup Factor |
|---|---|---|
| Dense matrix-vector product | GPU Tensor Cores + HBM bandwidth | 10--100x |
| O(N^2) scaling per iteration | FFT convolution on GPU (cuFFT) | N / log N |
| Memory wall (dense matrix) | 80--192 GB HBM per GPU, multi-GPU | 10--50x |
| Iterative solver convergence | Neural operator surrogates | 1000x+ |
| Pre-processing (geometry) | Semi-analytical Gabor coefficients | 20x |
| Multi-GPU communication | NVLink / NVSwitch (AI interconnect) | 10x |
| Spectral singularities | Complex-contour Gabor-frame VIE | exact |
| Interface discontinuities | Li factorization + normal-vector fields | convergence restored |

**The key insight:** None of these technologies were developed *for* VIE. They
were developed for training large language models and vision transformers. But
the computational primitives are identical: dense matrix operations,
FFT-based convolutions, and iterative solvers on massive grids. VIE is a
**free rider** on the AI hardware and software revolution.

### Gabor frames occupy a special position in this convergence

They were originally motivated by **physical** considerations (wave propagation
in heterogeneous media, phase-space representations in quantum mechanics and
seismology) long before the AI era. What AI provides is the **hardware** to
exploit their theoretical advantages at scale: the massively parallel
architectures that can evaluate thousands of Gabor coefficients simultaneously,
the high-bandwidth memory that can hold the phase-space representation, and the
software ecosystem (CUDA, cuFFT, CuPy, PyTorch) that makes GPU programming
accessible.

---

## 7. Implications for Elastic Wave Scattering

For the elastic VIE (Lippmann-Schwinger equation for seismic/ultrasonic
scattering), these developments are directly applicable:

1. **FFT-accelerated GMRES** on GPU: The convolution structure of the elastic
   Green's tensor on a regular grid maps directly to cuFFT. This is already
   implemented in the `sphere_scattering_fft.py` module of this project.

2. **Gabor-frame VIE for layered elastic media**: The Eindhoven framework could
   be adapted from Maxwell's equations to elastodynamics. The Green's tensor
   for layered elastic media has the same pole/branch-cut structure as the EM
   case, and the Li factorization rules have direct analogues for elastic
   displacement-traction continuity.

3. **Mixed-precision solvers**: The MAGMA/CUSOLVER mixed-precision iterative
   refinement is directly applicable to the dense blocks that arise in
   multi-scatterer VIE (e.g., Lax-Foldy with FFT-accelerated self-interaction).

4. **Neural operator surrogates**: For effective medium computations requiring
   ensemble averages over random scatterer configurations, a neural operator
   trained on VIE solutions could replace thousands of individual solves.

---

## References (Consolidated)

### GPU Hardware and Mixed-Precision Linear Algebra
1. [NVIDIA Hopper Architecture In-Depth](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)
2. Haidar, A., et al. "Harnessing GPU Tensor Cores for Fast FP16 Arithmetic to Speed up Mixed-Precision Iterative Refinement Solvers," *SC'18*, 2018. [(PDF)](https://www.netlib.org/utk/people/JackDongarra/PAPERS/haidar_fp16_sc18.pdf)
3. Haidar, A., et al. "Mixed-precision iterative refinement using tensor cores on GPUs," *Proc. R. Soc. A*, 476, 2020. [(DOI)](https://royalsocietypublishing.org/doi/10.1098/rspa.2020.0110)
4. [Using Tensor Cores for Mixed-Precision Scientific Computing (NVIDIA)](https://developer.nvidia.com/blog/tensor-cores-mixed-precision-scientific-computing/)
5. [Tensor Processing Units as Scientific Supercomputers (UC Berkeley)](https://physics.berkeley.edu/news/tensor-processing-units-tpus-scientific-supercomputers)

### FFT-Accelerated VIE
6. Bleszynski, M., E. Bleszynski, and T. Jaroszewicz. "A CG-FFT approach to 3D elastic scattering problems," *J. Comput. Phys.*, 227(12), 5597--5614, 2008. [(ScienceDirect)](https://www.sciencedirect.com/science/article/abs/pii/S0021999108004117)
7. [cuFFT Documentation (NVIDIA)](https://docs.nvidia.com/cuda/cufft/)
8. Seo, S. M. and J.-F. Lee. "IE-FFT Algorithm for a Nonconformal VIE," *IEEE TAP*, 56(10), 2008. [(IEEE Xplore)](https://ieeexplore.ieee.org/document/4526967/)

### Gabor-Frame Spatial-Spectral VIE (Eindhoven Program)
9. Dilz, R. J. et al. "2D TM scattering ... employing Gabor frames," *JOSA A*, 34(8), 2017. [(Optica)](https://opg.optica.org/josaa/abstract.cfm?uri=josaa-34-8-1315)
10. Dilz, R. J. and M. C. van Beurden. "A domain integral equation approach ... Gabor frame discretization," *J. Comput. Phys.*, 345, 2017. [(ScienceDirect)](https://www.sciencedirect.com/science/article/pii/S0021999117304096)
11. Dilz, R. J. et al. "A 3D spatial spectral integral equation method ...," *Opt. Quantum Electron.*, 50, 2018. [(PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC6445559/)
12. Eijsvogel, S. et al. "Describing discontinuous finite 3D scattering objects in Gabor coefficients," *JOSA A*, 39(1), 2022. [(Optica)](https://opg.optica.org/josaa/abstract.cfm?uri=josaa-39-1-86)
13. Eijsvogel, S. et al. "Computing Gabor Coefficients ... Super Exponential Converging Accuracy," *ICEAA 2022*. [(IEEE)](https://ieeexplore.ieee.org/document/9899938)
14. Eijsvogel, S. et al. "A Parallel 3D Spatial Spectral VIE Method," *PIER B*, 102, 2023. [(PIER)](https://www.jpier.org/issues/volume.html?paper=23060708)
15. van Beurden, M. C. et al. "Phaseless inverse scattering ... spatial spectral VIE," *JOSA A*, 41(11), 2024. [(Optica)](https://opg.optica.org/josaa/abstract.cfm?uri=josaa-41-11-2076)

### Gabor Frames and Gaussian Beams in Seismology
16. Lugara, D. and C. Letrou. "Frame-based Gaussian beam summation method," *Radio Science*, 38(2), 2003. [(Wiley)](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2001RS002593)
17. Nowack, R. L. "Frame-based Gaussian beam summation and seismic head waves," *Purdue GMIG*, 2008. [(PDF)](https://gmig.science.purdue.edu/pdfs/2008/08-05.pdf)
18. Nowack, R. L. "Dynamically Focused Gaussian Beams for Seismic Imaging," *Int. J. Geophys.*, 2011. [(Wiley)](https://onlinelibrary.wiley.com/doi/10.1155/2011/316581)
19. Huang, X. et al. "Born modeling for heterogeneous media using Gaussian beam summation Green's function," *J. Appl. Geophys.*, 131, 2016. [(ScienceDirect)](https://www.sciencedirect.com/science/article/abs/pii/S0926985116301604)
20. Shlivinski, A. "Gaussian-windowed frame based MoM ... for extended apertures," *J. Comput. Phys.*, 315, 2016. [(ScienceDirect)](https://sciencedirect.com/science/article/pii/S0021999115008475)

### Neural Operators and Physics-Informed Surrogates
21. Guo, R. et al. "Physics Embedded Deep Neural Network for Solving VIE: 2-D Case," *IEEE TAP*, 69(10), 2021. [(IEEE)](https://ieeexplore.ieee.org/document/9397371)
22. "Acceleration of Solving VIE through Neural-Network-Assisted Approach," *PIER*, 183, 2025. [(PIER)](https://www.jpier.org/ac_api/download.php?id=25012103)
23. Li, Z. et al. "Physics-Informed Neural Operator for Learning PDEs," *J. Data Science*, 2024. [(ACM)](https://dl.acm.org/doi/10.1145/3648506)
24. "Variational Physics-informed Neural Operator (VINO)," *CMAME*, 2025. [(ScienceDirect)](https://www.sciencedirect.com/science/article/pii/S004578252500057X)
25. *PIER*, 2025 --- Neural-network modified contrast operators for coarse VIE. [(PIER)](https://www.jpier.org/issues/volume.html?paper=25082502)
