"""Galerkin scattering by Legendre cells near a welded interface between two half-spaces (step 2).

Plan: docs/2026-10-10-stratified-reference-legendre-cells-3d.md (step 2).
Blocks: ``interface_image`` (closed forms) and ``interface_spectral`` (the split K = D + A0 + w^2 A1 [+ w^4 A2]).

THE GRID.  Cells of half-width h at integer indices (iz, ix, iy), centres h (2 iz + 1, 2 ix, 2 iy): the interface
z = 0 is a plane of cell faces, medium A (``above``) for iz < 0 and B (``below``) for iz >= 0 (z points down, axes
(z, x, y) as everywhere in the package).  In the units of ``interface_spectral`` (centres in h) a cell is
(2 iz + 1, 2 ix, 2 iy).

THE SYSTEM.  As ``graded_voxel.solver`` (Paper 2):
    A[(m, a, i), (n, b, j)] = delta_mn M_ab delta_ij - sum_c K_mn[a, c][i, :] E^(n)_cb[:, j],
with the coupling of the stratified reference medium in place of the whole-space one:
    same side:      K_mn = K^ws_side(o_mn) + K^refl_mn     (Paper 2's block in that side's medium, plus the reflection)
    opposite sides: K_mn = K^trans_mn                      (the whole coupling across the interface)
Each cell's contrast is taken against its own side's medium.  The right-hand side is the reference medium's
field for an incident plane wave from A: the incident and reflected waves in A, the transmitted ones in B
(``stratified_march.interface_rt``), their cell moments in closed form for complex wave vectors.

THE SWEEP.  The interface blocks' frequency-independent parts (A0, A1) and lateral tables are set up once per
symmetry class of pairs: the interface keeps the lateral symmetries x -> -x, y -> -y, x <-> y, so a pair's block
is the canonical one (lateral offset (a, b), a >= b >= 0) mapped by the signed permutation (``fft._transform``).
Each frequency then costs one truncated plane-wave integral per class and one whole-space block per offset.
"""

import itertools
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import scipy.linalg
from numpy.typing import NDArray

from . import interface_spectral as isp
from .effective_contrasts import MaterialContrast, ReferenceMedium
from .graded_voxel.basis import (
    SOURCE_EXPONENTS,
    SOURCE_EXPONENTS_QUARTIC,
    gram_test,
    source_expansion,
)
from .graded_voxel.blocks import coupling_block
from .graded_voxel.fft import _transform, signed_permutations
from .graded_voxel.site import cell_contrast_coefficients
from .graded_voxel.solver import field_sizes
from .stratified_march import interface_rt, mode_family, receiver_moments

# the field functions of the solver and of the interface blocks are the same, in the same order
assert [tuple(e) for e in SOURCE_EXPONENTS] == [
    tuple(e) for e in SOURCE_EXPONENTS_QUARTIC[: len(SOURCE_EXPONENTS)]
]


@dataclass(frozen=True)
class InterfaceGrid:
    """Cells near a welded interface: integer indices (iz, ix, iy), half-width h (km), the two media."""

    idx: NDArray  # (N, 3) int
    h: float
    above: ReferenceMedium
    below: ReferenceMedium

    def __post_init__(self) -> None:
        idx = np.asarray(self.idx, dtype=int).reshape(-1, 3)
        if len({tuple(i) for i in idx}) != len(idx):
            raise ValueError("InterfaceGrid: two cells share an index.")
        object.__setattr__(self, "idx", idx)

    @property
    def centres(self) -> NDArray:
        """Cell centres, shape (N, 3), km."""
        return self.h * self.units

    @property
    def units(self) -> NDArray:
        """Centres in units of h: (2 iz + 1, 2 ix, 2 iy), the convention of ``interface_spectral``."""
        return np.stack(
            [2 * self.idx[:, 0] + 1, 2 * self.idx[:, 1], 2 * self.idx[:, 2]], -1
        )

    @property
    def in_above(self) -> NDArray:
        """True for cells in medium A (z < 0)."""
        return self.idx[:, 0] < 0

    def medium(self, m: int) -> ReferenceMedium:
        return self.above if self.in_above[m] else self.below


# ---------------------------------------------------------------------------
# The reference medium's field
# ---------------------------------------------------------------------------

_MODE = {"P": 0, "SV": 1, "SH": 2}


def reference_waves(
    kx: float, ky: float, omega: float, above: ReferenceMedium, below: ReferenceMedium, wave: str
) -> list[tuple[bool, NDArray, NDArray]]:
    """The plane waves of the reference medium's field for a down-going ``wave`` ("P", "SV", "SH") of unit
    amplitude in A with lateral wavenumber (kx, ky): (in A, wave vector (3,) complex, 9-state (9,)), the phases
    referred to the interface z = 0, the amplitudes those of ``mode_family`` (bilinear-unit polarisations)."""
    if wave not in _MODE:
        raise ValueError(f"reference_waves: wave must be 'P', 'SV' or 'SH', got {wave!r}")
    kxa, kya = np.array([kx], float), np.array([ky], float)
    rt = interface_rt(kxa, kya, omega, above, below)
    n = _MODE[wave]

    def modes(ref, sign):
        out = []
        for w in ("P", "S"):
            out += [(k[0], d[0]) for k, d, _ in mode_family(kxa, kya, omega, ref, w, sign)]
        return out  # (P, SV, SH)

    down_a, up_a, down_b = modes(above, 1.0), modes(above, -1.0), modes(below, 1.0)
    waves = [(True, *down_a[n])]
    waves += [(True, up_a[m][0], rt["R_d"][0, m, n] * up_a[m][1]) for m in range(3)]
    waves += [(False, down_b[m][0], rt["T_d"][0, m, n] * down_b[m][1]) for m in range(3)]
    return waves


def reference_moments(
    grid: InterfaceGrid, omega: float, kx: float, ky: float, wave: str, n_field: int
) -> NDArray:
    """<Q_a, u_ref> over each cell, shape (N, n_field, 9): the reference medium's field on the cells."""
    out = np.zeros((len(grid.idx), n_field, 9), dtype=complex)
    c = grid.centres
    for in_a, k, d in reference_waves(kx, ky, omega, grid.above, grid.below, wave):
        cells = np.flatnonzero(grid.in_above == in_a)
        mom = receiver_moments(k[None, :], grid.h, n_field)[0]  # <Q_a, e^{ik.(x - c)}>
        ph = np.exp(1j * c[cells] @ k)
        out[cells] += ph[:, None, None] * mom[None, :, None] * d[None, None, :]
    return out


# ---------------------------------------------------------------------------
# The couplings
# ---------------------------------------------------------------------------


def _canonical(du: int, dv: int) -> tuple[tuple[int, int], NDArray]:
    """The canonical lateral offset (a, b), a >= b >= 0, and the signed permutation Q (fixing z) with
    Q (0, a, b) = (0, du, dv)."""
    a, b = sorted((abs(du), abs(dv)), reverse=True)
    target = np.array([0.0, du, dv])
    for q in signed_permutations():
        if q[0, 0] == 1.0 and np.array_equal(q @ np.array([0.0, a, b]), target):
            return (a, b), q
    raise AssertionError("unreachable: the lateral symmetries reach every offset")


def _parts_job(args):
    rec, src, h, above, below, n_source, n_test, q0h, q1h, window, omega4 = args
    parts = isp.frequency_independent_parts(
        rec, src, h, above, below, n_source, n_test, q0h, q1h, window, omega4=omega4
    )
    table = isp.lateral_harmonics([(rec[1] - src[1], rec[2] - src[2])], h, q0h, n_source, n_test)
    return parts, table


@dataclass
class InterfaceCouplings:
    """The frequency-independent parts of every interface block of a grid, one per symmetry class.

    Args:
        grid: The cells.
        n_source: Source monomials (10 for p <= 1 with r <= 1, ``field_sizes``).
        n_test: Field functions (4 or 10).
        q0h: The split's cutoff Q0 h (``interface_spectral.sweep``: 10 suits omega h / beta <= 1/6, 40 to 1).
        q1h: Kept for ``frequency_independent_parts``.
        window: Optional smooth window of the split.
        omega4: Also the omega^4 correction A2 (no window).
        workers: Processes for the set-up (default: the CPUs).
    """

    grid: InterfaceGrid
    n_source: int = 10
    n_test: int = 4
    q0h: float = 10.0
    q1h: float = 160.0
    window: isp.Window | None = None
    omega4: bool = False
    workers: int | None = None
    classes: dict = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        u = self.grid.units
        keys = set()
        for m, n in itertools.product(range(len(u)), repeat=2):
            (a, b), _ = _canonical(int(u[m, 1] - u[n, 1]), int(u[m, 2] - u[n, 2]))
            keys.add((int(u[m, 0]), int(u[n, 0]), a, b))
        keys = sorted(keys)
        g = self.grid
        jobs = [
            ((zr, a, b), (zs, 0, 0), g.h, g.above, g.below, self.n_source, self.n_test, self.q0h,
             self.q1h, self.window, self.omega4)
            for zr, zs, a, b in keys
        ]
        workers = self.workers or os.cpu_count() or 1
        if workers > 1 and len(jobs) > 1:
            with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as ex:
                results = list(ex.map(_parts_job, jobs))
        else:
            results = [_parts_job(j) for j in jobs]
        self.classes = dict(zip(keys, results, strict=True))

    def interface_block(self, m: int, n: int, omega: float) -> NDArray:
        """The reflected (same side) or transmitted (opposite sides) block of receiver m, source n."""
        u = self.grid.units
        (a, b), q = _canonical(int(u[m, 1] - u[n, 1]), int(u[m, 2] - u[n, 2]))
        zr, zs = int(u[m, 0]), int(u[n, 0])
        parts, table = self.classes[zr, zs, a, b]
        g = self.grid
        blk = isp.interface_block(
            (zr, a, b), (zs, 0, 0), g.h, omega, g.above, g.below, self.n_source, self.n_test,
            self.q0h, self.q1h, parts, table, self.window,
        )
        return _transform(q, blk)

    def blocks(self, omega: float) -> dict[tuple[int, int], NDArray]:
        """K_mn for every pair at one frequency: the whole-space block of the side plus the reflection, or the
        transmission."""
        g = self.grid
        n_cells = len(g.idx)
        ws: dict = {}
        canon: dict = {}
        out = {}
        u = g.units
        for m, n in itertools.product(range(n_cells), repeat=2):
            (a, b), q = _canonical(int(u[m, 1] - u[n, 1]), int(u[m, 2] - u[n, 2]))
            key = (int(u[m, 0]), int(u[n, 0]), a, b)
            if key not in canon:
                parts, table = self.classes[key]
                canon[key] = isp.interface_block(
                    (key[0], a, b), (key[1], 0, 0), g.h, omega, g.above, g.below, self.n_source,
                    self.n_test, self.q0h, self.q1h, parts, table, self.window,
                )
            k = _transform(q, canon[key])
            if g.in_above[m] == g.in_above[n]:
                off = tuple(int(v) for v in g.idx[m] - g.idx[n])
                wkey = (off, bool(g.in_above[m]))
                if wkey not in ws:
                    ws[wkey] = coupling_block(off, g.h, omega, g.medium(m), self.n_source, self.n_test)  # type: ignore[arg-type]
                k = k + ws[wkey]
            out[m, n] = k
        return out


# ---------------------------------------------------------------------------
# The solve
# ---------------------------------------------------------------------------


def cell_contrasts(
    grid: InterfaceGrid,
    omega: float,
    contrasts: list[MaterialContrast | None],
    r: int = 0,
) -> NDArray:
    """Each cell's contrast operator coefficients against its own side's medium, shape (N, 4|10, 9, 9); a
    ``None`` contrast is a cell of the reference medium (a receiver)."""
    n_contrast = 10 if r == 2 else 4
    out = np.zeros((len(grid.idx), n_contrast, 9, 9), dtype=complex)
    for m, (c, ctr) in enumerate(zip(grid.centres, contrasts, strict=True)):
        if ctr is not None:
            out[m] = cell_contrast_coefficients(
                lambda x: 1.0, c, grid.h, ctr, grid.medium(m), omega, degree=r
            )
    return out


@dataclass
class InterfaceResult:
    """The Galerkin solution for one incident wave."""

    grid: InterfaceGrid
    omega: float
    delta: NDArray  # (N, 4|10, 9, 9)
    reference: NDArray  # (N, n_field, 9): the reference medium's field moments
    psi: NDArray  # (N, n_field, 9): the total field's coefficients
    p: int
    r: int


def assemble(
    grid: InterfaceGrid, blocks: dict, delta: NDArray, p: int, r: int
) -> NDArray:
    """The dense system matrix, (N na 9) square."""
    n = len(grid.idx)
    na, n_field, _, n_source = field_sizes(p, r)
    dim = n * na * 9
    a = np.zeros((dim, dim), dtype=complex)
    m9 = np.kron(gram_test(grid.h, n_field)[:na, :na], np.eye(9))
    for col in range(n):
        en = source_expansion(delta[col], n_field)[:, :na]
        en = en.transpose(0, 2, 1, 3).reshape(n_source * 9, na * 9)
        for row in range(n):
            k = blocks[row, col][:na, :n_source].transpose(0, 2, 1, 3).reshape(na * 9, n_source * 9)
            blk = -(k @ en)
            if row == col:
                blk = blk + m9
            a[row * na * 9 : (row + 1) * na * 9, col * na * 9 : (col + 1) * na * 9] = blk
    return a


def solve_interface(
    couplings: InterfaceCouplings,
    omega: float,
    contrasts: list[MaterialContrast | None],
    kx: float,
    ky: float,
    wave: str,
    p: int = 1,
    r: int = 0,
    blocks: dict | None = None,
) -> InterfaceResult:
    """Solve (M - K E) psi = <Q, u_ref> densely for a plane wave incident from A.

    Args:
        couplings: The grid's interface set-up (n_source, n_test must suit p and r: ``field_sizes``).
        omega: Angular frequency (rad/s, km and km/s units as the media).
        contrasts: Each cell's contrast against its side's medium, None for a cell of the reference medium.
        kx, ky: The incident wave's lateral wavenumbers.
        wave: "P", "SV" or "SH", down-going in A.
        p: Field degree (0, 1, 2).
        r: Contrast degree (0, 1, 2).
        blocks: ``couplings.blocks(omega)``, to reuse across incident waves.
    """
    grid = couplings.grid
    na, n_field, _, n_source = field_sizes(p, r)
    if couplings.n_source < n_source or couplings.n_test < n_field:
        raise ValueError(
            f"solve_interface: p = {p}, r = {r} need n_source >= {n_source}, n_test >= {n_field}; the couplings"
            f" have {couplings.n_source}, {couplings.n_test}."
        )
    delta = cell_contrasts(grid, omega, contrasts, r)
    blocks = couplings.blocks(omega) if blocks is None else blocks
    a = assemble(grid, blocks, delta, p, r)
    ref = reference_moments(grid, omega, kx, ky, wave, n_field)
    psi = scipy.linalg.solve(a, ref[:, :na].ravel(), check_finite=False).reshape(len(grid.idx), na, 9)
    full = np.zeros((len(grid.idx), n_field, 9), dtype=complex)
    full[:, :na] = psi
    return InterfaceResult(grid, omega, delta, ref, full, p, r)
