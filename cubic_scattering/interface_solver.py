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

import contextlib
import itertools
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import scipy.fft
import scipy.linalg
from numpy.typing import NDArray

from . import interface_spectral as isp
from . import stratified_apply as sap
from .effective_contrasts import MaterialContrast, ReferenceMedium
from .graded_voxel.basis import (
    SOURCE_EXPONENTS,
    SOURCE_EXPONENTS_QUARTIC,
    gram_test,
    source_expansion,
)
from .graded_voxel.blocks import (
    coupling_block,
    far_block_series,
    far_series_coefficients,
    near_block_series,
)
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
    kx: float,
    ky: float,
    omega: float,
    above: ReferenceMedium,
    below: ReferenceMedium,
    wave: str,
) -> list[tuple[bool, NDArray, NDArray]]:
    """The plane waves of the reference medium's field for a down-going ``wave`` ("P", "SV", "SH") of unit
    amplitude in A with lateral wavenumber (kx, ky): (in A, wave vector (3,) complex, 9-state (9,)), the phases
    referred to the interface z = 0, the amplitudes those of ``mode_family`` (bilinear-unit polarisations)."""
    if wave not in _MODE:
        raise ValueError(
            f"reference_waves: wave must be 'P', 'SV' or 'SH', got {wave!r}"
        )
    kxa, kya = np.array([kx], float), np.array([ky], float)
    rt = interface_rt(kxa, kya, omega, above, below)
    n = _MODE[wave]

    def modes(ref, sign):
        out = []
        for w in ("P", "S"):
            out += [
                (k[0], d[0]) for k, d, _ in mode_family(kxa, kya, omega, ref, w, sign)
            ]
        return out  # (P, SV, SH)

    down_a, up_a, down_b = modes(above, 1.0), modes(above, -1.0), modes(below, 1.0)
    waves = [(True, *down_a[n])]
    waves += [(True, up_a[m][0], rt["R_d"][0, m, n] * up_a[m][1]) for m in range(3)]
    waves += [
        (False, down_b[m][0], rt["T_d"][0, m, n] * down_b[m][1]) for m in range(3)
    ]
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
    return isp.frequency_independent_parts(
        rec, src, h, above, below, n_source, n_test, q0h, q1h, window, omega4=omega4
    )


def _table_job(args):
    offset, h, q0h, n_source, n_test = args
    return isp.lateral_harmonics([offset], h, q0h, n_source, n_test)


def _setup_job(job):
    kind, args = job
    if kind == "far":
        # Paper 2's whole-space series coefficients of an orbit (frequency-independent)
        return far_series_coefficients(*args)
    return _parts_job(args) if kind == "parts" else _table_job(args)


_BLAS_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


@contextlib.contextmanager
def _one_blas_thread():
    """Workers started meanwhile inherit one BLAS / OpenMP thread each: a process per CPU, each with a BLAS
    thread per CPU, would oversubscribe the CPUs.  Set in the parent's environment while the workers start,
    since a spawned worker imports numpy (and its BLAS) before any initializer runs; the parent's own BLAS,
    already loaded, is unaffected."""
    old = {k: os.environ.get(k) for k in _BLAS_THREAD_VARS}
    os.environ.update(dict.fromkeys(_BLAS_THREAD_VARS, "1"))
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def interface_gap(zr: int, zs: int) -> int:
    """The two cells' distances from the interface summed, in h, from their centres' z in h (odd): 0 for two
    touching cells."""
    return abs(zr) - 1 + abs(zs) - 1


@dataclass
class InterfaceCouplings:
    """The frequency-independent parts of every interface block of a grid, one per symmetry class.

    Args:
        grid: The cells.
        n_source: Source monomials (10 for p <= 1 with r <= 1, ``field_sizes``).
        n_test: Field functions (4 or 10).
        q0h: The split's cutoff Q0 h.  The dropped remainder at Q0 h = 10 is ~3.5e-10 of a touching block at
            omega h / beta = 1/6 and ~2.5e-8 at 1/2 (scripts/measure_interface_q0_omega4.py), far below the
            discretisation error (p = 1 against p = 2: ~2e-4 at 1/6); D's cost per frequency does not depend on Q0.
        q1h: Kept for ``frequency_independent_parts``.
        window: Optional smooth window of the split.
        omega4: Also the omega^4 correction A2, so the remainder falls as Q0^-8 (~5e-11 to omega h / beta = 1/2 at
            Q0 h = 10): for reference-quality blocks.  It changes no per-frequency cost but about 2.5x the set-up
            (the closed-form K2), and the solution does not see it below the discretisation error.  Not with a
            window.
        radius: Lateral distance between cell centres (in h) beyond which the frequency-independent corrections
            A0, A1 are dropped and the block is the windowed plane-wave part D alone; needs a smooth ``window``.
            With the C-infinity window from Q0 / 4 the corrections fall below 1e-6, 1e-8 and 1e-10 of the touching
            block beyond 12 h, 16 h and 24 h (cells touching the interface, the slowest case; the paper's
            tab:locality), so the set-up grows with the radius, not with the grid.  None: every class.
        depth_radius: Total gap g = d_r + d_s (in h) of the two cells from the interface at and beyond which A0, A1 are
            dropped likewise; needs a smooth ``window``.  The corrections fall by ~1e-4 per 2 h of g: below 1e-6 and
            1e-8 of the touching static block for g >= 4 h, below 1e-10 for g >= 6 h
            (scripts/measure_interface_depth_locality.py), so only the layers nearest the interface carry them and
            the set-up does not grow with the number of layers.  None: every depth.
        split: D = D_low + D_high (scaling plan S4; needs a smooth ``window``).  D_low, the band below the window's
            ramp (q h < window.start Q0 h) that holds the branch points and D's slow ~h/r tail, stays a per-offset
            radial integral, with tables over that short band only; D_high, smooth in k, is sampled on the FFT
            grid of a periodic box (``interface_spectral.high_band_blocks``), with no per-offset work.
        split_pad: Cells between the box's farthest offset and the first periodic image of D_high.  D_high
            decays faster than any power: below 1e-8 of the touching static block at ~26 cells (omega h / beta =
            1/6) to ~34 (1/2), 1e-10 at ~40 to ~55.
        matrix_free: Build only what ``MatrixFreeOperator`` uses: the self offset's lateral table (D itself comes
            from the plane-wave modes) and the whole-space series coefficients of planes up to 2 apart (the
            default ``stored_planes``; farther planes go through the march).  ``LateralOperator`` and the full ``interface_canonical`` then refuse.
            Needs ``split``.
        workers: Processes for the set-up (default: the CPUs), one BLAS thread each.  The set-up also computes the
            whole-space blocks' series coefficients of every orbit the grid needs (``far_coeffs``), so a frequency
            is only their short sums.  A frequency runs in this process, where BLAS threads use the CPUs (a pool of
            one task per pair of depths was slower: 77 s against 23 s per frequency at 288 cells, split).
    """

    grid: InterfaceGrid
    n_source: int = 10
    n_test: int = 4
    q0h: float = 10.0
    q1h: float = 160.0
    window: isp.Window | None = None
    omega4: bool = False
    radius: float | None = None
    depth_radius: float | None = None
    split: bool = False
    split_pad: int = 40
    matrix_free: bool = False
    workers: int | None = None
    classes: dict = field(default_factory=dict, init=False)
    far_coeffs: dict = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.window is not None and self.omega4:
            raise ValueError(
                "InterfaceCouplings: the omega^4 split has no windowed form; pass omega4=False with a window."
            )
        if (
            self.radius is not None or self.depth_radius is not None
        ) and self.window is None:
            raise ValueError(
                "InterfaceCouplings: a radius needs a smooth window; with the sharp cutoff the corrections ring and\n"
                "  decay only as d^-3/2.  Pass window=interface_spectral.Window(0.25)."
            )
        if self.matrix_free and not self.split:
            raise ValueError(
                "InterfaceCouplings: matrix_free needs split=True (MatrixFreeOperator's D)."
            )
        if self.split and self.window is None:
            raise ValueError(
                "InterfaceCouplings: split needs a smooth window (its band ends where the window's ramp starts)."
            )
        g = self.grid
        # the classes: per pair of layers, the canonical lateral offsets (a >= b >= 0, in h) its cells realise
        lat = {z: g.idx[g.idx[:, 0] == z, 1:] for z in np.unique(g.idx[:, 0])}
        keys = set()
        for zr, zs in itertools.product(lat, repeat=2):
            d = np.abs(lat[zr][:, None, :] - lat[zs][None, :, :]).reshape(-1, 2)
            ab = np.unique(np.sort(d, axis=1)[:, ::-1], axis=0)
            keys |= {
                (2 * int(zr) + 1, 2 * int(zs) + 1, 2 * int(a), 2 * int(b))
                for a, b in ab
            }
        keys = sorted(keys)
        # beyond either radius a class has no corrections: its block is the windowed D alone
        local = [k for k in keys if self.corrected(k)]
        # S_n(q) depends on the lateral offset alone, not on the depths or media: one table per offset,
        # shared by its classes.  Its cost grows as the reach squared, so the farthest go first.
        offsets = sorted({k[2:] for k in keys}, key=lambda o: -np.hypot(*o))
        if (
            self.matrix_free
        ):  # D comes from the plane-wave modes: only the self blocks' table is used
            offsets = [(0, 0)]
        # with the split, D's per-offset part is the low band alone: tables over q h < band_qh only
        table_qh = self.band_qh if self.split else self.q0h
        jobs = [
            ("table", (o, g.h, table_qh, self.n_source, self.n_test)) for o in offsets
        ] + [
            (
                "parts",
                (
                    (zr, a, b),
                    (zs, 0, 0),
                    g.h,
                    g.above,
                    g.below,
                    self.n_source,
                    self.n_test,
                    self.q0h,
                    self.q1h,
                    self.window,
                    self.omega4,
                ),
            )
            for zr, zs, a, b in local
        ]
        # the whole-space blocks' orbits (same side, not touching): their series coefficients, once
        idx = g.idx
        far: set = set()
        for side in (g.in_above, ~g.in_above):
            sub = idx[side]
            if not len(sub):
                continue
            # |offsets| between the side's cells: lateral differences of its positions, times the layer differences
            pos = np.unique(sub[:, 1:], axis=0)
            dxy = np.unique(
                np.abs(pos[:, None, :] - pos[None, :, :]).reshape(-1, 2), axis=0
            )
            zs = np.unique(sub[:, 0])
            dz = np.unique(np.abs(zs[:, None] - zs[None, :]))
            if self.matrix_free:
                # the stored whole-space kernels: planes up to MatrixFreeOperator's default stored_planes (2) apart
                dz = dz[dz <= 2]
            for z in dz:
                rows = np.column_stack([np.full(len(dxy), z), dxy])
                far |= {
                    c
                    for c in (tuple(sorted(map(int, r), reverse=True)) for r in rows)
                    if max(c) > 1
                }
        far_keys = sorted(far)
        jobs += [("far", (c, self.n_source, self.n_test)) for c in far_keys]
        workers = self.workers or os.cpu_count() or 1
        if workers > 1 and len(jobs) > 1:
            # spawn: forking a process whose BLAS runs threads can deadlock
            ctx = multiprocessing.get_context("spawn")
            with (
                _one_blas_thread(),
                ProcessPoolExecutor(
                    max_workers=min(workers, len(jobs)), mp_context=ctx
                ) as ex,
            ):
                results = list(ex.map(_setup_job, jobs))
        else:
            results = [_setup_job(j) for j in jobs]
        n_t, n_p = len(offsets), len(local)
        tables = dict(zip(offsets, results[:n_t], strict=True))
        parts = dict(zip(local, results[n_t : n_t + n_p], strict=True))
        self.far_coeffs = dict(zip(far_keys, results[n_t + n_p :], strict=True))
        self.classes = {k: (parts.get(k), tables.get(k[2:])) for k in keys}

    def release_far_coeffs(self) -> None:
        """Drop the whole-space series coefficients (~2.5 MB per orbit: 3.3 GB at 36 x 36 cells, 10 GB at 64 x 64).
        They are needed only while an operator for a frequency is built; call this after the last one.  A later
        operator recomputes the coefficients it needs, serially (~3 s per orbit)."""
        self.far_coeffs = {}

    @property
    def band_qh(self) -> float:
        """The upper end of the split's low band: where the window's ramp starts."""
        return self.window.start * self.q0h

    def low_pass(self, omega: float) -> isp.Window:
        """The split's band at a frequency: its ramp from twice the highest branch point (at least 0.35 / h; a
        wider ramp makes D_high decay faster) to ``band_qh``."""
        g = self.grid
        top = max(
            omega * g.h / c for m in (g.above, g.below) for c in (m.alpha, m.beta)
        )
        lo = max(0.35, 2 * top)
        if lo >= self.band_qh:
            raise ValueError(
                f"InterfaceCouplings: the split's band [{lo:g}, {self.band_qh:g}] (q h) is empty at this frequency;"
                " raise q0h or the window's start."
            )
        return isp.Window(lo / self.band_qh)

    def corrected(self, key: tuple) -> bool:
        """Whether the class (zr, zs, a, b) carries A0, A1: within the lateral radius and the depth radius."""
        zr, zs, a, b = key
        lateral = self.radius is None or np.hypot(a, b) <= self.radius
        depth = self.depth_radius is None or interface_gap(zr, zs) < self.depth_radius
        return lateral and depth

    def _zero_parts(self) -> tuple:
        z = np.zeros((self.n_test, self.n_source, 9, 9), dtype=complex)
        return (z, z)

    def interface_block(self, m: int, n: int, omega: float) -> NDArray:
        """The reflected (same side) or transmitted (opposite sides) block of receiver m, source n."""
        if self.split:
            raise NotImplementedError(
                "InterfaceCouplings.interface_block: with split the blocks come from interface_canonical."
            )
        u = self.grid.units
        (a, b), q = _canonical(int(u[m, 1] - u[n, 1]), int(u[m, 2] - u[n, 2]))
        zr, zs = int(u[m, 0]), int(u[n, 0])
        parts, table = self.classes[zr, zs, a, b]
        parts = self._zero_parts() if parts is None else parts
        g = self.grid
        blk = isp.interface_block(
            (zr, a, b),
            (zs, 0, 0),
            g.h,
            omega,
            g.above,
            g.below,
            self.n_source,
            self.n_test,
            self.q0h,
            self.q1h,
            parts,
            table,
            self.window,
        )
        return _transform(q, blk)

    def interface_canonical(
        self, omega: float, keys: list | None = None
    ) -> dict[tuple, NDArray]:
        """The interface block of every symmetry class at one frequency (or of ``keys`` only): the truncated
        plane-wave integrals D batched (one interface solve and frame per radial node and pair of depths, shared
        by every lateral offset; sources in B mirrored onto A with the media exchanged), plus the class's
        A0 + w^2 A1 (+ w^4 A2)."""
        g = self.grid
        keys = list(self.classes) if keys is None else list(keys)
        tables = [self.classes[k][1] for k in keys]
        if any(t is None for t in tables):
            raise ValueError(
                "InterfaceCouplings.interface_canonical: the set-up has no lateral tables for these classes\n"
                "  (matrix_free=True keeps the self offset's only); use MatrixFreeOperator."
            )
        table = dict(tables[0])
        for t in tables:
            table.update({k: v for k, v in t.items() if isinstance(k, tuple)})
        if self.split:
            lp = self.low_pass(omega)
            # D's per-offset part: the low band, below the window
            qh_d, window_d, extra = self.band_qh, None, {"low_pass": lp}
        else:
            qh_d, window_d, extra = self.q0h, self.window, {}
        d: dict = {}
        for from_a in (True, False):
            sel = [k for k in keys if (k[1] < 0) == from_a]
            if not sel:
                continue
            pairs = [((zr, a, b), (zs, 0, 0)) for zr, zs, a, b in sel]
            # a source in B: the mirrored cells (z -> -z, media exchanged), mapped back
            media = (g.above, g.below) if from_a else (g.below, g.above)
            back = (lambda x: x) if from_a else isp._mirror
            if not from_a:
                pairs = [(isp._mirrored(r), isp._mirrored(s)) for r, s in pairs]
            got = [
                back(x)
                for x in isp.truncated_blocks(
                    pairs,
                    g.h,
                    omega,
                    *media,
                    self.n_source,
                    self.n_test,
                    qh_d,
                    table,
                    window_d,
                    **extra,
                )
            ]
            if self.split:
                # D_high: one periodic grid per pair of depths, shared by every lateral offset
                reach = max(max(k[2], k[3]) for k in sel) // 2  # cells
                n_grid = scipy.fft.next_fast_len(reach + 1 + self.split_pad)
                for zr, zs in sorted({k[:2] for k in sel}):
                    rz, sz = (zr, zs) if from_a else (-zr, -zs)
                    at = [i for i, k in enumerate(sel) if k[:2] == (zr, zs)]
                    # the canonical offsets alone, by the lateral symmetries (an eighth of the wavenumbers)
                    hi = isp.high_band_blocks(
                        rz,
                        sz,
                        n_grid,
                        g.h,
                        omega,
                        *media,
                        self.n_source,
                        self.n_test,
                        self.q0h,
                        window=self.window,
                        low_pass=lp,
                        band_qh=self.band_qh,
                        offsets=[(sel[i][2] // 2, sel[i][3] // 2) for i in at],
                    )
                    for i, blk in zip(at, hi, strict=True):
                        got[i] = got[i] + back(blk)
            d.update(zip(sel, got, strict=True))
        out = {}
        for key in keys:
            parts = self.classes[key][0]
            if parts is None:  # beyond the radius
                out[key] = d[key]
                continue
            blk = d[key] + parts[0] + omega**2 * parts[1]
            if len(parts) > 2:
                blk = blk + omega**4 * parts[2]
            out[key] = blk
        return out

    def blocks(self, omega: float) -> dict[tuple[int, int], NDArray]:
        """K_mn for every pair at one frequency: the whole-space block of the side plus the reflection, or the
        transmission.  The whole-space blocks are Paper 2's (``_whole_space``), one per cube-group orbit and side,
        mapped by the signed permutation."""
        g = self.grid
        u = g.units
        canon = self.interface_canonical(omega)
        ws: dict = {}
        out = {}
        for m, n in itertools.product(range(len(g.idx)), repeat=2):
            (a, b), q = _canonical(int(u[m, 1] - u[n, 1]), int(u[m, 2] - u[n, 2]))
            k = _transform(q, canon[int(u[m, 0]), int(u[n, 0]), a, b])
            if g.in_above[m] == g.in_above[n]:
                k = k + _whole_space(
                    tuple(int(v) for v in g.idx[m] - g.idx[n]),
                    g.h,
                    omega,
                    g.medium(m),
                    self.n_source,
                    self.n_test,
                    ws,
                    self.far_coeffs,
                )
            out[m, n] = k
        return out


def _whole_space(off, h, omega, ref, n_source, n_test, cache, far=None) -> NDArray:
    """Paper 2's block for a grid offset: the orbit's canonical block mapped to it.  Touching offsets by
    ``near_block_series`` (its universal moments once per process, minutes; then milliseconds per frequency),
    the others by ``far_block_series`` (its coefficients once, then each frequency a short sum), Gauss where
    the far series would need too many terms."""
    c = tuple(sorted((abs(o) for o in off), reverse=True))
    key = (c, ref.alpha, ref.beta, ref.rho)
    if key not in cache:
        if max(c) <= 1:
            cache[key] = near_block_series(c, h, omega, ref, n_source, n_test)
        else:
            try:
                cache[key] = far_block_series(
                    c, h, omega, ref, n_source, n_test, coeffs=(far or {}).get(c)
                )
            except ValueError:  # too many series terms at this frequency: Gauss
                cache[key] = coupling_block(c, h, omega, ref, n_source, n_test)  # type: ignore[arg-type]
    cv, target = np.array(c, dtype=float), np.array(off, dtype=float)
    q = next(q for q in signed_permutations() if np.array_equal(q @ cv, target))
    return _transform(q, cache[key])


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
            k = (
                blocks[row, col][:na, :n_source]
                .transpose(0, 2, 1, 3)
                .reshape(na * 9, n_source * 9)
            )
            blk = -(k @ en)
            if row == col:
                blk = blk + m9
            a[row * na * 9 : (row + 1) * na * 9, col * na * 9 : (col + 1) * na * 9] = (
                blk
            )
    return a


def apply_coupling(
    grid: InterfaceGrid, blocks: dict, delta: NDArray, x: NDArray, p: int, r: int
) -> NDArray:
    """sum_n K_mn E^(n) x_n for coefficients x of shape (N, na, 9): the coupling term of the system, without
    forming it (the matvec of an iterative solve, or the right-hand side of the scattered-field system)."""
    n = len(grid.idx)
    na, n_field, _, n_source = field_sizes(p, r)
    ex = np.stack(
        [
            np.einsum(
                "cbij,bj->ci", source_expansion(delta[col], n_field)[:, :na], x[col]
            )
            for col in range(n)
        ]
    )  # (N, n_source, 9)
    return np.stack(
        [
            sum(
                np.einsum("acij,cj->ai", blocks[row, col][:na, :n_source], ex[col])
                for col in range(n)
            )
            for row in range(n)
        ]
    )


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
    psi = scipy.linalg.solve(a, ref[:, :na].ravel(), check_finite=False).reshape(
        len(grid.idx), na, 9
    )
    full = np.zeros((len(grid.idx), n_field, 9), dtype=complex)
    full[:, :na] = psi
    return InterfaceResult(grid, omega, delta, ref, full, p, r)


# ---------------------------------------------------------------------------
# The iterative solve: lateral FFT products and GMRES
# ---------------------------------------------------------------------------


class LateralOperator:
    """The coupling of a grid at one frequency as lateral convolutions, one per pair of depth layers.

    Every block depends on the two layers and the lateral offset only: K(iz_r, iz_s; dx, dy) is Paper 2's block of
    the side plus the reflection (same side) or the transmission (opposite sides).  For each pair of layers the
    blocks over the offsets of the grid's lateral box are transformed once (zero padding to 2 n - 1); a product
    then costs one 2-D FFT per source layer, one inverse per receiver layer, and a contraction per pair of
    layers in the transformed domain.  Offsets no pair of the grid's cells realises are zero (they only reach
    positions without cells, whose outputs are discarded).  No block of a single pair is formed.

    Args:
        couplings: The grid's set-up.
        omega: Angular frequency.
        p, r: Field and contrast degree (``field_sizes``).
    """

    def __init__(
        self, couplings: InterfaceCouplings, omega: float, p: int = 1, r: int = 0
    ) -> None:
        if couplings.matrix_free:
            raise ValueError(
                "LateralOperator: the set-up was built for MatrixFreeOperator (matrix_free=True)."
            )
        g = couplings.grid
        self.grid, self.p, self.r = g, p, r
        na, n_field, _, n_source = field_sizes(p, r)
        if couplings.n_source < n_source or couplings.n_test < n_field:
            raise ValueError(
                f"LateralOperator: p = {p}, r = {r} need n_source >= {n_source}, n_test >= {n_field}."
            )
        self.na, self.n_field, self.n_source = na, n_field, n_source
        idx = g.idx
        self.layers = sorted(set(idx[:, 0].tolist()))
        lo = idx[:, 1:].min(axis=0)
        self.pos = idx[:, 1:] - lo  # lateral position in the box
        nx, ny = (idx[:, 1:].max(axis=0) - lo + 1).tolist()
        self.shape = (2 * nx - 1, 2 * ny - 1)
        rows, cols = na * 9, n_source * 9
        canon = couplings.interface_canonical(omega)
        ws_cache: dict = {}
        by_layer = {z: np.flatnonzero(idx[:, 0] == z) for z in self.layers}
        realised: dict = {}
        for zr, zs in itertools.product(self.layers, repeat=2):
            pr, ps = self.pos[by_layer[zr]], self.pos[by_layer[zs]]
            d = (pr[:, None, :] - ps[None, :, :]).reshape(-1, 2)
            realised[zr, zs] = {tuple(v) for v in np.unique(d, axis=0).tolist()}
        self.kernels: dict = {}
        self.self_blocks: dict = {}  # K_mm per layer, for the preconditioner
        for (zr, zs), offs in realised.items():
            kh = np.zeros((rows, cols) + self.shape, dtype=complex)
            same = (zr < 0) == (zs < 0)
            ur, us = 2 * zr + 1, 2 * zs + 1
            for dx, dy in offs:
                (a, b), q = _canonical(2 * dx, 2 * dy)
                k = _transform(q, canon[ur, us, a, b])
                if same:
                    k = k + _whole_space(
                        (zr - zs, dx, dy),
                        g.h,
                        omega,
                        g.above if zr < 0 else g.below,
                        couplings.n_source,
                        couplings.n_test,
                        ws_cache,
                        couplings.far_coeffs,
                    )
                kh[:, :, dx % self.shape[0], dy % self.shape[1]] = (
                    k[:na, :n_source].transpose(0, 2, 1, 3).reshape(rows, cols)
                )
                if dx == 0 and dy == 0 and zr == zs:
                    self.self_blocks[zr] = k[:na, :n_source]
            self.kernels[zr, zs] = np.fft.fft2(kh, axes=(2, 3)).reshape(rows, cols, -1)
        self.by_layer = by_layer
        self.m9 = gram_test(g.h, n_field)[:na, :na]

    def coupling(self, delta: NDArray, x: NDArray) -> NDArray:
        """sum_n K_mn E^(n) x_n, x of shape (N, na, 9): as ``apply_coupling``, by the lateral FFTs."""
        na, n_source = self.na, self.n_source
        e = np.array([source_expansion(d, self.n_field)[:n_source, :na] for d in delta])
        src = np.einsum("ncbij,nbj->nci", e, x).reshape(len(x), n_source * 9)
        return self._convolve(src)

    def _convolve(self, src: NDArray) -> NDArray:
        rows = self.na * 9
        out = np.zeros((len(src), rows), dtype=complex)
        spec = {}
        for zs, cells in self.by_layer.items():
            grid = np.zeros((src.shape[1],) + self.shape, dtype=complex)
            grid[:, self.pos[cells, 0], self.pos[cells, 1]] = src[cells].T
            spec[zs] = np.fft.fft2(grid, axes=(1, 2)).reshape(src.shape[1], -1)
        for zr, cells in self.by_layer.items():
            acc = np.zeros((rows, spec[zr].shape[1]), dtype=complex)
            for zs in self.by_layer:
                acc += np.einsum("rcf,cf->rf", self.kernels[zr, zs], spec[zs])
            y = np.fft.ifft2(acc.reshape((rows,) + self.shape), axes=(1, 2))
            out[cells] = y[:, self.pos[cells, 0], self.pos[cells, 1]].T
        return out.reshape(len(src), self.na, 9)


class MatrixFreeOperator:
    """``LateralOperator``'s product with no kernel stored per pair of depth layers (scaling plan S5).

    - **D**, the plane-wave part of every interface block, through the plane-wave modes
      (``stratified_apply``): the low band by NUFFTs over its polar quadrature (exact, non-periodic), the high
      band at the wavenumbers of the periodic box of ``InterfaceCouplings.interface_canonical`` (the same
      period); per wavenumber three mode amplitudes per source side, plus the window's static pseudo-modes.
      The factors are computed once per frequency.
    - **The corrections** A0 + w^2 A1, local within the lateral and depth radii, as stencils.
    - **The whole-space blocks**, Paper 2's: for planes up to ``stored_planes`` apart, stored lateral kernels,
      (2 s + 1)(2 nx - 1)(2 ny - 1) blocks per side whatever the number of layers; for planes farther apart, the
      plane-wave march in depth (``stratified_apply.WholeSpaceMarch``, the low band's nodes and the box's lattice to
      q h = 16 / s: s = 2, the default, needs a quarter of the wavenumbers of s = 1).

    ``cache``: keep the plane-wave factors between products (each later product is then the NUFFTs and the
    contractions alone; ~23 kB per node for D with six layers, ~12 kB per node for the march), or recompute them in
    each product (memory of the lateral kernels only).  ``eps``: the NUFFTs' tolerance, which now only the polar low-band
    nodes use (the lattice nodes go by FFT, exactly): 1e-9 leaves the product ~1e-11 from the stored kernels', below
    GMRES's 1e-10 and far below D1's 1e-8, and makes the NUFFTs ~1.4x faster than 1e-11.  Needs ``InterfaceCouplings(split=True)``.  The interface ``self_blocks`` (for the preconditioner) are the
    classes' own.
    """

    def __init__(
        self,
        couplings: InterfaceCouplings,
        omega: float,
        p: int = 1,
        r: int = 0,
        eps: float = 1e-9,
        cache: bool = True,
        stored_planes: int = 2,
    ) -> None:
        if not couplings.split:
            raise ValueError(
                "MatrixFreeOperator: needs InterfaceCouplings(split=True)."
            )
        g = couplings.grid
        self.grid, self.p, self.r = g, p, r
        na, n_field, _, n_source = field_sizes(p, r)
        if couplings.n_source < n_source or couplings.n_test < n_field:
            raise ValueError(
                f"MatrixFreeOperator: p = {p}, r = {r} need n_source >= {n_source}, n_test >= {n_field}."
            )
        self.na, self.n_field, self.n_source = na, n_field, n_source
        idx = g.idx
        self.layers = sorted(set(idx[:, 0].tolist()))
        lo = idx[:, 1:].min(axis=0)
        self.pos = idx[:, 1:] - lo
        nx, ny = (idx[:, 1:].max(axis=0) - lo + 1).tolist()
        self.shape = (nx, ny)
        self.by_layer = {z: np.flatnonzero(idx[:, 0] == z) for z in self.layers}
        zu = [2 * z + 1 for z in self.layers]  # centres' z in h
        h, above, below = g.h, g.above, g.below

        # the window's symbolic frame kernels, once per process (~30 s), for both orders of the media (a source in B
        # is the mirrored problem), so that no product pays for them
        for kind, order, media in itertools.product(
            "RT", (0, 1), ((above, below), (below, above))
        ):
            isp.frame_laurent(kind, order, isp._key(media[0]), isp._key(media[1]))
        # D: the low band (non-periodic) and the high band (the period of interface_canonical)
        lp, band = couplings.low_pass(omega), couplings.band_qh
        kx, ky, w = sap.low_band_nodes(
            2.0 * float(np.hypot(nx - 1, ny - 1)), h, omega, above, below, band, lp
        )
        self.low = sap.TwoSidedProduct(
            zu,
            zu,
            self.shape,
            kx,
            ky,
            w,
            h,
            omega,
            above,
            below,
            n_source,
            na,
            band,
            None,
            eps,
            cache=cache,
        )
        reach = max(max(k[2], k[3]) for k in couplings.classes) // 2
        n_grid = scipy.fft.next_fast_len(reach + 1 + couplings.split_pad)
        kx, ky, w = sap.high_band_nodes(n_grid, h, couplings.q0h, band, lp)
        self.high = sap.TwoSidedProduct(
            zu,
            zu,
            self.shape,
            kx,
            ky,
            w,
            h,
            omega,
            above,
            below,
            n_source,
            na,
            couplings.q0h,
            couplings.window,
            eps,
            cache=cache,
            lattice=sap.lattice_of(kx, ky, h, n_grid),
        )

        # the corrections: a stencil per pair of layers, over the offsets whose class carries them
        self.stencils: dict = {}
        for zr, zs in itertools.product(self.layers, repeat=2):
            st = []
            for dx in range(-(nx - 1), nx):
                for dy in range(-(ny - 1), ny):
                    (a, b), q = _canonical(2 * dx, 2 * dy)
                    parts = couplings.classes.get(
                        (2 * zr + 1, 2 * zs + 1, a, b), (None, None)
                    )[0]
                    if parts is None:
                        continue
                    blk = parts[0] + omega**2 * parts[1]
                    if len(parts) > 2:
                        blk = blk + omega**4 * parts[2]
                    st.append((dx, dy, _transform(q, blk)[:na, :n_source]))
            if st:
                self.stencils[zr, zs] = st

        # the whole-space blocks on each side: stored lateral kernels for planes up to stored_planes apart
        # (independent of the number of layers), the plane-wave march (stratified_apply.WholeSpaceMarch) beyond
        ws_cache: dict = {}
        self.sides = []
        lat = (2 * nx - 1, 2 * ny - 1)
        for side in (
            [z for z in self.layers if z < 0],
            [z for z in self.layers if z >= 0],
        ):
            if not side:
                continue
            medium = above if side[0] < 0 else below
            kernels = {}
            for dz in range(-stored_planes, stored_planes + 1):
                if abs(dz) > side[-1] - side[0]:
                    continue
                kh = np.zeros((na * 9, n_source * 9) + lat, dtype=complex)
                for dx, dy in itertools.product(
                    range(-(nx - 1), nx), range(-(ny - 1), ny)
                ):
                    k = _whole_space(
                        (dz, dx, dy),
                        h,
                        omega,
                        medium,
                        couplings.n_source,
                        couplings.n_test,
                        ws_cache,
                        couplings.far_coeffs,
                    )
                    kh[:, :, dx % lat[0], dy % lat[1]] = (
                        k[:na, :n_source]
                        .transpose(0, 2, 1, 3)
                        .reshape(na * 9, n_source * 9)
                    )
                kernels[dz] = np.fft.fft2(kh, axes=(2, 3)).reshape(
                    na * 9, n_source * 9, -1
                )
            march = None
            if side[-1] - side[0] > stored_planes:
                kx, ky, w, lattice = sap.whole_space_nodes(
                    2.0 * float(np.hypot(nx - 1, ny - 1)),
                    n_grid,
                    h,
                    omega,
                    medium,
                    band,
                    lp,
                    qmax_h=16.0 / stored_planes,
                )
                march = sap.WholeSpaceMarch(
                    side,
                    self.shape,
                    kx,
                    ky,
                    w,
                    h,
                    omega,
                    medium,
                    n_source,
                    na,
                    eps,
                    cache=cache,
                    min_gap=stored_planes + 1,
                    lattice=lattice,
                )
            self.sides.append((side, kernels, march))

        # the preconditioner's self blocks: the self class's interface block plus the whole-space self block
        self_keys = [(2 * z + 1, 2 * z + 1, 0, 0) for z in self.layers]
        canon = couplings.interface_canonical(omega, keys=self_keys)
        self.self_blocks = {}
        for z in self.layers:
            medium = above if z < 0 else below
            ws = _whole_space(
                (0, 0, 0),
                h,
                omega,
                medium,
                couplings.n_source,
                couplings.n_test,
                ws_cache,
                couplings.far_coeffs,
            )
            self.self_blocks[z] = (canon[2 * z + 1, 2 * z + 1, 0, 0] + ws)[
                :na, :n_source
            ]
        self.m9 = gram_test(h, n_field)[:na, :na]

    def coupling(self, delta: NDArray, x: NDArray) -> NDArray:
        """sum_n K_mn E^(n) x_n, x of shape (N, na, 9): as ``apply_coupling``."""
        na, n_source = self.na, self.n_source
        e = np.array([source_expansion(d, self.n_field)[:n_source, :na] for d in delta])
        src = np.einsum("ncbij,nbj->nci", e, x).reshape(len(x), n_source * 9)
        return self._convolve(src)

    def _convolve(self, src: NDArray) -> NDArray:
        na, n_source = self.na, self.n_source
        nx, ny = self.shape
        grids = {}
        for z, cells in self.by_layer.items():
            gz = np.zeros((nx, ny, n_source, 9), dtype=complex)
            gz[self.pos[cells, 0], self.pos[cells, 1]] = src[cells].reshape(
                -1, n_source, 9
            )
            grids[z] = gz
        zu = {2 * z + 1: grids[z] for z in self.layers}
        lo, hi = self.low.apply(zu), self.high.apply(zu)
        out_g = {z: lo[2 * z + 1] + hi[2 * z + 1] for z in self.layers}
        for (zr, zs), st in self.stencils.items():
            y, x = out_g[zr], grids[zs]
            for dx, dy, blk in st:
                rx, ry = (
                    slice(max(0, dx), min(nx, nx + dx)),
                    slice(max(0, dy), min(ny, ny + dy)),
                )
                sx, sy = (
                    slice(max(0, -dx), min(nx, nx - dx)),
                    slice(max(0, -dy), min(ny, ny - dy)),
                )
                y[rx, ry] += np.einsum("acij,xycj->xyai", blk, x[sx, sy])
        lat = (2 * nx - 1, 2 * ny - 1)
        spec = {
            z: np.fft.fft2(
                np.pad(
                    g.reshape(nx, ny, -1).transpose(2, 0, 1),
                    ((0, 0), (0, nx - 1), (0, ny - 1)),
                ),
                axes=(1, 2),
            ).reshape(n_source * 9, -1)
            for z, g in grids.items()
        }
        for side, kernels, march in self.sides:
            for z in side:
                acc = np.zeros((na * 9, lat[0] * lat[1]), dtype=complex)
                for dz, kf in kernels.items():
                    if z - dz in spec and (z - dz) in side:
                        acc += np.einsum("rcf,cf->rf", kf, spec[z - dz])
                y = np.fft.ifft2(acc.reshape((na * 9,) + lat), axes=(1, 2))
                out_g[z] += y[:, :nx, :ny].transpose(1, 2, 0).reshape(nx, ny, na, 9)
            if march is not None:
                got = march.apply({z: grids[z] for z in side})
                for z in side:
                    out_g[z] += got[z]
        out = np.zeros((len(src), na, 9), dtype=complex)
        for z, cells in self.by_layer.items():
            out[cells] = out_g[z][self.pos[cells, 0], self.pos[cells, 1]]
        return out


def solve_interface_gmres(
    couplings: InterfaceCouplings,
    omega: float,
    contrasts: list[MaterialContrast | None],
    kx: float,
    ky: float,
    wave: str,
    p: int = 1,
    r: int = 0,
    operator: "LateralOperator | MatrixFreeOperator | None" = None,
    tol: float = 1e-11,
    restart: int = 100,
    max_cycles: int = 20,
) -> tuple[InterfaceResult, int]:
    """``solve_interface`` by GMRES with the lateral FFT products and a block-Jacobi preconditioner
    (M - K_mm E_m)^-1, K_mm the self block with its own image.  Returns the result and the iteration count.

    Raises:
        RuntimeError: when GMRES does not reach ``tol`` within restart * max_cycles iterations.
    """
    from scipy.sparse.linalg import LinearOperator, gmres

    grid = couplings.grid
    op = LateralOperator(couplings, omega, p, r) if operator is None else operator
    na, n_field, n_source = op.na, op.n_field, op.n_source
    n = len(grid.idx)
    delta = cell_contrasts(grid, omega, contrasts, r)
    e = np.array([source_expansion(d, n_field)[:n_source, :na] for d in delta])
    rows = na * 9
    m9 = op.m9

    def matvec(v: NDArray) -> NDArray:
        x = v.reshape(n, na, 9)
        src = np.einsum("ncbij,nbj->nci", e, x).reshape(n, n_source * 9)
        return (np.einsum("ab,nbi->nai", m9, x) - op._convolve(src)).ravel()

    m_full = np.kron(m9, np.eye(9))
    pre = np.array(
        [
            np.linalg.inv(
                m_full
                - np.einsum(
                    "acij,cbjk->aibk", op.self_blocks[grid.idx[m, 0]], e[m]
                ).reshape(rows, rows)
            )
            for m in range(n)
        ]
    )

    def precondition(v: NDArray) -> NDArray:
        return np.einsum("nij,nj->ni", pre, v.reshape(n, rows)).ravel()

    ref = reference_moments(grid, omega, kx, ky, wave, n_field)
    rhs = ref[:, :na].ravel()
    dim = n * rows
    count = [0]

    def counted(v):
        count[0] += 1
        return matvec(v)

    sol, info = gmres(
        LinearOperator((dim, dim), matvec=counted, dtype=complex),
        rhs,
        x0=precondition(rhs),
        M=LinearOperator((dim, dim), matvec=precondition, dtype=complex),
        rtol=tol,
        atol=0.0,
        restart=restart,
        maxiter=max_cycles,
    )
    if info != 0:
        raise RuntimeError(
            f"solve_interface_gmres: GMRES did not reach rtol {tol:g} in {restart * max_cycles} iterations."
        )
    full = np.zeros((n, n_field, 9), dtype=complex)
    full[:, :na] = sol.reshape(n, na, 9)
    return InterfaceResult(grid, omega, delta, ref, full, p, r), count[0]
