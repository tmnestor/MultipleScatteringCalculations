"""Ocean-bottom reflection with heterogeneous sediment.

Simulates ocean-bottom reflection from a Ricker wavelet through a 3-layer model:
water (acoustic) | heterogeneous sediment (M×M×N_z slab) | elastic halfspace.

Uses the existing Kennett recursion (kennett_layers) for the complete homogeneous
background response. For the heterogeneous slab, R_slab is injected into the
sub-ocean reflectivity and dressed by the water-sed interface via a single
Kennett recursion step. At p>0, P-S conversion exists in the solid but the
fluid-solid interface PP path remains well-defined as coeff.Rd[0,0].
"""

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .effective_contrasts import MaterialContrast, ReferenceMedium
from .kennett_layers import (
    FluidLayer,
    IsotropicLayer,
    LayerStack,
    _complex_slowness,
    _vertical_slowness,
    kennett_layers,
    psv_fluid_solid,
)
from .seismic_survey import ricker_source_spectrum
from .slab_scattering import (
    SlabGeometry,
    SlabMaterial,
    compute_slab_scattering,
    compute_slab_tmatrices,
    random_slab_material,
    slab_rpp_periodic,
    uniform_slab_material,
)


@dataclass
class OceanBottomConfig:
    """Configuration for ocean-bottom reflection simulation.

    Args:
        water_alpha: Water P-wave velocity (m/s).
        water_rho: Water density (kg/m³).
        water_depth: Water column depth (m).
        sed_ref: Sediment background (reference for slab contrasts).
        hs_alpha: Halfspace P-wave velocity (m/s).
        hs_beta: Halfspace S-wave velocity (m/s).
        hs_rho: Halfspace density (kg/m³).
        geometry: M×M×N_z slab geometry.
        material: Per-cube contrasts relative to sed_ref.
        f_peak: Ricker wavelet peak frequency (Hz).
        T: Record length (s).
        nw: Number of positive frequencies.
        gamma: Damping (rad/s). None = auto (pi/T).
        f_min: Minimum active frequency (Hz).
        f_max: Maximum active frequency (Hz).
        free_surface: Include free-surface reverberations (water-column multiples).
        p: Horizontal slowness (s/m). Default 0.0 (normal incidence).
    """

    water_alpha: float
    water_rho: float
    water_depth: float
    sed_ref: ReferenceMedium
    hs_alpha: float
    hs_beta: float
    hs_rho: float
    geometry: SlabGeometry
    material: SlabMaterial
    f_peak: float
    T: float
    nw: int
    gamma: float | None = None
    f_min: float = 1.0
    f_max: float = 200.0
    free_surface: bool = False
    p: float = 0.0


@dataclass
class OceanBottomResult:
    """Result of ocean-bottom reflection simulation.

    Attributes:
        time: Time axis (s), shape (nt,).
        trace_total: Total seismogram (homogeneous + slab), shape (nt,).
        trace_homogeneous: Homogeneous layered seismogram, shape (nt,).
        R_bg: Background Kennett R_PP (from kennett_layers), shape (nw-1,), complex.
        R_slab: Raw slab scattering R_PP (before coupling), shape (nw-1,), complex.
        R_total: Total R_PP with slab dressed by interface coupling, shape (nw-1,), complex.
        omega_real: Real frequency axis (rad/s), shape (nw-1,).
        config: Configuration used.
        n_gmres_iters: GMRES iterations per active frequency.
        freq_elapsed: Per-frequency wall-clock time (s), one per active frequency.
        elapsed_seconds: Total wall-clock time (s).
    """

    time: NDArray[np.floating]
    trace_total: NDArray[np.floating]
    trace_homogeneous: NDArray[np.floating]
    R_bg: NDArray[np.complexfloating]
    R_slab: NDArray[np.complexfloating]
    R_total: NDArray[np.complexfloating]
    omega_real: NDArray[np.floating]
    config: OceanBottomConfig
    n_gmres_iters: list[int] = field(default_factory=list)
    freq_elapsed: list[float] = field(default_factory=list)
    elapsed_seconds: float = 0.0


def _kennett_water_step(
    MT_pp: NDArray,
    cfg: OceanBottomConfig,
) -> NDArray:
    """Kennett recursion at the water-sediment interface for PP.

    Computes R_surface_PP = Rd + Tu · MT · (1 − Ru · MT)⁻¹ · Td
    using modified fluid-solid scattering coefficients. At p>0, P-S
    conversion exists but the PP path through the fluid-solid interface
    is still well-defined as coeff.Rd[0,0].

    Args:
        MT_pp: Phase-shifted sub-ocean PP reflectivity, shape (nfreq,).
        cfg: Ocean-bottom configuration (provides water and sediment properties).

    Returns:
        PP reflection coefficient at the water-sed interface, shape (nfreq,).
    """
    p = cfg.p

    # Complex slownesses at given p
    s_water = _complex_slowness(cfg.water_alpha, np.inf)
    eta_water = _vertical_slowness(s_water, p)
    s_sed_p = _complex_slowness(cfg.sed_ref.alpha, np.inf)
    eta_sed = _vertical_slowness(s_sed_p, p)
    s_sed_s = _complex_slowness(cfg.sed_ref.beta, np.inf)
    neta_sed = _vertical_slowness(s_sed_s, p)
    beta_sed = 1.0 / s_sed_s

    # Fluid-solid interface R/T coefficients (modified form)
    coeff = psv_fluid_solid(
        p, eta_water, cfg.water_rho, eta_sed, neta_sed, cfg.sed_ref.rho, beta_sed
    )
    Rd = coeff.Rd[0, 0]
    Ru = coeff.Ru[0, 0]
    Td = coeff.Td[0, 0]
    Tu = coeff.Tu[0, 0]

    # Scalar Kennett recursion: PP path through fluid-solid interface
    U = 1.0 / (1.0 - Ru * MT_pp)
    return Rd + Tu * MT_pp * U * Td


def compute_ocean_bottom_reflection(
    config: OceanBottomConfig,
    *,
    volume_averaged: bool = False,
    n_orders: int = 2,
    gmres_tol: float = 1e-6,
    progress: bool = True,
) -> OceanBottomResult:
    """Compute ocean-bottom reflection with heterogeneous sediment slab.

    The homogeneous background is computed via the existing kennett_layers
    recursion for the full [water|sed|hs] stack. For the heterogeneous total,
    R_slab is injected into the sub-ocean reflectivity and dressed by the
    water-sed interface via a single Kennett step (Td·Tu coupling +
    sediment-internal reverberations).

    Optionally includes free-surface reverberations (water-column multiples).

    Args:
        config: Simulation configuration.
        volume_averaged: Use volume-averaged inter-voxel propagator.
        n_orders: Dynamic correction orders for volume-averaged propagator.
        gmres_tol: GMRES tolerance for slab solver.
        progress: Show tqdm progress bar.

    Returns:
        OceanBottomResult with time-domain traces and frequency-domain reflectivities.
    """
    t_start = time.perf_counter()
    cfg = config
    p = cfg.p

    # Validate: p must be below critical angle for water
    if p >= 1.0 / cfg.water_alpha:
        msg = (
            f"Horizontal slowness p={p:.6e} s/m exceeds critical angle "
            f"for water (1/α_water = {1.0 / cfg.water_alpha:.6e} s/m). "
            f"P-wave becomes evanescent in the water column."
        )
        raise ValueError(msg)

    # Damping
    gamma = cfg.gamma if cfg.gamma is not None else np.pi / cfg.T

    # Frequency grid
    dw = 2.0 * np.pi / cfg.T
    nwm = cfg.nw - 1
    omega_real = np.arange(1, nwm + 1, dtype=np.float64) * dw
    omega_damped = omega_real + 1j * gamma

    # Time axis
    nt = 2 * cfg.nw
    time_axis = np.arange(nt, dtype=np.float64) * (cfg.T / float(nt))

    # Active frequency band
    freq_hz = omega_real / (2.0 * np.pi)
    active_mask = (freq_hz >= cfg.f_min) & (freq_hz <= cfg.f_max)
    active_indices = np.where(active_mask)[0]

    # ── Vertical slownesses ───────────────────────────────────────────
    s_water = _complex_slowness(cfg.water_alpha, np.inf)
    eta_water = _vertical_slowness(s_water, p)
    s_sed_p = _complex_slowness(cfg.sed_ref.alpha, np.inf)
    eta_sed = _vertical_slowness(s_sed_p, p)

    # ── Kennett background R_PP (full 3-layer recursion) ──────────────
    H = cfg.geometry.N_z * cfg.geometry.d
    full_stack = LayerStack(
        layers=[
            FluidLayer(
                alpha=cfg.water_alpha, rho=cfg.water_rho, thickness=cfg.water_depth
            ),
            IsotropicLayer(
                alpha=cfg.sed_ref.alpha,
                beta=cfg.sed_ref.beta,
                rho=cfg.sed_ref.rho,
                thickness=H,
            ),
            IsotropicLayer(
                alpha=cfg.hs_alpha,
                beta=cfg.hs_beta,
                rho=cfg.hs_rho,
                thickness=np.inf,
            ),
        ]
    )
    omega_kennett = omega_damped[active_indices]
    kennett_result = kennett_layers(full_stack, p=p, omega=omega_kennett)

    R_bg = np.zeros(nwm, dtype=complex)
    R_bg[active_indices] = kennett_result.RPP

    # ── Sub-ocean sed-hs reflection (for slab injection) ──────────────
    sub_stack = LayerStack(
        layers=[
            IsotropicLayer(
                alpha=cfg.sed_ref.alpha,
                beta=cfg.sed_ref.beta,
                rho=cfg.sed_ref.rho,
                thickness=H,
            ),
            IsotropicLayer(
                alpha=cfg.hs_alpha,
                beta=cfg.hs_beta,
                rho=cfg.hs_rho,
                thickness=np.inf,
            ),
        ]
    )
    sub_result = kennett_layers(sub_stack, p=p, omega=omega_kennett)
    R_sed_hs = sub_result.RPP[0]

    # ── Sediment two-way phase ────────────────────────────────────────
    E2_sed = np.exp(2j * omega_damped * eta_sed * H)

    # ── Slab scattering R_PP ─────────────────────────────────────────
    R_slab = np.zeros(nwm, dtype=complex)
    # k_hat from slowness: [η_P·α, p·α, 0] in (z,x,y) — downgoing
    k_hat = np.array(
        [float(np.real(eta_sed * cfg.sed_ref.alpha)), float(p * cfg.sed_ref.alpha), 0.0]
    )
    n_gmres_iters: list[int] = []
    freq_elapsed: list[float] = []

    if progress:
        from tqdm.auto import tqdm

        iterator = tqdm(active_indices, desc="Slab scattering", leave=False)
    else:
        iterator = active_indices

    for iw in iterator:
        t_freq = time.perf_counter()
        w = float(omega_real[iw])
        result = compute_slab_scattering(
            cfg.geometry,
            cfg.material,
            w,
            k_hat,
            wave_type="P",
            gmres_tol=gmres_tol,
            periodic=True,
            volume_averaged=volume_averaged,
            n_orders=n_orders,
        )
        T_local = compute_slab_tmatrices(cfg.geometry, cfg.material, w)
        R_slab[iw] = slab_rpp_periodic(result, T_local, p=p)
        n_gmres_iters.append(result.n_gmres_iter)
        freq_elapsed.append(time.perf_counter() - t_freq)

    # ── Total R_PP with slab injection into Kennett recursion ─────────
    # MT = E²_sed · R_sed_hs + R_slab: sub-ocean reflectivity with slab
    MT_total = E2_sed * R_sed_hs + R_slab
    # Single Kennett step at water-sed interface dresses R_slab with
    # Td·Tu coupling and includes sediment-internal reverberations
    R_total = np.zeros(nwm, dtype=complex)
    R_total[active_indices] = _kennett_water_step(MT_total[active_indices], cfg)

    # ── Water column two-way phase ────────────────────────────────────
    water_phase = np.exp(2j * omega_damped * eta_water * cfg.water_depth)

    # ── Free surface reverberations ───────────────────────────────────
    if cfg.free_surface:

        def _apply_water(R: NDArray) -> NDArray:
            """Water phase + free-surface reverberations."""
            E2R = water_phase * R
            return E2R / (1.0 + E2R)
    else:

        def _apply_water(R: NDArray) -> NDArray:
            """Water phase only (no free surface)."""
            return water_phase * R

    R_obs_hom = _apply_water(R_bg)
    R_obs_total = _apply_water(R_total)

    # ── Source spectrum ───────────────────────────────────────────────
    S = ricker_source_spectrum(omega_real, cfg.f_peak)

    # ── IFFT with damping compensation ────────────────────────────────
    exp_decay = np.exp(-gamma * time_axis)

    def _ifft_trace(R: NDArray) -> NDArray:
        """Hermitian-symmetric IFFT with damping compensation."""
        U = S * R
        Uwk = np.zeros(nt, dtype=complex)
        Uwk[1 : cfg.nw] = U
        Uwk[cfg.nw + 1 :] = np.conj(U[::-1])
        seismogram_c = np.fft.fft(Uwk)
        return np.real(seismogram_c) * exp_decay

    trace_total = _ifft_trace(R_obs_total)
    trace_homogeneous = _ifft_trace(R_obs_hom)

    elapsed = time.perf_counter() - t_start

    return OceanBottomResult(
        time=time_axis,
        trace_total=trace_total,
        trace_homogeneous=trace_homogeneous,
        R_bg=R_bg,
        R_slab=R_slab,
        R_total=R_total,
        omega_real=omega_real,
        config=config,
        n_gmres_iters=n_gmres_iters,
        freq_elapsed=freq_elapsed,
        elapsed_seconds=elapsed,
    )


def write_log(result: OceanBottomResult, path: str | Path) -> None:
    """Write a structured log file for an ocean-bottom simulation.

    Records configuration, material statistics, per-frequency solver
    diagnostics, and summary timing.

    Args:
        result: Completed simulation result.
        path: Output log file path.
    """
    from datetime import datetime, timezone

    path = Path(path)
    cfg = result.config
    geom = cfg.geometry
    H = geom.N_z * geom.d

    # Material statistics
    Dl = cfg.material.Dlambda
    Dm = cfg.material.Dmu
    Dr = cfg.material.Drho
    n_cubes = Dl.size
    n_inclusions = int(np.sum(Dl > 0))
    phi_realised = n_inclusions / n_cubes if n_cubes > 0 else 0.0

    # Active frequencies
    freq_hz = result.omega_real / (2.0 * np.pi)
    active_mask = (freq_hz >= cfg.f_min) & (freq_hz <= cfg.f_max)
    active_indices = np.where(active_mask)[0]

    with path.open("w") as f:
        f.write("# Ocean-Bottom Reflection Log\n")
        f.write(f"# Generated: {datetime.now(timezone.utc).isoformat()}\n\n")

        # Configuration
        f.write("## Configuration\n\n")
        f.write(f"Water:        α = {cfg.water_alpha / 1e3:.3f} km/s, ")
        f.write(f"ρ = {cfg.water_rho / 1e3:.3f} g/cm³, ")
        f.write(f"depth = {cfg.water_depth / 1e3:.4f} km\n")
        f.write(f"Sediment:     α = {cfg.sed_ref.alpha / 1e3:.3f} km/s, ")
        f.write(f"β = {cfg.sed_ref.beta / 1e3:.3f} km/s, ")
        f.write(f"ρ = {cfg.sed_ref.rho / 1e3:.3f} g/cm³\n")
        f.write(f"Halfspace:    α = {cfg.hs_alpha / 1e3:.3f} km/s, ")
        f.write(f"β = {cfg.hs_beta / 1e3:.3f} km/s, ")
        f.write(f"ρ = {cfg.hs_rho / 1e3:.3f} g/cm³\n")
        f.write(f"Slab:         {geom.M}×{geom.M}×{geom.N_z}, ")
        f.write(f"a = {geom.a:.3f} m, d = {geom.d:.3f} m, H = {H:.3f} m\n")
        f.write(f"Slowness:     p = {cfg.p * 1e3:.6f} s/km\n")
        if cfg.p > 0:
            angle_deg = np.degrees(np.arcsin(cfg.p * cfg.water_alpha))
            f.write(f"              θ_water = {angle_deg:.2f}°\n")
        f.write(f"Free surface: {cfg.free_surface}\n")
        f.write(f"Wavelet:      Ricker f_peak = {cfg.f_peak} Hz\n")
        f.write(f"Recording:    T = {cfg.T} s, nw = {cfg.nw}, ")
        f.write(f"f = [{cfg.f_min}–{cfg.f_max}] Hz\n\n")

        # Material statistics
        f.write("## Material statistics\n\n")
        f.write(f"Cubes:        {n_cubes} ({geom.M}×{geom.M}×{geom.N_z})\n")
        f.write(f"Inclusions:   {n_inclusions} (φ_realised = {phi_realised:.3f})\n")
        if n_inclusions > 0 and n_inclusions < n_cubes:
            # Binary random — report moments
            incl_Dl = Dl[Dl > 0]
            f.write(f"Inclusion Δλ: {incl_Dl[0] / 1e9:.4f} GPa\n")
            f.write(f"Inclusion Δμ: {Dm[Dm > 0][0] / 1e9:.4f} GPa\n")
            f.write(f"Inclusion Δρ: {Dr[Dr > 0][0] / 1e3:.4f} g/cm³\n")
        f.write(f"Mean Δλ:      {np.mean(Dl) / 1e9:.6f} GPa\n")
        f.write(f"Mean Δμ:      {np.mean(Dm) / 1e9:.6f} GPa\n")
        f.write(f"Mean Δρ:      {np.mean(Dr) / 1e3:.6f} g/cm³\n")
        f.write(f"Var  Δλ:      {np.var(Dl) / 1e18:.6f} GPa²\n")
        f.write(f"Var  Δμ:      {np.var(Dm) / 1e18:.6f} GPa²\n")
        f.write(f"Var  Δρ:      {np.var(Dr) / 1e6:.6f} (g/cm³)²\n")
        if np.mean(Dl) != 0:
            cv_Dl = np.std(Dl) / np.mean(Dl)
            f.write(f"σ/μ (Δλ):     {cv_Dl:.3f}\n")
        f.write("\n")

        # Per-frequency table
        f.write("## Per-frequency diagnostics\n\n")
        f.write(f"{'freq_Hz':>10s}  {'|R_slab|':>10s}  {'|R_total|':>10s}  ")
        f.write(f"{'GMRES_iter':>10s}  {'elapsed_ms':>10s}\n")
        f.write("-" * 58 + "\n")

        for i, iw in enumerate(active_indices):
            fhz = freq_hz[iw]
            r_slab = abs(result.R_slab[iw])
            r_total = abs(result.R_total[iw])
            gm = result.n_gmres_iters[i] if i < len(result.n_gmres_iters) else -1
            dt = result.freq_elapsed[i] * 1e3 if i < len(result.freq_elapsed) else -1
            f.write(f"{fhz:10.2f}  {r_slab:10.2e}  {r_total:10.2e}  ")
            f.write(f"{gm:10d}  {dt:10.2f}\n")
        f.write("\n")

        # Summary
        f.write("## Summary\n\n")
        f.write(f"Active frequencies: {len(active_indices)}\n")
        if result.n_gmres_iters:
            gmres_iters = result.n_gmres_iters
            f.write(f"GMRES iters:  mean = {np.mean(gmres_iters):.1f}, ")
            f.write(f"min = {min(gmres_iters)}, max = {max(gmres_iters)}\n")
        if result.freq_elapsed:
            freq_times = result.freq_elapsed
            f.write(f"Per-freq (ms): mean = {np.mean(freq_times) * 1e3:.1f}, ")
            f.write(
                f"min = {min(freq_times) * 1e3:.1f}, max = {max(freq_times) * 1e3:.1f}\n"
            )
            f.write(f"Slab total:   {sum(freq_times):.2f} s\n")
        f.write(f"Total elapsed: {result.elapsed_seconds:.2f} s\n")
        f.write(f"Peak |R_bg|:   {np.max(np.abs(result.R_bg)):.6f}\n")
        f.write(f"Peak |R_slab|: {np.max(np.abs(result.R_slab)):.6f}\n")
        f.write(f"Peak |R_total|: {np.max(np.abs(result.R_total)):.6f}\n")


def load_ocean_bottom_config(path: str | Path) -> OceanBottomConfig:
    """Load ocean-bottom simulation configuration from a YAML file.

    YAML uses seismic units (km/s, g/cm³, GPa, km, s/km) for better
    conditioning. Internally the code uses SI (m/s, kg/m³, Pa, m, s/m).

    Args:
        path: Path to YAML config file.

    Returns:
        OceanBottomConfig populated from YAML (SI units internally).

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If required sections are missing.
    """
    import yaml

    path = Path(path)
    if not path.exists():
        msg = f"Config file not found: {path}"
        raise FileNotFoundError(msg)

    with path.open() as f:
        cfg = yaml.safe_load(f)

    # Validate required sections
    required = ["ocean", "sediment", "halfspace", "slab", "recording"]
    missing = [s for s in required if s not in cfg]
    if missing:
        msg = (
            f"Missing required config sections: {missing}. "
            f"Expected sections: {required}. "
            f"See configs/example_ocean_bottom.yml for an example."
        )
        raise ValueError(msg)

    oc = cfg["ocean"]
    sed = cfg["sediment"]
    hs = cfg["halfspace"]
    sl = cfg["slab"]
    rec = cfg["recording"]
    slow = cfg.get("slowness", {})

    # Unit conversions: seismic → SI
    # Velocity: km/s → m/s (×1e3)
    # Density: g/cm³ → kg/m³ (×1e3)
    # Length: km → m (×1e3)
    # Modulus: GPa → Pa (×1e9)
    # Slowness: s/km → s/m (×1e-3)

    # Build reference medium (SI)
    sed_ref = ReferenceMedium(
        alpha=sed["alpha"] * 1e3,
        beta=sed["beta"] * 1e3,
        rho=sed["rho"] * 1e3,
    )

    # Build slab geometry (SI: metres)
    geom = SlabGeometry(M=sl["M"], N_z=sl.get("N_z", 1), a=sl["a"] * 1e3)

    # Build material (SI: Pa, kg/m³)
    #
    # Two parameterizations:
    #   contrast: {Dlambda, Dmu, Drho}  — raw inclusion contrast
    #   mean:     {Dlambda, Dmu, Drho}  — ensemble mean; inclusion = mean/φ
    # Exactly one of "contrast" or "mean" must be present.
    has_contrast = "contrast" in sl
    has_mean = "mean" in sl
    if has_contrast == has_mean:
        msg = (
            "slab section must contain exactly one of 'contrast' or 'mean'. "
            "Use 'contrast' for raw inclusion values, 'mean' for ensemble-mean "
            "parameterization (inclusion contrast = mean / phi)."
        )
        raise ValueError(msg)

    mat_type = sl.get("material", "uniform")
    phi = sl.get("phi", 1.0 if mat_type == "uniform" else 0.3)

    if has_mean:
        m = sl["mean"]
        if mat_type == "uniform":
            # For uniform: mean = contrast (every cube is an inclusion)
            contrast = MaterialContrast(
                Dlambda=m["Dlambda"] * 1e9,
                Dmu=m["Dmu"] * 1e9,
                Drho=m["Drho"] * 1e3,
            )
        else:
            # For random binary: inclusion contrast = mean / φ
            contrast = MaterialContrast(
                Dlambda=m["Dlambda"] * 1e9 / phi,
                Dmu=m["Dmu"] * 1e9 / phi,
                Drho=m["Drho"] * 1e3 / phi,
            )
    else:
        c = sl["contrast"]
        contrast = MaterialContrast(
            Dlambda=c["Dlambda"] * 1e9,
            Dmu=c["Dmu"] * 1e9,
            Drho=c["Drho"] * 1e3,
        )

    if mat_type == "random":
        material = random_slab_material(
            geom,
            sed_ref,
            contrast,
            phi=phi,
            seed=sl.get("seed", 42),
        )
    else:
        material = uniform_slab_material(geom, sed_ref, contrast)

    return OceanBottomConfig(
        water_alpha=oc["water_alpha"] * 1e3,
        water_rho=oc["water_rho"] * 1e3,
        water_depth=oc["water_depth"] * 1e3,
        sed_ref=sed_ref,
        hs_alpha=hs["alpha"] * 1e3,
        hs_beta=hs["beta"] * 1e3,
        hs_rho=hs["rho"] * 1e3,
        geometry=geom,
        material=material,
        f_peak=rec["f_peak"],
        T=rec["T"],
        nw=rec["nw"],
        f_min=rec.get("f_min", 5.0),
        f_max=rec.get("f_max", 100.0),
        free_surface=oc.get("free_surface", False),
        p=slow.get("p", 0.0) * 1e-3,  # s/km → s/m
    )
