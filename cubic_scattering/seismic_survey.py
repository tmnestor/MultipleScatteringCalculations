"""Towed hydrophone marine seismic reflection survey simulation.

Simulates synthetic shot gathers for a marine towed-streamer survey with:
  - Kennett reflectivity (batched over slowness and frequency)
  - Free surface reverberations (water-column multiples)
  - Source and receiver ghost operators
  - Bessel-weighted wavenumber summation (slowness-to-offset)
  - IFFT with complex-frequency damping compensation

Pipeline:
    YAML config --> build_survey_stack() --> kennett_reflectivity_batch()
    --> free_surface_reverberations() --> ghosts --> bessel_summation()
    --> source spectrum --> IFFT --> ShotGatherResult

References:
    Bouchon, M. (1981). A simple method to calculate Green's functions
    for elastic layered media. BSSA, 71(4), 959-971.
    Kennett, B.L.N. (1983). Seismic Wave Propagation in Stratified Media.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.special import j0

from .kennett_layers import (
    FluidLayer,
    IsotropicLayer,
    LayerStack,
    _complex_slowness,
    _vertical_slowness,
    kennett_reflectivity_batch,
)

logger = logging.getLogger(__name__)


# ── Data structures ─────────────────────────────────────────────────────


@dataclass
class SurveyConfig:
    """Marine survey geometry.

    Attributes:
        source_depth: Source depth below free surface (m).
        receiver_depth: Streamer depth below free surface (m).
        receiver_type: 'hydrophone' (pressure) or 'geophone' (velocity).
        offsets: Receiver offsets from source (m), shape (nr,).
        water_depth: Water depth (m).
        water_alpha: Water P-wave velocity (m/s).
        water_rho: Water density (kg/m^3).
    """

    source_depth: float
    receiver_depth: float
    receiver_type: str
    offsets: np.ndarray
    water_depth: float
    water_alpha: float = 1500.0
    water_rho: float = 1025.0

    def __post_init__(self) -> None:
        if self.receiver_type not in ("hydrophone", "geophone"):
            msg = f"receiver_type must be 'hydrophone' or 'geophone', got '{self.receiver_type}'"
            raise ValueError(msg)
        if self.source_depth < 0:
            msg = f"source_depth must be >= 0, got {self.source_depth}"
            raise ValueError(msg)
        if self.receiver_depth < 0:
            msg = f"receiver_depth must be >= 0, got {self.receiver_depth}"
            raise ValueError(msg)
        if self.water_depth <= 0:
            msg = f"water_depth must be > 0, got {self.water_depth}"
            raise ValueError(msg)


@dataclass
class GatherConfig:
    """Parameters for the shot gather computation.

    Attributes:
        f_min: Minimum frequency (Hz).
        f_max: Maximum frequency (Hz).
        T: Record length (s).
        nw: Number of positive frequencies.
        np_slow: Number of slowness samples.
        p_max: Maximum slowness (s/m). None = auto (1.2/alpha_min).
        gamma: Complex frequency damping (rad/s). None = auto (pi/T).
        source_type: Source wavelet type.
        f_peak: Peak frequency for Ricker wavelet (Hz).
        free_surface: Include free surface reverberations.
    """

    f_min: float = 5.0
    f_max: float = 80.0
    T: float = 4.0
    nw: int = 4096
    np_slow: int = 8192
    p_max: float | None = None
    gamma: float | None = None
    source_type: str = "ricker"
    f_peak: float = 30.0
    free_surface: bool = True


@dataclass
class ShotGatherResult:
    """Result of shot gather computation.

    Attributes:
        time: Time axis (s), shape (nt,).
        offsets: Receiver offsets (m), shape (nr,).
        gather: Seismogram gather, shape (nr, nt).
        reflectivity: Raw PP reflectivity, shape (np_slow, nfreq).
        stack: Layer stack used.
    """

    time: np.ndarray
    offsets: np.ndarray
    gather: np.ndarray
    reflectivity: np.ndarray
    stack: LayerStack


# ── Ghost operators ─────────────────────────────────────────────────────


def source_ghost(
    omega: np.ndarray,
    p: np.ndarray,
    z_s: float,
    alpha: float,
) -> np.ndarray:
    """Source ghost operator for an explosive source below a free surface.

    Ghost = 1 - exp(2j omega q_alpha z_s) where q_alpha = sqrt(1/alpha^2 - p^2).
    The free surface reflection is -1 for pressure.

    Args:
        omega: Angular frequencies, shape (nfreq,).
        p: Horizontal slowness, shape (np_slow,).
        z_s: Source depth below free surface (m).
        alpha: Water P-wave velocity (m/s).

    Returns:
        Ghost operator, shape (np_slow, nfreq).
    """
    s_alpha = complex(1.0 / alpha)
    q = np.array([_vertical_slowness(s_alpha, float(pv)) for pv in p])  # (np_slow,)
    phase = 2j * q[:, np.newaxis] * omega[np.newaxis, :] * z_s  # (np_slow, nfreq)
    return 1.0 - np.exp(phase)


def receiver_ghost(
    omega: np.ndarray,
    p: np.ndarray,
    z_r: float,
    alpha: float,
    receiver_type: str = "hydrophone",
) -> np.ndarray:
    """Receiver ghost operator for a towed streamer.

    Hydrophone (pressure): 1 + exp(2j omega q z_r)  (pressure doubles)
    Geophone (velocity):   1 - exp(2j omega q z_r)  (velocity reverses)

    Args:
        omega: Angular frequencies, shape (nfreq,).
        p: Horizontal slowness, shape (np_slow,).
        z_r: Receiver depth below free surface (m).
        alpha: Water P-wave velocity (m/s).
        receiver_type: 'hydrophone' or 'geophone'.

    Returns:
        Ghost operator, shape (np_slow, nfreq).
    """
    s_alpha = complex(1.0 / alpha)
    q = np.array([_vertical_slowness(s_alpha, float(pv)) for pv in p])
    phase = 2j * q[:, np.newaxis] * omega[np.newaxis, :] * z_r
    if receiver_type == "hydrophone":
        return 1.0 + np.exp(phase)
    return 1.0 - np.exp(phase)


# ── Free surface reverberations ─────────────────────────────────────────


def free_surface_reverberations(
    RRd_PP: np.ndarray,
    eaea_water: np.ndarray,
) -> np.ndarray:
    """Apply free surface reverberations (water-column multiples).

    R_total = E^2 RRd / (1 + E^2 RRd)

    where E^2 is the two-way phase through the water column and
    R_fs = -1 (pressure-release surface).

    Args:
        RRd_PP: Sub-ocean PP reflectivity, shape (np_slow, nfreq).
        eaea_water: Two-way water phase exp(2j omega eta_w h_w),
            shape (np_slow, nfreq).

    Returns:
        Total reflectivity with multiples, shape (np_slow, nfreq).
    """
    numerator = eaea_water * RRd_PP
    denominator = 1.0 + eaea_water * RRd_PP
    return numerator / denominator


# ── Source spectrum ──────────────────────────────────────────────────────


def ricker_source_spectrum(
    omega: np.ndarray,
    f_peak: float,
) -> np.ndarray:
    """Ricker wavelet in frequency domain.

    S(omega) = (omega/omega_p)^2 exp(-(omega/omega_p)^2)

    Args:
        omega: Real angular frequencies, shape (nfreq,).
        f_peak: Peak frequency (Hz).

    Returns:
        Source spectrum, shape (nfreq,).
    """
    omega_p = 2.0 * np.pi * f_peak
    ratio_sq = (omega / omega_p) ** 2
    return ratio_sq * np.exp(-ratio_sq)


# ── Bessel summation ─────────────────────────────────────────────────────


def bessel_summation(
    R: np.ndarray,
    p_samples: np.ndarray,
    omega_real: np.ndarray,
    offsets: np.ndarray,
    dp: float,
) -> np.ndarray:
    """CPU Bessel-weighted wavenumber summation (slowness to offset).

    U[r, w] = dp * sum_j R[j,w] * J0(omega_w * p_j * r) * p_j * omega_w

    Args:
        R: Reflectivity, shape (np_slow, nfreq).
        p_samples: Slowness samples, shape (np_slow,).
        omega_real: Real angular frequencies, shape (nfreq,).
        offsets: Receiver offsets (m), shape (nr,).
        dp: Slowness spacing (s/m).

    Returns:
        Spectral displacement U(r, omega), shape (nr, nfreq).
    """
    np_slow = len(p_samples)
    nfreq = len(omega_real)
    nr = len(offsets)

    # Hann taper to suppress Gibbs ringing from slowness truncation
    p_max = p_samples[-1]
    taper = 0.5 * (1.0 + np.cos(np.pi * p_samples / p_max))

    # Weighted reflectivity: W[j, w] = R[j,w] * taper[j] * p_j * omega_w * dp
    weight = (
        taper[:, np.newaxis] * p_samples[:, np.newaxis] * omega_real[np.newaxis, :] * dp
    )
    W = R * weight  # (np_slow, nfreq)

    # Accumulate in chunks to manage memory
    from tqdm.auto import tqdm

    U = np.zeros((nr, nfreq), dtype=np.complex128)
    chunk_size = min(np_slow, 256)
    chunks = range(0, np_slow, chunk_size)

    for j_start in tqdm(chunks, desc="Bessel summation", leave=False):
        j_end = min(j_start + chunk_size, np_slow)
        W_chunk = W[j_start:j_end]  # (n_chunk, nfreq)
        p_chunk = p_samples[j_start:j_end]  # (n_chunk,)

        # Bessel arguments: (nr, n_chunk, nfreq)
        arg = (
            offsets[:, None, None] * p_chunk[None, :, None] * omega_real[None, None, :]
        )
        J0_vals = j0(arg)  # (nr, n_chunk, nfreq)

        # Accumulate: U[r, w] += sum_j J0[r,j,w] * W[j,w]
        U += np.einsum("rjw,jw->rw", J0_vals, W_chunk)

    return U


def bessel_summation_gpu(
    R: np.ndarray,
    p_samples: np.ndarray,
    omega_real: np.ndarray,
    offsets: np.ndarray,
    dp: float,
) -> np.ndarray:
    """GPU-accelerated Bessel-weighted wavenumber summation.

    Same as bessel_summation() but uses PyTorch for the einsum on GPU.
    Falls back to CPU if no GPU is available.

    Args:
        R: Reflectivity, shape (np_slow, nfreq).
        p_samples: Slowness samples, shape (np_slow,).
        omega_real: Real angular frequencies, shape (nfreq,).
        offsets: Receiver offsets (m), shape (nr,).
        dp: Slowness spacing (s/m).

    Returns:
        Spectral displacement U(r, omega), shape (nr, nfreq).
    """
    try:
        import torch
    except ImportError:
        logger.warning("PyTorch not available, falling back to CPU Bessel summation")
        return bessel_summation(R, p_samples, omega_real, offsets, dp)

    # Detect device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        return bessel_summation(R, p_samples, omega_real, offsets, dp)

    np_slow = len(p_samples)
    nfreq = len(omega_real)
    nr = len(offsets)

    p_max = p_samples[-1]
    taper = 0.5 * (1.0 + np.cos(np.pi * p_samples / p_max))
    weight = (
        taper[:, np.newaxis] * p_samples[:, np.newaxis] * omega_real[np.newaxis, :] * dp
    )
    W = R * weight

    from tqdm.auto import tqdm

    U = np.zeros((nr, nfreq), dtype=np.complex128)
    chunk_size = min(np_slow, 256)
    chunks = range(0, np_slow, chunk_size)

    off_t = torch.from_numpy(offsets.astype(np.float32)).to(device)
    omega_t = torch.from_numpy(omega_real.astype(np.float32)).to(device)

    for j_start in tqdm(chunks, desc="Bessel summation (GPU)", leave=False):
        j_end = min(j_start + chunk_size, np_slow)
        W_chunk = W[j_start:j_end]
        W_re = torch.from_numpy(W_chunk.real.astype(np.float32)).to(device)
        W_im = torch.from_numpy(W_chunk.imag.astype(np.float32)).to(device)
        p_chunk = torch.from_numpy(p_samples[j_start:j_end].astype(np.float32)).to(
            device
        )

        arg = off_t[:, None, None] * p_chunk[None, :, None] * omega_t[None, None, :]
        arg_cpu = arg.cpu().numpy()
        J0_cpu = j0(arg_cpu).astype(np.float32)
        J0_t = torch.from_numpy(J0_cpu).to(device)

        U_re_chunk = torch.einsum("rjw,jw->rw", J0_t, W_re)
        U_im_chunk = torch.einsum("rjw,jw->rw", J0_t, W_im)

        U += U_re_chunk.cpu().numpy().astype(np.float64)
        U += 1j * U_im_chunk.cpu().numpy().astype(np.float64)

    return U


# ── Stack builders ───────────────────────────────────────────────────────


def build_survey_stack(
    survey: SurveyConfig,
    sediment_layers: list[IsotropicLayer],
    half_space: IsotropicLayer,
) -> LayerStack:
    """Build a LayerStack with a water column on top.

    Args:
        survey: Survey configuration (water depth, velocity, density).
        sediment_layers: List of sub-ocean sediment/crust layers.
        half_space: Bottom half-space layer (thickness must be inf).

    Returns:
        LayerStack: water FluidLayer + sediment layers + half-space.
    """
    water = FluidLayer(
        alpha=survey.water_alpha,
        rho=survey.water_rho,
        thickness=survey.water_depth,
    )
    layers: list[FluidLayer | IsotropicLayer] = [water, *sediment_layers, half_space]
    return LayerStack(layers=layers)


# ── Main pipeline ────────────────────────────────────────────────────────


def compute_shot_gather(
    stack: LayerStack,
    survey: SurveyConfig,
    gather_config: GatherConfig,
    use_gpu: bool = False,
) -> ShotGatherResult:
    """Compute a synthetic shot gather via Kennett reflectivity.

    Pipeline:
        1. Build frequency/slowness grids
        2. Batched Kennett reflectivity
        3. Free surface reverberations (optional)
        4. Source/receiver ghosts
        5. Bessel summation (slowness -> offset)
        6. Source spectrum
        7. IFFT with damping compensation

    Args:
        stack: Layer stack (water + sediments + half-space).
        survey: Survey geometry.
        gather_config: Computation parameters.
        use_gpu: Use GPU for Bessel summation if available.

    Returns:
        ShotGatherResult with time-domain gather.
    """
    gc = gather_config

    # Damping
    gamma = gc.gamma if gc.gamma is not None else np.pi / gc.T

    # Frequency grid
    dw = 2.0 * np.pi / gc.T
    nwm = gc.nw - 1
    omega_real = np.arange(1, nwm + 1, dtype=np.float64) * dw
    omega_damped = omega_real + 1j * gamma

    # Time axis
    nt = 2 * gc.nw
    time = np.arange(nt, dtype=np.float64) * (gc.T / float(nt))

    # Slowness grid
    if gc.p_max is not None:
        p_max = gc.p_max
    else:
        # Auto: 1.2 / minimum velocity in stack
        alpha_min = min(lay.alpha for lay in stack.layers)
        p_max = 1.2 / alpha_min
    dp = p_max / gc.np_slow
    p_samples = np.arange(1, gc.np_slow + 1, dtype=np.float64) * dp

    logger.info(
        "Shot gather: %d offsets [%.0f-%.0f m], %d freqs, %d slowness, dp=%.2e s/m",
        len(survey.offsets),
        survey.offsets[0],
        survey.offsets[-1],
        nwm,
        gc.np_slow,
        dp,
    )

    from tqdm.auto import tqdm

    pipeline = tqdm(total=5, desc="Shot gather pipeline", leave=True)

    # Step 1: Batched Kennett reflectivity (below water)
    pipeline.set_postfix_str("Kennett reflectivity")
    sub_ocean_layers = stack.layers[1:]
    sub_ocean_stack = LayerStack(layers=sub_ocean_layers)
    RRd_PP = kennett_reflectivity_batch(sub_ocean_stack, p_samples, omega_damped)
    pipeline.update(1)

    # Step 2: Water column phase + free surface reverberations
    pipeline.set_postfix_str("Free surface reverberations")
    water_layer = stack.layers[0]
    s_water = _complex_slowness(water_layer.alpha, water_layer.Q_alpha)
    eta_water = np.array([_vertical_slowness(s_water, float(pv)) for pv in p_samples])
    eaea_water = np.exp(
        2j
        * eta_water[:, np.newaxis]
        * omega_damped[np.newaxis, :]
        * water_layer.thickness
    )

    if gc.free_surface:
        R = free_surface_reverberations(RRd_PP, eaea_water)
    else:
        R = eaea_water * RRd_PP
    pipeline.update(1)

    # Step 3: Source and receiver ghosts
    pipeline.set_postfix_str("Ghost operators")
    sg = source_ghost(omega_damped, p_samples, survey.source_depth, survey.water_alpha)
    rg = receiver_ghost(
        omega_damped,
        p_samples,
        survey.receiver_depth,
        survey.water_alpha,
        survey.receiver_type,
    )
    R = R * sg * rg
    pipeline.update(1)

    # Step 4: Bessel summation (slowness -> offset)
    pipeline.set_postfix_str("Bessel summation")
    bessel_func = bessel_summation_gpu if use_gpu else bessel_summation
    U = bessel_func(R, p_samples, omega_real, survey.offsets, dp)

    # Source spectrum
    S = ricker_source_spectrum(omega_real, gc.f_peak)
    U *= S[np.newaxis, :]
    pipeline.update(1)

    # Step 5: IFFT with damping compensation
    pipeline.set_postfix_str("IFFT")
    gather = np.zeros((len(survey.offsets), nt), dtype=np.float64)
    exp_decay = np.exp(-gamma * time)

    for ir in range(len(survey.offsets)):
        Uwk = np.zeros(nt, dtype=np.complex128)
        Uwk[1 : gc.nw] = U[ir, :]
        Uwk[gc.nw + 1 :] = np.conj(U[ir, ::-1])

        seismogram_c = np.fft.fft(Uwk)
        gather[ir, :] = np.real(seismogram_c) * exp_decay

    pipeline.update(1)
    pipeline.set_postfix_str("Done")
    pipeline.close()

    return ShotGatherResult(
        time=time,
        offsets=survey.offsets,
        gather=gather,
        reflectivity=RRd_PP,
        stack=stack,
    )


# ── YAML configuration ──────────────────────────────────────────────────


def load_survey_config(
    path: str | Path,
) -> tuple[LayerStack, SurveyConfig, GatherConfig]:
    """Load survey configuration from a YAML file.

    Args:
        path: Path to YAML config file.

    Returns:
        Tuple of (LayerStack, SurveyConfig, GatherConfig).
    """
    import yaml

    path = Path(path)
    if not path.exists():
        msg = f"Config file not found: {path}"
        raise FileNotFoundError(msg)

    with path.open() as f:
        cfg = yaml.safe_load(f)

    # Survey geometry
    sc = cfg["survey"]
    offsets_cfg = sc["offsets"]
    offsets = np.arange(
        offsets_cfg["min"],
        offsets_cfg["max"] + offsets_cfg["spacing"] / 2,
        offsets_cfg["spacing"],
    )
    survey = SurveyConfig(
        source_depth=sc["source_depth"],
        receiver_depth=sc["receiver_depth"],
        receiver_type=sc["receiver_type"],
        offsets=offsets,
        water_depth=sc["water_depth"],
        water_alpha=sc.get("water_alpha", 1500.0),
        water_rho=sc.get("water_rho", 1025.0),
    )

    # Gather parameters
    gc = cfg.get("gather", {})
    src = gc.get("source", {})
    gather = GatherConfig(
        f_min=gc.get("f_min", 5.0),
        f_max=gc.get("f_max", 80.0),
        T=gc.get("T", 4.0),
        nw=gc.get("nw", 4096),
        np_slow=gc.get("np_slow", 8192),
        free_surface=gc.get("free_surface", True),
        source_type=src.get("type", "ricker"),
        f_peak=src.get("f_peak", 30.0),
    )

    # Earth model
    em = cfg["earth_model"]
    sediment_layers: list[IsotropicLayer] = []
    layer_defs = em["layers"]
    for i, ld in enumerate(layer_defs):
        h = (
            np.inf
            if ld.get("thickness") in (None, "inf", float("inf"))
            else ld["thickness"]
        )
        if i == len(layer_defs) - 1:
            h = np.inf
        sediment_layers.append(
            IsotropicLayer(
                alpha=ld["alpha"],
                beta=ld["beta"],
                rho=ld["rho"],
                thickness=h,
                Q_alpha=ld.get("Q_alpha", np.inf),
                Q_beta=ld.get("Q_beta", np.inf),
            )
        )

    # Build full stack (water + sediments)
    half_space = sediment_layers.pop()
    stack = build_survey_stack(survey, sediment_layers, half_space)

    return stack, survey, gather
