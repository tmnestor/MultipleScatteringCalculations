"""YAML configuration for scattering solvers.

Loads, validates, and provides typed access to solver configuration.
Dispatches to GPU or CPU solver based on device config.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import yaml

from .effective_contrasts import MaterialContrast, ReferenceMedium
from .slab_scattering import (
    SlabGeometry,
    SlabResult,
    random_slab_material,
    uniform_slab_material,
)
from .sphere_scattering import SphereDecompositionResult
from .torch_gmres import get_device, select_dtype

__all__ = [
    "DeviceConfig",
    "ProblemConfig",
    "ScatteringConfig",
    "SolverConfig",
    "load_config",
    "run_from_config",
    "validate_config",
]


@dataclass
class DeviceConfig:
    """Device and dtype configuration.

    Args:
        backend: Device backend — "auto", "mps", "cuda", or "cpu".
        dtype: Complex dtype — "auto", "complex64", or "complex128".
        memory_limit_gb: Warn if estimated memory exceeds this (GB).
    """

    backend: str = "auto"
    dtype: str = "auto"
    memory_limit_gb: float = 8.0

    def resolve_device(self) -> torch.device:
        """Resolve the target PyTorch device.

        Returns:
            torch.device based on backend setting.

        Raises:
            ValueError: If requested backend is unavailable.
        """
        if self.backend == "auto":
            return get_device()
        if self.backend == "mps":
            if not torch.backends.mps.is_available():
                msg = (
                    "MPS backend requested but not available. "
                    "Requires Apple Silicon with macOS 12.3+."
                )
                raise ValueError(msg)
            return torch.device("mps")
        if self.backend == "cuda":
            if not torch.cuda.is_available():
                msg = "CUDA backend requested but no CUDA GPU found."
                raise ValueError(msg)
            return torch.device("cuda")
        if self.backend == "cpu":
            return torch.device("cpu")
        msg = f"Invalid backend '{self.backend}'. Valid options: auto, mps, cuda, cpu."
        raise ValueError(msg)

    def resolve_dtype(self, device: torch.device) -> torch.dtype:
        """Resolve the complex dtype for the given device.

        Args:
            device: Target device (used for auto selection).

        Returns:
            torch.complex64 or torch.complex128.

        Raises:
            ValueError: If dtype string is invalid.
        """
        if self.dtype == "auto":
            return select_dtype(device)
        if self.dtype == "complex64":
            return torch.complex64
        if self.dtype == "complex128":
            return torch.complex128
        msg = (
            f"Invalid dtype '{self.dtype}'. Valid options: auto, complex64, complex128."
        )
        raise ValueError(msg)


@dataclass
class SolverConfig:
    """GMRES solver parameters.

    Args:
        gmres_tol: Relative residual tolerance.
        max_iter: Maximum GMRES iterations.
        initial_guess: Initial guess strategy — "born" or "zero".
    """

    gmres_tol: float = 1e-6
    max_iter: int = 500
    initial_guess: str = "born"


@dataclass
class ProblemConfig:
    """Problem definition.

    Args:
        type: Problem type — "slab" or "sphere".
        geometry: Geometry parameters dict (M, N_z, a).
        reference: Reference medium dict (alpha, beta, rho).
        contrast: Material contrast dict (Dlambda, Dmu, Drho).
        frequency: Frequency dict (omega).
        incident: Incident wave dict (k_hat, wave_type).
        sphere: Sphere-specific params (radius, n_sub).
        slab: Slab-specific params (material_type, phi, seed).
    """

    type: str
    geometry: dict = field(default_factory=dict)
    reference: dict = field(default_factory=dict)
    contrast: dict = field(default_factory=dict)
    frequency: dict = field(default_factory=dict)
    incident: dict = field(default_factory=dict)
    sphere: dict | None = None
    slab: dict | None = None

    def to_reference_medium(self) -> ReferenceMedium:
        """Create ReferenceMedium from config."""
        return ReferenceMedium(
            alpha=float(self.reference["alpha"]),
            beta=float(self.reference["beta"]),
            rho=float(self.reference["rho"]),
        )

    def to_material_contrast(self) -> MaterialContrast:
        """Create MaterialContrast from config."""
        return MaterialContrast(
            Dlambda=float(self.contrast["Dlambda"]),
            Dmu=float(self.contrast["Dmu"]),
            Drho=float(self.contrast["Drho"]),
        )

    def to_slab_geometry(self) -> SlabGeometry:
        """Create SlabGeometry from config."""
        return SlabGeometry(
            M=int(self.geometry["M"]),
            N_z=int(self.geometry["N_z"]),
            a=float(self.geometry["a"]),
        )


@dataclass
class ScatteringConfig:
    """Top-level scattering configuration.

    Args:
        device: Device and dtype settings.
        solver: GMRES solver parameters.
        problem: Problem definition.
    """

    device: DeviceConfig
    solver: SolverConfig
    problem: ProblemConfig


def load_config(path: Path) -> ScatteringConfig:
    """Load and parse YAML configuration file.

    Args:
        path: Path to YAML config file.

    Returns:
        Parsed ScatteringConfig.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        ValueError: If required sections are missing.
    """
    path = Path(path)
    if not path.exists():
        msg = (
            f"Config file not found: {path}\n"
            f"Expected a YAML file with device, solver, and problem sections.\n"
            f"See configs/example_slab.yml for a template."
        )
        raise FileNotFoundError(msg)

    with path.open() as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        msg = f"Config file must contain a YAML mapping, got {type(raw).__name__}"
        raise ValueError(msg)

    # Parse device section
    dev_raw = raw.get("device", {})
    device_config = DeviceConfig(
        backend=dev_raw.get("backend", "auto"),
        dtype=dev_raw.get("dtype", "auto"),
        memory_limit_gb=float(dev_raw.get("memory_limit_gb", 8.0)),
    )

    # Parse solver section
    sol_raw = raw.get("solver", {})
    solver_config = SolverConfig(
        gmres_tol=float(sol_raw.get("gmres_tol", 1e-6)),
        max_iter=int(sol_raw.get("max_iter", 500)),
        initial_guess=sol_raw.get("initial_guess", "born"),
    )

    # Parse problem section (required)
    prob_raw = raw.get("problem")
    if prob_raw is None:
        msg = (
            "Config missing required 'problem' section.\n"
            "Must contain: type, geometry, reference, contrast, frequency, incident."
        )
        raise ValueError(msg)

    problem_config = ProblemConfig(
        type=prob_raw.get("type", "slab"),
        geometry=prob_raw.get("geometry", {}),
        reference=prob_raw.get("reference", {}),
        contrast=prob_raw.get("contrast", {}),
        frequency=prob_raw.get("frequency", {}),
        incident=prob_raw.get("incident", {}),
        sphere=prob_raw.get("sphere"),
        slab=prob_raw.get("slab"),
    )

    config = ScatteringConfig(
        device=device_config,
        solver=solver_config,
        problem=problem_config,
    )

    validate_config(config)
    return config


def validate_config(config: ScatteringConfig) -> None:
    """Validate configuration with actionable error messages.

    Args:
        config: Configuration to validate.

    Raises:
        ValueError: If any field is invalid, with specific remediation guidance.
    """
    # Validate problem type
    if config.problem.type not in ("slab", "sphere"):
        msg = (
            f"Invalid problem type '{config.problem.type}'. Must be 'slab' or 'sphere'."
        )
        raise ValueError(msg)

    # Validate reference medium
    ref = config.problem.reference
    for key in ("alpha", "beta", "rho"):
        if key not in ref:
            msg = (
                f"Missing 'problem.reference.{key}'. "
                f"Required fields: alpha (P-velocity m/s), "
                f"beta (S-velocity m/s), rho (density kg/m³)."
            )
            raise ValueError(msg)
        if float(ref[key]) <= 0:
            msg = f"problem.reference.{key} must be positive, got {ref[key]}"
            raise ValueError(msg)

    # Validate contrast
    contrast = config.problem.contrast
    for key in ("Dlambda", "Dmu", "Drho"):
        if key not in contrast:
            msg = (
                f"Missing 'problem.contrast.{key}'. "
                f"Required fields: Dlambda (Pa), Dmu (Pa), Drho (kg/m³)."
            )
            raise ValueError(msg)

    # Validate frequency
    if "omega" not in config.problem.frequency:
        msg = "Missing 'problem.frequency.omega'. Must be a positive float (rad/s)."
        raise ValueError(msg)
    if float(config.problem.frequency["omega"]) <= 0:
        msg = (
            f"problem.frequency.omega must be positive, "
            f"got {config.problem.frequency['omega']}"
        )
        raise ValueError(msg)

    # Validate geometry
    geom = config.problem.geometry
    for key in ("M", "N_z", "a"):
        if key not in geom:
            msg = (
                f"Missing 'problem.geometry.{key}'. "
                f"Required: M (int, horizontal grid), "
                f"N_z (int, vertical layers), a (float, half-width m)."
            )
            raise ValueError(msg)

    # Validate incident wave
    inc = config.problem.incident
    if "wave_type" not in inc:
        msg = "Missing 'problem.incident.wave_type'. Must be 'P' or 'S'."
        raise ValueError(msg)
    if inc["wave_type"] not in ("P", "S"):
        msg = f"Invalid wave_type '{inc['wave_type']}'. Must be 'P' or 'S'."
        raise ValueError(msg)
    if "k_hat" not in inc:
        msg = (
            "Missing 'problem.incident.k_hat'. "
            "Must be a 3-element list [z, x, y] direction."
        )
        raise ValueError(msg)

    # Validate solver
    if config.solver.initial_guess not in ("born", "zero"):
        msg = (
            f"Invalid initial_guess '{config.solver.initial_guess}'. "
            f"Must be 'born' or 'zero'."
        )
        raise ValueError(msg)

    # Validate device
    valid_backends = ("auto", "mps", "cuda", "cpu")
    if config.device.backend not in valid_backends:
        msg = (
            f"Invalid backend '{config.device.backend}'. "
            f"Valid options: {', '.join(valid_backends)}."
        )
        raise ValueError(msg)

    valid_dtypes = ("auto", "complex64", "complex128")
    if config.device.dtype not in valid_dtypes:
        msg = (
            f"Invalid dtype '{config.device.dtype}'. "
            f"Valid options: {', '.join(valid_dtypes)}."
        )
        raise ValueError(msg)

    # Sphere-specific validation
    if config.problem.type == "sphere":
        sp = config.problem.sphere
        if sp is None:
            msg = (
                "Problem type 'sphere' requires a 'sphere' section "
                "with 'radius' and 'n_sub'."
            )
            raise ValueError(msg)
        if "radius" not in sp or "n_sub" not in sp:
            msg = (
                "Sphere config requires 'radius' (m) and 'n_sub' (int). "
                f"Got keys: {list(sp.keys())}"
            )
            raise ValueError(msg)


def run_from_config(
    config: ScatteringConfig,
) -> SlabResult | SphereDecompositionResult:
    """Dispatch to GPU or CPU solver based on config.

    Args:
        config: Validated scattering configuration.

    Returns:
        SlabResult or SphereDecompositionResult depending on problem type.
    """
    device = config.device.resolve_device()
    dtype = config.device.resolve_dtype(device)
    use_gpu = device.type != "cpu"

    prob = config.problem
    omega = float(prob.frequency["omega"])
    k_hat = np.array(prob.incident["k_hat"], dtype=float)
    wave_type = prob.incident["wave_type"]
    ref = prob.to_reference_medium()
    contrast = prob.to_material_contrast()

    if prob.type == "slab":
        geometry = prob.to_slab_geometry()

        # Build material
        slab_cfg = prob.slab or {}
        material_type = slab_cfg.get("material_type", "uniform")
        if material_type == "random":
            phi = float(slab_cfg.get("phi", 0.1))
            seed = slab_cfg.get("seed")
            if seed is not None:
                seed = int(seed)
            material = random_slab_material(geometry, ref, contrast, phi, seed)
        else:
            material = uniform_slab_material(geometry, ref, contrast)

        if use_gpu:
            from .slab_scattering_gpu import compute_slab_scattering_gpu

            return compute_slab_scattering_gpu(
                geometry=geometry,
                material=material,
                omega=omega,
                k_hat=k_hat,
                wave_type=wave_type,
                gmres_tol=config.solver.gmres_tol,
                max_iter=config.solver.max_iter,
                initial_guess=config.solver.initial_guess,
                device=device,
                dtype=dtype,
            )
        else:
            from .slab_scattering import compute_slab_scattering

            return compute_slab_scattering(
                geometry=geometry,
                material=material,
                omega=omega,
                k_hat=k_hat,
                wave_type=wave_type,
                gmres_tol=config.solver.gmres_tol,
                max_iter=config.solver.max_iter,
            )

    # sphere
    sp = prob.sphere or {}
    radius = float(sp["radius"])
    n_sub = int(sp["n_sub"])

    if use_gpu:
        from .sphere_scattering_fft_gpu import compute_sphere_foldy_lax_fft_gpu

        return compute_sphere_foldy_lax_fft_gpu(
            omega=omega,
            radius=radius,
            ref=ref,
            contrast=contrast,
            n_sub=n_sub,
            k_hat=k_hat,
            wave_type=wave_type,
            gmres_tol=config.solver.gmres_tol,
            gmres_maxiter=config.solver.max_iter,
            initial_guess=config.solver.initial_guess,
            device=device,
            dtype=dtype,
        )
    else:
        from .sphere_scattering_fft import compute_sphere_foldy_lax_fft

        return compute_sphere_foldy_lax_fft(
            omega=omega,
            radius=radius,
            ref=ref,
            contrast=contrast,
            n_sub=n_sub,
            k_hat=k_hat,
            wave_type=wave_type,
            gmres_tol=config.solver.gmres_tol,
            gmres_maxiter=config.solver.max_iter,
        )
