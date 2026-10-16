"""Skip the PyTorch solvers' tests where torch is not installed (the package imports it lazily)."""

import importlib.util

collect_ignore = (
    []
    if importlib.util.find_spec("torch")
    else [
        "test_slab_scattering_gpu.py",
        "test_solver_config.py",
        "test_sphere_scattering_fft_gpu.py",
        "test_torch_gmres.py",
    ]
)
