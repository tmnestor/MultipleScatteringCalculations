"""Compare global vs trace normalization to confirm aliasing diagnosis.

Generates two gather plots:
1. gather_global_norm.png — global normalization (new default)
2. gather_trace_norm.png — per-trace normalization (old behaviour)

The horizontal aliasing bands should vanish in the global-norm plot,
confirming they were integration noise amplified by per-trace normalization
at offsets where the hyperbolic moveout exits the display window.
"""

from __future__ import annotations

import logging
import sys
import time as time_module

import numpy as np

sys.path.insert(0, ".")

from Kennett_Reflectivity.kennett_gather import compute_gather, plot_gather
from Kennett_Reflectivity.layer_model import LayerModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

model = LayerModel.from_arrays(
    alpha=[1.5, 1.6, 3.0, 5.0, 2.2],
    beta=[0.0, 0.3, 1.5, 3.0, 1.1],
    rho=[1.0, 2.0, 3.0, 3.0, 1.8],
    thickness=[2.0, 1.0, 1.0, 1.0, np.inf],
    Q_alpha=[20000, 100, 100, 100, 100],
    Q_beta=[1e10, 100, 100, 100, 100],
)

offsets = np.arange(0.5, 50.5, 0.5)

print("Computing gather (CPU, free surface)...")
t0 = time_module.perf_counter()
time_arr, offsets_out, gather = compute_gather(
    model,
    offsets,
    T=64.0,
    nw=2048,
    np_slow=4096,
    p_max=1.2,
    free_surface=True,
)
t1 = time_module.perf_counter()
print(f"Done in {t1 - t0:.1f}s")

# Plot 1: Global normalization (new default)
plot_gather(
    time_arr,
    offsets_out,
    gather,
    output="gather_global_norm.png",
    title=(
        "Kennett Reflectivity — Global Normalization\n"
        "5-layer ocean crust + free surface"
    ),
    t_max=30.0,
    norm="global",
    noise_floor_db=-60.0,
)
print("Saved gather_global_norm.png")

# Plot 2: Per-trace normalization (old behaviour — shows aliasing)
plot_gather(
    time_arr,
    offsets_out,
    gather,
    output="gather_trace_norm.png",
    title=(
        "Kennett Reflectivity — Per-Trace Normalization\n"
        "5-layer ocean crust + free surface (shows noise amplification)"
    ),
    t_max=30.0,
    norm="trace",
)
print("Saved gather_trace_norm.png")

# Diagnostic: show which traces are below noise floor
global_peak = np.max(np.abs(gather))
for ir in range(len(offsets_out)):
    tmask = time_arr <= 30.0
    trace_peak = np.max(np.abs(gather[ir, tmask]))
    ratio_db = 20.0 * np.log10(trace_peak / global_peak) if trace_peak > 0 else -np.inf
    if ratio_db < -40.0:
        print(
            f"  Trace {ir:3d} (r={offsets_out[ir]:5.1f} km): "
            f"peak = {trace_peak:.2e}, {ratio_db:.1f} dB below global peak"
        )
