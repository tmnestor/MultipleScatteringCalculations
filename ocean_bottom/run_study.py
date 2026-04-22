#!/usr/bin/env python
"""Ocean-bottom reflection study with YAML config.

Simulates reflection through:
    water (acoustic) | heterogeneous sediment slab | elastic halfspace

Usage:
    # YAML config mode (primary)
    python ocean_bottom/run_study.py ocean_bottom/example_config.yml

    # With CLI overrides
    python ocean_bottom/run_study.py ocean_bottom/example_config.yml --p 0.2
    python ocean_bottom/run_study.py ocean_bottom/example_config.yml --free-surface
    python ocean_bottom/run_study.py ocean_bottom/example_config.yml --M 8 --save result.npz
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# Add parent to path for script execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cubic_scattering.kennett_layers import (
    _complex_slowness,
    _vertical_slowness,
    psv_fluid_solid,
)
from cubic_scattering.ocean_bottom import (
    compute_ocean_bottom_reflection,
    load_ocean_bottom_config,
    write_log,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ocean-bottom reflection with heterogeneous sediment slab."
    )

    # Positional: YAML config
    parser.add_argument("config", type=str, help="Path to YAML config file")

    # CLI overrides (seismic units: km/s, g/cm³, GPa, km, s/km)
    parser.add_argument("--M", type=int, default=None, help="Override horizontal grid size (M×M)")
    parser.add_argument("--a", type=float, default=None, help="Override cube half-width (km)")
    parser.add_argument("--p", type=float, default=None, help="Override horizontal slowness (s/km)")
    parser.add_argument(
        "--free-surface", action="store_true", default=None, help="Enable free-surface reverberations"
    )
    parser.add_argument(
        "--no-free-surface", action="store_true", default=None, help="Disable free-surface reverberations"
    )
    parser.add_argument(
        "--volume-averaged", action="store_true", default=None, help="Use volume-averaged propagator"
    )
    parser.add_argument("--f-peak", type=float, default=None, help="Override Ricker peak frequency (Hz)")
    parser.add_argument("--f-min", type=float, default=None, help="Override minimum frequency (Hz)")
    parser.add_argument("--f-max", type=float, default=None, help="Override maximum frequency (Hz)")
    parser.add_argument("--save", type=str, default=None, help="Save results to .npz file")
    parser.add_argument("--no-plot", action="store_true", help="Skip plotting")

    args = parser.parse_args()

    # ── Load YAML config ──────────────────────────────────────────────
    config = load_ocean_bottom_config(args.config)

    # ── Apply CLI overrides ───────────────────────────────────────────
    if args.M is not None or args.a is not None:
        from cubic_scattering.effective_contrasts import MaterialContrast
        from cubic_scattering.slab_scattering import SlabGeometry, uniform_slab_material

        M = args.M if args.M is not None else config.geometry.M
        a = args.a * 1e3 if args.a is not None else config.geometry.a  # km → m
        geom = SlabGeometry(M=M, N_z=config.geometry.N_z, a=a)
        contrast = MaterialContrast(
            Dlambda=float(config.material.Dlambda[0, 0, 0]),
            Dmu=float(config.material.Dmu[0, 0, 0]),
            Drho=float(config.material.Drho[0, 0, 0]),
        )
        config = type(config)(
            **{
                **config.__dict__,
                "geometry": geom,
                "material": uniform_slab_material(geom, config.sed_ref, contrast),
            }
        )
    if args.p is not None:
        config = type(config)(**{**config.__dict__, "p": args.p * 1e-3})  # s/km → s/m
    if args.free_surface:
        config = type(config)(**{**config.__dict__, "free_surface": True})
    if args.no_free_surface:
        config = type(config)(**{**config.__dict__, "free_surface": False})
    if args.f_peak is not None:
        config = type(config)(**{**config.__dict__, "f_peak": args.f_peak})
    if args.f_min is not None:
        config = type(config)(**{**config.__dict__, "f_min": args.f_min})
    if args.f_max is not None:
        config = type(config)(**{**config.__dict__, "f_max": args.f_max})

    # Read solver settings from YAML (not overrideable via CLI for simplicity)
    import yaml

    with Path(args.config).open() as f:
        raw_cfg = yaml.safe_load(f)
    solver = raw_cfg.get("solver", {})
    output_cfg = raw_cfg.get("output", {})
    volume_averaged = args.volume_averaged or solver.get("volume_averaged", False)
    n_orders = solver.get("n_orders", 2)
    gmres_tol = solver.get("gmres_tol", 1e-6)
    save_path = args.save or output_cfg.get("save")
    do_plot = not args.no_plot and output_cfg.get("plot", True)

    cfg = config

    # ── Print setup ──────────────────────────────────────────────────────
    print("=" * 60)
    print("Ocean-Bottom Reflection Study")
    print("=" * 60)
    print(
        f"Water:       α={cfg.water_alpha / 1e3:.3f} km/s, "
        f"ρ={cfg.water_rho / 1e3:.3f} g/cm³, "
        f"depth={cfg.water_depth / 1e3:.3f} km"
    )
    print(
        f"Sediment:    α={cfg.sed_ref.alpha / 1e3:.3f} km/s, "
        f"β={cfg.sed_ref.beta / 1e3:.3f} km/s, "
        f"ρ={cfg.sed_ref.rho / 1e3:.3f} g/cm³"
    )
    print(
        f"Halfspace:   α={cfg.hs_alpha / 1e3:.3f} km/s, "
        f"β={cfg.hs_beta / 1e3:.3f} km/s, "
        f"ρ={cfg.hs_rho / 1e3:.3f} g/cm³"
    )
    H = cfg.geometry.N_z * cfg.geometry.d
    print(
        f"Slab:        {cfg.geometry.M}×{cfg.geometry.M}×{cfg.geometry.N_z}, "
        f"a={cfg.geometry.a:.3f} m, H={H:.1f} m"
    )
    print(f"Slowness:    p={cfg.p * 1e3:.6f} s/km")
    if cfg.p > 0:
        angle_deg = np.degrees(np.arcsin(cfg.p * cfg.water_alpha))
        print(f"             θ_water ≈ {angle_deg:.2f}°")
    print(f"Wavelet:     Ricker f_peak={cfg.f_peak} Hz")
    print(
        f"Recording:   T={cfg.T} s, nw={cfg.nw}, f=[{cfg.f_min}-{cfg.f_max}] Hz"
    )
    print(f"Free surf:   {cfg.free_surface}")

    # ── Coupling info ─────────────────────────────────────────────────
    s_water = _complex_slowness(cfg.water_alpha, np.inf)
    eta_water = _vertical_slowness(s_water, cfg.p)
    s_sed_p = _complex_slowness(cfg.sed_ref.alpha, np.inf)
    eta_sed = _vertical_slowness(s_sed_p, cfg.p)
    s_sed_s = _complex_slowness(cfg.sed_ref.beta, np.inf)
    neta_sed = _vertical_slowness(s_sed_s, cfg.p)
    beta_sed = 1.0 / s_sed_s
    coeff_ws = psv_fluid_solid(
        cfg.p, eta_water, cfg.water_rho, eta_sed, neta_sed, cfg.sed_ref.rho, beta_sed
    )
    R_ws = coeff_ws.Rd[0, 0]
    Td_Tu = coeff_ws.Td[0, 0] * coeff_ws.Tu[0, 0]
    print(f"|R_ws|:       {abs(R_ws):.6f}")
    print(f"|T_d·T_u|:    {abs(Td_Tu):.6f}")
    print("-" * 60)

    # ── Compute ──────────────────────────────────────────────────────────
    result = compute_ocean_bottom_reflection(
        cfg,
        volume_averaged=volume_averaged,
        n_orders=n_orders,
        gmres_tol=gmres_tol,
        progress=True,
    )

    # ── Summary ──────────────────────────────────────────────────────────
    active = np.abs(result.R_slab) > 0
    n_active = int(np.sum(active))
    avg_iters = np.mean(result.n_gmres_iters) if result.n_gmres_iters else 0

    print("-" * 60)
    print(f"Elapsed:     {result.elapsed_seconds:.1f} s")
    print(f"Active freq: {n_active}")
    print(f"Avg GMRES:   {avg_iters:.1f} iterations")
    print(f"Peak |R_bg|: {np.max(np.abs(result.R_bg)):.6f}")
    print(f"Peak |R_slab|: {np.max(np.abs(result.R_slab)):.6f}")
    print(f"Peak |R_total|: {np.max(np.abs(result.R_total)):.6f}")
    print(f"Peak trace (hom):   {np.max(np.abs(result.trace_homogeneous)):.6e}")
    print(f"Peak trace (total): {np.max(np.abs(result.trace_total)):.6e}")

    # ── Log ──────────────────────────────────────────────────────────────
    log_path = Path(args.config).stem + ".log"
    write_log(result, log_path)
    print(f"Log:         {log_path}")

    # ── Save ─────────────────────────────────────────────────────────────
    if save_path:
        out = Path(save_path)
        np.savez(
            out,
            time=result.time,
            trace_total=result.trace_total,
            trace_homogeneous=result.trace_homogeneous,
            R_bg=result.R_bg,
            R_slab=result.R_slab,
            R_total=result.R_total,
            omega_real=result.omega_real,
            p=cfg.p,
        )
        print(f"Saved to {out}")

    # ── Plot (if matplotlib available) ───────────────────────────────────
    if do_plot:
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 1, figsize=(10, 8))

            # Time domain
            ax = axes[0]
            ax.plot(
                result.time, result.trace_homogeneous, "b-", label="Homogeneous", alpha=0.7
            )
            ax.plot(
                result.time, result.trace_total, "r-", label="Total (hom + slab)", alpha=0.7
            )
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude")
            title = "Ocean-Bottom Reflection"
            if cfg.p > 0:
                title += f" (p={cfg.p:.4e} s/m)"
            else:
                title += " (normal incidence)"
            ax.set_title(title)
            ax.legend()
            ax.set_xlim(0, min(cfg.T, 0.5))

            # Frequency domain
            ax = axes[1]
            freq_hz = result.omega_real / (2.0 * np.pi)
            ax.semilogy(freq_hz, np.abs(result.R_bg), "b-", label="|R_bg|", alpha=0.7)
            ax.semilogy(
                freq_hz, np.abs(result.R_slab), "g-", label="|R_slab|", alpha=0.7
            )
            ax.semilogy(
                freq_hz, np.abs(result.R_total), "r-", label="|R_total|", alpha=0.7
            )
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("|R|")
            ax.set_title("Reflection Coefficients")
            ax.legend()
            ax.set_xlim(cfg.f_min, cfg.f_max)

            plt.tight_layout()
            plt.savefig("ocean_bottom_study.png", dpi=150)
            print("Plot saved to ocean_bottom_study.png")
            plt.show()
        except ImportError:
            print("(matplotlib not available — skipping plot)")


if __name__ == "__main__":
    main()
