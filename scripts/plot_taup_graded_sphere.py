#!/usr/bin/env python3
"""Figure of the tau-p sections written by ``taup_graded_sphere.py``: exact, cells, difference, wiggles.

Variable density on a diverging scale (blue <-> red about a neutral grey: the sign of the displacement), one
symmetric scale for the exact and the cell sections, the difference magnified on the same scale, and a panel
of wiggle traces at a few slownesses (exact solid, cells dashed). Time runs down, as seismic sections do.

Run:  python scripts/plot_taup_graded_sphere.py [tag=p1_n8] [component=uz]
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ALPHA, BETA = 5000.0, 3000.0
INK, INK2, MUTED, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#fcfcfb"
# the reference diverging pair: blue <-> red about the neutral light grey
DIVERGING = LinearSegmentedColormap.from_list(
    "blue_red", ["#104281", "#2a78d6", "#9ec5f4", "#f0efec", "#f3a3a2", "#e34948", "#8f1d1c"]
)


def main() -> int:
    tag = sys.argv[1] if len(sys.argv) > 1 else "p1_n8"
    comp = sys.argv[2] if len(sys.argv) > 2 else "uz"
    d = np.load(ROOT / "scratch" / "taup" / f"taup_{tag}.npz")
    tau, slow = d["tau"], d["slow"]
    ex, vx = d[f"exact_{comp}"], d[f"cells_{comp}"]
    # the FFT period is circular: intercept times past half the period are negative times, wrapped
    period = tau[1] - tau[0] + tau[-1]
    order = np.argsort(np.where(tau > period / 2, tau - period, tau))
    tau = np.where(tau > period / 2, tau - period, tau)[order]
    ex, vx = ex[order], vx[order]
    peak = np.abs(ex).max()
    ex, vx = ex / peak, vx / peak
    # the time window: where the exact section carries energy, padded
    env = np.abs(ex).max(axis=1)
    live = np.where(env > 1e-3)[0]
    pad = max(10, (live[-1] - live[0]) // 6)
    i0, i1 = max(live[0] - pad, 0), min(live[-1] + pad, len(tau) - 1)
    t_ms = tau[i0 : i1 + 1] * 1e3
    ex, vx = ex[i0 : i1 + 1], vx[i0 : i1 + 1]
    diff = vx - ex
    mag = 10 ** np.ceil(np.log10(0.5 / max(np.abs(diff).max(), 1e-30)))  # magnify the difference to a visible size
    p_ms = slow * 1e3  # s/km
    extent = (p_ms[0], p_ms[-1], t_ms[-1], t_ms[0])

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.titlecolor": INK, "figure.facecolor": SURFACE,
                         "axes.facecolor": SURFACE})  # fmt: skip
    fig, axes = plt.subplots(1, 4, figsize=(13, 5.2), constrained_layout=True, sharey=True)
    name = {"uz": "vertical", "ux": "horizontal"}[comp]
    for ax, data, title in ((axes[0], ex, "exact sphere"), (axes[1], vx, "Legendre cells"),
                            (axes[2], diff * mag, f"cells − exact, × {mag:g}")):  # fmt: skip
        im = ax.imshow(data, aspect="auto", extent=extent, cmap=DIVERGING, vmin=-1, vmax=1, interpolation="bilinear")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("horizontal slowness p (s/km)")
        for pb, lab in ((1e3 / ALPHA, "1/α"), (1e3 / BETA, "1/β")):
            ax.axvline(pb, color=MUTED, lw=0.8, ls=(0, (4, 3)))
            ax.text(pb, t_ms[0], f" {lab}", color=INK2, va="top", ha="left", fontsize=8)
    axes[0].set_ylabel("intercept time τ (ms)")
    cb = fig.colorbar(im, ax=axes[:3], shrink=0.8, pad=0.01)
    cb.set_label("displacement / peak of the exact section", color=INK2)
    cb.outline.set_edgecolor(MUTED)
    # wiggles
    ax = axes[3]
    picks = np.unique(np.linspace(2, len(slow) - 3, 7).astype(int))
    gap = (p_ms[picks[1]] - p_ms[picks[0]]) if len(picks) > 1 else 1.0
    for j in picks:
        base = p_ms[j]
        ax.plot(base + 0.45 * gap * ex[:, j], t_ms, color=INK, lw=1.0)
        ax.plot(base + 0.45 * gap * vx[:, j], t_ms, color="#2a78d6", lw=1.2, ls=(0, (3, 2)))
    ax.plot([], [], color=INK, lw=1.0, label="exact")
    ax.plot([], [], color="#2a78d6", lw=1.2, ls=(0, (3, 2)), label="cells")
    ax.legend(loc="lower right", frameon=False, fontsize=8, labelcolor=INK2)
    ax.set_title("traces", fontsize=10)
    ax.set_xlabel("horizontal slowness p (s/km)")
    ax.set_ylim(t_ms[-1], t_ms[0])
    for a in axes:
        a.tick_params(length=3)
    fig.suptitle(f"Plane P wave on the graded sphere: τ-p section of the {name} displacement 1.5a above the "
                 f"centre (cells of degree {tag.split('_')[0][1:]}, {tag.split('_')[1][1:]} across)", color=INK, fontsize=11)  # fmt: skip
    out = ROOT / "scratch" / "taup" / f"taup_{tag}_{comp}.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}; difference max {np.abs(diff).max():.2e} of the exact peak")
    return 0


if __name__ == "__main__":
    sys.exit(main())
