#!/usr/bin/env python3
"""The tau-p figure of Paper 2 (``LatexPDFs/ExactCouplingIntegrals/figures/fig_taup.pdf``).

Two rows (vertical and horizontal displacement on the receiver plane) by four columns (the exact sphere, the
Legendre cells, their difference magnified, and traces at a few slownesses), from the sections written by
``taup_graded_sphere.py``. Variable density on a diverging scale (blue <-> red about a neutral grey), one
symmetric scale per row, time running down.

Run:  python scripts/plot_taup_paper.py [tag=p1_n8]
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from plot_taup_graded_sphere import ALPHA, BETA, DIVERGING, INK, INK2, MUTED  # noqa: E402


def load(tag, comp):
    d = np.load(ROOT / "scratch" / "taup" / f"taup_{tag}.npz")
    tau, slow = d["tau"], d["slow"]
    period = tau[1] - tau[0] + tau[-1]
    t = np.where(tau > period / 2, tau - period, tau)
    order = np.argsort(t)
    return t[order], slow, d[f"exact_{comp}"][order], d[f"cells_{comp}"][order]


def main() -> int:
    tag = sys.argv[1] if len(sys.argv) > 1 else "p1_n8"
    plt.rcParams.update({"font.size": 7.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.titlecolor": INK, "axes.linewidth": 0.6,
                         "xtick.major.width": 0.6, "ytick.major.width": 0.6, "pdf.fonttype": 42})  # fmt: skip
    fig, axes = plt.subplots(2, 4, figsize=(7.1, 5.4), constrained_layout=True, sharey=True,
                             gridspec_kw={"width_ratios": [1, 1, 1, 0.9]})  # fmt: skip
    window = (-25.0, 25.0)  # intercept time (ms) shown
    for row, comp in enumerate(("uz", "ux")):
        tau, slow, ex, vx = load(tag, comp)
        keep = (tau * 1e3 >= window[0]) & (tau * 1e3 <= window[1])
        t_ms, ex, vx = tau[keep] * 1e3, ex[keep], vx[keep]
        peak = np.abs(ex).max()
        ex, vx = ex / peak, vx / peak
        diff = vx - ex
        mag = 10 ** np.ceil(np.log10(0.5 / np.abs(diff).max()))
        p_skm = slow * 1e3
        extent = (p_skm[0], p_skm[-1], t_ms[-1], t_ms[0])
        name = {"uz": "$u_z$", "ux": "$u_x$"}[comp]
        for col, (data, title) in enumerate(((ex, f"{name}: exact sphere"), (vx, f"{name}: Legendre cells"),
                                              (diff * mag, f"{name}: cells − exact, ×{mag:.0e}".replace("e+0", "0^").replace("10^4", "10⁴")))):  # fmt: skip
            ax = axes[row, col]
            im = ax.imshow(data, aspect="auto", extent=extent, cmap=DIVERGING, vmin=-1, vmax=1,
                           interpolation="bilinear", rasterized=True)  # fmt: skip
            ax.set_title(title, fontsize=8)
            for pb, lab in ((1e3 / ALPHA, "$1/\\alpha$"), (1e3 / BETA, "$1/\\beta$")):
                ax.axvline(pb, color=MUTED, lw=0.6, ls=(0, (3, 2)))
                if row == 0:
                    ax.text(pb, window[0], f" {lab}", color=INK2, va="top", ha="left", fontsize=7)
            if row == 1:
                ax.set_xlabel("slowness $p$ (s/km)")
        axes[row, 0].set_ylabel("intercept time $\\tau$ (ms)")
        ax = axes[row, 3]
        picks = np.unique(np.linspace(2, len(slow) - 3, 7).astype(int))
        gap = p_skm[picks[1]] - p_skm[picks[0]]
        for j in picks:
            ax.plot(p_skm[j] + 0.45 * gap * ex[:, j], t_ms, color=INK, lw=0.8)
            ax.plot(p_skm[j] + 0.45 * gap * vx[:, j], t_ms, color="#2a78d6", lw=0.9, ls=(0, (2.5, 1.5)))
        ax.set_title(f"{name}: traces", fontsize=8)
        ax.set_xlim(0, p_skm[-1] + 0.02)
        ax.set_ylim(window[1], window[0])
        if row == 1:
            ax.set_xlabel("slowness $p$ (s/km)")
            ax.plot([], [], color=INK, lw=0.8, label="exact")
            ax.plot([], [], color="#2a78d6", lw=0.9, ls=(0, (2.5, 1.5)), label="cells")
            ax.legend(loc="lower right", frameon=False, fontsize=7, labelcolor=INK2)
    cb = fig.colorbar(im, ax=axes[:, :3], shrink=0.6, pad=0.01, aspect=30)
    cb.set_label("displacement / peak of the exact section", color=INK2)
    cb.outline.set_edgecolor(MUTED)
    cb.outline.set_linewidth(0.6)
    out = ROOT / "LatexPDFs" / "ExactCouplingIntegrals" / "figures" / "fig_taup.pdf"
    fig.savefig(out, dpi=300)
    fig.savefig(ROOT / "scratch" / "taup" / f"fig_taup_{tag}.png", dpi=150)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
