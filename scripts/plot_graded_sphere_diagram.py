#!/usr/bin/env python3
"""The graded sphere of Paper 2, in greyscale for print (``LatexPDFs/ExactCouplingIntegrals/figures/fig_sphere.pdf``).

Left: the section through the centre in the plane of incidence, in the seismic convention (z down): the
contrast shaded by the profile, the homogeneous core, the cells of the 8-across model that overlap the sphere,
the plane P wave coming down, and the receiver plane of the tau-p section at depth zero, 1.5 a above the
centre. Right: the profile, the fraction of the full contrast against the radius.

Run:  python scripts/plot_graded_sphere_diagram.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, Rectangle

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from plot_taup_graded_sphere import INK, INK2, MUTED

A, CORE, ZC, N = 1.0, 0.1, 1.5, 8  # radius, core radius, depth of the centre, cells across (in radii)
RAMP = LinearSegmentedColormap.from_list("contrast", ["#ffffff", "#cfcfcf", "#8c8c8c", "#4a4a4a"])


def profile(r):
    """Fraction of the full contrast: 1 in the core, the smoothstep 10x^3 - 15x^4 + 6x^5 across the shell."""
    x = np.clip((A - r) / (A - CORE), 0.0, 1.0)
    return 10 * x**3 - 15 * x**4 + 6 * x**5


def main() -> int:
    plt.rcParams.update({"font.size": 7.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.linewidth": 0.6, "pdf.fonttype": 42})  # fmt: skip
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.1, 3.3), gridspec_kw={"width_ratios": [1.15, 1]},
                                 constrained_layout=True)  # fmt: skip
    # the contrast, shaded inside the sphere
    g = np.linspace(-1.05, 1.05, 600)
    X, Z = np.meshgrid(g, g)
    R = np.hypot(X, Z)
    img = np.where(R <= A, profile(R), np.nan)
    ax.imshow(img, extent=(g[0], g[-1], ZC + g[-1], ZC + g[0]), cmap=RAMP, vmin=0, vmax=1, interpolation="bilinear",
              rasterized=True, zorder=1)  # fmt: skip
    ax.add_patch(Circle((0, ZC), A, fill=False, ec=INK, lw=0.8, zorder=3))
    ax.add_patch(Circle((0, ZC), CORE, fill=False, ec=INK, lw=0.6, ls=(0, (2, 1.5)), zorder=3))
    # the cells of the section y in [0, h], kept where they overlap the sphere
    w = 2 * A / N
    for i in range(N):
        for k in range(N):
            x0, z0 = -A + i * w, -A + k * w
            nx = min(max(0.0, x0), x0 + w) if x0 > 0 or x0 + w < 0 else 0.0
            nz = min(max(0.0, z0), z0 + w) if z0 > 0 or z0 + w < 0 else 0.0
            near = np.hypot(min(abs(x0), abs(x0 + w)) if x0 * (x0 + w) > 0 else 0.0,
                            min(abs(z0), abs(z0 + w)) if z0 * (z0 + w) > 0 else 0.0)  # fmt: skip
            del nx, nz
            if near < A:
                ax.add_patch(Rectangle((x0, ZC + z0), w, w, fill=False, ec=INK2, lw=0.35, zorder=2))
    # receivers at z = 0, and the incident plane P wave
    xs = np.linspace(-1.4, 1.4, 15)
    ax.plot([-1.5, 1.5], [0, 0], color=INK, lw=0.8, zorder=3)
    ax.plot(xs, np.zeros_like(xs), "v", ms=3.2, color=INK, zorder=4, clip_on=False)
    for zf in (-0.62, -0.47, -0.32):
        ax.plot([-1.5, 1.5], [zf, zf], color=INK2, lw=0.7)
    for xa in (-1.2, 0.0, 1.2):
        ax.annotate("", xy=(xa, -0.12), xytext=(xa, -0.72),
                    arrowprops={"arrowstyle": "-|>", "color": INK2, "lw": 0.8, "mutation_scale": 7})  # fmt: skip
    ax.text(1.58, -0.47, "plane P wave", color=INK2, va="center", ha="left", fontsize=7)
    ax.text(1.58, 0.0, "receivers, $z=0$", color=INK, va="center", ha="left", fontsize=7)
    # dimensions
    ax.annotate("", xy=(np.cos(np.pi / 4) * A, ZC - np.sin(np.pi / 4) * A), xytext=(0, ZC),
                arrowprops={"arrowstyle": "-|>", "color": INK, "lw": 0.6, "mutation_scale": 6})  # fmt: skip
    ax.text(0.42, ZC - 0.28, "$a$", color=INK, fontsize=8)
    ax.annotate("core, $a/10$", xy=(0.06, ZC + 0.08), xytext=(0.55, ZC + 0.75), fontsize=7, color=INK,
                arrowprops={"arrowstyle": "-", "color": INK, "lw": 0.5})  # fmt: skip
    ax.annotate("", xy=(-1.45, ZC), xytext=(-1.45, 0.0),
                arrowprops={"arrowstyle": "<|-|>", "color": INK2, "lw": 0.6, "mutation_scale": 6})  # fmt: skip
    ax.text(-1.5, 0.75, "$1.5a$", color=INK2, ha="right", va="center", fontsize=7)
    ax.plot([-1.5, 0], [ZC, ZC], color=MUTED, lw=0.4, ls=(0, (2, 2)))
    ax.set_xlim(-1.7, 2.6)
    ax.set_ylim(2.65, -0.85)
    ax.set_aspect("equal")
    ax.set_xlabel("$x/a$")
    ax.set_ylabel("depth $z/a$")
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([0, 0.5, 1, 1.5, 2, 2.5])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    # the profile
    r = np.linspace(0, 1.15, 600)
    bx.fill_between(r, profile(r), color="#d9d9d9", lw=0)
    bx.plot(r, profile(r), color=INK, lw=1.2)
    bx.axvline(CORE, color=MUTED, lw=0.6, ls=(0, (2, 1.5)))
    bx.axvline(A, color=MUTED, lw=0.6, ls=(0, (2, 1.5)))
    bx.text(CORE + 0.02, 0.5, "core", color=INK2, rotation=90, va="center", fontsize=7)
    bx.text(0.55, 0.62, "smoothstep shell\n$10x^3-15x^4+6x^5$", color=INK, fontsize=7, ha="left")
    bx.text(0.03, 1.07, "$\\Delta\\lambda=2$ GPa, $\\Delta\\mu=1$ GPa, $\\Delta\\rho=100$ kg m$^{-3}$", color=INK,
            fontsize=7)  # fmt: skip
    bx.set_xlim(0, 1.15)
    bx.set_ylim(0, 1.18)
    bx.set_xlabel("radius $r/a$")
    bx.set_ylabel("fraction of the full contrast")
    for side in ("top", "right"):
        bx.spines[side].set_visible(False)
    out = ROOT / "LatexPDFs" / "ExactCouplingIntegrals" / "figures" / "fig_sphere.pdf"
    fig.savefig(out, dpi=300)
    fig.savefig(ROOT / "scratch" / "taup" / "fig_sphere.png", dpi=150)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
