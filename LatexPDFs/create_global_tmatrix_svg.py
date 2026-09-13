#!/usr/bin/env python3
"""
create_global_tmatrix_svg.py
============================
Reproducible generator for the Global T-Matrix schematic figure.

Produces an SVG depicting cubic heterogeneities superimposed on a
layered elastic medium, annotated with:
  - Explosion source (filled star) at the free surface
  - Incident wavefield u₀ radiating downward
  - Intra-site T₀ (highlighted cube with self-scattering)
  - Inter-site G₀ arrows (same-layer, 1-interface, multi-interface)
  - Right-panel annotation boxes (T₀, G₀, Global T)
  - Bottom key-insight box

IMPORTANT DESIGN CONSTRAINT:
  Layer boundaries occur ONLY between cube rows (in the inter-row gaps),
  never through a cube.  This is enforced by deriving all layer geometry
  from `rows_per_layer` and the cube grid step.

Usage:
    python create_global_tmatrix_svg.py            # writes SVG to default path
    python create_global_tmatrix_svg.py -o fig.svg  # custom output path

Author: T.M. Nestor (2026)
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass, field
from pathlib import Path

# ──────────────────────────────────────────────────────────────────
# Configuration dataclass — edit these to change the figure
# ──────────────────────────────────────────────────────────────────


@dataclass
class FigureConfig:
    """All tuneable parameters for the schematic."""

    # Canvas
    width: int = 1100
    height: int = 720
    bg_colour: str = "#faf8f0"
    font_family: str = "'Georgia', 'Times New Roman', serif"

    # Main diagram region (translated from top-left)
    diagram_x: int = 50
    diagram_y: int = 75
    diagram_w: int = 520  # width of the layered block

    # Cube grid
    cube_size: int = 55
    cube_gap: int = 5  # gap between cubes (border to border)
    cube_margin: int = 20  # left margin of first cube column
    cube_top: int = 10  # top offset of first cube row
    n_cols: int = 8  # cubes per row

    # How many cube rows belong to each layer (sum = total rows).
    # Layer boundaries are placed in the gap between the last row of
    # one layer and the first row of the next — NEVER through a cube.
    rows_per_layer: list = field(default_factory=lambda: [2, 2, 1, 2, 1])

    # Layer colour definitions: (grad_top, grad_bot, stroke, label_colour)
    layer_colours: list = field(
        default_factory=lambda: [
            ("#e8d4a0", "#d4bc82", "#8a7a5a", "#5a4a2a"),  # Layer 1
            ("#b8c4a0", "#9aad82", "#6a8a5a", "#3a5a2a"),  # Layer 2
            ("#a0b4c4", "#829aad", "#5a6a8a", "#2a3a5a"),  # Layer 3
            ("#c4a0b4", "#ad829a", "#8a5a6a", "#5a2a3a"),  # Layer 4
            ("#8a9ab0", "#6e7e94", "#5a6a7a", "#3a4a5a"),  # Layer 5
        ]
    )

    # Cube fill palette: base RGB per layer index
    cube_rgb: list = field(
        default_factory=lambda: [
            (180, 120, 80),  # Layer 1 (warm brown)
            (100, 160, 90),  # Layer 2 (green)
            (80, 120, 160),  # Layer 3 (blue)
            (160, 80, 110),  # Layer 4 (rose)
            (80, 100, 130),  # Layer 5 (slate)
        ]
    )
    cube_strokes: list = field(
        default_factory=lambda: ["#6a4a2a", "#4a6a3a", "#3a4a6a", "#6a3a4a", "#3a4a5a"]
    )
    cube_alphas: list = field(
        default_factory=lambda: [0.35, 0.30, 0.40, 0.30, 0.45, 0.30, 0.35, 0.30]
    )

    # Highlighted T₀ cube: (row_idx, col_idx) in the grid (0-based)
    t0_row: int = 3
    t0_col: int = 2

    # Source position (relative to diagram group)
    source_x: int = 80
    source_y_offset: int = -18  # above the free surface line

    # Right panel
    panel_x: int = 810
    panel_y: int = 85

    # Bottom annotation
    bottom_y: int = 590
    bottom_h: int = 118

    # Colours
    col_t0: str = "#d4a017"
    col_t0_dark: str = "#8b6914"
    col_g0_same: str = "#2980b9"
    col_g0_cross: str = "#c0392b"
    col_text: str = "#1a1a2e"
    col_wave: str = "#2c3e50"
    col_g0_cross_dark: str = "#922b21"
    col_global: str = "#27774a"
    col_global_dark: str = "#1a5032"


# ──────────────────────────────────────────────────────────────────
# Geometry helpers
# ──────────────────────────────────────────────────────────────────


def compute_layer_geometry(cfg: FigureConfig):
    """
    Derive layer rectangles and interface positions from the cube grid.

    Returns:
        layer_rects: list of (y_top, height) for each layer
        interface_ys: list of y-positions for internal interfaces
        total_h: total height of the layered block
        row_tops: list of y-coordinate for top of each cube row
        row_layer: list of layer index for each cube row
    """
    step = cfg.cube_size + cfg.cube_gap
    n_rows = sum(cfg.rows_per_layer)
    row_tops = [cfg.cube_top + r * step for r in range(n_rows)]

    # Assign each row to a layer
    row_layer = []
    for li, n in enumerate(cfg.rows_per_layer):
        row_layer.extend([li] * n)

    # Layer boundaries sit at the midpoint of the gap between the
    # last row of one layer and the first row of the next.
    layer_rects = []
    interface_ys = []
    y_start = 0
    row_cursor = 0
    for li, n in enumerate(cfg.rows_per_layer):
        last_row_bottom = row_tops[row_cursor + n - 1] + cfg.cube_size
        if li < len(cfg.rows_per_layer) - 1:
            next_row_top = row_tops[row_cursor + n]
            boundary = (last_row_bottom + next_row_top) // 2
            interface_ys.append(boundary)
            layer_rects.append((y_start, boundary - y_start))
            y_start = boundary
        else:
            # Last layer: extend to bottom of last cube + small margin
            total_h = last_row_bottom + cfg.cube_top
            layer_rects.append((y_start, total_h - y_start))
        row_cursor += n

    total_h = sum(h for _, h in layer_rects)
    return layer_rects, interface_ys, total_h, row_tops, row_layer


def star_points(
    cx: float, cy: float, r_outer: float = 13, r_inner: float = 6, n_pts: int = 5
) -> str:
    """Generate SVG polygon points for an n-pointed star."""
    pts = []
    for i in range(2 * n_pts):
        angle = math.pi / 2 + i * math.pi / n_pts
        r = r_outer if i % 2 == 0 else r_inner
        x = cx + r * math.cos(angle)
        y = cy - r * math.sin(angle)
        pts.append(f"{x:.1f},{y:.1f}")
    return " ".join(pts)


# ──────────────────────────────────────────────────────────────────
# Main SVG assembly
# ──────────────────────────────────────────────────────────────────


def build_svg(cfg: FigureConfig) -> str:
    """Assemble the complete SVG string."""
    lines: list[str] = []

    def L(s: str) -> None:  # noqa: E741
        lines.append(s)

    n_layers = len(cfg.rows_per_layer)
    layer_rects, interface_ys, total_h, row_tops, row_layer = compute_layer_geometry(
        cfg
    )
    step = cfg.cube_size + cfg.cube_gap
    half = cfg.cube_size // 2
    n_rows = sum(cfg.rows_per_layer)

    # ── Header ──
    L(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {cfg.width} {cfg.height}" '
        f'font-family="{cfg.font_family}">'
    )

    # ── Defs ──
    L("  <defs>")
    for i, (g_top, g_bot, _, _) in enumerate(cfg.layer_colours, start=1):
        L(f'    <linearGradient id="layer{i}" x1="0" y1="0" x2="0" y2="1">')
        L(f'      <stop offset="0%" stop-color="{g_top}"/>')
        L(f'      <stop offset="100%" stop-color="{g_bot}"/>')
        L("    </linearGradient>")
    L(
        '    <marker id="arrowG0" markerWidth="10" markerHeight="7" '
        'refX="9" refY="3.5" orient="auto" markerUnits="strokeWidth">'
    )
    L(f'      <polygon points="0 0, 10 3.5, 0 7" fill="{cfg.col_g0_cross}"/>')
    L("    </marker>")
    L(
        '    <marker id="arrowG0blue" markerWidth="10" markerHeight="7" '
        'refX="9" refY="3.5" orient="auto" markerUnits="strokeWidth">'
    )
    L(f'      <polygon points="0 0, 10 3.5, 0 7" fill="{cfg.col_g0_same}"/>')
    L("    </marker>")
    L(
        '    <marker id="arrowWave" markerWidth="8" markerHeight="6" '
        'refX="7" refY="3" orient="auto" markerUnits="strokeWidth">'
    )
    L(f'      <polygon points="0 0, 8 3, 0 6" fill="{cfg.col_wave}"/>')
    L("    </marker>")
    L('    <filter id="glowT0" x="-20%" y="-20%" width="140%" height="140%">')
    L('      <feGaussianBlur stdDeviation="3" result="coloredBlur"/>')
    L(
        '      <feMerge><feMergeNode in="coloredBlur"/>'
        '<feMergeNode in="SourceGraphic"/></feMerge>'
    )
    L("    </filter>")
    L("  </defs>")
    L("")

    # ── Background & title ──
    L(f'  <rect width="{cfg.width}" height="{cfg.height}" fill="{cfg.bg_colour}"/>')
    cx = cfg.width // 2
    L(
        f'  <text x="{cx}" y="32" text-anchor="middle" font-size="18" '
        f'font-weight="bold" fill="{cfg.col_text}">'
    )
    L("    Cubic Heterogeneities Superimposed on a Layered Medium")
    L("  </text>")
    L(
        f'  <text x="{cx}" y="54" text-anchor="middle" font-size="13" '
        f'fill="#555" font-style="italic">'
    )
    L("    Intra-site scattering (T&#x2080;) vs. inter-site propagation (G&#x2080;)")
    L("  </text>")
    L("")

    # ── Diagram group ──
    L(f'  <g transform="translate({cfg.diagram_x}, {cfg.diagram_y})">')

    # Layer rectangles (derived from cube grid — boundaries between rows)
    for i, (y_top, h) in enumerate(layer_rects):
        stroke = cfg.layer_colours[i][2]
        L(
            f'    <rect x="0" y="{y_top}" width="{cfg.diagram_w}" '
            f'height="{h}" fill="url(#layer{i + 1})" stroke="{stroke}" '
            f'stroke-width="0.5"/>'
        )

    # Layer labels
    sub = [
        "\u2081",
        "\u2082",
        "\u2083",
        "\u2084",
        "\u2085",
        "\u2086",
        "\u2087",
        "\u2088",
        "\u2089",
    ]
    lx = cfg.diagram_w + 15
    for i, (y_top, h) in enumerate(layer_rects):
        my = y_top + h // 2 + 4
        lcol = cfg.layer_colours[i][3]
        si = sub[i] if i < len(sub) else str(i + 1)
        L(
            f'    <text x="{lx}" y="{my}" font-size="11" fill="{lcol}" '
            f'font-style="italic">'
            f"Layer {i + 1}: \u03b1{si}, \u03b2{si}, \u03c1{si}</text>"
        )

    # Interface dashed lines
    for iy in interface_ys:
        L(
            f'    <line x1="0" y1="{iy}" x2="{cfg.diagram_w}" y2="{iy}" '
            f'stroke="#444" stroke-width="1.5" stroke-dasharray="4,3"/>'
        )
    L("")

    # ── Cube grid ──
    cube_centres: list[list[tuple[int, int]]] = []  # [row][col] = (cx, cy)
    for ri in range(n_rows):
        y = row_tops[ri]
        li = row_layer[ri]
        r, g, b = cfg.cube_rgb[li]
        stroke = cfg.cube_strokes[li]
        row_c = []
        for ci in range(cfg.n_cols):
            x = cfg.cube_margin + ci * step
            alpha = cfg.cube_alphas[ci % len(cfg.cube_alphas)]
            dr = (ci * 17 + ri * 7) % 30 - 15
            L(
                f'    <rect x="{x}" y="{y}" width="{cfg.cube_size}" '
                f'height="{cfg.cube_size}" '
                f'fill="rgba({r + dr},{g + dr},{b + dr},{alpha})" '
                f'stroke="{stroke}" stroke-width="1"/>'
            )
            row_c.append((x + half, y + half))
        cube_centres.append(row_c)
    L("")

    # Centre dots (every other column)
    L(f'    <g fill="{cfg.col_text}" opacity="0.6">')
    for ri, row in enumerate(cube_centres):
        for ci, (ccx, ccy) in enumerate(row):
            if ci % 2 == 0:
                L(f'      <circle cx="{ccx}" cy="{ccy}" r="2.5"/>')
    L("    </g>")
    L("")

    # ── Highlighted T₀ cube ──
    t0_cx, t0_cy = cube_centres[cfg.t0_row][cfg.t0_col]
    t0_x = t0_cx - half
    t0_y = t0_cy - half
    L(
        f'    <rect x="{t0_x}" y="{t0_y}" width="{cfg.cube_size}" '
        f'height="{cfg.cube_size}" fill="rgba(230,180,50,0.5)" '
        f'stroke="{cfg.col_t0}" stroke-width="3" filter="url(#glowT0)"/>'
    )
    L(f'    <circle cx="{t0_cx}" cy="{t0_cy}" r="4" fill="{cfg.col_t0}"/>')
    L(
        f'    <path d="M {t0_cx - 12} {t0_cy - 12} Q {t0_cx - 17} {t0_cy}, '
        f'{t0_cx - 9} {t0_cy + 10}" fill="none" stroke="{cfg.col_t0_dark}" '
        f'stroke-width="1.5" stroke-dasharray="3,2"/>'
    )
    L(
        f'    <path d="M {t0_cx + 11} {t0_cy + 10} Q {t0_cx + 16} {t0_cy}, '
        f'{t0_cx + 9} {t0_cy - 12}" fill="none" stroke="{cfg.col_t0_dark}" '
        f'stroke-width="1.5" stroke-dasharray="3,2"/>'
    )
    L(
        f'    <text x="{t0_cx}" y="{t0_y + cfg.cube_size + 13}" '
        f'text-anchor="middle" font-size="10" font-weight="bold" '
        f'fill="{cfg.col_t0_dark}">T&#x2080;</text>'
    )
    L("")

    # ── G₀ arrows ──
    # Same-layer (horizontal)
    tgt_cx = cube_centres[cfg.t0_row][cfg.t0_col + 2][0]
    L(
        f'    <line x1="{t0_cx + 5}" y1="{t0_cy}" '
        f'x2="{tgt_cx}" y2="{t0_cy}" '
        f'stroke="{cfg.col_g0_same}" stroke-width="2.2" '
        f'marker-end="url(#arrowG0blue)" opacity="0.85"/>'
    )
    mx = (t0_cx + tgt_cx) // 2
    L(
        f'    <text x="{mx}" y="{t0_cy - 7}" text-anchor="middle" '
        f'font-size="9" fill="{cfg.col_g0_same}" font-weight="bold">'
        f"G&#x2080;</text>"
    )
    L(
        f'    <text x="{mx}" y="{t0_cy + 15}" text-anchor="middle" '
        f'font-size="7.5" fill="{cfg.col_g0_same}" font-style="italic">'
        f"(same layer)</text>"
    )
    L("")

    # 1-interface (curved upward to row t0_row-2)
    t1_cx, t1_cy = cube_centres[cfg.t0_row - 2][cfg.t0_col]
    qx = t0_cx + 33
    qy = (t0_cy + t1_cy) // 2
    L(
        f'    <path d="M {t0_cx + 5} {t0_cy} Q {qx} {qy}, {t1_cx} {t1_cy + 5}" '
        f'fill="none" stroke="{cfg.col_g0_cross}" stroke-width="2.2" '
        f'marker-end="url(#arrowG0)" opacity="0.85"/>'
    )
    # Interface crossing marker (nearest interface between T₀ layer and target)
    t0_layer = row_layer[cfg.t0_row]
    if t0_layer > 0:
        cross_y = interface_ys[t0_layer - 1]
        L(
            f'    <circle cx="{(t0_cx + qx) // 2}" cy="{cross_y}" r="5" '
            f'fill="none" stroke="{cfg.col_g0_cross}" stroke-width="1" '
            f'stroke-dasharray="2,2" opacity="0.6"/>'
        )
    L(
        f'    <text x="{qx + 10}" y="{qy - 5}" font-size="9" '
        f'fill="{cfg.col_g0_cross}" font-weight="bold">G&#x2080;</text>'
    )
    L(
        f'    <text x="{qx + 10}" y="{qy + 7}" font-size="7.5" '
        f'fill="{cfg.col_g0_cross}" font-style="italic">(1 interface)</text>'
    )
    L("")

    # Multi-interface (dashed, to top-right corner)
    tt_cx, tt_cy = cube_centres[0][-2]
    L(
        f'    <path d="M {t0_cx + 5} {t0_cy} Q 320 180, 380 80, '
        f'{tt_cx} {tt_cy}" fill="none" stroke="{cfg.col_g0_cross}" '
        f'stroke-width="2.5" marker-end="url(#arrowG0)" opacity="0.9" '
        f'stroke-dasharray="6,3"/>'
    )
    for iy in interface_ys[:2]:
        icx = 300 if iy == interface_ys[0] else 370
        L(
            f'    <circle cx="{icx}" cy="{iy}" r="5" fill="none" '
            f'stroke="{cfg.col_g0_cross}" stroke-width="1.2" '
            f'stroke-dasharray="2,2" opacity="0.5"/>'
        )
    L(
        f'    <text x="370" y="125" font-size="9" '
        f'fill="{cfg.col_g0_cross}" font-weight="bold">G&#x2080;</text>'
    )
    L(
        f'    <text x="370" y="137" font-size="7.5" '
        f'fill="{cfg.col_g0_cross}" font-style="italic">(multiple</text>'
    )
    L(
        f'    <text x="370" y="148" font-size="7.5" '
        f'fill="{cfg.col_g0_cross}" font-style="italic">interfaces)</text>'
    )
    L("")

    # ── Free surface ──
    L(
        f'    <line x1="-10" y1="0" x2="{cfg.diagram_w + 10}" y2="0" '
        f'stroke="{cfg.col_text}" stroke-width="2.5"/>'
    )
    L(
        f'    <text x="{cfg.diagram_w + 15}" y="-4" font-size="10" '
        f'fill="{cfg.col_text}" font-weight="bold">Free surface</text>'
    )
    L("")

    # ── Explosion source (filled star) ──
    pts = star_points(cfg.source_x, cfg.source_y_offset)
    L(
        f'    <polygon points="{pts}" '
        f'fill="{cfg.col_text}" stroke="{cfg.col_g0_cross}" stroke-width="1"/>'
    )
    L(
        f'    <text x="{cfg.source_x}" y="{cfg.source_y_offset - 18}" '
        f'text-anchor="middle" font-size="9" font-weight="bold" '
        f'fill="{cfg.col_text}">Source</text>'
    )
    L("")

    # ── Incident wavefield u₀ ──
    sx = cfg.source_x
    L(
        f'    <path d="M {sx - 30} 10 Q {sx} 30, {sx + 30} 10" fill="none" '
        f'stroke="{cfg.col_wave}" stroke-width="1.2" opacity="0.5"/>'
    )
    L(
        f'    <path d="M {sx - 45} 30 Q {sx} 60, {sx + 45} 30" fill="none" '
        f'stroke="{cfg.col_wave}" stroke-width="1.0" opacity="0.4"/>'
    )
    L(
        f'    <path d="M {sx - 60} 55 Q {sx} 95, {sx + 60} 55" fill="none" '
        f'stroke="{cfg.col_wave}" stroke-width="0.8" opacity="0.3"/>'
    )
    L(
        f'    <line x1="{sx}" y1="0" x2="{sx}" y2="60" '
        f'stroke="{cfg.col_wave}" stroke-width="1.8" '
        f'marker-end="url(#arrowWave)" opacity="0.7"/>'
    )
    L(
        f'    <text x="{sx + 12}" y="40" font-size="10" '
        f'fill="{cfg.col_wave}" font-weight="bold">u&#x2080;</text>'
    )
    L(
        f'    <text x="{sx + 12}" y="52" font-size="8" '
        f'fill="{cfg.col_wave}" font-style="italic">incident</text>'
    )
    L("")

    L("  </g>")  # end diagram
    L("")

    # ── Right panel ──
    L(f'  <g transform="translate({cfg.panel_x}, {cfg.panel_y})">')
    bw = 255

    # Box 1: T₀
    L(
        f'    <rect x="0" y="0" width="{bw}" height="130" rx="8" '
        f'fill="#fdf6e3" stroke="{cfg.col_t0}" stroke-width="2"/>'
    )
    L(
        f'    <text x="{bw // 2}" y="22" text-anchor="middle" font-size="13" '
        f'font-weight="bold" fill="{cfg.col_t0_dark}">Intra-Site: T&#x2080;</text>'
    )
    L(
        f'    <line x1="15" y1="30" x2="{bw - 15}" y2="30" '
        f'stroke="{cfg.col_t0}" stroke-width="0.5"/>'
    )
    for dy, txt in [
        (48, "Self-consistent scattering within"),
        (62, "a single cube. Computed"),
        (76, "analytically (Part II)."),
        (96, "Absorbs all multiple scattering"),
        (110, "inside the cell \u2014 exact to O(ka)\u2077."),
    ]:
        L(f'    <text x="15" y="{dy}" font-size="10.5" fill="#333">{txt}</text>')
    L(
        f'    <text x="{bw // 2}" y="126" text-anchor="middle" font-size="10" '
        f'fill="{cfg.col_t0_dark}" font-style="italic">'
        f"\u2714 Solved completely</text>"
    )

    # Box 2: G₀
    g0_y = 148
    L(
        f'    <rect x="0" y="{g0_y}" width="{bw}" height="210" rx="8" '
        f'fill="#f0e6e6" stroke="{cfg.col_g0_cross}" stroke-width="2"/>'
    )
    L(
        f'    <text x="{bw // 2}" y="{g0_y + 22}" text-anchor="middle" '
        f'font-size="13" font-weight="bold" fill="{cfg.col_g0_cross_dark}">'
        f"Inter-Site: G&#x2080;</text>"
    )
    L(
        f'    <line x1="15" y1="{g0_y + 30}" x2="{bw - 15}" y2="{g0_y + 30}" '
        f'stroke="{cfg.col_g0_cross}" stroke-width="0.5"/>'
    )
    g0_items = [
        (48, 15, "#333", "", "10.5", "Propagation between cube centres"),
        (62, 15, "#333", "", "10.5", "through the layered medium."),
        (82, 15, cfg.col_g0_cross_dark, ' font-weight="bold"', "10.5", "Challenges:"),
        (98, 15, "#333", "", "10", "\u2022 Reflections at every interface"),
        (113, 15, "#333", "", "10", "\u2022 P \u2194 SV mode conversions"),
        (128, 15, "#333", "", "10", "\u2022 Head waves, turning waves"),
        (143, 15, "#333", "", "10", "\u2022 Evanescent coupling near"),
        (156, 25, "#333", "", "10", "critical angles"),
        (174, 15, "#333", "", "10.5", "Computed via Kennett (1983)"),
        (188, 15, "#333", "", "10.5", "reflectivity method."),
    ]
    for dy, ix, col, bld, fs, txt in g0_items:
        L(
            f'    <text x="{ix}" y="{g0_y + dy}" font-size="{fs}" '
            f'fill="{col}"{bld}>{txt}</text>'
        )
    L(
        f'    <text x="{bw // 2}" y="{g0_y + 206}" text-anchor="middle" '
        f'font-size="10" fill="{cfg.col_g0_cross}" font-style="italic">'
        f"\u2718 Computationally expensive</text>"
    )

    # Box 3: Global T
    gt_y = 376
    L(
        f'    <rect x="0" y="{gt_y}" width="{bw}" height="115" rx="8" '
        f'fill="#e8f0e8" stroke="{cfg.col_global}" stroke-width="2"/>'
    )
    L(
        f'    <text x="{bw // 2}" y="{gt_y + 22}" text-anchor="middle" '
        f'font-size="13" font-weight="bold" fill="{cfg.col_global_dark}">'
        f"Global: T = T&#x2080;(I \u2212 G&#x2080;T&#x2080;)\u207b\u00b9</text>"
    )
    L(
        f'    <line x1="15" y1="{gt_y + 30}" x2="{bw - 15}" y2="{gt_y + 30}" '
        f'stroke="{cfg.col_global}" stroke-width="0.5"/>'
    )
    for dy, txt in [
        (48, "Sums all multiple-scattering"),
        (62, "paths between cubes to all"),
        (76, "orders. Solved via Krylov"),
        (90, "methods (Nestor, 1996) and"),
        (104, "Fourier diagonalisation."),
    ]:
        L(f'    <text x="15" y="{gt_y + dy}" font-size="10.5" fill="#333">{txt}</text>')

    L("  </g>")
    L("")

    # ── Bottom annotation ──
    bw_full = cfg.width - 2 * cfg.diagram_x + 10
    L(f'  <g transform="translate({cfg.diagram_x}, {cfg.bottom_y})">')
    L(
        f'    <rect x="0" y="0" width="{bw_full}" height="{cfg.bottom_h}" '
        f'rx="6" fill="#f5f5f0" stroke="#888" stroke-width="0.8"/>'
    )
    L(
        f'    <text x="15" y="20" font-size="11" font-weight="bold" '
        f'fill="{cfg.col_text}">Key insight: cubes tile space</text>'
    )
    L(
        '    <text x="15" y="38" font-size="10" fill="#444">'
        "Every point in the medium belongs to exactly one cube. "
        "The T&#x2080; captures all physics inside each cell;</text>"
    )
    L(
        '    <text x="15" y="52" font-size="10" fill="#444">'
        "the G&#x2080; captures all physics between cells. "
        "Together, they give the complete wavefield.</text>"
    )
    L(
        f'    <line x1="15" y1="62" x2="{bw_full - 15}" y2="62" '
        f'stroke="#ccc" stroke-width="0.5"/>'
    )
    L(
        '    <text x="15" y="78" font-size="10" fill="#444">'
        "The G&#x2080; propagator is the bottleneck: in a layered medium "
        "it requires the Kennett reflectivity recursion</text>"
    )
    L(
        '    <text x="15" y="92" font-size="10" fill="#444">'
        "(wavenumber integration, reflection/transmission matrices at each "
        "interface, mode conversions) for every</text>"
    )
    L(
        '    <text x="15" y="106" font-size="10" fill="#444">'
        "pair of cube centres. Horizontal translational invariance "
        "allows 2D FFT acceleration.</text>"
    )
    L("  </g>")
    L("")
    L("</svg>")
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Global T-Matrix schematic (SVG)"
    )
    parser.add_argument(
        "-o",
        "--output",
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "global_tmatrix_schematic.svg"
        ),
        help="Output SVG path (default: same directory as script)",
    )
    args = parser.parse_args()

    cfg = FigureConfig()
    svg = build_svg(cfg)

    out = Path(args.output)
    out.write_text(svg, encoding="utf-8")
    print(f"Wrote {len(svg):,} bytes -> {out}")


if __name__ == "__main__":
    main()
