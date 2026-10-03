#!/usr/bin/env python3
# ruff: noqa: E501 (reason: the TikZ template below keeps its LaTeX lines whole, so that it reads as TikZ)
"""Write fig_octree.tex: the octree of the paper, drawn as its two-dimensional slice (a quadtree).

The leaves are generated, not drawn by hand: a square of side 8 is cut into 4 x 4 cells and a cell is
split to half-width 0.25 where it straddles the graded part of a compact feature (radius 0.7 to 1.9),
and to half-width 0.5 near the feature; cells outside the body (radius 4) are dropped. The grey level of a leaf is its mean
contrast: a weak halo over the body plus the feature.

Run:  python make_fig_octree.py   (then lualatex, or pdflatex, fig_octree.tex)
"""

from pathlib import Path

import numpy as np

R_BODY, R_IN, R_OUT, HALO, H_MIN = 4.0, 0.7, 1.9, 0.12, 0.25


def s5(x):
    x = np.clip(x, 0.0, 1.0)
    return x**3 * (10 - 15 * x + 6 * x**2)


def profile(r):
    return HALO * s5((R_BODY - r) / R_BODY) + (1 - HALO) * s5((R_OUT - r) / (R_OUT - R_IN))


def corners_r(x, y, s):
    xs = np.linspace(x, x + s, 9)
    ys = np.linspace(y, y + s, 9)
    gx, gy = np.meshgrid(xs, ys)
    return np.hypot(gx, gy)


def leaves():
    out = []
    todo = [(x, y, 2.0) for x in np.arange(-4.0, 4.0, 2.0) for y in np.arange(-4.0, 4.0, 2.0)]
    while todo:
        x, y, s = todo.pop()
        r = corners_r(x, y, s)
        if r.min() >= R_BODY:
            continue  # outside the body: no contrast, no leaf
        straddles = r.min() < R_OUT and r.max() > R_IN  # the graded edge of the feature
        near = r.min() < R_OUT + 0.9  # the feature's neighbourhood: the middle size
        if (straddles and s > 2 * H_MIN * 1.0001) or (near and s > 4 * H_MIN * 1.0001):
            h = s / 2
            todo += [(x, y, h), (x + h, y, h), (x, y + h, h), (x + h, y + h, h)]
        else:
            out.append((x, y, s, float(profile(r).mean())))
    return out


def main() -> int:
    lv = leaves()
    cells = "\n".join(
        f"    \\filldraw[fill=black!{int(round(8 + 62 * c))}, draw=black!70, line width=0.25pt] "
        f"({x:.3f},{y:.3f}) rectangle ++({s:.3f},{s:.3f});"
        for x, y, s, c in lv
    )
    outline = "\n".join(
        f"    \\draw[black!35, line width=0.2pt] ({x:.3f},{y:.3f}) rectangle ++({s:.3f},{s:.3f});"
        for x, y, s, _ in lv
    )
    sizes = sorted({s for *_, s, _ in lv}, reverse=True)
    tex = TEMPLATE.replace("%CELLS%", cells).replace("%OUTLINE%", outline)
    Path(__file__).with_name("fig_octree.tex").write_text(tex)
    print(f"{len(lv)} leaves of sizes {sizes}")
    return 0


TEMPLATE = r"""% Standalone source of the figure fig:octree of the octree paper, written by make_fig_octree.py.
% Build: lualatex (or pdflatex) fig_octree.tex; the paper includes the PDF. Legible in grey scale.
\documentclass[border=3pt]{standalone}
\usepackage{amsmath,amssymb}
\usepackage{tikz}
\usetikzlibrary{arrows.meta,decorations.pathmorphing,calc}
\begin{document}
\begin{tikzpicture}[x=0.52cm, y=0.52cm, font=\small, >={Stealth[length=4pt]}]
  % ---------------------------------------------------------------- (a) the tree over the medium
  \begin{scope}
%CELLS%
    \draw[thick] (0,0) circle (4);
    \node at (-4.3,4.6) {(a)};
    \node[align=center, font=\footnotesize] at (0,-5.4) {leaves of three sizes;\\grey: the mean contrast of the leaf};
  \end{scope}
  % ---------------------------------------------------------------- (b) the first-order term, one pass
  \begin{scope}[shift={(13.5,0)}]
%OUTLINE%
    \draw[thick] (0,0) circle (4);
    % incident plane wave
    \foreach \y in {-3.2,-1.6,0,1.6,3.2} \draw[->, thick] (-7.2,\y) -- (-4.8,\y);
    \foreach \xx in {-6.8,-6.3,-5.8} \draw[black!60] (\xx,-3.6) -- (\xx,3.6);
    \node[font=\footnotesize, align=center] at (-6.3,4.3) {$\mathrm{e}^{\mathrm{i}\mathbf{k}_{\mathrm{in}}\cdot\mathbf{x}}$};
    % one leaf, highlighted, radiating to the observer
    \fill[black!45] (2,-2) rectangle ++(2,2);
    \draw[very thick] (2,-2) rectangle ++(2,2);
    \draw[->, very thick, dashed] (3.6,-0.4) -- (6.4,2.6) node[above, font=\footnotesize] {$\mathbf{k}_{\mathrm{out}}$};
    \node at (-4.3,4.6) {(b)};
    \node[align=center, font=\footnotesize] at (0,-5.4) {one pass over the leaves, no solve: each leaf\\projects the wave on its polynomials and radiates\\its exact Born term times $1+E_p(\mathbf{k}_{\mathrm{in}},\mathbf{k}_{\mathrm{out}},h)$};
  \end{scope}
  % ---------------------------------------------------------------- (c) blocks between unequal leaves
  \begin{scope}[shift={(25.5,0)}]
    \draw[very thick] (-3.5,-1) rectangle ++(2,2);
    \draw[dashed, black!60] (-2.5,-1) -- ++(0,2) (-3.5,0) -- ++(2,0);
    \foreach \dx/\dy in {-3/-0.5,-2/-0.5,-3/0.5,-2/0.5} \fill (\dx,\dy) circle (1.6pt);
    \node[font=\footnotesize] at (-2.5,-1.6) {large leaf $L$};
    \draw[very thick] (2.5,-0.5) rectangle ++(1,1);
    \node[font=\footnotesize] at (3,-1.1) {leaf $S$};
    \foreach \dx/\dy in {-3/-0.5,-2/-0.5,-3/0.5,-2/0.5} \draw[->, black!70] (\dx,\dy) to[bend left=8] (2.5,0);
    \node[font=\footnotesize, align=center] at (0,3.4)
      {$\mathbf{K}(S,L)=\sum_{d\,\subset\,L}\mathbf{K}(S,d)\,\mathbf{R}_d$};
    \node[font=\footnotesize, align=center] at (0,-5.4)
      {$L$ re-expanded on its descendants $d$\\at the size of $S$:\\only blocks between equal cells};
    \node at (-4.3,4.6) {(c)};
  \end{scope}
\end{tikzpicture}
\end{document}
"""

if __name__ == "__main__":
    raise SystemExit(main())
