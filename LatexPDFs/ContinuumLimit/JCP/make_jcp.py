#!/usr/bin/env python3
"""Build the Journal of Computational Physics version of the paper from the article source.

The article (``../ContinuumLimit.tex``) stays the master; this script regenerates ``ContinuumLimit_JCP.tex``
from it: Elsevier's elsarticle class (review mode: double spacing) with line numbers, the front matter in
Elsevier's form (title, author, affiliation, abstract, highlights, keywords), numbered references from the
shared ``../references.bib`` (style ``elsarticle-num-names``), and the closing statements Elsevier asks for.
The abstract is the journal's own: JCP limits it to 250 words and reads a method first.

With ``--submission`` it also assembles ``submission/``: the files to upload to Elsevier's Editorial
Manager, all at ONE folder level (the system rejects LaTeX submissions with subfolders), with the figures
as separate PDF files, and test-compiles them there with pdfLaTeX and BibTeX, as the submission system
does. Only the upload files are kept: the .tex, its .bbl, the .bib, the .bst and the figure PDFs.

Run from this directory:  python3 make_jcp.py [--submission]
"""

import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = (HERE.parent / "ContinuumLimit.tex").read_text()


MATH = re.compile(
    r"\$\$(.+?)\$\$|\$(.+?)\$|\\\[(.+?)\\\]"
    r"|\\begin\{(equation|align|gather|multline)\*?\}(.+?)\\end\{\4\*?\}",
    re.S,
)
# an italic differential: "\,dz", "\,d\mathbf y", "\,d^2x", or d/dz written as \frac{d}{dz}
ITALIC_DIFFERENTIAL = re.compile(r"\\[,;!]\s*d(?=[A-Za-z\\^])|\\frac\{d[^{}]*\}\{d[A-Za-z\\]")


def check_upright_differentials(text: str, name: str) -> None:
    """Fail unless every differential is the upright \\dd and every integral carries one.

    An italic d in this paper is the lattice pitch, so an italic differential changes the meaning.
    """
    src = re.sub(r"(?<!\\)%.*", "", text)
    problems = []
    for m in re.finditer(r"\\mathrm\{d\}", src):
        if not src[max(0, m.start() - 20) : m.start()].endswith("\\newcommand{\\dd}{"):
            problems.append((m.start(), "\\mathrm{d} written out", "use \\dd"))
    for m in ITALIC_DIFFERENTIAL.finditer(src):
        problems.append((m.start(), f"italic differential {m.group(0)!r}", "write \\dd in place of d"))
    # "\ddV" is one undefined control sequence, not \dd followed by V
    for m in re.finditer(r"\\dd[A-Za-z]+", src):
        problems.append((m.start(), f"{m.group(0)} is not \\dd", f"write \\dd {m.group(0)[3:]}"))
    for m in MATH.finditer(src):
        block = next(g for g in (m.group(1), m.group(2), m.group(3), m.group(5)) if g is not None)
        signs = len(re.findall(r"\\o?int(?![a-zA-Z])", block)) + 2 * block.count("\\iint")
        signs += 3 * block.count("\\iiint")
        dds = len(re.findall(r"\\dd(?![A-Za-z])", block))
        if signs > dds:
            problems.append(
                (
                    m.start(),
                    f"{signs} integral sign(s) but {dds} \\dd",
                    "end each integrand with \\,\\dd <variable>",
                )
            )
    if problems:
        lines = [
            f"  line {src.count(chr(10), 0, pos) + 1}: {what}; fix: {fix}"
            for pos, what, fix in sorted(problems)
        ]
        raise SystemExit(
            f"{name}: the differential of every integral and derivative must be upright.\n"
            + "\n".join(lines)
            + "\nExample: \\int_V G\\,\\dd V, \\int f(z')\\,\\dd z', \\frac{\\dd u}{\\dd z}"
            " (\\dd is \\mathrm{d}, defined in the Macros block).\n"
            f"Fix the lines above in {name}, then rerun make_jcp.py."
        )


check_upright_differentials(SRC, str(HERE.parent / "ContinuumLimit.tex"))


def between(text: str, start: str, end: str) -> str:
    a = text.index(start) + len(start)
    return text[a : text.index(end, a)]


body = SRC[SRC.index("\\section{Introduction}") : SRC.index("\\vfill\n\\noindent\\rule")].rstrip()
macros = between(SRC, "% --- Macros ---\n", "\\title").strip()

FIGURES = sorted((HERE.parent / "figures").glob("fig_*.pdf"))
# the master's figure paths, relative to its own folder; the local JCP build sits one level down
body = body.replace("{figures/fig_", "{../figures/fig_")
# elsarticle's \paragraph adds its own full stop: drop the source's, or headings end in ".."
body = re.sub(r"\\paragraph\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\.\}", r"\\paragraph{\1}", body)
# notebook names: \path breaks at underscores and dots
body = re.sub(r"\\nb\{([^{}]*)\}", lambda m: r"\path{" + m.group(1).replace(r"\_", "_") + "}", body)

ABSTRACT = r"""
To model seismic waves efficiently in a realistically complicated medium, volume-integral
(coupled-dipole) methods divide it into cubic voxels coupled through the background Green's tensor.
Each voxel must represent the medium within it and the wavefield within it. We ask which basis each
needs and what error a given choice leaves, which cannot be measured where no exact solution exists.
We answer on a layer of voxels: because cubes tile space, $n$ planes of voxels are the layer itself,
with no shape error, and its exact solution isolates the discretisation error. The source-averaged
coupling equals the continuum's exactly; the cube's single-site $T$-matrix follows from distributional
moments of the Green's tensor. With the wavefield expanded to Legendre degree $p$ and the medium's
contrast to degree $r$ in each voxel, the scheme converges at order $\min(2p+2,2r+2)$, with leading
error $c_p(kd)^{2p+2}$ in closed form: $c_0=1/12$, $c_1=1/720$, $c_2=1/100800$; for $p\le1$ also for
SV and SH waves and at oblique incidence. The multiple-scattering error equals
the relative projection error of the medium at degree $\min(p,r)$, exactly in the long-wave limit:
the two bases must be matched, and the error is known from the medium before anything is solved.
Where the medium is piecewise constant on the voxels it vanishes: in a randomly stratified layer one
first-moment voxel per layer reaches $10^{-6}$. A smoothly graded sphere converges at second order
against its exact solution. Every result is verified by an independent implementation or an exact
solution."""
words = len(re.sub(r"\$[^$]*\$", "x", ABSTRACT).split())
assert words <= 250, f"the JCP abstract is {words} words; the limit is 250"

HIGHLIGHTS = [
    "A voxel needs two bases: one for the medium and one for the wavefield",
    "On a layer of cubes the discretisation error is isolated, with no shape error",
    "The order is min(2p+2, 2r+2) for field degree p and medium degree r",
    "The multiple-scattering error is the medium's relative projection error",
    "The error is known in advance, from the medium and the grid alone",
]
assert all(len(h) <= 85 for h in HIGHLIGHTS), "a highlight exceeds 85 characters"

PREAMBLE = r"""\documentclass[preprint,review,12pt]{elsarticle}
% Journal of Computational Physics version, generated by make_jcp.py from the article ContinuumLimit.tex
\usepackage{amsmath,amssymb,bm}
\usepackage{mathtools}
\usepackage{booktabs}
\usepackage{array}
\usepackage{graphicx}
\usepackage{placeins}
\usepackage{url}
\usepackage{lineno}
\providecommand{\texorpdfstring}[2]{#1}
\setcounter{secnumdepth}{3}
\journal{Journal of Computational Physics}

"""
FRONT = (
    r"""
\begin{document}
\begin{frontmatter}
\title{The continuum limit of a space-filling layer of cubes: a tiling identity, an exact lattice kernel
and the closed-form error of the discrete layer}
\author{T. M. Nestor}
\affiliation{organization={Independent researcher}, country={Australia}}

\begin{abstract}
"""
    + ABSTRACT
    + r"""
\end{abstract}

\begin{highlights}
"""
    + "\n".join(r"\item " + h for h in HIGHLIGHTS)
    + r"""
\end{highlights}

\begin{keyword}
Elastic wave scattering \sep Volume integral equation \sep Foldy--Lax multiple scattering \sep
Discrete dipole approximation \sep Galerkin method \sep Convergence analysis
\end{keyword}
\end{frontmatter}

\linenumbers

"""
)
BACK = r"""

\section*{Acknowledgments}
This work began thirty years ago in the author's doctoral research at the Research School of Earth
Sciences of the Australian National University, Canberra \citep{Nestor1996}.

\section*{Declaration of competing interest}
The author declares no known competing financial interests or personal relationships that could have
appeared to influence the work reported in this paper.

\section*{Declaration of generative AI and AI-assisted technologies in the manuscript preparation process}
During the preparation of this work, the author used Claude (Anthropic) in order to check the existing
Mathematica notebooks, Python and Fortran code, to build tests to ensure their correctness, and to
write the TikZ code of the diagrams. After using this tool, the author reviewed and edited the content
as needed and takes full responsibility for the content of the published article.

\section*{Data availability}
The Mathematica notebooks and Python scripts that derive and check every result, and the package they
test, are available at \url{https://github.com/tmnestor/MultipleScatteringCalculations}.

\bibliographystyle{elsarticle-num-names}
\bibliography{../references}

\end{document}
"""

out = PREAMBLE + macros + "\n" + FRONT + body + BACK
(HERE / "ContinuumLimit_JCP.tex").write_text(out)
print(f"wrote ContinuumLimit_JCP.tex ({len(out.splitlines())} lines); abstract {words} words")


def assemble_submission() -> None:
    """Write submission/ flat, test-compile it with pdfLaTeX + BibTeX, keep only the upload files."""
    tools = {t: shutil.which(t) for t in ("pdflatex", "bibtex")}
    missing = [t for t, p in tools.items() if p is None]
    if missing:
        raise SystemExit(
            f"cannot test-compile the submission: {', '.join(missing)} not on PATH.\n"
            "Fix: install TeX Live (or TinyTeX) and make sure pdflatex and bibtex are on PATH, then rerun\n"
            "  python3 make_jcp.py --submission"
        )
    sub = HERE / "submission"
    sub.mkdir(exist_ok=True)
    for f in sub.iterdir():
        f.unlink()
    flat = out.replace("{../figures/fig_", "{fig_")
    flat = flat.replace("\\bibliography{../references}", "\\bibliography{references}")
    assert "../" not in flat, "a relative path to another folder remains in the submission source"
    (sub / "ContinuumLimit_JCP.tex").write_text(flat)
    for src in [HERE.parent / "references.bib", HERE / "elsarticle-num-names.bst", *FIGURES]:
        shutil.copy2(src, sub / src.name)

    def run(*cmd: str) -> None:
        subprocess.run(cmd, cwd=sub, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)

    run(tools["pdflatex"], "-interaction=nonstopmode", "ContinuumLimit_JCP.tex")
    run(tools["bibtex"], "ContinuumLimit_JCP")
    for _ in range(2):
        run(tools["pdflatex"], "-interaction=nonstopmode", "ContinuumLimit_JCP.tex")
    log = (sub / "ContinuumLimit_JCP.log").read_text(errors="replace")
    errors = [ln for ln in log.splitlines() if ln.startswith("! ")]
    undefined = [ln for ln in log.splitlines() if "undefined" in ln]
    if errors or undefined or not (sub / "ContinuumLimit_JCP.pdf").exists():
        raise SystemExit(
            "the submission folder does NOT compile with pdfLaTeX:\n  "
            + "\n  ".join((errors + undefined)[:10])
            + f"\nSee {sub / 'ContinuumLimit_JCP.log'} for the full log."
        )
    keep = {
        "ContinuumLimit_JCP.tex",
        "ContinuumLimit_JCP.bbl",
        "references.bib",
        "elsarticle-num-names.bst",
    }
    keep |= {f.name for f in FIGURES}
    for f in sub.iterdir():
        if f.name not in keep:
            f.unlink()
    print(f"submission/: compiles with pdfLaTeX; upload these {len(keep)} files: {', '.join(sorted(keep))}")


if "--submission" in sys.argv[1:]:
    assemble_submission()
