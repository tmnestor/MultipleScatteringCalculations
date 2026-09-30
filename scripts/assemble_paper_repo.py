#!/usr/bin/env python3
"""Assemble the self-contained reproduction repository for the two voxel papers.

The research repository holds far more than the papers need. A reviewer wants a small tree that
reproduces every table and figure, with a README that maps each to one command. This script builds
that tree from an explicit manifest, so it can be regenerated at every revision and never drifts:

  * the ``cubic_scattering`` package (the object under test, copied whole, tests included);
  * the scripts the papers cite, plus the two they import;
  * the Mathematica notebooks the papers cite, the notebooks they load, and their saved outputs;
  * the two papers' LaTeX sources, bibliographies and figures, and the journal build script;
  * a licence, a minimal environment file, a two-tier reproduction script and a README.

The tree keeps the research repository's layout (``cubic_scattering/``, ``scripts/``,
``Mathematica/``, ``LatexPDFs/``), so every relative path in the scripts keeps working.

Before the tree is declared built, it is scrubbed: any absolute home-directory path, any file that
does not belong in a public release, and any text matching the exclusion patterns fails the build.

Usage (from the research repository root):
    python scripts/assemble_paper_repo.py ../continuum-limit-paper [--force]
"""

import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# --- manifest ----------------------------------------------------------------------------------------

SCRIPTS = [
    # continuum-limit paper: cross-checks, gates, pilots, reference dumps, figure
    "crosscheck_first_moment_voxel.py",
    "crosscheck_second_moment_voxel.py",
    "crosscheck_graded_contrast.py",
    "crosscheck_graded_sphere.py",
    "gate_cube_shear_split.py",
    "gate_sphere_closure_vs_mie.py",
    "gate_heterogeneous_reference_vs_kennett.py",
    "gate_sphere_cell_average_vs_mie.py",  # imported by the sphere pilots
    "pilot_sphere_voxel_vs_mie.py",
    "pilot_graded_sphere_vs_exact.py",  # imported by the graded-sphere scripts
    "pilot_graded_voxel_sphere.py",
    "continuum_limit_reference.py",
    "continuum_limit_specular.py",
    "continuum_limit_averaged.py",
    "continuum_limit_cubet.py",
    "continuum_limit_chain.py",
    "plot_convergence_orders.py",
    # graded-voxel paper
    "measure_graded_voxel_site.py",
    "measure_graded_voxel_t2.py",
    "measure_graded_voxel_resolution.py",
    "check_graded_voxel_subdivision.py",
    "plot_graded_voxel_orders.py",
    # this script and its templates, so the release records how it was made
    "assemble_paper_repo.py",
    "paper_repo/README.md",
    "paper_repo/reproduce.sh",
    "paper_repo/environment.yml",
    "paper_repo/gitignore",
    "paper_repo/ruff.toml",
]

NOTEBOOKS = [
    "ContinuumLimit_Reference.wl",
    "ContinuumLimit_ThinLayer.wl",
    "ContinuumLimit_SpecularSums.wl",
    "ContinuumLimit_StaticLocal.wl",
    "ContinuumLimit_AveragedSums.wl",
    "ContinuumLimit_CubeT.wl",
    "ContinuumLimit_Chain.wl",
    "ContinuumLimit_ErrorConstant.wl",
    "ContinuumLimit_FourthOrder.wl",
    "ContinuumLimit_FourthOrderError.wl",
    "ContinuumLimit_Reduction3D.wl",
    "ContinuumLimit_Oblique.wl",
    "ContinuumLimit_ObliqueExport.wl",
    "ContinuumLimit_IncidentS.wl",
    "ContinuumLimit_IncidentSExport.wl",
    "ContinuumLimit_Heterogeneous.wl",
    "ContinuumLimit_SecondMoment.wl",
    "ContinuumLimit_GradedContrast.wl",
    "ContinuumLimit_GradedSphere.wl",
    "ContinuumLimit_Figure.wl",
    "TakeuchiSaito.wl",
    "SphereClosureDynamic.wl",
    "MieAsymptoticRaw.wl",  # loaded by SphereClosureDynamic.wl
    "CubeMomentCore.wl",
    "CubeMomentCoreTest.wl",
    "CubeA22Block.wl",
    "CubeShearSplit.wl",
    "CubeMomentArchiveCheck.wl",
    "CubeAnalytic.wl",  # loaded by the cube notebooks
    "CubeT6Masters.wl",  # loaded by the cube notebooks
    "GradedVoxel_WeakKernel.wl",
    "GradedVoxel_LayerReduction.wl",
]

NOTEBOOK_OUTPUTS_GLOB = ["ContinuumLimit_*.json", "GradedVoxel_*.jsonl"]

PAPERS = {
    "LatexPDFs/ContinuumLimit": [
        "ContinuumLimit.pdf",
        "ContinuumLimit.tex",
        "references.bib",
        "figures/fig_preamble.tex",
        "figures/fig_lattice.tex",
        "figures/fig_poisson.tex",
        "figures/fig_tiling.tex",
        "figures/fig_lattice.pdf",
        "figures/fig_poisson.pdf",
        "figures/fig_tiling.pdf",
        "figures/fig_convergence.pdf",
        "figures/data_graded_sphere_ka0.5.json",
        "figures/data_graded_sphere_ka1.0.json",
        "JCP/ContinuumLimit_JCP.pdf",
        "JCP/make_jcp.py",
        "JCP/elsarticle-num-names.bst",
    ],
    "LatexPDFs/GradedVoxel": [
        "GradedVoxel.pdf",
        "GradedVoxel.tex",
        "references.bib",
        "figures/fig_orders.pdf",
        "figures/data_born_ka0.5.json",
        "figures/data_born_ka1.0.json",
        "figures/data_full_ka0.5.json",
        "figures/data_full_ka1.0.json",
        "figures/data_resolution_core0.25.json",
        "figures/data_resolution_core0.5.json",
    ],
}

# Nothing machine-local may reach the release: no home-directory path, and none of the local tooling
# that this machine keeps out of every repository through its global git excludes file. The names are
# read from that file rather than written here, so the release rule and the machine rule cannot drift.


def local_exclusions() -> tuple[set[str], list[str]]:
    """Names and text stems of local tooling, from the machine's global git excludes file.

    Returns:
        (names, stems): file or directory names that must not be copied, and lower-case word stems that
        must not appear in any text file. A stem is taken from a pattern part that is marked as tooling
        (a leading dot, a wildcard or an upper-case name); plain words such as ``docs`` are not stems.
    """
    out = subprocess.run(["git", "config", "--get", "core.excludesFile"], capture_output=True, text=True)
    raw = out.stdout.strip()
    path = Path(raw).expanduser() if raw else None
    if path is None or not path.exists():
        raise SystemExit(
            "the machine's global git excludes file is not configured or does not exist\n"
            "  What: core.excludesFile is unset or points to a missing file, so the release scrub cannot\n"
            "        learn which local tooling names to exclude\n"
            f"  Where: git config core.excludesFile (currently {raw!r})\n"
            "  Should look like: core.excludesFile = ~/.gitignore_global, a file listing the local\n"
            "        tooling directories and files, one pattern per line\n"
            "  Fix: git config --global core.excludesFile ~/.gitignore_global, and create the file"
        )
    names: set[str] = set()
    stems: list[str] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [q for q in line.strip("/").split("/") if q and q != "**"]
        names.add(parts[-1].replace("*", ""))
        for part in parts:
            marked = part.startswith(".") or "*" in part or part.isupper() or part.split(".")[0].isupper()
            stem = part.replace("*", "").lstrip(".").split(".")[0].lower()
            if marked and stem.isalpha() and len(stem) >= 4 and stem not in stems:
                stems.append(stem)
    return names, stems


_NAMES, _STEMS = local_exclusions()
# the parent of the home directory, so any absolute path into a user account is caught on any platform
HOME_ROOT = str(Path.home().parent) + "/"
FORBIDDEN = re.compile("|".join([re.escape(HOME_ROOT)] + [re.escape(w) for w in _STEMS]), re.IGNORECASE)
FORBIDDEN_NAMES = _NAMES | {"docs", "memory", "plans"}
DECLARATION_HEADING = "Declaration of generative AI"
TEXT_SUFFIXES = {
    ".py",
    ".wl",
    ".tex",
    ".bib",
    ".md",
    ".yml",
    ".yaml",
    ".sh",
    ".json",
    ".jsonl",
    ".bst",
    ".txt",
    "",
}


# The release's own files are kept as templates beside this script, so they can be edited as text.
TEMPLATES = ROOT / "scripts" / "paper_repo"
ENVIRONMENT = (TEMPLATES / "environment.yml").read_text()
GITIGNORE = (TEMPLATES / "gitignore").read_text()
REPRODUCE = (TEMPLATES / "reproduce.sh").read_text()
README = (TEMPLATES / "README.md").read_text()
RUFF = (TEMPLATES / "ruff.toml").read_text()


# --- assembly ----------------------------------------------------------------------------------------


def copy(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)


def assemble(dest: Path) -> list[str]:
    """Copy the manifest into ``dest``. Returns the list of copied files relative to dest."""
    copied: list[str] = []

    def take(rel: str) -> None:
        src = ROOT / rel
        if not src.exists():
            raise FileNotFoundError(
                f"manifest names a file that does not exist: {rel}\n"
                f"  What: the reproduction manifest in scripts/assemble_paper_repo.py is out of date\n"
                f"  Where: {ROOT / 'scripts' / 'assemble_paper_repo.py'}, "
                f"the SCRIPTS / NOTEBOOKS / PAPERS lists\n"
                f"  Fix: correct or remove the entry, then rerun"
            )
        copy(src, dest / rel)
        copied.append(rel)

    for path in sorted((ROOT / "cubic_scattering").rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        take(str(path.relative_to(ROOT)))
    for name in SCRIPTS:
        take(f"scripts/{name}")
    for name in NOTEBOOKS:
        take(f"Mathematica/{name}")
    for pattern in NOTEBOOK_OUTPUTS_GLOB:
        for path in sorted((ROOT / "Mathematica").glob(pattern)):
            take(str(path.relative_to(ROOT)))
    for folder, files in PAPERS.items():
        for name in files:
            take(f"{folder}/{name}")
    take("LICENSE")

    (dest / "environment.yml").write_text(ENVIRONMENT)
    (dest / ".gitignore").write_text(GITIGNORE)
    (dest / "ruff.toml").write_text(RUFF)
    (dest / "README.md").write_text(README)
    (dest / "reproduce.sh").write_text(REPRODUCE)
    (dest / "reproduce.sh").chmod(0o755)
    return copied


def scrub(dest: Path) -> list[str]:
    """Return every violation of the release rules found in ``dest``: forbidden names and text."""
    problems: list[str] = []
    for path in sorted(dest.rglob("*")):
        rel_parts = path.relative_to(dest).parts
        if rel_parts and rel_parts[0] in {".git", ".ruff_cache", "__pycache__", ".pytest_cache"}:
            continue
        if path.name in FORBIDDEN_NAMES or any(
            part in FORBIDDEN_NAMES for part in path.relative_to(dest).parts
        ):
            problems.append(f"forbidden name: {path.relative_to(dest)}")
            continue
        if path.is_file() and path.suffix in TEXT_SUFFIXES:
            try:
                text = path.read_text(errors="replace")
            except OSError:
                continue
            # The papers' declaration of AI-assisted preparation, which the journals require, is the
            # one place a tool may be named: the block from its heading to the next sectioning command.
            in_declaration = False
            for i, line in enumerate(text.splitlines(), 1):
                if DECLARATION_HEADING in line:
                    in_declaration = True
                elif in_declaration and line.lstrip().startswith(
                    ("\\section", "\\bibliographystyle", "\\end{document}")
                ):
                    in_declaration = False
                if in_declaration:
                    continue
                m = FORBIDDEN.search(line)
                if m:
                    problems.append(f"{path.relative_to(dest)}:{i}: '{m.group(0)}'")
    return problems


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    force = "--force" in sys.argv
    if len(args) != 1:
        print(__doc__)
        return 2
    dest = Path(args[0]).resolve()
    if dest.exists():
        if not force:
            print(f"{dest} exists; pass --force to replace its contents (its .git directory is kept)")
            return 2
        for child in dest.iterdir():
            if child.name == ".git":
                continue
            shutil.rmtree(child) if child.is_dir() else child.unlink()
    dest.mkdir(parents=True, exist_ok=True)

    copied = assemble(dest)
    print(f"copied {len(copied)} files into {dest}")
    problems = scrub(dest)
    if problems:
        print("the release fails the scrub; fix these in the research repository and rerun:")
        for p in problems:
            print("  " + p)
        return 1
    print("scrub clean: no home-directory paths, no forbidden names or text")
    if (dest / ".git").exists():
        status = subprocess.run(["git", "status", "--short"], cwd=dest, capture_output=True, text=True)
        changed = len(status.stdout.splitlines())
        print(f"{dest.name}: {changed} paths changed since the last commit there")
    else:
        print(
            f"no git repository in {dest}; initialise one with:  git -C {dest} init && git -C {dest} add -A"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
