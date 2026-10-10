# Standalone reproduction repositories, one per paper: design

**Status:** design only. Build it as the **last step before each paper is submitted**, from that paper's
final `.tex`. Until then `scripts/assemble_paper_repo.py` and `scripts/paper_repo/` are left as they are
(out of date: they describe the old combined continuum-limit paper).

## 1. Why the current script is wrong

- It assembles one release for the old combined paper. That paper is now split: the layer law and the
  sphere are in Paper 3, the hierarchy is archived.
- Its manifest is three hand-kept lists (`SCRIPTS`, `NOTEBOOKS`, `PAPERS`) and a hand-written README
  mapping results to sections (§3, §6.4, §9, ...). Every reorganisation of the papers broke them.
- Its `reproduce.sh` runs hierarchy scripts that no paper now cites.

## 2. Principle: the paper is the manifest

Each release is generated from the paper's final `.tex`. The script reads what the paper cites, adds
what those files load, and copies the result. Nothing is listed by hand except the few files no paper
cites but every release needs (§4). A release can then only drift if the paper's own citations are wrong,
and the build reports those.

## 3. Papers

| Paper | Directory | Default release name |
|---|---|---|
| 1. Single-site T-matrix of a Legendre cell | `LatexPDFs/LegendreCellTMatrix/` | `legendre-cell-tmatrix` |
| 2. Coupling integrals of polynomial cubes | `LatexPDFs/ExactCouplingIntegrals/` | `exact-coupling-integrals` |
| 3. The error of polynomial voxels | `LatexPDFs/PolynomialVoxelError/` | `polynomial-voxel-error` |

Archives (`OctreeRefinementArchive/`, `CartesianMultipoleHierarchyArchive/`) get no release.
`GradedVoxel/` is outside the series; add it to the table only if it is submitted.

Usage: `python scripts/assemble_paper_repo.py <paper> <dest> [--force]`, with `<paper>` a key of the
table, or `all`.

## 4. What a release contains

1. **The paper's directory, whole**, less build products (`.aux`, `.log`, `.bbl`, `.blg`, `.out`,
   `.synctex.gz`): the `.tex`, the `.pdf`, `references.bib`, `figures/` with their sources and data.
2. **Files the paper cites**, found by scanning the `.tex` (§5).
3. **Files those files load**, transitively (§6).
4. **The package `cubic_scattering/`, whole, tests included**, with its non-Python files
   (`numerics.yml`, the Fortran sources of the point kernel and `.f2py_f2cmap`). Every script imports it,
   and the papers cite modules and tests by name. The built extension is never copied.
5. **Fixed files:** `LICENSE`, `environment.yml`, `.gitignore`, `ruff.toml` (templates in
   `scripts/paper_repo/`), and the generated `README.md` and `reproduce.sh` (§7).
6. **This script and its templates**, so the release records how it was made.

## 5. Finding the citations

Scan the `.tex` for `\path{...}`, `\nb{...}` and `\texttt{...}`, with `\_` read as `_`. Resolve each
token against, in order: the repository root; `Mathematica/` (bare notebook names, e.g. `TakeuchiSaito.wl`);
`scripts/`; `cubic_scattering/` (e.g. `tests/test_graded_voxel_fft.py`); the paper's own directory
(e.g. Paper 1's `data/self_blocks_irreps.txt`). Forms seen in the papers that need care:

| Form | Example | Handling |
|---|---|---|
| Wildcard | `scripts/continuum_limit_*.py` | expand the glob |
| Named family | `Mathematica/ContinuumLimit_X.wl, for X in Reference, ThinLayer, ... and Figure` | parse the list after "for X in" up to the next `;` or "and their" |
| Directory | `scratch/taup/`, `scratch/legendre_sphere_series/` | every file in it (the outputs of the runs the paper reports) |
| Module or function | `blocks.far_block_series`, `octree.born_series_octree` | ignore; covered by the package |
| Non-file `\texttt` | `source\_moments`, `PhD\_fortran\_code` | ignore if it resolves to nothing; `PhD_fortran_code` is a folder of the research repository, cited for history, not copied |
| Truncated by a line break | `\path{scripts/pilot_graded_voxel_sphere.py` | take the token up to the first space |

**Any token that looks like a file (has a path separator or a known suffix) and resolves to nothing fails
the build**, with the line of the `.tex` it came from. That is the check that the paper's citations are
correct.

## 6. Transitive dependencies

- **Python:** for each script, the local modules it imports (`import X` / `from X import` where
  `scripts/X.py` exists), recursively. Data files it reads by a literal path that exists in the research
  repository (`*.json`, `*.npz`, `*.txt`, `*.pkl`) are copied too; caches it writes
  (`scripts/data/legendre_far_series/`, `scripts/data/octree_finest_blocks/`, gitignored) are not, and
  the README says they are rebuilt on first run, with the time.
- **Mathematica:** for each notebook, every quoted `"*.wl"`, `"*.m"` or `"*.json"` name that exists in
  `Mathematica/` (covers `Get`, `<<`, `Needs` by file, `Import`, `FileNameJoin`), recursively, and the
  saved outputs it exports. The notebooks carry the symbolic derivations behind every paper and are the
  centre of the release, not an extra: a notebook whose output a paper quotes must ship with that output,
  so the number can be checked without running it.

## 7. Generated README and `reproduce.sh`

**README**, generated per paper:
- Title and one-paragraph summary, taken from the `.tex` (`\title`, the abstract's first sentences).
- **Map of the paper**: one row per section, with the scripts and notebooks cited in it, found by the
  section in which each citation occurs. This replaces the hand-written table.
- **Environment:** `conda env create -f environment.yml`, building the Fortran point kernel, running the
  package tests.
- **Mathematica:** the notebooks need Wolfram Mathematica (the author holds Wolfram's permission to use
  the academic licence; a reader needs their own) or the free Wolfram Engine with `wolframscript`. One
  `wolframscript -file Mathematica/<name>.wl` line per notebook, its run time, and the output file it
  writes. Saved outputs are included for readers without a licence.
- **Timings:** quick and full tiers, with the measured time of each long run.
- **Licence and citation.**

**`reproduce.sh`**, generated per paper:
- `quick`: the package tests, then every cited `gate_*`, `crosscheck_*` and `test_*` script, and the
  figure scripts, from saved data (minutes).
- `full`: every cited script, with the arguments of the `Run:` line in its docstring where it has one,
  else none (hours).
- `notebooks`: every cited notebook through `wolframscript`, if it is on the path; otherwise a message
  saying which notebooks were skipped and that their saved outputs are in the release.

## 8. Checks before a release is declared built

1. Every file-like citation resolved (§5).
2. The scrub of the current script, kept: no absolute home-directory path, no local tooling names from
   the machine's global git excludes, no forbidden folders (`docs`, `memory`, `plans`). The papers'
   declaration of AI assistance is the one place a tool may be named.
3. The release's own `.tex` builds with `lualatex` and `bibtex`, with no undefined references.
4. `reproduce.sh quick` passes inside the release, in a fresh environment.
5. If `wolframscript` is available, each notebook runs and reproduces its saved output.
6. A size report, with any file over 5 MB listed.

## 9. Order of work at submission time

1. Freeze the paper's text.
2. Rewrite `scripts/assemble_paper_repo.py` to this design, with tests on all three papers.
3. Build the paper's release, run §8, publish it, and put its address in the paper's Data availability
   section.
4. Rebuild the PDF with that address.
