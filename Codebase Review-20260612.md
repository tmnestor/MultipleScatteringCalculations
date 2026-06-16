Codebase Review

Overall health: strong

This is a mature, well-organized research codebase. The things that are usually wrong with research code are not
wrong here:

- Clean public API — __init__.py exports 124 symbols deliberately; internals stay hidden.
- Lint/types clean — ruff and mypy both pass with zero issues on all 61 files.
- YAML discipline holds — solver_config.py is the single config source; no hardcoded shadows found. The two
except fallbacks that exist (singular Mie matrix → physically-zero coefficient; PyTorch missing → logged CPU
fallback) are documented and legitimate.
- No god objects — the biggest modules (effective_contrasts.py at ~1900 lines, lattice_greens.py at ~1500) are
large because the physics is, not because of tangling. Dependency flow is acyclic and matches the CLAUDE.md
architecture diagram.
- Validation culture — Mie sphere as an independent check on the cube machinery, Kennett as the reference for
slab R_PP, Rayleigh-limit regression for the resonance solver.

Issues found (all minor)

1. Stale TODO pointer — cubic_scattering/__init__.py:24 says "see TODO in resonance_tmatrix.py", but no TODO
exists in that file anymore. Either the caveat about the approximate Voigt conversion is resolved (remove the
note) or the TODO was lost (restore it where the limitation lives).
2. Scratch scripts inside the package — baseline_fft_final.py, baseline_kx_residue.py, baseline_kz_residue.py,
greens_fft_cli.py, verify_prefactor.py, debug_body_bilinear.py (~2,200 lines total) are development/verification
artifacts not imported by anything. They belong in scripts/ or derivations/, not in the importable package.
(shifted_master_values.py and compute_gerade_blocks.py are load-bearing — keep those.)
3. Indirect-only test coverage on core modules — effective_contrasts.py, voigt_tmatrix.py, and lattice_greens.py
are exercised only through integration tests. They work, but a regression in (say) one irrep block would
surface as a confusing downstream failure rather than a pointed unit failure.

Suggested next steps

Cross-referencing the plans/ directory, the LaTeX "future work" sections, and the ocean-bottom README, here's
how I'd prioritize:

Near term — finish what's started

1. Complete the Mie verification plan (plans/Complete_Verification.md). Only P-incidence is validated; the plan
calls for all five channels (P→P, P→SV, SV→P, SV→SV, SH→SH) with phase-sensitive Re/Im comparison. This is the
highest-leverage item because it hardens the validation backbone everything else rests on — especially before
adding S-wave incidence anywhere else.
2. Mode-converted reflections in the slab/ocean-bottom solver. The ocean-bottom study currently extracts only
R_PP; R_PS, R_SP, R_SS need SV/SH-incidence solves. This pairs naturally with item 1 — validate the channels on
the sphere first, then extend the slab. It also unlocks the AVO/Zoeppritz validation listed in
marine_survey_explanation.tex.
3. Dynamic body bilinear (plans/dynamic_body_bilinear.md). The math framework (radiation damping via the smooth
polynomial part of the Green's tensor, MpPoly/K1atPoly master integrals) is fully specified and "ready for
coding" — the cheapest plan to land, and it improves T-matrix accuracy at finite ka.

Medium term — assemble the big pieces

4. Full T₂₇ lattice Foldy-Lax (plans/multiple_scattering_t27.md). The irrep blocks exist and the inter-voxel
propagator is complete through ω⁶; what's missing is the assembly into the FFT-accelerated multi-voxel solver.
This is the codebase's stated end-goal — strong-contrast multiple scattering with the 27-component basis instead
of the 9-component Rayleigh one.
5. CPA-homogenised background. Use the existing cpa_iteration.py effective medium as the slab/survey background
(listed as a future extension in both slab_scattering_explanation.tex and marine_survey_explanation.tex). Most
of the machinery exists; this is mostly plumbing plus a validation study.
6. Resonance T₀ in the slab solver. slab_scattering.py currently uses compute_cube_tmatrix (Rayleigh); swapping
in compute_resonance_tmatrix for ka > 0.3 would extend the ocean-bottom study to higher frequencies / larger
voxels. Resolving the stale Voigt-conversion TODO (issue 1 above) is a prerequisite.

Longer term — research directions from your own plans

7. Frequency-sweep parallelization and attenuation (Q) for the ocean-bottom study — both flagged in the LaTeX
future-work sections, both incremental.
8. The VIE/GPU program (plans/AI_Architecture_Enables_VIE.md) — Gabor-frame spatial-spectral methods and neural
surrogates. This is the ambitious horizon; the T₂₇ lattice solver (item 4) is the stepping stone toward it.

Hygiene (do alongside, not instead)

- Fix the stale TODO pointer in __init__.py.
- Move the six scratch scripts out of the package.
- Add direct unit tests for effective_contrasts.py, voigt_tmatrix.py, and lattice_greens.py.


RESOLVE why Claude heredocs hang in this shell setup - fix the zsh setup if necessary




Two findings worth carrying forward (both in memory): the radiation part barely moves normal-incidence near-field results (~0.5%) but is O(1) of the G block by ka≈0.5 and essential for distant-coupling/far-field physics — exactly your point. And off-normal, volume averaging matches Kennett but
doesn't beat the point propagator, because it corrects vertical coupling, not the ~1 rad/cell horizontal phase — a real, documented limitation and a clean candidate for future work.


Websearch whether the (ka)^4 correct single site cubic T-matrix result is already known in the seismic literature.

We need a paragraph to explains the relative advantages of the current physics based model compared to the direct discretization of the elastodynamic wave equations.


########################################################################################
The research questions described below, began 30+ years ago - in my PhD. I would like to complete this chapter of my life now!
I need a plan that uses (and extends if necessary) this codebase to address the research questions described below:
Research Question 1: Near Field Interactions are error-prone and computationally expensive - is it better to calculate near-field interaction within the depth fixed-planes only, and then use kennet recursions to include interactions between the depth planes - or should we use full 3-D multiple scattering near field (assuming homogeneous background)
I need luatex generated PDF that includes the background theory required to understand the Analytical Sphere solution, the full T₂₇ lattice Foldy-Lax Multiple Scattering Representation and T_9 solution. This should include all necessary Academic References, Tables and TiKz generated figures. Some of this latex components are in LatexPDFs/ but some content is now out of date!


❯Most of the errors related back to the incorrect Galerkin bi-linear representation. Initial analysis predicted that the bilinear form would improve near-field interactions - but this was later refuted by the agents. I am wondering whether we should fall back to point source representations only?