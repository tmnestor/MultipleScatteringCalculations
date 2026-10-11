"""Audit: the stable evaluation against quadrature at self, face, edge and corner contact, term by term.

Paper 2 (ExactCouplingIntegrals), "Self and face contact": for all 41 static terms and all pairs of the degree-one
field, at h = 1.25 (so that the physical scaling of each piece by powers of h is checked too), the largest
difference relative to the largest entry of each term, worst and median over the terms. Quadrature: Duffy
pyramids and Gauss rules of 20 points per axis (now from the 30-digit table of cubic_scattering.gauss_rules).

Run:  PYTHONPATH=. python scripts/audit_self_face_contact.py
"""

import numpy as np

from cubic_scattering.effective_contrasts import ReferenceMedium
from cubic_scattering.graded_voxel import blocks

H = 1.25
REF = ReferenceMedium(alpha=5000.0, beta=3000.0, rho=2500.0)


def main() -> None:
    terms = sorted(blocks.static_term_table(REF.alpha, REF.beta, REF.rho))
    for name, off in (("self", (0, 0, 0)), ("face", (1, 0, 0)), ("edge", (1, 1, 0)), ("corner", (1, 1, 1))):
        errs = []
        for m, idx in terms:
            closed = blocks.static_term_integral_closed(m, idx, off, H, 10, 4)
            quad = blocks.static_term_integral(m, idx, off, H, 20, 10, 4)
            scale = np.abs(quad).max()
            if scale > 0:
                errs.append(float(np.abs(closed - quad).max() / scale))
        print(f"{name:6s}: worst term {max(errs):.1e}, median {np.median(errs):.1e} ({len(errs)} terms)", flush=True)


if __name__ == "__main__":
    main()
