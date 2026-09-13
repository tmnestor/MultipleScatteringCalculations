#!/usr/bin/env python3
"""GATE, rung 3L: the STRATIFIED vertical sweep operator.

The production vertical operator is the thesis Chapter 5 stratified propagator
``Q^d = (I - S_int E)^-1 S_int`` (GstratRep.tex, Eq. PstratDef), reached through
the validated route: the Riccati layered Green's function plus the three wrapper
corrections D1/D2/D3 (docs/wrapper_problem_state_2026-09-13.md).

  [3L-a] HOMOGENEOUS REDUCTION. On a uniform model with the plane pair taken
         deep enough that the free surface is attenuated, the stratified
         operator must equal the whole-space kernel built by an INDEPENDENT
         construction -- the k_z residue of the whole-space Green's tensor,
         itself gated against the closed-form Kupradze propagator at rung 3.
         Exact: 1e-13.

  [3L-b] ATTENUATION CONSISTENCY, the control. Rebuilding the same quantity with
         the source/receiver operators taken in the UNDAMPED medium -- what
         GlobalMatrix.layered_greens_9x9 and scripts/composed_matvec
         .resolved_9x9_grid both do, via _interface_elastic_properties ->
         float(model.alpha[j]) -- must FAIL here. If it does not, this gate is
         not discriminating and [3L-a] proves nothing.

  [3L-c] LAYERING IS ACTUALLY PRESENT. With a real contrast between the planes,
         the stratified operator must differ substantially from the whole-space
         one. Otherwise [3L-a] could pass on an operator that silently ignores
         the model.

Run:  conda run -n seismic python scripts/gate_sweep_rung3_layered.py
Seismic units (km/s, g/cm3), time convention e^{-i omega t}.
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "PhD_fortran_code"))
sys.path.insert(0, "/Users/tod/Desktop/SeismicInversion")

from cubic_scattering.effective_contrasts import ReferenceMedium  # noqa: E402
from cubic_scattering.layered_correction import (  # noqa: E402
    correct_6x6,
    corrected_layered_9x9,
    source_jump_operator,
    strain_from_state,
)
from cubic_scattering.sweep_kernels import vertical_kernel_9x9  # noqa: E402
from Kennett_Reflectivity.layer_model import LayerModel  # noqa: E402

ALPHA, BETA, RHO = 4.0, 2.22, 2.6
Q_LIMIT = 2.0  # heavy damping: kills the free-surface return in a few km
PITCH = 1.0
# 16 crust layers, matching the model the wrapper resolution was validated on.
# With fewer, the interface pair used below lands on the half-space boundary and
# layered_greens_6x6 returns zeros -- which reads as a 1.000 relative residual,
# not as an error.
N_LAY = 16
FREQ = 6.0

KX = np.array([0.4, 1.1, 1.9])
KY = np.full_like(KX, 0.3)


def uniform_model(q: float = Q_LIMIT) -> LayerModel:
    """Ocean over N_LAY identical crust layers; interface k at the bottom of layer k.

    The deep pair (8, 9) sits ~8 km below the free surface, which at Q = 2 is
    many attenuation lengths away -- so those two interfaces see a whole space.
    """
    return LayerModel.from_arrays(
        alpha=[1.5, *([ALPHA] * N_LAY), ALPHA],
        beta=[0.0, *([BETA] * N_LAY), BETA],
        rho=[1.03, *([RHO] * N_LAY), RHO],
        thickness=[3.0, *([PITCH] * N_LAY), np.inf],
        Q_alpha=[q] * (N_LAY + 2),
        Q_beta=[1e10, *([q] * N_LAY), q],
    )


def contrast_model(q: float = Q_LIMIT) -> LayerModel:
    """Same, but with a fast slab strictly BETWEEN the plane pair.

    The planes sit at interfaces 7 and 11; the fast layers are 9 and 10, so the
    material is continuous across both planes (layers 7|8 and 11|12 match) while
    a strong contrast is crossed twice in between -- transmission, reflection,
    conversion and interbed multiples all present. Putting the contrast ON a
    plane instead would violate the layer-interior rule that the correction
    operator K requires, and assert_interface_continuous would reject it.
    """
    al = [1.5, *([ALPHA] * N_LAY), ALPHA]
    be = [0.0, *([BETA] * N_LAY), BETA]
    rh = [1.03, *([RHO] * N_LAY), RHO]
    for lay in (9, 10):
        al[lay], be[lay], rh[lay] = 6.5, 3.7, 3.3
    return LayerModel.from_arrays(
        alpha=al,
        beta=be,
        rho=rh,
        thickness=[3.0, *([PITCH] * N_LAY), np.inf],
        Q_alpha=[q] * (N_LAY + 2),
        Q_beta=[1e10, *([q] * N_LAY), q],
    )


def whole_space_stack(model: LayerModel, omega: complex, dz: float) -> np.ndarray:
    """The independent whole-space kernel, in the model's own complex medium."""
    s_p, s_s = model.complex_slowness_p(), model.complex_slowness_s()
    ref = ReferenceMedium(1.0 / s_p[1], 1.0 / s_s[1], model.rho[1])
    return np.stack([vertical_kernel_9x9(np.array([k]), KY[0], dz, omega, ref)[:, :, 0] for k in KX])


def undamped_operator_variant(model, omega, j, i) -> np.ndarray:
    """The same 9x9 but with A and B in the UNDAMPED medium -- the control."""
    from GlobalMatrix.layered_greens import layered_greens_6x6

    raw = layered_greens_6x6(model, omega, KX, KY, source_iface=j, receiver_iface=i)
    s_s = model.complex_slowness_s()
    out = np.empty((KX.size, 9, 9), dtype=complex)
    for t in range(KX.size):
        g_c = correct_6x6(raw[t], omega, s_s[max(j, 1)], s_s[max(i, 1)], KX[t], KY[t])
        a = strain_from_state(KX[t], KY[t], model.rho[i], model.alpha[i], model.beta[i])
        b = source_jump_operator(KX[t], KY[t], model.rho[j], model.alpha[j], model.beta[j])
        out[t] = a @ g_c @ b
    return out


def main() -> int:
    omega = 2 * np.pi * FREQ
    print("=" * 76)
    print("GATE rung 3L -- stratified vertical operator (thesis Ch.5 Q^d)")
    print(f"  crust alpha={ALPHA} beta={BETA} rho={RHO}, Q={Q_LIMIT}, pitch={PITCH} km")
    print(f"  freq={FREQ} Hz, plane pair at interfaces 8 and 9 (~8 km below the surface)")
    print("=" * 76)

    uni = uniform_model()
    ok = True

    print("\n[3L-a] HOMOGENEOUS REDUCTION vs the independent whole-space kernel")
    print(f"      {'source':>7} {'recv':>5} {'dz':>6}   {'relative residual':>18}")
    for j, i, dz in ((9, 8, -PITCH), (8, 9, +PITCH), (9, 7, -2 * PITCH)):
        got = corrected_layered_9x9(uni, omega, KX, KY, j, i)
        want = whole_space_stack(uni, omega, dz)
        rel = float(np.abs(got - want).max() / np.abs(want).max())
        good = rel < 1e-13
        ok = ok and good
        print(f"      {j:7d} {i:5d} {dz:+6.1f}   {rel:18.3e}")
    print(f"      target < 1e-13  ->  {'PASS' if ok else 'FAIL'}")

    print("\n[3L-b] CONTROL -- the undamped-operator variant must FAIL the same test")
    bad = undamped_operator_variant(uni, omega, 9, 8)
    want = whole_space_stack(uni, omega, -PITCH)
    rel_bad = float(np.abs(bad - want).max() / np.abs(want).max())
    okc = rel_bad > 1e-2
    print(f"      A, B built from float(model.alpha[j]) -- undamped: {rel_bad:.3e}")
    print("      This is what GlobalMatrix.layered_greens_9x9 and")
    print("      scripts/composed_matvec.resolved_9x9_grid compute. At field Q the")
    print("      error is ~0.1% and invisible; here it is order one.")
    print(f"      the control discriminates  ->  {'PASS' if okc else 'FAIL'}")
    ok = ok and okc

    print("\n[3L-c] LAYERING IS PRESENT -- a real contrast must change the answer")
    con = contrast_model()
    got_c = corrected_layered_9x9(con, omega, KX, KY, 11, 7)
    got_u = corrected_layered_9x9(uni, omega, KX, KY, 11, 7)
    rel_c = float(np.abs(got_c - got_u).max() / np.abs(got_u).max())
    okl = rel_c > 1e-2
    print("      planes at interfaces 7 and 11; fast layers 9-10 strictly between")
    print(f"      change vs the uniform model: {rel_c:.3e}")
    print(f"      the operator sees the model  ->  {'PASS' if okl else 'FAIL'}")
    ok = ok and okl

    print("\n" + "=" * 76)
    print(f"GATE rung 3L: {'PASS' if ok else 'FAIL'}")
    print("=" * 76)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
