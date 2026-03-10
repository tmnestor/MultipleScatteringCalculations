"""Diagnostic: test SH far-field formulas with corrected M-type sign."""

import numpy as np

from cubic_scattering import MaterialContrast, ReferenceMedium
from cubic_scattering.sphere_scattering import (
    _dPn1_dtheta,
    _mie_swave_fields,
    _Pn1_over_sintheta,
    _spherical_h1_complex,
    compute_elastic_mie,
    compute_sphere_foldy_lax,
    decompose_SV_SH,
    foldy_lax_far_field,
)

REF = ReferenceMedium(alpha=5000.0, beta=3000.0, rho=2500.0)
CONTRAST = MaterialContrast(Dlambda=2.0e9, Dmu=1.0e9, Drho=100.0)
RADIUS = 10.0
N_SUB = 6
THETA_ARR = np.linspace(0.2, np.pi - 0.2, 15)
K_HAT_Z = np.array([1.0, 0.0, 0.0])

for ka_target in [0.1, 0.5]:
    omega = ka_target * REF.beta / RADIUS
    kS = omega / REF.beta
    r_eval = 1.0e6 * RADIUS

    mie = compute_elastic_mie(omega, RADIUS, REF, CONTRAST)
    n_max = mie.n_max

    print(f"\n{'='*70}")
    print(f"ka_S = {ka_target}, omega = {omega:.4f}, kS = {kS:.6f}, n_max = {n_max}")
    print(f"{'='*70}")

    # --- Formula 1: Current (N-type only) ---
    f_current = np.zeros(len(THETA_ARR), dtype=complex)
    for i, theta in enumerate(THETA_ARR):
        u_phi = 0.0j
        for n in range(1, n_max + 1):
            renorm = -1.0 / (n * (n + 1))
            pi_n = _Pn1_over_sintheta(n, theta)
            _, ut_S, _, _ = _mie_swave_fields(n, kS, r_eval, REF.mu, "h1")
            u_phi += mie.b_n_sv[n] * renorm * ut_S * pi_n
        f_current[i] = u_phi * r_eval * np.exp(-1j * kS * r_eval)

    # --- Formula 2: M-type with POSITIVE h_n (code convention) ---
    # The SH boundary system uses z_n(kr) as the displacement variable
    # (not -z_n), so scattered = c_n * h_n(kSr), not c_n * (-h_n)
    f_mtype_pos = np.zeros(len(THETA_ARR), dtype=complex)
    for i, theta in enumerate(THETA_ARR):
        u_phi = 0.0j
        for n in range(1, n_max + 1):
            renorm = -1.0 / (n * (n + 1))
            tau_n = _dPn1_dtheta(n, theta)
            hn_val = _spherical_h1_complex(n, kS * r_eval)
            # +h_n (code convention), then multiply by ikS to match utS asymptotically
            u_phi += mie.c_n[n] * renorm * 1j * kS * hn_val * tau_n
        f_mtype_pos[i] = u_phi * r_eval * np.exp(-1j * kS * r_eval)

    # --- Formula 3: M-type neg h_n (physical M_phi = -h_n * tau) ---
    f_mtype_neg = np.zeros(len(THETA_ARR), dtype=complex)
    for i, theta in enumerate(THETA_ARR):
        u_phi = 0.0j
        for n in range(1, n_max + 1):
            renorm = -1.0 / (n * (n + 1))
            tau_n = _dPn1_dtheta(n, theta)
            hn_val = _spherical_h1_complex(n, kS * r_eval)
            u_phi += mie.c_n[n] * renorm * (-1j * kS) * hn_val * tau_n
        f_mtype_neg[i] = u_phi * r_eval * np.exp(-1j * kS * r_eval)

    # --- Formula 4: M(+h_n) + N-type (no i factor on N-type) ---
    f_M_plus_N = np.zeros(len(THETA_ARR), dtype=complex)
    for i, theta in enumerate(THETA_ARR):
        u_phi = 0.0j
        for n in range(1, n_max + 1):
            renorm = -1.0 / (n * (n + 1))
            tau_n = _dPn1_dtheta(n, theta)
            pi_n = _Pn1_over_sintheta(n, theta)
            hn_val = _spherical_h1_complex(n, kS * r_eval)
            _, ut_S, _, _ = _mie_swave_fields(n, kS, r_eval, REF.mu, "h1")
            u_phi += renorm * (
                mie.c_n[n] * 1j * kS * hn_val * tau_n  # M-type (+h_n * ikS)
                + mie.b_n_sv[n] * ut_S * pi_n           # N-type (no extra i)
            )
        f_M_plus_N[i] = u_phi * r_eval * np.exp(-1j * kS * r_eval)

    # --- Formula 5: M(-h_n) + N-type (no i) ---
    f_Mneg_plus_N = np.zeros(len(THETA_ARR), dtype=complex)
    for i, theta in enumerate(THETA_ARR):
        u_phi = 0.0j
        for n in range(1, n_max + 1):
            renorm = -1.0 / (n * (n + 1))
            tau_n = _dPn1_dtheta(n, theta)
            pi_n = _Pn1_over_sintheta(n, theta)
            hn_val = _spherical_h1_complex(n, kS * r_eval)
            _, ut_S, _, _ = _mie_swave_fields(n, kS, r_eval, REF.mu, "h1")
            u_phi += renorm * (
                mie.c_n[n] * (-1j * kS) * hn_val * tau_n  # M-type (-h_n * ikS)
                + mie.b_n_sv[n] * ut_S * pi_n              # N-type
            )
        f_Mneg_plus_N[i] = u_phi * r_eval * np.exp(-1j * kS * r_eval)

    # --- Formula 6: N-type only, with utS * tau_n instead of utS * pi_n ---
    # (test: what if we use tau_n with b_n_sv?)
    f_bn_tau = np.zeros(len(THETA_ARR), dtype=complex)
    for i, theta in enumerate(THETA_ARR):
        u_phi = 0.0j
        for n in range(1, n_max + 1):
            renorm = -1.0 / (n * (n + 1))
            tau_n = _dPn1_dtheta(n, theta)
            _, ut_S, _, _ = _mie_swave_fields(n, kS, r_eval, REF.mu, "h1")
            u_phi += mie.b_n_sv[n] * renorm * ut_S * tau_n
        f_bn_tau[i] = u_phi * r_eval * np.exp(-1j * kS * r_eval)

    # --- Foldy-Lax reference ---
    fl = compute_sphere_foldy_lax(
        omega, RADIUS, REF, CONTRAST, n_sub=N_SUB, k_hat=K_HAT_Z, wave_type="S"
    )
    r_distance = 500.0 * RADIUS
    M_obs = len(THETA_ARR)
    r_hat_arr = np.zeros((M_obs, 3))
    r_hat_arr[:, 0] = np.cos(THETA_ARR)
    r_hat_arr[:, 1] = np.sin(THETA_ARR)

    pol = np.array([0.0, 0.0, 1.0])
    u_P, u_S = foldy_lax_far_field(fl, r_hat_arr, r_distance, K_HAT_Z, pol, wave_type="S")
    _, f_SH_fl = decompose_SV_SH(u_S, r_hat_arr, K_HAT_Z)
    phase_S = np.exp(1j * kS * r_distance) / r_distance
    f_fl = f_SH_fl / phase_S

    # --- Errors ---
    formulas = [
        ("Current (N-type, pi_n)", f_current),
        ("M(+ikS h_n) only", f_mtype_pos),
        ("M(-ikS h_n) only", f_mtype_neg),
        ("M(+ikS h_n) + N(pi_n)", f_M_plus_N),
        ("M(-ikS h_n) + N(pi_n)", f_Mneg_plus_N),
        ("N-type with tau_n", f_bn_tau),
    ]
    for label, f_mie in formulas:
        ref_mag = max(np.max(np.abs(f_mie)), np.max(np.abs(f_fl)), 1e-30)
        err_re = np.max(np.abs(f_fl.real - f_mie.real)) / ref_mag
        err_im = np.max(np.abs(f_fl.imag - f_mie.imag)) / ref_mag
        err_mag = np.max(np.abs(np.abs(f_fl) - np.abs(f_mie))) / ref_mag
        print(f"  {label:28s}: err_Re={err_re:.4f}, err_Im={err_im:.4f}, err_|f|={err_mag:.4f}")

    # Sample values
    idx = len(THETA_ARR) // 3
    theta_deg = np.degrees(THETA_ARR[idx])
    print(f"\n  Sample at theta={theta_deg:.1f} deg:")
    print(f"    LF:                       {f_fl[idx]:.6e}")
    for label, f_mie in formulas:
        print(f"    {label:28s}: {f_mie[idx]:.6e}")
