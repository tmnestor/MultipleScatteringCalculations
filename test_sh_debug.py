"""Debug the SH far-field fix: check c_n values and formulas."""

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

    # Print coefficients
    for n in range(0, min(4, n_max + 1)):
        print(f"  n={n}: c_n = {mie.c_n[n]:.6e}, b_n_sv = {mie.b_n_sv[n]:.6e}")
        if n > 0 and abs(mie.b_n_sv[n]) > 1e-30:
            print(f"         c_n/b_n_sv = {mie.c_n[n]/mie.b_n_sv[n]:.4f}")

    # Test multiple SH far-field formulas
    formulas = {}

    # Formula A: current code (c_n, M-type with (-1)^n * (-ikS) * h_n * tau_n)
    f_A = np.zeros(len(THETA_ARR), dtype=complex)
    for i, theta in enumerate(THETA_ARR):
        u_phi = 0.0j
        for n in range(1, n_max + 1):
            renorm = -1.0 / (n * (n + 1))
            tau_n = _dPn1_dtheta(n, theta)
            h_n = _spherical_h1_complex(n, kS * r_eval)
            radial = (-1.0)**n * (-1j * kS) * h_n
            u_phi += mie.c_n[n] * renorm * radial * tau_n
        f_A[i] = u_phi * r_eval * np.exp(-1j * kS * r_eval)
    formulas["c_n (-1)^n (-ikS) h_n tau_n"] = f_A

    # Formula B: c_n, M-type WITHOUT (-1)^n (test)
    f_B = np.zeros(len(THETA_ARR), dtype=complex)
    for i, theta in enumerate(THETA_ARR):
        u_phi = 0.0j
        for n in range(1, n_max + 1):
            renorm = -1.0 / (n * (n + 1))
            tau_n = _dPn1_dtheta(n, theta)
            h_n = _spherical_h1_complex(n, kS * r_eval)
            radial = (-1j * kS) * h_n
            u_phi += mie.c_n[n] * renorm * radial * tau_n
        f_B[i] = u_phi * r_eval * np.exp(-1j * kS * r_eval)
    formulas["c_n (-ikS) h_n tau_n"] = f_B

    # Formula C: c_n, M-type with (+ikS) * h_n * tau_n
    f_C = np.zeros(len(THETA_ARR), dtype=complex)
    for i, theta in enumerate(THETA_ARR):
        u_phi = 0.0j
        for n in range(1, n_max + 1):
            renorm = -1.0 / (n * (n + 1))
            tau_n = _dPn1_dtheta(n, theta)
            h_n = _spherical_h1_complex(n, kS * r_eval)
            radial = 1j * kS * h_n
            u_phi += mie.c_n[n] * renorm * radial * tau_n
        f_C[i] = u_phi * r_eval * np.exp(-1j * kS * r_eval)
    formulas["c_n (+ikS) h_n tau_n"] = f_C

    # Formula D: c_n with N-type ut_S and tau_n (c_n replaces b_n_sv)
    f_D = np.zeros(len(THETA_ARR), dtype=complex)
    for i, theta in enumerate(THETA_ARR):
        u_phi = 0.0j
        for n in range(1, n_max + 1):
            renorm = -1.0 / (n * (n + 1))
            tau_n = _dPn1_dtheta(n, theta)
            _, ut_S, _, _ = _mie_swave_fields(n, kS, r_eval, REF.mu, "h1")
            u_phi += mie.c_n[n] * renorm * ut_S * tau_n
        f_D[i] = u_phi * r_eval * np.exp(-1j * kS * r_eval)
    formulas["c_n ut_S tau_n"] = f_D

    # Formula E: c_n with N-type ut_S and pi_n
    f_E = np.zeros(len(THETA_ARR), dtype=complex)
    for i, theta in enumerate(THETA_ARR):
        u_phi = 0.0j
        for n in range(1, n_max + 1):
            renorm = -1.0 / (n * (n + 1))
            pi_n = _Pn1_over_sintheta(n, theta)
            _, ut_S, _, _ = _mie_swave_fields(n, kS, r_eval, REF.mu, "h1")
            u_phi += mie.c_n[n] * renorm * ut_S * pi_n
        f_E[i] = u_phi * r_eval * np.exp(-1j * kS * r_eval)
    formulas["c_n ut_S pi_n"] = f_E

    # Formula F: OLD formula (b_n_sv with ut_S and pi_n)
    f_F = np.zeros(len(THETA_ARR), dtype=complex)
    for i, theta in enumerate(THETA_ARR):
        u_phi = 0.0j
        for n in range(1, n_max + 1):
            renorm = -1.0 / (n * (n + 1))
            pi_n = _Pn1_over_sintheta(n, theta)
            _, ut_S, _, _ = _mie_swave_fields(n, kS, r_eval, REF.mu, "h1")
            u_phi += mie.b_n_sv[n] * renorm * ut_S * pi_n
        f_F[i] = u_phi * r_eval * np.exp(-1j * kS * r_eval)
    formulas["OLD: b_n_sv ut_S pi_n"] = f_F

    # Formula G: c_n, just h_n * tau_n (no ikS factor)
    f_G = np.zeros(len(THETA_ARR), dtype=complex)
    for i, theta in enumerate(THETA_ARR):
        u_phi = 0.0j
        for n in range(1, n_max + 1):
            renorm = -1.0 / (n * (n + 1))
            tau_n = _dPn1_dtheta(n, theta)
            h_n = _spherical_h1_complex(n, kS * r_eval)
            u_phi += mie.c_n[n] * renorm * h_n * tau_n
        f_G[i] = u_phi * r_eval * np.exp(-1j * kS * r_eval)
    formulas["c_n h_n tau_n (no kS)"] = f_G

    # --- Foldy-Lax reference ---
    fl = compute_sphere_foldy_lax(
        omega, RADIUS, REF, CONTRAST, n_sub=6, k_hat=K_HAT_Z, wave_type="S"
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
    for label, f_mie in formulas.items():
        ref_mag = max(np.max(np.abs(f_mie)), np.max(np.abs(f_fl)), 1e-30)
        err_re = np.max(np.abs(f_fl.real - f_mie.real)) / ref_mag
        err_im = np.max(np.abs(f_fl.imag - f_mie.imag)) / ref_mag
        err_mag = np.max(np.abs(np.abs(f_fl) - np.abs(f_mie))) / ref_mag
        print(f"  {label:40s}: err_Re={err_re:.4f}, err_Im={err_im:.4f}, err_|f|={err_mag:.4f}")

    # Sample values
    idx = len(THETA_ARR) // 3
    theta_deg = np.degrees(THETA_ARR[idx])
    print(f"\n  Sample at theta={theta_deg:.1f} deg:")
    print(f"    LF:                                    {f_fl[idx]:.6e}")
    for label, f_mie in formulas.items():
        print(f"    {label:40s}: {f_mie[idx]:.6e}")
