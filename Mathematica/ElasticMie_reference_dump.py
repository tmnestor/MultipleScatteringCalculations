import json
import sys

sys.path.insert(0, "/Users/tod/Desktop/MultipleScatteringCalculations")
import numpy as np

# --- find the ReferenceMedium / MaterialContrast dataclasses ---
from cubic_scattering.effective_contrasts import ReferenceMedium
from cubic_scattering.sphere_scattering import (
    _mie_matrix_psv,
    _mie_matrix_sh,
    _mie_pwave_fields,
    _mie_swave_fields,
    _spherical_h1_complex,
    _spherical_h1_deriv,
    _spherical_jn_complex,
    _spherical_jn_deriv,
    compute_elastic_mie,
)

# MaterialContrast location:
try:
    from cubic_scattering.sphere_scattering import MaterialContrast
except Exception:
    from cubic_scattering.effective_contrasts import MaterialContrast

ref = ReferenceMedium(alpha=5000.0, beta=3000.0, rho=2500.0)
contrast = MaterialContrast(Dlambda=2.0e9, Dmu=1.0e9, Drho=100.0)
a = 1.0
omega = 4500.0   # ka_S = omega a / beta = 1.5 ; ka_P = 0.9

lam_out, mu_out = ref.lam, ref.mu
lam_in = lam_out + contrast.Dlambda
mu_in = mu_out + contrast.Dmu
rho_in = ref.rho + contrast.Drho
alpha_in = np.sqrt((lam_in + 2 * mu_in) / rho_in)
beta_in = np.sqrt(mu_in / rho_in)
kP_out, kS_out = omega / ref.alpha, omega / ref.beta
kP_in, kS_in = omega / alpha_in, omega / beta_in


def c2(z):
    z = complex(z)
    return [z.real, z.imag]


def mat(M):
    return [[c2(M[i, j]) for j in range(M.shape[1])] for i in range(M.shape[0])]


out = {
    "params": {
        "alpha": ref.alpha, "beta": ref.beta, "rho": ref.rho,
        "Dlam": contrast.Dlambda, "Dmu": contrast.Dmu, "Drho": contrast.Drho,
        "a": a, "omega": omega,
        "lam_out": lam_out, "mu_out": mu_out,
        "lam_in": lam_in, "mu_in": mu_in, "rho_in": rho_in,
        "alpha_in": alpha_in, "beta_in": beta_in,
        "kP_out": kP_out, "kS_out": kS_out, "kP_in": kP_in, "kS_in": kS_in,
    },
    "pfields": {}, "sfields": {}, "Mpsv": {}, "Msh": {},
}

# field operators at r=a for n=2 (validate symbolic operators)
for n in (1, 2, 3):
    out["pfields"][f"n{n}_jPout"] = [c2(v) for v in _mie_pwave_fields(n, kP_out, a, lam_out, mu_out, "j")]
    out["pfields"][f"n{n}_hPout"] = [c2(v) for v in _mie_pwave_fields(n, kP_out, a, lam_out, mu_out, "h1")]
    out["pfields"][f"n{n}_jPin"]  = [c2(v) for v in _mie_pwave_fields(n, kP_in, a, lam_in, mu_in, "j")]
    out["sfields"][f"n{n}_jSout"] = [c2(v) for v in _mie_swave_fields(n, kS_out, a, mu_out, "j")]
    out["sfields"][f"n{n}_hSout"] = [c2(v) for v in _mie_swave_fields(n, kS_out, a, mu_out, "h1")]
    out["sfields"][f"n{n}_jSin"]  = [c2(v) for v in _mie_swave_fields(n, kS_in, a, mu_in, "j")]
    out["Mpsv"][f"n{n}"] = mat(_mie_matrix_psv(n, omega, a, ref, contrast))
    out["Msh"][f"n{n}"] = mat(_mie_matrix_sh(n, omega, a, ref, contrast))

# full Mie coefficients
res = compute_elastic_mie(omega, a, ref, contrast, n_max=4)
out["coeffs"] = {
    "a_n": [c2(v) for v in res.a_n],
    "b_n": [c2(v) for v in res.b_n],
    "c_n": [c2(v) for v in res.c_n],
    "a_n_sv": [c2(v) for v in res.a_n_sv],
    "b_n_sv": [c2(v) for v in res.b_n_sv],
    "n_max": res.n_max,
}

# corrected SH c_n: rebuild 2x2 with the -z_n/r term, P-incidence-irrelevant; SH incidence
def sh_c_corrected(n):
    z_out = kS_out * a
    z_in = kS_in * a
    h = _spherical_h1_complex(n, z_out); hp = _spherical_h1_deriv(n, z_out)
    j = _spherical_jn_complex(n, z_in);  jp = _spherical_jn_deriv(n, z_in)
    # u_phi ~ z_n ; tau_rphi = mu (kS z_n' - z_n/a)
    M = np.array([[h, -j],
                  [mu_out * (kS_out * hp - h / a), -mu_in * (kS_in * jp - j / a)]], dtype=complex)
    z_inc = kS_out * a
    j_inc = _spherical_jn_complex(n, z_inc); jp_inc = _spherical_jn_deriv(n, z_inc)
    coeff = (2 * n + 1) * (1j) ** n / (1j * kS_out)
    rhs = np.array([-coeff * j_inc,
                    -coeff * mu_out * (kS_out * jp_inc - j_inc / a)], dtype=complex)
    sol = np.linalg.solve(M, rhs)
    return ((-1.0) ** n) * sol[0]

out["c_n_corrected"] = [c2(0)] + [c2(sh_c_corrected(n)) for n in (1, 2, 3, 4)]

print(json.dumps(out))
