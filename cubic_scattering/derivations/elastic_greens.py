"""
elastic_greens.py
Symbolic elastodynamic Green's function and its spatial derivative
for a homogeneous isotropic background medium.

Implements Appendix A of Shekhar et al. (2023):
  "Integral equation method for microseismic wavefield modelling
   in anisotropic elastic media"

All expressions are symbolic (SymPy). Use .subs() or lambdify()
for numerical evaluation.
"""

import sympy as sp
from sympy import (
    I,
    KroneckerDelta,
    Matrix,
    Rational,
    cos,
    exp,
    pi,
    sin,
    sqrt,
    symbols,
)

# ================================================================
# Symbolic parameters
# ================================================================
omega, alpha, beta, rho = symbols(r"\omega \alpha \beta \rho", positive=True)
a_sym = symbols("a", positive=True)
r_sym = symbols("r", positive=True)
theta_sym, phi_sym = symbols(r"\theta \phi", real=True)

# Cartesian position components (relative: x_r - x_s)
x1, x2, x3 = symbols("x_1 x_2 x_3", real=True)
x_vec = [x1, x2, x3]
r_cart = sqrt(x1**2 + x2**2 + x3**2)


# ================================================================
# Green's tensor radial decomposition
# ================================================================
# G_{ij}(x) = f(r) delta_{ij} + g(r) n_i n_j
# where n_i = x_i / r
#
# From Appendix A:
#   C   = (1/(8*pi*rho*beta^2)) * (1 - beta^2/alpha^2)
#   X(r) = exp(i*omega*r/alpha) / (4*pi*rho*alpha^2)
#   V(r) = exp(i*omega*r/beta)  / (4*pi*rho*beta^2)
#
#   f(r) = V(r)/r - C/r
#   g(r) = 3C/r + X(r)/r - V(r)/r


def C_nf():
    """Near-field constant."""
    return (1 - beta**2 / alpha**2) / (8 * pi * rho * beta**2)


def X_func(r):
    """P-wave far-field radial function."""
    return exp(I * omega * r / alpha) / (4 * pi * rho * alpha**2)


def V_func(r):
    """S-wave far-field radial function."""
    return exp(I * omega * r / beta) / (4 * pi * rho * beta**2)


def f_rad(r):
    """Isotropic part of G_{ij}: coefficient of delta_{ij}."""
    return V_func(r) / r - C_nf() / r


def g_rad(r):
    """Anisotropic part of G_{ij}: coefficient of n_i n_j."""
    return 3 * C_nf() / r + X_func(r) / r - V_func(r) / r


def greens_cart(i: int, j: int):
    """
    Full Green's tensor G_{ij} in Cartesian coordinates.
    i, j in {0, 1, 2} (0-indexed).
    Returns a SymPy expression in (x1, x2, x3).
    """
    r = r_cart
    ni = x_vec[i] / r
    nj = x_vec[j] / r
    return f_rad(r) * KroneckerDelta(i, j) + g_rad(r) * ni * nj


def greens_matrix_cart():
    """Full 3x3 Green's tensor as a SymPy Matrix."""
    return Matrix(3, 3, lambda i, j: greens_cart(i, j))


# ================================================================
# Near-field, P-wave, S-wave components (for inspection)
# ================================================================


def G_NF(i, j):
    """Near-field Green's tensor G^NF_{ij}."""
    r = r_cart
    ni, nj = x_vec[i] / r, x_vec[j] / r
    return -C_nf() / r * (KroneckerDelta(i, j) - 3 * ni * nj)


def G_P(i, j):
    """P-wave far-field Green's tensor G^P_{ij}."""
    r = r_cart
    ni, nj = x_vec[i] / r, x_vec[j] / r
    return X_func(r) * ni * nj / r


def G_S(i, j):
    """S-wave far-field Green's tensor G^S_{ij}."""
    r = r_cart
    ni, nj = x_vec[i] / r, x_vec[j] / r
    return V_func(r) * (KroneckerDelta(i, j) - ni * nj) / r


# ================================================================
# Spatial derivatives
# ================================================================


def greens_deriv(i, j, k):
    """G_{ij,k} = dG_{ij}/dx_k in Cartesian coordinates."""
    return sp.diff(greens_cart(i, j), x_vec[k])


def greens_deriv2(i, j, k, l):
    """G_{ij,kl} = d^2 G_{ij} / (dx_k dx_l) in Cartesian."""
    return sp.diff(greens_cart(i, j), x_vec[k], x_vec[l])


# ================================================================
# H tensor (Eq. 10)
# ================================================================


def H_tensor(i, j, k):
    """
    H^(0)_{ijk} = (1/2)(G_{ij,k} + G_{ik,j})
    First-order spatial derivative of background Green's function.
    """
    return Rational(1, 2) * (greens_deriv(i, j, k) + greens_deriv(i, k, j))


# ================================================================
# Utility: convert Cartesian -> spherical
# ================================================================


def to_spherical(expr):
    """
    Substitute x1, x2, x3 -> spherical coordinates (r, theta, phi).
    Uses r_sym, theta_sym, phi_sym as the spherical symbols.
    """
    return expr.subs(
        {
            x1: r_sym * sin(theta_sym) * cos(phi_sym),
            x2: r_sym * sin(theta_sym) * sin(phi_sym),
            x3: r_sym * cos(theta_sym),
        }
    )
