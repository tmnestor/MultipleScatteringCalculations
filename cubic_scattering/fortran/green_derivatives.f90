!=======================================================================================================
! green_derivatives.f90 -- every Cartesian derivative, to any order, of the two radial functions that
!                          make up the elastodynamic Green's tensor, at many points, OpenMP over points
!=======================================================================================================
!
! WHAT IT COMPUTES
! ----------------
! The Green's tensor and all its derivatives are built from two radial functions,
!
!   g_S = e^{i k_S r} / r,        B = (g_S - g_P) / k_S^2,        k_S = omega / beta,  k_P = omega / alpha,
!
! through  d_D G_in = (1 / 4 pi mu) [ delta_in d_D g_S + d_{D + e_i + e_n} B ]  (assembled in Python).
! For each point x (non-zero) this routine returns d^a g_S for the first n_s multi-indices a and d^a B
! for the first n_b, in the order Python lists them (graded_voxel.derivatives.multi_indices).
!
! It is a term-for-term transcription of the Python reference, cubic_scattering/graded_voxel/
! derivatives.py (functions radial_ladders and scalar_derivative_fields_python), and
! tests/test_green_derivatives.py checks that the two agree to round-off and that the reference agrees
! with a 40-digit evaluation. If you change one, change the other.
!
! THE LADDER, AND THE TERMS
! -------------------------
! As in point_kernel.f90 (whose header explains it, with a reading guide to the Fortran used), every
! derivative of a radial function f is a sum of terms  coefficient * x^e * F_q,  F_q = ((1/r) d/dr)^q f.
! Which terms make up which derivative is NOT coded here: Python computes the table once
! (derivatives.derivative_terms) and passes it in, flattened:
!
!   term_start(a) .. term_start(a+1) - 1     the terms of multi-index number a
!   term_coef(t), term_exp(1:3, t), term_q(t)  coefficient, exponents of (x, y, z), ladder index q
!
! THE TWO BRANCHES OF THE LADDER (the cancellation in B)
! ------------------------------------------------------
!  * |k_S| r <= series_limit: the power series  e^{ikr}/r = sum_t (ik)^t r^(t-1) / t!.  B's t = 0 term,
!    the 1/r singularity that g_S and g_P share, is never formed, so nothing cancels. F_q of r^m is
!    falling(m, q) r^(m - 2q),  falling(m, q) = m (m-2) ... (m-2q+2).
!  * otherwise the closed forms F_q = e^{ikr} p_q(ikr) / r^(2q+1), with the polynomial coefficients
!    poly(q, j) of p_q passed in from Python (derivatives.bessel_polynomials).
!
!-------------------------------------------------------------------------------------------------------

module green_derivatives
  implicit none
  private
  public :: scalar_derivative_fields

  integer, parameter :: dp = kind(1.0d0)
  complex(dp), parameter :: iu = (0.0_dp, 1.0_dp)

contains

  ! m (m-2) ... (m-2q+2): the coefficient of r^(m-2q) in F_q of r^m (1 for q = 0)
  pure function falling(m, q) result(c)
    integer, intent(in) :: m, q
    real(dp) :: c
    integer :: s
    c = 1.0_dp
    do s = 0, q - 1
      c = c * real(m - 2 * s, dp)
    end do
  end function falling

  subroutine scalar_derivative_fields(x, n, omega, alpha, beta, series_limit, n_series, q_max, poly, &
                                      n_idx, term_start, n_terms, term_coef, term_exp, term_q, &
                                      n_s, n_b, out_s, out_b)
    integer, intent(in) :: n, n_series, q_max, n_idx, n_terms, n_s, n_b
    real(dp), intent(in) :: x(3, n)                 ! separations, one per column
    complex(dp), intent(in) :: omega                ! complex when attenuated
    real(dp), intent(in) :: alpha, beta, series_limit
    real(dp), intent(in) :: poly(0:q_max, 0:q_max)  ! p_q(x) = sum_j poly(q, j) x^j
    integer, intent(in) :: term_start(n_idx + 1)    ! 1-based, into the term arrays
    real(dp), intent(in) :: term_coef(n_terms)
    integer, intent(in) :: term_exp(3, n_terms), term_q(n_terms)
    complex(dp), intent(out) :: out_s(n_s, n), out_b(n_b, n)

    complex(dp) :: ks, kp, fs(0:q_max), fb(0:q_max), aks, akp, a_t, b_t, z, ph, pq, val
    real(dp) :: r, fact, c, pw, xp(0:q_max, 3), mono
    integer :: p, t, q, j, a, k, ax

    ks = omega / beta
    kp = omega / alpha

    !$omp parallel do schedule(static) &
    !$omp private(r, fs, fb, aks, akp, a_t, b_t, fact, c, pw, z, ph, pq, val, xp, mono, t, q, j, a, k, ax)
    do p = 1, n
      r = sqrt(x(1, p)**2 + x(2, p)**2 + x(3, p)**2)
      fs = (0.0_dp, 0.0_dp)
      fb = (0.0_dp, 0.0_dp)

      if (abs(ks) * r <= series_limit) then
        aks = (1.0_dp, 0.0_dp)   ! (i k_S)^t
        akp = (1.0_dp, 0.0_dp)   ! (i k_P)^t
        fact = 1.0_dp            ! t!
        do t = 0, n_series - 1
          if (t > 0) then
            aks = aks * iu * ks
            akp = akp * iu * kp
            fact = fact * real(t, dp)
          end if
          a_t = aks / fact
          b_t = (aks - akp) / (fact * ks**2)
          do q = 0, q_max
            c = falling(t - 1, q)
            if (c == 0.0_dp) cycle
            pw = c * r**(t - 1 - 2 * q)
            fs(q) = fs(q) + a_t * pw
            if (t >= 1) fb(q) = fb(q) + b_t * pw   ! t = 0, the shared singularity, never enters B
          end do
        end do
      else
        do k = 1, 2   ! 1: S, 2: P
          if (k == 1) then
            z = iu * ks * r
          else
            z = iu * kp * r
          end if
          ph = exp(z)
          do q = 0, q_max
            pq = (0.0_dp, 0.0_dp)
            do j = q_max, 0, -1   ! Horner
              pq = pq * z + poly(q, j)
            end do
            val = ph * pq / r**(2 * q + 1)
            if (k == 1) then
              fs(q) = val
              fb(q) = fb(q) + val / ks**2
            else
              fb(q) = fb(q) - val / ks**2
            end if
          end do
        end do
      end if

      ! powers of the coordinates, x_ax^e for e = 0..q_max
      do ax = 1, 3
        xp(0, ax) = 1.0_dp
        do j = 1, q_max
          xp(j, ax) = xp(j - 1, ax) * x(ax, p)
        end do
      end do

      do a = 1, max(n_s, n_b)
        if (a <= n_s) out_s(a, p) = (0.0_dp, 0.0_dp)
        if (a <= n_b) out_b(a, p) = (0.0_dp, 0.0_dp)
        do t = term_start(a), term_start(a + 1) - 1
          mono = term_coef(t) * xp(term_exp(1, t), 1) * xp(term_exp(2, t), 2) * xp(term_exp(3, t), 3)
          if (a <= n_s) out_s(a, p) = out_s(a, p) + mono * fs(term_q(t))
          if (a <= n_b) out_b(a, p) = out_b(a, p) + mono * fb(term_q(t))
        end do
      end do
    end do
    !$omp end parallel do
  end subroutine scalar_derivative_fields

end module green_derivatives
