!=======================================================================================================
! kseries_moments.f90: the universal coefficients of the frequency-amortised receiver-cell average,
!                      U(t, a)(o) = < d^a r^(t-1) > over the unit cube, for many offsets, OpenMP over offsets
!=======================================================================================================
!
! WHAT IT COMPUTES
! ----------------
! For an integer offset o between a point source and the centre of a unit receiver cube,
!
!   U(t, a)(o) = sum_{nodes} w  d^a r^(t-1) (o - xi),        t = 0 .. t_max,  |a| <= 4,
!
! with xi, w a tensor Gauss rule on [-1/2, 1/2]^3 (weights summing to 1). Every averaged derivative of
! g = e^{i kappa r} / r is then a power series in kappa with these coefficients,
!
!   < d^a g >(s) = sum_t (i kappa)^t / t!  d^(t - 1 - |a|)  U(t, a)(s / d),      d the cell side,
!
! so the coefficients do not depend on the frequency, the cell size or the medium: universal constants,
! computed once and stored (cell_average_kseries.py), after which every frequency of a sweep costs a
! short weighted sum per offset. This is the principle the project works to: analytic or tabulated
! integrals, where possible, are amortised across frequency.
!
! It is a term-for-term transcription of the Python reference, graded_voxel/derivatives.py
! (kseries_coefficients, with the weight W = 0), and tests/test_cell_average_kseries.py checks that the
! two agree to round-off. If you change one, change the other.
!
! WHY THIS STEP IS IN FORTRAN (21 October 2026)
! ---------------------------------------------
! The coefficients are needed for every canonical offset of the grid: about 2,600 at n_sub = 25, each a
! 24- or 28-point cubed Gauss rule (13,824 to 21,952 nodes) evaluated for 49 powers and 35 derivatives.
! The Python reference takes 0.13 to 0.19 s per offset, so about 6.5 minutes for n_sub = 25: acceptable
! once ever, but it would make the first run on a new machine, or any growth of the grid, slower than
! the quadrature it replaces. Compiled, with OpenMP over offsets, the same table takes about a minute of
! CPU. The table is stored on disk and grows on demand, so this cost is paid once per machine; the
! per-frequency work that follows is in NumPy and needs no compiled code.
!
! THE TERMS
! ---------
! d^a f of a radial f is a sum of  coefficient * x^e * F_q,  F_q = ((1/r) d/dr)^q f  (graded_voxel.
! derivatives.derivative_terms, passed in flattened). For f = r^m, F_q = falling(m, q) r^(m - 2q),
! falling(m, q) = m (m-2) ... (m-2q+2), which vanishes once 2q > m for even m >= 0: the derivatives of an
! even power, a polynomial, end. Powers of x and of r are built by multiplication per node.
!
! Each offset is computed by one thread with no reduction across offsets, so the result does not depend on
! the number of OpenMP threads.
!-------------------------------------------------------------------------------------------------------

module kseries_moments_mod
  implicit none
  private
  public :: kseries_moments

  integer, parameter :: dp = kind(1.0d0)

contains

  ! m (m-2) ... (m-2q+2): the coefficient of r^(m-2q) in F_q of r^m (1 for q = 0)
  pure function falling(m, q) result(out)
    integer, intent(in) :: m, q
    real(dp) :: out
    integer :: s
    out = 1.0_dp
    do s = 0, q - 1
      out = out * real(m - 2 * s, dp)
    end do
  end function falling

  !-----------------------------------------------------------------------------------------------------
  ! kseries_moments: U(t, a) at n offsets.
  !
  ! Arguments (supplied by the Python wrapper, cell_average_kseries.coefficients_fortran):
  !   o(3, n)                 integer offsets (as doubles), in the package's (z, x, y) order, non-zero.
  !   x1(ng_max, n_rules), w1(ng_max, n_rules), rule_n(n_rules)
  !                           the distinct one-dimensional Gauss rules on [-1/2, 1/2] (weights summing to 1),
  !                           the first rule_n(k) entries of column k being rule k.
  !   rule(n)                 which rule (1-based) each offset uses, per axis.
  !   t_max                   the highest power index t (r^(t-1)).
  !   term_start(n_a + 1), term_coef(n_terms), term_exp(3, n_terms), term_q(n_terms)
  !                           the terms of each multi-index a (1-based starts into the term arrays).
  !   q_max, e_max            the largest q and the largest exponent in the terms.
  !   u(0:t_max, n_a, n)      output.
  !-----------------------------------------------------------------------------------------------------
  subroutine kseries_moments(o, n, x1, w1, ng_max, n_rules, rule_n, rule, t_max, n_a, term_start, &
                             n_terms, term_coef, term_exp, term_q, q_max, e_max, u)
    integer, intent(in) :: n, ng_max, n_rules, t_max, n_a, n_terms, q_max, e_max
    real(dp), intent(in) :: o(3, n), x1(ng_max, n_rules), w1(ng_max, n_rules)
    integer, intent(in) :: rule_n(n_rules), rule(n)
    integer, intent(in) :: term_start(n_a + 1), term_exp(3, n_terms), term_q(n_terms)
    real(dp), intent(in) :: term_coef(n_terms)
    real(dp), intent(out) :: u(0:t_max, n_a, n)

    real(dp) :: fall(0:t_max, 0:q_max), x(3), xp(0:e_max, 3), rp(-1 - 2 * q_max:t_max), r, w, acc, c
    integer :: p, k, ng, a, b, cc, t, ia, j, ax, e, pw

    do t = 0, t_max
      do j = 0, q_max
        fall(t, j) = falling(t - 1, j)
      end do
    end do

    !$omp parallel do schedule(dynamic) private(k, ng, a, b, cc, w, x, r, xp, rp, ax, e, pw, t, ia, j, acc, c)
    do p = 1, n
      u(:, :, p) = 0.0_dp
      k = rule(p)
      ng = rule_n(k)
      do a = 1, ng
        do b = 1, ng
          do cc = 1, ng
            w = w1(a, k) * w1(b, k) * w1(cc, k)
            x(1) = o(1, p) - x1(a, k)
            x(2) = o(2, p) - x1(b, k)
            x(3) = o(3, p) - x1(cc, k)
            r = sqrt(x(1)**2 + x(2)**2 + x(3)**2)
            do ax = 1, 3
              xp(0, ax) = 1.0_dp
              do e = 1, e_max
                xp(e, ax) = xp(e - 1, ax) * x(ax)
              end do
            end do
            rp(0) = 1.0_dp
            do pw = 1, t_max
              rp(pw) = rp(pw - 1) * r
            end do
            do pw = -1, -1 - 2 * q_max, -1
              rp(pw) = rp(pw + 1) / r
            end do
            do t = 0, t_max
              do ia = 1, n_a
                acc = 0.0_dp
                do j = term_start(ia), term_start(ia + 1) - 1
                  c = term_coef(j) * fall(t, term_q(j))
                  if (c == 0.0_dp) cycle
                  acc = acc + c * xp(term_exp(1, j), 1) * xp(term_exp(2, j), 2) * xp(term_exp(3, j), 3) &
                    * rp(t - 1 - 2 * term_q(j))
                end do
                u(t, ia, p) = u(t, ia, p) + w * acc
              end do
            end do
          end do
        end do
      end do
    end do
    !$omp end parallel do
  end subroutine kseries_moments

end module kseries_moments_mod
