!=======================================================================================================
! cell_average.f90: the receiver-cell average of the derivative tensors of D = g_S - g_P and of g_S,
!                   for many lattice offsets at once, OpenMP over offsets
!=======================================================================================================
!
! WHAT IT COMPUTES
! ----------------
! For each separation s between a point source and the centre of a cubic receiver cell of half-width h,
!
!   <d^n f>(s) = sum_{a,b,c} w_a w_b w_c  d^n f(s - (u_a, u_b, u_c)),
!
! with (u, w) the Gauss rule on [-h, h] (weights summing to 1), for
!
!   f = D = g_S - g_P   at orders n = 2, 3, 4      (the d_i d_j D part of G and its two derivatives)
!   f = g_S             at orders n = 0, 1, 2      (the delta_ij k_S^2 g_S part)
!   g_c = e^{i kappa_c r} / (4 pi r).
!
! These are exactly the tensors the 9 x 9 block is assembled from (Python:
! kupradze_derivatives.greens_from_difference, then resonance_tmatrix._voigt_contract), so the assembly
! stays in Python and the expensive part, the cubed Gauss rule, runs here.
!
! It is a term-for-term transcription of the Python reference, cubic_scattering/cell_averaged_pair.py
! (_receiver_average, with kupradze_derivatives.difference_radial_ladder, radial_ladder, _hankel1 and
! _tensors_from_ladder), and tests/test_cell_average_fortran.py checks that the two agree to round-off.
! If you change one, change the other.
!
! THE TWO BRANCHES OF THE DIFFERENCE LADDER
! -----------------------------------------
! F_q = ((1/r) d/dr)^q f.  D's two ladders cancel their leading 1/r terms against each other, so
!  * max(|kappa_P|, |kappa_S|) r <= series_limit: the power series
!      D = (1/4 pi) sum_{t>=1} i^t (kappa_S^t - kappa_P^t) r^(t-1) / t!,
!    whose t = 0 term is never formed, so nothing cancels; F_q of r^m is falling(m, q) r^(m-2q);
!  * otherwise the closed form of each mode, differenced:
!      F_q[g] = (i kappa^(q+1) / 4 pi) (-1)^q h_q(kappa r) / r^q,
!    with h_q from its terminating series (complex kappa allowed).
! g_S always takes the closed form, as in Python.
!
! THE TENSORS FROM THE LADDER (radial_component below)
! ----------------------------------------------------
!   n = 0:  F_0
!   n = 1:  x_i F_1
!   n = 2:  delta_ij F_1 + x_i x_j F_2
!   n = 3:  (delta_ij x_k + delta_ik x_j + delta_jk x_i) F_2 + x_i x_j x_k F_3
!   n = 4:  (the three delta delta pairings) F_2 + (the six delta x x pairings) F_3 + x_i x_j x_k x_m F_4
! with x the CARTESIAN separation (not the unit vector).
!
! The nodes are summed in Python's order (z outermost, then x, then y), the weight formed as
! (w_z w_x) w_y, so the two implementations differ only by the round-off of individual operations.
! Each offset is averaged by one thread with no reduction across offsets, so the result does not depend
! on the number of OpenMP threads.
!-------------------------------------------------------------------------------------------------------

module cell_average
  implicit none
  private
  public :: averaged_tensors

  integer, parameter :: dp = kind(1.0d0)
  complex(dp), parameter :: iu = (0.0_dp, 1.0_dp)
  real(dp), parameter :: pi = 3.141592653589793238462643383279502884_dp

contains

  ! n! as a double (exact for the small n used here)
  pure function factorial(n) result(out)
    integer, intent(in) :: n
    real(dp) :: out
    integer :: j
    out = 1.0_dp
    do j = 2, n
      out = out * real(j, dp)
    end do
  end function factorial

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

  ! spherical Hankel h_n^(1)(x) from its terminating series. Python: kupradze_derivatives._hankel1
  pure function hankel1(n, x) result(out)
    integer, intent(in) :: n
    complex(dp), intent(in) :: x
    complex(dp) :: out, acc
    integer :: s
    acc = (0.0_dp, 0.0_dp)
    do s = 0, n
      acc = acc + (iu**s / (factorial(s) * (2.0_dp * x)**s)) * (factorial(n + s) / factorial(n - s))
    end do
    out = (-iu)**(n + 1) * (exp(iu * x) / x) * acc
  end function hankel1

  ! F_q of g = e^{i kappa r} / (4 pi r), q = 0..4, closed form. Python: kupradze_derivatives.radial_ladder
  pure subroutine closed_ladder(r, kappa, f)
    real(dp), intent(in) :: r
    complex(dp), intent(in) :: kappa
    complex(dp), intent(out) :: f(0:4)
    integer :: q
    do q = 0, 4
      f(q) = iu * kappa**(q + 1) / (4.0_dp * pi) * real((-1)**q, dp) * hankel1(q, kappa * r) / r**q
    end do
  end subroutine closed_ladder

  ! F_q of D = g_S - g_P, q = 0..4. Python: kupradze_derivatives.difference_radial_ladder
  pure subroutine difference_ladder(r, kp, ks, series_limit, n_series, f)
    real(dp), intent(in) :: r, series_limit
    complex(dp), intent(in) :: kp, ks
    integer, intent(in) :: n_series
    complex(dp), intent(out) :: f(0:4)
    complex(dp) :: fs(0:4), fp(0:4), coef
    real(dp) :: c
    integer :: t, m, q
    if (max(abs(kp), abs(ks)) * r > series_limit) then
      call closed_ladder(r, ks, fs)
      call closed_ladder(r, kp, fp)
      f = fs - fp
      return
    end if
    f = (0.0_dp, 0.0_dp)
    do t = 1, n_series - 1
      coef = (iu**t) * (ks**t - kp**t) / (4.0_dp * pi * factorial(t))
      m = t - 1
      do q = 0, 4
        c = falling(m, q)
        if (c /= 0.0_dp) f(q) = f(q) + coef * c * r**(m - 2 * q)
      end do
    end do
  end subroutine difference_ladder

  ! the Kronecker delta, as a double
  pure function dl(a, b) result(out)
    integer, intent(in) :: a, b
    real(dp) :: out
    out = merge(1.0_dp, 0.0_dp, a == b)
  end function dl

  ! one component d_{idx(1)} ... d_{idx(n)} f (0 <= n <= 4) from the ladder f(0:4) and the separation x
  pure function radial_component(f, x, idx, n) result(out)
    complex(dp), intent(in) :: f(0:4)
    real(dp), intent(in) :: x(3)
    integer, intent(in) :: idx(4), n
    complex(dp) :: out
    integer :: i, j, k, m
    real(dp) :: pairs, mixed, sym
    select case (n)
    case (0)
      out = f(0)
    case (1)
      out = x(idx(1)) * f(1)
    case (2)
      i = idx(1); j = idx(2)
      out = dl(i, j) * f(1) + x(i) * x(j) * f(2)
    case (3)
      i = idx(1); j = idx(2); k = idx(3)
      sym = dl(i, j) * x(k) + dl(i, k) * x(j) + dl(j, k) * x(i)
      out = sym * f(2) + x(i) * x(j) * x(k) * f(3)
    case default                                ! n = 4
      i = idx(1); j = idx(2); k = idx(3); m = idx(4)
      pairs = dl(i, j) * dl(k, m) + dl(i, k) * dl(j, m) + dl(i, m) * dl(j, k)
      mixed = dl(i, j) * x(k) * x(m) + dl(i, k) * x(j) * x(m) + dl(i, m) * x(j) * x(k) &
        + dl(j, k) * x(i) * x(m) + dl(j, m) * x(i) * x(k) + dl(k, m) * x(i) * x(j)
      out = pairs * f(2) + mixed * f(3) + x(i) * x(j) * x(k) * x(m) * f(4)
    end select
  end function radial_component

  !-----------------------------------------------------------------------------------------------------
  ! averaged_tensors: the receiver-cell averages at n separations.
  !
  ! Arguments (all supplied by the Python wrapper, cell_average_fortran.averaged_tensors_fortran):
  !   s(3, n)               separations, one per column, in the package's (z, x, y) order; the wrapper has
  !                         checked that no receiver cell reaches its source, so r > 0 at every node.
  !   kp, ks                kappa_P = omega / alpha and kappa_S = omega / beta (complex when attenuated).
  !   nodes(ng_max, n_rules), weights(ng_max, n_rules), rule_n(n_rules)
  !                         the distinct Gauss rules on [-h, h], already scaled (Python: _cell_nodes), the
  !                         first rule_n(k) entries of column k being rule k.
  !   rule(n)               which rule (1-based) each separation uses.
  !   series_limit, n_series  the branch switch and series length (Python: DIFFERENCE_SERIES_LIMIT,
  !                         DIFFERENCE_N_SERIES), passed in so the two implementations cannot disagree.
  !   dd2, dd3, dd4         output: averaged d^2 D, d^3 D, d^4 D, last index the separation.
  !   ds0, ds1, ds2         output: averaged g_S, d g_S, d^2 g_S.
  !-----------------------------------------------------------------------------------------------------
  subroutine averaged_tensors(s, n, kp, ks, nodes, weights, ng_max, n_rules, rule_n, rule, series_limit, &
                              n_series, dd2, dd3, dd4, ds0, ds1, ds2)
    integer, intent(in) :: n, ng_max, n_rules, n_series
    real(dp), intent(in) :: s(3, n)
    complex(dp), intent(in) :: kp, ks
    real(dp), intent(in) :: nodes(ng_max, n_rules), weights(ng_max, n_rules)
    integer, intent(in) :: rule_n(n_rules), rule(n)
    real(dp), intent(in) :: series_limit
    complex(dp), intent(out) :: dd2(3, 3, n), dd3(3, 3, 3, n), dd4(3, 3, 3, 3, n)
    complex(dp), intent(out) :: ds0(n), ds1(3, n), ds2(3, 3, n)

    complex(dp) :: fd(0:4), fs(0:4)
    real(dp) :: x(3), r, w
    integer :: p, k, ng, a, b, c, i, j, l, m

    !$omp parallel do schedule(dynamic) private(fd, fs, x, r, w, k, ng, a, b, c, i, j, l, m)
    do p = 1, n
      dd2(:, :, p) = (0.0_dp, 0.0_dp)
      dd3(:, :, :, p) = (0.0_dp, 0.0_dp)
      dd4(:, :, :, :, p) = (0.0_dp, 0.0_dp)
      ds0(p) = (0.0_dp, 0.0_dp)
      ds1(:, p) = (0.0_dp, 0.0_dp)
      ds2(:, :, p) = (0.0_dp, 0.0_dp)
      k = rule(p)
      ng = rule_n(k)
      do a = 1, ng            ! z
        do b = 1, ng          ! x
          do c = 1, ng        ! y
            w = weights(a, k) * weights(b, k) * weights(c, k)
            x(1) = s(1, p) - nodes(a, k)
            x(2) = s(2, p) - nodes(b, k)
            x(3) = s(3, p) - nodes(c, k)
            r = sqrt(x(1)**2 + x(2)**2 + x(3)**2)
            call difference_ladder(r, kp, ks, series_limit, n_series, fd)
            call closed_ladder(r, ks, fs)

            ds0(p) = ds0(p) + w * radial_component(fs, x, [1, 1, 1, 1], 0)
            do i = 1, 3
              ds1(i, p) = ds1(i, p) + w * radial_component(fs, x, [i, 1, 1, 1], 1)
              do j = 1, 3
                ds2(i, j, p) = ds2(i, j, p) + w * radial_component(fs, x, [i, j, 1, 1], 2)
                dd2(i, j, p) = dd2(i, j, p) + w * radial_component(fd, x, [i, j, 1, 1], 2)
                do l = 1, 3
                  dd3(i, j, l, p) = dd3(i, j, l, p) + w * radial_component(fd, x, [i, j, l, 1], 3)
                  do m = 1, 3
                    dd4(i, j, l, m, p) = dd4(i, j, l, m, p) + w * radial_component(fd, x, [i, j, l, m], 4)
                  end do
                end do
              end do
            end do
          end do
        end do
      end do
    end do
    !$omp end parallel do
  end subroutine averaged_tensors

end module cell_average
