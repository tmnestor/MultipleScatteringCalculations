!=======================================================================================================
! point_kernel.f90 -- the 9 x 9 point propagator [[G, C], [H, S]] at many separations, OpenMP over points
!=======================================================================================================
!
! WHAT IT COMPUTES
! ----------------
! For each separation x = x_field - x_source (non-zero), the 9 x 9 block that couples a 9-component
! source (3 forces, 6 Voigt stresses) to a 9-component field (3 displacements, 6 Voigt strains):
!
!       [ G  C ]      G_ij      displacement from force         (3 x 3)
!   P = [      ]      C_i,beta  displacement from stress        (3 x 6)
!       [ H  S ]      H_alpha,j strain from force               (6 x 3)
!                     S_ab      strain from stress              (6 x 6)
!
! C, H and S are contractions of the first and second derivatives of the Green's tensor
!
!   G_ij(x) = (1 / 4 pi mu) [ delta_ij g_b(r) + d_i d_j f_b(r) ],
!   g_b = e^{i k_b r} / r,    f_b = (g_b - g_a) / k_b^2,    k_a = omega / alpha (P),  k_b = omega / beta (S).
!
! It is a term-for-term transcription of the Python reference, cubic_scattering/graded_voxel/kernel.py
! (functions greens_tensors and kernel_9x9), and tests/test_kernel_fortran.py checks the two agree to
! round-off and both agree with a 50-digit evaluation. If you change one, change the other.
!
! THE RADIAL LADDER (the one idea everything is built on)
! -------------------------------------------------------
! g_b and f_b depend on x only through r = |x|. Every Cartesian derivative of such a radial function
! can be written with the "ladder"
!
!   F_q = ((1/r) d/dr)^q f,   q = 0..4,
!
! times polynomials in x and Kronecker deltas, for example
!
!   d_i f       = x_i F_1
!   d_i d_j f   = delta_ij F_1 + x_i x_j F_2
!
! (radial_component below has all four orders). So per point we need only ten complex numbers, F_0..F_4
! of g_b (called fa) and of f_b (called fb); the tensors follow from them.
!
! TWO WAYS OF GETTING THE LADDER, BECAUSE OF A CANCELLATION
! ---------------------------------------------------------
! f_b = (g_b - g_a) / k_b^2 is a difference of two functions that share the same 1/r singularity, so as
! k r -> 0 its closed form cancels and loses digits as eps / (k r)^2 (4e-7 relative at k r = 1e-4).
! Hence two branches, switching at |k_b| r = series_limit (0.5):
!
!  * small |k_b| r: the power series  e^{ikr}/r = sum_t (ik)^t r^(t-1) / t!.  For f_b the t = 0 term
!    (the shared singularity) vanishes identically and is simply never added, so nothing cancels. Each
!    power r^m has ladder F_q[r^m] = m (m-2) ... (m-2q+2) r^(m-2q) (function falling).
!
!  * large |k_b| r: the closed forms, with x = i k r,
!      F_q[e^{ikr}/r] = e^{ikr} p_q(x) / r^(2q+1),
!      p_0 = 1,  p_1 = x - 1,  p_2 = x^2 - 3x + 3,  p_3 = x^3 - 6x^2 + 15x - 15,
!      p_4 = x^4 - 10x^3 + 45x^2 - 105x + 105        (reverse Bessel polynomials; derived with sympy).
!
! omega may be complex (attenuation, the physical case), so the switch is on |k_b| r.
!
! STATIC AND DYNAMIC PARTS
! ------------------------
! The static (Kelvin) part is the t = 0 term of the delta series (1/r) and the t = 2 term of the
! derivative series (b2 r, b2 = -(1 - beta^2 / alpha^2) / 2). use_static / use_dynamic select either
! part or both, exactly as the Python flags static= and dynamic=.
!
! THE VOIGT CONTRACTION IS NOT CODED HERE
! ---------------------------------------
! How derivatives of G map onto C, H and S (index pairs, the factor of two on engineering shear strains)
! is the package's convention, defined once in Python (resonance_tmatrix._voigt_contract). Python probes it
! into three linear maps (graded_voxel.kernel.voigt_maps) and passes them in as mc, mh, ms, so the
! convention cannot drift between the two implementations. The maps are 97 per cent zeros, so they are
! turned into short index lists once, before the loop over points.
!
!-------------------------------------------------------------------------------------------------------
! A READING GUIDE TO THE FORTRAN USED HERE (Fortran 2008)
!-------------------------------------------------------------------------------------------------------
!  ! text                   a comment, to the end of the line.
!  &                        at the end of a line: the statement continues on the next line.
!  a = 1; b = 2             ; separates two statements on one line.
!  module ... contains      a module groups data and procedures; procedures follow "contains".
!  private / public :: x    only names listed public are visible outside the module (here: kernel_9x9).
!  implicit none            every variable must be declared (no implicit i-n integers). Always use it.
!  integer, parameter :: dp = kind(1.0d0)
!                           a named constant: the "kind" number of double precision. real(dp) is a
!                           64-bit real, complex(dp) a pair of them. 1.0_dp is the literal 1.0 in kind dp.
!                           (f2py needs .f2py_f2cmap to know that dp means double; without it it would
!                           silently pass single precision.)
!  intent(in) / intent(out) an argument that is only read / only written.
!  real(dp) :: x(3, n)      an array; indices start at 1 by default and run to 3 and to n.
!  complex(dp) :: f(0:4)    explicit bounds: this one is indexed f(0) .. f(4), like the ladder q = 0..4.
!  x(:, ip)                 a whole column (all of dimension 1, at ip). Arrays are stored column-major:
!                           the FIRST index varies fastest, the opposite of NumPy's default.
!  pw = fall(t, :) * rt     whole-array arithmetic: element by element, no loop needed.
!  [i, j, 0, 0]             an array constructor (a literal array).
!  do i = 1, 3 ... end do   a loop, both bounds inclusive. "cycle" skips to the next iteration.
!  select case (n) ... case (0) ... case default ... end select     a switch.
!  merge(a, b, cond)        cond ? a : b (both a and b are evaluated).
!  cmplx(re, im, dp)        build a complex(dp) value. x**n is a power.
!  pure                     the procedure has no side effects (needed to call it safely inside OpenMP).
!  function f(...) result(out)    the value returned is the variable "out".
!  subroutine s(...); call s(...) a procedure with no return value; arguments carry results out.
!  /= .and. .or. .not.      not-equal, and, or, not.
!
!  OpenMP (directives start with !$omp, so a compiler without OpenMP treats them as comments):
!  !$omp parallel do        split the next do-loop's iterations across threads.
!  default(none)            every variable used in the loop must be declared shared or private below
!                           (a guard: a forgotten scratch variable becomes a compile error, not a race).
!  shared(...)              one copy, seen by all threads: inputs read by everyone, and the output p,
!                           which is safe because each iteration writes only its own column p(:, :, ip).
!  private(...)             each thread gets its own copy: the loop index and every scratch variable.
!  schedule(static)         iterations are dealt out in fixed contiguous blocks.
!  Each point is computed by one thread, start to finish, with no sum across points, so the result is
!  bit-for-bit the same for any number of threads (tested).
!
!  Calling from Python: f2py turns kernel_9x9 into _point_kernel.point_kernel.kernel_9x9; the wrapper is
!  cubic_scattering/graded_voxel/kernel_fortran.py. Build: python -m cubic_scattering.fortran.build.
!=======================================================================================================

module point_kernel
  implicit none
  private
  public :: kernel_9x9

  integer, parameter :: dp = kind(1.0d0)                                ! double precision
  real(dp), parameter :: pi = 3.14159265358979323846264338327950288_dp

contains

  !-----------------------------------------------------------------------------------------------------
  ! falling(m, q) = m (m - 2) (m - 4) ... (m - 2q + 2)   (q factors; 1 when q = 0)
  !
  ! The ladder of a power: F_q[r^m] = falling(m, q) r^(m - 2q), because each (1/r) d/dr lowers the power
  ! by two and multiplies by the current exponent. Python: graded_voxel.kernel.falling.
  !-----------------------------------------------------------------------------------------------------
  pure function falling(m, q) result(out)
    integer, intent(in) :: m, q
    real(dp) :: out
    integer :: s
    out = 1.0_dp
    do s = 0, q - 1
      out = out * real(m - 2 * s, dp)            ! real(..., dp) converts the integer to double
    end do
  end function falling

  !-----------------------------------------------------------------------------------------------------
  ! closed_ladder: F_0 .. F_4 of e^{ikr}/r in closed form (the large-|k| r branch).
  !
  ! F_q = e^{ikr} p_q(ikr) / r^(2q+1), each polynomial p_q written in nested (Horner) form, which costs
  ! fewer multiplications and loses less to round-off than expanding the powers.
  ! Python: graded_voxel.kernel._closed_F_functions (the same functions, derived symbolically by sympy).
  !-----------------------------------------------------------------------------------------------------
  pure subroutine closed_ladder(k, r, f)
    complex(dp), intent(in) :: k              ! wavenumber, may be complex
    real(dp), intent(in) :: r                 ! distance, > 0
    complex(dp), intent(out) :: f(0:4)        ! the ladder F_0 .. F_4
    complex(dp) :: x, e
    x = cmplx(0.0_dp, 1.0_dp, dp) * k * r     ! x = i k r
    e = exp(x)
    f(0) = e / r
    f(1) = e * (x - 1.0_dp) / r**3
    f(2) = e * ((x - 3.0_dp) * x + 3.0_dp) / r**5
    f(3) = e * (((x - 6.0_dp) * x + 15.0_dp) * x - 15.0_dp) / r**7
    f(4) = e * ((((x - 10.0_dp) * x + 45.0_dp) * x - 105.0_dp) * x + 105.0_dp) / r**9
  end subroutine closed_ladder

  !-----------------------------------------------------------------------------------------------------
  ! radial_component: one component d_{idx(1)} ... d_{idx(n)} f of the derivative tensor of order n
  ! (0 <= n <= 4) of a radial function f, from its ladder f(0:4) and the separation x.
  !
  !   n = 0:  f
  !   n = 1:  x_i F_1
  !   n = 2:  delta_ij F_1 + x_i x_j F_2
  !   n = 3:  (delta_ij x_k + delta_ik x_j + delta_jk x_i) F_2 + x_i x_j x_k F_3
  !   n = 4:  (delta_ij delta_km + delta_ik delta_jm + delta_im delta_jk) F_2
  !         + (the six delta x x pairings) F_3 + x_i x_j x_k x_m F_4
  !
  ! Only the first n entries of idx are used. Python: graded_voxel.kernel.radial_component.
  !-----------------------------------------------------------------------------------------------------
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
  ! dl(a, b): the Kronecker delta, as a double (1 when a == b, else 0).
  !-----------------------------------------------------------------------------------------------------
  pure function dl(a, b) result(out)
    integer, intent(in) :: a, b
    real(dp) :: out
    out = merge(1.0_dp, 0.0_dp, a == b)
  end function dl

  !-----------------------------------------------------------------------------------------------------
  ! kernel_9x9: the propagator at n separations.
  !
  ! Arguments (all supplied by the Python wrapper, kernel_fortran.kernel_9x9_fortran):
  !   x(3, n)          separations x - x', one per column, in the package's (z, x, y) order; non-zero
  !                    (the wrapper rejects r = 0, where the propagator is a distribution).
  !   n                number of points (f2py fills it from the shape of x).
  !   omega            angular frequency, complex for an attenuating medium.
  !   alpha, beta, mu  P and S speeds and shear modulus of the background.
  !   use_static, use_dynamic
  !                    which parts to include (both: the full propagator).
  !   series_limit, n_series
  !                    the branch switch (0.5) and the number of series terms (24): kernel.SERIES_LIMIT
  !                    and kernel.N_SERIES, passed in so that the two implementations cannot disagree.
  !   mc, mh, ms       the Voigt maps: C(i, a) = sum_z mc(i, a, z) gd(z), and likewise for H and S.
  !   p(9, 9, n)       output: p(:, :, ip) is the block at point ip (the wrapper reorders it to (n, 9, 9)).
  !
  ! Layout of the work:
  !   1. once: the wavenumbers, the series coefficients, and the sparse index lists of the Voigt maps;
  !   2. per point, in parallel: the ladders fa and fb (series or closed form), then G, dG, ddG, then the
  !      Voigt contraction into p(:, :, ip).
  !-----------------------------------------------------------------------------------------------------
  subroutine kernel_9x9(x, n, omega, alpha, beta, mu, use_static, use_dynamic, series_limit, n_series, &
                        mc, mh, ms, p)
    integer, intent(in) :: n, n_series
    real(dp), intent(in) :: x(3, n), alpha, beta, mu, series_limit
    complex(dp), intent(in) :: omega
    logical, intent(in) :: use_static, use_dynamic
    real(dp), intent(in) :: mc(3, 6, 27), mh(6, 3, 27), ms(6, 6, 81)
    complex(dp), intent(out) :: p(9, 9, n)

    complex(dp), parameter :: iu = (0.0_dp, 1.0_dp)     ! the imaginary unit i
    integer, parameter :: max_c = 4                      ! most non-zero map weights for any one output

    ! Wavenumbers and the ladders. fa: of g_b (the delta part); fb: of f_b = (g_b - g_a) / k_b^2.
    ! ga, gb: scratch ladders (closed forms of g_a and g_b, then reused for the static parts).
    complex(dp) :: ka, kb, fa(0:4), fb(0:4), fa_tot(0:4), fb_tot(0:4), ga(0:4), gb(0:4)
    ! G (3 x 3), its first derivatives d_k G_ij (27) and second derivatives d_k d_l G_ij (81), the
    ! derivatives flattened in C (row-major) order of (i, j, k[, l]), because that is the order in which
    ! Python built the Voigt maps.
    complex(dp) :: gmat(3, 3), gd(27), gdd(81), acc
    ! Series coefficients of the delta part (a) and the difference part (b), term t = 0 .. n_series - 1:
    !   a_t = (i k_b)^t / t!,   b_t = ((i k_b)^t - (i k_a)^t) / (t! k_b^2).
    complex(dp) :: a_coef(0:n_series - 1), b_coef(0:n_series - 1)
    real(dp) :: r, xp(3), b2, fac, pw(0:4), rt, r2inv(0:4)
    real(dp) :: fall(0:n_series - 1, 0:4)                ! fall(t, q) = falling(t - 1, q)
    logical :: use_a(0:n_series - 1), use_b(0:n_series - 1)   ! which series terms are included
    ! Sparse Voigt maps: for output (i, a) of C, c_n(i, a) non-zero weights c_w(1:c_n, i, a) at entries
    ! c_z(1:c_n, i, a) of gd; likewise h_* for H (from gd) and s_* for S (from gdd).
    integer :: c_n(3, 6), c_z(max_c, 3, 6), h_n(6, 3), h_z(max_c, 6, 3), s_n(6, 6), s_z(max_c, 6, 6)
    real(dp) :: c_w(max_c, 3, 6), h_w(max_c, 6, 3), s_w(max_c, 6, 6)
    integer :: ip, t, q, i, j, k, l, a, b, z, idx(4), c

    ka = omega / alpha
    kb = omega / beta
    b2 = -(1.0_dp - beta**2 / alpha**2) / 2.0_dp        ! the static Kelvin coefficient of d_i d_j r

    ! ---- 1a. Series coefficients, which do not depend on the point. -------------------------------
    ! Which terms are kept (Python: use_a / use_b in kernel.greens_tensors):
    !   delta part:      t = 0 is static (1/r), t >= 1 dynamic;
    !   difference part: t = 0 and t = 1 never (t = 0 cancels exactly; t = 1 is a constant, which every
    !                    derivative removes), t = 2 static (b2 r), t >= 3 dynamic.
    fac = 1.0_dp
    do t = 0, n_series - 1
      if (t > 0) fac = fac * real(t, dp)                ! fac = t!
      a_coef(t) = (iu * kb)**t / fac
      b_coef(t) = ((iu * kb)**t - (iu * ka)**t) / (fac * kb**2)
      use_a(t) = merge(use_static, use_dynamic, t == 0)
      use_b(t) = merge(use_static, (t >= 2) .and. use_dynamic, t == 2)
      do q = 0, 4
        fall(t, q) = falling(t - 1, q)                  ! term t is the power r^(t - 1)
      end do
    end do

    ! ---- 1b. The Voigt maps as index lists of their non-zero weights. -----------------------------
    c_n = 0; h_n = 0; s_n = 0                            ! whole-array assignment: all counts to zero
    do a = 1, 6
      do i = 1, 3
        do z = 1, 27
          if (mc(i, a, z) /= 0.0_dp) then
            c_n(i, a) = c_n(i, a) + 1; c_z(c_n(i, a), i, a) = z; c_w(c_n(i, a), i, a) = mc(i, a, z)
          end if
          if (mh(a, i, z) /= 0.0_dp) then
            h_n(a, i) = h_n(a, i) + 1; h_z(h_n(a, i), a, i) = z; h_w(h_n(a, i), a, i) = mh(a, i, z)
          end if
        end do
      end do
      do b = 1, 6
        do z = 1, 81
          if (ms(a, b, z) /= 0.0_dp) then
            s_n(a, b) = s_n(a, b) + 1; s_z(s_n(a, b), a, b) = z; s_w(s_n(a, b), a, b) = ms(a, b, z)
          end if
        end do
      end do
    end do

    ! ---- 2. The points, in parallel. Each iteration reads only x(:, ip) and the shared tables, and
    ! writes only p(:, :, ip), so there is nothing to synchronise and no race.
    !$omp parallel do default(none) schedule(static) &
    !$omp shared(x, n, ka, kb, b2, use_static, use_dynamic, series_limit, n_series, mu, p, a_coef, b_coef, &
    !$omp        use_a, use_b, fall, c_n, c_z, c_w, h_n, h_z, h_w, s_n, s_z, s_w) &
    !$omp private(ip, r, xp, fa, fb, fa_tot, fb_tot, ga, gb, pw, rt, r2inv, t, q, &
    !$omp         gmat, gd, gdd, i, j, k, l, a, b, z, idx, c, acc)
    do ip = 1, n
      xp = x(:, ip)                                     ! this point's separation
      r = sqrt(xp(1)**2 + xp(2)**2 + xp(3)**2)
      fa = (0.0_dp, 0.0_dp)
      fb = (0.0_dp, 0.0_dp)

      if (abs(kb) * r <= series_limit) then
        ! ---- 2a. Small |k| r: the ladders from the series. ------------------------------------------
        ! The ladder is linear in the series, so the terms' ladders are summed and the tensors built once.
        ! Term t is the power r^(t - 1), with ladder fall(t, q) r^(t - 1 - 2q). Rather than calling **
        ! 120 times, r^(t - 1) is kept as a running product rt, and r^(-2q) is the fixed table r2inv(q).
        r2inv(0) = 1.0_dp
        do q = 1, 4
          r2inv(q) = r2inv(q - 1) / (r * r)
        end do
        rt = 1.0_dp / r                                 ! r^(t - 1) at t = 0
        do t = 0, n_series - 1
          if (use_a(t) .or. use_b(t)) then
            pw = fall(t, :) * rt * r2inv                ! pw(q) = ladder of r^(t - 1), q = 0..4
            if (use_a(t)) fa = fa + a_coef(t) * pw
            if (use_b(t)) fb = fb + b_coef(t) * pw
          end if
          rt = rt * r
        end do
      else
        ! ---- 2b. Large |k| r: the closed forms. ---------------------------------------------------
        call closed_ladder(kb, r, gb)
        call closed_ladder(ka, r, ga)
        fa_tot = gb
        fb_tot = (gb - ga) / kb**2                      ! safe here: |k| r > 0.5, the loss is at most 4x
        if (use_static .and. use_dynamic) then
          fa = fa_tot
          fb = fb_tot
        else
          ! The static Kelvin part, to keep or to remove: the ladder of 1/r for the delta part and of
          ! b2 r for the difference part. (ga and gb are reused as scratch from here.)
          do q = 0, 4
            pw(q) = falling(-1, q) * r**(-1 - 2 * q)
          end do
          ga = pw
          do q = 0, 4
            pw(q) = b2 * falling(1, q) * r**(1 - 2 * q)
          end do
          gb = pw
          if (use_static) then
            fa = ga
            fb = gb
          else if (use_dynamic) then
            fa = fa_tot - ga
            fb = fb_tot - gb
          end if
        end if
      end if

      ! ---- 2c. G, d_k G_ij and d_k d_l G_ij from the two ladders. -----------------------------------
      !   G_ij        = delta_ij fa      + d_i d_j fb
      !   d_k G_ij    = delta_ij d_k fa  + d_i d_j d_k fb
      !   d_k d_l G_ij = delta_ij d_k d_l fa + d_i d_j d_k d_l fb
      ! (the 1 / 4 pi mu is applied at the end). z is the C-order flat index of (i, j, k) or (i, j, k, l),
      ! counting from 1: Python's reshape of a (3, 3, 3) array to 27, plus one.
      do i = 1, 3
        do j = 1, 3
          idx = [i, j, 0, 0]
          gmat(i, j) = dl(i, j) * fa(0) + radial_component(fb, xp, idx, 2)
          do k = 1, 3
            z = ((i - 1) * 3 + (j - 1)) * 3 + k
            idx = [k, 0, 0, 0]
            gd(z) = dl(i, j) * radial_component(fa, xp, idx, 1)
            idx = [i, j, k, 0]
            gd(z) = gd(z) + radial_component(fb, xp, idx, 3)
            do l = 1, 3
              z = (((i - 1) * 3 + (j - 1)) * 3 + (k - 1)) * 3 + l
              idx = [k, l, 0, 0]
              gdd(z) = dl(i, j) * radial_component(fa, xp, idx, 2)
              idx = [i, j, k, l]
              gdd(z) = gdd(z) + radial_component(fb, xp, idx, 4)
            end do
          end do
        end do
      end do

      ! ---- 2d. Assemble [[G, C], [H, S]] with the sparse Voigt maps, then scale by 1 / (4 pi mu). -----
      p(:, :, ip) = (0.0_dp, 0.0_dp)
      p(1:3, 1:3, ip) = gmat                            ! the G block: rows 1-3, columns 1-3
      do a = 1, 6
        do i = 1, 3
          acc = (0.0_dp, 0.0_dp)
          do c = 1, c_n(i, a)                           ! C_i,a: displacement i from stress a
            acc = acc + c_w(c, i, a) * gd(c_z(c, i, a))
          end do
          p(i, 3 + a, ip) = acc
          acc = (0.0_dp, 0.0_dp)
          do c = 1, h_n(a, i)                           ! H_a,i: strain a from force i
            acc = acc + h_w(c, a, i) * gd(h_z(c, a, i))
          end do
          p(3 + a, i, ip) = acc
        end do
        do b = 1, 6
          acc = (0.0_dp, 0.0_dp)
          do c = 1, s_n(a, b)                           ! S_a,b: strain a from stress b
            acc = acc + s_w(c, a, b) * gdd(s_z(c, a, b))
          end do
          p(3 + a, 3 + b, ip) = acc
        end do
      end do
      p(:, :, ip) = p(:, :, ip) / (4.0_dp * pi * mu)
    end do
    !$omp end parallel do
  end subroutine kernel_9x9

end module point_kernel
