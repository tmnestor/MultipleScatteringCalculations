!=======================================================================================================
! kernel_apply.f90: the transformed kernel applied to the transformed sources, in OpenMP over the Fourier
!                   points (the product at the heart of every GMRES step of the graded-voxel FFT solve)
!=======================================================================================================
!
! WHAT IT COMPUTES
! ----------------
!   yh(f, r) = sum_c kh(f, c, r) sh(f, c)      for every Fourier point f and kernel row r,
!
! kh the transformed kernel as the solve stores it (graded_voxel.fft.transformed_kernel), sh the
! transformed sources. Python holds kh as kh[row, col, i, j, k] in C order, which is exactly
! kh(f, c, r) in Fortran order with f = (i, j, k) flattened: the wrapper passes it as it lies, without a
! copy, which matters because it is 6.5 GB at N = 32 in single precision and may be memory-mapped from
! disk. The sum over c is accumulated in double precision whatever the kernel's precision, as the NumPy
! product it replaces (einsum over a complex64 kernel and complex128 sources) does.
!
! WHY THIS STEP IS IN FORTRAN (21 October 2026)
! ---------------------------------------------
! The product was a single-threaded einsum, row by row, over the whole kernel: about 3.4 s per GMRES step
! at N = 32 with the kernel in memory (9 s from disk), 6 steps per frequency. The arithmetic is a few
! operations per kernel entry, so the step is bound by reading the kernel: one pass, in order, by all
! threads, is the most it can cost. Each Fourier point is summed by one thread over c in order, so the
! result does not depend on the number of threads.
!
! THE KERNEL AT THE CANONICAL k ONLY (apply_canon_*, item 9 of the speed-ups plan)
! --------------------------------------------------------------------------------
! The cube group acts on Fourier space as on the offsets: K^(Q k) = R K^(k) R^T, R the signed
! permutation of Q on the rows (pr, sr) and on the columns (pc, sc). So only the canonical k are stored,
! kc(r, c, k) for the C(N + 2, 3) orbits (1/42 of the points at N = 32), and at a point f = Q k_c
!
!     s(pc(c)) = sc(c) sh(c, f),    z = kc(:, :, k_c) s,    yh(r, f) = sr(r) z(pr(r)).
!
! The arithmetic is that of apply_hat; the kernel read falls by the orbit size. The sources and results
! are point-major, sh(c, f) and yh(r, f), so each point's values are contiguous, and the points are
! visited in ``order``, grouped by orbit, so that each canonical block (51 KB in double precision) stays
! in cache across its orbit. Each point is summed by one thread over c in order: the result does not
! depend on the number of threads. The tables are 0-based, as Python builds them.
!-------------------------------------------------------------------------------------------------------

module kernel_apply
  implicit none
  private
  public :: apply_hat_c64, apply_hat_c128, apply_canon_c64, apply_canon_c128

  integer, parameter :: sp = kind(1.0), dp = kind(1.0d0)
  integer, parameter :: block = 2048          ! Fourier points per unit of parallel work

contains

  ! kh in single precision (complex64): the production path
  subroutine apply_hat_c64(kh, nf, nc, nr, sh, yh)
    integer, intent(in) :: nf, nc, nr
    complex(sp), intent(in) :: kh(nf, nc, nr)
    complex(dp), intent(in) :: sh(nf, nc)
    complex(dp), intent(out) :: yh(nf, nr)
    integer :: f0, f1, r, c
    !$omp parallel do default(none) schedule(static) shared(kh, sh, yh, nf, nc, nr) private(f0, f1, r, c)
    do f0 = 1, nf, block
      f1 = min(nf, f0 + block - 1)
      do r = 1, nr
        yh(f0:f1, r) = (0.0_dp, 0.0_dp)
        do c = 1, nc
          yh(f0:f1, r) = yh(f0:f1, r) + cmplx(kh(f0:f1, c, r), kind=dp) * sh(f0:f1, c)
        end do
      end do
    end do
    !$omp end parallel do
  end subroutine apply_hat_c64

  ! kh in double precision (complex128)
  subroutine apply_hat_c128(kh, nf, nc, nr, sh, yh)
    integer, intent(in) :: nf, nc, nr
    complex(dp), intent(in) :: kh(nf, nc, nr)
    complex(dp), intent(in) :: sh(nf, nc)
    complex(dp), intent(out) :: yh(nf, nr)
    integer :: f0, f1, r, c
    !$omp parallel do default(none) schedule(static) shared(kh, sh, yh, nf, nc, nr) private(f0, f1, r, c)
    do f0 = 1, nf, block
      f1 = min(nf, f0 + block - 1)
      do r = 1, nr
        yh(f0:f1, r) = (0.0_dp, 0.0_dp)
        do c = 1, nc
          yh(f0:f1, r) = yh(f0:f1, r) + kh(f0:f1, c, r) * sh(f0:f1, c)
        end do
      end do
    end do
    !$omp end parallel do
  end subroutine apply_hat_c128

  ! the canonical kernel in single precision, accumulated in double
  subroutine apply_canon_c64(kc, nr, nc, nk, sh, nf, qf, kf, order, pr, sr, pc, sc, nq, yh)
    integer, intent(in) :: nr, nc, nk, nf, nq
    complex(sp), intent(in) :: kc(nr, nc, nk)
    complex(dp), intent(in) :: sh(nc, nf)
    integer, intent(in) :: qf(nf), kf(nf), order(nf), pr(nr, nq), pc(nc, nq)
    real(dp), intent(in) :: sr(nr, nq), sc(nc, nq)
    complex(dp), intent(out) :: yh(nr, nf)
    complex(dp) :: s(nc), z(nr)
    integer :: i, f, q, k, c
    !$omp parallel do default(none) schedule(static) &
    !$omp   shared(kc, sh, qf, kf, order, pr, sr, pc, sc, yh, nr, nc, nf) private(i, f, q, k, c, s, z)
    do i = 1, nf
      f = order(i) + 1
      q = qf(f) + 1
      k = kf(f) + 1
      do c = 1, nc
        s(pc(c, q) + 1) = sc(c, q) * sh(c, f)
      end do
      z = (0.0_dp, 0.0_dp)
      do c = 1, nc
        z = z + cmplx(kc(:, c, k), kind=dp) * s(c)
      end do
      yh(:, f) = sr(:, q) * z(pr(:, q) + 1)
    end do
    !$omp end parallel do
  end subroutine apply_canon_c64

  ! the canonical kernel in double precision: the production path (0.3 GB at N = 32)
  subroutine apply_canon_c128(kc, nr, nc, nk, sh, nf, qf, kf, order, pr, sr, pc, sc, nq, yh)
    integer, intent(in) :: nr, nc, nk, nf, nq
    complex(dp), intent(in) :: kc(nr, nc, nk)
    complex(dp), intent(in) :: sh(nc, nf)
    integer, intent(in) :: qf(nf), kf(nf), order(nf), pr(nr, nq), pc(nc, nq)
    real(dp), intent(in) :: sr(nr, nq), sc(nc, nq)
    complex(dp), intent(out) :: yh(nr, nf)
    complex(dp) :: s(nc), z(nr)
    integer :: i, f, q, k, c
    !$omp parallel do default(none) schedule(static) &
    !$omp   shared(kc, sh, qf, kf, order, pr, sr, pc, sc, yh, nr, nc, nf) private(i, f, q, k, c, s, z)
    do i = 1, nf
      f = order(i) + 1
      q = qf(f) + 1
      k = kf(f) + 1
      do c = 1, nc
        s(pc(c, q) + 1) = sc(c, q) * sh(c, f)
      end do
      z = (0.0_dp, 0.0_dp)
      do c = 1, nc
        z = z + kc(:, c, k) * s(c)
      end do
      yh(:, f) = sr(:, q) * z(pr(:, q) + 1)
    end do
    !$omp end parallel do
  end subroutine apply_canon_c128

end module kernel_apply
