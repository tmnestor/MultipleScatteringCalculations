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
!-------------------------------------------------------------------------------------------------------

module kernel_apply
  implicit none
  private
  public :: apply_hat_c64, apply_hat_c128

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

end module kernel_apply
