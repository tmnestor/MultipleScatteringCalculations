      SUBROUTINE JACOBI( N, B, X, WORK, LDW, ITER, RESID, MATVEC, 
     $                   INFO )
*
*  -- Iterative template routine --
*     Univ. of Tennessee and Oak Ridge National Laboratory
*     October 1, 1993
*     Details of this algorithm are described in "Templates for the 
*     Solution of Linear Systems: Building Blocks for Iterative 
*     Methods", Barrett, Berry, Chan, Demmel, Donato, Dongarra, 
*     Eijkhout, Pozo, Romine, and van der Vorst, SIAM Publications,
*     1993. (ftp netlib2.cs.utk.edu; cd linalg; get templates.ps).
*
*     .. Scalar Arguments ..
      INTEGER            N, LDW, ITER, INFO
      DOUBLE PRECISION   RESID
*     ..
*     .. Array Arguments ..
      DOUBLE PRECISION   B( * ), X( * ), WORK( LDW,* )
*     ..
*     .. Function Arguments ..
      EXTERNAL           MATVEC
*
*  Purpose
*  =======
*
*  JACOBI solves the linear system Ax = b using the Jacobi iterative 
*  method. The matrix splitting should be accomplished before calling
*  this routine. The diagonal elements of the matrix must be passed into
*  this routine in the first column of matrix WORK. 
*
*  Relative error measured: norm( X - X_1 ) / norm( X ).
*
*  Arguments
*  =========
*
*  N       (input) INTEGER 
*          On entry, the dimension of the matrix.
*          Unchanged on exit.
*
*  B       (input) DOUBLE PRECISION array, dimension N
*          On entry, right hand side vector B.
*          Unchanged on exit.
*
*  X       (input/output) DOUBLE PRECISION array, dimension N.
*          On input, the initial guess. This is commonly set to
*          the zero vector.
*          On exit, if INFO = 0, the iterated approximate solution.
*
*  WORK    (workspace) DOUBLE PRECISION array, dimension (LDW,3)
*          Workspace for residual, direction vector, etc.
*          
*  LDW     (input) INTEGER
*          The leading dimension of the array WORK. LDW >= max(1,N).
*
*  ITER    (input/output) INTEGER
*          On input, the maximum iterations to be performed.
*          On output, actual number of iterations performed.
*
*  RESID   (input/output) DOUBLE PRECISION
*          On input, the allowable convergence measure for
*          norm( x - x_1 ) / norm( x ).
*          On output, the final value of this measure.
*
*  MATVEC  (external subroutine)
*          The user must provide a subroutine to perform the
*          matrix-vector product
*
*               y := alpha*A*x + beta*y,
*
*          where alpha and beta are scalars, x and y are vectors,
*          and A is a matrix. Vector x must remain unchanged.
*          The solution is over-written on vector y.
*
*          The call is:
*
*             CALL MATVEC( ALPHA, X, BETA, Y )
*
*          The matrix is passed into the routine in a common block.
*
*  INFO    (output) INTEGER
*
*          =  0: Successful exit. Iterated approximate solution returned.
*
*          >  0: Convergence to tolerance not achieved. This will be
*                set to the number of iterations performed.
*
*          <  0: Illegal input parameter, or breakdown occurred
*                during iteration.
*
*                Illegal parameter:
*
*                   -1: matrix dimension N < 0
*                   -2: LDW < N
*                   -3: Maximum number of iterations ITER <= 0.
*
*  BLAS CALLS:   DAXPY, DCOPY, DNRM2
*  ==========================================================
*
*     .. Parameters ..
      DOUBLE PRECISION   ONE
      PARAMETER        ( ONE = 1.0D+0 )
*     ..
*     .. Local Scalars ..
      INTEGER            I, MAXIT, X1, MM, TEMP
      DOUBLE PRECISION   TOL, DNRM2
*     ..
*     .. External Routines ..
*
      EXTERNAL           DAXPY, DCOPY, DNRM2
*     ..
*     .. Executable Statements ..
*
      INFO = 0
*
*     Test the input parameters.
*
      IF ( N.LT.0 ) THEN
         INFO = -1
      ELSE IF ( LDW.LT.MAX( 1, N ) ) THEN
         INFO = -2
      ELSE IF ( ITER.LE.0 ) THEN
         INFO = -3
      ENDIF
      IF ( INFO.NE.0 ) RETURN
*
      MAXIT = ITER
      TOL   = RESID
*
*     Alias workspace columns.
*
      MM   = 1
      X1   = 2
      TEMP = 3
*
      ITER = 0
*
*     Form matrix splitting inv(M) and N.
*
      CALL MATSPLIT( ONE, B, WORK( 1,MM ), LDW,'JACOBI','SPLIT' )
*
   10 CONTINUE
*
*        Perform Jacobi iteration
*
         ITER = ITER + 1
*
*        Save the current approximation to X in X1.
*
         CALL DCOPY( N, X, 1, WORK( 1,X1 ),  1 )
*
*        Apply iteration; result is updated approximation vector x.
*
         CALL DCOPY( N, B, 1, WORK( 1,TEMP ), 1 )
         CALL MATVEC( ONE, X, ONE, WORK( 1,TEMP ) )
         DO 15 I = 1, N
            X( I ) = WORK( I,MM ) *  WORK( I,TEMP )
   15    CONTINUE
*
*        Compute error and check for acceptable convergence.
*
         CALL DAXPY( N, -ONE, X, 1, WORK( 1,X1 ), 1 )
         RESID = DNRM2( N, WORK( 1,X1 ), 1 ) / DNRM2( N, X, 1 )
*
         IF ( RESID.LE.TOL  ) GO TO 30
         IF ( ITER.EQ.MAXIT ) GO TO 20
*
      GO TO 10
*
   20 CONTINUE
*
*     Iteration fails
*
      INFO = ITER
      GO TO 30
*
   30 CONTINUE 
*
*     Iteration successful. Reconstruct matrix A.
*
      CALL MATSPLIT( ONE, B, WORK( 1,MM ), LDW, 'JACOBI','RECONSTRUCT' )
*
      RETURN
*
*     End of JACOBI
*
      END







****************************************************************************************
      subroutine born(N, B, X, Xold , ITER, RESID, INFO,
     &            Nl,NLm1,Nk,Nscatx,Nscatz,jskip,
     &            R11,R12,R21,R22,S11,S12,S21,S22,
     &            U11,U12,U21,U22,T11,T12,T21,T22,
     &            P1,P2,Q1,Q2,PC,SY,PSY,PD,SD,PU,SU,Sigp,SigS,
     &            ea,eb,PP11,PP22,PP31,PP33,PP51,PS11,PS51,
     &            PS22,PS31,SS11,SS51,SS22)
	 
	  Implicit    none
      INTEGER     N, ITER, INFO
      Real*8      RESID
      Complex*16  B( N ), X( N ), Xold( N )



      INTEGER     Nl,Nlm1,Nk,Nscatx,Nscatz,jskip
	  
      Complex*16  PC(Nk,-2:2,2,Nscatz)
 
      Complex*16  T11(Nk,1:Nlm1),T12(Nk,1:Nlm1)
      Complex*16  T21(Nk,1:Nlm1),T22(Nk,1:Nlm1)
      Complex*16  U11(Nk,0:Nlm1),U12(Nk,0:Nlm1)
      Complex*16  U21(Nk,0:Nlm1),U22(Nk,0:Nlm1)
      Complex*16  R11(Nk,1:Nl),R12(Nk,1:Nl)
      Complex*16  R21(Nk,1:Nl),R22(Nk,1:Nl)
      Complex*16  S11(Nk,1:Nlm1),S12(Nk,1:Nlm1)
      Complex*16  S21(Nk,1:Nlm1),S22(Nk,1:Nlm1)
      Complex*16  P1(Nk,1:Nl),P2(Nk,1:Nl)
      Complex*16  Q1(Nk,1:Nl),Q2(Nk,1:Nl)

      Complex*16  PD(Nk),SD(Nk),PU(Nk),SU(Nk)
      Complex*16  sigP(Nk),sigS(Nk)
      Complex*16  ea(Nk,1:Nl),eb(Nk,1:Nl)
	 
	  Complex*16  PP11(Nscatx,Nscatz),PP22(Nscatx,Nscatz),PS31(Nscatx,Nscatz)
	  Complex*16  PP31(Nscatx,Nscatz),PP33(Nscatx,Nscatz),PP51(Nscatx,Nscatz)
	  Complex*16  PS11(Nscatx,Nscatz),PS51(Nscatx,Nscatz),PS22(Nscatx,Nscatz)
	  Complex*16  SS11(Nscatx,Nscatz),SS51(Nscatx,Nscatz),SS22(Nscatx,Nscatz)

      Complex*16  SY(Nscatx,-2:2,2,Nscatz),PSY(Nscatx,-2:2,2,Nscatz)

      integer     iter,iscatx,iscatz,j,m
	  real*8      nrm1,nrm2,err,errb
	  


*
*  Purpose
*  =======
*
*  Born solves the linear system x = b + PS x using the Born iterative 
*  method.  
*
*  Relative error measured: norm( X - X_1 ) / norm( X ).
*
*  Arguments
*  =========
*
*  N       (input) INTEGER 
*          On entry, the dimension of the matrix.
*          Unchanged on exit.
*
*  B       (input) Complex array, dimension N
*          On entry, right hand side vector B.
*          Unchanged on exit.
*
*  X       (input/output) Complex array, dimension N.
*          On input, the initial guess. This is commonly set to
*          the zero vector.
*          On exit, if INFO = 0, the iterated approximate solution.
*
*  Xold    (workspace) Complex  vector, dimension ( N )
*          Workspace for residual, direction vector, etc.
*
*  ITER    (input/output) INTEGER
*          On input, the maximum iterations to be performed.
*          On output, actual number of iterations performed.
*
*  RESID   (input/output) Real*8                  
*          On input, the allowable convergence measure for
*          norm( x - x_1 ) / norm( x ).
*          On output, the final value of this measure.
*
*  INFO    (output) INTEGER
*
*          =  0: Successful exit. Iterated approximate solution returned.
*
*          >  0: Convergence to tolerance not achieved. This will be
*                set to the number of iterations performed.
*
*          <  0: Illegal input parameter, or breakdown occurred
*                during iteration.
*
*                Illegal parameter:
*
*                   -1: matrix dimension N < 0
*                   -3: Maximum number of iterations ITER <= 0.
*
*  BLAS CALLS:   ZAXPY, ZCOPY, DZNRM2
*  ==========================================================
*
*     .. Parameters ..
      Complex*16         ONE
      PARAMETER        ( ONE = ( 1.0D+0 , 0.D0 ) )
*     ..
*     .. Local Scalars ..
      INTEGER            I, MAXIT, X1, MM, TEMP
      Real*8             TOL, DZNRM2                  
*     ..
*     .. External Routines ..
*
      EXTERNAL           ZAXPY, ZCOPY, DZNRM2                  
*     ..
*     .. Executable Statements ..
*
      INFO = 0
*
*     Test the input parameters.
*
      IF ( N.LT.0 ) THEN
         INFO = -1
      ELSE IF ( ITER.LE.0 ) THEN
         INFO = -3
      ENDIF
      IF ( INFO.NE.0 ) RETURN
*
      MAXIT = ITER
      TOL   = RESID
*
*     Alias workspace columns.
*
*
      ITER = 0

*
*     X=B.
*
   CALL ZCOPY( N, B, 1, X , 1 )
*
   10 CONTINUE
*
*        Perform Jacobi iteration
*
         ITER = ITER + 1
*
*        Save the current approximation to X in Xold.
*
         CALL ZCOPY( N, X, 1, Xold ,  1 )
*
*        Apply iteration; result is updated approximation vector x.
*

*        set SY=S*X
	     call setSY(Nscatx,Nscatz,PP11,PP22,PP31,
     &     PP33,PP51,PS11,PS51,PS22,PS31,SS11,
     &     SS51,SS22,X,SY)

*        set PSY=P*SY
         call setPSY(Nl,Nlm1,Nk,Nscatx,Nscatz,jskip,
     &           R11,R12,R21,R22,S11,S12,S21,S22,
     &           U11,U12,U21,U22,T11,T12,T21,T22,
     &           ea,eb,PD,SD,PU,SU,sigP,sigS,P1,P2,Q1,Q2,PC,SY,PSY)

         DO 15 I = 1, N
            X( I ) = B( I ) +  PSY( I )
   15    CONTINUE

*
*        Compute error and check for acceptable convergence.
*
         CALL ZAXPY( N, -ONE, X, 1, Xold , 1 )
         RESID = DZNRM2( N, Xold , 1 ) / DZNRM2( N, X, 1 )
		 
		 type*,'iter,resid=',iter,resid
*
         IF ( RESID.LE.TOL  ) GO TO 30
         IF ( ITER.EQ.MAXIT ) GO TO 20
*
      GO TO 10
*
   20 CONTINUE
*
*     Iteration fails
*
      INFO = ITER
      GO TO 30
*
   30 CONTINUE 

      RETURN
*
*     End of JACOBI
*
      END
	  
	  
