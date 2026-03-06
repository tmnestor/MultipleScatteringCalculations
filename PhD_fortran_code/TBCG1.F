      Program  TBCG
      Implicit none
      Integer     Nl,Np,N,N1,il,ilp1,i,j,ip,ioff,itype
      Parameter   (Np=3) !must be odd
      Parameter   (Nl=2,N1=Nl+1,N=4*Nl*Np)
      Real*8      rho0(0:N1),rho(0:N1),drho(Nl),QB(0:N1)
      complex*16  alpha0(0:N1),beta0(0:N1),alpha(0:N1),beta(0:N1)
      complex*16  dalpha(Nl),dbeta(Nl)
      complex*16  w,p(Np)
      Real*8      dh(0:Nl),Zs,Xs,Xr,pmax,dp,rt3
      complex*16  CV
      Integer  is  !is=1=>Exp; is=2=>Doub
      Integer  ir  !ir=1=>X-disp; ir=2=>Z-disp
      logical     exact
      complex*16  resp(Np),XSY,XY !DELETE
*******************************************************
      Complex*16  Idvec(N),Amat(N,N),Vexact(N)  !DELETE
      INTEGER     IPIV(N),INDXR(N),INDXC(N)
******************************************************

*     TBCG declarations
      INTEGER      ITER, INFO
      parameter    (Iter=N)
      real*8       RESID
      parameter    (resid=1.D-8)
      complex*16   X1(4*Nl)
      complex*16   A(N,N),B( N ), X( N )
      complex*16   WORK(N,3)

******************************************************
      DATA     beta0/1.01,1.51,2.51,3.51/	
      DATA     rho0/1.01,1.51,2.51,3.51/
      DATA     QB/4*1.E10/
      DATA     drho/0.2,0.2/
      DATA     dbeta/0.2,0.2/
      DATA     Zs,Xs,Xr/0.25,0.0,10.0/
      DATA     is,ir/2,1/
      DATA     pmax/1.0/
      DATA     W,dh/1.0,0.5,.1,.1/
      DATA     exact/.true./

      dp=2.0*pmax/float(Np-1)
      Do ip=1,Np
        p(ip)=float(ip-1)*dp-pmax
        type*,'ip,p(ip)=',ip,p(ip)
      Enddo

      rt3=Sqrt(3.0)
      rho(0)=rho0(0)
      rho(NL+1)=rho0(NL+1)
      beta0(0)=CV(w,beta0(0),QB)
      beta(0)=beta0(0)
      beta0(Nl+1)=CV(w,beta0(Nl+1),QB)
      beta(Nl+1)=beta0(Nl+1)
      alpha0(0)=rt3*beta0(0)
      alpha(0)=alpha0(0)
      alpha0(Nl+1)=rt3*beta0(Nl+1)
      alpha(Nl+1)=alpha0(Nl+1)
      Do il=1,Nl
        beta0(il)=CV(w,beta0(il),QB)
        beta(il)=beta0(il)+CV(w,dbeta(il),QB)
        alpha0(il)=rt3*beta0(il)
        alpha(il)=rt3*beta(il)
        rho(il)=rho0(il)+drho(il)
      Enddo

      Do il=0,Nl+1
        TYpe*,'il,rho0(il)=',il,rho0(il)
      Enddo

      Do il=0,Nl+1
        TYpe*,'il,alpha0(il)=',il,alpha0(il)
      Enddo

      Do il=0,Nl+1
        TYpe*,'il,beta0(il)=',il,beta0(il)
      Enddo

      Do il=0,Nl+1
        TYpe*,'il,rho(il)=',il,rho(il)
      Enddo

      Do il=0,Nl+1
        TYpe*,'il,alpha(il)=',il,alpha(il)
      Enddo

      Do il=0,Nl+1
        TYpe*,'il,beta(il)=',il,beta(il)
      Enddo

      Do ip=1,Np
          call setInc(Nl,p(ip),w,dh,Zs,rho0,alpha0,beta0,
     &       rho,alpha,beta,Resp(ip),X1)
        Do il=1,Nl
          ioff=4*(il-1)
           Do itype=1,4
             B(4*Np*(il-1)+Np*(itype-1)+ip)=X1(ioff+itype)
           Enddo !itype
         Enddo !il
*        Type*,'Resp=',Resp(ip)
      Enddo !ip

	  Type*,'Incident field='
	  Do i=1,N
	    Type*,'i,B(i)=',i,B(i)
	  Enddo
*      pause
*********************************************
*     generate exact solution
**********************************************
      if(exact) then
*     set Amat(N,N),
      Do j=1,N
        Do i=1,N
          Idvec(i)=(0.0,0.0)
        Enddo !i
        Idvec(j)=(1.0,0.0)
        call GEMVmask(Nl,Np,w,p,dh,rho0,alpha0,beta0,
     &             rho,alpha,beta,.false.,Idvec,Amat(1,j))
      Enddo !j

	  Do i=1,N
	  	Do j=1,N
          A(i,j)=Amat(i,j)
*		  Type*,'i,j,Amat(i,j)=',i,j,Amat(i,j)
		Enddo !i
      Enddo!j

      Do i=1,N
        Vexact(i)=b(i)
      Enddo !i

      call CMPLXGJ(Amat,N,N,Vexact,1,1,IPIV,INDXR,INDXC,N)

      Do i=1,N
        Type*,'i,Vexact(i)=',i,Vexact(i)
      Enddo !i
      endif
**************************************************************
      Do i=1,N
        X(i)=(0.0,0.0) !BORN trial solution
      Enddo
      call BICG( N,A, B, X, WORK, ITER, RESID,INFO,Nl,
     &      Np,w,p,dh,rho0,alpha0,beta0,rho,alpha,beta )
      type*,'Info,iter,resid=',info,iter,resid
      Do i=1,N
        TYpe*, 'X(i)=',X(i)
        TYpe*
      Enddo !i

      if(exact) then
      Do i=1,N
        TYpe*, 'Cabs(Vexact(i)-X(i))/Cabs(Vexact(i))=',
     &          i,Cabs(Vexact(i)-X(i))/Cabs(Vexact(i))
        TYpe*
      Enddo !i
      endif

      stop'normal completion'
      end

******************************************************************
      SUBROUTINE BICG( N,A, B, X, WORK, ITER, RESID,INFO,
     &           Nl,Np,w,pvec,dh,den0,alf0,bet0,den,alf,bet )

*     .. Scalar Arguments ..
      IMPLICIT NONE
      INTEGER            N, ITER, INFO
      Real*8             RESID
*     ..
*     .. Array Arguments ..
      Complex*16         A(N,N), X( N ), B( N ), WORK( N,* )
****************************************************************
      Integer     Nl,Np,k
      Real*8      den0(0:Nl),den(0:Nl)
      complex*16  alf0(0:Nl),bet0(0:Nl)
      complex*16  alf(0:Nl),bet(0:Nl)	  
      Real*8      dh(0:Nl)
      complex*16  w,pvec(Np)
****************************************************************


*     ..
*
*  Purpose
*  =======
*
*  BiCG solves the linear system Ax = b using the
*  BiConjugate Gradient iterative method with preconditioning.
*
*  Convergence test: ( norm( b - A*x ) / norm( b ) ) < TOL.
*  For other measures, see the above reference.
*
*  Arguments
*  =========
*
*  N       (input) INTEGER. 
*          On entry, the dimension of the matrix.
*          Unchanged on exit.
*
*  B       (input) Complex*16 array, dimension N.
*          On entry, right hand side vector B.
*          Unchanged on exit.
*
*  X      (input/output) Complex*16 array, dimension N.
*          On input, the initial guess; on exit, the iterated solution.
*
*  WORK    (workspace) Complex*16 array, dimension (N,6).
*          Workspace for residual, direction vector, etc.
*          Note that Z and Q, and ZTLD and QTLD share workspace.
*
*  N     (input) INTEGER
*          The leading dimension of the array WORK. N >= 1.
*
*  ITER    (input/output) INTEGER
*          On input, the maximum iterations to be performed.
*          On output, actual number of iterations performed.
*
*  RESID   (input/output) Real*8
*          On input, the allowable convergence measure for
*          norm( b - A*x ) / norm( b ).
*          On output, the final value of this measure.
*
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
*                BREAKDOWN: If parameters RHO or OMEGA become smaller
*                   than some tolerance, the program will terminate.
*                   Here we check against tolerance BREAKTOL.
*
*                  -10: RHO < BREAKTOL: RHO and RTLD have become
*                                       orthogonal.
*
*                  BREAKTOL is set in function GETBREAK.
*
*  BLAS CALLS:     ZAXPY, ZCOPY, ZDOTU, DZNRM2
*  ==============================================================
*
*     .. Parameters ..
      Implicit none
      Complex*16             ZERO, ONE
      PARAMETER  ( ZERO = (0.0D+0,0.0D+0), ONE = (1.0D+0,0.0D+0) )
*     ..
*     .. Local Scalars ..
      INTEGER            R, RTLD, Z, ZTLD, P, PTLD, Q, QTLD, MAXIT
      Real*8             TOL, RHOTOL, GETBREAK, DZNRM2, BNRM2 
      Complex*16         ALPHA, BETA, RHO, RHO1, ZDOTU,XSY
*     ..
*     .. External Routines ..
      EXTERNAL           ZAXPY, ZCOPY, ZDOTU, DZNRM2, XSY
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
      R    = 1
      Z    = 2
      P    = 3
      Q    = 2

*
*     Set breakdown parameters.
*
      RHOTOL = GETBREAK()
* 
*     Set initial residual.
*
*     r0=b
      CALL ZCOPY( N, B, 1, WORK(1,R), 1 )
*     BNRM2 = ||B||2
      BNRM2 = DZNRM2( N, B, 1 )
      IF ( BNRM2.EQ.ZERO ) BNRM2 = 1.D0
*
      ITER = 0
*
   10 CONTINUE
*
*     Perform BiConjugate Gradient iteration.
*
         ITER = ITER + 1
		 type*,'iter=',iter
*
*        Compute direction vectors PK.
*

         CALL ZCOPY( N, WORK(1,R), 1, WORK(1,Z), 1 )

*        rho=z.rtld
         RHO = XSY(Nl,Np,w,pvec,dh,den0,alf0,bet0,den,
     &                 alf,bet,WORK(1,Z), WORK(1,R))
	      type*,'rho=',rho
         IF ( ABS( RHO ).LT.RHOTOL ) GO TO 25
*
         IF ( ITER.GT.1 ) THEN
            BETA = RHO / RHO1
	        type*,'beta=',beta

*           p=z+beta*p
            CALL ZAXPY( N, BETA, WORK(1,P), 1, WORK(1,Z), 1 )
            CALL ZCOPY( N, WORK(1,Z), 1, WORK(1,P), 1 )
         ELSE
*           p=z
            CALL ZCOPY( N, WORK(1,Z), 1, WORK(1,P), 1 )
         ENDIF
*

*        q=A*p
         CALL ZGEMV( 'N', N, N, one, A, N, 
     *        WORK(1,P), 1, zero, WORK(1,Q), 1 )

         ALPHA = RHO / XSY(Nl,Np,w,pvec,dh,den0,alf0,bet0,den,
     &                 alf,bet,WORK(1,P), WORK(1,Q))

	      type*,'alpha=',alpha

*
*        Compute current solution vector x.
*
*        x=x+alpha*p
         CALL ZAXPY( N, ALPHA, WORK(1,P), 1, X, 1 )
*
*        Compute residual vector rk, find norm,
*        then check for tolerance.
*
*        r=r-alpha*q
         CALL ZAXPY( N, -ALPHA, WORK(1,Q), 1, WORK(1,R), 1 )
*
         RESID = DZNRM2( N, WORK(1,R), 1 ) / BNRM2
         IF ( RESID.LE.TOL  ) GO TO 30
         IF ( ITER.EQ.MAXIT ) GO TO 20
		 
         RHO1 = RHO
*
      GO TO 10
*
   20 CONTINUE
*
*     Iteration fails.
*
      INFO = ITER
      RETURN
*
   25 CONTINUE
*
*     Set breakdown flag.
*
      INFO = -10
      RETURN
*
   30 CONTINUE
*
*     Iteration successful; return.
*
      RETURN
*
*     End of BICG
*
      END
***************************************************************
      subroutine  zcopy(n,zx,incx,zy,incy)
c
c     copies a vector, x, to a vector, y.
c     jack dongarra, linpack, 4/11/78.
c     modified 12/3/93, array(1) declarations changed to array(*)
c
      double complex zx(*),zy(*)
      integer i,incx,incy,ix,iy,n
c
      if(n.le.0)return
      if(incx.eq.1.and.incy.eq.1)go to 20
c
c        code for unequal increments or equal increments
c          not equal to 1
c
      ix = 1
      iy = 1
      if(incx.lt.0)ix = (-n+1)*incx + 1
      if(incy.lt.0)iy = (-n+1)*incy + 1
      do 10 i = 1,n
        zy(iy) = zx(ix)
        ix = ix + incx
        iy = iy + incy
   10 continue
      return
c
c        code for both increments equal to 1
c
   20 do 30 i = 1,n
        zy(i) = zx(i)
   30 continue
      return
      end
*********************************************
      DOUBLE PRECISION FUNCTION DZNRM2( N, X, INCX )
*     .. Scalar Arguments ..
      INTEGER                           INCX, N
*     .. Array Arguments ..
      COMPLEX*16                        X( * )
*     ..
*
*  DZNRM2 returns the euclidean norm of a vector via the function
*  name, so that
*
*     DZNRM2 := sqrt( conjg( x' )*x )
*
*
*
*  -- This version written on 25-October-1982.
*     Modified on 14-October-1993 to inline the call to ZLASSQ.
*     Sven Hammarling, Nag Ltd.
*
*
*     .. Parameters ..
      DOUBLE PRECISION      ONE         , ZERO
      PARAMETER           ( ONE = 1.0D+0, ZERO = 0.0D+0 )
*     .. Local Scalars ..
      INTEGER               IX
      DOUBLE PRECISION      NORM, SCALE, SSQ, TEMP
*     .. Intrinsic Functions ..
      INTRINSIC             ABS, DIMAG, DBLE, SQRT
*     ..
*     .. Executable Statements ..
      IF( N.LT.1 .OR. INCX.LT.1 )THEN
         NORM  = ZERO
      ELSE
         SCALE = ZERO
         SSQ   = ONE
*        The following loop is equivalent to this call to the LAPACK
*        auxiliary routine:
*        CALL ZLASSQ( N, X, INCX, SCALE, SSQ )
*
         DO 10, IX = 1, 1 + ( N - 1 )*INCX, INCX
            IF( DBLE( X( IX ) ).NE.ZERO )THEN
               TEMP = ABS( DBLE( X( IX ) ) )
               IF( SCALE.LT.TEMP )THEN
                  SSQ   = ONE   + SSQ*( SCALE/TEMP )**2
                  SCALE = TEMP
               ELSE
                  SSQ   = SSQ   +     ( TEMP/SCALE )**2
               END IF
            END IF
            IF( DIMAG( X( IX ) ).NE.ZERO )THEN
               TEMP = ABS( DIMAG( X( IX ) ) )
               IF( SCALE.LT.TEMP )THEN
                  SSQ   = ONE   + SSQ*( SCALE/TEMP )**2
                  SCALE = TEMP
               ELSE
                  SSQ   = SSQ   +     ( TEMP/SCALE )**2
               END IF
            END IF
   10    CONTINUE
         NORM  = SCALE * SQRT( SSQ )
      END IF
*
      DZNRM2 = NORM
      RETURN
*
*     End of DZNRM2.
*
      END
*****************************************************

      SUBROUTINE ZGEMV ( TRANS, M, N, ALPHA, A, LDA, X, INCX,
     $                   BETA, Y, INCY )
*     .. Scalar Arguments ..
      COMPLEX*16         ALPHA, BETA
      INTEGER            INCX, INCY, LDA, M, N
      CHARACTER*1        TRANS
*     .. Array Arguments ..
      COMPLEX*16         A( LDA, * ), X( * ), Y( * )
*     ..
*
*  Purpose
*  =======
*
*  ZGEMV  performs one of the matrix-vector operations
*
*     y := alpha*A*x + beta*y,   or   y := alpha*A'*x + beta*y,   or
*
*     y := alpha*conjg( A' )*x + beta*y,
*
*  where alpha and beta are scalars, x and y are vectors and A is an
*  m by n matrix.
*
*  Parameters
*  ==========
*
*  TRANS  - CHARACTER*1.
*           On entry, TRANS specifies the operation to be performed as
*           follows:
*
*              TRANS = 'N' or 'n'   y := alpha*A*x + beta*y.
*
*              TRANS = 'T' or 't'   y := alpha*A'*x + beta*y.
*
*              TRANS = 'C' or 'c'   y := alpha*conjg( A' )*x + beta*y.
*
*           Unchanged on exit.
*
*  M      - INTEGER.
*           On entry, M specifies the number of rows of the matrix A.
*           M must be at least zero.
*           Unchanged on exit.
*
*  N      - INTEGER.
*           On entry, N specifies the number of columns of the matrix A.
*           N must be at least zero.
*           Unchanged on exit.
*
*  ALPHA  - COMPLEX*16      .
*           On entry, ALPHA specifies the scalar alpha.
*           Unchanged on exit.
*
*  A      - COMPLEX*16       array of DIMENSION ( LDA, n ).
*           Before entry, the leading m by n part of the array A must
*           contain the matrix of coefficients.
*           Unchanged on exit.
*
*  LDA    - INTEGER.
*           On entry, LDA specifies the first dimension of A as declared
*           in the calling (sub) program. LDA must be at least
*           max( 1, m ).
*           Unchanged on exit.
*
*  X      - COMPLEX*16       array of DIMENSION at least
*           ( 1 + ( n - 1 )*abs( INCX ) ) when TRANS = 'N' or 'n'
*           and at least
*           ( 1 + ( m - 1 )*abs( INCX ) ) otherwise.
*           Before entry, the incremented array X must contain the
*           vector x.
*           Unchanged on exit.
*
*  INCX   - INTEGER.
*           On entry, INCX specifies the increment for the elements of
*           X. INCX must not be zero.
*           Unchanged on exit.
*
*  BETA   - COMPLEX*16      .
*           On entry, BETA specifies the scalar beta. When BETA is
*           supplied as zero then Y need not be set on input.
*           Unchanged on exit.
*
*  Y      - COMPLEX*16       array of DIMENSION at least
*           ( 1 + ( m - 1 )*abs( INCY ) ) when TRANS = 'N' or 'n'
*           and at least
*           ( 1 + ( n - 1 )*abs( INCY ) ) otherwise.
*           Before entry with BETA non-zero, the incremented array Y
*           must contain the vector y. On exit, Y is overwritten by the
*           updated vector y.
*
*  INCY   - INTEGER.
*           On entry, INCY specifies the increment for the elements of
*           Y. INCY must not be zero.
*           Unchanged on exit.
*
*
*  Level 2 Blas routine.
*
*  -- Written on 22-October-1986.
*     Jack Dongarra, Argonne National Lab.
*     Jeremy Du Croz, Nag Central Office.
*     Sven Hammarling, Nag Central Office.
*     Richard Hanson, Sandia National Labs.
*
*
*     .. Parameters ..
      COMPLEX*16         ONE
      PARAMETER        ( ONE  = ( 1.0D+0, 0.0D+0 ) )
      COMPLEX*16         ZERO
      PARAMETER        ( ZERO = ( 0.0D+0, 0.0D+0 ) )
*     .. Local Scalars ..
      COMPLEX*16         TEMP
      INTEGER            I, INFO, IX, IY, J, JX, JY, KX, KY, LENX, LENY
      LOGICAL            NOCONJ
*     .. External Functions ..
      LOGICAL            LSAME
      EXTERNAL           LSAME
*     .. External Subroutines ..
      EXTERNAL           XERBLA
*     .. Intrinsic Functions ..
      INTRINSIC          DCONJG, MAX
*     ..
*     .. Executable Statements ..
*
*     Test the input parameters.
*
      INFO = 0
      IF     ( .NOT.LSAME( TRANS, 'N' ).AND.
     $         .NOT.LSAME( TRANS, 'T' ).AND.
     $         .NOT.LSAME( TRANS, 'C' )      )THEN
         INFO = 1
      ELSE IF( M.LT.0 )THEN
         INFO = 2
      ELSE IF( N.LT.0 )THEN
         INFO = 3
      ELSE IF( LDA.LT.MAX( 1, M ) )THEN
         INFO = 6
      ELSE IF( INCX.EQ.0 )THEN
         INFO = 8
      ELSE IF( INCY.EQ.0 )THEN
         INFO = 11
      END IF
      IF( INFO.NE.0 )THEN
         CALL XERBLA( 'ZGEMV ', INFO )
         RETURN
      END IF
*
*     Quick return if possible.
*
      IF( ( M.EQ.0 ).OR.( N.EQ.0 ).OR.
     $    ( ( ALPHA.EQ.ZERO ).AND.( BETA.EQ.ONE ) ) )
     $   RETURN
*
      NOCONJ = LSAME( TRANS, 'T' )
*
*     Set  LENX  and  LENY, the lengths of the vectors x and y, and set
*     up the start points in  X  and  Y.
*
      IF( LSAME( TRANS, 'N' ) )THEN
         LENX = N
         LENY = M
      ELSE
         LENX = M
         LENY = N
      END IF
      IF( INCX.GT.0 )THEN
         KX = 1
      ELSE
         KX = 1 - ( LENX - 1 )*INCX
      END IF
      IF( INCY.GT.0 )THEN
         KY = 1
      ELSE
         KY = 1 - ( LENY - 1 )*INCY
      END IF
*
*     Start the operations. In this version the elements of A are
*     accessed sequentially with one pass through A.
*
*     First form  y := beta*y.
*
      IF( BETA.NE.ONE )THEN
         IF( INCY.EQ.1 )THEN
            IF( BETA.EQ.ZERO )THEN
               DO 10, I = 1, LENY
                  Y( I ) = ZERO
   10          CONTINUE
            ELSE
               DO 20, I = 1, LENY
                  Y( I ) = BETA*Y( I )
   20          CONTINUE
            END IF
         ELSE
            IY = KY
            IF( BETA.EQ.ZERO )THEN
               DO 30, I = 1, LENY
                  Y( IY ) = ZERO
                  IY      = IY   + INCY
   30          CONTINUE
            ELSE
               DO 40, I = 1, LENY
                  Y( IY ) = BETA*Y( IY )
                  IY      = IY           + INCY
   40          CONTINUE
            END IF
         END IF
      END IF
      IF( ALPHA.EQ.ZERO )
     $   RETURN
      IF( LSAME( TRANS, 'N' ) )THEN
*
*        Form  y := alpha*A*x + y.
*
         JX = KX
         IF( INCY.EQ.1 )THEN
            DO 60, J = 1, N
               IF( X( JX ).NE.ZERO )THEN
                  TEMP = ALPHA*X( JX )
                  DO 50, I = 1, M
                     Y( I ) = Y( I ) + TEMP*A( I, J )
   50             CONTINUE
               END IF
               JX = JX + INCX
   60       CONTINUE
         ELSE
            DO 80, J = 1, N
               IF( X( JX ).NE.ZERO )THEN
                  TEMP = ALPHA*X( JX )
                  IY   = KY
                  DO 70, I = 1, M
                     Y( IY ) = Y( IY ) + TEMP*A( I, J )
                     IY      = IY      + INCY
   70             CONTINUE
               END IF
               JX = JX + INCX
   80       CONTINUE
         END IF
      ELSE
*
*        Form  y := alpha*A'*x + y  or  y := alpha*conjg( A' )*x + y.
*
         JY = KY
         IF( INCX.EQ.1 )THEN
            DO 110, J = 1, N
               TEMP = ZERO
               IF( NOCONJ )THEN
                  DO 90, I = 1, M
                     TEMP = TEMP + A( I, J )*X( I )
   90             CONTINUE
               ELSE
                  DO 100, I = 1, M
                     TEMP = TEMP + DCONJG( A( I, J ) )*X( I )
  100             CONTINUE
               END IF
               Y( JY ) = Y( JY ) + ALPHA*TEMP
               JY      = JY      + INCY
  110       CONTINUE
         ELSE
            DO 140, J = 1, N
               TEMP = ZERO
               IX   = KX
               IF( NOCONJ )THEN
                  DO 120, I = 1, M
                     TEMP = TEMP + A( I, J )*X( IX )
                     IX   = IX   + INCX
  120             CONTINUE
               ELSE
                  DO 130, I = 1, M
                     TEMP = TEMP + DCONJG( A( I, J ) )*X( IX )
                     IX   = IX   + INCX
  130             CONTINUE
               END IF
               Y( JY ) = Y( JY ) + ALPHA*TEMP
               JY      = JY      + INCY
  140       CONTINUE
         END IF
      END IF
*
      RETURN
*
*     End of ZGEMV .
*
      END
**************************************************

      subroutine  zdscal(n,da,zx,incx)
c
c     scales a vector by a constant.
c     jack dongarra, 3/11/78.
c     modified 3/93 to return if incx .le. 0.
c     modified 12/3/93, array(1) declarations changed to array(*)
c
      double complex zx(*)
      double precision da
      integer i,incx,ix,n
c
      if( n.le.0 .or. incx.le.0 )return
      if(incx.eq.1)go to 20
c
c        code for increment not equal to 1
c
      ix = 1
      do 10 i = 1,n
        zx(ix) = dcmplx(da,0.0d0)*zx(ix)
        ix = ix + incx
   10 continue
      return
c
c        code for increment equal to 1
c
   20 do 30 i = 1,n
        zx(i) = dcmplx(da,0.0d0)*zx(i)
   30 continue
      return
      end
************************************************
      double complex function zdotu(n,zx,incx,zy,incy)
c
c     forms the dot product of two vectors.
c     jack dongarra, 3/11/78.
c     modified 12/3/93, array(1) declarations changed to array(*)
c
      double complex zx(*),zy(*),ztemp
      integer i,incx,incy,ix,iy,n
      ztemp = (0.0d0,0.0d0)
      zdotu = (0.0d0,0.0d0)
      if(n.le.0)return
      if(incx.eq.1.and.incy.eq.1)go to 20
c
c        code for unequal increments or equal increments
c          not equal to 1
c
      ix = 1
      iy = 1
      if(incx.lt.0)ix = (-n+1)*incx + 1
      if(incy.lt.0)iy = (-n+1)*incy + 1
      do 10 i = 1,n
        ztemp = ztemp + zx(ix)*zy(iy)
        ix = ix + incx
        iy = iy + incy
   10 continue
      zdotu = ztemp
      return
c
c        code for both increments equal to 1
c
   20 do 30 i = 1,n
        ztemp = ztemp + zx(i)*zy(i)
   30 continue
      zdotu = ztemp
      return
      end

*******************************************************
      subroutine zaxpy(n,za,zx,incx,zy,incy)
c
c     constant times a vector plus a vector.
c     jack dongarra, 3/11/78.
c     modified 12/3/93, array(1) declarations changed to array(*)
c
      double complex zx(*),zy(*),za
      integer i,incx,incy,ix,iy,n
      double precision dcabs1
      if(n.le.0)return
      if (dcabs1(za) .eq. 0.0d0) return
      if (incx.eq.1.and.incy.eq.1)go to 20
c
c        code for unequal increments or equal increments
c          not equal to 1
c
      ix = 1
      iy = 1
      if(incx.lt.0)ix = (-n+1)*incx + 1
      if(incy.lt.0)iy = (-n+1)*incy + 1
      do 10 i = 1,n
        zy(iy) = zy(iy) + za*zx(ix)
        ix = ix + incx
        iy = iy + incy
   10 continue
      return
c
c        code for both increments equal to 1
c
   20 do 30 i = 1,n
        zy(i) = zy(i) + za*zx(i)
   30 continue
      return
      end
****************************************************

      subroutine  zscal(n,za,zx,incx)
c
c     scales a vector by a constant.
c     jack dongarra, 3/11/78.
c     modified 3/93 to return if incx .le. 0.
c     modified 12/3/93, array(1) declarations changed to array(*)
c
      double complex za,zx(*)
      integer i,incx,ix,n
c
      if( n.le.0 .or. incx.le.0 )return
      if(incx.eq.1)go to 20
c
c        code for increment not equal to 1
c
      ix = 1
      do 10 i = 1,n
        zx(ix) = za*zx(ix)
        ix = ix + incx
   10 continue
      return
c
c        code for increment equal to 1
c
   20 do 30 i = 1,n
        zx(i) = za*zx(i)
   30 continue
      return
      end
*******************************************************


      SUBROUTINE ZTRSV ( UPLO, TRANS, DIAG, N, A, LDA, X, INCX )
*     .. Scalar Arguments ..
      INTEGER            INCX, LDA, N
      CHARACTER*1        DIAG, TRANS, UPLO
*     .. Array Arguments ..
      COMPLEX*16         A( LDA, * ), X( * )
*     ..
*
*  Purpose
*  =======
*
*  ZTRSV  solves one of the systems of equations
*
*     A*x = b,   or   A'*x = b,   or   conjg( A' )*x = b,
*
*  where b and x are n element vectors and A is an n by n unit, or
*  non-unit, upper or lower triangular matrix.
*
*  No test for singularity or near-singularity is included in this
*  routine. Such tests must be performed before calling this routine.
*
*  Parameters
*  ==========
*
*  UPLO   - CHARACTER*1.
*           On entry, UPLO specifies whether the matrix is an upper or
*           lower triangular matrix as follows:
*
*              UPLO = 'U' or 'u'   A is an upper triangular matrix.
*
*              UPLO = 'L' or 'l'   A is a lower triangular matrix.
*
*           Unchanged on exit.
*
*  TRANS  - CHARACTER*1.
*           On entry, TRANS specifies the equations to be solved as
*           follows:
*
*              TRANS = 'N' or 'n'   A*x = b.
*
*              TRANS = 'T' or 't'   A'*x = b.
*
*              TRANS = 'C' or 'c'   conjg( A' )*x = b.
*
*           Unchanged on exit.
*
*  DIAG   - CHARACTER*1.
*           On entry, DIAG specifies whether or not A is unit
*           triangular as follows:
*
*              DIAG = 'U' or 'u'   A is assumed to be unit triangular.
*
*              DIAG = 'N' or 'n'   A is not assumed to be unit
*                                  triangular.
*
*           Unchanged on exit.
*
*  N      - INTEGER.
*           On entry, N specifies the order of the matrix A.
*           N must be at least zero.
*           Unchanged on exit.
*
*  A      - COMPLEX*16       array of DIMENSION ( LDA, n ).
*           Before entry with  UPLO = 'U' or 'u', the leading n by n
*           upper triangular part of the array A must contain the upper
*           triangular matrix and the strictly lower triangular part of
*           A is not referenced.
*           Before entry with UPLO = 'L' or 'l', the leading n by n
*           lower triangular part of the array A must contain the lower
*           triangular matrix and the strictly upper triangular part of
*           A is not referenced.
*           Note that when  DIAG = 'U' or 'u', the diagonal elements of
*           A are not referenced either, but are assumed to be unity.
*           Unchanged on exit.
*
*  LDA    - INTEGER.
*           On entry, LDA specifies the first dimension of A as declared
*           in the calling (sub) program. LDA must be at least
*           max( 1, n ).
*           Unchanged on exit.
*
*  X      - COMPLEX*16       array of dimension at least
*           ( 1 + ( n - 1 )*abs( INCX ) ).
*           Before entry, the incremented array X must contain the n
*           element right-hand side vector b. On exit, X is overwritten
*           with the solution vector x.
*
*  INCX   - INTEGER.
*           On entry, INCX specifies the increment for the elements of
*           X. INCX must not be zero.
*           Unchanged on exit.
*
*
*  Level 2 Blas routine.
*
*  -- Written on 22-October-1986.
*     Jack Dongarra, Argonne National Lab.
*     Jeremy Du Croz, Nag Central Office.
*     Sven Hammarling, Nag Central Office.
*     Richard Hanson, Sandia National Labs.
*
*
*     .. Parameters ..
      COMPLEX*16         ZERO
      PARAMETER        ( ZERO = ( 0.0D+0, 0.0D+0 ) )
*     .. Local Scalars ..
      COMPLEX*16         TEMP
      INTEGER            I, INFO, IX, J, JX, KX
      LOGICAL            NOCONJ, NOUNIT
*     .. External Functions ..
      LOGICAL            LSAME
      EXTERNAL           LSAME
*     .. External Subroutines ..
      EXTERNAL           XERBLA
*     .. Intrinsic Functions ..
      INTRINSIC          DCONJG, MAX
*     ..
*     .. Executable Statements ..
*
*     Test the input parameters.
*
      INFO = 0
      IF     ( .NOT.LSAME( UPLO , 'U' ).AND.
     $         .NOT.LSAME( UPLO , 'L' )      )THEN
         INFO = 1
      ELSE IF( .NOT.LSAME( TRANS, 'N' ).AND.
     $         .NOT.LSAME( TRANS, 'T' ).AND.
     $         .NOT.LSAME( TRANS, 'C' )      )THEN
         INFO = 2
      ELSE IF( .NOT.LSAME( DIAG , 'U' ).AND.
     $         .NOT.LSAME( DIAG , 'N' )      )THEN
         INFO = 3
      ELSE IF( N.LT.0 )THEN
         INFO = 4
      ELSE IF( LDA.LT.MAX( 1, N ) )THEN
         INFO = 6
      ELSE IF( INCX.EQ.0 )THEN
         INFO = 8
      END IF
      IF( INFO.NE.0 )THEN
         CALL XERBLA( 'ZTRSV ', INFO )
         RETURN
      END IF
*
*     Quick return if possible.
*
      IF( N.EQ.0 )
     $   RETURN
*
      NOCONJ = LSAME( TRANS, 'T' )
      NOUNIT = LSAME( DIAG , 'N' )
*
*     Set up the start point in X if the increment is not unity. This
*     will be  ( N - 1 )*INCX  too small for descending loops.
*
      IF( INCX.LE.0 )THEN
         KX = 1 - ( N - 1 )*INCX
      ELSE IF( INCX.NE.1 )THEN
         KX = 1
      END IF
*
*     Start the operations. In this version the elements of A are
*     accessed sequentially with one pass through A.
*
      IF( LSAME( TRANS, 'N' ) )THEN
*
*        Form  x := inv( A )*x.
*
         IF( LSAME( UPLO, 'U' ) )THEN
            IF( INCX.EQ.1 )THEN
               DO 20, J = N, 1, -1
                  IF( X( J ).NE.ZERO )THEN
                     IF( NOUNIT )
     $                  X( J ) = X( J )/A( J, J )
                     TEMP = X( J )
                     DO 10, I = J - 1, 1, -1
                        X( I ) = X( I ) - TEMP*A( I, J )
   10                CONTINUE
                  END IF
   20          CONTINUE
            ELSE
               JX = KX + ( N - 1 )*INCX
               DO 40, J = N, 1, -1
                  IF( X( JX ).NE.ZERO )THEN
                     IF( NOUNIT )
     $                  X( JX ) = X( JX )/A( J, J )
                     TEMP = X( JX )
                     IX   = JX
                     DO 30, I = J - 1, 1, -1
                        IX      = IX      - INCX
                        X( IX ) = X( IX ) - TEMP*A( I, J )
   30                CONTINUE
                  END IF
                  JX = JX - INCX
   40          CONTINUE
            END IF
         ELSE
            IF( INCX.EQ.1 )THEN
               DO 60, J = 1, N
                  IF( X( J ).NE.ZERO )THEN
                     IF( NOUNIT )
     $                  X( J ) = X( J )/A( J, J )
                     TEMP = X( J )
                     DO 50, I = J + 1, N
                        X( I ) = X( I ) - TEMP*A( I, J )
   50                CONTINUE
                  END IF
   60          CONTINUE
            ELSE
               JX = KX
               DO 80, J = 1, N
                  IF( X( JX ).NE.ZERO )THEN
                     IF( NOUNIT )
     $                  X( JX ) = X( JX )/A( J, J )
                     TEMP = X( JX )
                     IX   = JX
                     DO 70, I = J + 1, N
                        IX      = IX      + INCX
                        X( IX ) = X( IX ) - TEMP*A( I, J )
   70                CONTINUE
                  END IF
                  JX = JX + INCX
   80          CONTINUE
            END IF
         END IF
      ELSE
*
*        Form  x := inv( A' )*x  or  x := inv( conjg( A' ) )*x.
*
         IF( LSAME( UPLO, 'U' ) )THEN
            IF( INCX.EQ.1 )THEN
               DO 110, J = 1, N
                  TEMP = X( J )
                  IF( NOCONJ )THEN
                     DO 90, I = 1, J - 1
                        TEMP = TEMP - A( I, J )*X( I )
   90                CONTINUE
                     IF( NOUNIT )
     $                  TEMP = TEMP/A( J, J )
                  ELSE
                     DO 100, I = 1, J - 1
                        TEMP = TEMP - DCONJG( A( I, J ) )*X( I )
  100                CONTINUE
                     IF( NOUNIT )
     $                  TEMP = TEMP/DCONJG( A( J, J ) )
                  END IF
                  X( J ) = TEMP
  110          CONTINUE
            ELSE
               JX = KX
               DO 140, J = 1, N
                  IX   = KX
                  TEMP = X( JX )
                  IF( NOCONJ )THEN
                     DO 120, I = 1, J - 1
                        TEMP = TEMP - A( I, J )*X( IX )
                        IX   = IX   + INCX
  120                CONTINUE
                     IF( NOUNIT )
     $                  TEMP = TEMP/A( J, J )
                  ELSE
                     DO 130, I = 1, J - 1
                        TEMP = TEMP - DCONJG( A( I, J ) )*X( IX )
                        IX   = IX   + INCX
  130                CONTINUE
                     IF( NOUNIT )
     $                  TEMP = TEMP/DCONJG( A( J, J ) )
                  END IF
                  X( JX ) = TEMP
                  JX      = JX   + INCX
  140          CONTINUE
            END IF
         ELSE
            IF( INCX.EQ.1 )THEN
               DO 170, J = N, 1, -1
                  TEMP = X( J )
                  IF( NOCONJ )THEN
                     DO 150, I = N, J + 1, -1
                        TEMP = TEMP - A( I, J )*X( I )
  150                CONTINUE
                     IF( NOUNIT )
     $                  TEMP = TEMP/A( J, J )
                  ELSE
                     DO 160, I = N, J + 1, -1
                        TEMP = TEMP - DCONJG( A( I, J ) )*X( I )
  160                CONTINUE
                     IF( NOUNIT )
     $                  TEMP = TEMP/DCONJG( A( J, J ) )
                  END IF
                  X( J ) = TEMP
  170          CONTINUE
            ELSE
               KX = KX + ( N - 1 )*INCX
               JX = KX
               DO 200, J = N, 1, -1
                  IX   = KX
                  TEMP = X( JX )
                  IF( NOCONJ )THEN
                     DO 180, I = N, J + 1, -1
                        TEMP = TEMP - A( I, J )*X( IX )
                        IX   = IX   - INCX
  180                CONTINUE
                     IF( NOUNIT )
     $                  TEMP = TEMP/A( J, J )
                  ELSE
                     DO 190, I = N, J + 1, -1
                        TEMP = TEMP - DCONJG( A( I, J ) )*X( IX )
                        IX   = IX   - INCX
  190                CONTINUE
                     IF( NOUNIT )
     $                  TEMP = TEMP/DCONJG( A( J, J ) )
                  END IF
                  X( JX ) = TEMP
                  JX      = JX   - INCX
  200          CONTINUE
            END IF
         END IF
      END IF
*
      RETURN
*
*     End of ZTRSV .
*
      END
*************************************************
      double precision function dcabs1(z)
      double complex z,zz
      double precision t(2)
      equivalence (zz,t(1))
      zz = z
      dcabs1 = dabs(t(1)) + dabs(t(2))
      return
      end

*************************************************
      LOGICAL          FUNCTION LSAME( CA, CB )
*
*  -- LAPACK auxiliary routine (version 2.0) --
*     Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
*     Courant Institute, Argonne National Lab, and Rice University
*     January 31, 1994
*
*     .. Scalar Arguments ..
      CHARACTER          CA, CB
*     ..
*
*  Purpose
*  =======
*
*  LSAME returns .TRUE. if CA is the same letter as CB regardless of
*  case.
*
*  Arguments
*  =========
*
*  CA      (input) CHARACTER*1
*  CB      (input) CHARACTER*1
*          CA and CB specify the single characters to be compared.
*
* =====================================================================
*
*     .. Intrinsic Functions ..
      INTRINSIC          ICHAR
*     ..
*     .. Local Scalars ..
      INTEGER            INTA, INTB, ZCODE
*     ..
*     .. Executable Statements ..
*
*     Test if the characters are equal
*
      LSAME = CA.EQ.CB
      IF( LSAME )
     $   RETURN
*
*     Now test for equivalence if both characters are alphabetic.
*
      ZCODE = ICHAR( 'Z' )
*
*     Use 'Z' rather than 'A' so that ASCII can be detected on Prime
*     machines, on which ICHAR returns a value with bit 8 set.
*     ICHAR('A') on Prime machines returns 193 which is the same as
*     ICHAR('A') on an EBCDIC machine.
*
      INTA = ICHAR( CA )
      INTB = ICHAR( CB )
*
      IF( ZCODE.EQ.90 .OR. ZCODE.EQ.122 ) THEN
*
*        ASCII is assumed - ZCODE is the ASCII code of either lower or
*        upper case 'Z'.
*
         IF( INTA.GE.97 .AND. INTA.LE.122 ) INTA = INTA - 32
         IF( INTB.GE.97 .AND. INTB.LE.122 ) INTB = INTB - 32
*
      ELSE IF( ZCODE.EQ.233 .OR. ZCODE.EQ.169 ) THEN
*
*        EBCDIC is assumed - ZCODE is the EBCDIC code of either lower or
*        upper case 'Z'.
*
         IF( INTA.GE.129 .AND. INTA.LE.137 .OR.
     $       INTA.GE.145 .AND. INTA.LE.153 .OR.
     $       INTA.GE.162 .AND. INTA.LE.169 ) INTA = INTA + 64
         IF( INTB.GE.129 .AND. INTB.LE.137 .OR.
     $       INTB.GE.145 .AND. INTB.LE.153 .OR.
     $       INTB.GE.162 .AND. INTB.LE.169 ) INTB = INTB + 64
*
      ELSE IF( ZCODE.EQ.218 .OR. ZCODE.EQ.250 ) THEN
*
*        ASCII is assumed, on Prime machines - ZCODE is the ASCII code
*        plus 128 of either lower or upper case 'Z'.
*
         IF( INTA.GE.225 .AND. INTA.LE.250 ) INTA = INTA - 32
         IF( INTB.GE.225 .AND. INTB.LE.250 ) INTB = INTB - 32
      END IF
      LSAME = INTA.EQ.INTB
*
*     RETURN
*
*     End of LSAME
*
      END
****************************************************

      SUBROUTINE XERBLA( SRNAME, INFO )
*
*  -- LAPACK auxiliary routine (preliminary version) --
*     Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
*     Courant Institute, Argonne National Lab, and Rice University
*     February 29, 1992
*
*     .. Scalar Arguments ..
      CHARACTER*6        SRNAME
      INTEGER            INFO
*     ..
*
*  Purpose
*  =======
*
*  XERBLA  is an error handler for the LAPACK routines.
*  It is called by an LAPACK routine if an input parameter has an
*  invalid value.  A message is printed and execution stops.
*
*  Installers may consider modifying the STOP statement in order to
*  call system-specific exception-handling facilities.
*
*  Arguments
*  =========
*
*  SRNAME  (input) CHARACTER*6
*          The name of the routine which called XERBLA.
*
*  INFO    (input) INTEGER
*          The position of the invalid parameter in the parameter list
*          of the calling routine.
*
*
      WRITE( *, FMT = 9999 )SRNAME, INFO
*
      STOP
*
 9999 FORMAT( ' ** On entry to ', A6, ' parameter number ', I2, ' had ',
     $      'an illegal value' )
*
*     End of XERBLA
*
      END
*******************************************************
* 
*     ================================================================
      DOUBLE PRECISION FUNCTION GETBREAK()
*
*     Get breakdown parameter tolerance; for the test routine,
*     set to machine precision.
*
      DOUBLE PRECISION EPS, DLAMCH
*
*      EPS = DLAMCH('EPS') !!!!!!!!AMMEND ACCORDINLY
*      GETBREAK = EPS**2
      GETBREAK = (1.D-12)

*
      RETURN
*
      END
*     ===============================================================
      DOUBLE PRECISION FUNCTION SCALEDRESID( ANORM, N, X, RK, TOL )
*
*     Returns |B-A*X| / ( |A||X|*N*TOL ), using the infinity norm.
*
      INTEGER            N, IDAMAX
      DOUBLE PRECISION   ANORM, TOL, XNORM, RESNORM,
     $                   X( * ), RK( * )
*
      XNORM   = ABS( X( IDAMAX( N, X, 1 ) ) )
      RESNORM = ABS( RK( IDAMAX( N, RK, 1 ) ) )
*
      SCALEDRESID = RESNORM / ( TOL * N * ANORM * XNORM )
*
      RETURN
*
      END
*     ===========================================================
      DOUBLE PRECISION FUNCTION MATNORM( N, A, LDA )
*
*     Compute infinity norm of matrix A.
*
      INTEGER            N, LDA, I, J
      DOUBLE PRECISION   ROWSUM, ZERO, TEMP, A( LDA,* )
      PARAMETER        ( ZERO = 0.0D+0 )
*
      TEMP = ZERO
      DO 20 I = 1, N
         ROWSUM = ZERO
         DO 10 J = 1, N
            ROWSUM = ROWSUM + ABS( A( I,J ) )
   10    CONTINUE
         TEMP = MAX( ROWSUM, TEMP )
   20 CONTINUE
*
      MATNORM = TEMP
*
      RETURN
*
      END
*
*     ===============================================================


      complex*16 FUNCTION Vslo(C,P)
*     compute the vertical slowness
      complex*16 P,C,Cinv
      Real*8 eps
      parameter(eps=1.0E-8)
      complex*16 T
      if(abs(c).le.eps)then
        PAUSE 'propagation velocity too small'
      else
        Cinv=1.0/c
      end if
      T=dcmplx((Cinv+P)*(Cinv-P))
      Vslo=csqrt(T)
      IF(dimag(Vslo) .LT. 0.0)THEN 
        Vslo=-Vslo
      END IF
      If(dimag(Vslo)*dble(Vslo).lt. 0.0) then
        Pause'DANGER! nonThermodynamic Vslo'
      Endif	  
      If(Cabs(Vslo).lt. eps) then
        Pause'DANGER! hit turning point'
      Endif
      RETURN
      END

************************************

      complex*16 FUNCTION CV(W,C0,Q0)
*     DETERMINE THE FREQUENCY DEPENDENT RECIPROCAL PROPAGATION 
*     VELOCITY (SLOWNESS) ACCORDING TO THE POWER LAW OF STRICK
      Real*8    Q0
      complex*16 C0,W
*     local variables      
      Real*8    EPS,Pi,W0,SIGMA,T1
      complex*16 T,CW
      PARAMETER(Pi=3.14159265358979324,W0=1.0,SIGMA=0.1,EPS=0.001)
      T=cdexp(-SIGMA*CLOG(dcmplx(EPS+dimag(W),-dble(W))))
      T1=W0**SIGMA/SIN(SIGMA*PI/2)
      CW=(1.0,0.0)+dcmplx(0.5*T1/Q0,0.0)*T
      CV=C0/CW
      RETURN
      END

******************************************************
      SUBROUTINE setinc(Nl,p,w,h,Zs,rho0,alpha0,beta0,
     &           rho,alpha,beta,Resp0,X)

*     set down reflexion incident field
      Implicit none
      INTEGER  Nl,Nlm,NLM4,NLM2,N1
      parameter(Nlm=2,NLM2=2*NLM,NLM4=4*Nlm,N1=Nlm+1)
      Integer  ilp1,ilm1
      Real*8     h(0:Nl),Zs
      Real*8     rho0(0:N1),rho(0:N1)
      complex*16  alpha0(0:N1),beta0(0:N1)
      complex*16  alpha(0:N1),beta(0:N1)
      complex*16  w,p

      complex*16  Resp0,X(NLM4)

****************************************************************   
*      LOCAL variables  
*****************************************************************
      Integer  ioff4,ioff8,Nl2,nl2m1,IL,il2,ioff,i1,i2
      complex*16  Xwk(4*2*Nlm)
      complex*16  Rf11,Rf12,Rf21,Rf22,InvUv11,InvUv12,InvUv21,InvUv22
      complex*16  Nd(2,2),Md(2,2),Nu(2,2),Mu(2,2),Vslo
      complex*16  cbeta1,cbeta2,temp,Ci,C1
      complex*16 T11,T12,T21,T22
      complex*16 U11,U12,U21,U22
      complex*16 P1,P2
      complex*16 Q1,Q2
      complex*16 B11,B12
      complex*16 B21,B22
      complex*16 RR11,RR12,RR21,RR22
      complex*16 R11(0:NLm2),R12(0:NLm2),R21(0:NLm2),R22(0:NLm2)
      complex*16 S11(0:NLm2),S12(0:NLm2),S21(0:NLm2),S22(0:NLm2)
      complex*16 Rd11(0:NLm2),Rd12(0:NLm2),Rd21(0:NLm2),Rd22(0:NLm2)
      complex*16 Tu11(0:NLm2),Tu12(0:NLm2),Tu21(0:NLm2),Tu22(0:NLm2)
      complex*16 Td11(0:NLm2),Td12(0:NLm2),Td21(0:NLm2),Td22(0:NLm2)
      complex*16 Ru11(0:NLm2),Ru12(0:NLm2),Ru21(0:NLm2),Ru22(0:NLm2)
      complex*16 ea0,eb0,easf,ebsf,eas0,ebs0,SigPU,SigSU,SigPD,SigSD
      complex*16 eta0(0:N1),neta0(0:N1)
      complex*16 W11,W12,W21,W22,V11,V12,V21,V22,InvDet
      complex*16 Df1,Df2,Uf1,Uf2
     
      if(Nl.gt.NLm) then
         stop 'Nl.gt.NLm in setinc'
       Endif
       Nl2=2*Nl


      C1=(1.0,0.0)	  
      Ci=(0.0,1.0)
*     *******************************************************
*     set free surface reflection array Rf11,Rf12,Rf21,Rf22
*     and free-surface amplitude factor
*     *******************************************************
      do il=0,Nl+1
        eta0(il)=Vslo(alpha0(il),P)
        neta0(il)=Vslo(beta0(il),P)
      end do !il
      Temp=CI*w*h(0)*eta0(0)
      EA0=cdexp(Temp)
      Temp=CI*w*h(0)*neta0(0)
      EB0=cdexp(Temp)		

      Temp=CI*w*Zs*eta0(0)
      EAsf=cdexp(Temp)
      Temp=CI*w*Zs*neta0(0)
      EBsf=cdexp(Temp)		

      Temp=CI*w*(h(0)-Zs)*eta0(0)
      EAs0=cdexp(Temp)
      Temp=CI*w*(h(0)-Zs)*neta0(0)
      EBs0=cdexp(Temp)		

*     SOURCE TERMS
      SigPu=EAsf*(1.0,0.0)
      SigSu=EBsf*(0.0,0.0)
      SigPd=EAs0*(1.0,0.0)
      SigSd=EBs0*(0.0,0.0)

      call Rfs(P,eta0(0),neta0(0),rho0(0),beta0(0),
     &     Rf11,Rf12,Rf21,Rf22)
*	  TYpe*,'Rf11=',Rf11
*	  TYpe*,'Rf12=',Rf12
*	  TYpe*,'Rf21=',Rf21
*	  TYpe*,'Rf22=',Rf22
*     set reference medium interfacial scattering arrays
      do il=0,Nl
        ilp1=il+1
        il2=2*il
        cbeta1=dcmplx(beta0(il),0.0)
        cbeta2=dcmplx(beta0(ilp1),0.0)
*       set the reference medium interfacial scattering matrices
        call ScatMat(p,eta0(il),neta0(il),rho0(il),CBeta1,eta0(ilp1),
     &     neta0(ilp1),rho0(ilp1),CBeta2,Rd11(il2),Rd12(il2),Rd21(il2),
     &     Rd22(il2),Tu11(il2),Tu12(il2),Tu21(il2),Tu22(il2),Td11(il2),
     &     Td12(il2),Td21(il2),Td22(il2),Ru11(il2),Ru12(il2),Ru21(il2),
     &     Ru22(il2))
       end do

*     set layer scattering arrays
      do il=1,Nl
        il2=2*(il-1)+1

        call  SLVD(il,Nl,w,p,h,rho0,alpha0,beta0,rho,alpha,beta,
     &  Rd11(il2),Rd12(il2),Rd21(il2),Rd22(il2),Tu11(il2),Tu12(il2),
     &  Tu21(il2),Tu22(il2),Td11(il2),Td12(il2),Td21(il2),Td22(il2),
     &  Ru11(il2),Ru12(il2),Ru21(il2),Ru22(il2))

*        call SetS(il,Nl,w,p,h,rho0,alpha0,beta0,rho,alpha,beta,
*     &  Rd11(il2),Rd12(il2),Rd21(il2),Rd22(il2),Tu11(il2),Tu12(il2),
*     &  Tu21(il2),Tu22(il2),Td11(il2),Td12(il2),Td21(il2),Td22(il2),
*     &  Ru11(il2),Ru12(il2),Ru21(il2),Ru22(il2))
      end do
*******************************************************
*     Factor the System Matrix
*     R=Rd(nl2)
      R11(nl2)=Rd11(nl2)
      R12(nl2)=Rd12(nl2)
      R21(nl2)=Rd21(nl2)
      R22(nl2)=Rd22(nl2)
*	  type*,'nl2,Rd11(nl2)=',nl2,Rd11(nl2)
*	  type*,'nl2,Rd12(nl2)=',nl2,Rd12(nl2)
*	  type*,'nl2,Rd21(nl2)=',nl2,Rd21(nl2)
*	  type*,'nl2,Rd22(nl2)=',nl2,Rd22(nl2)

      nl2m1=nl2-1
      do il=nl2m1,0,-1
        ilp1=il+1
*       U=(I-Ru*R(ilp1))^(-1)
        W11=C1-Ru21(il)*R12(ilp1)-Ru22(il)*R22(ilp1)
        W12=Ru11(il)*R12(ilp1)+Ru12(il)*R22(ilp1)
        W21=Ru21(il)*R11(ilp1)+Ru22(il)*R21(ilp1)
        W22=C1-Ru11(il)*R11(ilp1)-Ru12(il)*R21(ilp1)
        InvDet=C1/(W11*W22-W12*W21)
        U11=W11*InvDet
        U12=W12*InvDet
        U21=W21*InvDet
        U22=W22*InvDet
*		type*,'il,U11,U12,U21,U22=',il
*		type*,U11,U12,U21,U22
*        type*
*       W=R*U
        W11=R11(ilp1)*U11+R12(ilp1)*U21
        W12=R11(ilp1)*U12+R12(ilp1)*U22
        W21=R21(ilp1)*U11+R22(ilp1)*U21
        W22=R21(ilp1)*U12+R22(ilp1)*U22

*       T=Tu*W
        T11=Tu11(il)*W11+Tu12(il)*W21
        T12=Tu11(il)*W12+Tu12(il)*W22
        T21=Tu21(il)*W11+Tu22(il)*W21
        T22=Tu21(il)*W12+Tu22(il)*W22
*		type*,'il,T11,T12,T21,T22=',il
*		type*,T11,T12,T21,T22
*        type*


*       S=U*Td(il)
        S11(il)=U11*Td11(il)+U12*Td21(il)
        S12(il)=U11*Td12(il)+U12*Td22(il)
        S21(il)=U21*Td11(il)+U22*Td21(il)
        S22(il)=U21*Td12(il)+U22*Td22(il)

*		type*,'il,S11,S12,S21,S22=',il
*		type*,S11(il),S12(il),S21(il),S22(il)
*        type*

*       W=T*Td(il)
        W11=T11*Td11(il)+T12*Td21(il)
        W12=T11*Td12(il)+T12*Td22(il)
        W21=T21*Td11(il)+T22*Td21(il)
        W22=T21*Td12(il)+T22*Td22(il)

*       R=Rd+W
        R11(il)=Rd11(il)+W11
        R12(il)=Rd12(il)+W12
        R21(il)=Rd21(il)+W21
        R22(il)=Rd22(il)+W22

*		type*,'il,R11,R12,R21,R22=',il
*		type*,R11(il),R12(il),R21(il),R22(il)
*        type*
        end do !il
************************************************
*     B1=Rd(0)*SD
      B11=Rd11(0)*SigPD+Rd12(0)*SigSD
      B12=Rd21(0)*SigPD+Rd22(0)*SigSD

*     B2=Td(0)*SD
      B21=Td11(0)*SigPD+Td12(0)*SigSD
      B22=Td21(0)*SigPD+Td22(0)*SigSD

*     P(0)=B1+T*B2
      P1=B11+T11*B21+T12*B22
      P2=B12+T21*B21+T22*B22

*     Q(0)=U*B2
      Q1=U11*B21+U12*B22
      Q2=U21*B21+U22*B22

**************************************************
*     set Df=downgoing from Free surface
*     Df=Rf*(1-E0*R(0)*E0*Rf)^(-1)*[Esf*SigU+ea0*P0]
      RR11=R11(0)*ea0*ea0
      RR12=R12(0)*eb0*ea0
      RR21=R21(0)*ea0*eb0
      RR22=R22(0)*eb0*eb0

*     V=(I-RR*Rf)^(-1)
      W11=C1-RR21*Rf12-RR22*Rf22
      W12=RR11*Rf12+RR12*Rf22
      W21=RR21*Rf11+RR22*Rf21
      W22=C1-RR11*Rf11-RR12*Rf21
      InvDet=C1/(W11*W22-W12*W21)
      V11=W11*InvDet
      V12=W12*InvDet
      V21=W21*InvDet
      V22=W22*InvDet

*     W=Rf*V
      W11=Rf11*V11+Rf12*V21
      W12=Rf11*V12+Rf12*V22
      W21=Rf21*V11+Rf22*V21
      W22=Rf21*V12+Rf22*V22

*     Df=W*(SigU+e0*P(0)) !downgoing from the free-surface
      Df1=W11*(SigPU+ea0*P1)+W12*(SigSU+eb0*P2)
      Df2=W21*(SigPU+ea0*P1)+W22*(SigSU+eb0*P2)
*
*     N.B. SigU=DIRECT FIELD AT SURFACE RECEIVER
*
*     Uf=RR*Df+E0*P(0)+SigU
      Uf1=RR11*Df1+RR12*Df2+ea0*P1+SigPu
      Uf2=RR21*Df1+RR22*Df2+eb0*P1+SigSu

*     Resp0=Mu(0)*Uf+Md(0)*Df

***************************************

      Xwk(1)=Q1+S11(0)*ea0*Df1+S12(0)*eb0*Df2
      Xwk(2)=Q2+S21(0)*ea0*Df1+S22(0)*eb0*Df2
      Xwk(3)=R11(1)*Xwk(1)+R12(1)*Xwk(2)
      Xwk(4)=R21(1)*Xwk(1)+R22(1)*Xwk(2)
****************************************
      do il=2,nl2
        i1=(il-2)*4
        i2=i1+4
        ilm1=il-1
        Xwk(i2+1)=S11(ilm1)*Xwk(i1+1)+S12(ilm1)*Xwk(i1+2)
        Xwk(i2+2)=S21(ilm1)*Xwk(i1+1)+S22(ilm1)*Xwk(i1+2)
        Xwk(i2+3)=R11(il)*Xwk(i2+1)+R12(il)*Xwk(i2+2)
        Xwk(i2+4)=R21(il)*Xwk(i2+1)+R22(il)*Xwk(i2+2)
      end do !il
*     
*     Load X
*
      Do il=1,Nl
        ioff4=4*(il-1)
        ioff8=8*(il-1)
        X(ioff4+1)=Xwk(ioff8+1)
        X(ioff4+2)=Xwk(ioff8+2)
        X(ioff4+3)=Xwk(ioff8+7)
        X(ioff4+4)=Xwk(ioff8+8)
      Enddo !il

      RETURN
      END

***********************************
      SUBROUTINE setPSY(Nl,p,w,h,rho0,alpha0,beta0,rho,alpha,beta,SY,X)

*     set X=P*SY
      Implicit none
      INTEGER  Nl,Nlm,NLM4,NLM2,N1
      parameter(Nlm=2,NLM2=2*NLM,NLM4=4*Nlm,N1=Nlm+1)
      Integer  ilp1,ilm1
      Real*8     h(0:Nl)
      Real*8     rho0(0:N1),rho(0:N1)
      complex*16  alpha0(0:N1),beta0(0:N1)
      complex*16  alpha(0:N1),beta(0:N1)
      complex*16  w,p
      complex*16  SY(NLM4),X(NLM4)

****************************************************************   
*      LOCAL variables  
*****************************************************************
      Integer  ioff4,ioff8,Nl2,IL,il2,ioff
      complex*16  Xwk(4*2*Nlm)
      complex*16  Rf11,Rf12,Rf21,Rf22,InvUv11,InvUv12,InvUv21,InvUv22
      complex*16  Nd(2,2),Md(2,2),Nu(2,2),Mu(2,2),Vslo
      complex*16  cbeta1,cbeta2,temp,Ci,C1
      complex*16 T11(0:NLm2),T12(0:NLm2),T21(0:NLm2),T22(0:NLm2)
      complex*16 U11(0:NLm2),U12(0:NLm2),U21(0:NLm2),U22(0:NLm2)
      complex*16 P1(0:NLm2),P2(0:NLm2)
      complex*16 Q1(0:NLm2),Q2(0:NLm2)
      complex*16 B11(0:NLm2),B12(0:NLm2)
      complex*16 B21(0:NLm2),B22(0:NLm2)
      complex*16 R11(0:NLm2),R12(0:NLm2),R21(0:NLm2),R22(0:NLm2)
      complex*16 S11(0:NLm2),S12(0:NLm2),S21(0:NLm2),S22(0:NLm2)
      complex*16 Rd11(0:NLm2),Rd12(0:NLm2),Rd21(0:NLm2),Rd22(0:NLm2)
      complex*16 Tu11(0:NLm2),Tu12(0:NLm2),Tu21(0:NLm2),Tu22(0:NLm2)
      complex*16 Td11(0:NLm2),Td12(0:NLm2),Td21(0:NLm2),Td22(0:NLm2)
      complex*16 Ru11(0:NLm2),Ru12(0:NLm2),Ru21(0:NLm2),Ru22(0:NLm2)
      complex*16 ea0,eb0,eta0(0:N1),neta0(0:N1)
      complex*16 W11,W12,W21,W22,V11,V12,V21,V22,InvDet,Df1,Df2

      if(Nl.gt.NLm) then
        stop 'Nl.gt.NLm in setPSY'
      Endif

      C1=(1.0,0.0)	  
      Ci=(0.0,1.0)
*     *******************************************************
*     set free surface reflection array Rf11,Rf12,Rf21,Rf22
*     and free-surface amplitude factor
*     *******************************************************
      do il=0,Nl+1
        eta0(il)=Vslo(alpha0(il),P)
        neta0(il)=Vslo(beta0(il),P)
      end do !il
      Temp=CI*w*h(0)*eta0(0)
      EA0=cdexp(Temp)
      Temp=CI*w*h(0)*neta0(0)
      EB0=cdexp(Temp)		

      call Rfs(P,eta0(0),neta0(0),rho0(0),beta0(0),
     &     Rf11,Rf12,Rf21,Rf22)
*     set reference medium interfacial scattering arrays
      do il=0,Nl
        ilp1=il+1
        il2=2*il
        cbeta1=beta0(il)
        cbeta2=beta0(ilp1)
*       set the reference medium interfacial scattering matrices
        call ScatMat(p,eta0(il),neta0(il),rho0(il),CBeta1,eta0(ilp1),
     &     neta0(ilp1),rho0(ilp1),CBeta2,Rd11(il2),Rd12(il2),Rd21(il2),
     &     Rd22(il2),Tu11(il2),Tu12(il2),Tu21(il2),Tu22(il2),Td11(il2),
     &     Td12(il2),Td21(il2),Td22(il2),Ru11(il2),Ru12(il2),Ru21(il2),
     &     Ru22(il2))
      end do

*     set layer scattering arrays
      do il=1,Nl
        il2=2*(il-1)+1
        call  SLVD(il,Nl,w,p,h,rho0,alpha0,beta0,rho,alpha,beta,
     &  Rd11(il2),Rd12(il2),Rd21(il2),Rd22(il2),Tu11(il2),Tu12(il2),
     &  Tu21(il2),Tu22(il2),Td11(il2),Td12(il2),Td21(il2),Td22(il2),
     &  Ru11(il2),Ru12(il2),Ru21(il2),Ru22(il2))
*        TYpe*,'Rd11(il2)=',Rd11(il2)
*        TYpe*,'Rd21(il2)=',Rd21(il2)
*        TYpe*,'Rd22(il2)=',Rd22(il2)

*        TYpe*,'Td11(il2)=',Td11(il2)
*        TYpe*,'Td21(il2)=',Td21(il2)
*        TYpe*,'Td22(il2)=',Td22(il2)

*        call SetS(il,Nl,w,p,h,rho0,alpha0,beta0,rho,alpha,beta,
*     &  Rd11(il2),Rd12(il2),Rd21(il2),Rd22(il2),Tu11(il2),Tu12(il2),
*     &  Tu21(il2),Tu22(il2),Td11(il2),Td12(il2),Td21(il2),Td22(il2),
*     &  Ru11(il2),Ru12(il2),Ru21(il2),Ru22(il2))

      end do
*******************************************************
      Nl2=2*Nl
*     Factor the System Matrix
      call    Factor(Nl2,T11,T12,T21,T22,U11,U12,U21,U22,
     &        R11,R12,R21,R22,S11,S12,S21,S22,Rd11,Rd12,Rd21,Rd22,
     &        Tu11,Tu12,Tu21,Tu22,Ru11,Ru12,Ru21,Ru22,Td11,Td12,Td21,
     &        Td22)


      call  RHS(NL2,Rd11,Rd12,Rd21,Rd22,Td11,Td12,Td21,Td22,
     &          Ru11,Ru12,Ru21,Ru22,Tu11,Tu12,Tu21,Tu22,
     &          B11,B12,B21,B22,SY)

      call  UpwardSweep(NL2,B11,B12,B21,B22,Ru11,Ru12,Ru21,Ru22,
     &      Tu11,Tu12,Tu21,Tu22,T11,T12,T21,T22,U11,U12,U21,U22,P1,
     &      P2,Q1,Q2)

*     set Df=downgoing from Free surface
*     Df=E0*Rf*E0*(1-R(0)*E0*Rf*E0)^(-1)*P0
      Rf11=Rf11*ea0*ea0
      Rf12=Rf12*eb0*ea0
      Rf21=Rf21*ea0*eb0
      Rf22=Rf22*eb0*eb0

*     U=(I-Ru*Rf)^(-1)
      W11=C1-R21(0)*Rf12-R22(0)*Rf22
      W12=R11(0)*Rf12+R12(0)*Rf22
      W21=R21(0)*Rf11+R22(0)*Rf21
      W22=C1-R11(0)*Rf11-R12(0)*Rf21
      InvDet=C1/(W11*W22-W12*W21)
      V11=W11*InvDet
      V12=W12*InvDet
      V21=W21*InvDet
      V22=W22*InvDet

*     W=Rf*V
      W11=Rf11*V11+Rf12*V21
      W12=Rf11*V12+Rf12*V22
      W21=Rf21*V11+Rf22*V21
      W22=Rf21*V12+Rf22*V22

*     Df=W*P(0) !downgoing from the free-surface
      Df1=(W11*P1(0)+W12*P2(0))
      Df2=(W21*P1(0)+W22*P2(0))


      call DownwardSweep(NL2,R11,R12,R21,R22,S11,S12,
     &                S21,S22,P1,P2,Q1,Q2,Df1,Df2,Xwk)
*     Load X
      Do il=1,Nl
        ioff4=4*(il-1)
        ioff8=8*(il-1)
        X(ioff4+1)=Xwk(ioff8+1)
        X(ioff4+2)=Xwk(ioff8+2)
        X(ioff4+3)=Xwk(ioff8+7)
        X(ioff4+4)=Xwk(ioff8+8)
      Enddo !il

      RETURN
      END

      Subroutine Factor(Nl,T11,T12,T21,T22,U11,U12,U21,U22,
     &     R11,R12,R21,R22,S11,S12,S21,S22,Rd11,Rd12,Rd21,Rd22,
     &     Tu11,Tu12,Tu21,Tu22,Ru11,Ru12,Ru21,Ru22,Td11,Td12,
     &     Td21,Td22)
      Implicit None
      integer il,nl,ilp1,nlm1
      complex*16 T11(0:Nl),T12(0:Nl),T21(0:Nl),T22(0:Nl)
      complex*16 U11(0:Nl),U12(0:Nl),U21(0:Nl),U22(0:Nl)
      complex*16 R11(0:Nl),R12(0:Nl),R21(0:Nl),R22(0:Nl)
      complex*16 S11(0:Nl),S12(0:Nl),S21(0:Nl),S22(0:Nl)
      complex*16 W11,W12,W21,W22,V11,V12,V21,V22,C1,InvDet
      complex*16 Rd11(0:Nl),Rd12(0:Nl),Rd21(0:Nl),Rd22(0:Nl)
      complex*16 Tu11(0:Nl),Tu12(0:Nl),Tu21(0:Nl),Tu22(0:Nl)
      complex*16 Td11(0:Nl),Td12(0:Nl),Td21(0:Nl),Td22(0:Nl)
      complex*16 Ru11(0:Nl),Ru12(0:Nl),Ru21(0:Nl),Ru22(0:Nl)
      C1=(1.0,0.0)
*     R=Rd(nl)
      R11(nl)=Rd11(nl)
      R12(nl)=Rd12(nl)
      R21(nl)=Rd21(nl)
      R22(nl)=Rd22(nl)
*	  type*,'nl,Rd11(nl)=',nl,Rd11(nl)
*	  type*,'nl,Rd12(nl)=',nl,Rd12(nl)
*	  type*,'nl,Rd21(nl)=',nl,Rd21(nl)
*	  type*,'nl,Rd22(nl)=',nl,Rd22(nl)

*     S=Td(nl)
      S11(nl)=Td11(nl)
      S12(nl)=Td12(nl)
      S21(nl)=Td21(nl)
      S22(nl)=Td22(nl)
*	  type*,'nl,S11(nl)=',nl,S11(nl)
*	  type*,'nl,S12(nl)=',nl,S12(nl)
*	  type*,'nl,S21(nl)=',nl,S21(nl)
*	  type*,'nl,S22(nl)=',nl,S22(nl)
      nlm1=nl-1
      do il=Nlm1,0,-1
        ilp1=il+1
*       U=(I-Ru*R(ilp1))^(-1)
        W11=C1-Ru21(il)*R12(ilp1)-Ru22(il)*R22(ilp1)
        W12=Ru11(il)*R12(ilp1)+Ru12(il)*R22(ilp1)
        W21=Ru21(il)*R11(ilp1)+Ru22(il)*R21(ilp1)
        W22=C1-Ru11(il)*R11(ilp1)-Ru12(il)*R21(ilp1)
        InvDet=C1/(W11*W22-W12*W21)
        U11(il)=W11*InvDet
        U12(il)=W12*InvDet
        U21(il)=W21*InvDet
        U22(il)=W22*InvDet
*		type*,'il,U11,U12,U21,U22=',il
*		type*,U11(il),U12(il),U21(il),U22(il)
*        type*
*       W=R*U
        W11=R11(ilp1)*U11(il)+R12(ilp1)*U21(il)
        W12=R11(ilp1)*U12(il)+R12(ilp1)*U22(il)
        W21=R21(ilp1)*U11(il)+R22(ilp1)*U21(il)
        W22=R21(ilp1)*U12(il)+R22(ilp1)*U22(il)

*       T=Tu*W
        T11(il)=Tu11(il)*W11+Tu12(il)*W21
        T12(il)=Tu11(il)*W12+Tu12(il)*W22
        T21(il)=Tu21(il)*W11+Tu22(il)*W21
        T22(il)=Tu21(il)*W12+Tu22(il)*W22
*		type*,'il,T11,T12,T21,T22=',il
*		type*,T11(il),T12(il),T21(il),T22(il)
*        type*


*       S=U*Td(il)
        S11(il)=U11(il)*Td11(il)+U12(il)*Td21(il)
        S12(il)=U11(il)*Td12(il)+U12(il)*Td22(il)
        S21(il)=U21(il)*Td11(il)+U22(il)*Td21(il)
        S22(il)=U21(il)*Td12(il)+U22(il)*Td22(il)

*		type*,'il,S11,S12,S21,S22=',il
*		type*,S11(il),S12(il),S21(il),S22(il)
*        type*

*       W=T*Td(il)
        W11=T11(il)*Td11(il)+T12(il)*Td21(il)
        W12=T11(il)*Td12(il)+T12(il)*Td22(il)
        W21=T21(il)*Td11(il)+T22(il)*Td21(il)
        W22=T21(il)*Td12(il)+T22(il)*Td22(il)

*       R=Rd+W
        R11(il)=Rd11(il)+W11
        R12(il)=Rd12(il)+W12
        R21(il)=Rd21(il)+W21
        R22(il)=Rd22(il)+W22

*		type*,'il,R11,R12,R21,R22=',il
*		type*,R11(il),R12(il),R21(il),R22(il)
*        type*
      end do !il

      return
      end

      Subroutine  RHS(NL2,Rd11,Rd12,Rd21,Rd22,Td11,Td12,Td21,Td22,
     &            Ru11,Ru12,Ru21,Ru22,Tu11,Tu12,Tu21,Tu22,B11,B12,
     &            B21,B22,SY)
      Implicit None
      Integer NL2,il,ioff,isc,ioffsc
      complex*16 SY(*)
      complex*16 B11(0:NL2),B12(0:NL2),B21(0:NL2),B22(0:NL2)
      complex*16 Rd11(0:NL2),Rd12(0:NL2),Rd21(0:NL2),Rd22(0:NL2)
      complex*16 Td11(0:NL2),Td12(0:NL2),Td21(0:NL2),Td22(0:NL2)
      complex*16 Ru11(0:NL2),Ru12(0:NL2),Ru21(0:NL2),Ru22(0:NL2)
      complex*16 Tu11(0:NL2),Tu12(0:NL2),Tu21(0:NL2),Tu22(0:NL2)
      complex*16 C0
      logical scat

      B11(0)=(0.0,0.0)
      B12(0)=(0.0,0.0)
      B21(0)=(0.0,0.0)
      B22(0)=(0.0,0.0)    
      scat=.true.
      isc=0
      Do il=1,NL2-1
        ioffsc=4*isc
        If(scat) then
          isc=isc+1
          B11(il)=SY(ioffsc+1)
          B12(il)=SY(ioffsc+2)
          B21(il)=SY(ioffsc+3)
          B22(il)=SY(ioffsc+4)
        else
          B11(il)=(0.0,0.0)
          B12(il)=(0.0,0.0)
          B21(il)=(0.0,0.0)
          B22(il)=(0.0,0.0)
        Endif
        scat=.not.scat
      Enddo
      B11(Nl2)=(0.0,0.0)
      B12(Nl2)=(0.0,0.0)
      B21(NL2)=(0.0,0.0)
      B22(NL2)=(0.0,0.0)    

      return
      end



      Subroutine DownwardSweep(NL,R11,R12,R21,R22,S11,S12,S21,
     &                         S22,P1,P2,Q1,Q2,Df1,Df2,Xwk)
      Implicit None
      Integer NL,il,ilm1,i1,i2
      complex*16 P1(0:Nl),P2(0:Nl),Q1(0:Nl),Q2(0:Nl)
      complex*16 Xwk(4*Nl)
      complex*16 R11(0:Nl),R12(0:Nl),R21(0:Nl),R22(0:Nl),Df1,Df2
      complex*16 S11(0:Nl),S12(0:Nl),S21(0:Nl),S22(0:Nl)

***************************************

      Xwk(1)=Q1(0)+S11(0)*Df1+S12(0)*Df2
      Xwk(2)=Q2(0)+S21(0)*Df1+S22(0)*Df2
      Xwk(3)=P1(1)+R11(1)*Xwk(1)+R12(1)*Xwk(2)
      Xwk(4)=P2(1)+R21(1)*Xwk(1)+R22(1)*Xwk(2)
****************************************
      do il=2,nl
        i1=(il-2)*4
        i2=i1+4
        ilm1=il-1
        Xwk(i2+1)=Q1(ilm1)+S11(ilm1)*Xwk(i1+1)+S12(ilm1)*Xwk(i1+2)
        Xwk(i2+2)=Q2(ilm1)+S21(ilm1)*Xwk(i1+1)+S22(ilm1)*Xwk(i1+2)
        Xwk(i2+3)=P1(il)+R11(il)*Xwk(i2+1)+R12(il)*Xwk(i2+2)
        Xwk(i2+4)=P2(il)+R21(il)*Xwk(i2+1)+R22(il)*Xwk(i2+2)
      end do !il
      return
      end


      Subroutine UpwardSweep(NL,B11,B12,B21,B22,Ru11,
     &           Ru12,Ru21,Ru22,Tu11,Tu12,Tu21,Tu22,T11,T12,
     &           T21,T22,U11,U12,U21,U22,P1,P2,Q1,Q2)
      Implicit None
      Integer NL,NLM1,il,ilp1
      complex*16 T11(0:Nl),T12(0:Nl),T21(0:Nl),T22(0:Nl)
      complex*16 U11(0:Nl),U12(0:Nl),U21(0:Nl),U22(0:Nl)
      complex*16 P1(0:Nl),P2(0:Nl)
      complex*16 Q1(0:Nl),Q2(0:Nl)
      complex*16 B11(0:Nl),B12(0:Nl)
      complex*16 B21(0:Nl),B22(0:Nl)
      complex*16 Tu11(0:Nl),Tu12(0:Nl),Tu21(0:Nl),Tu22(0:Nl)
      complex*16 Ru11(0:Nl),Ru12(0:Nl),Ru21(0:Nl),Ru22(0:Nl)
      complex*16 W11,W21,V11,V21,M11,M21
***********************************************
*     set P and Q	 
***********************************************
      P1(nl)=B11(nl)
      P2(nl)=B12(nl)
      Q1(nl)=B21(nl)
      Q2(nl)=B22(nl)
*	  Type*,'P1(nl)=',P1(nl)
*	  Type*,'P2(nl)=',P2(nl)
*	  Type*,'Q1(nl)=',Q1(nl)
*	  Type*,'Q2(nl)=',Q2(nl)
      nlm1=nl-1
      do il=nlm1,0,-1
        ilp1=il+1

*       W=Tu*P(ilp1)
        W11=Tu11(il)*P1(ilp1)+Tu12(il)*P2(ilp1)
        W21=Tu21(il)*P1(ilp1)+Tu22(il)*P2(ilp1)

*       M=Ru*P(ilp1)
        M11=Ru11(il)*P1(ilp1)+Ru12(il)*P2(ilp1)
        M21=Ru21(il)*P1(ilp1)+Ru22(il)*P2(ilp1)

*       V=B1+W
        V11=B11(il)+W11
        V21=B12(il)+W21

*       W=B2+M
        W11=B21(il)+M11
        W21=B22(il)+M21

*	    P(il)=V+T(il)*W
        P1(il)=V11+T11(il)*W11+T12(il)*W21
        P2(il)=V21+T21(il)*W11+T22(il)*W21
*		type*,'il,P1,P2=',il,P1(il),P2(il)

*       Q(il)=U(il)*W
        Q1(il)=U11(il)*W11+U12(il)*W21
        Q2(il)=U21(il)*W11+U22(il)*W21
*		type*,'il,Q1,Q2=',il,Q1(il),Q2(il)
      end do !il
      return
      end

      Subroutine GEMVmask(Nl,Np,w,p,h,rho0,alpha0,beta0,
     &                  rho,alpha,beta,getres,Y,X)
*     ********************************************************
*     set X=(1-P*dS)*Y
*     ********************************************************
      Implicit none
      Integer      Nl,Np,i,k,j,Mp,ip,il
      logical      getres
      Real*8       rho0(0:Nl),rho(0:Nl)
      complex*16   alpha0(0:Nl),beta0(0:Nl)
      complex*16   alpha(0:Nl),beta(0:Nl)	  
      Real*8       h(0:Nl)
      complex*16   w,p(Np)
      complex*16   X(4*Nl*Np),Y(4*Nl*Np)
      Integer      Nlmax,Npmax,NLM4
*     local variables !ensure sufficient space has been allocated
      parameter    (Nlmax=2,Npmax=45,NLM4=4*Nlmax*Npmax)
      complex*16   b(4*Nlmax),c(4*Nlmax),SY(4*Nlmax*Npmax)
      complex*16   Y1(4*Nlmax*Npmax),Dum
*      type*,'entered GEMVmask'
*	  Do i=1,4*Nl*Np
*	    Type*,'i,X(i)=',i,X(i)
*	  Enddo
*	  pause
      if(Nl.gt.NLmax) then
         stop 'Nl.gt.NLmax in GEMVmask'
       Endif

      if(Np.gt.Npmax) then
        stop 'Np.gt.Npmax in GEMVmask'
      Endif


*     SY=dS*Y
      call setSY(Nl,Np,w,p,h,rho0,alpha0,beta0,rho,alpha,beta,Y,SY)
*      type*,'left setSy'
*	  Do i =1, 4*Nl*Np
*	    Type*,'SY(i)=',i,SY(i)
*	  Enddo
*	  pause
*     Y1=P*SY
      Do ip=1,Np
        Do il=1,Nl
          Do j=1,4
            b((il-1)*4+j)=SY(4*Np*(il-1)+Np*(j-1)+ip)
          Enddo!j
        Enddo !il
        call setPSY(Nl,p(ip),w,h,rho0,alpha0,beta0,rho,alpha,beta,b,c)
*      type*,'left setPSY'
*	  Type*,'p(ip)=',p(ip)
*	  Do i =1, 4*Nl
*	    Type*,'SY(i)=',i,b(i)
*	  Enddo
*	  Do i =1, 4*Nl
*	    Type*,'PSY(i)=',i,c(i)
*	  Enddo
*	  pause
      Do il=1,Nl
        Do j=1,4
          Y1(4*Np*(il-1)+Np*(j-1)+ip)=c((il-1)*4+j)
        Enddo!j
      Enddo !il
      Enddo !ip
      
      if(getres)then
        Do i=1,4*Nl*Np
          X(i)=X(i)-Y(i)+Y1(i)
        Enddo !i
      else
        Do i=1,4*Nl*Np
          X(i)=Y(i)-Y1(i)
        Enddo !i
      endif

*      Type*,'Leaving amat'
*	  Do i=1,4*Nl*Np
*	    Type*,'i,X(i)=',i,X(i)
*	  Enddo !i
      return
      end

      subroutine Rfs(P,ETA,NETA,RHO,BETA,Rf11,Rf12,Rf21,Rf22)
*********************************************************
*     form the layer eigen arrays Md,Nd
*********************************************************
      implicit none
      Real*8    rho
      complex*16 beta,twomup,p
      complex*16 Rf11,Rf12,Rf21,Rf22
      complex*16 eta,neta,Nd(2,2),Nu(2,2)
      complex*16 ci,epsa,epsb,c1,InvDet
      Ci=dcmplx(0.0,1.0)
      C1=dcmplx(1.0,0.0)
      twomup=2.0*rho*beta*beta*p
      epsa=C1/Csqrt(2.0*rho*eta)
      epsb=C1/Csqrt(2.0*rho*neta)

      Nd(2,1)=(twomup*p-rho)*epsa
      Nd(1,2)=-(twomup*p-rho)*epsb
      Nd(1,1)=-twomup*eta*epsa
      Nd(2,2)=-twomup*neta*epsb

      Nu(2,1)=Nd(2,1)
      Nu(2,2)=-Nd(2,2)
      Nu(1,1)=-Nd(1,1)
      Nu(1,2)=Nd(1,2)

      InvDet=ND(1,1)*ND(2,2)-ND(1,2)*ND(2,1)
      If(cabs(InvDet).lt.1.E-3) then
        Pause'singular InvDet'
      else
        InvDet=(1.0,0.0)/InvDet
      endif
      Rf11=InvDet*(ND(1,2)*NU(2,1)-ND(2,2)*Nu(1,1))
      Rf12=InvDet*(ND(1,2)*NU(2,2)-ND(2,2)*Nu(1,2))
      Rf21=InvDet*(ND(2,1)*NU(1,1)-ND(1,1)*Nu(2,1))
      Rf22=InvDet*(ND(2,1)*NU(1,2)-ND(1,1)*Nu(2,2))

      return
      end

      Subroutine  ScatMat(p,eta1,neta1,rho1,beta1,eta2,neta2,rho2,
     &            beta2,Rd11,Rd12,Rd21,Rd22,Tu11,Tu12,Tu21,Tu22,
     &            Td11,Td12,Td21,Td22,Ru11,Ru12,Ru21,Ru22)
*     **************************************************************
*     plane-wave interfacial scattering matrices  
*     modified from Aki&Richards pp149-51
*     **************************************************************
      Real*8    RHO1,RHO2,rtrho1,rtrho2,m2
      complex*16 ETA1,NETA1,ETA2,NETA2,BETA1,BETA2
      complex*16 p,PSQ,CRHO1,CRHO2,DRHO,Dmu
      complex*16 rteta1,rteta2,rtneta1,rtneta2
      complex*16 Rd11,Rd12,Rd21,Rd22
      complex*16 Tu11,Tu12,Tu21,Tu22,Td11,Td12,Td21
      complex*16 Td22,Ru11,Ru12,Ru21,Ru22
      complex*16 PSQD,a,b,c,d,Det,E,F,G,H,Q,R,S,T,U,V
      complex*16 rtza1,rtza2,rtzb1,rtzb2

      m2=-2.0
      rtrho1=sqrt(rho1)
      rtrho2=sqrt(rho2)
      rteta1=csqrt(eta1)
      rteta2=csqrt(eta2)
      rtneta1=csqrt(neta1)
      rtneta2=csqrt(neta2)
      rtza1=rteta1*rtrho1
      rtza2=rteta2*rtrho2
      rtzb1=rtneta1*rtrho1
      rtzb2=rtneta2*rtrho2
      psq=p*p
      crho1=dcmplx(rho1,0.0)
      crho2=dcmplx(rho2,0.0)
      drho=crho2-crho1
      dmu=crho2*beta2*beta2-crho1*beta1*beta1

      d=2.0*dmu
      psqd=psq*d
      a=drho-psqd
      b=crho2-psqd
      c=crho1+psqd

      E=(b*eta1+c*eta2)
      F=(b*neta1+c*neta2)
      G=(a-d*eta1*neta2)
      H=(a-d*eta2*neta1)
      Det=E*F+G*H*psq
      If(Cabs(Det).lt.0.000001) then 
        TYpe*,'hit Stonley pole in ScatMat, p=',p
        Pause
      Endif
      E=E/Det
      F=F/Det
      G=G/Det
      H=H/Det

      Q=(B*ETA1-C*ETA2)*F
      R=(a+d*eta1*neta2)*H*psq
      S=(a*b+c*d*eta2*neta2)*p/Det
      T=(b*neta1-c*neta2)*E
      U=(a+d*eta2*neta1)*G*psq
      V=(a*c+b*d*eta1*neta1)*p/Det

      Rd11=Q-R
      Rd21=m2*rteta1*rtneta1*S
      Rd12=-Rd21
      Rd22=T-U
      Td11=2*rtza1*rtza2*F
      Td21=m2*rtza1*rtzb2*H*p
      Td12=2.0*rtzb1*rtza2*G*p
      Td22=2.0*rtzb1*rtzb2*E
      Tu11=Td11
      Tu21=-Td12
      Tu12=-Td21
      Tu22=Td22
      Ru11=-(Q+U)
      Ru21=m2*rteta2*rtneta2*V
      Ru12=-Ru21
      Ru22=-(T+R)	  
      return
      end	  


      Subroutine setSY(Nl,Np,w,p,h,rho0,alpha0,
     &                 beta0,rho,alpha,beta,Y,SY)
      Implicit none
      integer  Nl,Np,il,ioffo,ioffi,ipi,ipo,jo,ji
      complex*16  SY(4*Nl*Np),Y(4*Nl*Np)
      Real*8     rho0(0:Nl),h(0:Nl),rho(0:Nl)
      complex*16  w,p(Np)
      complex*16  alpha0(0:Nl),beta0(0:Nl),alpha(0:Nl),beta(0:Nl)
      complex*16  S(4,4),sum  
*      Type*,'entered setSY'
*	  pause
      Do il=1,4*Np*Nl
        SY(il)=(0.0,0.0)
      Enddo
*     set SY=S*Y
      Do il=1,Nl
        Do ipo=1,Np !loop over scattered slownesses
          ioffo=4*Np*(il-1)+ipo
          Do ipi=1,Np !loop over incident slownesses
            ioffi=4*Np*(il-1)+ipi

            If(ipi.eq.ipo) then
              S(1,1)=(0.0,0.0)
              S(1,2)=(0.0,0.0)
              S(1,3)=(0.0,0.0)
              S(1,4)=(0.0,0.0)
              S(2,1)=(0.0,0.0)
              S(2,2)=(0.0,0.0)
              S(2,3)=(0.0,0.0)
              S(2,4)=(0.0,0.0)
              S(3,1)=(0.0,0.0)
              S(3,2)=(0.0,0.0)
              S(3,3)=(0.0,0.0)
              S(3,4)=(0.0,0.0)
              S(4,1)=(0.0,0.0)
              S(4,2)=(0.0,0.0)
              S(4,3)=(0.0,0.0)
              S(4,4)=(0.0,0.0)
            else !offdiagonal coupling
              call SLVOD(il,Nl,w,p(ipi),p(ipo),h,rho0,alpha0,
     &         beta0,rho,alpha,beta,S)
            Endif

            Do jo=1,4	
              sum=(0.0,0.0)
                 Do ji=1,4
                   sum=sum+S(jo,ji)*Y(ioffi+Np*(ji-1))
                 Enddo !ji
                 SY(ioffo+Np*(jo-1))=SY(ioffo+Np*(jo-1))+sum
               Enddo !jo
             Enddo !ipi
*		  pause
          Enddo !ipo
*		pause
        End do !il
      return
      end



      subroutine SetD(P,ETA,NETA,RHO,BETA,Md,Nd,Mu,Nu)
*********************************************************
*     form the layer eigen arrays Md,Nd
*********************************************************
      implicit none
      Real*8    rho
      complex*16 beta,twomup,p
      complex*16 eta,neta,Nd(2,2),Md(2,2),Nu(2,2),Mu(2,2)
      complex*16 ci,epsa,epsb,c1
      Ci=dcmplx(0.0,1.0)
      C1=dcmplx(1.0,0.0)
      twomup=2.0*rho*beta*beta*p
      epsa=C1/Csqrt(2.0*rho*eta)
      epsb=C1/Csqrt(2.0*rho*neta)

      Md(2,1)=Ci*eta*epsa
      Md(2,2)=Ci*p*epsb
      Md(1,1)=Ci*p*epsa
      Md(1,2)=-Ci*neta*epsb

      Nd(2,1)=-Ci*(twomup*p-rho)*epsa
      Nd(1,2)=Ci*(twomup*p-rho)*epsb
      Nd(1,1)=Ci*twomup*eta*epsa
      Nd(2,2)=Ci*twomup*neta*epsb

      Mu(2,1)=-Md(2,1)
      Mu(2,2)=Md(2,2)
      Mu(1,1)=Md(1,1)
      Mu(1,2)=-Md(1,2)

      Nu(2,1)=Nd(2,1)
      Nu(2,2)=-Nd(2,2)
      Nu(1,1)=-Nd(1,1)
      Nu(1,2)=Nd(1,2)

      return
      end

      Subroutine SetS(il,Nl,w,p,h,rho0,alpha0,beta0,rho,alpha,beta,
     &                     Rd11,Rd12,Rd21,Rd22,Tu11,Tu12,Tu21,Tu22,
     &                     Td11,Td12,Td21,Td22,Ru11,Ru12,Ru21,Ru22)
      Implicit None
      Integer il,Nl,i,j
      Real*8     rho0(0:Nl),h(0:Nl)
      Real*8     rho(0:Nl)
      complex*16  w,p
      complex*16  alpha0(0:Nl),beta0(0:Nl),alpha(0:Nl),beta(0:Nl)
      complex*16  Rd11,Rd12,Rd21,Rd22,Tu11,Tu12,Tu21,Tu22
      complex*16  Td11,Td12,Td21,Td22,Ru11,Ru12,Ru21,Ru22


      complex*16 InvDet,a,b,c,d,C1,Ci
      complex*16 RRt11,RRt21,RRt12,RRt22
      complex*16 RRb11,RRb21,RRb12,RRb22
      complex*16 ea,eb,ea0,eb0,eta0,neta0,eta,neta
      complex*16 RRd11,RRd12,RRd21,RRd22
      complex*16 TTd11,TTd12,TTd21,TTd22
      complex*16 Tu11t,Tu12t,Tu21t,Tu22t
      complex*16 Td11t,Td12t,Td21t,Td22t
      complex*16 Rd11t,Rd12t,Rd21t,Rd22t
      complex*16 Ru11t,Ru12t,Ru21t,Ru22t
      complex*16 Tu11b,Tu12b,Tu21b,Tu22b
      complex*16 Td11b,Td12b,Td21b,Td22b
      complex*16 Rd11b,Rd12b,Rd21b,Rd22b
      complex*16 Ru11b,Ru12b,Ru21b,Ru22b
      complex*16 Vslo,cbeta1,cbeta2

*	  type*,'rho0=',rho0(il)
*	  type*,'rho=',rho(il)

*	  type*,'alpha0=',alpha0(il)
*	  type*,'alpha=',alpha(il)

*	  type*,'beta0=',beta0(il)
*	  type*,'beta=',beta(il)

      C1=dcmplx(1.0,0.0)
      Ci=(0.0,1.0)

      eta0=Vslo(alpha0(il),P)
      neta0=Vslo(beta0(il),P)

      eta=Vslo(alpha(il),P)
      neta=Vslo(beta(il),P)

      Ea0=cdexp(CI*w*h(il)*eta0)
      Eb0=cdexp(CI*w*h(il)*neta0)

      Ea=cdexp(CI*w*h(il)*eta)
      Eb=cdexp(CI*w*h(il)*neta)

****************************************************************
*     set top interface
      cbeta1=dcmplx(beta0(il),0.0)
      cbeta2=dcmplx(beta(il),0.0)

      call ScatMat(p,eta0,neta0,rho0(il),cbeta1,eta,
     &  neta,rho(il),cbeta2,Rd11t,Rd12t,Rd21t,Rd22t,
     &  Tu11t,Tu12t,Tu21t,Tu22t,Td11t,Td12t,
     &  Td21t,Td22t,Ru11t,Ru12t,Ru21t,Ru22t)

****************************************************************
*     set bottom interface

      cbeta1=dcmplx(beta(il),0.0)
      cbeta2=dcmplx(beta0(il),0.0)

      call ScatMat(p,eta,neta,rho(il),cbeta1,eta0,
     &  neta0,rho0(il),cbeta2,Rd11b,Rd12b,Rd21b,
     &  Rd22b,Tu11b,Tu12b,Tu21b,Tu22b,Td11b,Td12b,
     &  Td21b,Td22b,Ru11b,Ru12b,Ru21b,Ru22b)

****************************************************************

      RRd11=Rd11b
      RRd12=Rd12b
      RRd21=Rd21b
      RRd22=Rd22b

      RRt11=RRd11*ea*ea
      RRt12=RRd12*ea*eb
      RRt21=RRd21*ea*eb
      RRt22=RRd22*eb*eb

      a=(C1-RRt22*Ru22t-RRt12*Ru21t)
      b=(RRt22*Ru12t+RRt12*Ru11t)	  
      c=(RRt21*Ru22t+RRt11*Ru21t)
      d=(C1-RRt21*Ru12t-RRt11*Ru11t)
      InvDet=C1/(a*d-b*c)

      a=InvDet*a
      b=InvDet*b
      c=InvDet*c
      d=InvDet*d

      TTd11=Td11t*a+Td21t*b
      TTd12=Td12t*a+Td22t*b
      TTd21=Td11t*c+Td21t*d
      TTd22=Td12t*c+Td22t*d


      Rd11=Rd11t+RRt22*TTd21*Tu12t+RRt21*TTd11*Tu12t+
     &          RRt12*TTd21*Tu11t+RRt11*TTd11*Tu11t

      Rd21=Rd21t+RRt22*TTd21*Tu22t+RRt21*TTd11*Tu22t+
     &          RRt12*TTd21*Tu21t+RRt11*TTd11*Tu21t

      Rd22=Rd22t+RRt22*TTd22*Tu22t+RRt21*TTd12*Tu22t+
     &          RRt12*TTd22*Tu21t+RRt11*TTd12*Tu21t


      Td11=eb*TTd21*Td12b+ea*TTd11*Td11b 
      Td21=eb*TTd21*Td22b+ea*TTd11*Td21b
      Td22=eb*TTd22*Td22b+ea*TTd12*Td21b

      Rd12=-Rd21
      Tu11=Td11
      Tu12=-Td21
      Tu21=-Td21
      Tu22=Td22
      Td12=Td21
      Ru11=Rd11
      Ru12=Rd21
      Ru21=-Rd21
      Ru22=Rd22
      return
      end

      Real*8 FUNCTION KTAPER(KS,KLT,KRT,KF,K)
*     APPLIES A TUCKEY WINDOW TO THE SLOWNESS CONTOUR
      Real*8 KS,PI,KLT,KRT,KF,K
      parameter(Pi=3.14159265358979324)
      IF(k.ge.KLT .and. k.le.KRT) THEN
        KTAPER=1.0
      ELSEIF(k .gt. KRT .and. K .lt. KF) THEN
        KTAPER=0.5*(1.+COS(PI*(K-KRT)/(KF-KRT)))
      ELSEIF(K .gt. KS .and. K .Lt. KLT) THEN
        KTAPER=.5*(1.+COS(PI*(KLT-K)/(KLT-KS)))
      ELSE
        KTAPER=0.0
      END IF
      RETURN
      END

      complex*16 FUNCTION S(OMEGA,OmegaMax)
*     ***************************************************************
*     Fourier transform of Ricker wavelet
*     ****************************************************************
      complex*16 OMEGA,CI,Z,OmegaMax,Alfa,T0
      CI=(0.0,1.0)
      alfa=.12*omegamax
      T0=5.5/(ALFA*SQRT(2.0))
      Z=OMEGA*OMEGA/(4.*ALFA*ALFA)
      S=Z*cdexp(-Z+CI*OMEGA*T0)
      RETURN
      END


      complex*16   function couple(dk)
*     tanh attenuated sinc spectrum of x1<=x<=x2 pulse
      Implicit  none
      Real*8      dk,Pi,x1,x2,L,xmd2,xpd2,lambda
      parameter (lambda=100.0,x1=0.5,x2=1.5)
      parameter (L=(x2-x1),xmd2=0.5*L,xpd2=0.5*(x2+x1))
      parameter (Pi=3.14159265358979324)
      Real*8      e1,temp
      complex*16   e2
      If(abs(dk).gt.1E-10)then
        e1=exp(0.5*Pi*dk/lambda)
        e2=cdexp(dcmplx(0.0,-dk*xpd2))
        temp=2.0*Pi*sin(0.5*dk*L)*e1/(e1**2-1.0)/lambda
        couple=dcmplx(temp*dble(e2),temp*dimag(e2))
*        couple=dcmplx(0.0,0.0) !DELETE
      else
        couple=dcmplx(L)
*        couple=dcmplx(1.0,0.0) !DELETE
      Endif
      return
      end

      subroutine Emat(P,ETA,NETA,RHO,BETA,Md,Nd,Mu,Nu)
*********************************************************
*     form the layer eigen arrays Md,Nd
*********************************************************
      implicit none
      complex*16 twomup,beta,p
      Real*8    rho
      complex*16 eta,neta,Nd(2,2),Md(2,2),Nu(2,2),Mu(2,2),ci,epsa,epsb,c1
      Ci=dcmplx(0.0,1.0)
      C1=dcmplx(1.0,0.0)
      twomup=2.0*rho*beta*beta*p
      epsa=C1/Csqrt(2.0*rho*eta)
      epsb=C1/Csqrt(2.0*rho*neta)
      Md(2,1)=Ci*eta*epsa
      Md(2,2)=Ci*p*epsb
      Md(1,1)=Ci*p*epsa
      Md(1,2)=-Ci*neta*epsb
      Nd(2,1)=(twomup*p-rho)*epsa
      Nd(1,2)=-(twomup*p-rho)*epsb
      Nd(1,1)=-twomup*eta*epsa
      Nd(2,2)=-twomup*neta*epsb
      Mu(2,1)=-Md(2,1)
      Mu(2,2)=Md(2,2)
      Mu(1,1)=Md(1,1)
      Mu(1,2)=-Md(1,2)

      Nu(2,1)=Nd(2,1)
      Nu(2,2)=-Nd(2,2)
      Nu(1,1)=-Nd(1,1)
      Nu(1,2)=Nd(1,2)

      return
      end

******************************************************

       subroutine SLVOD(il,Nl,w,pi,po,h,rho0,alpha0,beta0,
     &           rho,alpha,beta,S)
       Implicit none
      Integer il,Nl,i,j,jp1
      Real*8    x1,x2,Eps
      complex*16 pi,po,w
      Parameter(Eps=1.0E-10)
      complex*16 S(4,4)
      complex*16 dS11,dS12,dS21,dS22,dS31,dS32,dS41,dS42
      complex*16 dS13,dS14,dS23,dS24,dS33,dS34,dS43,dS44
      Real*8     rho0(0:Nl),rho(0:Nl),h(0:Nl)
      complex*16  alpha0(0:Nl),beta0(0:Nl),alpha(0:Nl),beta(0:Nl)
      Real*8    dk
      complex*16 alphasq0,betasq0,alphasq,betasq,nu1,nu10
      complex*16 Ndi(2,2),Mdi(2,2),Nui(2,2),Mui(2,2)
      complex*16 Ndo(2,2),Mdo(2,2),Nuo(2,2),Muo(2,2)
      complex*16 dA21,dA34,dA13,dA24,dA31,dA42
      complex*16 Ci,C1,C0
      complex*16 Vslo,couple,formfac
      complex*16 etai,netai,etao,netao,eai,ebi,eao,ebo
      complex*16 eaoai,eaobi,eboai,ebobi
      complex*16 sum21,sum24,sum13,sum42,sum31,dewp,ewp


      Ci=(0.0,1.0)
      C0=(0.0,0.0)
      C1=(1.0,0.0)
      etai=Vslo(alpha0(il),pi)
      netai=Vslo(beta0(il),pi)
      etao=Vslo(alpha0(il),po)
      netao=Vslo(beta0(il),po)
      EAi=cdexp(Ci*w*h(il)*etai)
      EBi=cdexp(Ci*w*h(il)*netai)
      EAo=cdexp(Ci*w*h(il)*etao)
      EBo=cdexp(Ci*w*h(il)*netao)
*
*     **************************************************
*     set the Pi->Po coupling elements  Ci*w*<Po|dA|Pi>
*     ***************************************************
*
      dk=dble(w*(po-pi))
*	  type*,'dk,couple(dk)=',dk,couple(dk)
      formfac=Ci*w*couple(dk)
      alphasq0=alpha0(il)*alpha0(il)
      betasq0=beta0(il)*beta0(il)
      nu10=4.0*betasq0*(1.0-betasq0/alphasq0)
      alphasq=alpha(il)*alpha(il)
      betasq=beta(il)*beta(il)
      nu1=4.0*betasq*(1.0-betasq/alphasq)
      sum21=(betasq/alphasq-betasq0/alphasq0)*formfac
      sum24=(1.0/(rho(il)*alphasq)-1.0/(rho0(il)*alphasq0))*formfac
      sum13=(1.0/(rho(il)*betasq)-1.0/(rho0(il)*betasq0))*formfac
      sum42=(rho(il)-rho0(il))*formfac
      sum31=(rho(il)*nu1-rho0(il)*nu10)*formfac

      dA21=pi*2.0*sum21
      dA34=po*2.0*sum21
      dA24=sum24
      dA13=sum13
      dA42=sum42
      dA31=dA42-pi*po*sum31

*      type*,'dA21=',dA21
*      type*,'dA34=',dA34
*      type*,'dA24=',dA24
*      type*,'dA13=',dA13
*      type*,'dA42=',dA42
*      type*,'dA31=',dA31
*     ************************************************************
*     set the first-Born scattering elements for Pi->Po transition
*     ************************************************************
      call SetD(Pi,ETAi,NETAi,rho0(il),beta0(il),Mdi,Ndi,Mui,Nui)
      call SetD(Po,ETAo,NETAo,rho0(il),beta0(il),Mdo,Ndo,Muo,Nuo)

      dS11=-(Ndi(2,1)*dA34+Mdi(1,1)*dA31)*Mdo(1,1)+
     &      (-Ndi(2,1)*dA24-Mdi(1,1)*dA21)*Ndo(2,1)+Mdi(2,1)*
     &      Mdo(2,1)*dA42+Ndi(1,1)*Ndo(1,1)*dA13
      dS12=-(Ndi(2,2)*dA34+Mdi(1,2)*dA31)*Mdo(1,1)+(-Ndi(2,2)*dA24-
     &      Mdi(1,2)*dA21)*Ndo(2,1)+Mdi(2,2)*Mdo(2,1)*dA42+
     &      Ndi(1,2)*Ndo(1,1)*dA13
      dS13=-(Nui(2,1)*dA34+Mui(1,1)*dA31)*Mdo(1,1)+(-Nui(2,1)*dA24-
     &     Mui(1,1)*dA21)*Ndo(2,1)+Mui(2,1)*Mdo(2,1)*dA42+Nui(1,1)*
     &     Ndo(1,1)*dA13
      dS14=-(Nui(2,2)*dA34+Mui(1,2)*dA31)*Mdo(1,1)+(-Nui(2,2)*dA24-
     &      Mui(1,2)*dA21)*Ndo(2,1)+Mui(2,2)*Mdo(2,1)*dA42+Nui(1,2)*
     &      Ndo(1,1)*dA13
      dS21=(Ndi(2,1)*dA34+Mdi(1,1)*dA31)*Mdo(1,2)-(-Ndi(2,1)*dA24-
     &      Mdi(1,1)*dA21)*Ndo(2,2)-Mdi(2,1)*Mdo(2,2)*dA42-Ndi(1,1)*
     &      Ndo(1,2)*dA13
      dS22=(Ndi(2,2)*dA34+Mdi(1,2)*dA31)*Mdo(1,2)-(-Ndi(2,2)*dA24-
     &     Mdi(1,2)*dA21)*Ndo(2,2)-Mdi(2,2)*Mdo(2,2)*dA42-Ndi(1,2)*
     &     Ndo(1,2)*dA13
      dS23=(Nui(2,1)*dA34+Mui(1,1)*dA31)*Mdo(1,2)-(-Nui(2,1)*dA24-
     &     Mui(1,1)*dA21)*Ndo(2,2)-Mui(2,1)*Mdo(2,2)*dA42-Nui(1,1)*
     &     Ndo(1,2)*dA13
      dS24=(Nui(2,2)*dA34+Mui(1,2)*dA31)*Mdo(1,2)-(-Nui(2,2)*dA24-
     &     Mui(1,2)*dA21)*Ndo(2,2)-Mui(2,2)*Mdo(2,2)*dA42-Nui(1,2)*
     &     Ndo(1,2)*dA13
      dS31=-(Ndi(2,1)*dA34+Mdi(1,1)*dA31)*Muo(1,1)+(-Ndi(2,1)*dA24-
     &     Mdi(1,1)*dA21)*Nuo(2,1)+Mdi(2,1)*Muo(2,1)*dA42+Ndi(1,1)*
     &     Nuo(1,1)*dA13

      dS32=-(Ndi(2,2)*dA34+Mdi(1,2)*dA31)*Muo(1,1)+(-Ndi(2,2)*dA24-
     &     Mdi(1,2)*dA21)*Nuo(2,1)+Mdi(2,2)*Muo(2,1)*dA42+Ndi(1,2)*
     &     Nuo(1,1)*dA13

      dS33=-(Nui(2,1)*dA34+Mui(1,1)*dA31)*Muo(1,1)+(-Nui(2,1)*dA24-
     &     Mui(1,1)*dA21)*Nuo(2,1)+Mui(2,1)*Muo(2,1)*dA42+Nui(1,1)*
     &     Nuo(1,1)*dA13

      dS34=-(Nui(2,2)*dA34+Mui(1,2)*dA31)*Muo(1,1)+(-Nui(2,2)*dA24-
     &     Mui(1,2)*dA21)*Nuo(2,1)+Mui(2,2)*Muo(2,1)*dA42+Nui(1,2)*
     &     Nuo(1,1)*dA13
      dS41=(Ndi(2,1)*dA34+Mdi(1,1)*dA31)*Muo(1,2)-(-Ndi(2,1)*dA24-
     &    Mdi(1,1)*dA21)*Nuo(2,2)-Mdi(2,1)*Muo(2,2)*dA42-Ndi(1,1)*
     &    Nuo(1,2)*dA13
      dS42=(Ndi(2,2)*dA34+Mdi(1,2)*dA31)*Muo(1,2)-(-Ndi(2,2)*dA24-
     &    Mdi(1,2)*dA21)*Nuo(2,2)-Mdi(2,2)*Muo(2,2)*dA42-Ndi(1,2)*
     &    Nuo(1,2)*dA13
      dS43=(Nui(2,1)*dA34+Mui(1,1)*dA31)*Muo(1,2)-(-Nui(2,1)*dA24-
     &    Mui(1,1)*dA21)*Nuo(2,2)-Mui(2,1)*Muo(2,2)*dA42-Nui(1,1)*
     &    Nuo(1,2)*dA13
      dS44=(Nui(2,2)*dA34+Mui(1,2)*dA31)*Muo(1,2)-(-Nui(2,2)*dA24-
     &    Mui(1,2)*dA21)*Nuo(2,2)-Mui(2,2)*Muo(2,2)*dA42-Nui(1,2)*
     &    Nuo(1,2)*dA13

      S(1,1)=(eao*eai-C1)*dS11/(Ci*w*(etao+etai))
      S(1,2)=(eao*ebi-C1)*dS12/(Ci*w*(etao+netai))
      S(2,1)=(ebo*eai-C1)*dS21/(Ci*w*(netao+etai))
      S(2,2)=(ebo*ebi-C1)*dS22/(Ci*w*(netao+netai))

      if(Cabs(etai-etao).gt.1.E-08) then
        S(3,1)=(eai-eao)*dS31/(Ci*w*(etai-etao))
        S(1,3)=(eao-eai)*dS13/(Ci*w*(etao-etai))
      else
        S(3,1)=h(il)*eao*dS31
        S(1,3)=h(il)*eai*dS13
      endif

      if(Cabs(netai-etao).gt.1.E-08) then
         S(3,2)=(ebi-eao)*dS32/(Ci*w*(netai-etao))
      else
         S(3,2)=h(il)*eao*dS32
      endif

      if(Cabs(etai-netao).gt.1.E-08) then
         S(4,1)=(eai-ebo)*dS41/(Ci*w*(etai-netao))
      else
         S(4,1)=h(il)*ebo*dS41
      endif

      if(Cabs(netai-netao).gt.1.E-08) then	  
         S(4,2)=(ebi-ebo)*dS42/(Ci*w*(netai-netao))
         S(2,4)=(ebo-ebi)*dS24/(Ci*w*(netao-netai))
      else
         S(4,2)=h(il)*ebo*dS42
        S(2,4)=h(il)*ebi*dS24
      endif

      if(Cabs(etao-netai).gt.1.E-08) then	  
        S(1,4)=(eao-ebi)*dS14/(Ci*w*(etao-netai))
      else
        S(1,4)=h(il)*ebi*dS14
      Endif

      if(Cabs(netao-etai).gt.1.E-08) then	  
        S(2,3)=(ebo-eai)*dS23/(Ci*w*(netao-etai))
      else
        S(2,3)=h(il)*eai*dS23
      endif

      S(3,3)=(eao*eai-C1)*dS33/(Ci*w*(etao+etai))
      S(3,4)=(eao*ebi-C1)*dS34/(Ci*w*(etao+netai))
      S(4,3)=(ebo*eai-C1)*dS43/(Ci*w*(netao+etai))
      S(4,4)=(ebo*ebi-C1)*dS44/(Ci*w*(netao+netai))

      return
      End


      subroutine SLVD(il,Nl,w,pi,h,rho0,alpha0,beta0,rho,alpha,beta,
     &                     Rd11,Rd12,Rd21,Rd22,Tu11,Tu12,Tu21,Tu22,
     &                     Td11,Td12,Td21,Td22,Ru11,Ru12,Ru21,Ru22)
      Implicit    none
      Integer     il,Nl,i,j,jp1
      complex*16  pi,w
      Real*8      x1,x2,Eps,recipN
      Parameter   (Eps=1.0E-10)
      complex*16  Rd11,Rd12,Rd21,Rd22,Tu11,Tu12,Tu21,Tu22
      complex*16  Td11,Td12,Td21,Td22,Ru11,Ru12,Ru21,Ru22
      complex*16  dS11,dS12,dS21,dS22,dS31,dS32,dS41,dS42
      complex*16  dS13,dS14,dS23,dS24,dS33,dS34,dS43,dS44
      Real*8      rho0(0:Nl),rho(0:Nl),h(0:Nl)
      complex*16  alpha0(0:Nl),beta0(0:Nl),alpha(0:Nl),beta(0:Nl)
      Real*8      dk
      complex*16  alphasq0,betasq0,alphasq,betasq,nu1,nu10,contrast
      complex*16  Ndi(2,2),Mdi(2,2),Nui(2,2),Mui(2,2)
      complex*16  dA21,dA34,dA13,dA24,dA31,dA42
      complex*16  Ci,C1,C0
      complex*16  Vslo,couple,formfac
      complex*16  etai,netai,rteai,rtebi,eai,ebi,eaiai,ebiai,eaibi,ebibi
      complex*16  sum21,sum24,sum13,sum42,sum31,dewp,ewp


      If(w.eq.0.0) stop'static scattering case not yet implemented'
      Ci=(0.0,1.0)
      C0=(0.0,0.0)
      C1=(1.0,0.0)
      etai=Vslo(alpha0(il),pi)
      netai=Vslo(beta0(il),pi)
      Eai=cdexp(Ci*w*h(il)*etai)
      Ebi=cdexp(Ci*w*h(il)*netai)
*
*     **************************************************
*     set the Pi->Pi diagonal elements  Ci*w*<Pi|dA|Pi>
*     ***************************************************
*
      dk=0.0
      formfac=Ci*w*couple(dk)
      alphasq0=alpha0(il)*alpha0(il)
      betasq0=beta0(il)*beta0(il)
      nu10=4.0*betasq0*(1.0-betasq0/alphasq0)
      alphasq=alpha(il)*alpha(il)
      betasq=beta(il)*beta(il)
      nu1=4.0*betasq*(1.0-betasq/alphasq)
      sum21=(betasq/alphasq-betasq0/alphasq0)*formfac
      sum24=(1.0/(rho(il)*alphasq)-1.0/(rho0(il)*alphasq0))*formfac
      sum13=(1.0/(rho(il)*betasq)-1.0/(rho0(il)*betasq0))*formfac
      sum42=(rho(il)-rho0(il))*formfac
      sum31=(rho(il)*nu1-rho0(il)*nu10)*formfac

      dA21=pi*2.0*sum21
      dA34=pi*2.0*sum21
      dA24=sum24
      dA13=sum13
      dA42=sum42
      dA31=dA42-pi*pi*sum31
*      type*,'dA21=',dA21
*      type*,'dA34=',dA34
*      type*,'dA24=',dA24
*      type*,'dA13=',dA13
*      type*,'dA42=',dA42
*      type*,'dA31=',dA31

*     ************************************************************
*     set the first-Born scattering elements for Pi->Pi transition
*     ************************************************************
      call SetD(Pi,ETAi,NETAi,rho0(il),beta0(il),Mdi,Ndi,Mui,Nui)

      dS11=-(Ndi(2,1)*dA34+Mdi(1,1)*dA31)*Mdi(1,1)+
     &      (-Ndi(2,1)*dA24-Mdi(1,1)*dA21)*Ndi(2,1)+Mdi(2,1)*
     &      Mdi(2,1)*dA42+Ndi(1,1)*Ndi(1,1)*dA13
      dS12=-(Ndi(2,2)*dA34+Mdi(1,2)*dA31)*Mdi(1,1)+(-Ndi(2,2)*dA24-
     &      Mdi(1,2)*dA21)*Ndi(2,1)+Mdi(2,2)*Mdi(2,1)*dA42+
     &      Ndi(1,2)*Ndi(1,1)*dA13
      dS21=(Ndi(2,1)*dA34+Mdi(1,1)*dA31)*Mdi(1,2)-(-Ndi(2,1)*dA24-
     &      Mdi(1,1)*dA21)*Ndi(2,2)-Mdi(2,1)*Mdi(2,2)*dA42-Ndi(1,1)*
     &      Ndi(1,2)*dA13
      dS22=(Ndi(2,2)*dA34+Mdi(1,2)*dA31)*Mdi(1,2)-(-Ndi(2,2)*dA24-
     &     Mdi(1,2)*dA21)*Ndi(2,2)-Mdi(2,2)*Mdi(2,2)*dA42-Ndi(1,2)*
     &     Ndi(1,2)*dA13
      dS31=-(Ndi(2,1)*dA34+Mdi(1,1)*dA31)*Mui(1,1)+(-Ndi(2,1)*dA24-
     &     Mdi(1,1)*dA21)*Nui(2,1)+Mdi(2,1)*Mui(2,1)*dA42+Ndi(1,1)*
     &     Nui(1,1)*dA13
      dS32=-(Ndi(2,2)*dA34+Mdi(1,2)*dA31)*Mui(1,1)+(-Ndi(2,2)*dA24-
     &     Mdi(1,2)*dA21)*Nui(2,1)+Mdi(2,2)*Mui(2,1)*dA42+Ndi(1,2)*
     &     Nui(1,1)*dA13
      dS41=(Ndi(2,1)*dA34+Mdi(1,1)*dA31)*Mui(1,2)-(-Ndi(2,1)*dA24-
     &    Mdi(1,1)*dA21)*Nui(2,2)-Mdi(2,1)*Mui(2,2)*dA42-Ndi(1,1)*
     &    Nui(1,2)*dA13
      dS42=(Ndi(2,2)*dA34+Mdi(1,2)*dA31)*Mui(1,2)-(-Ndi(2,2)*dA24-
     &    Mdi(1,2)*dA21)*Nui(2,2)-Mdi(2,2)*Mui(2,2)*dA42-Ndi(1,2)*
     &    Nui(1,2)*dA13

      eaiai=eai*eai
      eaibi=eai*ebi
      ebiai=eaibi
      ebibi=ebi*ebi

      Rd11=(eaiai-C1)*dS11/(2.0*Ci*w*etai)
      Rd12=(eaibi-C1)*dS12/(Ci*w*(etai+netai))
      Rd21=(ebiai-C1)*dS21/(Ci*w*(etai+netai))
      Rd22=(ebibi-C1)*dS22/(2.0*Ci*w*netai)

      Td11=eai*(C1+h(il)*dS31)
      Td12=(ebi-eai)*dS32/(Ci*w*(netai-etai))
      Td21=(eai-ebi)*dS41/(Ci*w*(etai-netai))
      Td22=ebi*(C1+h(il)*dS42)

      Tu11=Td11
      Tu12=-Td21
      Tu21=-Td12
      Tu22=Td22

      Ru11=Rd11
      Ru12=-Rd12
      Ru21=-Rd21
      Ru22=Rd22
      return
      End

****************************************
      complex*16 function XSY(Nl,Np,w,p,h,rho0,alpha0,
     &                 beta0,rho,alpha,beta,X,Y)
      Implicit none
      integer  Nl,Np,il,ioffo,ioffi,ipi,ipo,jo,ji,j
      complex*16  Y(4*Nl*Np),X(4*Nl*Np)
      Real*8     h(0:Nl),rho0(0:Nl),rho(0:Nl)
      complex*16  alpha0(0:Nl),beta0(0:Nl)
      complex*16  alpha(0:Nl),beta(0:Nl),w,p(Np)
      complex*16  S(4,4),sum0,Sum1  
*      Type*,'entered XSY'
*	  pause
*	  Do il=1,4*Nl*Np
*	    Type*,'X(il),Y(il)=',il,X(il),Y(il)
*	  Enddo
*     set X*S*Y

*     set X=X(I(*)K)
      call swapper1(x,Np,Nl) !!!!DELETE

      Sum1=(0.0,0.0)
      j=0
      Do il=1,Nl
      Do ipo=1,Np !loop over scattered slownesses
      ioffo=4*Np*(il-1)+ipo
      Do ipi=1,Np !loop over incident slownesses
      ioffi=4*Np*(il-1)+ipi

      If(ipi.eq.ipo) then
	    S(1,1)=(0.0,0.0)
		S(1,2)=(0.0,0.0)
		S(1,3)=(0.0,0.0)
		S(1,4)=(0.0,0.0)
		S(2,1)=(0.0,0.0)
		S(2,2)=(0.0,0.0)
		S(2,3)=(0.0,0.0)
		S(2,4)=(0.0,0.0)
		S(3,1)=(0.0,0.0)
		S(3,2)=(0.0,0.0)
		S(3,3)=(0.0,0.0)
		S(3,4)=(0.0,0.0)
		S(4,1)=(0.0,0.0)
		S(4,2)=(0.0,0.0)
		S(4,3)=(0.0,0.0)
		S(4,4)=(0.0,0.0)
	  else 
	    call SLVOD(il,Nl,w,p(ipi),p(ipo),h,rho0,alpha0,
     &         beta0,rho,alpha,beta,S)
	  Endif
	  
	  Do jo=1,4	
	    Sum0=(0.0,0.0)
	    Do ji=1,4
	      Sum0=Sum0+S(jo,ji)*Y(ioffi+Np*(ji-1))
	    Enddo !ji
        j=j+1
	    Sum1=Sum1+X(ioffo+Np*(jo-1))*Sum0
	  Enddo !jo
	  
	  Enddo !ipi
	  Enddo !ipo
	  End do !il
	  XSY=Sum1

*     set X=X(I(*)K)
      call swapper1(x,Np,Nl) !!!!swap back again DELETE

	  return
	  end

      subroutine XS(Nl,Np,w,p,h,rho0,alpha0,
     &                 beta0,rho,alpha,beta,X,Y)
      Implicit none
      integer  Nl,Np,il,ioffo,ioffi,ipi,ipo,jo,ji
      complex*16  Y(4*Nl*Np),X(4*Nl*Np)
      Real*8     h(0:Nl),rho0(0:Nl),rho(0:Nl)
      complex*16  alpha0(0:Nl),beta0(0:Nl)
      complex*16  alpha(0:Nl),beta(0:Nl),w,p(Np)
      complex*16  S(4,4),sum  
*      Type*,'entered XS'
*	  pause
	  Do il=1,4*Nl*Np
	    Y(il)=(0.0,0.0)
	  Enddo
*     set Y=X*S
      Do il=1,Nl


        Do ipi=1,Np !loop over incident slownesses
          ioffi=4*Np*(il-1)+ipi

          Do ipo=1,Np !loop over scattered slownesses
            ioffo=4*Np*(il-1)+ipo

            If(ipi.eq.ipo) then
	           S(1,1)=(0.0,0.0)
			   S(1,2)=(0.0,0.0)
			   S(1,3)=(0.0,0.0)
			   S(1,4)=(0.0,0.0)
			   S(2,1)=(0.0,0.0)
			   S(2,2)=(0.0,0.0)
			   S(2,3)=(0.0,0.0)
			   S(2,4)=(0.0,0.0)
			   S(3,1)=(0.0,0.0)
			   S(3,2)=(0.0,0.0)
			   S(3,3)=(0.0,0.0)
			   S(3,4)=(0.0,0.0)
			   S(4,1)=(0.0,0.0)
			   S(4,2)=(0.0,0.0)
			   S(4,3)=(0.0,0.0)
			   S(4,4)=(0.0,0.0)
	         else 
	           call SLVOD(il,Nl,w,p(ipi),p(ipo),h,rho0,alpha0,
     &         beta0,rho,alpha,beta,S)
	         Endif
	  
	  Do ji=1,4	
	    Sum=(0.0,0.0)
	    Do jo=1,4
	      Sum=Sum+X(ioffo+Np*(jo-1))*S(jo,ji)
	    Enddo !ji
	    Y(ioffi+Np*(ji-1))=Y(ioffi+Np*(ji-1))+Sum
	  Enddo !jo
	  
	  Enddo !ipo
	  Enddo !ipi
	  End do !il
	  return
	  end

      complex*16 function XY(Nl,Np,X,Y)
*     set X*Y
      Implicit none
      integer  Nl,Np,i,N
      complex*16  Y(4*Nl*Np),X(4*Nl*Np),sum
      N=4*Nl*Np
      Sum=(0.0,0.0)
      Do i=1,N
        sum=sum+X(i)*Y(i)
	  End do !i
	  XY=Sum
	  return
	  end

      subroutine swapper(y,yswapp,Np,Nl)
      implicit none
      integer  i,j,Np,Mp,Nl,k
      complex*16  Dum,y(*),yswapp(*) 

      Mp=(Np+1)/2
      Do i=1,4*Nl
        k=(i-1)*Np
        Yswapp(k+Mp)=Y(k+Mp)
        Do j=1,Mp-1
          Yswapp(k+Mp-j)=Y(k+Mp+j)
          Yswapp(k+Mp+j)=Y(k+Mp-j)
        Enddo !j
       Enddo!i
       return
       end

      subroutine swapper1(y,Np,Nl)
      implicit none
      integer  i,j,Np,Mp,Nl,k
      complex*16  Dum,y(*) 

      Mp=(Np+1)/2
      Do i=1,4*Nl
        k=(i-1)*Np
        Do j=1,Mp-1
          Dum=Y(k+Mp+j)
          Y(k+Mp+j)=Y(k+Mp-j)
          Y(k+Mp-j)=Dum
        Enddo !j
       Enddo!i
       return
       end

      SUBROUTINE CMPLXGJ(A,N,NP,B,M,MP,IPIV,INDXR,INDXC,NMAX)	
      IMPLICIT NONE
      INTEGER  NMAX,NP,N,MP,M,IPIV(NMAX),INDXR(NMAX),INDXC(NMAX)
      INTEGER  IROW,ICOL,I,J,K,L,LL
      REAL*8     BIG,EPSMIN
      complex*16  A(NP,NP),B(NP,MP),DUM,C1,PIVINV
      PARAMETER (C1=(1.0,0.0),EPSMIN=1.0E-6)
      DO J=1,N
        IPIV(J)=0
      END DO
      DO I=1,N
        BIG=0.0
          DO J=1,N
            IF(IPIV(J) .NE. 1)THEN
              DO K=1,N
                 IF(IPIV(K) .EQ. 0)THEN
                   IF(CABS(A(J,K)).GE.BIG)THEN
                     BIG=CABS(A(J,K))
                     IROW=J
                     ICOL=K
                   END IF
                 ELSE IF (IPIV(K) .GT. 1) THEN
                   PAUSE 'IPIV(K) .GT. 1 MATRIX SINGULAR'
                 END IF
               END DO
             END IF
           END DO
           IPIV(ICOL)=IPIV(ICOL)+1
           IF(IROW .NE. ICOL)THEN
             DO L=1,N
               DUM=A(IROW,L)
               A(IROW,L)=A(ICOL,L)
               A(ICOL,L)=DUM
             END DO
             DO L=1,M
               DUM=B(IROW,L)
               B(IROW,L)=B(ICOL,L)
               B(ICOL,L)=DUM
             END DO
           END IF
           INDXR(I)=IROW
           INDXC(I)=ICOL
           IF(CABS(A(ICOL,ICOL)) .LT. EPSMIN) PAUSE 'SINGULAR MATRIX'
             PIVINV=C1/A(ICOL,ICOL)
             A(ICOL,ICOL)=C1
             DO L=1,N
               A(ICOL,L)=A(ICOL,L)*PIVINV
             END DO
             DO L=1,M
               B(ICOL,L)=B(ICOL,L)*PIVINV
             END DO
             DO LL=1,N
               IF(LL .NE. ICOL)THEN
                 DUM=A(LL,ICOL)
                 A(LL,ICOL)=(0.0,0.0)
                 DO L=1,N
                   A(LL,L)=A(LL,L)-A(ICOL,L)*DUM
                 END DO
                 DO L=1,M
                   B(LL,L)=B(LL,L)-B(ICOL,L)*DUM
                 END DO
               ENDIF
             END DO
           END DO
           DO L=N,1,-1
             IF(INDXR(L).NE.INDXC(L))THEN
             DO K=1,N
               DUM=A(K,INDXR(L))
               A(K,INDXR(L))=A(K,INDXC(L))
               A(K,INDXC(L))=DUM
             END DO
           END IF
         END DO
       RETURN
       END

