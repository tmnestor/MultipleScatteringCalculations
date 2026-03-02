      Program      GMRESP
*******************************************************************
*     GMRES with (1+PS) preconditioning and Householder factorization
*******************************************************************
      implicit     none
      INTEGER      Nk,Nl,Nlm1,Nl1,NL2
      INTEGER      iz,ix,Nh,Nhm1,j,il,i,ilm1
      Parameter    (Nk=1024,Nh=Nk/2,Nhm1=Nh-1)

      parameter    (Nl=2,Nlm1=Nl-1,NL1=Nl+1,NL2=NL+2)
	  
      Integer      Nx,Nz,Ns,jskip
      Parameter    (Nx=2, Nz=Nl, Ns=Nx*Nz, jskip=8)
	  
      Real*8       h(0:Nl),xs,zs,xr,zr,dela
      Parameter    (xr=0,xs=0)
      Real*8       rho0(0:Nl1),rho(Nx,Nz)
      Complex*16   alpha0(0:Nl1),beta0(0:Nl1)
      Complex*16   alpha(Nx,Nz),beta(Nx,Nz)
      Complex*16   w,CV
      Real*8       Pi,freq,QB(0:Nl1),rt3
      Parameter   (Pi=3.14159265358979324)

*******************************************************
*     storage for init
*******************************************************
      Complex*16  PC(Nk,9,Nz)

      Complex*16  U11(Nk,0:Nlm1)
      Complex*16  U21(Nk,0:Nlm1),U22(Nk,0:Nlm1)
      Complex*16  R11(Nk,Nl),R21(Nk,Nl),R22(Nk,Nl)
      Complex*16  S11(Nk,Nlm1),S12(Nk,Nlm1),S21(Nk,Nlm1),S22(Nk,Nlm1)
      Complex*16  P1(Nk,Nl),P2(Nk,Nl),Q1(Nk,Nl),Q2(Nk,Nl)
 
      Complex*16  Rd11(Nk),Rd12(Nk),Rd21(Nk),Rd22(Nk)
      Complex*16  Tu11(Nk),Tu12(Nk),Tu21(Nk),Tu22(Nk)
      Complex*16  Td11(Nk),Td12(Nk),Td21(Nk),Td22(Nk)
      Complex*16  Ru11(Nk),Ru12(Nk),Ru21(Nk),Ru22(Nk)
      Complex*16  Rf11(Nk),Rf21(Nk),Rf22(Nk)
      Complex*16  etavec(Nk,2),netavec(Nk,2)
      Complex*16  Ea(Nk,Nl),Eb(Nk,Nl)
	  
      Complex*16  PP11(Nx,Nz),PP22(Nx,Nz),PS31(Nx,Nz)
      Complex*16  PP31(Nx,Nz),PP33(Nx,Nz),PP51(Nx,Nz)
      Complex*16  PS11(Nx,Nz),PS51(Nx,Nz),PS22(Nx,Nz)
      Complex*16  SS11(Nx,Nz),SS51(Nx,Nz),SS22(Nx,Nz)

      Complex*16  PD(Nk),SD(Nk),PU(Nk),SU(Nk)
      Complex*16  sigP(Nk),sigS(Nk)

*******************************************************
*     iterative scheme storage
*******************************************************

      INTEGER      N, RESTRT, MAXIT, ITER, INFO, Nmeths
      parameter    ( MAXIT=40, Restrt=5, Nmeths=1)
      Parameter    (N=9*Ns)
      real*8       TOL, RESID, err(MAXIT,Nmeths)	  
      parameter    ( TOL= 1.D-6 )
      complex*16   X( N ),Svec( N ),Rvec( N ),SY( N ),PSY( N )
      complex*16   WORK(N,RESTRT+4), Hmat(N,RESTRT+2),dS
      REAL*8      c(RESTRT+1)
      Complex*16  Rmat(RESTRT,RESTRT),Jmat(2,RESTRT)

*******************************************************
*     set and solve system array A=I-PS to check accuracy
*******************************************************
*      Complex*16  A(N,N)
*      INTEGER     IPIV(N),INDXR(N),INDXC(N)
*      Logical     Exact
*      Parameter   (Exact=.true.)
*      Complex*16  exactTrans
	  

*******************************************************
*     set p Lattice Data (- Kmax<= w p <= Kmax)
*******************************************************
	  
*     spatial interval 
      Real*8      dx
*     wave number interval dk
      Real*8      dk,Kmax
      Complex*16  pvec(Nk)

*     material properties for Poisson solid
      Real*8      drho(Nx,Nz)
      Complex*16  dbeta(Nx,Nz)
	  
*     scattering transitions
      complex*16   setXKSY,GMRESresp,GMRESHresp, BORNresp
      complex*16   BCGresp ,BCGSTABresp  


      DATA        freq/2.D0/

      type*,'N=',N

************************************************************
      QB(0)=5.D1  
      beta0(0)=(2.8D0,0.D0)  
      rho0(0)=2.D0
      do il=1,Nl1
        QB(il)=5.D1    
        ilm1=il-1
        beta0(il)=beta0(ilm1)+.1D0
        rho0(il)=rho0(ilm1)+.1D0
      Enddo

      Do iz=1,Nz
        Do ix=1,Nx
          drho(ix,iz)=0.1D0*rho0(iz)
          dbeta(ix,iz)=0.1D0*beta0(iz)
        Enddo
      Enddo
************************************************************
      w=dcmplx(2.D0*Pi*freq,0.2D0)
      Kmax=32.0*Real(w/beta0(0))
      dk=2.D0*Kmax/dfloat(Nk)
      dx=Pi/Kmax
      dela=(jskip*dx)*(jskip*dx)
*************************************************************	  
*     Reference Layer Depths
      Do il=0,Nl
	h(il)=jskip*dx
      Enddo
      zs=0.5*h(0)
*
      zr=zs           !only case implemented
*	  
      rt3=DSqrt(3.0D0)
      Do 12 il=0,Nl1
	beta0(il)=CV(w,beta0(il),QB(il))
	alpha0(il)=rt3*beta0(il)
 12   continue

      Do 15 iz=1,nz
	Do 13 ix=1,nx
          rho(ix,iz)=rho0(iz)+drho(ix,iz)
          beta(ix,iz)=beta0(iz)+dbeta(ix,iz)
          alpha(ix,iz)=rt3*beta(ix,iz)
  13    continue
  15  continue
 
*     ********************************************************
*     FFT ordered slowness lattice
*     ********************************************************
      Do i=1,Nhm1
	    pvec(i+1)=float(i)*dk/w
	    pvec(i+1+Nh)=float(i-Nh)*dk/w
      Enddo
      pvec(1)=(0.0,0.0)
      pvec(Nh+1)=-Nh*dk/w
 
*     ************************************************************
*     init R/T coeffs ,factor System matrix and 
*     set PC,PP11,...,SS22
*     ************************************************************
      type*,'entering init'
      call Init(Nl,Nlm1,Nl1,Nx,Nz,jskip,
     &           Nk,dk,pvec,etavec,netavec,Svec,Rvec,
     &           PC,w,h,dela,xs,zs,xr,zr,rho0,alpha0,
     &           beta0,rho,alpha,beta,Rf11,Rf21,Rf22,
     &           Rd11,Rd12,Rd21,Rd22,Td11,Td12,
     &           Td21,Td22,Tu11,Tu12,Tu21,Tu22,Ru11,Ru12,
     &           Ru21,Ru22,Ea,Eb,P1,P2,Q1,Q2,R11,R21,R22,
     &           S11,S12,S21,S22,U11,U21,U22,PD,SD,PU,SU,
     &           SigP,SigS,PP11,PP22,PP31,PP33,PP51,
     &           PS11,PS51,PS22,PS31,SS11,SS51,SS22)
      type*,'leaving init'
*********************************************
*     generate exact solution if required
**********************************************

*      if(exact) then
*       set A(N,N),
*        Do j=1,N
*          Do i=1,N
*            X(i)=(0.0,0.0)
*          Enddo !i
*          X(j)=(1.D0,0.D0)
*         set col. j of A
*          type*,'setting column',j
*          call AtimesV(N,Nl,NLm1,Nk,Nx,Nz,jskip,
*     &            R11,R21,R22,S11,S12,S21,S22,U11,U21,U22,
*     &            P1,P2,Q1,Q2,PC,SY,PSY,PD,SD,PU,SU,Sigp,SigS,
*     &            ea,eb,PP11,PP22,PP31,PP33,PP51,PS11,PS51,
*     &            PS22,PS31,SS11,SS51,SS22,.false.,X,A(1,j))
*        Enddo !j


*        CALL ZCOPY( N, Svec, 1, X, 1 )

*        call CMPLXGJ(A,N,N,X,1,1,IPIV,INDXR,INDXC,N)
*       the exact transition
*        exactTrans=setXKSY(Nx,Nz,PP11,PP22,PP31,PP33,PP51,
*     &            PS11,PS51,PS22,PS31,SS11,SS51,SS22,Rvec,X)
*	    type*,'exactTrans=',exactTrans
*      endif

**************************************************************
*      test each iterative scheme in turn
**************************************************************
      type*,'GMRESH'	 
      Do i=1,N
        X(i)=(0.D0,0.D0)
      Enddo
      call GMRESH( N, Svec, X, RESTRT, WORK, Hmat, 
     &           MAXIT,ITER, TOL,RESID,INFO,err(1,1),
     &            c,Rmat,Jmat,Nl,NLm1,Nk,Nx,Nz,
     &            jskip,R11,R21,R22,S11,S12,S21,S22,U11,U21,
     &            U22,P1,P2,Q1,Q2,PC,SY,PSY,PD,SD,PU,SU,Sigp,
     &            SigS,ea,eb,PP11,PP22,PP31,PP33,PP51,PS11,
     &            PS51,PS22,PS31,SS11,SS51,SS22)
	  GMRESHresp=setXKSY(Nx,Nz,PP11,PP22,PP31,PP33,PP51,
     &            PS11,PS51,PS22,PS31,SS11,SS51,SS22,Rvec,X)

      type*,'info,iter,GMRESHresp=',info,iter,GMRESHresp

	  
      stop'normal completion'
      end



********************************************************
      subroutine GMRESH( N, B, X, RESTRT, WORK, Hmat, 
     &           MAXIT,ITER, TOL,RESID,INFO,err,
     &            c,Rmat,Jmat,Nl,NLm1,Nk,Nx,Nz,
     &            jskip,R11,R21,R22,S11,S12,S21,S22,U11,U21,
     &            U22,P1,P2,Q1,Q2,PC,SY,PSY,PD,SD,PU,SU,Sigp,
     &            SigS,ea,eb,PP11,PP22,PP31,PP33,PP51,PS11,
     &            PS51,PS22,PS31,SS11,SS51,SS22)

*
*  Purpose
*  =======
*
*  GMRES solves the linear system Ax = b using the
*  Generalized Minimal Residual iterative method
*  with QR factorization implememented with Householder
*  reflections
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
*  B       (input) complex*16 array, dimension N.
*          On entry, right hand side vector B.
*          Unchanged on exit.
*
*  X       (input/output) complex*16 array, dimension N.
*          On input, the initial guess; on exit, the iterated solution.
*
*  RESTRT  (input) INTEGER
*          Restart parameter, <= N. This parameter controls the amount
*          of memory required for matrix Hmat (see WORK and Hmat).
*
*  WORK    (workspace) complex*16 array, dimension (N,3).
*
*  Hmat    (workspace) complex*16 array, dimension (N,RESTRT+1).
*          This workspace is used for constructing and storing the
*          Householder reflections.
*
*  MAXIT    (input) INTEGER
*          The maximum iterations to be performed.
*
*  ITER    (output) INTEGER
*          On input, the maximum iterations to be performed.
*          On output, actual number of iterations performed.
*
*  TOL     (input) Real*8
*          The allowable convergence measure for
*          norm( b - A*x ) / norm( b ).
*
*  RESID   (output) Real*8
*          The final value of norm( b - A*x ) / norm( b ).
*
*  INFO    (output) INTEGER
*
*          =  0: Successful exit. Iterated approximate solution returned.
*
*          >  0: Convergence to tolerance not achieved. This will be
*                set to the number of iterations performed.
*
*          <  0: Illegal input parameter.
*
*                   -1: matrix dimension N < 0
*                   -3: Maximum number of iterations ITER <= 0.
*                   -4: N < RESTRT
*========================================================================
      Implicit    none
      INTEGER     N, MAXIT, ITER, INFO, RESTRT
      Real*8      TOL, RESID , err(MAXIT)
      complex*16  B( N ), X( N )
      complex*16  WORK(N,3), Hmat(N,RESTRT+2)
      REAL*8      c(RESTRT+1)
      Complex*16  Rmat(RESTRT,RESTRT),Jmat(2,RESTRT)

*************************************************************
*     storage passed to AtimesV
*************************************************************
      INTEGER     Nl,Nlm1,Nk,Nx,Nz,jskip
      Complex*16  PC(Nk,9,Nz), SY( N ), PSY( N )
      Complex*16  U11(Nk,0:Nlm1)
      Complex*16  U21(Nk,0:Nlm1),U22(Nk,0:Nlm1)
      Complex*16  R11(Nk,Nl),R21(Nk,Nl),R22(Nk,Nl)
      Complex*16  S11(Nk,Nlm1),S12(Nk,Nlm1)
      Complex*16  S21(Nk,Nlm1),S22(Nk,Nlm1)
      Complex*16  P1(Nk,Nl),P2(Nk,Nl),Q1(Nk,Nl),Q2(Nk,Nl)
      Complex*16  PD(Nk),SD(Nk),PU(Nk),SU(Nk)
      Complex*16  sigP(Nk),sigS(Nk)
      Complex*16  ea(Nk,1:Nl),eb(Nk,1:Nl)
      Complex*16  PP11(Nx,Nz),PP22(Nx,Nz),PS31(Nx,Nz)
      Complex*16  PP31(Nx,Nz),PP33(Nx,Nz),PP51(Nx,Nz)
      Complex*16  PS11(Nx,Nz),PS51(Nx,Nz),PS22(Nx,Nz)
      Complex*16  SS11(Nx,Nz),SS51(Nx,Nz),SS22(Nx,Nz)
*************************************************************
*     local variables
      Integer     i,j,k,R,Wvec,ek,IZAMAX
      Real*8      One, Zero, rmax, BNRM2, DZNRM2
      logical     sing
      Parameter   (ONE=1.D0,ZERO=0.D0)

      INFO = 0
*
*     Test the input parameters.
*

      IF ( N.LT.0 ) THEN
         INFO = -1
      ELSE IF ( MAXIT.LE.0 ) THEN
         INFO = -3
      ELSE IF ( N.LT.RESTRT+1 ) THEN
         INFO = -4
      ENDIF
      IF ( INFO.NE.0 ) RETURN
*
*     Alias workspace columns.
*
      R    = 1
      Wvec = 2
      ek   = 3
 

      type*,'in GMRESit,N=',N
************************************************************ 
*     set B=(1+P*S)*b0
************************************************************

*     set SY=S*B
	  call setSY(Nx,Nz,PP11,PP22,PP31,
     &     PP33,PP51,PS11,PS51,PS22,PS31,SS11,
     &     SS51,SS22,B,SY)

*     set PSY=P*SY
      call setPSY(Nl,Nlm1,Nk,Nx,Nz,jskip,
     &     R11,R21,R22,S11,S12,S21,S22,U11,U21,U22,
     &     ea,eb,PD,SD,PU,SU,sigP,sigS,P1,P2,Q1,Q2,PC,SY,PSY)

*      B=B+PSY        
       CALL ZAXPY( N, (1.D0,0.D0), PSY, 1, B, 1 )

************************************************************ 
*     set initial residual R=b0-Ax
************************************************************
*
*     (ek is temporary workspace here).
*
      CALL ZCOPY( N, B, 1, WORK(1,ek), 1 )
      IF ( DZNRM2( N, X, 1 ).NE.ZERO ) THEN
*
*        ek is temporary workspace here.
*
         CALL ZCOPY( N, B, 1, WORK(1,ek), 1 )

         call AtimesV(N,Nl,NLm1,Nk,Nx,Nz,jskip,
     &            R11,R21,R22,S11,S12,S21,S22,U11,U21,U22,
     &            P1,P2,Q1,Q2,PC,SY,PSY,PD,SD,PU,SU,Sigp,SigS,
     &            ea,eb,PP11,PP22,PP31,PP33,PP51,PS11,PS51,
     &            PS22,PS31,SS11,SS51,SS22,.true.,X,WORK(1,ek))

		 
      ENDIF
	  
      CALL ZCOPY( N, WORK( 1,ek ), 1, WORK( 1,R ), 1 )
*       Do i=1,N
*	      type*,'R0(i)=',i,WORK(i,R)
*      enddo
*      pause


      BNRM2 = DZNRM2( N, B, 1 )
      IF ( BNRM2.EQ.ZERO ) BNRM2 = ONE
	
*     initial approximation is already accurate enough
      IF ( ( DZNRM2( N, WORK(1,R), 1 ) / BNRM2 ).LT.TOL ) GO TO 70
*
      ITER = 0
*
   10 CONTINUE

         I = 0


*     set R=H1*R and store Housholder vector in Hmat(.,1)
      call setHkV(Hmat,WORK(1,R),N,1,N,RESTRT+1,c,sing)
*     righthand side vector Wvec=H1*R
      WORK(1,Wvec)=WORK(1,R)
      DO 99998 j=2,N
	    WORK(j,Wvec)=(0.0,0.0)
99998 continue

   30    CONTINUE
*
           I = I + 1
           ITER = ITER + 1

*
*     ******************************************
*     set Col-i of upper triangular Rmat(RESTRT,RESTRT)
*     ******************************************
*

*       set unit vector ek
        Do 9 j=1,N
	      WORK(j,ek)=(0.0,0.0)
9       continue
	    WORK(i,ek)=(1.0,0.0)
*
*       ******************************************
*       set ek=H(1)*....H(i)*ek
*       ******************************************
*
        call QV(Hmat,N,i,N,c,WORK(1,ek))
*
*       ******************************************
*       set r=(1-PdS)*ek
*       ******************************************
*
         call AtimesV(N,Nl,NLm1,Nk,Nx,Nz,jskip,
     &            R11,R21,R22,S11,S12,S21,S22,U11,U21,U22,
     &            P1,P2,Q1,Q2,PC,SY,PSY,PD,SD,PU,SU,Sigp,SigS,
     &            ea,eb,PP11,PP22,PP31,PP33,PP51,PS11,PS51,
     &            PS22,PS31,SS11,SS51,SS22,.false.,
     &            WORK(1,ek),WORK(1,R))

 
*	  Do j=1,N
*	     type*,'j,r(j)=',j,WORK(j,R)
*	  Enddo
      

*
*       ******************************************
*       set r=H(i)*....*H(1)*r
*       ******************************************
*
        call QtV(Hmat,N,i,N,c,WORK(1,R))
*	    Do j=1,N
*	     type*,'j,r(j)=',j,WORK(j,R)
*	    Enddo


		rmax= CDABS( WORK( i+IZAMAX(N-i,WORK(i+1,R),1), R) )

*        rmax=0.D0
*		do 11 j=i+1,N
*            rmax=max(rmax,CDabs(WORK(j,R)))
*11      continue

        If(rmax.gt.0.0)then
		  call setHkV(Hmat,WORK(1,R),N,i+1,N,RESTRT+1,c,sing)
		else !zero Householder vector
		  type*,'rmax=0.0 case'
		  Do 12 j= i+1,N         !recomment out
		    Hmat(j,i+1)=(0.0,0.0)
12        continue		
	    Endif

        if(i.gt.1) then
*
*       ******************************************
*       set r=J(i-1)*....*J(1)*r
*       ******************************************
*
        Do 13 j=1,i-1
          call rotV(WORK(1,R),j,Jmat(1,j),Jmat(2,j))
13      continue
        Endif
	    if(CDabs(WORK(i+1,R)).ne.0.0)then
	       Jmat(1,i)=WORK(i,R)
	       Jmat(2,i)=WORK(i+1,R)
	    else
	       Jmat(1,i)=(1.0,0.0)
	       Jmat(2,i)=(0.0,0.0)
	    Endif
        call rotV(WORK(1,R),i,Jmat(1,i),Jmat(2,i))
        call rotV(WORK(1,Wvec),i,Jmat(1,i),Jmat(2,i))
*
*       ******************************************
*       set Rmat(.,i)=[Rmat(.,i-1),r]
*       ******************************************
*
        Do 14 j=1,i
	      Rmat(j,i)=WORK(j,R)
14      continue


*
*       ******************************************
*       Test for convergence
*       ******************************************
*
        resid=CDabs(WORK(i+1,Wvec))/BNRM2
        type*,'iter,resid=',iter,resid
			err(iter)=resid
            IF ( RESID.LE.TOL ) THEN
              call UPDATEH(I, N, RESTRT, X, Hmat, Rmat, 
     &       WORK(1,ek), WORK(1,Wvec), c )
               GO TO 70
            ENDIF

            IF ( ITER.EQ.MAXIT ) GO TO 50
            IF ( I.LT.RESTRT )   GO TO 30
*
   50    CONTINUE
*
*        Compute current solution vector X.
*
         call UPDATEH(RESTRT, N, RESTRT, X, Hmat, Rmat, 
     &       WORK(1,ek), WORK(1,Wvec), c )
         Do j=1,N
           Type*,'j,X(j)=',j,X(j)
         Enddo

*
*        Compute residual vector R, find norm, then check for tolerance.
*        (ek is temporary workspace here.)
*
         CALL ZCOPY( N, B, 1, WORK( 1,ek ), 1 )
		 
         call AtimesV(N,Nl,NLm1,Nk,Nx,Nz,jskip,
     &            R11,R21,R22,S11,S12,S21,S22,U11,U21,U22,
     &            P1,P2,Q1,Q2,PC,SY,PSY,PD,SD,PU,SU,Sigp,SigS,
     &            ea,eb,PP11,PP22,PP31,PP33,PP51,PS11,PS51,
     &            PS22,PS31,SS11,SS51,SS22,.true.,X,WORK(1,ek))

         CALL ZCOPY( N, WORK(1,ek), 1, WORK( 1,R ), 1 )
		 
         RESID = DZNRM2( N, WORK( 1,R ), 1 ) / BNRM2

		 err(iter)=dlog10(resid)
         type*,'iter, resid=',iter, resid

         IF ( RESID.LE.TOL  ) GO TO 70
         IF ( ITER.EQ.MAXIT ) GO TO 60
*
*        Restart.
*
         type*,'restarting'
         GO TO 10
*
   60 CONTINUE
*
*     Iteration fails.
*
      INFO = ITER
      type*,'iteration fails'

      RETURN

   70 CONTINUE
*
*     Iteration successful; return.
*

      type*,'iteration successful'
      RETURN
*
*     End of GMRES
*
      END

*************************************************************
      subroutine UPDATEH(I, N, RESTRT, X, Hmat, Rmat, ek, Wvec, c )
      Implicit   none
      Integer    I,N,RESTRT
      REAL*8     c(N)
      Complex*16 X(N),Hmat(N,*),Rmat(RESTRT,RESTRT)
      Complex*16 ek(N),Wvec(N)

      integer    k,j
*
*     ******************************************
*     solve Rmat(.,i)*Wvec = Wvec
*     for upper triangualar Rmat
*     ******************************************

      CALL ZTRSV( 'UPPER', 'NOTRANS', 'NONUNIT', I, 
     &     Rmat, RESTRT, Wvec, 1 )

*
*     ******************************************
*     X=X+V*Wvec
*     ******************************************
*
 
      Do 250 k=1,i
*
*       set unit vector ek
        Do 210 j=1,N
		  ek(j)=(0.0,0.0)
210     continue
	    ek(k)=(1.0,0.0)
*
*       ******************************************
*       set ek=H(1)*....H(k)*ek
*       ******************************************
*
        call QV(Hmat,N,k,N,c,ek)
*	    Do 215 j=1,N
*	      X(j)=X(j)+ek(j)*Wvec(k)
*215     continue
        call ZAXPY(N, Wvec(k), ek, 1, X, 1 )
 
250   continue

      return
      end

      subroutine rotV(V,i,a11,a21)
      implicit none
      integer  i
      Complex*16  a11,a21,V(*)
      Real*8     CDabsa,CDabsb,m
      Complex*16  c,s,w,y
      CDabsa=CDabs(a11)
      CDabsb=CDabs(a21)
      m=sqrt(CDabsa**2+CDabsb**2)
      if(m.gt.1.D-10)then
      c=a11/m
      s=a21/m
      else
      c=(1.D0,0.D0)
      s=(0.D0,0.D0)
      endif
      y=V(i)
      w=V(i+1)
      V(i)=conjg(c)*y+conjg(s)*w
      V(i+1)=c*w-s*y
      return
      end
 
      subroutine QtV(a,n,k,np,c,V)
      implicit none
*     H(k)...*H(j)*..H(1)*V
      integer  n,np,k
      Complex*16  a(np,np),V(n)
      Real*8     c(n)
      integer  i,j
      Complex*16  zdotc,sum,tau
      do 13 j=1,k

*          sum=(0.0,0.0)
*          do 11 i=j,n
*            sum=sum+conjg(a(i,j))*V(i)
*11        continue
*          tau=sum/c(j)

          tau=zdotc(n-j+1,a(j,j), 1, V(j) ,1 )/c(j)

*          do 12 i=j,n
*            V(i)=V(i)-tau*a(i,j)
*12        continue

          call ZAXPY(n-j+1, -tau, a(j,j), 1, V(j), 1 )

13    continue
      return
      end
 
      subroutine QV(a,n,k,np,c,V)
      implicit none
*     H(1)...*H(j)*..H(k)*V
      integer  n,np,k
      Complex*16  a(np,np),V(n)
      Real*8     c(n)
      integer  i,j
      Complex*16  zdotc,sum,tau
      do 13 j=k,1,-1

*          sum=(0.0,0.0)
*          do 11 i=j,n
*            sum=sum+conjg(a(i,j))*V(i)
*11        continue
*          tau=sum/c(j)

           tau=zdotc(n-j+1,a(j,j), 1, V(j) ,1 )/c(j)

*          do 12 i=j,n
*            V(i)=V(i)-tau*a(i,j)
*12        continue

          call ZAXPY(n-j+1, -tau, a(j,j), 1, V(j), 1 )

13    continue
      return
      end
 
      SUBROUTINE setHkV(a,V,n,k,np,mp,c,sing)
      implicit none
      INTEGER  n,np,mp,k
      Complex*16  a(np,np),V(n)
      Real*8     c(n)
      LOGICAL  sing
      INTEGER  i,IZAMAX
      Real*8     DZNRM2,scale,L2,absakk
      Complex*16  sigma
	  if(k.lt.1) Pause'k.lt.1 in setHkV'
	  if(k.gt.Mp) Pause'k.gt.Mp in setHkV'
      sing=.false.

*      scale=0.
*      do 11 i=k,n
*            scale=max(scale,CDabs(V(i)))
*11     continue

       scale=CDABS(  V(IZAMAX(N,V,1)) )

       if(scale.lt.1.D-8)then

            c(k)=1.D0
       else

*         do 12 i=k,n
*              V(i)=V(i)/scale
*12       continue

          call ZDSCAL(N-k+1,1.D0/scale,V(k),1)

*         L2=0.0
*         do 13 i=k,n
*              L2=L2+V(i)*conjg(V(i))
*13       continue
*         L2=sqrt(L2)  ! L2 norm of V

         L2=DZNRM2(N-k+1, V(k), 1)

         absakk=CDabs(V(k))
         if(absakk.lt.1.D-6) then
           pause 'absakk.lt.1.D-6'
         endif

*        type*,'k,V(k)=',k,V(k)
         sigma=L2*V(k)/absakk
         a(k,k)=V(k)+sigma
         c(k)=L2*(L2+absakk)
         V(k)=-scale*sigma
         do 15 i=k+1,n
           a(i,k)=V(i)
	       V(i)=(0.0,0.0)
15      continue
      endif
      return
      END



* 
*     ================================================================
      DOUBLE PRECISION FUNCTION GETBREAK()
*
*     Get breakdown parameter tolerance; for the test routine,
*     set to machine precision.
*
      DOUBLE PRECISION EPS, DLAMCH
      eps=1.D-18
*
*      EPS = DLAMCH('EPS') !!!!!!!!AMMEND ACCORDINLY
*       type*,'eps=',eps
*      GETBREAK = EPS**2
      GETBREAK = (1.D08)*EPS

*
      RETURN
*
      END
*     ===============================================================
      subroutine ZROTG(da,db,c,s)
c
c     construct complex*16 givens plane rotation.
c
	  implicit none
      real*8        absa,absb
      complex*16    da,db,c,s,C0,C1,r
c
      C0=(0.D0,0.D0)
      C1=(1.D0,0.D0)
	  absa=abs(da)
	  absb=abs(db)

      if( (absa+absb) .eq. 0.0D0 ) then
         c = C1
         s = C0
         go to 20
	  Endif
	  if( absa .gt. absb ) then
	     r=absa*sqrt(1.D0+(absb/absa)**2)
	  else
	     r=absb*sqrt(1.D0+(absa/absb)**2)
	  endif
      c = da/r
      s = db/r
*      type*,'c,s=',c,s
   20 continue
      return
      end
	  
      subroutine  ZROT (n,dx,incx,dy,incy,c,s)
c
c     applies a complex*16 plane rotation.
c
	  implicit none
      complex*16    dx(1),dy(1),dtemp,c,s
      integer       i,incx,incy,ix,iy,n
c
      if(n.le.0)return
      if(incx.eq.1.and.incy.eq.1)go to 20
c
c       code for unequal increments or equal increments not equal
c         to 1
c
      ix = 1
      iy = 1
      if(incx.lt.0)ix = (-n+1)*incx + 1
      if(incy.lt.0)iy = (-n+1)*incy + 1
      do 10 i = 1,n
        dtemp = conjg(c)*dx(ix) + conjg(s)*dy(iy)
        dy(iy) = c*dy(iy) - s*dx(ix)
        dx(ix) = dtemp
        ix = ix + incx
        iy = iy + incy
   10 continue
      return
c
c       code for both increments equal to 1
c
   20 do 30 i = 1,n
        dtemp = conjg(c)*dx(i) + conjg(s)*dy(i)
        dy(i) = c*dy(i) - s*dx(i)
        dx(i) = dtemp
   30 continue
      return
      end


*******************************************************************************

	 
      SUBROUTINE Init(Nl,Nlm1,Nl1,Nx,Nz,jskip,
     &           Nk,dk,pvec,etavec,netavec,Svec,Rvec,
     &           PC,w,h,dela,xs,zs,xr,zr,rho0,alpha0,
     &           beta0,rho,alpha,beta,Rf11,Rf21,Rf22,
     &           Rd11,Rd12,Rd21,Rd22,Td11,Td12,
     &           Td21,Td22,Tu11,Tu12,Tu21,Tu22,Ru11,Ru12,
     &           Ru21,Ru22,Ea,Eb,P1,P2,Q1,Q2,R11,R21,R22,
     &           S11,S12,S21,S22,U11,U21,U22,PD,SD,PU,SU,
     &           SigP,SigS,PP11,PP22,PP31,PP33,PP51,
     &           PS11,PS51,PS22,PS31,SS11,SS51,SS22)
	
*     set up interfacial scattering arrays and factorise propagator
      Implicit    none
      INTEGER     Nl,NLm1,Nl1,Nx,Nz,Nk,jskip
      Real*8       h(0:Nl),xs,zs,xr,zr,dela
      Real*8       rho0(0:Nl1)
      Complex*16   alpha0(0:Nl1),beta0(0:Nl1)
	  Real*8       rho(Nx,Nz)
      Complex*16   alpha(Nx,Nz),beta(Nx,Nz)
	  Real*8      kx,dk,Pi
	  Parameter   (Pi=3.14159265358979324)

      Complex*16  w,pvec(Nk)

      Complex*16  PC(Nk,9,Nz)
      Complex*16  Svec(Nx,9,Nz),Rvec(Nx,9,Nz)


      Complex*16  U11(Nk,0:Nlm1)
      Complex*16  U21(Nk,0:Nlm1),U22(Nk,0:Nlm1)
      Complex*16  R11(Nk,Nl),R21(Nk,Nl),R22(Nk,Nl)
      Complex*16  S11(Nk,Nlm1),S12(Nk,Nlm1)
      Complex*16  S21(Nk,Nlm1),S22(Nk,Nlm1)
      Complex*16  P1(Nk,Nl),P2(Nk,Nl),Q1(Nk,Nl),Q2(Nk,Nl)
 
      Complex*16  Rd11(Nk),Rd12(Nk)
      Complex*16  Rd21(Nk),Rd22(Nk)
      Complex*16  Tu11(Nk),Tu12(Nk)
      Complex*16  Tu21(Nk),Tu22(Nk)
      Complex*16  Td11(Nk),Td12(Nk)
      Complex*16  Td21(Nk),Td22(Nk)
      Complex*16  Ru11(Nk),Ru12(Nk)
      Complex*16  Ru21(Nk),Ru22(Nk)
      Complex*16  Rf11(Nk)
      Complex*16  Rf21(Nk),Rf22(Nk)
      Complex*16  etavec(Nk,2),netavec(Nk,2)
      Complex*16  Ea(Nk,Nl),Eb(Nk,Nl)

      Complex*16  PD(Nk),SD(Nk),PU(Nk),SU(Nk)
      Complex*16  sigP(Nk),sigS(Nk)
	  
	  Complex*16  PP11(Nx,Nz),PP22(Nx,Nz),PS31(Nx,Nz)
	  Complex*16  PP31(Nx,Nz),PP33(Nx,Nz),PP51(Nx,Nz)
	  Complex*16  PS11(Nx,Nz),PS51(Nx,Nz),PS22(Nx,Nz)
	  Complex*16  SS11(Nx,Nz),SS51(Nx,Nz),SS22(Nx,Nz)
 
      Complex*16  Ka0,Kb0,kz,C0,C1,Ci,eas0,ebs0,easf,ebsf
      Complex*16  W11,W12,W21,W22,InvDet,temp,temp1
      Complex*16  tempS11,tempS12,tempS21,tempS22
      Complex*16  tempR11,tempR12,tempR21,tempR22
	  Complex*16  MT11,MT12,MT21,MT22,MB11,MB12,MB21,MB22
	  Complex*16  V11,V12,V21,V22,RRu11,RRu12,RRu21,RRu22
      integer     ik,il,ilp1
 
      C0=(0.D0,0.D0)
      C1=(1.D0,0.D0)	  	
      Ci=(0.D0,1.D0)

      Do il=1,Nl
	    Do ik=1,Nk
	      P1(ik,il)=C0	
	      P2(ik,il)=C0	
	      Q1(ik,il)=C0	
	      Q2(ik,il)=C0	
		Enddo
	  Enddo
 
      call Vslo(Nk,alpha0(Nl+1),pvec,etavec(1,1))
      call Vslo(Nk,beta0(Nl+1),pvec,netavec(1,1))

*******************************************************
*     set R,T,U,S
*******************************************************
      il=Nl
	  ilp1=il+1

      Do ik=1,Nk
	    etavec(ik,2)=etavec(ik,1)
	    netavec(ik,2)=netavec(ik,1)
	  Enddo
*     set the vertical slowness
      call Vslo(Nk,alpha0(il),pvec,etavec(1,1))
      call Vslo(Nk,beta0(il),pvec,netavec(1,1))

*     set the reference medium interfacial scattering matrices at interface Nl
      call IntScat(Nk,pvec,etavec(1,1),netavec(1,1),
     &    rho0(il),beta0(il),etavec(1,2),netavec(1,2),
     &    rho0(ilp1),beta0(ilp1),Rd11,Rd12,Rd21,Rd22,Tu11,Tu12,
     &    Tu21,Tu22,Td11,Td12,Td21,Td22,Ru11,Ru12,Ru21,Ru22)
	 

      DO ik=1,Nk
*       set half layer phase shift
        Ea(ik,il)=CDexp(0.5D0*Ci*w*h(il)*etavec(ik,1))
        Eb(ik,il)=CDexp(0.5D0*Ci*w*h(il)*netavec(ik,1))
		
*       R=Ea(nl)*Rd(nl)*Ea(nl)
		R11(ik,Nl)=ea(ik,Nl)*Rd11(ik)*ea(ik,Nl)
		R21(ik,Nl)=eb(ik,Nl)*Rd21(ik)*ea(ik,Nl)
		R22(ik,Nl)=eb(ik,Nl)*Rd22(ik)*eb(ik,Nl)
      Enddo

*       set PC for layer Nl
	Ka0=w/alpha0(il)
	Kb0=w/beta0(il)
	Do ik=1,Nk
	  kx=w*pvec(ik)
*P
          kz=w*etavec(ik,1)
          PC(ik,3,il)=CDsqrt(dk/(Pi*kz))
          temp=((kx/ka0)+Ci*(kz/ka0))
	  PC(ik,4,il)=PC(ik,3,il)*temp
	  PC(ik,5,il)=PC(ik,4,il)*temp
          temp=((kx/ka0)-Ci*(kz/ka0))
	  PC(ik,2,il)=PC(ik,3,il)*temp
	  PC(ik,1,il)=PC(ik,2,il)*temp
*SV	  
	  kz=w*netavec(ik,1)
	  temp1=CDsqrt(dk/(Pi*kz))
	  temp=((kx/kb0)+Ci*(kz/kb0))
	  PC(ik,8,il)=temp1*temp
	  PC(ik,9,il)=PC(ik,8,il)*temp
	  temp=((kx/kb0)-Ci*(kz/kb0))
	  PC(ik,7,il)=temp1*temp
	  PC(ik,6,il)=PC(ik,7,il)*temp
	Enddo

      

*      pause'check point 1'
 
      DO 99998 IL=NLM1,1,-1
        ilp1=il+1

        Do ik=1,Nk
	      etavec(ik,2)=etavec(ik,1)
	      netavec(ik,2)=netavec(ik,1)
	    Enddo
*       set the vertical slowness
        call Vslo(Nk,alpha0(il),pvec,etavec(1,1))
        call Vslo(Nk,beta0(il),pvec,netavec(1,1))

*       set the reference medium interfacial scattering matrices at interface Nl
        call IntScat(Nk,pvec,etavec(1,1),netavec(1,1),
     &    rho0(il),beta0(il),etavec(1,2),netavec(1,2),
     &    rho0(ilp1),beta0(ilp1),Rd11,Rd12,Rd21,Rd22,Tu11,Tu12,
     &    Tu21,Tu22,Td11,Td12,Td21,Td22,Ru11,Ru12,Ru21,Ru22)
	 

*       set half layer phase shift
        DO ik=1,Nk
          Ea(ik,il)=CDexp(0.5D0*Ci*w*h(il)*etavec(ik,1))
          Eb(ik,il)=CDexp(0.5D0*Ci*w*h(il)*netavec(ik,1))
        Enddo

        DO ik=1,Nk
		  Rd11(ik)=ea(ik,il)*Rd11(ik)*ea(ik,il)
		  Rd21(ik)=eb(ik,il)*Rd21(ik)*ea(ik,il)
		  Rd22(ik)=eb(ik,il)*Rd22(ik)*eb(ik,il)

		  Td11(ik)=ea(ik,ilp1)*Td11(ik)*ea(ik,il)
		  Td12(ik)=ea(ik,ilp1)*Td12(ik)*eb(ik,il)
		  Td21(ik)=eb(ik,ilp1)*Td21(ik)*ea(ik,il)
		  Td22(ik)=eb(ik,ilp1)*Td22(ik)*eb(ik,il)

		  Tu11(ik)=ea(ik,il)*Tu11(ik)*ea(ik,ilp1)
		  Tu12(ik)=ea(ik,il)*Tu12(ik)*eb(ik,ilp1)
		  Tu21(ik)=eb(ik,il)*Tu21(ik)*ea(ik,ilp1)
		  Tu22(ik)=eb(ik,il)*Tu22(ik)*eb(ik,ilp1)

		  Ru11(ik)=ea(ik,ilp1)*Ru11(ik)*ea(ik,ilp1)
		  Ru12(ik)=ea(ik,ilp1)*Ru12(ik)*eb(ik,ilp1)
		  Ru21(ik)=eb(ik,ilp1)*Ru21(ik)*ea(ik,ilp1)
		  Ru22(ik)=eb(ik,ilp1)*Ru22(ik)*eb(ik,ilp1)

*      type*,'ik,Rd11(ik)=',ik,Rd11(ik)
*      type*,'ik,Rd12(ik)=',ik,Rd12(ik)
*      type*,'ik,Rd21(ik)=',ik,Rd21(ik)
*      type*,'ik,Rd22(ik)=',ik,Rd22(ik)

*      type*,'ik,Td11(ik)=',ik,Td11(ik)
*      type*,'ik,Td12(ik)=',ik,Td12(ik)
*      type*,'ik,Td21(ik)=',ik,Td21(ik)
*      type*,'ik,Td22(ik)=',ik,Td22(ik)

*      type*,'ik,Ru11(ik)=',ik,Ru11(ik)
*      type*,'ik,Ru12(ik)=',ik,Ru12(ik)
*      type*,'ik,Ru21(ik)=',ik,Ru21(ik)
*      type*,'ik,Ru22(ik)=',ik,Ru22(ik)

*      type*,'ik,Tu11(ik)=',ik,Tu11(ik)
*      type*,'ik,Tu12(ik)=',ik,Tu12(ik)
*      type*,'ik,Tu21(ik)=',ik,Tu21(ik)
*      type*,'ik,Tu22(ik)=',ik,Tu22(ik)

       Enddo


*      pause'check point 2'
 
      DO 99997 ik=1,Nk
*      pause'check point 3'

*       MT=R(ilp1)
        MT11=R11(ik,ilp1)
        MT12=-R21(ik,ilp1)
        MT21=R21(ik,ilp1)
        MT22=R22(ik,ilp1)
 
*       W=(I-Ru*MT)^(-1)
	W11=C1-Ru21(ik)*MT12-Ru22(ik)*MT22
	W12=Ru11(ik)*MT12+Ru12(ik)*MT22
	W21=Ru21(ik)*MT11+Ru22(ik)*MT21
	W22=C1-Ru11(ik)*MT11-Ru12(ik)*MT21
	InvDet=C1/(W11*W22-W12*W21)


	W11=W11*InvDet
	W12=W12*InvDet
	W21=W21*InvDet
	W22=W22*InvDet

*       U=((I-Ru*MT)^(-1))*Ru
	U11(ik,il)=W11*Ru11(ik)+W12*Ru21(ik)
	U21(ik,il)=W21*Ru11(ik)+W22*Ru21(ik)
	U22(ik,il)=W21*Ru12(ik)+W22*Ru22(ik)
		
		
*       S=((I-Ru*MT)^(-1))*Td
        S11(ik,il)=W11*Td11(ik)+W12*Td21(ik)
	S12(ik,il)=W11*Td12(ik)+W12*Td22(ik)
	S21(ik,il)=W21*Td11(ik)+W22*Td21(ik)
	S22(ik,il)=W21*Td12(ik)+W22*Td22(ik)

*       W=MT*S
        W11=MT11*S11(ik,il)+MT12*S21(ik,il)
	W12=-MT11*S21(ik,il)+MT12*S22(ik,il)
	W21=MT21*S11(ik,il)+MT22*S21(ik,il)
	W22=-MT21*S21(ik,il)+MT22*S22(ik,il)
 
 
*       R=Rd+Tu*W
        R11(ik,il)=Rd11(ik)+Tu11(ik)*W11+Tu12(ik)*W21
	R21(ik,il)=Rd21(ik)+Tu21(ik)*W11+Tu22(ik)*W21
	R22(ik,il)=Rd22(ik)+Tu21(ik)*W12+Tu22(ik)*W22

99997 continue

*       set PC for layer il

	Ka0=w/alpha0(il)
	Kb0=w/beta0(il)
	Do ik=1,Nk
	  kx=w*pvec(ik)
*P
          kz=w*etavec(ik,1)
          PC(ik,3,il)=CDsqrt(dk/(Pi*kz))
	  temp=((kx/ka0)+Ci*(kz/ka0))
	  PC(ik,4,il)=PC(ik,3,il)*temp
	  PC(ik,5,il)=PC(ik,4,il)*temp
	  temp=((kx/ka0)-Ci*(kz/ka0))
	  PC(ik,2,il)=PC(ik,3,il)*temp
	  PC(ik,1,il)=PC(ik,2,il)*temp
*SV	  
	  kz=w*netavec(ik,1)
	  temp1=CDsqrt(dk/(Pi*kz))
	  temp=((kx/kb0)+Ci*(kz/kb0))
	  PC(ik,8,il)=temp1*temp
	  PC(ik,9,il)=PC(ik,8,il)*temp
	  temp=((kx/kb0)-Ci*(kz/kb0))
	  PC(ik,7,il)=temp1*temp
	  PC(ik,6,il)=PC(ik,7,il)*temp
	Enddo
 
99998 continue

****************************************************************************
*       set the reference medium interfacial scattering matrices at interface 0
****************************************************************************
        il=0
	ilp1=il+1

        Do ik=1,Nk
	  etavec(ik,2)=etavec(ik,1)
	  netavec(ik,2)=netavec(ik,1)
	Enddo
*       set the vertical slowness
        call Vslo(Nk,alpha0(il),pvec,etavec(1,1))
        call Vslo(Nk,beta0(il),pvec,netavec(1,1))

*       set the reference medium interfacial scattering matrices at interface Nl
        call IntScat(Nk,pvec,etavec(1,1),netavec(1,1),
     &    rho0(il),beta0(il),etavec(1,2),netavec(1,2),
     &    rho0(ilp1),beta0(ilp1),Rd11,Rd12,Rd21,Rd22,Tu11,Tu12,
     &    Tu21,Tu22,Td11,Td12,Td21,Td22,Ru11,Ru12,Ru21,Ru22)

*       set half layer phase shift

        DO ik=1,Nk
          Eas0=CDexp(Ci*w*(h(il)-zs)*etavec(ik,1))
          Ebs0=CDexp(Ci*w*(h(il)-zs)*netavec(ik,1))

	  Rd11(ik)=eas0*Rd11(ik)*eas0
	  Rd12(ik)=eas0*Rd12(ik)*ebs0
	  Rd21(ik)=ebs0*Rd21(ik)*eas0
	  Rd22(ik)=ebs0*Rd22(ik)*ebs0

	  Td11(ik)=ea(ik,ilp1)*Td11(ik)*eas0
	  Td12(ik)=ea(ik,ilp1)*Td12(ik)*ebs0
	  Td21(ik)=eb(ik,ilp1)*Td21(ik)*eas0
	  Td22(ik)=eb(ik,ilp1)*Td22(ik)*ebs0

	  Tu11(ik)=eas0*Tu11(ik)*ea(ik,ilp1)
	  Tu12(ik)=eas0*Tu12(ik)*eb(ik,ilp1)
	  Tu21(ik)=ebs0*Tu21(ik)*ea(ik,ilp1)
	  Tu22(ik)=ebs0*Tu22(ik)*eb(ik,ilp1)

	  Ru11(ik)=ea(ik,ilp1)*Ru11(ik)*ea(ik,ilp1)
	  Ru12(ik)=ea(ik,ilp1)*Ru12(ik)*eb(ik,ilp1)
	  Ru21(ik)=eb(ik,ilp1)*Ru21(ik)*ea(ik,ilp1)
	  Ru22(ik)=eb(ik,ilp1)*Ru22(ik)*eb(ik,ilp1)
		  
*      type*,'ik,Rd11(ik)=',ik,Rd11(ik)
*      type*,'ik,Rd12(ik)=',ik,Rd12(ik)
*      type*,'ik,Rd21(ik)=',ik,Rd21(ik)
*      type*,'ik,Rd22(ik)=',ik,Rd22(ik)

*      type*,'ik,Td11(ik)=',ik,Td11(ik)
*      type*,'ik,Td12(ik)=',ik,Td12(ik)
*      type*,'ik,Td21(ik)=',ik,Td21(ik)
*      type*,'ik,Td22(ik)=',ik,Td22(ik)

*      type*,'ik,Ru11(ik)=',ik,Ru11(ik)
*      type*,'ik,Ru12(ik)=',ik,Ru12(ik)
*      type*,'ik,Ru21(ik)=',ik,Ru21(ik)
*      type*,'ik,Ru22(ik)=',ik,Ru22(ik)

*      type*,'ik,Tu11(ik)=',ik,Tu11(ik)
*      type*,'ik,Tu12(ik)=',ik,Tu12(ik)
*      type*,'ik,Tu21(ik)=',ik,Tu21(ik)
*      type*,'ik,Tu22(ik)=',ik,Tu22(ik)
       Enddo
*       pause
 
      DO    75 ik=1,Nk
        call Rfs(pvec(ik),etavec(ik,1),netavec(ik,1),
     &       rho0(0),beta0(0),Rf11(ik),Rf21(ik),Rf22(ik))

*       EXCLUDE FREE SURFACE
*        Rf11(ik)=C0
*        Rf21(ik)=C0
*        Rf22(ik)=C0

        easf=CDexp(Ci*w*zs*etavec(ik,1))
        ebsf=CDexp(Ci*w*zs*netavec(ik,1))
        Rf11(ik)=Easf*Rf11(ik)*Easf
        Rf21(ik)=Ebsf*Rf21(ik)*Easf
        Rf22(ik)=Ebsf*Rf22(ik)*Ebsf

   75 continue


****************************************************************************
*     set Source contribution at the top of the layer stack
*     via S(0)*(1-Rf*R(0))^(-1)(SigD+Rf*SigU)
****************************************************************************
      Ka0=w/alpha0(0)
      Do ik=1,Nk
*       W=(I-Ru*R(1))^(-1)
	W11=C1+Ru21(ik)*R21(ik,1)-Ru22(ik)*R22(ik,1)
	W12=-Ru11(ik)*R21(ik,1)+Ru12(ik)*R22(ik,1)
	W21=Ru21(ik)*R11(ik,1)+Ru22(ik)*R21(ik,1)
	W22=C1-Ru11(ik)*R11(ik,1)-Ru12(ik)*R21(ik,1)
	InvDet=C1/(W11*W22-W12*W21)


	W11=W11*InvDet
	W12=W12*InvDet
	W21=W21*InvDet
	W22=W22*InvDet

*       S=((I-Ru*R)^(-1))*Td
        tempS11=W11*Td11(ik)+W12*Td21(ik)
        tempS12=W11*Td12(ik)+W12*Td22(ik)
	tempS21=W21*Td11(ik)+W22*Td21(ik)
	tempS22=W21*Td12(ik)+W22*Td22(ik)


*       W=R(1)*S
        W11=R11(ik,1)*tempS11-R21(ik,1)*tempS21
	W12=R11(ik,1)*tempS12-R21(ik,1)*tempS22
	W21=R21(ik,1)*tempS11+R22(ik,1)*tempS21
	W22=R21(ik,1)*tempS12+R22(ik,1)*tempS22
 
 
*       R=Rd+Tu*W
        tempR11=Rd11(ik)+Tu11(ik)*W11+Tu12(ik)*W21
	tempR12=Rd12(ik)+Tu11(ik)*W12+Tu12(ik)*W22
	tempR21=Rd21(ik)+Tu21(ik)*W11+Tu22(ik)*W21
	tempR22=Rd22(ik)+Tu21(ik)*W12+Tu22(ik)*W22

*       W=(I-Rf(0)*tempR)^(-1)
	W11=C1-Rf21(ik)*tempR12-Rf22(ik)*tempR22
	W12=Rf11(ik)*tempR12-Rf21(ik)*tempR22
	W21=Rf21(ik)*tempR11+Rf22(ik)*tempR21
	W22=C1-Rf11(ik)*tempR11+Rf21(ik)*tempR21
	InvDet=C1/(W11*W22-W12*W21)
        W11=W11*InvDet
        W12=W12*InvDet
        W21=W21*InvDet
        W22=W22*InvDet

*       V=tempS*W
        V11=tempS11*W11+tempS12*W21
	V12=tempS11*W12+tempS12*W22
	V21=tempS21*W11+tempS22*W21
	V22=tempS21*W12+tempS22*W22

*       Source
	PD(ik)=Ci*(Ka0**2)*CDSqrt(dk/(4.*Pi*rho0(0)*w*w*w*etavec(ik,1)))*
     *         CDexp(-Ci*w*pvec(ik)*xs)
	PU(ik)=PD(ik)
	SD(ik)=C0
	SU(ik)=C0

*       Sig=D+Rf*U
        SigP(ik)=PD(ik)+Rf11(ik)*PU(ik)-Rf21(ik)*SU(ik)
        SigS(ik)=SD(ik)+Rf21(ik)*PU(ik)+Rf22(ik)*SU(ik)

*       D=V*Sig
        Q1(ik,1)=V11*SigP(ik)+V12*SigS(ik)
        Q2(ik,1)=V21*SigP(ik)+V22*SigS(ik)
*		type*,'Q1(ik,1)=',Q1(ik,1)
*		type*,'Q2(ik,1)=',Q2(ik,1)


*       Reciprocal Receiver for Z-component displacement at (zr,xr)
*       store Q(ik,1) of reciprocal receiver in U11(0),U21(0) for later use
	PD(ik)=w*etavec(ik,1)*CDSqrt(dk/(4.*Pi*rho0(0)*w*w*w*etavec(ik,1)))*
     &         CDexp(-Ci*w*pvec(ik)*xr)
	PU(ik)=-PD(ik)
	SD(ik)=-w*pvec(ik)*CDSqrt(dk/(4.*Pi*rho0(0)*w*w*w*netavec(ik,1)))*
     &         CDexp(-Ci*w*pvec(ik)*xr)
	SU(ik)=SD(ik)

*       Sig=D+Rf*U
        SigP(ik)=PD(ik)+Rf11(ik)*PU(ik)-Rf21(ik)*SU(ik)
        SigS(ik)=SD(ik)+Rf21(ik)*PU(ik)+Rf22(ik)*SU(ik)

*       Q=V*Sig
        U11(ik,0)=V11*SigP(ik)+V12*SigS(ik)
        U21(ik,0)=V21*SigP(ik)+V22*SigS(ik)
*		type*,'PD(ik)=',PD(ik)
*		type*,'SD(ik)=',SD(ik)
      Enddo
	  
	  
      call DownSweep(Nk,NL,Nlm1,Nx,Nz,jskip,R11,R21,R22,
     &     S11,S12,S21,S22,PD,SD,PU,SU,SigP,SigS,PC,P1,P2,Q1,Q2,Svec)	  
	  
*     set Q of reciprocal receiver
	  Do ik=1,Nk
	    Q1(ik,1)=U11(ik,0)
	    Q2(ik,1)=U21(ik,0)
	  Enddo

      call DownSweep(Nk,NL,Nlm1,Nx,Nz,jskip,R11,R21,R22,
     &     S11,S12,S21,S22,PD,SD,PU,SU,SigP,SigS,PC,P1,P2,Q1,Q2,Rvec)	  

****************************************************************************
*      set U(0)
****************************************************************************

      DO 99999 ik=1,Nk
*      pause'check point 3'

*       set RRu
*       MB=Rf
        MB11=Rf11(ik)
        MB12=-Rf21(ik)
        MB21=Rf21(ik)
        MB22=Rf22(ik)
 
*       T(0)=(I-Rd(0)*MB)^(-1)
	W11=C1-Rd21(ik)*MB12-Rd22(ik)*MB22
	W12=Rd11(ik)*MB12+Rd12(ik)*MB22
	W21=Rd21(ik)*MB11+Rd22(ik)*MB21
	W22=C1-Rd11(ik)*MB11-Rd12(ik)*MB21
	InvDet=C1/(W11*W22-W12*W21)


        W11=W11*InvDet
        W12=W12*InvDet
        W21=W21*InvDet
        W22=W22*InvDet

        V11=W11*Tu11(ik)+W12*Tu21(ik)
	V12=W11*Tu12(ik)+W12*Tu22(ik)
	V21=W21*Tu11(ik)+W22*Tu21(ik)
	V22=W21*Tu12(ik)+W22*Tu22(ik)

*         W=MB*V
        W11=MB11*V11+MB12*V21
	W12=MB11*V12+MB12*V22
	W21=MB21*V11+MB22*V21
	W22=MB21*V12+MB22*V22
 
 
*         RRu=Ru+Td*W
        RRu11=Ru11(ik)+Td11(ik)*W11+Td12(ik)*W21
	RRu12=Ru12(ik)+Td11(ik)*W12+Td12(ik)*W22
	RRu21=Ru21(ik)+Td21(ik)*W11+Td22(ik)*W21
	RRu22=Ru22(ik)+Td21(ik)*W12+Td22(ik)*W22

*       MT=R(1)
        MT11=R11(ik,1)
        MT12=-R21(ik,1)
        MT21=R21(ik,1)
        MT22=R22(ik,1)
 
*       W=(I-RRu*MT)^(-1)
	W11=C1-RRu21*MT12-RRu22*MT22
	W12=RRu11*MT12+RRu12*MT22
	W21=RRu21*MT11+RRu22*MT21
	W22=C1-RRu11*MT11-RRu12*MT21
	InvDet=C1/(W11*W22-W12*W21)


	W11=W11*InvDet
	W12=W12*InvDet
	W21=W21*InvDet
	W22=W22*InvDet

*       U(0)=(I-RRu*MT)^(-1)*RRu
	U11(ik,0)=W11*RRu11+W12*RRu21
	U21(ik,0)=W21*RRu11+W22*RRu21
	U22(ik,0)=W21*RRu12+W22*RRu22
99999 continue


*******************************************************
*     set T-Matrix
*******************************************************
      call setT(w,dela,Nx,Nz,
     &            rho0,rho,alpha0,alpha,beta0,beta,
     &            PP11,PP22,PP31,PP33,PP51,PS11,PS51,
     &            PS22,PS31,SS11,SS51,SS22)

      Do il=1,Nl
        DO ik=1,Nk
          Ea(ik,il)=Ea(ik,il)*Ea(ik,il)
          Eb(ik,il)=Eb(ik,il)*Eb(ik,il)
        Enddo
	  Enddo
	  
      RETURN
      END


*************************************************************************
       Subroutine setT(w,dela,Nx,Nz,
     &            rho0,rho,alpha0,alpha,beta0,beta,
     &            PP11,PP22,PP31,PP33,PP51,PS11,PS51,
     &            PS22,PS31,SS11,SS51,SS22)

      Implicit none
      Integer     Nx,Nz,ix,iz
      Complex*16  w,w2
	   
	   
      Real*8      rho0(0:*),rho(Nx,Nz)
      Complex*16  alpha0(0:*),beta0(0:*)
      Complex*16  alpha(Nx,Nz),beta(Nx,Nz)

      Complex*16  PP11(Nx,Nz),PP22(Nx,Nz),PS31(Nx,Nz)
      Complex*16  PP31(Nx,Nz),PP33(Nx,Nz),PP51(Nx,Nz)
      Complex*16  PS11(Nx,Nz),PS51(Nx,Nz),PS22(Nx,Nz)
      Complex*16  SS11(Nx,Nz),SS51(Nx,Nz),SS22(Nx,Nz)

      Real*8      dela,drho,den0,den1  
      Complex*16  ca0,ca1,cb0,cb1,Cb12,Ca12,ca02,cb02,ca04,cb04
      Complex*16  mu,lam,mu2,lam2
      Complex*16  da,db,gam1,gam0,zet1,zet0,dgam,dzet

***************************************************************************
      Do iz=1,Nz
	ca0=alpha0(iz)
	cb0=beta0(iz)
        den0=rho0(iz)
        Do ix=1,Nx
	   ca1=alpha(ix,iz)
	   cb1=beta(ix,iz)
           den1=rho(ix,iz)
		   
           Cb12=cb1*cb1
	   Ca12=ca1*ca1
	   ca02=ca0*ca0
	   cb02=cb0*cb0
	   ca04=ca02*ca02
	   cb04=cb02*cb02
	   mu=den0*cb02
	   lam=den0*(Ca02-2.0*Cb02)
	   w2=w*w*dela
		   
	   mu2=mu*mu
	   lam2=lam*lam
		   
	   drho=den1-den0
	   da=1.0/(den1*Ca12)-1.0/(den0*Ca02)
	   db=1.0/(den1*Cb12)-1.0/(den0*Cb02)
	   gam1=1.0-2.0*cb12/ca12
	   gam0=1.0-2.0*cb02/ca02
	   zet1=4.0*den1*cb12*(1.0-cb12/ca12)
	   zet0=4.0*den0*Cb02*(1.0-cb02/ca02)
	   dgam=gam1-gam0
	   dzet=zet1-zet0

	   PP11(ix,iz)=dCmplx(0.,-.015625)*(dzet - 4.*dgam*mu - 
     -        4.*da*mu2 - 4.*db*mu2)*w2/(ca04*den0)
	  
	   PP22(ix,iz)=dCmplx(0.,.125)*drho*w2/(ca02*den0)

	   PP31(ix,iz)=dCmplx(0.,-.03125)*(dzet + 2*dgam*lam +  
     -       4.*da*lam*mu + 4.*da*mu2)*w2/(ca04*den0)
	  
	   PP33(ix,iz)=dCmplx(0.,-1./16)*(dzet + 4.*dgam*lam -  
     -     4.*da*lam2 + 4.*dgam*mu - 8*da*lam*mu - 4.*da*mu2)*w2/
     -     (ca04*den0)
	  
	   PP51(ix,iz)=dCmplx(0.,-.015625)*(dzet - 4.*dgam*mu -  
     -       4.*da*mu2 + 4.*db*mu2)*w2/(ca04*den0)
	  
	   PS11(ix,iz)=(-dzet + 4.*dgam*mu + 4.*da*mu2 + 4.*db*mu2)*
     -       w2/(64*ca02*cb02*den0)
	  
	   PS22(ix,iz)=drho*w2/(8*ca0*cb0*den0)

	   PS31(ix,iz)=-((dzet + 2*dgam*lam + 4.*da*lam*mu + 4.*da*mu2)*
     -       w2)/(32*ca02*cb02*den0)
	  
	   PS51(ix,iz)=-((dzet - 4.*dgam*mu - 4.*da*mu2 + 4.*db*mu2)*
     -       w2)/(64*ca02*cb02*den0)
	
	   SS11(ix,iz)=dCmplx(0.,-.015625)*(dzet - 4.*dgam*mu -  
     -       4.*da*mu2 - 4.*db*mu2)*w2/(cb04*den0)
	  
	   SS22(ix,iz)=dCmplx(0.,.125)*drho*w2/(cb02*den0)

	   SS51(ix,iz)=dCmplx(0.,.015625)*(dzet - 4.*dgam*mu -  
     -       4.*da*mu2 + 4.*db*mu2)*w2/(cb04*den0)
        Enddo
      Enddo
      return
      end

****************************************************************************************
      Complex*16 function setXKSY(Nx,Nz,PP11,PP22,PP31,PP33,PP51,
     &            PS11,PS51,PS22,PS31,SS11,SS51,SS22,X,Y)
*     set X*K*Smat*Y
      Implicit    none
      Integer     Nx,Nz,ix,iz
      Complex*16  X(Nx,9,Nz),Y(Nx,9,Nz)
      Complex*16  PP11(Nx,Nz),PP22(Nx,Nz),PS31(Nx,Nz)
      Complex*16  PP31(Nx,Nz),PP33(Nx,Nz),PP51(Nx,Nz)
      Complex*16  PS11(Nx,Nz),PS51(Nx,Nz),PS22(Nx,Nz)
      Complex*16  SS11(Nx,Nz),SS51(Nx,Nz),SS22(Nx,Nz)
      Complex*16  XKSY,P1,P2,P3,P4,P5,S1,S2,S4,S5
      Complex*16  dP1,dP2,dP3,dP4,dP5,dS1,dS2,dS4,dS5

      XKSY=(0.D0,0.D0)
      Do iz=1,Nz
	    Do ix=1,Nx
		  P1=Y(ix,1,iz)
		  P2=Y(ix,2,iz)
		  P3=Y(ix,3,iz)
		  P4=Y(ix,4,iz)
		  P5=Y(ix,5,iz)
		  S1=Y(ix,6,iz)
		  S2=Y(ix,7,iz)
		  S4=Y(ix,8,iz)
		  S5=Y(ix,9,iz)

		  dP1=X(ix,1,iz)
		  dP2=X(ix,2,iz)
		  dP3=X(ix,3,iz)
		  dP4=X(ix,4,iz)
		  dP5=X(ix,5,iz)
		  dS1=X(ix,6,iz)
		  dS2=X(ix,7,iz)
		  dS4=X(ix,8,iz)
		  dS5=X(ix,9,iz)

          XKSY=XKSY+(P5*PP51(ix,iz)+P3*PP31(ix,iz)+P1*PP11(ix,iz)-
     &        PS51(ix,iz)*S5+PS11(ix,iz)*S1)*dP5-(P2*PP22(ix,iz)+
     &        PS22(ix,iz)*S2)*dP4+(P5*PP31(ix,iz)+P1*PP31(ix,iz)+
     &        P3*PP33(ix,iz)-PS31(ix,iz)*S5+PS31(ix,iz)*S1)*dP3+
     &        (-P4*PP22(ix,iz)+PS22(ix,iz)*S4)*dP2+(P1*PP51(ix,iz)+
     &        P3*PP31(ix,iz)+P5*PP11(ix,iz)-PS11(ix,iz)*S5+
     &        PS51(ix,iz)*S1)*dP1+(-P5*PS51(ix,iz)-P3*PS31(ix,iz)-
     &        P1*PS11(ix,iz)+S5*SS51(ix,iz)+S1*SS11(ix,iz))*dS5+
     &        (P2*PS22(ix,iz)-S2*SS22(ix,iz))*dS4-(P4*PS22(ix,iz)+
     &        S4*SS22(ix,iz))*dS2+(P1*PS51(ix,iz)+P3*PS31(ix,iz)+
     &        P5*PS11(ix,iz)+S1*SS51(ix,iz)+S5*SS11(ix,iz))*dS1
	 
	Enddo
      Enddo
      setXKSY=XKSY
      return
      end
****************************************************************************************
      subroutine AtimesV(N,Nl,NLm1,Nk,Nx,Nz,jskip,
     &            R11,R21,R22,S11,S12,S21,S22,U11,U21,U22,
     &            P1,P2,Q1,Q2,PC,SY,PSY,PD,SD,PU,SU,Sigp,SigS,
     &            ea,eb,PP11,PP22,PP31,PP33,PP51,PS11,PS51,
     &            PS22,PS31,SS11,SS51,SS22,getres,V,X)
***************************************************************
*     If[getres,set  X=X-(1-P*dS)V, set X=(1-P*dS)V ]
***************************************************************
      Implicit    none
      logical     getres
      INTEGER     N,Nl,Nlm1,Nk,Nx,Nz,jskip
      Complex*16  V(N),X(N)
      Complex*16  SY(N),PSY(N)
      Complex*16  PC(Nk,9,Nz)
 
      Complex*16  U11(Nk,0:Nlm1)
      Complex*16  U21(Nk,0:Nlm1),U22(Nk,0:Nlm1)
      Complex*16  R11(Nk,Nl),R21(Nk,Nl),R22(Nk,Nl)
      Complex*16  S11(Nk,Nlm1),S12(Nk,Nlm1)
      Complex*16  S21(Nk,Nlm1),S22(Nk,Nlm1)
      Complex*16  P1(Nk,Nl),P2(Nk,Nl),Q1(Nk,Nl),Q2(Nk,Nl)

      Complex*16  PD(Nk),SD(Nk),PU(Nk),SU(Nk)
      Complex*16  sigP(Nk),sigS(Nk)
      Complex*16  ea(Nk,1:Nl),eb(Nk,1:Nl)
	 
      Complex*16  PP11(Nx,Nz),PP22(Nx,Nz),PS31(Nx,Nz)
      Complex*16  PP31(Nx,Nz),PP33(Nx,Nz),PP51(Nx,Nz)
      Complex*16  PS11(Nx,Nz),PS51(Nx,Nz),PS22(Nx,Nz)
      Complex*16  SS11(Nx,Nz),SS51(Nx,Nz),SS22(Nx,Nz)
	  
      Complex*16  C1

      C1=(1.D0,0.D0)
	  
*      type*,'entered AtimesV'
 
*     set SY=S*V
	  call setSY(Nx,Nz,PP11,PP22,PP31,
     &     PP33,PP51,PS11,PS51,PS22,PS31,SS11,
     &     SS51,SS22,V,SY)

	  
*     set PSY=P*SY
      call setPSY(Nl,Nlm1,Nk,Nx,Nz,jskip,
     &     R11,R21,R22,S11,S12,S21,S22,U11,U21,U22,
     &     ea,eb,PD,SD,PU,SU,sigP,sigS,P1,P2,Q1,Q2,PC,SY,PSY)

*      V=V-PSY        
       CALL ZAXPY( N, -C1, PSY, 1, V, 1 )

*     set SY=S*V
	  call setSY(Nx,Nz,PP11,PP22,PP31,
     &     PP33,PP51,PS11,PS51,PS22,PS31,SS11,
     &     SS51,SS22,V,SY)

	  
*     set PSY=P*SY
      call setPSY(Nl,Nlm1,Nk,Nx,Nz,jskip,
     &     R11,R21,R22,S11,S12,S21,S22,U11,U21,U22,
     &     ea,eb,PD,SD,PU,SU,sigP,sigS,P1,P2,Q1,Q2,PC,SY,PSY)

*      V=V+PSY        
       CALL ZAXPY( N, C1, PSY, 1, V, 1 )

      if(getres)then
*        X=X-V        
         CALL ZAXPY( N, -C1, V, 1, X, 1 )
      else
*       X=V        
        CALL ZCOPY( N, V, 1, X, 1 )
      Endif
*      pause 'leaving AtimesV'
      return
      end

      Subroutine setSY(Nx,Nz,PP11,PP22,PP31,
     &            PP33,PP51,PS11,PS51,PS22,PS31,SS11,
     &            SS51,SS22,Y,SY)
      Implicit    none
      Integer     Nx,Nz,ix,iz
      Complex*16  Y(Nx,9,Nz),SY(Nx,9,Nz)
      Complex*16  PP11(Nx,Nz),PP22(Nx,Nz),PS31(Nx,Nz)
      Complex*16  PP31(Nx,Nz),PP33(Nx,Nz),PP51(Nx,Nz)
      Complex*16  PS11(Nx,Nz),PS51(Nx,Nz),PS22(Nx,Nz)
      Complex*16  SS11(Nx,Nz),SS51(Nx,Nz),SS22(Nx,Nz)
      Complex*16  P1,P2,P3,P4,P5,S1,S2,S4,S5

      Do iz=1,Nz
	Do ix=1,Nx
		  P1=Y(ix,1,iz)
		  P2=Y(ix,2,iz)
		  P3=Y(ix,3,iz)
		  P4=Y(ix,4,iz)
		  P5=Y(ix,5,iz)
		  S1=Y(ix,6,iz)
		  S2=Y(ix,7,iz)
		  S4=Y(ix,8,iz)
		  S5=Y(ix,9,iz)
		  
	      SY(ix,1,iz)=P5*PP51(ix,iz)+P3*PP31(ix,iz)+
     &          P1*PP11(ix,iz)-PS51(ix,iz)*S5+PS11(ix,iz)*S1
		  SY(ix,2,iz)=P2*PP22(ix,iz)+PS22(ix,iz)*S2
		  SY(ix,3,iz)=P5*PP31(ix,iz)+P1*PP31(ix,iz)+
     &          P3*PP33(ix,iz)+PS31(ix,iz)*(S1-S5)
		  SY(ix,4,iz)=P4*PP22(ix,iz)-PS22(ix,iz)*S4
		  SY(ix,5,iz)=P1*PP51(ix,iz)+P3*PP31(ix,iz)+
     &          P5*PP11(ix,iz)-PS11(ix,iz)*S5+PS51(ix,iz)*S1
		  SY(ix,6,iz)=-P5*PS51(ix,iz)-P3*PS31(ix,iz)-
     &          P1*PS11(ix,iz)+S5*SS51(ix,iz)+S1*SS11(ix,iz)
		  SY(ix,7,iz)=-P2*PS22(ix,iz)+S2*SS22(ix,iz)
		  SY(ix,8,iz)=P4*PS22(ix,iz)+S4*SS22(ix,iz)
		  SY(ix,9,iz)=P1*PS51(ix,iz)+P3*PS31(ix,iz)+
     &          P5*PS11(ix,iz)+S1*SS51(ix,iz)+S5*SS11(ix,iz)
	  Enddo
	Enddo
      return
      end


      SUBROUTINE setPSY(Nl,Nlm1,Nk,Nx,Nz,jskip,
     &           R11,R21,R22,S11,S12,S21,S22,U11,U21,U22,
     &           ea,eb,PD,SD,PU,SU,sigP,sigS,P1,P2,Q1,Q2,PC,SY,PSY)
	
*     set X=P*SY
      Implicit none
      INTEGER     Nl,Nlm1,Nk,jskip,Nx,Nz
      complex*16  SY(Nx,9,Nz),PSY(Nx,9,Nz)	  
      complex*16  PC(Nk,9,Nz)

      Complex*16  U11(Nk,0:Nlm1)
      Complex*16  U21(Nk,0:Nlm1),U22(Nk,0:Nlm1)
      Complex*16  R11(Nk,Nl),R21(Nk,Nl),R22(Nk,Nl)
      Complex*16  S11(Nk,Nlm1),S12(Nk,Nlm1)
      Complex*16  S21(Nk,Nlm1),S22(Nk,Nlm1)
      Complex*16  P1(Nk,Nl),P2(Nk,Nl),Q1(Nk,Nl),Q2(Nk,Nl)
      Complex*16  ea(Nk,1:Nl),eb(Nk,1:Nl)

      Complex*16  PD(Nk),SD(Nk),PU(Nk),SU(Nk)
      Complex*16  sigP(Nk),sigS(Nk)


****************************************************************
*     LOCAL variables
*****************************************************************
*      type*,'entered setPSY'

      call UpSweep(Nk,Nl,Nlm1,Nx,Nz,jskip,PC,SY,
     &           R11,R21,R22,S11,S12,S21,S22,U11,U21,U22,
     &           SigP,SigS,PD,SD,PU,SU,P1,P2,Q1,Q2)
	 
      call DownSweep(Nk,NL,Nlm1,Nx,Nz,jskip,R11,R21,R22,
     &     S11,S12,S21,S22,PD,SD,PU,SU,SigP,SigS,PC,P1,P2,Q1,Q2,PSY) 
	 
	  If(Nx.gt.1)then
	    call LRsweep(Nk,Nz,Nx,PC,SY,PSY,PU,SU,PD,SD,Ea,Eb)
      Endif
      RETURN
      END

      Subroutine UpSweep(Nk,Nl,Nlm1,Nx,Nz,jskip,PC,SY,
     &           R11,R21,R22,S11,S12,S21,S22,U11,U21,U22,
     &           SigP,SigS,PD,SD,PU,SU,P1,P2,Q1,Q2)
	 
      Implicit    None
      Integer     NL,Nlm1,Nx,Nz,jskip,Nk
      complex*16  SY(Nx,9,Nz)	  
      complex*16  PC(Nk,9,Nz)
 
      Complex*16  U11(Nk,0:Nlm1)
      Complex*16  U21(Nk,0:Nlm1),U22(Nk,0:Nlm1)
      Complex*16  R11(Nk,Nl),R21(Nk,Nl),R22(Nk,Nl)
      Complex*16  S11(Nk,Nlm1),S12(Nk,Nlm1)
      Complex*16  S21(Nk,Nlm1),S22(Nk,Nlm1)
      Complex*16  P1(Nk,Nl),P2(Nk,Nl),Q1(Nk,Nl),Q2(Nk,Nl)

      Complex*16  PD(Nk),SD(Nk),PU(Nk),SU(Nk)
      Complex*16  sigP(Nk),sigS(Nk)
	  
      Integer     mm(9),ik,il,ilm1,ilp1,m,j,ix
      Real*8      signi
      Complex*16  C0
 
      
      C0=(0.D0,0.D0)
      mm(1)=5
      mm(2)=4
      mm(3)=3
      mm(4)=2
      mm(5)=1
      mm(6)=9
      mm(7)=8
      mm(8)=7
      mm(9)=6
	  
      Do ik=1,Nk
        P1(ik,Nl)=C0
        P2(ik,Nl)=C0
      Enddo
	  
      DO 99998 il=NL,1,-1
        ilm1=il-1
	ilp1=il+1
	Do ik=1,Nk
          PD(ik)=C0
	  PU(ik)=C0

	  SD(ik)=C0
	  SU(ik)=C0
	Enddo

	Do 99996 m=1,5
	  Do ik=1,Nk
	    sigP(ik)=C0
	   Enddo
          j=1
	  Do ix=1,Nx
	    sigP(j)=SY(ix,m,il)
	    j=j+jskip
	  Enddo
	  signi=-1.0
	  call FFT(sigP,Nk,signi) !x->k
	  Do ik=1,Nk	  
          PU(ik)=PU(ik)+PC(ik,m,il)*sigP(ik)
          PD(ik)=PD(ik)+PC(ik,mm(m),il)*sigP(ik)
	Enddo
99996   Continue
		
	Do 99997 m=6,9
	  Do ik=1,Nk
	    sigS(ik)=C0
	  Enddo
          j=1
	  Do ix=1,Nx
	    sigS(j)=SY(ix,m,il)
	    j=j+jskip
	  Enddo
	  signi=-1.0
	  call FFT(sigS,Nk,signi) !x->k
	  Do ik=1,Nk	  
            SU(ik)=SU(ik)+PC(ik,m,il)*sigS(ik)
            SD(ik)=SD(ik)+PC(ik,mm(m),il)*sigS(ik)
	  Enddo
99997    Continue
		
        DO ik=1,Nk
*         P(il)=P(il)+R(il)*D(il)
          P1(ik,il)=P1(ik,il)+R11(ik,il)*PD(ik)-R21(ik,il)*SD(ik)
          P2(ik,il)=P2(ik,il)+R21(ik,il)*PD(ik)+R22(ik,il)*SD(ik)
        Enddo

        if(il.ne.nl)then
          DO ik=1,Nk
*           Q(il)=Q(il)+S(il)*D(il)
            Q1(ik,ilp1)=Q1(ik,ilp1)+S11(ik,il)*PD(ik)+S12(ik,il)*SD(ik)
            Q2(ik,ilp1)=Q2(ik,ilp1)+S21(ik,il)*PD(ik)+S22(ik,il)*SD(ik)
          Enddo
        endif

        DO ik=1,Nk
*         PU=(PU+P1) & SU=(SU+P2)
          PU(ik)=PU(ik)+P1(ik,il)
          SU(ik)=SU(ik)+P2(ik,il)
        Enddo
		
        DO ik=1,Nk
*         Q(il)=U(ilm1)*U
          Q1(ik,il)=U11(ik,ilm1)*PU(ik)-U21(ik,ilm1)*SU(ik)
          Q2(ik,il)=U21(ik,ilm1)*PU(ik)+U22(ik,ilm1)*SU(ik)
        Enddo
		
	if(il.ne.1) then
          DO ik=1,Nk
*           P(ilm1)=T(ilm1)*U
            P1(ik,ilm1)=S11(ik,ilm1)*PU(ik)-S21(ik,ilm1)*SU(ik)
            P2(ik,ilm1)=-S12(ik,ilm1)*PU(ik)+S22(ik,ilm1)*SU(ik)
          Enddo
	endif		
99998 continue

      return
      end
 
      Subroutine DownSweep(Nk,NL,Nlm1,Nx,Nz,jskip,R11,R21,R22,
     & S11,S12,S21,S22,PD,SD,PU,SU,SigP,SigS,PC,P1,P2,Q1,Q2,PSY)
      Implicit   None
      Integer     Nk,NL,Nlm1,Nx,Nz,jskip

      Complex*16  R11(Nk,1:Nl)
      Complex*16  R21(Nk,1:Nl),R22(Nk,1:Nl)
      Complex*16  S11(Nk,1:Nlm1),S12(Nk,1:Nlm1)
      Complex*16  S21(Nk,1:Nlm1),S22(Nk,1:Nlm1)
      Complex*16  P1(Nk,1:Nl),P2(Nk,1:Nl)
      Complex*16  Q1(Nk,1:Nl),Q2(Nk,1:Nl)

      Complex*16  PSY(Nx,9,Nz)
      complex*16  PC(Nk,9,Nz)

      Complex*16  PD(Nk),SD(Nk),PU(Nk),SU(Nk)
      Complex*16  sigP(Nk),sigS(Nk)

      Integer     mm(9),il,ilm1,ik,m,j,ix
      Real*8      signi	  
	  Complex*16  tp,ts,C0
   
      C0=(0.0,0.0)
      mm(1)=5
      mm(2)=4
      mm(3)=3
      mm(4)=2
      mm(5)=1
      mm(6)=9
      mm(7)=8
      mm(8)=7
      mm(9)=6
	  
      DO 99998 IL=1,Nl
	ilm1=il-1
        if(il.ne.1)then
	  Do ik=1,Nk
	    tp=Q1(ik,il)+S11(ik,ilm1)*PD(ik)+S12(ik,ilm1)*SD(ik)
	    ts=Q2(ik,il)+S21(ik,ilm1)*PD(ik)+S22(ik,ilm1)*SD(ik)
            PD(ik)=tp
	    SD(ik)=ts
	  Enddo
        else
	  Do ik=1,Nk
            PD(ik)=Q1(ik,il)
	    SD(ik)=Q2(ik,il)
	  Enddo
        endif

	Do ik=1,Nk	  
          PU(ik)=P1(ik,il)+R11(ik,il)*PD(ik)-R21(ik,il)*SD(ik)
          SU(ik)=P2(ik,il)+R21(ik,il)*PD(ik)+R22(ik,il)*SD(ik)
	Enddo

        Do 99996 m=1,5
	  Do ik=1,Nk
            SigP(ik)=PC(ik,m,il)*PD(ik)+PC(ik,mm(m),il)*PU(ik)
	  Enddo
	  signi=1.0
	  call FFT(SigP,Nk,signi) !k->x
	  j=1
	  Do ix=1,Nx	  
            PSY(ix,m,il)=SigP(j)
	    j=j+jskip
	  Enddo
99996	continue

        Do 99997 m=6,9
	  Do ik=1,Nk
            SigS(ik)=PC(ik,m,il)*SD(ik)+PC(ik,mm(m),il)*SU(ik)
	  Enddo
	  signi=1.0
	  call FFT(SigS,Nk,signi) !k->x
	  j=1
	  Do ix=1,Nx	  
            PSY(ix,m,il)=SigS(j)
	    j=j+jskip
	  Enddo
99997	continue
99998 continue

      return
      end
	
      subroutine LRsweep(Nk,Nz,Nx,PC,SY,PSY,PU,SU,PD,SD,Ea,Eb)
      Implicit   none
      integer    kstep,Nk,Nz,Nx
*     halve the wavenumber sampling
      Parameter  (kstep=2) 
      Complex*16 PC(Nk,9,Nz),SY(Nx,9,Nz)
      Complex*16 PSY(Nx,9,Nz)
      Complex*16 PU(Nk),SU(Nk),PD(Nk),SD(Nk),Ea(Nk,Nz),Eb(Nk,Nz)
	  
      Integer    mm(9),iz,ix,ik,m
      Real*8     fkstep
      Complex*16 C0,Ci,Cim(9)
	  
      fkstep=Sqrt(float(kstep))
      C0=(0.D0,0.D0)
      Ci=(0.D0,1.D0)
      mm(1)=5
      mm(2)=4
      mm(3)=3
      mm(4)=2
      mm(5)=1
      mm(6)=9
      mm(7)=8
      mm(8)=7
      mm(9)=6

      Cim(1)=-fkstep
      Cim(2)=-Ci*fkstep
      Cim(3)=fkstep
      Cim(4)=Ci*fkstep
      Cim(5)=-fkstep
      Cim(6)=-fkstep
      Cim(7)=-Ci*fkstep
      Cim(8)=Ci*fkstep
      Cim(9)=-fkstep
	  
      Do 5000 iz=Nz,1,-1
	  
	 Do ik=1,Nk,2
	   PU(ik)=C0
	   SU(ik)=C0
	 Enddo


*       begin right sweep
        Do 3000 ix=1,Nx
*         type*,'ix=',ix

*k->x
        Do 90 m=1,5
	  Do ik=1,Nk,kstep
            PSY(ix,m,iz)=PSY(ix,m,iz)+Cim(m)*PC(ik,mm(m),iz)*PU(ik)
	  Enddo
90	continue

        Do 91 m=6,9
	  Do ik=1,Nk,kstep
            PSY(ix,m,iz)=PSY(ix,m,iz)+Cim(m)*PC(ik,mm(m),iz)*SU(ik)
	  Enddo
91	continue
		
	Do 30 m=1,5
	  Do ik=1,Nk,kstep	  
            PU(ik)=PU(ik)+Cim(mm(m))*PC(ik,m,iz)*SY(ix,m,iz)
	  Enddo
30      continue		

	Do 31 m=6,9
	  Do ik=1,Nk,kstep	  
            SU(ik)=SU(ik)+Cim(mm(m))*PC(ik,m,iz)*SY(ix,m,iz)
	  Enddo
31      continue		

	Do ik=1,Nk,kstep	  
	  PU(ik)=Ea(ik,iz)*PU(ik)
	  SU(ik)=Eb(ik,iz)*SU(ik)
        Enddo

3000  continue  !end right sweep

       Do ik=1,Nk
	  PD(ik)=C0
	  SD(ik)=C0
       Enddo
      
*     begin left sweep
      Do 4000 ix=Nx,1,-1
	  
*	    type*,'ix=',ix


        Do 110 m=1,5
	  Do ik=1,Nk,kstep
            PSY(ix,m,iz)=PSY(ix,m,iz)+Cim(m)*PC(ik,m,iz)*PD(ik)
	  Enddo
110	continue

        Do 111 m=6,9
	  Do ik=1,Nk,kstep
            PSY(ix,m,iz)=PSY(ix,m,iz)+Cim(m)*PC(ik,m,iz)*SD(ik)
	  Enddo
111	continue
	  
		
	Do 120 m=1,5
	  Do ik=1,Nk,kstep	  
            PD(ik)=PD(ik)+Cim(mm(m))*PC(ik,mm(m),iz)*SY(ix,m,iz)
	   Enddo
120     continue

	Do 121 m=6,9
	  Do ik=1,Nk,kstep	  
            SD(ik)=SD(ik)+Cim(mm(m))*PC(ik,mm(m),iz)*SY(ix,m,iz)
	  Enddo
121     continue


	Do ik=1,Nk,kstep	  
	  PD(ik)=Ea(ik,iz)*PD(ik)
	  SD(ik)=Eb(ik,iz)*SD(ik)
	Enddo


4000  continue  !end leftsweep
5000  continue  !end depth loop
      return
      end	


      COMPLEX*16 FUNCTION CV(W,C0,Q0)
      implicit none
*     DETERMINE THE FREQUENCY DEPENDENT RECIPROCAL PROPAGATION
*     VELOCITY (SLOWNESS) ACCORDING TO THE POWER LAW OF STRICK
      REAL*8    Q0
      COMPLEX*16 C0,W
*     local variables
      REAL*8    EPS,Pi,W0,SIGMA,T1
      Complex*16 T,CW
      PARAMETER(Pi=3.14159265358979324D0,W0=1.D0,SIGMA=0.1D0,EPS=1.D-3)
      T=CDEXP(-SIGMA*CDLOG(DCMPLX(EPS+DIMAG(W),-DBLE(W))))
*	  type*,'T=',T
      T1=W0**SIGMA/DSIN(SIGMA*PI/2)
*	  type*,'T1=',T1
      CW=(1.D0,0.D0)+CMPLX(0.5D0*T1/Q0,0.D0)*T
*	  type*,'CW=',CW
      CV=C0/CW
      RETURN
      END
 
      subroutine Vslo(Nk,C,pvec,Vslovec)
      implicit none
      Integer  Nk,ik,ierr
*     compute the vertical slowness
      Complex*16 pvec(Nk),Vslovec(Nk),C,Cinv
      Real*8 eps
      parameter(eps=1.0D-8)
      Complex*16 T
      ierr=0
      Do 10 ik=1,Nk
      if(CDabs(c).le.eps)then
        PAUSE 'propagation velocity too small'
      else
        Cinv=1.D0/c
      end if
      T=CMPLX((Cinv+pvec(ik))*(Cinv-pvec(ik)))
      Vslovec(ik)=CDsqrt(T)
      IF(DIMAG(Vslovec(ik)) .LT. 0.0)THEN
        Vslovec(ik)=-Vslovec(ik)
      END IF
      If(Dimag(Vslovec(ik))*DBLE(Vslovec(ik)).lt. 0.D0) then
        ierr=1
*        Pause'DANGER! nonThermodynamic Vslo'
      Endif	
      If(CDabs(Vslovec(ik)).lt. eps) then
*        Pause'DANGER! hit turning point'
        ierr=2
      Endif
   10 CONTINUE
      If(ierr.ne.0) then
        type*,'ierr=',ierr,'in Vslo'
        stop
      Endif
      RETURN
      END
 
 
      subroutine Rfs(P,ETA,NETA,RHO,BETA,Rf11,Rf21,Rf22)
*********************************************************
*     form the Free surface reflection response
*********************************************************
      implicit none
      Real*8    rho
      Complex*16 beta,twomup,p
      Complex*16 Rf11,Rf21,Rf22
      Complex*16 eta,neta,Nd(2,2),Nu(2,2)
      Complex*16 ci,epsa,epsb,c1,InvDet
      Ci=Dcmplx(0.D0,1.D0)
      C1=Dcmplx(1.D0,0.D0)
      twomup=2.D0*rho*beta*beta*p
      epsa=C1/CDsqrt(2.D0*rho*eta)
      epsb=C1/CDsqrt(2.D0*rho*neta)
 
      Nd(2,1)=(twomup*p-rho)*epsa
      Nd(1,2)=-(twomup*p-rho)*epsb
      Nd(1,1)=-twomup*eta*epsa
      Nd(2,2)=-twomup*neta*epsb
 
      Nu(2,1)=Nd(2,1)
      Nu(2,2)=-Nd(2,2)
      Nu(1,1)=-Nd(1,1)
      Nu(1,2)=Nd(1,2)
 
      InvDet=ND(1,1)*ND(2,2)-ND(1,2)*ND(2,1)
      If(CDabs(InvDet).lt.1.D-3) then
        stop'ERROR!! singular InvDet'
      else
        InvDet=C1/InvDet
      endif
      Rf11=InvDet*(ND(1,2)*NU(2,1)-ND(2,2)*Nu(1,1))
      Rf21=InvDet*(ND(2,1)*NU(1,1)-ND(1,1)*Nu(2,1))
      Rf22=InvDet*(ND(2,1)*NU(1,2)-ND(1,1)*Nu(2,2))
      return
      end
 
      Subroutine  IntScat(Nk,pvec,etavec1,netavec1,rho1,
     &      beta1,etavec2,netavec2,rho2,beta2,Rd11,Rd12,
     &      Rd21,Rd22,Tu11,Tu12,Tu21,Tu22,Td11,Td12,Td21,
     &      Td22,Ru11,Ru12,Ru21,Ru22)
*     **************************************************************
*     plane-wave interfacial scattering matrices
*     modified from Aki&Richards pp149-51
*     **************************************************************
      implicit   none
      Integer    Nk,ik,ierr
      Real*8     RHO1,RHO2,rtrho1,rtrho2,m2
      Complex*16 ETAvec1(Nk),NETAvec1(Nk)
      Complex*16 ETAvec2(Nk),NETAvec2(Nk)
      Complex*16 BETA1,BETA2
      Complex*16 pvec(Nk),PSQ,CRHO1,CRHO2,DRHO,Dmu
      Complex*16 rteta1,rteta2,rtneta1,rtneta2
      Complex*16 Rd11(Nk),Rd12(Nk),Rd21(Nk),Rd22(Nk)
      Complex*16 Tu11(Nk),Tu12(Nk),Tu21(Nk),Tu22(Nk)
      Complex*16 Td11(Nk),Td12(Nk),Td21(Nk),Td22(Nk)
      Complex*16 Ru11(Nk),Ru12(Nk),Ru21(Nk),Ru22(Nk)
      Complex*16 PSQD,a,b,c,d,Det,E,F,G,H,Q,R,S,T,U,V
      Complex*16 rtza1,rtza2,rtzb1,rtzb2
 
      m2=-2.D0
      rtrho1=Dsqrt(rho1)
      rtrho2=Dsqrt(rho2)
      ierr=0
      Do 10 ik=1,Nk
 
      rteta1=CDsqrt(etavec1(ik))
      rteta2=CDsqrt(etavec2(ik))
      rtneta1=CDsqrt(netavec1(ik))
      rtneta2=CDsqrt(netavec2(ik))
      rtza1=rteta1*rtrho1
      rtza2=rteta2*rtrho2
      rtzb1=rtneta1*rtrho1
      rtzb2=rtneta2*rtrho2
      psq=pvec(ik)*pvec(ik)
      crho1=Dcmplx(rho1,0.D0)
      crho2=Dcmplx(rho2,0.D0)
      drho=crho2-crho1
      dmu=crho2*beta2*beta2-crho1*beta1*beta1
 
      d=2.D0*dmu
      psqd=psq*d
      a=drho-psqd
      b=crho2-psqd
      c=crho1+psqd
 
      E=(b*etavec1(ik)+c*etavec2(ik))
      F=(b*netavec1(ik)+c*netavec2(ik))
      G=(a-d*etavec1(ik)*netavec2(ik))
      H=(a-d*etavec2(ik)*netavec1(ik))
      Det=E*F+G*H*psq
      If(CDabs(Det).lt.1.D-6) then
         ierr=1
      Endif
      E=E/Det
      F=F/Det
      G=G/Det
      H=H/Det
 
      Q=(B*ETAvec1(ik)-C*ETAvec2(ik))*F
      R=(a+d*etavec1(ik)*netavec2(ik))*H*psq
      S=(a*b+c*d*etavec2(ik)*netavec2(ik))*pvec(ik)/Det
      T=(b*netavec1(ik)-c*netavec2(ik))*E
      U=(a+d*etavec2(ik)*netavec1(ik))*G*psq
      V=(a*c+b*d*etavec1(ik)*netavec1(ik))*pvec(ik)/Det
 
      Rd11(ik)=Q-R
      Rd21(ik)=m2*rteta1*rtneta1*S
      Rd12(ik)=-Rd21(ik)
      Rd22(ik)=T-U
      Td11(ik)=2.D0*rtza1*rtza2*F
      Td21(ik)=m2*rtza1*rtzb2*H*pvec(ik)
      Td12(ik)=2.D0*rtzb1*rtza2*G*pvec(ik)
      Td22(ik)=2.D0*rtzb1*rtzb2*E
      Tu11(ik)=Td11(ik)
      Tu21(ik)=-Td12(ik)
      Tu12(ik)=-Td21(ik)
      Tu22(ik)=Td22(ik)
      Ru11(ik)=-(Q+U)
      Ru21(ik)=m2*rteta2*rtneta2*V
      Ru12(ik)=-Ru21(ik)
      Ru22(ik)=-(T+R)
   10 continue	
      if(ierr.ne.0)then
        type*,'hit Stonley pole in ScatMat'
        stop
      endif
      return
      end


      SUBROUTINE FFT(CX,LX,SIGNI)
      Implicit  none
      Integer    LX,J,L,I,M,ISTEP
      COMPLEX*16 CX(LX),CTEMP,CW
      Real*8     Pi,Signi,AA
      Parameter (Pi=3.14159265358979324)
      J=1
      DO 3 I=1,LX
      IF(I.GT. J) GO TO 1
      CTEMP=CX(J)
      CX(J)=CX(I)
      CX(I)=CTEMP
 1    M=LX/2
 2    IF(J.LE.M) GO TO 3
      J=J-M
      M=M/2
      IF(M.GE.1) GO TO 2
 3    J=J+M
      L=1
 4    ISTEP=2*L
      DO 5 M=1,L
      AA=Pi*SIGNI*DFLOAT((M-1))/FLOAT(L)
      CW=CMPLX(DCOS(AA),DSIN(AA))
      DO 5 I=M,LX,ISTEP
      CTEMP=CW*CX(I+L)
      CX(I+L)=CX(I)-CTEMP
 5    CX(I)=CX(I)+CTEMP
      L=ISTEP
      IF(L.LT.LX) GO TO 4
      RETURN
      END

*******************************************************************************
*******************************************************************************
*******************************************************************************
*******************************************************************************
*     ===============================================================
*     ===============================================================
*     ===============================================================
*     ===============================================================

*     BLAS ROUTINES
*     ===============================================================
*     ===============================================================
*     ===============================================================
*     ===============================================================
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
      double complex function zdotc(n,zx,incx,zy,incy)
c
c     forms the dot product of a vector.
c     jack dongarra, 3/11/78.
c     modified 12/3/93, array(1) declarations changed to array(*)
c
      double complex zx(*),zy(*),ztemp
      integer i,incx,incy,ix,iy,n
      ztemp = (0.0d0,0.0d0)
      zdotc = (0.0d0,0.0d0)
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
        ztemp = ztemp + dconjg(zx(ix))*zy(iy)
        ix = ix + incx
        iy = iy + incy
   10 continue
      zdotc = ztemp
      return
c
c        code for both increments equal to 1
c
   20 do 30 i = 1,n
        ztemp = ztemp + dconjg(zx(i))*zy(i)
   30 continue
      zdotc = ztemp
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

      integer function izamax(n,zx,incx)
c
c     finds the index of element having max. absolute value.
c     jack dongarra, 1/15/85.
c     modified 3/93 to return if incx .le. 0.
c     modified 12/3/93, array(1) declarations changed to array(*)
c
      double complex zx(*)
      double precision smax
      integer i,incx,ix,n
      double precision dcabs1
c
      izamax = 0
      if( n.lt.1 .or. incx.le.0 )return
      izamax = 1
      if(n.eq.1)return
      if(incx.eq.1)go to 20
c
c        code for increment not equal to 1
c
      ix = 1
      smax = dcabs1(zx(1))
      ix = ix + incx
      do 10 i = 2,n
         if(dcabs1(zx(ix)).le.smax) go to 5
         izamax = i
         smax = dcabs1(zx(ix))
    5    ix = ix + incx
   10 continue
      return
c
c        code for increment equal to 1
c
   20 smax = dcabs1(zx(1))
      do 30 i = 2,n
         if(dcabs1(zx(i)).le.smax) go to 30
         izamax = i
         smax = dcabs1(zx(i))
   30 continue
      return
      end

**************************************************
