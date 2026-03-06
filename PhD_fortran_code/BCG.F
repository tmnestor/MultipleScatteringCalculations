      Program      BCG
*******************************************************************
*     biconjugate gradient with a stratified Background
*******************************************************************
      implicit     none
	  logical      pointscat
	  parameter    (pointscat=.true.)
      INTEGER      Nk,Nl,Nlm1,Nl1,Nscatx,Nscatz,Nscat,jskip
	  INTEGER      iscatz,iscatx,Nh,Nhm1,j,m,il,i
      Parameter    (Nk=4096,Nh=Nk/2,Nhm1=Nh-1)

      parameter    (Nl=2,Nlm1=Nl-1,NL1=Nl+1)
	  Parameter    (Nscatx=9,Nscatz=Nl,jskip=8,Nscat=Nscatx*Nscatz)
	  
      Real*8       h(0:Nl),xs,zs,xr,zr,dela
	  Parameter    (xr=0,xs=0)
      Real*8       rho0(0:Nl1),rho(Nscatx,Nscatz)
      Complex*16   alpha0(0:Nl1),beta0(0:Nl1)
      Complex*16   alpha(Nscatx,Nscatz),beta(Nscatx,Nscatz)
      Complex*16   w,QB(0:Nl1)
      Real*8       rt3,errb,nrm1

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
 
      Complex*16  Rd11(Nk),Rd12(Nk)
      Complex*16  Rd21(Nk),Rd22(Nk)
      Complex*16  Tu11(Nk),Tu12(Nk)
      Complex*16  Tu21(Nk),Tu22(Nk)
      Complex*16  Td11(Nk),Td12(Nk)
      Complex*16  Td21(Nk),Td22(Nk)
      Complex*16  Ru11(Nk),Ru12(Nk)
      Complex*16  Ru21(Nk),Ru22(Nk)
      Complex*16  Rf11(Nk),Rf12(Nk)
      Complex*16  Rf21(Nk),Rf22(Nk)
      Complex*16  etavec(Nk,2),netavec(Nk,2)
      Complex*16  Ea(Nk,Nl),Eb(Nk,Nl)
	  
	  Complex*16  PP11(Nscatx,Nscatz),PP22(Nscatx,Nscatz),PS31(Nscatx,Nscatz)
	  Complex*16  PP31(Nscatx,Nscatz),PP33(Nscatx,Nscatz),PP51(Nscatx,Nscatz)
	  Complex*16  PS11(Nscatx,Nscatz),PS51(Nscatx,Nscatz),PS22(Nscatx,Nscatz)
	  Complex*16  SS11(Nscatx,Nscatz),SS51(Nscatx,Nscatz),SS22(Nscatx,Nscatz)

      Complex*16  PD(Nk),SD(Nk),PU(Nk),SU(Nk)
      Complex*16  sigP(Nk),sigS(Nk)

*******************************************************
*     BCG storage
*******************************************************

      INTEGER      N,ITER, INFO
	  Parameter    (N=10*Nscatx*Nscatz)
      parameter    (Iter=N)
      real*8       RESID
      parameter    (resid=1.D-8)
      complex*16   X( N ),WORK(N,3)
      Complex*16   Svec(N),Rvec(N)
      Complex*16   SY(N),PSY(N)

*******************************************************
*     set p Lattice Data (- Kmax<= w p <= Kmax)
*******************************************************
	  
*     spatial period 
      Real*8      L,dx
	  Parameter   (L=1.D02)
*     wave number interval dk
      Real*8      dk,Kmax,kvec(Nk)

      Complex*16  pvec(Nk)

      Real*8      drho
      Complex*16  dbeta
	  
      DATA     w/12.0/
	  DATA     beta0/3.5,3.6,3.7,3.8/	
      DATA     rho0/3.0,3.1,3.2,3.3/
	  DATA     drho,dbeta/0.1,0.2/

************************************************************
	  Kmax=20.0*Real(w/beta0(0))
	  dk=2.*Kmax/float(Nk)
	  dx=Pi/Kmax
	  dela=(jskip*dx)**2
*************************************************************	  

*     Reference Layer Depths
	  Do il=0,Nl
	    h(il)=jskip*dx
	  Enddo
      zs=0.5*h(0)
	  zr=zs
	  


      rt3=DSqrt(3.0D0)
	  Do 12 il=0,Nl1
	    alpha0(il)=rt3*beta0(il)
  12  continue
      Do 15 iscatz=1,nscatz
	    Do 13 iscatx=1,nscatx
          rho(iscatx,iscatz)=rho0(iscatz)+drho
          beta(iscatx,iscatz)=beta0(iscatz)+dbeta
          alpha(iscatx,iscatz)=rt3*beta(iscatx,iscatz)
  13    continue
  15  continue
 
*     ********************************************************
*     FFT ordered slowness lattice
*     ********************************************************
	  Do i=1,Nhm1
		kvec(i+1)=float(i)*dk
		kvec(i+1+Nh)=float(i-Nh)*dk
      Enddo
	  kvec(1)=0.0
	  kvec(Nh+1)=-Nh*dk
      Do i=1,Nk
        pvec(i)=kvec(i)/w
*        type*,'i,pvec(i)=',i,pvec(i)
      enddo
 
*     ************************************************************
*     initialise  R/T coeffs and factorise System matrix
*     ************************************************************
      call Init(pointscat,Nl,Nlm1,Nl1,Nscatx,Nscatz,jskip,
     &           Nk,dk,kvec,pvec,etavec,netavec,Svec,Rvec,
     &           PC,w,h,dela,xs,zs,xr,zr,rho0,alpha0,beta0,rho,alpha,beta,
     &           Rf11,Rf12,Rf21,Rf22,Rd11,Rd12,Rd21,Rd22,Td11,Td12,
     &           Td21,Td22,Tu11,Tu12,Tu21,Tu22,Ru11,Ru12,
     &           Ru21,Ru22,Ea,Eb,P1,P2,Q1,Q2,R11,R12,R21,R22,
     &           S11,S12,S21,S22,U11,U12,U21,U22,T11,T12,T21,T22,
     &           PD,SD,PU,SU,SigP,SigS,PP11,PP22,PP31,PP33,PP51,
     &           PS11,PS51,PS22,PS31,SS11,SS51,SS22)
	 
*      type*,'R12=',R12
*      type*,'R21=',R21
*      type*,'U12=',U12
*      type*,'U21=',U21
*      type*,'S11=',S11
*      type*,'S12=',S12
*      type*,'S21=',S21
*      type*,'S22=',S22
*      type*,'T11=',T11
*      type*,'T12=',T12
*      type*,'T21=',T21
*      type*,'T22=',T22

      call BICG(N,Svec, X, WORK, ITER, RESID,INFO,
     &            Nl,NLm1,Nk,Nscatx,Nscatz,jskip,
     &            R11,R12,R21,R22,S11,S12,S21,S22,
     &            U11,U12,U21,U22,T11,T12,T21,T22,
     &            P1,P2,Q1,Q2,PC,SY,PSY,PD,SD,PU,SU,Sigp,SigS,
     &            ea,eb,PP11,PP22,PP31,PP33,PP51,PS11,PS51,
     &            PS22,PS31,SS11,SS51,SS22)
	 

      stop'normal completion'
      end



******************************************************************
      SUBROUTINE BICG(N,B, X, WORK, ITER, RESID,INFO,
     &            Nl,NLm1,Nk,Nscatx,Nscatz,jskip,
     &            R11,R12,R21,R22,S11,S12,S21,S22,
     &            U11,U12,U21,U22,T11,T12,T21,T22,
     &            P1,P2,Q1,Q2,PC,SY,PSY,PD,SD,PU,SU,Sigp,SigS,
     &            ea,eb,PP11,PP22,PP31,PP33,PP51,PS11,PS51,
     &            PS22,PS31,SS11,SS51,SS22)
	 
	  Implicit    none
      INTEGER     N, ITER, INFO
      Real*8      RESID
      Complex*16  X( N ), B( N ), WORK( N,* )
      Complex*16  SY( N ),PSY( N )

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
      Complex*16  setXKSY


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
      Complex*16         ZERO, ONE
      PARAMETER  	     ( ZERO = (0.D0,0.D0), ONE = (1.D0,0.D0) )
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
*      RHOTOL = GETBREAK()

       RHOTOL = 1.D-16
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
	     RHO=setXKSY(Nscatx,Nscatz,PP11,PP22,PP31,PP33,PP51,
     &      PS11,PS51,PS22,PS31,SS11,SS51,SS22,WORK(1,Z), WORK(1,R))

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
         call AtimesV(Nl,NLm1,Nk,Nscatx,Nscatz,jskip,
     &            R11,R12,R21,R22,S11,S12,S21,S22,
     &            U11,U12,U21,U22,T11,T12,T21,T22,
     &            P1,P2,Q1,Q2,PC,SY,PSY,PD,SD,PU,SU,Sigp,SigS,
     &            ea,eb,PP11,PP22,PP31,PP33,PP51,PS11,PS51,
     &            PS22,PS31,SS11,SS51,SS22,.false.,WORK(1,P),
     &            WORK(1,Q))

         ALPHA = RHO / setXKSY(Nscatx,Nscatz,PP11,PP22,PP31,PP33,PP51,
     &           PS11,PS51,PS22,PS31,SS11,SS51,SS22,WORK(1,P), WORK(1,Q))
	 
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
		 type*,'resid=',resid
		 
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


************************************************************
      SUBROUTINE Init(pointscat,Nl,Nlm1,Nl1,Nscatx,Nscatz,jskip,
     &           Nk,dk,kvec,pvec,etavec,netavec,Svec,Rvec,
     &           PC,w,h,dela,xs,zs,xr,zr,rho0,alpha0,beta0,rho,alpha,beta,
     &           Rf11,Rf12,Rf21,Rf22,Rd11,Rd12,Rd21,Rd22,Td11,Td12,
     &           Td21,Td22,Tu11,Tu12,Tu21,Tu22,Ru11,Ru12,
     &           Ru21,Ru22,Ea,Eb,P1,P2,Q1,Q2,R11,R12,R21,R22,
     &           S11,S12,S21,S22,U11,U12,U21,U22,T11,T12,T21,T22,
     &           PD,SD,PU,SU,SigP,SigS,PP11,PP22,PP31,PP33,PP51,
     &           PS11,PS51,PS22,PS31,SS11,SS51,SS22)
	
*     set up interfacial scattering arrays and factorise propagator
      Implicit    none
	  logical     pointscat
      INTEGER     Nl,NLm1,Nl1,Nscatx,Nscatz,Nk,jskip
      Real*8       h(0:Nl),xs,zs,xr,zr,dela
      Real*8       rho0(0:Nl1)
      Complex*16   alpha0(0:Nl1),beta0(0:Nl1)
	  Real*8       rho(Nscatx,Nscatz)
      Complex*16   alpha(Nscatx,Nscatz),beta(Nscatx,Nscatz)
	  Real*8      kvec(Nk),kx,dk,Pi
	  Parameter   (Pi=3.14159265358979324)

      Complex*16  w,pvec(Nk)

      Complex*16  PC(Nk,-2:2,2,Nscatz)
      Complex*16  Svec(Nscatx,-2:2,2,Nscatz),Rvec(Nscatx,-2:2,2,Nscatz)

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
 
      Complex*16  Rd11(Nk),Rd12(Nk)
      Complex*16  Rd21(Nk),Rd22(Nk)
      Complex*16  Tu11(Nk),Tu12(Nk)
      Complex*16  Tu21(Nk),Tu22(Nk)
      Complex*16  Td11(Nk),Td12(Nk)
      Complex*16  Td21(Nk),Td22(Nk)
      Complex*16  Ru11(Nk),Ru12(Nk)
      Complex*16  Ru21(Nk),Ru22(Nk)
      Complex*16  Rf11(Nk),Rf12(Nk)
      Complex*16  Rf21(Nk),Rf22(Nk)
      Complex*16  etavec(Nk,2),netavec(Nk,2)
      Complex*16  Ea(Nk,Nl),Eb(Nk,Nl)

      Complex*16  PD(Nk),SD(Nk),PU(Nk),SU(Nk)
      Complex*16  sigP(Nk),sigS(Nk)
	  
	  Complex*16  PP11(Nscatx,Nscatz),PP22(Nscatx,Nscatz),PS31(Nscatx,Nscatz)
	  Complex*16  PP31(Nscatx,Nscatz),PP33(Nscatx,Nscatz),PP51(Nscatx,Nscatz)
	  Complex*16  PS11(Nscatx,Nscatz),PS51(Nscatx,Nscatz),PS22(Nscatx,Nscatz)
	  Complex*16  SS11(Nscatx,Nscatz),SS51(Nscatx,Nscatz),SS22(Nscatx,Nscatz)
 
      Complex*16  p,eta,neta,Ka0,Kb0,kz,C0,C1,Ci,eas0,ebs0,easf,ebsf
      Complex*16  W11,W12,W21,W22,InvDet,temp
      Complex*16  tempS11,tempS12,tempS21,tempS22
      Complex*16  tempR11,tempR12,tempR21,tempR22
	  Complex*16  MT11,MT12,MT21,MT22,MB11,MB12,MB21,MB22
	  Complex*16  V11,V12,V21,V22,RRu11,RRu12,RRu21,RRu22
      integer     i,j,ik,iscatz,il,ilp1
 
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
		R12(ik,Nl)=ea(ik,Nl)*Rd12(ik)*eb(ik,Nl)
		R21(ik,Nl)=eb(ik,Nl)*Rd21(ik)*ea(ik,Nl)
		R22(ik,Nl)=eb(ik,Nl)*Rd22(ik)*eb(ik,Nl)
      Enddo

*       set PC for layer Nl
	    Ka0=w/alpha0(il)
	    Kb0=w/beta0(il)
*P
	    Do ik=1,Nk
	      kx=kvec(ik)
		  kz=w*etavec(ik,1)
		  PC(ik,0,1,il)=Csqrt(dk/(Pi*kz))
		  temp=((kx/ka0)+Ci*(kz/ka0))
		  PC(ik,1,1,il)=PC(ik,0,1,il)*temp
	      PC(ik,2,1,il)=PC(ik,1,1,il)*temp
		  temp=((kx/ka0)-Ci*(kz/ka0))
		  PC(ik,-1,1,il)=PC(ik,0,1,il)*temp
	      PC(ik,-2,1,il)=PC(ik,-1,1,il)*temp
	  Enddo
*SV	  
	  Do ik=1,Nk
	      kx=kvec(ik)
		  kz=w*netavec(ik,1)
		  PC(ik,0,2,il)=Csqrt(dk/(Pi*kz))
		  temp=((kx/kb0)+Ci*(kz/kb0))
		  PC(ik,1,2,il)=PC(ik,0,2,il)*temp
	      PC(ik,2,2,il)=PC(ik,1,2,il)*temp
		  temp=((kx/kb0)-Ci*(kz/kb0))
		  PC(ik,-1,2,il)=PC(ik,0,2,il)*temp
	      PC(ik,-2,2,il)=PC(ik,-1,2,il)*temp
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
		  Rd12(ik)=ea(ik,il)*Rd12(ik)*eb(ik,il)
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
        MT12=R12(ik,ilp1)
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
	    U12(ik,il)=W11*Ru12(ik)+W12*Ru22(ik)
	    U21(ik,il)=W21*Ru11(ik)+W22*Ru21(ik)
	    U22(ik,il)=W21*Ru12(ik)+W22*Ru22(ik)
		
		
*       S=((I-Ru*MT)^(-1))*Td
		S11(ik,il)=W11*Td11(ik)+W12*Td21(ik)
		S12(ik,il)=W11*Td12(ik)+W12*Td22(ik)
		S21(ik,il)=W21*Td11(ik)+W22*Td21(ik)
		S22(ik,il)=W21*Td12(ik)+W22*Td22(ik)

*         W=MT*S
        W11=MT11*S11(ik,il)+MT12*S21(ik,il)
		W12=MT11*S12(ik,il)+MT12*S22(ik,il)
		W21=MT21*S11(ik,il)+MT22*S21(ik,il)
		W22=MT21*S12(ik,il)+MT22*S22(ik,il)
 
 
*         R=Rd+Tu*W
        R11(ik,il)=Rd11(ik)+Tu11(ik)*W11+Tu12(ik)*W21
		R12(ik,il)=Rd12(ik)+Tu11(ik)*W12+Tu12(ik)*W22
		R21(ik,il)=Rd21(ik)+Tu21(ik)*W11+Tu22(ik)*W21
		R22(ik,il)=Rd22(ik)+Tu21(ik)*W12+Tu22(ik)*W22

*       W=(I-MT*Ru)^(-1)
	    W11=C1-MT21*Ru12(ik)-MT22*Ru22(ik)
	    W12=MT11*Ru12(ik)+MT12*Ru22(ik)
	    W21=MT21*Ru11(ik)+MT22*Ru21(ik)
	    W22=C1-MT11*Ru11(ik)-MT12*Ru21(ik)
	    InvDet=C1/(W11*W22-W12*W21)


	    W11=W11*InvDet
	    W12=W12*InvDet
	    W21=W21*InvDet
	    W22=W22*InvDet

*       T=Tu*(I-MT*Ru)^(-1)

        T11(ik,il)=Tu11(ik)*W11+Tu12(ik)*W21
		T12(ik,il)=Tu11(ik)*W12+Tu12(ik)*W22
		T21(ik,il)=Tu21(ik)*W11+Tu22(ik)*W21
		T22(ik,il)=Tu21(ik)*W12+Tu22(ik)*W22

99997 continue

*       set PC for layer il

	    Ka0=w/alpha0(il)
	    Kb0=w/beta0(il)
*P
	    Do ik=1,Nk
	      kx=kvec(ik)
		  kz=w*etavec(ik,1)
		  PC(ik,0,1,il)=Csqrt(dk/(Pi*kz))
		  temp=((kx/ka0)+Ci*(kz/ka0))
		  PC(ik,1,1,il)=PC(ik,0,1,il)*temp
	      PC(ik,2,1,il)=PC(ik,1,1,il)*temp
		  temp=((kx/ka0)-Ci*(kz/ka0))
		  PC(ik,-1,1,il)=PC(ik,0,1,il)*temp
	      PC(ik,-2,1,il)=PC(ik,-1,1,il)*temp
	  Enddo

*SV	  
	  Do ik=1,Nk
	      kx=kvec(ik)
		  kz=w*netavec(ik,1)
		  PC(ik,0,2,il)=Csqrt(dk/(Pi*kz))
		  temp=((kx/kb0)+Ci*(kz/kb0))
		  PC(ik,1,2,il)=PC(ik,0,2,il)*temp
	      PC(ik,2,2,il)=PC(ik,1,2,il)*temp
		  temp=((kx/kb0)-Ci*(kz/kb0))
		  PC(ik,-1,2,il)=PC(ik,0,2,il)*temp
	      PC(ik,-2,2,il)=PC(ik,-1,2,il)*temp
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
     &       rho0(0),beta0(0),Rf11(ik),Rf12(ik),Rf21(ik),Rf22(ik))

        easf=CDexp(Ci*w*zs*etavec(ik,1))
        ebsf=CDexp(Ci*w*zs*netavec(ik,1))
        Rf11(ik)=Easf*Rf11(ik)*Easf
        Rf12(ik)=Easf*Rf12(ik)*Ebsf
        Rf21(ik)=Ebsf*Rf21(ik)*Easf
        Rf22(ik)=Ebsf*Rf22(ik)*Ebsf

*       EXCLUDE FREE SURFACE******************************
*        Rf11(ik)=C0
*        Rf12(ik)=C0
*        Rf21(ik)=C0
*        Rf22(ik)=C0
   75 continue


****************************************************************************
*     set Source contribution at the top of the layer stack
*     via S(0)*(1-Rf*R(0))^(-1)(SigD+Rf*SigU)
****************************************************************************
      Ka0=w/alpha0(0)
      Do ik=1,Nk
*       W=(I-Ru*R(1))^(-1)
	    W11=C1-Ru21(ik)*R12(ik,1)-Ru22(ik)*R22(ik,1)
	    W12=Ru11(ik)*R12(ik,1)+Ru12(ik)*R22(ik,1)
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


*         W=R(1)*S
        W11=R11(ik,1)*tempS11+R12(ik,1)*tempS21
		W12=R11(ik,1)*tempS12+R12(ik,1)*tempS22
		W21=R21(ik,1)*tempS11+R22(ik,1)*tempS21
		W22=R21(ik,1)*tempS12+R22(ik,1)*tempS22
 
 
*         R=Rd+Tu*W
        tempR11=Rd11(ik)+Tu11(ik)*W11+Tu12(ik)*W21
		tempR12=Rd12(ik)+Tu11(ik)*W12+Tu12(ik)*W22
		tempR21=Rd21(ik)+Tu21(ik)*W11+Tu22(ik)*W21
		tempR22=Rd22(ik)+Tu21(ik)*W12+Tu22(ik)*W22

*       W=(I-Rf(0)*tempR)^(-1)
	    W11=C1-Rf21(ik)*tempR12-Rf22(ik)*tempR22
	    W12=Rf11(ik)*tempR12+Rf12(ik)*tempR22
	    W21=Rf21(ik)*tempR11+Rf22(ik)*tempR21
	    W22=C1-Rf11(ik)*tempR11-Rf12(ik)*tempR21
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
        SigP(ik)=PD(ik)+Rf11(ik)*PU(ik)+Rf12(ik)*SU(ik)
        SigS(ik)=SD(ik)+Rf21(ik)*PU(ik)+Rf22(ik)*SU(ik)

*       D=V*Sig
        Q1(ik,1)=V11*SigP(ik)+V12*SigS(ik)
        Q2(ik,1)=V21*SigP(ik)+V22*SigS(ik)
*		type*,'Q1(ik,1)=',Q1(ik,1)
*		type*,'Q2(ik,1)=',Q2(ik,1)


*       Reciprocal Receiver for Z-component displacement at (zr,xr)
*       store Q(ik,1) of reciprocal receiver in U11(0),U21(0) for later use
	    PD(ik)=w*etavec(ik,1)*CDSqrt(dk/(4.*Pi*rho0(0)*w*w*w*etavec(ik,1)))*
     *         CDexp(-Ci*w*pvec(ik)*xr)
	    PU(ik)=-PD(ik)
	    SD(ik)=-w*pvec(ik)*CDSqrt(dk/(4.*Pi*rho0(0)*w*w*w*netavec(ik,1)))*
     *         CDexp(-Ci*w*pvec(ik)*xr)
	    SU(ik)=SD(ik)

*       Sig=D+Rf*U
        SigP(ik)=PD(ik)+Rf11(ik)*PU(ik)+Rf12(ik)*SU(ik)
        SigS(ik)=SD(ik)+Rf21(ik)*PU(ik)+Rf22(ik)*SU(ik)

*       Q=V*Sig
        U11(ik,0)=V11*SigP(ik)+V12*SigS(ik)
        U21(ik,0)=V21*SigP(ik)+V22*SigS(ik)
*		type*,'PD(ik)=',PD(ik)
*		type*,'SD(ik)=',SD(ik)
      Enddo
	  
	  If(pointscat)then
      Do il=1,Nl
	    Do ik=1,Nk
	      PC(ik,0,2,il)=C0
		Enddo
	  Enddo
	  Endif
	  
      call DownSweep(Nk,NL,Nlm1,Nscatx,Nscatz,jskip,R11,R12,R21,R22,
     &     S11,S12,S21,S22,PD,SD,PU,SU,SigP,SigS,PC,P1,P2,Q1,Q2,Svec)	  
	  
*     set Q of reciprocal receiver
	  Do ik=1,Nk
	    Q1(ik,1)=U11(ik,0)
	    Q2(ik,1)=U21(ik,0)
	  Enddo

      call DownSweep(Nk,NL,Nlm1,Nscatx,Nscatz,jskip,R11,R12,R21,R22,
     &     S11,S12,S21,S22,PD,SD,PU,SU,SigP,SigS,PC,P1,P2,Q1,Q2,Rvec)	  

*      dump
*      stop
****************************************************************************
*      set U(0)
****************************************************************************

      DO 99999 ik=1,Nk
*      pause'check point 3'

*       set RRu
*       MB=Rf
        MB11=Rf11(ik)
        MB12=Rf12(ik)
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
        MT12=R12(ik,1)
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
	    U12(ik,0)=W11*RRu12+W12*RRu22
	    U21(ik,0)=W21*RRu11+W22*RRu21
	    U22(ik,0)=W21*RRu12+W22*RRu22


99999 continue


*******************************************************
*     set T-Matrix
*******************************************************
      call setT(w,dela,Nscatx,Nscatz,
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
       Subroutine setT(w,dela,Nscatx,Nscatz,
     &            rho0,rho,alpha0,alpha,beta0,beta,
     &            PP11,PP22,PP31,PP33,PP51,PS11,PS51,
     &            PS22,PS31,SS11,SS51,SS22)

      Implicit none
	  Integer     Nscatx,Nscatz,iscatx,iscatz
      Complex*16  w,w2
	   
	   
      Real*8      rho0(0:*),rho(Nscatx,Nscatz)
      Complex*16  alpha0(0:*),beta0(0:*)
      Complex*16  alpha(Nscatx,Nscatz),beta(Nscatx,Nscatz)

	  Complex*16  PP11(Nscatx,Nscatz),PP22(Nscatx,Nscatz),PS31(Nscatx,Nscatz)
	  Complex*16  PP31(Nscatx,Nscatz),PP33(Nscatx,Nscatz),PP51(Nscatx,Nscatz)
	  Complex*16  PS11(Nscatx,Nscatz),PS51(Nscatx,Nscatz),PS22(Nscatx,Nscatz)
	  Complex*16  SS11(Nscatx,Nscatz),SS51(Nscatx,Nscatz),SS22(Nscatx,Nscatz)

      Real*8      dela,drho,den0,den1  
	  Complex*16  ca0,ca1,cb0,cb1,Cb12,Ca12,ca02,cb02,ca04,cb04
	  Complex*16  mu,lam,mu2,lam2
	  Complex*16  da,db,gam1,gam0,zet1,zet0,dgam,dzet

***************************************************************************
      Do iscatz=1,Nscatz
	    ca0=alpha0(iscatz)
	    cb0=beta0(iscatz)
		den0=rho0(iscatz)
        Do iscatx=1,Nscatx
	       ca1=alpha(iscatx,iscatz)
	       cb1=beta(iscatx,iscatz)
		   den1=rho(iscatx,iscatz)
		   
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

	       PP11(iscatx,iscatz)=dCmplx(0.,-.015625)*(dzet - 4.*dgam*mu - 4.*da*mu2 - 
     -       4.*db*mu2)*w2/(ca04*den0)
	  
	       PP22(iscatx,iscatz)=dCmplx(0.,.125)*drho*w2/(ca02*den0)

	       PP31(iscatx,iscatz)=dCmplx(0.,-.03125)*(dzet + 2*dgam*lam + 4.*da*lam*mu + 
     -       4.*da*mu2)*w2/(ca04*den0)
	  
	       PP33(iscatx,iscatz)=dCmplx(0.,-1./16)*(dzet + 4.*dgam*lam - 4.*da*lam2 + 
     -       4.*dgam*mu - 8*da*lam*mu - 4.*da*mu2)*w2/(ca04*den0)
	  
	  
	       PP51(iscatx,iscatz)=dCmplx(0.,-.015625)*(dzet - 4.*dgam*mu - 4.*da*mu2 + 
     -       4.*db*mu2)*w2/(ca04*den0)
	  
	       PS11(iscatx,iscatz)=(-dzet + 4.*dgam*mu + 4.*da*mu2 + 4.*db*mu2)*
     -       w2/(64*ca02*cb02*den0)
	  
	       PS22(iscatx,iscatz)=drho*w2/(8*ca0*cb0*den0)

	       PS31(iscatx,iscatz)=-((dzet + 2*dgam*lam + 4.*da*lam*mu + 4.*da*mu2)*
     -       w2)/(32*ca02*cb02*den0)
	  
	       PS51(iscatx,iscatz)=-((dzet - 4.*dgam*mu - 4.*da*mu2 + 4.*db*mu2)*
     -       w2)/(64*ca02*cb02*den0)
	
	       SS11(iscatx,iscatz)=dCmplx(0.,-.015625)*(dzet - 4.*dgam*mu - 4.*da*mu2 - 
     -       4.*db*mu2)*w2/(cb04*den0)
	  
	       SS22(iscatx,iscatz)=dCmplx(0.,.125)*drho*w2/(cb02*den0)

	       SS51(iscatx,iscatz)=dCmplx(0.,.015625)*(dzet - 4.*dgam*mu - 4.*da*mu2 + 
     -       4.*db*mu2)*w2/(cb04*den0)
        Enddo
      Enddo
      return
      end

****************************************************************************************
      subroutine AtimesV(Nl,NLm1,Nk,Nscatx,Nscatz,jskip,
     &            R11,R12,R21,R22,S11,S12,S21,S22,
     &            U11,U12,U21,U22,T11,T12,T21,T22,
     &            P1,P2,Q1,Q2,PC,SY,PSY,PD,SD,PU,SU,Sigp,SigS,
     &            ea,eb,PP11,PP22,PP31,PP33,PP51,PS11,PS51,
     &            PS22,PS31,SS11,SS51,SS22,getres,V,X)
***************************************************************
*     If[getres,set  X=X-(1-P*dS)V, set X=(1-P*dS)V ]
***************************************************************
      Implicit    none
      logical     getres
      INTEGER     Nl,Nlm1,Nk,Nscatx,Nscatz,jskip
      Complex*16  V(Nscatx,-2:2,2,Nscatz),X(Nscatx,-2:2,2,Nscatz)
      Complex*16  SY(Nscatx,-2:2,2,Nscatz),PSY(Nscatx,-2:2,2,Nscatz)
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
***********************************************************
	 
	 
      Integer     m,j,iscatx,iscatz

*      type*,'entered AtimesV'
      if(getres)then
	    Do iscatz=1,Nscatz
		Do j=1,2
		Do m=-2,2
	    Do iscatx=1,Nscatx
          X(iscatx,m,j,iscatz)=X(iscatx,m,j,iscatz)-V(iscatx,m,j,iscatz)        
		Enddo
		Enddo
		Enddo
		Enddo
      else
	    Do iscatz=1,Nscatz
		Do j=1,2
		Do m=-2,2
	    Do iscatx=1,Nscatx
          X(iscatx,m,j,iscatz)=V(iscatx,m,j,iscatz)        
		Enddo
		Enddo
		Enddo
		Enddo
      Endif
 
*     set SY=S*V
	  call setSY(Nscatx,Nscatz,PP11,PP22,PP31,
     &     PP33,PP51,PS11,PS51,PS22,PS31,SS11,
     &     SS51,SS22,V,SY)

	  Do iscatz=1,Nscatz	  
      Do j=1,2
	  Do m=-2,2
	  Do iscatx=1,Nscatx
*	    type*,'iscatx,m,j,iscatz=',iscatx,m,j,iscatz
*		type*,'SY(iscatx,m,j,iscatz)=',SY(iscatx,m,j,iscatz)
*        PSY(iscatx,m,j,iscatz)=(0.,0.) 
	  Enddo
	  Enddo
	  Enddo
	  Enddo
	  
*     set PSY=P*SY
      call setPSY(Nl,Nlm1,Nk,Nscatx,Nscatz,jskip,
     &           R11,R12,R21,R22,S11,S12,S21,S22,
     &           U11,U12,U21,U22,T11,T12,T21,T22,
     &           ea,eb,PD,SD,PU,SU,sigP,sigS,P1,P2,Q1,Q2,PC,SY,PSY)
      if(getres)then
*       X=X+PSY
	    Do iscatz=1,Nscatz
		Do j=1,2
		Do m=-2,2
	    Do iscatx=1,Nscatx
          X(iscatx,m,j,iscatz)=X(iscatx,m,j,iscatz)+PSY(iscatx,m,j,iscatz)        
		Enddo
		Enddo
		Enddo
		Enddo
      else
*       X=X-PSY
	    Do iscatz=1,Nscatz
		Do j=1,2
		Do m=-2,2
	    Do iscatx=1,Nscatx
          X(iscatx,m,j,iscatz)=X(iscatx,m,j,iscatz)-PSY(iscatx,m,j,iscatz)        
		Enddo
		Enddo
		Enddo
		Enddo
      endif
*      pause 'leaving AtimesV'
      return
      end

      Subroutine setSY(Nscatx,Nscatz,PP11,PP22,PP31,
     &            PP33,PP51,PS11,PS51,PS22,PS31,SS11,
     &            SS51,SS22,Y,SY)
	  Implicit    none
	  Integer     Nscatx,Nscatz,iscatx,iscatz
      Complex*16  Y(Nscatx,-2:2,2,Nscatz),SY(Nscatx,-2:2,2,Nscatz)
	  Complex*16  PP11(Nscatx,Nscatz),PP22(Nscatx,Nscatz),PS31(Nscatx,Nscatz)
	  Complex*16  PP31(Nscatx,Nscatz),PP33(Nscatx,Nscatz),PP51(Nscatx,Nscatz)
	  Complex*16  PS11(Nscatx,Nscatz),PS51(Nscatx,Nscatz),PS22(Nscatx,Nscatz)
	  Complex*16  SS11(Nscatx,Nscatz),SS51(Nscatx,Nscatz),SS22(Nscatx,Nscatz)
	  Complex*16  P1,P2,P3,P4,P5,S1,S2,S4,S5

      Do iscatz=1,Nscatz
	    Do iscatx=1,Nscatx
		  P1=Y(iscatx,-2,1,iscatz)
		  P2=Y(iscatx,-1,1,iscatz)
		  P3=Y(iscatx,0,1,iscatz)
		  P4=Y(iscatx,1,1,iscatz)
		  P5=Y(iscatx,2,1,iscatz)
		  S1=Y(iscatx,-2,2,iscatz)
		  S2=Y(iscatx,-1,2,iscatz)
		  S4=Y(iscatx,1,2,iscatz)
		  S5=Y(iscatx,2,2,iscatz)
	      SY(iscatx,-2,1,iscatz)=P5*PP51(iscatx,iscatz)+P3*PP31(iscatx,iscatz)+
     &          P1*PP11(iscatx,iscatz)-PS51(iscatx,iscatz)*S5+PS11(iscatx,iscatz)*S1
		  SY(iscatx,-1,1,iscatz)=P2*PP22(iscatx,iscatz)+PS22(iscatx,iscatz)*S2
		  SY(iscatx,0,1,iscatz)=P5*PP31(iscatx,iscatz)+P1*PP31(iscatx,iscatz)+
     &          P3*PP33(iscatx,iscatz)+PS31(iscatx,iscatz)*(S1-S5)
		  SY(iscatx,1,1,iscatz)=P4*PP22(iscatx,iscatz)-PS22(iscatx,iscatz)*S4
		  SY(iscatx,2,1,iscatz)=P1*PP51(iscatx,iscatz)+P3*PP31(iscatx,iscatz)+
     &          P5*PP11(iscatx,iscatz)-PS11(iscatx,iscatz)*S5+PS51(iscatx,iscatz)*S1
		  SY(iscatx,-2,2,iscatz)=-P5*PS51(iscatx,iscatz)-P3*PS31(iscatx,iscatz)-
     &          P1*PS11(iscatx,iscatz)+S5*SS51(iscatx,iscatz)+S1*SS11(iscatx,iscatz)
		  SY(iscatx,-1,2,iscatz)=-P2*PS22(iscatx,iscatz)+S2*SS22(iscatx,iscatz)

	      SY(iscatx,0,2,iscatz)=(0.0,0.0)

		  SY(iscatx,1,2,iscatz)=P4*PS22(iscatx,iscatz)+S4*SS22(iscatx,iscatz)
		  SY(iscatx,2,2,iscatz)=P1*PS51(iscatx,iscatz)+P3*PS31(iscatx,iscatz)+
     &          P5*PS11(iscatx,iscatz)+S1*SS51(iscatx,iscatz)+S5*SS11(iscatx,iscatz)
		Enddo
	  Enddo

      return
      end


      SUBROUTINE setPSY(Nl,Nlm1,Nk,Nscatx,Nscatz,jskip,
     &           R11,R12,R21,R22,S11,S12,S21,S22,
     &           U11,U12,U21,U22,T11,T12,T21,T22,
     &           ea,eb,PD,SD,PU,SU,sigP,sigS,P1,P2,Q1,Q2,PC,SY,PSY)
	
*     set X=P*SY
      Implicit none
      INTEGER     Nl,Nlm1,Nk,jskip,Nscatx,Nscatz
      complex*16  SY(Nscatx,-2:2,2,Nscatz),PSY(Nscatx,-2:2,2,Nscatz)	  
      complex*16  PC(Nk,-2:2,2,Nscatz)
 
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
      Complex*16  ea(Nk,1:Nl),eb(Nk,1:Nl)

      Complex*16  PD(Nk),SD(Nk),PU(Nk),SU(Nk)
      Complex*16  sigP(Nk),sigS(Nk)


****************************************************************
*     LOCAL variables
*****************************************************************
*      type*,'entered setPSY'

      call UpSweep(Nk,Nl,Nlm1,Nscatx,Nscatz,jskip,PC,SY,
     &           T11,T12,T21,T22,U11,U12,U21,U22,R11,R12,R21,R22,
     &           S11,S12,S21,S22,SigP,SigS,PD,SD,PU,SU,P1,P2,Q1,Q2)
	 
      call DownSweep(Nk,NL,Nlm1,Nscatx,Nscatz,jskip,R11,R12,R21,R22,
     &           S11,S12,S21,S22,PD,SD,PU,SU,SigP,SigS,PC,P1,P2,Q1,Q2,PSY) 
	 
	  If(Nscatx.gt.1)then
	    call LRsweep(Nk,Nscatz,Nscatx,PC,SY,PSY,PU,SU,PD,SD,Ea,Eb)
      Endif
      RETURN
      END

      Subroutine UpSweep(Nk,Nl,Nlm1,Nscatx,Nscatz,jskip,PC,SY,
     &           T11,T12,T21,T22,U11,U12,U21,U22,R11,R12,R21,R22,
     &           S11,S12,S21,S22,SigP,SigS,PD,SD,PU,SU,P1,P2,Q1,Q2)
      Implicit None
      Integer     NL,Nlm1,Nscatx,Nscatz,jskip,Nk
      complex*16  SY(Nscatx,-2:2,2,Nscatz)	  
      complex*16  PC(Nk,-2:2,2,Nscatz)
 
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
	  
	  Integer     ik,il,ilm1,ilp1,m,j,jscat
	  Real*8      signi
      Complex*16  C0
 
      
      C0=(0.D0,0.D0)
	  
      Do ik=1,Nk
        P1(ik,Nl)=C0
        P2(ik,Nl)=C0
      Enddo
      DO 99998 il=NL,1,-1
		ilm1=il-1
		ilp1=il+1
	    Do ik=1,Nk
		  PD(ik)=C0
		  SD(ik)=C0
		  PU(ik)=C0
		  SU(ik)=C0
		Enddo
		
		Do 99997 m=-2,2
*		  pause 'chk 1'
		
	      Do ik=1,Nk
		    sigP(ik)=C0
		    sigS(ik)=C0
		  Enddo

          j=1
		  Do jscat=1,Nscatx
	        sigP(j)=SY(jscat,m,1,il)
	        j=j+jskip
		  Enddo

*		  pause 'chk 2'

		  if(m.ne.0)then
            j=1
		    Do jscat=1,Nscatx
	          sigS(j)=SY(jscat,m,2,il)
	          j=j+jskip
		    Enddo
		  endif
		
		  signi=-1.0
		  call FFT(sigP,Nk,signi) !x->k
		  if(m.ne.0)then
		    call FFT(sigS,Nk,signi) !x->k
		  endif


		  Do ik=1,Nk	  
            PU(ik)=PU(ik)+PC(ik,m,1,il)*sigP(ik)
            PD(ik)=PD(ik)+PC(ik,-m,1,il)*sigP(ik)
		  Enddo
		  if(m.ne.0)then
		    Do ik=1,Nk	  
              SU(ik)=SU(ik)+PC(ik,m,2,il)*sigS(ik)
              SD(ik)=SD(ik)+PC(ik,-m,2,il)*sigS(ik)
		    Enddo
		  endif
99997    Continue
		
        DO ik=1,Nk
*         P(il)=P(il)+R(il)*D(il)
          P1(ik,il)=P1(ik,il)+R11(ik,il)*PD(ik)+R12(ik,il)*SD(ik)
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
          Q1(ik,il)=U11(ik,ilm1)*PU(ik)+U12(ik,ilm1)*SU(ik)
          Q2(ik,il)=U21(ik,ilm1)*PU(ik)+U22(ik,ilm1)*SU(ik)
        Enddo
		
		if(il.ne.1) then
          DO ik=1,Nk
*           P(ilm1)=T(ilm1)*U
            P1(ik,ilm1)=T11(ik,ilm1)*PU(ik)+T12(ik,ilm1)*SU(ik)
            P2(ik,ilm1)=T21(ik,ilm1)*PU(ik)+T22(ik,ilm1)*SU(ik)
          Enddo
		endif
		
		
99998 continue

      return
      end
 
      Subroutine DownSweep(Nk,NL,Nlm1,Nscatx,Nscatz,jskip,R11,R12,R21,R22,
     &           S11,S12,S21,S22,PD,SD,PU,SU,SigP,SigS,PC,P1,P2,Q1,Q2,PSY)
      Implicit   None
      Integer     Nk,NL,Nlm1,Nscatx,Nscatz,jskip

      Complex*16  R11(Nk,1:Nl),R12(Nk,1:Nl)
      Complex*16  R21(Nk,1:Nl),R22(Nk,1:Nl)
      Complex*16  S11(Nk,1:Nlm1),S12(Nk,1:Nlm1)
      Complex*16  S21(Nk,1:Nlm1),S22(Nk,1:Nlm1)
      Complex*16  P1(Nk,1:Nl),P2(Nk,1:Nl)
      Complex*16  Q1(Nk,1:Nl),Q2(Nk,1:Nl)

      Complex*16  PSY(Nscatx,-2:2,2,Nscatz)
      complex*16  PC(Nk,-2:2,2,Nscatz)

      Complex*16  PD(Nk),SD(Nk),PU(Nk),SU(Nk)
      Complex*16  sigP(Nk),sigS(Nk)

      Integer     il,ilm1,ik,m,j,jscat
      Real*8      signi	  
	  Complex*16  tp,ts,C0
   
      C0=(0.0,0.0)
	  
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
		  PU(ik)=P1(ik,il)+R11(ik,il)*PD(ik)+R12(ik,il)*SD(ik)
		  SU(ik)=P2(ik,il)+R21(ik,il)*PD(ik)+R22(ik,il)*SD(ik)
		Enddo

        Do 99997 m=-2,2

		  Do ik=1,Nk
              SigP(ik)=PC(ik,m,1,il)*PD(ik)+PC(ik,-m,1,il)*PU(ik)
		  Enddo
		  
		  signi=1.0
		  call FFT(SigP,Nk,signi) !k->x
          
		  j=1
		  Do jscat=1,Nscatx	  
            PSY(jscat,m,1,il)=SigP(j)
			j=j+jskip
		  Enddo
			
          If(m.ne.0) then
		    Do ik=1,Nk
              SigS(ik)=PC(ik,m,2,il)*SD(ik)+PC(ik,-m,2,il)*SU(ik)
		    Enddo
		  
		    signi=1.0
		    call FFT(SigS,Nk,signi) !k->x
          
		    j=1
		    Do jscat=1,Nscatx	  
              PSY(jscat,m,2,il)=SigS(j)
			  j=j+jskip
		    Enddo
          Endif		  
99997	continue
*       zero unused component
		j=1
		Do jscat=1,Nscatx	  
            PSY(jscat,0,2,il)=C0
			j=j+jskip
		Enddo
*		
99998 continue

      return
      end
	
      subroutine LRsweep(Nk,Nscatz,Nscatx,PC,SY,PSY,PU,SU,PD,SD,Ea,Eb)
	  Implicit   none
	  integer    Nk,Nscatz,Nscatx
      Complex*16 PC(Nk,-2:2,2,Nscatz),SY(Nscatx,-2:2,2,Nscatz)
      Complex*16 PSY(Nscatx,-2:2,2,Nscatz)
	  Complex*16 PU(Nk),SU(Nk),PD(Nk),SD(Nk),Ea(Nk,Nscatz),Eb(Nk,Nscatz)
	  
	  Integer    iscat,jscat,ik,i,m
	  Complex*16 C0,Ci,im
	  
      C0=(0.D0,0.D0)
      Ci=(0.D0,1.D0)
	  
      Do 5000 iscat=Nscatz,1,-1
	  
*	    type*,'iscat=',iscat

	    Do i=1,Nk,2
	      PU(i)=C0
	      SU(i)=C0
	    Enddo


*       begin right sweep
		Do 3000 jscat=1,Nscatx
*		  type*,'jscat=',jscat

*k->x
        Do 90 m=-2,2
		  im=2.0*((Ci)**m)
		  Do ik=1,Nk,2
            PSY(jscat,m,1,iscat)=PSY(jscat,m,1,iscat)+im*PC(ik,-m,1,iscat)*PU(ik)
            PSY(jscat,m,2,iscat)=PSY(jscat,m,2,iscat)+im*PC(ik,-m,2,iscat)*SU(ik)
		  Enddo
90		continue
		
		Do 30 m=-2,2
		  im=(-Ci)**m
		  Do ik=1,Nk,2	  
            PU(ik)=PU(ik)+im*PC(ik,m,1,iscat)*SY(jscat,m,1,iscat)
            SU(ik)=SU(ik)+im*PC(ik,m,2,iscat)*SY(jscat,m,2,iscat)
		  Enddo
30      continue		

		Do ik=1,Nk,2	  
		  PU(ik)=Ea(ik,iscat)*PU(ik)
		  SU(ik)=Eb(ik,iscat)*SU(ik)
		Enddo

3000  continue  !end right sweep

	  Do ik=1,Nk
	    PD(ik)=C0
	    SD(ik)=C0
	  Enddo
      
*     begin left sweep
      Do 4000 jscat=Nscatx,1,-1
	  
*	    type*,'jscat=',jscat


        Do 110 m=-2,2
		
          im=2.*((Ci)**m)
		  Do ik=1,Nk,2
            PSY(jscat,m,1,iscat)=PSY(jscat,m,1,iscat)+im*PC(ik,m,1,iscat)*PD(ik)
            PSY(jscat,m,2,iscat)=PSY(jscat,m,2,iscat)+im*PC(ik,m,2,iscat)*SD(ik)
		  Enddo
110		continue
	  
		
		Do 120 m=-2,2
          im=(-Ci)**m
		  Do ik=1,Nk,2	  
            PD(ik)=PD(ik)+im*PC(ik,-m,1,iscat)*SY(jscat,m,1,iscat)
            SD(ik)=SD(ik)+im*PC(ik,-m,2,iscat)*SY(jscat,m,2,iscat)
		  Enddo
120      continue

		Do ik=1,Nk,2	  
		  PD(ik)=Ea(ik,iscat)*PD(ik)
		  SD(ik)=Eb(ik,iscat)*SD(ik)
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
 
 
      subroutine Rfs(P,ETA,NETA,RHO,BETA,Rf11,Rf12,Rf21,Rf22)
*********************************************************
*     form the Free surface reflection response
*********************************************************
      implicit none
      Real*8    rho
      Complex*16 beta,twomup,p
      Complex*16 Rf11,Rf12,Rf21,Rf22
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
      Rf12=InvDet*(ND(1,2)*NU(2,2)-ND(2,2)*Nu(1,2))
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


      Complex*16 function setXKSY(Nscatx,Nscatz,PP11,PP22,PP31,
     &            PP33,PP51,PS11,PS51,PS22,PS31,SS11,SS51,SS22,X,Y)
*     set X*K*Smat*Y
	  Implicit    none
	  Integer     Nscatx,Nscatz,iscatx,iscatz
      Complex*16  X(Nscatx,-2:2,2,Nscatz),Y(Nscatx,-2:2,2,Nscatz)
	  Complex*16  PP11(Nscatx,Nscatz),PP22(Nscatx,Nscatz),PS31(Nscatx,Nscatz)
	  Complex*16  PP31(Nscatx,Nscatz),PP33(Nscatx,Nscatz),PP51(Nscatx,Nscatz)
	  Complex*16  PS11(Nscatx,Nscatz),PS51(Nscatx,Nscatz),PS22(Nscatx,Nscatz)
	  Complex*16  SS11(Nscatx,Nscatz),SS51(Nscatx,Nscatz),SS22(Nscatx,Nscatz)
	  Complex*16  XKSY,P1,P2,P3,P4,P5,S1,S2,S4,S5
	  Complex*16  dP1,dP2,dP3,dP4,dP5,dS1,dS2,dS4,dS5

      XKSY=(0.D0,0.D0)
      Do iscatz=1,Nscatz
	    Do iscatx=1,Nscatx
		  P1=Y(iscatx,-2,1,iscatz)
		  P2=Y(iscatx,-1,1,iscatz)
		  P3=Y(iscatx,0,1,iscatz)
		  P4=Y(iscatx,1,1,iscatz)
		  P5=Y(iscatx,2,1,iscatz)
		  S1=Y(iscatx,-2,2,iscatz)
		  S2=Y(iscatx,-1,2,iscatz)
		  S4=Y(iscatx,1,2,iscatz)
		  S5=Y(iscatx,2,2,iscatz)

		  dP1=X(iscatx,-2,1,iscatz)
		  dP2=X(iscatx,-1,1,iscatz)
		  dP3=X(iscatx,0,1,iscatz)
		  dP4=X(iscatx,1,1,iscatz)
		  dP5=X(iscatx,2,1,iscatz)
		  dS1=X(iscatx,-2,2,iscatz)
		  dS2=X(iscatx,-1,2,iscatz)
		  dS4=X(iscatx,1,2,iscatz)
		  dS5=X(iscatx,2,2,iscatz)

          XKSY=XKSY+(P5*PP51(iscatx,iscatz)+P3*PP31(iscatx,iscatz)+P1*PP11(iscatx,iscatz)-
     &        PS51(iscatx,iscatz)*S5+PS11(iscatx,iscatz)*S1)*dP5-(P2*PP22(iscatx,iscatz)+
     &        PS22(iscatx,iscatz)*S2)*dP4+(P5*PP31(iscatx,iscatz)+P1*PP31(iscatx,iscatz)+
     &        P3*PP33(iscatx,iscatz)-PS31(iscatx,iscatz)*S5+PS31(iscatx,iscatz)*S1)*dP3+
     &        (-P4*PP22(iscatx,iscatz)+PS22(iscatx,iscatz)*S4)*dP2+(P1*PP51(iscatx,iscatz)+
     &        P3*PP31(iscatx,iscatz)+P5*PP11(iscatx,iscatz)-PS11(iscatx,iscatz)*S5+
     &        PS51(iscatx,iscatz)*S1)*dP1+(-P5*PS51(iscatx,iscatz)-P3*PS31(iscatx,iscatz)-
     &        P1*PS11(iscatx,iscatz)+S5*SS51(iscatx,iscatz)+S1*SS11(iscatx,iscatz))*dS5+
     &        (P2*PS22(iscatx,iscatz)-S2*SS22(iscatx,iscatz))*dS4-(P4*PS22(iscatx,iscatz)+
     &        S4*SS22(iscatx,iscatz))*dS2+(P1*PS51(iscatx,iscatz)+P3*PS31(iscatx,iscatz)+
     &        P5*PS11(iscatx,iscatz)+S1*SS51(iscatx,iscatz)+S5*SS11(iscatx,iscatz))*dS1
	 
		Enddo
	  Enddo
      setXKSY=XKSY
      return
      end

*******************************************************************************
*******************************************************************************
*******************************************************************************
*******************************************************************************
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

      double precision function dcabs1(z)
      double complex z,zz
      double precision t(2)
      equivalence (zz,t(1))
      zz = z
      dcabs1 = dabs(t(1)) + dabs(t(2))
      return
      end

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
