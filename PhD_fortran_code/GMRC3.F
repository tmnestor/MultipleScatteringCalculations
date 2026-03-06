      Program GMRC3
*******************************************************************
*     Homogeneous Background
*******************************************************************
      implicit     none
      INTEGER      N,Np,Np4,Nl,Nl1
      Parameter    (Np=257,Np4=4*Np)
      parameter    (Nl=2,NL1=Nl+1,N=Nl*Np4)
      Real*8       h(0:Nl),xleft(Nl),xright(Nl)
      Real*8       rho0(0:Nl1),rho(0:Nl1),drho(NL)
      Complex*16   alpha0(0:Nl1),beta0(0:Nl1),dbeta(NL)
      Complex*16   alpha(0:Nl1),beta(0:Nl1)
      Complex*16   QB(0:Nl1)
      Complex*16   RespX(Np),RespZ(Np),CV
      Complex*16   RespX0(Np),RespZ0(Np)
      Real*8       rt3
      integer      il,i
      integer      i1,i2
      Real*8       vu,cpu
******************************************************************
      Complex*16  Vexact(N)
      logical     exact
******************************************************************
*     TGMRES1 declarations
      INTEGER      RESTRT, ITER, INFO
      parameter    (Restrt=10,Iter=11)
      real*8       RESID
      parameter    (resid=1.D-6)
      complex*16   B( N ), BB( N ), X( N )
      complex*16   WORK(N,RESTRT+4), Hmat(N,RESTRT+2)
*********************************************************
      Complex*16  SY(N),PSY(N)

      Complex*16  T11(Np,0:Nl),T12(Np,0:Nl)
      Complex*16  T21(Np,0:Nl),T22(Np,0:Nl)
      Complex*16  U11(Np,0:Nl),U12(Np,0:Nl)
      Complex*16  U21(Np,0:Nl),U22(Np,0:Nl)
      Complex*16  R11(Np,0:Nl),R12(Np,0:Nl)
      Complex*16  R21(Np,0:Nl),R22(Np,0:Nl)
      Complex*16  S11(Np,0:Nl),S12(Np,0:Nl)
      Complex*16  S21(Np,0:Nl),S22(Np,0:Nl)
      Complex*16  P1(Np,0:Nl),P2(Np,0:Nl)
      Complex*16  Q1(Np,0:Nl),Q2(Np,0:Nl)
 
      Complex*16  Rd11(Np,0:Nl),Rd12(Np,0:Nl)
      Complex*16  Rd21(Np,0:Nl),Rd22(Np,0:Nl)
      Complex*16  Tu11(Np,0:Nl),Tu12(Np,0:Nl)
      Complex*16  Tu21(Np,0:Nl),Tu22(Np,0:Nl)
      Complex*16  Td11(Np,0:Nl),Td12(Np,0:Nl)
      Complex*16  Td21(Np,0:Nl),Td22(Np,0:Nl)
      Complex*16  Ru11(Np,-1:Nl),Ru12(Np,-1:Nl)
      Complex*16  Ru21(Np,-1:Nl),Ru22(Np,-1:Nl)
      Complex*16  etamat(Np,0:Nl1),netamat(Np,0:Nl1)
      Complex*16  rtEa(Np,0:Nl),rtEb(Np,0:Nl),Ea(Np,0:Nl),Eb(Np,0:Nl)
      Complex*16  SigPu(Np),SigSu(Np),SigPd(Np),SigSd(Np)
      Complex*16  RecPu(Np),RecSu(Np),RecPd(Np),RecSd(Np)
 
*******************************************************
*     Source-Receiver information
*******************************************************

*     reflection geometry
      logical reflex
      parameter(reflex=.true.)
*     source type
      Integer is
*     is=1 => double couple
*     is=2 => explosion
*     is=3 => Incident plane P-Wave
*     Source depth Zs
      Real*8  Zs
*     Source offset Xs
      Real*8  Xs
      Parameter(is=3,Zs=0.5D0,Xs=0.D0)
*      Parameter(is=2,Zs=0.D0,Xs=0.D0)

*     receiver type
      Integer ir
*     ir=1 => Z-component response
*     ir=2 => X-component response
*     Receiver depth Zr
      Real*8  Zr
*     Receiver offset Xr
      Real*8  Xr
      Parameter(ir=1,Zr=0.5D0,Xr=0.D0)

*******************************************************
*     set p Lattice Data (- Kmax<= w p <= Kmax)
*******************************************************
      Integer     K1,ip
	  
*     spatial period 
      Real*8      L
	  Parameter   (L=1.D02)
*     wave number interval dk
      Real*8      dk
      parameter   (dk=2.D0*3.14159265358979324D0/L)

      Real*8      Kmax
      parameter   (Kmax=0.5D0*(Np-1)*dk)

      Complex*16  pvec(Np)
*******************************************************
*     TIME SERIES DATA
*******************************************************
      Integer    Nw,iw,iwmax
      Parameter  (iwmax=31)
      Integer    iwvec(iwmax)
      Real*8     fmin,fmax !fmax-fmin=bandwidth
      Real*8     Wmin,dw,eps
      Parameter  (fmin=1.0D0,fmax=5.0D0) !Hz
      Real*8     T  !time series length
      Parameter  (T=12.5,dw=2.D0*3.14159265358979324D0/T)
      Parameter  (Wmin=2.D0*3.14159265358979324D0*fmin)
      Complex*16 w
*******************************************************
* IN/OUT FILES
*******************************************************
       DATA Exact/.false./  !read in exact answer
*******************************************************
*     Scattering Model Data
*     2 scattering layer-2 cylinder case
      Integer    M,Bs,iz
	  Parameter  (M=2,Bs=2*M+1)
	  Complex*16 Tpp(Bs,Nl),Tps(Bs,Nl),Tsp(Bs,Nl),Tss(Bs,Nl)
      Real*8     Xc(Nl),Rad,z
      Parameter  (Rad=.115D0)
	  Data       Xc/2*(0.D0)/
*******************************************************

	  DATA     beta0/4*(3.5)/	
      DATA     rho0/4*(3.0)/

      DATA     drho/2*0.5D0/
      DATA     dbeta/2*0.4D0/


      DATA     QB/4*1.D10/

*     Reference Layer Depths
      DATA     h/3*1.0/



      OPEN( UNIT=11, FILE='GMRCRespP',
     &     STATUS='UNKNOWN')
      OPEN( UNIT=12, FILE='GMRCRespS',
     &     STATUS='UNKNOWN')
 
      rt3=DSqrt(3.0D0)
      rho(0)=rho0(0)
      rho(Nl1)=rho0(Nl1)
      Do 15 il=1,NL
          rho(il)=rho0(il)+drho(il)
  15  continue
 
*     damping factor
      eps=dlog(20.D0)/T
      w=dcmplx(Wmin,eps)
**************************************************
*     frequency loop
**************************************************
      Nw=(fmax-fmin)*T+1
*      Do 111 iw=1,iwmax
*      Do 111 iw=1,128
       Do 111 iw=1,1

         w=dcmplx(dFloat(iw)*dw,eps)
		 type*,'w=',w

*         w=dcmplx(iwvec(iw)*dw,eps)

        beta0(0)=CV(w,beta0(0),QB)
	beta(0)=beta0(0)
	beta0(Nl1)=CV(w,beta0(Nl1),QB)
	beta(Nl1)=beta0(Nl1)
	alpha0(0)=rt3*beta0(0)
	alpha(0)=alpha0(0)
	alpha0(Nl1)=rt3*beta0(Nl1)
	alpha(Nl1)=alpha0(Nl1)
        DO    99 IL=1,NL
          beta0(il)=CV(w,beta0(il),QB)
	  beta(il)=beta0(il)+CV(w,dbeta(il),QB)
	  alpha0(il)=rt3*beta0(il)
	  alpha(il)=rt3*beta(il)
   99   continue
   
      Do il=1,Nl
	    Type*,'il=',il
		TYpe*,'w=',w
	    Type*,'rho0(il)=',rho0(il)
	    Type*,'beta0(il)=',beta0(il)
	    Type*,'alpha0(il)=',alpha0(il)
	    Type*,'rho(il)=',rho(il)
	    Type*,'beta(il)=',beta(il)
	    Type*,'alpha(il)=',alpha(il)
		Type*,'radius=',0.5*h(il)
	  Enddo
	  pause
	  
	  TPP(1,1)=(0.0001022340025694636049,-0.00008327192297714449804)
	  TPS(1,1)=(-0.0002498485049468744366,-0.0003065456620455889155)
	  TSP(1,1)=(0.0002498485049468743807,0.000306545662045588931)
	  TSS(1,1)=(0.0009191681078175447773,-0.0007496434134140165369)
	  
	  TPP(2,1)=(-0.0002692346165448888915,0.0002175383866810270383)
	  TPS(2,1)=(0.0003768256005807844324,0.0004661528989507522429)
	  TSP(2,1)=(-0.0003768256005807844067,-0.0004661528989507522968)
	  TSS(2,1)=(-0.0008063496726406734319,0.0006529045848155467789)
	  
	  TPP(3,1)=(0.0004095131079242027192,-0.0003327286329148201023)
	  TPS(3,1)=(0.,0.)
	  TSP(3,1)=(0.,0.)
	  TSS(3,1)=(-1.22986446543895287e-6,-2.597571420802442845e-7)
	  
	  TPP(4,1)=(-0.0002692346165448888988,0.0002175383866810270689)
	  TPS(4,1)=(-0.0003768256005807844857,-0.000466152898950752248)
	  TSP(4,1)=(0.0003768256005807844598,0.0004661528989507523094)
	  TSS(4,1)=(-0.0008063496726406734407,0.0006529045848155468712)
	  
	  TPP(5,1)=(0.0001022340025694636044,-0.00008327192297714449767)
	  TPS(5,1)=(0.000249848504946874442,0.0003065456620455889222)
	  TSP(5,1)=(-0.0002498485049468743796,-0.0003065456620455889296)
	  TSS(5,1)=(0.0009191681078175447973,-0.0007496434134140165531)
	  
	  TPP(1,2)=(0.00006016309946747488812,-0.00004891287688336352571)
	  TPS(1,2)=(-0.0001467539035846403964,-0.0001804166712641851761)
	  TSP(1,2)=(0.0001467539035846404318,0.0001804166712641851516)
	  TSS(1,2)=(0.0005410321893308980612,-0.0004403074183311272072)
	  
	  TPP(2,2)=(-0.0001239578024784236645,0.0001003615866860314024)
	  TPS(2,2)=(0.0001738422161152468688,0.0002146507928305652192)
	  TSP(2,2)=(-0.0001738422161152467562,-0.0002146507928305652269)
	  TSS(2,2)=(-0.0003713680481063097738,0.0003011921599462799122)
	  
	  TPP(3,2)=(0.0002408671952211800404,-0.0001955237977180720383)
	  TPS(3,2)=(0.,0.)
	  TSP(3,2)=(0.,0.)
	  TSS(3,2)=(-4.478960725693927248e-7,-9.46219035232695101e-8)
	  
	  TPP(4,2)=(-0.0001239578024784236645,0.0001003615866860314024)
	  TPS(4,2)=(-0.0001738422161152468688,-0.0002146507928305652192)
	  TSP(4,2)=(0.0001738422161152467562,0.0002146507928305652269)
	  TSS(4,2)=(-0.0003713680481063097738,0.0003011921599462799122)
	  
	  TPP(5,2)=(0.00006016309946747488812,-0.00004891287688336352571)
	  TPS(5,2)=(0.0001467539035846403964,0.0001804166712641851761)
	  TSP(5,2)=(-0.0001467539035846404318,-0.0001804166712641851516)
	  TSS(5,2)=(0.0005410321893308980612,-0.0004403074183311272072)

**************************************************
*     slowness lattice
**************************************************

      Do i=1,np
        pvec(i)=Cmplx(-kmax+(i-1)*dk)/w
*        type*,'i,pvec(i)=',i,pvec(i)
      enddo

*        pause
 
*       calculate layer stack reflection response
	
*************************************************************
*     initialise  R/T coeffs and factorise System matrix
*************************************************************
      call Init(Nl,Nl1,Np,Np4,pvec,etamat,netamat,
     &           w,h,xleft,xright,rho0,alpha0,beta0,rho,
     &           alpha,beta,Rd11,Rd12,Rd21,Rd22,Td11,Td12,
     &           Td21,Td22,Tu11,Tu12,Tu21,Tu22,Ru11,Ru12,
     &           Ru21,Ru22,rtEa,rtEb,Ea,Eb,R11,R12,R21,R22,
     &           S11,S12,S21,S22,U11,U12,U21,U22,T11,T12,T21,T22)
 
 
****************************************************************
*     set incident field B and Reciprocal field BB basis states
***************************************************************

      call setRefBasis(reflex,is,ir,Nl,Nl1,N,Np,Zs,Xs,
     &           Zr,Xr,rho(0),alpha(0),beta(0),pvec,w,h,etamat,
     &           netamat,R11,R12,R21,R22,S11,S12,S21,S22,
     &           U11,U12,U21,U22,T11,T12,T21,T22,
     &           Rd11,Rd12,Rd21,Rd22,Td11,Td12,
     &           Td21,Td22,Tu11,Tu12,Tu21,Tu22,
     &           Ru11,Ru12,Ru21,Ru22,
     &           P1,P2,Q1,Q2,ea,eb,
     &           SigPu,SigPd,SigSu,SigSd,
     &           RecPu,RecPd,RecSu,RecSd,
     &           RespX0,RespZ0,B,BB)

	   
 
*      Do i=1,N
*	    type*,'i,B(i)=',i,B(i)
*      Enddo

*      Do i=1,N
*	    type*,'i,BB(i)=',i,BB(i)
*      Enddo
*	  pause
 
****************************************************
*     Read in exact solution if exact=.true.
****************************************************
      if(exact) then
        open(19,file='ExactDat',status='old')
      DO 99999 IL=1,N
	  Read(19,*) Vexact(il)
*	  type*,'Vexact(il)=',il,Vexact(il)
99999 continue
	close (19)
      endif
****************************************************
*     set initial approx X0
      DO 99998 I=1,N
	    X(i)=B(i)       !BORN trial solution
*	    X(i)=Vexact(i)   !DELETE
99998 continue
 
       call GMRES( N,  B, X, RESTRT, WORK, Hmat, ITER,
     &           RESID, INFO,Nl,Np,Np4,
     &            M,Bs,w,L,Xc,alpha0,beta0,pvec,etamat,netamat,
     &            rtEa,rtEb,Tpp,Tps,Tsp,Tss, 
     &            R11,R12,R21,R22,S11,S12,S21,S22,
     &            U11,U12,U21,U22,T11,T12,T21,T22,
     &            P1,P2,Q1,Q2,SY,PSY,ea,eb)

       type*,'iter,resid,info=',iter,resid,info
       pause
 
        if(exact) then
          type*,'iterate,Solution:',iter
      DO 99997 I=1,N
	    type*, 'CDabs(Vexact(i)-X(i))/CDabs(Vexact(i))=',
     &          i,CDabs(Vexact(i)-X(i))/CDabs(Vexact(i))
            type*
99997 continue
        else
            type*,'solution coverged after iteration',iter
            type*,'approximate L2 error=',resid
       DO 99996 I=1,N
	      type*, 'X(i)=',i,X(i)
          type*
99996 continue
        endif
 
*     set the receiver response
        call setResp(Nl,Np,ea,eb,T11,T12,T21,T22,X,RespX,RespZ)
 
*       write TOTAL=REFERENCE+SCATTERED surface 
*       response at freq. iw to file
        Write(11,*)  iwvec(iw),w
 
        DO    91 I=1,Np
          Write(11,*)  i,Pvec(i)
   91   continue
 
        DO    97 I=1,Np
          type*,'i,RespX(i)=',i,RespX(i)
          Write(11,*)  RespX0(i)+RespX(i)
   97   continue
 
        Write(12,*)  iwvec(iw),w
 
        DO    96 I=1,Np
          type*,'i,RespZ(i)=',i,RespZ(i)
          Write(12,*)  RespZ0(i)+RespZ(i)
   96   continue
 
*        w=w+dcmplx(dw)
**************************************************
  111 continue !end frequency loop
**************************************************
 
*      call clockv(vu,cpu,i1,i2)
      Write(11,*) 'normal completion'
      Write(12,*) 'normal completion'
*      Write(11,*)  'vu=',vu
*      Write(11,*)  'cpu=',cpu
 
*      Write(12,*)  'vu=',vu
*      Write(12,*)  'cpu=',cpu
 
      close (11)
      close (12)
 
      stop
      end

********************************************************
      SUBROUTINE GMRES( N,  B, X, RESTRT, WORK, Hmat, ITER,
     &           RESID, INFO,Nl,Np,Np4,
     &            M,Bs,w,L,Xc,alpha0,beta0,pvec,etamat,netamat,
     &            rtEa,rtEb,Tpp,Tps,Tsp,Tss,	 
     &            R11,R12,R21,R22,S11,S12,S21,S22,
     &            U11,U12,U21,U22,T11,T12,T21,T22,
     &            P1,P2,Q1,Q2,SY,PSY,ea,eb)


*     .. Scalar Arguments ..
      INTEGER            N, RESTRT, ITER, INFO
      real*8             RESID
*     ..
*     .. Array Arguments ..
      complex*16   B( N ), X( N )
      complex*16   WORK(N,RESTRT+4), Hmat(N,RESTRT+2)
****************************************************************
      INTEGER     Nl,Np,Np4
      Complex*16  ea(Np,0:Nl),eb(Np,0:Nl)
      Complex*16  SY(N),PSY(N)
 
      Complex*16  T11(Np,0:Nl),T12(Np,0:Nl)
      Complex*16  T21(Np,0:Nl),T22(Np,0:Nl)
      Complex*16  U11(Np,0:Nl),U12(Np,0:Nl)
      Complex*16  U21(Np,0:Nl),U22(Np,0:Nl)
      Complex*16  R11(Np,0:Nl),R12(Np,0:Nl)
      Complex*16  R21(Np,0:Nl),R22(Np,0:Nl)
      Complex*16  S11(Np,0:Nl),S12(Np,0:Nl)
      Complex*16  S21(Np,0:Nl),S22(Np,0:Nl)
      Complex*16  P1(Np,0:Nl),P2(Np,0:Nl)
      Complex*16  Q1(Np,0:Nl),Q2(Np,0:Nl)

****************************************************************
      Integer     M,Bs
      Real*8      w,L,Xc(Nl)
      Complex*16  alpha0(0:Nl),beta0(0:Nl),pvec(Np)
      Complex*16  etamat(Np,0:Nl),netamat(Np,0:Nl)
      Complex*16  rtEa(Np,0:Nl),rtEb(Np,0:Nl)
      Complex*16  Tpp(Bs,Nl),Tps(Bs,Nl),Tsp(Bs,Nl),Tss(Bs,Nl)
	 
***********************************************************
	  
*
*  Purpose
*  =======
*
*  GMRES solves the linear system Ax = b using the
*  Generalized Minimal Residual iterative method with preconditioning.
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
*  WORK    (workspace) complex*16 array, dimension (N,RESTRT+4).
*
*  Hmat       (workspace) complex*16 array, dimension (N,RESTRT+2).
*          This workspace is used for constructing and storing the
*          upper Hessenberg matrix. The two extra columns are used to
*          store the Givens rotation matrices.
*
*  N    (input) INTEGER
*          The leading dimension of the array Hmat. N >= max(1,RESTRT+1).
*
*  ITER    (input/output) INTEGER
*          On input, the maximum iterations to be performed.
*          On output, actual number of iterations performed.
*
*  RESID   (input/output) complex*16
*          On input, the allowable convergence measure for
*          norm( b - A*x ) / norm( b ).
*          On output, the final value of this measure.
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
*
*  BLAS CALLS:   ZAXPY, ZCOPY, ZDOTC, DZNRM2, ZROT, ZROTG, ZDSCAL
*  ============================================================
*
*     .. Parameters ..
      Implicit none
	  real*8            ZERO,ONE
	  Parameter         ( ZERO=0.0D+0 , ONE=1.0D+0 )
      complex*16         C0,C1
      PARAMETER         (C0 = (0.0D+0,0.0D+0), C1 = (1.0D+0,0.0D+0))
*     ..
*     .. Local Scalars ..
      INTEGER             I, K, MAXIT, AV, CS, SN, R, S, V, W1, Y
	  real*8              BNRM2, RNORM, TOL, DZNRM2
      complex*16          AA, BB, ZDOT
*     ..
*     .. External Routines ..
      EXTERNAL            ZAXPY, ZCOPY, ZDOT, DZNRM2, ZROT, ZROTG, ZSCAL
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
      ELSE IF ( N.LT.RESTRT+1 ) THEN
         INFO = -4
      ENDIF
      IF ( INFO.NE.0 ) RETURN
*
      MAXIT = ITER
      TOL   = RESID
*
*     Alias workspace columns. 
*
      R  = 1
      S  = R + 1
      W1  = S + 1
      Y  = W1
      AV = Y
      V  = AV + 1
*
*     Store the Givens parameters in matrix Hmat.
*
      CS = RESTRT + 1
      SN = CS + 1
*
*     Set initial residual (AV is temporary workspace here).
*
      CALL ZCOPY( N, B, 1, WORK(1,AV), 1 )
      IF ( DZNRM2( N, X, 1 ).NE.ZERO ) THEN
*
*        AV is temporary workspace here.
*
         CALL ZCOPY( N, B, 1, WORK(1,AV), 1 )

*         CALL ZGEMV( 'N', N, N, -C1, A, N, 
*     *        X, 1, C1, WORK(1,AV), 1 )

*         call GEMVmask(Nl,Np,om,pvec,dh,den0,alf0,bet0,
*     &        den,alf,bet,.true.,X,WORK(1,AV))
      call AtimesV(N,Nl,Np,Np4,
     &            M,Bs,w,L,Xc,alpha0,beta0,pvec,etamat,netamat,
     &            rtEa,rtEb,Tpp,Tps,Tsp,Tss,
     &            R11,R12,R21,R22,S11,S12,S21,S22,
     &            U11,U12,U21,U22,T11,T12,T21,T22,
     &            P1,P2,Q1,Q2,SY,PSY,ea,eb,
     &            .true.,X,WORK(1,AV)) 

		 
      ENDIF
	  
      CALL ZCOPY( N, WORK( 1,AV ), 1, WORK( 1,R ), 1 )

      BNRM2 = DZNRM2( N, B, 1 )
      IF ( BNRM2.EQ.ZERO ) BNRM2 = ONE
      IF ( ( DZNRM2( N, WORK(1,R), 1 ) / BNRM2 ).LT.TOL ) GO TO 70
*
      ITER = 0
*
   10 CONTINUE
*
         type*,'DZNRM2( N, WORK(1,R), 1 )=',DZNRM2( N, WORK(1,R), 1 )
         I = 0
*
*        Construct the first column of V.
*
         CALL ZCOPY( N, WORK( 1,R ), 1, WORK( 1,V ), 1 )
         RNORM = DZNRM2( N, WORK( 1,V ), 1 )
         CALL ZDSCAL( N, ONE / RNORM, WORK( 1,V ), 1 )
*
*        Initialize S to the elementary vector E1 scaled by RNORM.
*
         WORK( 1,S ) = RNORM
         DO 20 K = 2, N
            WORK( K,S ) = C0
   20    CONTINUE
*
   30    CONTINUE
*
            I = I + 1
            ITER = ITER + 1
*

*            CALL ZGEMV( 'N', N, N, C1, A, N, 
*     &        WORK( 1,V+I-1 ), 1, C0, WORK(1,AV), 1 )

*             call GEMVmask(Nl,Np,om,pvec,dh,den0,alf0,bet0,
*     &        den,alf,bet,.false.,WORK( 1,V+I-1 ), WORK(1,AV))

      call AtimesV(N,Nl,Np,Np4,
     &            M,Bs,w,L,Xc,alpha0,beta0,pvec,etamat,netamat,
     &            rtEa,rtEb,Tpp,Tps,Tsp,Tss,
     &            R11,R12,R21,R22,S11,S12,S21,S22,
     &            U11,U12,U21,U22,T11,T12,T21,T22,
     &            P1,P2,Q1,Q2,SY,PSY,ea,eb,
     &            .false.,WORK( 1,V+I-1 ),WORK(1,AV)) 

			
            CALL ZCOPY( N, WORK(1,AV), 1, WORK( 1,W1 ), 1 )
			
*
*           Construct I-th column of Hmat orthnormal to the previous 
*           I-1 columns.
*
            CALL BASIS( I, N, Hmat( 1,I ), WORK( 1,V ), N, WORK( 1,W1 ) )
*
*           Apply Givens rotations to the I-th column of Hmat. This
*           "updating" of the QR factorization effectively reduces
*           the Hessenberg matrix to upper triangular form during
*           the RESTRT iterations.
*
            DO 40 K = 1, I-1
               CALL ZROT( 1, Hmat( K,I ), N, Hmat( K+1,I ), N,
     &            Hmat( K,CS ), Hmat( K,SN ) )
   40       CONTINUE
*
*           Construct the I-th rotation matrix, and apply it to Hmat so that
*           Hmat(I+1,I) = 0.
*
            AA = Hmat( I,I )
            BB = Hmat( I+1,I )
            CALL ZROTG( AA, BB, Hmat( I,CS ), Hmat( I,SN ) )
            CALL ZROT( 1, Hmat( I,I ), N, Hmat( I+1,I ), N,
     &            Hmat( I,CS ), Hmat( I,SN ) )
*
*           Apply the I-th rotation matrix to [ S(I), S(I+1) ]'. This
*           gives an approximation of the residual norm. If less than
*           tolerance, update the approximation vector X and quit.
*
            CALL ZROT( 1, WORK( I,S ), N, WORK( I+1,S ), N,
     &         Hmat( I,CS ), Hmat( I,SN ) )
            RESID = ABS( WORK( I+1,S ) ) / BNRM2
            type*,'iter,resid=',iter,resid
            IF ( RESID.LE.TOL ) THEN
               CALL UPDATE( I, N, X, Hmat, WORK( 1,Y ), WORK( 1,S ),
     &            WORK( 1,V ))
               GO TO 70
            ENDIF
            IF ( ITER.EQ.MAXIT ) GO TO 50
            IF ( I.LT.RESTRT )   GO TO 30
*
   50    CONTINUE
*
*        Compute current solution vector X.
*
         CALL UPDATE( RESTRT, N, X, Hmat, WORK( 1,Y ), WORK( 1,S ),
     &                WORK( 1,V ) )
*
*        Compute residual vector R, find norm, then check for tolerance.
*        (AV is temporary workspace here.)
*
         CALL ZCOPY( N, B, 1, WORK( 1,AV ), 1 )
		 
*         CALL ZGEMV( 'N', N, N, -C1, A, N, 
*     &        X, 1, C1, WORK(1,AV), 1 )
*         call GEMVmask(Nl,Np,om,pvec,dh,den0,alf0,bet0,
*     &        den,alf,bet,.true.,X, WORK(1,AV))
      call AtimesV(N,Nl,Np,Np4,
     &            M,Bs,w,L,Xc,alpha0,beta0,pvec,etamat,netamat,
     &            rtEa,rtEb,Tpp,Tps,Tsp,Tss,
     &            R11,R12,R21,R22,S11,S12,S21,S22,
     &            U11,U12,U21,U22,T11,T12,T21,T22,
     &            P1,P2,Q1,Q2,SY,PSY,ea,eb,
     &            .true.,X,WORK(1,AV)) 
			

         CALL ZCOPY( N, WORK(1,AV), 1, WORK( 1,R ), 1 )
		 
         WORK( I+1,S ) = DZNRM2( N, WORK( 1,R ), 1 )
         RESID = WORK( I+1,S ) / BNRM2
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
*
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

********************************************************
      subroutine AtimesV(N,Nl,Np,Np4,
     &            M,Bs,w,L,Xc,alpha0,beta0,pvec,etamat,netamat,
     &            rtEa,rtEb,Tpp,Tps,Tsp,Tss,
     &            R11,R12,R21,R22,S11,S12,S21,S22,
     &            U11,U12,U21,U22,T11,T12,T21,T22,
     &            P1,P2,Q1,Q2,SY,PSY,ea,eb,getres,V,X)
***************************************************************
*     If[getres,set  X=X-(1-P*dS)V, set X=(1-P*dS)V ]
***************************************************************
      Implicit    none
      logical     getres
      INTEGER     N,Nl,Np,Np4,i
      Complex*16  ea(Np,0:Nl),eb(Np,0:Nl)
      Complex*16  SY(N),PSY(N),V(N),X(N)
 
      Complex*16  T11(Np,0:Nl),T12(Np,0:Nl)
      Complex*16  T21(Np,0:Nl),T22(Np,0:Nl)
      Complex*16  U11(Np,0:Nl),U12(Np,0:Nl)
      Complex*16  U21(Np,0:Nl),U22(Np,0:Nl)
      Complex*16  R11(Np,0:Nl),R12(Np,0:Nl)
      Complex*16  R21(Np,0:Nl),R22(Np,0:Nl)
      Complex*16  S11(Np,0:Nl),S12(Np,0:Nl)
      Complex*16  S21(Np,0:Nl),S22(Np,0:Nl)
      Complex*16  P1(Np,0:Nl),P2(Np,0:Nl)
      Complex*16  Q1(Np,0:Nl),Q2(Np,0:Nl)
***********************************************************
      Integer     M,Bs
      Real*8      w,L,Xc(Nl)
      Complex*16  alpha0(0:Nl),beta0(0:Nl),pvec(Np)
      Complex*16  etamat(Np,0:Nl),netamat(Np,0:Nl)
      Complex*16  rtEa(Np,0:Nl),rtEb(Np,0:Nl)
      Complex*16  Tpp(Bs,Nl),Tps(Bs,Nl),Tsp(Bs,Nl),Tss(Bs,Nl)
	 
***********************************************************
	 
	 
      Integer     il,ioff
      Complex*16  alf,bet

*      type*,'entered AtimesV'
      if(getres)then
*       X=X-V        
        call ZAXPY(N,(-1.D0,0.D0),V,1,X,1)        
      else
*       X=V        
        call ZCOPY(N,V,1,X,1)
      Endif
 
*     SY=S*V
      alf=(1.D0,0.D0)
      bet=(0.D0,0.D0)

*      Do i=1,N
*        type*,'V(i)=',i,V(i)
*      enddo

*     set SY
      DO 99999 IL=1,NL
        ioff=Np4*(il-1)
		call PwCylScat(Np,M,Bs,w,L,alpha0(il),beta0(il),pvec,etamat(1,il),
     &           netamat(1,il),rtEa(1,il),rtEb(1,il),Tpp(1,il),Tps(1,il),
     &           Tsp(1,il),Tss(1,il),Xc(il),V(ioff+1),SY(ioff+1))
	 
*        call ZGEMV('N',Np4,Np4,alf,dS(1,1,il),
*     &        Np4,V(ioff+1),1,bet,SY(ioff+1),1)
99999 continue


*      Do i=1,N
*        type*,'SY(i)=',i,SY(i)
*      enddo

      call setPSY(Nl,N,Np,ea,eb,
     &           R11,R12,R21,R22,S11,S12,S21,S22,
     &           U11,U12,U21,U22,T11,T12,T21,T22,
     &           P1,P2,Q1,Q2,SY,PSY)

*      Do i=1,N
*        type*,'PSY(i)=',i,PSY(i)
*      enddo
 
      if(getres)then
*       X=X+PSY
        call ZAXPY(N,(1.D0,0.D0),PSY,1,X,1)
      else
*       X=X-PSY
        call ZAXPY(N,(-1.D0,0.D0),PSY,1,X,1)
      endif

*      Do i=1,N
*        type*,'X(i)=',i,X(i)
*      enddo
*      pause 'leaving AtimesV'
 
      return
      end

      Subroutine PwCylScat(Np,M,Bs,w,L,alpha0,beta0,pvec,etavec,
     &           netavec,rtEa,rtEb,Tpp,Tps,Tsp,Tss,Xc,Y,SY)
      Implicit none
      Integer    M,Bs,Np
      Real*8     Xc,L,PPi,dk,kmin
      parameter  (PPi=3.14159265358979324D0)

      Complex*16 Y(Np,4),SY(Np,4)
      Complex*16 w,dp,alpha0,beta0,Pvec(Np)
      Complex*16 etavec(Np),netavec(np),rtea(np),rteb(np)
      Complex*16 Tpp(Bs),Tps(Bs),Tsp(Bs),Tss(Bs)
      
      Integer    Mmax,il,ipi,ipo,mi,mo
	  Parameter  (Mmax=2)
	  Complex*16 Ytemp(Mmax,2),SYtemp(Mmax,2)

      Complex*16 fac,facp,facs,pi,po
      Complex*16 etai,etao,netai,netao,dEx,Ex0,Exi,Exo
      Complex*16 eadi,ebdi,eaui,ebui
      Complex*16 eado,ebdo,eauo,ebuo
      Complex*16 Ci,C0,sump,sums,sumPU,sumSU,sumPD,sumSD
      Complex*16 rtetai,rtnetai,rtetao,rtnetao      
      Complex*16 phidai,phiuai,phidao,phiuao    
      Complex*16 phidbi,phiubi,phidbo,phiubo    
	  
	  If(M.gt. Mmax) Stop "M.gt.Mmax"


      Ci=(0.D0,1.D0)
      C0=(0.D0,0.D0)
      
       
      dk=2.D0*PPi/L
      kmin=Dble(w*pvec(1))
      fac=CDsqrt(2.D0/(w*L))
      
      
      il=0
      Do 1000 mi=-M,M
      il=il+1
      sump=C0
      sums=C0

      dEx=CDexp(Ci*dk*Xc)
      Exi=CDexp(Ci*kmin*Xc)
      Do 100 ipi=1,Np
        pi=pvec(ipi)
        etai=etavec(ipi)
        netai=netavec(ipi)
        rtetai=CDsqrt(etai)
        rtnetai=CDsqrt(netai)
        eadi=rtea(ipi)
        ebdi=rteb(ipi)
        eaui=rtea(ipi)
        ebui=rteb(ipi)
        facp=fac*Exi/(rtetai)
        facs=fac*Exi/(rtnetai)

        phidai=facp*eadi*(alpha0*(pi+Ci*etai))**mi
        phiuai=facp*eaui*(alpha0*(pi-Ci*etai))**mi
        sump=sump+phidai*Y(ipi,1)+phiuai*Y(ipi,3)
        
        phidbi=facs*ebdi*(beta0*(pi+Ci*netai))**mi
        phiubi=facs*ebui*(beta0*(pi-Ci*netai))**mi
        sums=sums+phidbi*Y(ipi,2)+phiubi*Y(ipi,4)
        Exi=dEx*Exi
100	  Continue
      Ytemp(il,1)=sump
      Ytemp(il,2)=sums

1000  Continue

      Do 1100 il=1,M
        SYtemp(il,1) = Tpp(il)*Y(il,1) + Tps(il)*Y(il,2)
        SYtemp(il,2) = Tsp(il)*Y(il,1) + Tss(il)*Y(il,2)
1100  Continue

      dEx=CDexp(-Ci*dk*Xc)
      Exo=CDexp(-Ci*kmin*Xc)
      Do 999  ipo=1,Np
	  po=pvec(ipo)
	  etao=etavec(ipo)
	  netao=netavec(ipo)
	  rtetao=CDsqrt(etao)
	  rtnetao=CDsqrt(netao)
	  
	  eado=rtea(ipo)
	  ebdo=rteb(ipo)
	  eauo=rtea(ipo)
	  ebuo=rteb(ipo)
	  facp=fac*Exo/(rtetao)
	  facs=fac*Exo/(rtnetao)
          
      sumPU=C0
      sumSU=C0
      sumPD=C0
      sumSD=C0
      il=0
	  Do 993 mo=-M,M
	    il=il+1
		phidao=facp*eado*(alpha0*(po+Ci*etao))**mo
		SumPU=SumPU+phidao*SYtemp(il,1)
		
		phidbo=facs*ebdo*(beta0*(po+Ci*netao))**mo
		SumSU=SumSU+phidbo*SYtemp(il,2)
		
		phiuao=facp*eauo*(alpha0*(po-Ci*etao))**mo
		SumPD=SumPD+phiuao*SYtemp(il,1)
		
		phiubo=facs*ebuo*(beta0*(po-Ci*netao))**mo
		SumSD=SumSD+phiubo*SYtemp(il,2)

993	  Continue
      SY(ipo,1)=SumPU
      SY(ipo,2)=SumSU
      SY(ipo,3)=SumPD
      SY(ipo,4)=SumSD
      Exo=dEx*Exo
999   Continue

      return
      end 

      subroutine setResp(Nl,Np,ea,eb,T11,T12,T21,T22,X,RespX,RespZ)
***************************************************************
*     Resp=P*dS*X
***************************************************************
      Implicit    none
      Integer     Nl,Np,ip,ip1,ip2,Np2
      complex*16  ea(Np,0:Nl),eb(Np,0:Nl)
      Complex*16  X(*),RespX(Np),RespZ(Np)
 
      Complex*16  T11(Np,0:Nl),T12(Np,0:Nl)
      Complex*16  T21(Np,0:Nl),T22(Np,0:Nl)

***********************************************************
*     Resp=T(0)*X(Np2+ip)
***********************************************************
      Np2=2*Np
      Do 99999 ip=1,Np
        ip1=Np2+ip
        ip2=ip1+Np
*       Resp=T(0)*X
        Respx(ip)=T11(ip,0)*X(ip1)+T12(ip,0)*X(ip2)
        Respz(ip)=T21(ip,0)*X(ip1)+T22(ip,0)*X(ip2)
*        type*,'ip,RespX(ip)=',ip,RespX(ip)
*        type*,'ip,RespZ(ip)=',ip,RespZ(ip)
99999 Continue
      return
      end

      SUBROUTINE setPSY(Nl,N,Np,ea,eb,
     &           R11,R12,R21,R22,S11,S12,S21,S22,
     &           U11,U12,U21,U22,T11,T12,T21,T22,
     &           P1,P2,Q1,Q2,SY,X)
	
*     set X=P*SY
      Implicit none
      INTEGER     Nl,N,Np
      complex*16  ea(Np,0:Nl),eb(Np,0:Nl)
      Complex*16  SY(N),X(N)
 
      Complex*16  T11(Np,0:Nl),T12(Np,0:Nl)
      Complex*16  T21(Np,0:Nl),T22(Np,0:Nl)
      Complex*16  U11(Np,0:Nl),U12(Np,0:Nl)
      Complex*16  U21(Np,0:Nl),U22(Np,0:Nl)
      Complex*16  R11(Np,0:Nl),R12(Np,0:Nl)
      Complex*16  R21(Np,0:Nl),R22(Np,0:Nl)
      Complex*16  S11(Np,0:Nl),S12(Np,0:Nl)
      Complex*16  S21(Np,0:Nl),S22(Np,0:Nl) 
      Complex*16  P1(Np,0:Nl),P2(Np,0:Nl)
      Complex*16  Q1(Np,0:Nl),Q2(Np,0:Nl)
      Integer     Np2,Np3,Np6,Np7
****************************************************************
*     LOCAL variables
*****************************************************************
      Integer    ioff4,ioff8,IL,ip
*      type*,'entered setPSY'


      call  UpSweep(Np,Nl,SY,ea,eb,T11,T12,T21,
     &           T22,U11,U12,U21,U22,R11,R12,R21,R22,
     &           S11,S12,S21,S22,P1,P2,Q1,Q2)

*        Do ip=1,Np
*      Do il=0,Nl
*          type*,'il,ip,P1(ip,il),P2(ip,il)=',ip,il,P1(ip,il),P2(ip,il)
*        Enddo
*      Enddo

*        Do ip=1,Np
*      Do il=0,Nl
*          type*,'il,ip,Q1(ip,il),Q2(ip,il)=',ip,il,Q1(ip,il),Q2(ip,il)
*        Enddo
*      Enddo
 
      call DownSweep(Np,NL,R11,R12,R21,R22,
     &     S11,S12,S21,S22,P1,P2,Q1,Q2,ea,eb,X)
 
      RETURN
      END

      Subroutine Factor(Nl,Np,T11,T12,T21,T22,U11,U12,U21,U22,
     &           R11,R12,R21,R22,S11,S12,S21,S22,Ea,Eb,Rd11,Rd12,
     &           Rd21,Rd22,Tu11,Tu12,Tu21,Tu22,Ru11,Ru12,Ru21,
     &           Ru22,Td11,Td12,Td21,Td22)
      Implicit    None
      integer     Np,ip,il,nl,ilp1,nlm1
      Complex*16  Ea(Np,0:Nl),Eb(Np,0:Nl)
      Complex*16  T11(Np,0:Nl),T12(Np,0:Nl)
      Complex*16  T21(Np,0:Nl),T22(Np,0:Nl)
      Complex*16  U11(Np,0:Nl),U12(Np,0:Nl)
      Complex*16  U21(Np,0:Nl),U22(Np,0:Nl)
      Complex*16  R11(Np,0:Nl),R12(Np,0:Nl)
      Complex*16  R21(Np,0:Nl),R22(Np,0:Nl)
      Complex*16  S11(Np,0:Nl),S12(Np,0:Nl)
      Complex*16  S21(Np,0:Nl),S22(Np,0:Nl)
 
      Complex*16  Rd11(Np,0:Nl),Rd12(Np,0:Nl)
      Complex*16  Rd21(Np,0:Nl),Rd22(Np,0:Nl)
      Complex*16  Tu11(Np,0:Nl),Tu12(Np,0:Nl)
      Complex*16  Tu21(Np,0:Nl),Tu22(Np,0:Nl)
      Complex*16  Td11(Np,0:Nl),Td12(Np,0:Nl)
      Complex*16  Td21(Np,0:Nl),Td22(Np,0:Nl)
      Complex*16  Ru11(Np,-1:Nl),Ru12(Np,-1:Nl)
      Complex*16  Ru21(Np,-1:Nl),Ru22(Np,-1:Nl)
      Complex*16  W11,W12,W21,W22
      Complex*16  V11,V12,V21,V22
      Complex*16  RRu11,RRu12,RRu21,RRu22
      Complex*16  MT11,MT12,MT21,MT22
      Complex*16  MB11,MB12,MB21,MB22
      Complex*16  C1,InvDet
 
      C1=(1.0,0.0)
 
*     R=Rd(nl)
      DO 99999 IP=1,NP
        R11(ip,nl)=Rd11(ip,nl)
        R12(ip,nl)=Rd12(ip,nl)
        R21(ip,nl)=Rd21(ip,nl)
        R22(ip,nl)=Rd22(ip,nl)
 
 
*       S=Td(nl)
        S11(ip,nl)=Td11(ip,nl)
        S12(ip,nl)=Td12(ip,nl)
        S21(ip,nl)=Td21(ip,nl)
        S22(ip,nl)=Td22(ip,nl)
99999 continue
*      pause'check point 1'
 
      nlm1=nl-1
      DO 99998 IL=NLM1,0,-1
        ilp1=il+1

*      pause'check point 2'
 
      DO 99997 IP=1,NP
*      pause'check point 3'

*       MT=E(ilp1)*R(ilp1)*E(ilp1)
        MT11=Ea(ip,ilp1)*R11(ip,ilp1)*Ea(ip,ilp1)
        MT12=Ea(ip,ilp1)*R12(ip,ilp1)*Eb(ip,ilp1)
        MT21=Eb(ip,ilp1)*R21(ip,ilp1)*Ea(ip,ilp1)
        MT22=Eb(ip,ilp1)*R22(ip,ilp1)*Eb(ip,ilp1)
 
*       W=(I-Ru*MT)^(-1)
	    W11=C1-Ru21(ip,il)*MT12-Ru22(ip,il)*MT22
	    W12=Ru11(ip,il)*MT12+Ru12(ip,il)*MT22
	    W21=Ru21(ip,il)*MT11+Ru22(ip,il)*MT21
	    W22=C1-Ru11(ip,il)*MT11-Ru12(ip,il)*MT21
	    InvDet=C1/(W11*W22-W12*W21)


	    W11=W11*InvDet
	    W12=W12*InvDet
	    W21=W21*InvDet
	    W22=W22*InvDet

*       U=((I-Ru*MT)^(-1))*Ru
	    U11(ip,il)=W11*Ru11(ip,il)+W12*Ru21(ip,il)
	    U12(ip,il)=W11*Ru12(ip,il)+W12*Ru22(ip,il)
	    U21(ip,il)=W21*Ru11(ip,il)+W22*Ru21(ip,il)
	    U22(ip,il)=W21*Ru12(ip,il)+W22*Ru22(ip,il)
		
		
*       S=((I-Ru*MT)^(-1))*Td
		S11(ip,il)=W11*Td11(ip,il)+W12*Td21(ip,il)
		S12(ip,il)=W11*Td12(ip,il)+W12*Td22(ip,il)
		S21(ip,il)=W21*Td11(ip,il)+W22*Td21(ip,il)
		S22(ip,il)=W21*Td12(ip,il)+W22*Td22(ip,il)

*         W=MT*S
        W11=MT11*S11(ip,il)+MT12*S21(ip,il)
		W12=MT11*S12(ip,il)+MT12*S22(ip,il)
		W21=MT21*S11(ip,il)+MT22*S21(ip,il)
		W22=MT21*S12(ip,il)+MT22*S22(ip,il)
 
 
*         R=Rd+Tu*W
        R11(ip,il)=Rd11(ip,il)+Tu11(ip,il)*W11+Tu12(ip,il)*W21
		R12(ip,il)=Rd12(ip,il)+Tu11(ip,il)*W12+Tu12(ip,il)*W22
		R21(ip,il)=Rd21(ip,il)+Tu21(ip,il)*W11+Tu22(ip,il)*W21
		R22(ip,il)=Rd22(ip,il)+Tu21(ip,il)*W12+Tu22(ip,il)*W22

*       W=(I-MT*Ru)^(-1)
	    W11=C1-MT21*Ru12(ip,il)-MT22*Ru22(ip,il)
	    W12=MT11*Ru12(ip,il)+MT12*Ru22(ip,il)
	    W21=MT21*Ru11(ip,il)+MT22*Ru21(ip,il)
	    W22=C1-MT11*Ru11(ip,il)-MT12*Ru21(ip,il)
	    InvDet=C1/(W11*W22-W12*W21)


	    W11=W11*InvDet
	    W12=W12*InvDet
	    W21=W21*InvDet
	    W22=W22*InvDet

*       T=Tu*(I-MT*Ru)^(-1)

        T11(ip,il)=Tu11(ip,il)*W11+Tu12(ip,il)*W21
		T12(ip,il)=Tu11(ip,il)*W12+Tu12(ip,il)*W22
		T21(ip,il)=Tu21(ip,il)*W11+Tu22(ip,il)*W21
		T22(ip,il)=Tu21(ip,il)*W12+Tu22(ip,il)*W22

99997 continue
 
99998 continue
 
      DO 99996 IP=1,NP
*      pause'check point 3'

*       set RRu
*       MB=E(0)*Ru(-1)*E(0)
        MB11=Ea(ip,0)*Ru11(ip,-1)*Ea(ip,0)
        MB12=Ea(ip,0)*Ru12(ip,-1)*Eb(ip,0)
        MB21=Eb(ip,0)*Ru21(ip,-1)*Ea(ip,0)
        MB22=Eb(ip,0)*Ru22(ip,-1)*Eb(ip,0)
 
*       T(0)=(I-Rd(0)*MB)^(-1)
	    W11=C1-Rd21(ip,0)*MB12-Rd22(ip,0)*MB22
	    W12=Rd11(ip,0)*MB12+Rd12(ip,0)*MB22
	    W21=Rd21(ip,0)*MB11+Rd22(ip,0)*MB21
	    W22=C1-Rd11(ip,0)*MB11-Rd12(ip,0)*MB21
	    InvDet=C1/(W11*W22-W12*W21)


        T11(ip,0)=W11*InvDet
        T12(ip,0)=W12*InvDet
        T21(ip,0)=W21*InvDet
        T22(ip,0)=W22*InvDet

		V11=T11(ip,0)*Tu11(ip,0)+T12(ip,0)*Tu21(ip,0)
		V12=T11(ip,0)*Tu12(ip,0)+T12(ip,0)*Tu22(ip,0)
		V21=T21(ip,0)*Tu11(ip,0)+T22(ip,0)*Tu21(ip,0)
		V22=T21(ip,0)*Tu12(ip,0)+T22(ip,0)*Tu22(ip,0)

*         W=MB*V
        W11=MB11*V11+MB12*V21
		W12=MB11*V12+MB12*V22
		W21=MB21*V11+MB22*V21
		W22=MB21*V12+MB22*V22
 
 
*         RRu=Ru+Td*W
        RRu11=Ru11(ip,0)+Td11(ip,0)*W11+Td12(ip,0)*W21
		RRu12=Ru12(ip,0)+Td11(ip,0)*W12+Td12(ip,0)*W22
		RRu21=Ru21(ip,0)+Td21(ip,0)*W11+Td22(ip,0)*W21
		RRu22=Ru22(ip,0)+Td21(ip,0)*W12+Td22(ip,0)*W22
*****************************************************************
*****************************************************************
*****************************************************************

*       MT=E(1)*R(1)*E(1)
        MT11=Ea(ip,1)*R11(ip,1)*Ea(ip,1)
        MT12=Ea(ip,1)*R12(ip,1)*Eb(ip,1)
        MT21=Eb(ip,1)*R21(ip,1)*Ea(ip,1)
        MT22=Eb(ip,1)*R22(ip,1)*Eb(ip,1)
 
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
	    U11(ip,0)=W11*RRu11+W12*RRu21
	    U12(ip,0)=W11*RRu12+W12*RRu22
	    U21(ip,0)=W21*RRu11+W22*RRu21
	    U22(ip,0)=W21*RRu12+W22*RRu22


*!!!!!!!!!!!!!!!!DDDEEELLLEEETTTEEEEE
*       T(0)=(I-R(0)*MB)^(-1)
*	    W11=C1-R21(ip,0)*MB12-R22(ip,0)*MB22
*	    W12=R11(ip,0)*MB12+R12(ip,0)*MB22
*	    W21=R21(ip,0)*MB11+R22(ip,0)*MB21
*	    W22=C1-R11(ip,0)*MB11-R12(ip,0)*MB21
*	    InvDet=C1/(W11*W22-W12*W21)


*        T11(ip,0)=W11*InvDet
*        T12(ip,0)=W12*InvDet
*        T21(ip,0)=W21*InvDet
*        T22(ip,0)=W22*InvDet
*!!!!!!!!!!!!!!!!DDDEEELLLEEETTTEEEEE

*       V=Tu(0)*E(1)

        V11=Tu11(ip,0)*Ea(ip,1)
        V12=Tu12(ip,0)*Eb(ip,1)
        V21=Tu21(ip,0)*Ea(ip,1)
        V22=Tu22(ip,0)*Eb(ip,1)

*       W=T(0)*V
        W11=T11(ip,0)*V11+T12(ip,0)*V21
        W12=T11(ip,0)*V12+T12(ip,0)*V22
        W21=T21(ip,0)*V11+T22(ip,0)*V21
        W22=T21(ip,0)*V12+T22(ip,0)*V22

*       T=E(0)*W
        T11(ip,0)=Ea(ip,0)*W11
        T12(ip,0)=Ea(ip,0)*W12
        T21(ip,0)=Eb(ip,0)*W21
        T22(ip,0)=Eb(ip,0)*W22


**************************************************

99996 continue
      return
      end
 
      Subroutine DownSweep(Np,NL,R11,R12,R21,R22,
     &           S11,S12,S21,S22,P1,P2,Q1,Q2,ea,eb,X)
      Implicit   None
      Integer     Np,NL
      complex*16  ea(Np,0:Nl),eb(Np,0:Nl)
      Complex*16  R11(Np,0:Nl),R12(Np,0:Nl)
      Complex*16  R21(Np,0:Nl),R22(Np,0:Nl)
      Complex*16  S11(Np,0:Nl),S12(Np,0:Nl)
      Complex*16  S21(Np,0:Nl),S22(Np,0:Nl)
 
      Complex*16  P1(Np,0:Nl),P2(Np,0:Nl)
      Complex*16  Q1(Np,0:Nl),Q2(Np,0:Nl)
      Complex*16  X(*),eax,ebx
      Integer     il,ilm1,i1,i2,i3,ip
      Integer     ip1,ip2,ip3,ip4,Np2,Np4,Nlm1
	  
**********************************************************
      DO 99995 IP=1,NP
        ip1=ip
        ip2=ip1+Np
        X(ip1)=Q1(ip,0)
        X(ip2)=Q2(ip,0)
99995 continue
      
      Nlm1=Nl-1
      Np2=2*Np
      Np4=4*Np

      DO 99998 IL=1,Nlm1
	    i1=(il-1)*Np4
	    i2=i1+Np
	    i3=i2+Np

        DO 99996 IP=1,NP
          ip1=i3+ip
		  ip2=ip1+Np
		  ip3=ip2+Np
		  ip4=ip3+Np
		  eaX=ea(ip,il)*X(i1+ip)
		  ebX=eb(ip,il)*X(i2+ip)
		  X(ip1)=P1(ip,il)+R11(ip,il)*eaX+R12(ip,il)*ebX
		  X(ip2)=P2(ip,il)+R21(ip,il)*eaX+R22(ip,il)*ebX
		  X(ip3)=Q1(ip,il)+S11(ip,il)*eaX+S12(ip,il)*ebX
		  X(ip4)=Q2(ip,il)+S21(ip,il)*eaX+S22(ip,il)*ebX

99996 continue
99998 continue

****************************************
*     Bottom interface
****************************************
	  i1=Nlm1*Np4
	  i2=i1+Np
	  i3=i2+Np
	  DO 99999 IP=1,NP
        ip1=i3+ip
		ip2=ip1+Np
		eaX=ea(ip,Nl)*X(i1+ip)
		ebX=eb(ip,Nl)*X(i2+ip)
		X(ip1)=P1(ip,Nl)+R11(ip,Nl)*eaX+R12(ip,Nl)*ebX
		X(ip2)=P2(ip,Nl)+R21(ip,Nl)*eaX+R22(ip,Nl)*ebX
99999 continue
      return
      end
	
	
      Subroutine UpSweep(Np,Nl,SY,ea,eb,T11,T12,T21,
     &           T22,U11,U12,U21,U22,R11,R12,R21,R22,
     &           S11,S12,S21,S22,P1,P2,Q1,Q2)
      Implicit None
      Integer     NL,Np,Np4,Np2,ip,NLM1,il,ilp1
      Complex*16  SY(*)
	  complex*16  ea(Np,0:nl),eb(Np,0:nl)
 
      Complex*16  T11(Np,0:Nl),T12(Np,0:Nl)
      Complex*16  T21(Np,0:Nl),T22(Np,0:Nl)
      Complex*16  U11(Np,0:Nl),U12(Np,0:Nl)
      Complex*16  U21(Np,0:Nl),U22(Np,0:Nl)

      Complex*16  R11(Np,0:Nl),R12(Np,0:Nl)
      Complex*16  R21(Np,0:Nl),R22(Np,0:Nl)
      Complex*16  S11(Np,0:Nl),S12(Np,0:Nl)
      Complex*16  S21(Np,0:Nl),S22(Np,0:Nl)
 
      Complex*16  P1(Np,0:Nl),P2(Np,0:Nl)
      Complex*16  Q1(Np,0:Nl),Q2(Np,0:Nl)
 
 
      Complex*16 Y11,Y21
      Complex*16 C0
      logical    scat
      Integer    ioff,ip1,ip2,ip3,ip4,i
 
      C0=(0.D0,0.D0)
      Np4=4*Np
      Np2=2*Np


***********************************************
*     set P and Q	
***********************************************
      ioff=Np4*(Nl-1)+Np2
      DO 99999 IP=1,NP
*       P=R(NL)*SY(NL)
        P1(ip,Nl)=R11(ip,Nl)*SY(ioff+ip)+R12(ip,Nl)*SY(ioff+Np+ip)
        P2(ip,Nl)=R21(ip,Nl)*SY(ioff+ip)+R22(ip,Nl)*SY(ioff+Np+ip)
*        type*,'ip,Nl,P1(ip,Nl),P2(ip,Nl)=',ip,Nl,P1(ip,Nl),P2(ip,Nl)

        Q1(ip,Nl)=C0
        Q2(ip,Nl)=C0
99999 continue

      nlm1=Nl-1
      DO 99998 IL=NLM1,1,-1
	    ilp1=il+1
        ioff=Np4*(il-1)+Np2
        DO 99997 IP=1,NP
          ip1=ioff+ip
          ip2=ip1+Np
          ip3=ip2+Np
          ip4=ip3+Np

*         Y=SY(ip3)+E(ilp1)*P(ilp1)
          Y11=SY(ip3)+ea(ip,ilp1)*P1(ip,ilp1)
          Y21=SY(ip4)+eb(ip,ilp1)*P2(ip,ilp1)

*         P=R*SY(ip1)+T*Y
          P1(ip,il)=R11(ip,il)*SY(ip1)+R12(ip,il)*SY(ip2)+
     &              T11(ip,il)*Y11+T12(ip,il)*Y21
          P2(ip,il)=R21(ip,il)*SY(ip1)+R22(ip,il)*SY(ip2)+
     &              T21(ip,il)*Y11+T22(ip,il)*Y21
*        type*,'ip,il,P1(ip,il),P2(ip,il)=',ip,il,P1(ip,il),P2(ip,il)

*         Q=S*SY(ip1)+U*Y
          Q1(ip,il)=S11(ip,il)*SY(ip1)+S12(ip,il)*SY(ip2)+
     &              U11(ip,il)*Y11+U12(ip,il)*Y21
          Q2(ip,il)=S21(ip,il)*SY(ip1)+S22(ip,il)*SY(ip2)+
     &              U21(ip,il)*Y11+U22(ip,il)*Y21
*        type*,'ip,il,Q1(ip,il),Q2(ip,il)=',ip,il,Q1(ip,il),Q2(ip,il)
 
99997 continue
99998 continue

      DO 99996 IP=1,NP
          ip1=ip
          ip2=ip+Np

*         Y=SY(ip1)+E(1)*P(1)
          Y11=SY(ip1)+ea(ip,1)*P1(ip,1)
          Y21=SY(ip2)+eb(ip,1)*P2(ip,1)
*        type*,'ip,il,SY(ip1),SY(ip2)=',ip,0,SY(ip1),SY(ip2)

*         Q(0)=U*Y
          Q1(ip,0)=U11(ip,0)*Y11+U12(ip,0)*Y21
          Q2(ip,0)=U21(ip,0)*Y11+U22(ip,0)*Y21

*        type*,'ip,il,Q1(ip,il),Q2(ip,il)=',ip,0,Q1(ip,0),Q2(ip,0)
 
99996 continue
      return

      end


      Subroutine setRefBasis(reflex,is,ir,Nl,Nl1,N,Np,Zs,Xs,
     &           Zr,Xr,rho,alpha,beta,pvec,w,h,etamat,
     &           netamat,R11,R12,R21,R22,S11,S12,S21,S22,
     &           U11,U12,U21,U22,T11,T12,T21,T22,
     &           Rd11,Rd12,Rd21,Rd22,Td11,Td12,
     &           Td21,Td22,Tu11,Tu12,Tu21,Tu22,
     &           Ru11,Ru12,Ru21,Ru22,
     &           P1,P2,Q1,Q2,ea,eb,
     &           SigPu,SigPd,SigSu,SigSd,
     &           RecPu,RecPd,RecSu,RecSd,
     &           RespX0,RespZ0,B,BB)
	
*     set the reference basis elements B and BB
      Implicit none
      logical     reflex
      INTEGER     is,ir,Nl,Nl1,Nl2,N,Np
      Real*8      h(0:Nl),rho,Zs,Xs,Zr,Xr
      Complex*16  w,pvec(Np)
      Complex*16  B(N),BB(N),alpha,beta
 
      Complex*16  T11(Np,0:Nl),T12(Np,0:Nl)
      Complex*16  T21(Np,0:Nl),T22(Np,0:Nl)
      Complex*16  U11(Np,0:Nl),U12(Np,0:Nl)
      Complex*16  U21(Np,0:Nl),U22(Np,0:Nl)
      Complex*16  R11(Np,0:Nl),R12(Np,0:Nl)
      Complex*16  R21(Np,0:Nl),R22(Np,0:Nl)
      Complex*16  S11(Np,0:Nl),S12(Np,0:Nl)
      Complex*16  S21(Np,0:Nl),S22(Np,0:Nl)
 
      Complex*16  P1(Np,0:Nl),P2(Np,0:Nl)
      Complex*16  Q1(Np,0:Nl),Q2(Np,0:Nl)
      Complex*16  Rd11(Np,0:Nl),Rd12(Np,0:Nl)
      Complex*16  Rd21(Np,0:Nl),Rd22(Np,0:Nl)
      Complex*16  Td11(Np,0:Nl),Td12(Np,0:Nl)
      Complex*16  Td21(Np,0:Nl),Td22(Np,0:Nl)
      Complex*16  Tu11(Np,0:Nl),Tu12(Np,0:Nl)
      Complex*16  Tu21(Np,0:Nl),Tu22(Np,0:Nl)
      Complex*16  Ru11(Np,-1:Nl),Ru12(Np,-1:Nl)
      Complex*16  Ru21(Np,-1:Nl),Ru22(Np,-1:Nl)
      Complex*16  etamat(Np,0:Nl1),netamat(Np,0:Nl1)
      Complex*16  RespX0(Np),RespZ0(Np),Ea(Np,0:Nl),Eb(Np,0:Nl)
      Complex*16  B11,B12,B21,B22,Temp,C1,C0,Ci
      Integer     ioff,nlm1,ilp1,il,i1,i2,ilm1,np2,np3,np6,np7
      Integer     Np4,ip,ip1,ip2,ip3,ip4,ioff4,ioff8
******************************************************************
*     Source\Receiver data
      Complex*16  SigPu(Np),SigSu(Np),SigPd(Np),SigSd(Np)
      Complex*16  RecPu(Np),RecSu(Np),RecPd(Np),RecSd(Np)
      Complex*16  Eas0,Ebs0,Easf,Ebsf,Ear0,Ebr0,Earf,Ebrf
      Complex*16  Ndi11,Ndi21,etai,netai
      Complex*16  pi,twomup,epsa,epsb,Ciwh,Ciwx,Ciwz,Ciwhmz
      Complex*16  Ex,bsq,rbsq
******************************************************************
 
 
      Complex*16 Y11,Y21,MB11,MB12,MB21,MB22,W11,W12,W21,W22
      Complex*16 InvDet,Sig11,Sig21,V11,V21,Rec11,Rec21
 
      C0=(0.D0,0.D0)
      C1=(1.D0,0.D0)
      Ci=(0.D0,1.D0)

      Np4=4*Np
      Np2=2*Np

***********************************************
*     set SigU and SigD	
***********************************************
      Ciwh=CI*w*h(0)
      Ciwx=CI*w*Xs
      Ciwz=CI*w*Zs
      Ciwhmz=CI*w*(h(0)-Zs)
      bsq=beta*beta
      rbsq=rho*bsq
 
      DO   10000 IP=1,Np
        Temp=Ciwh*etamat(ip,0)
	    Temp=Ciwh*netamat(ip,0)
		Temp=CIwz*etamat(ip,0)
		EAsf=CDEXP(Temp)
		Temp=CIwz*netamat(ip,0)
		EBsf=CDEXP(Temp)		
		
		Temp=Ciwhmz*etamat(ip,0)
		EAs0=CDEXP(Temp)
		Temp=Ciwhmz*netamat(ip,0)
		EBs0=CDEXP(Temp)	
 
        TEMP=-CIwx*Pvec(ip)
        Ex=CDEXP(Temp)
 
*       Double couple
        if(is.eq.1) then
          pi=Pvec(ip)
		  etai=etamat(ip,0)
		  netai=netamat(ip,0)
		  
		  twomup=2.D0*rbsq*pi
		  epsa=C1/CDsqrt(2.D0*rho*etai)
		  epsb=C1/CDsqrt(2.D0*rho*netai)
		  
		  Ndi11=Ci*twomup*etai*epsa
		  Ndi21=-Ci*(twomup*pi-rho)*epsa
		  
		  SigPu(ip)=-Ci*EAsf*Ex*Ndi11/(rbsq)
		  SigSu(ip)=-Ci*EBsf*Ex*Ndi21/(rbsq)
		  SigPd(ip)=Ci*EAs0*Ex*Ndi11/(rbsq)
		  SigSd(ip)=-Ci*EBs0*Ex*Ndi21/(rbsq)
 
        elseif(is.eq.2) then
**************************************************	
 
*         POINT EXPLOSION SOURCE TERMS
          epsa=Ex/cdsqrt(2.0*rho*etamat(ip,0))
		  SigPu(ip)=-Ci*EAsf*epsa/(alpha*alpha)
		  SigSu(ip)=C0
		  SigPd(ip)=-Ci*EAs0*epsa/(alpha*alpha)
		  SigSd(ip)=C0
        else 
*         PLANE WAVE SOURCE down P
*	        SigPu(ip)=Easf
	        SigPu(ip)=C0
			SigSu(ip)=C0
			SigPd(ip)=EAs0
			SigSd(ip)=C0
        Endif
10000 continue

*     Init P and Q
      Do il=0,Nl
	    Do ip=1,Np
		  Q1(ip,il)=C0
		  Q2(ip,il)=C0
		  P1(ip,il)=C0
		  P2(ip,il)=C0
		Enddo
	  Enddo

      if(reflex) then
***********************************************
*    set  D(0)	for B
***********************************************
        Do 99992 ip=1,Np
*         MB=E(0)*Ru(-1)*E(0)
          MB11=Ea(ip,0)*Ru11(ip,-1)*Ea(ip,0)
          MB12=Ea(ip,0)*Ru12(ip,-1)*Eb(ip,0)
          MB21=Eb(ip,0)*Ru21(ip,-1)*Ea(ip,0)
          MB22=Eb(ip,0)*Ru22(ip,-1)*Eb(ip,0)
*          type*,'MB11,MB12,MB21,MB22=',MB11,MB12,MB21,MB22

*         W=(I-MB*R)^(-1)
	      W11=C1-MB21*R12(ip,0)-MB22*R22(ip,0)
		  W12=MB11*R12(ip,0)+MB12*R22(ip,0)
		  W21=MB21*R11(ip,0)+MB22*R21(ip,0)
		  W22=C1-MB11*R11(ip,0)-MB12*R21(ip,0)
		  InvDet=C1/(W11*W22-W12*W21)
		  
		  
		  W11=W11*InvDet
		  W12=W12*InvDet
		  W21=W21*InvDet
		  W22=W22*InvDet


*          type*,'W11,W12,W21,W22=',W11,W12,W21,W22

          Sig11=Ru11(ip,-1)*SigPU(ip)+Ru12(ip,-1)*SigSU(ip)
          Sig21=Ru21(ip,-1)*SigPU(ip)+Ru22(ip,-1)*SigSU(ip)
          Sig11=SigPD(ip)+ea(ip,0)*Sig11
          Sig21=SigSD(ip)+eb(ip,0)*Sig21

*          type*,'Sig11,Sig21=',Sig11,Sig21

		  V11=W11*Sig11+W12*Sig21
		  V21=W21*Sig11+W22*Sig21
		  
*         D(0)=S(0)*V
		  Q1(ip,0)=S11(ip,0)*V11+S12(ip,0)*V21
		  Q2(ip,0)=S21(ip,0)*V11+S22(ip,0)*V21

99992 continue

*     ******************************************************
      else  ! upward transmission geometry
*     ******************************************************
      ioff=Np4*(Nl-1)+Np2
      DO 99999 IP=1,NP
*       P=Tu(NL)*Sig
        P1(ip,Nl)=Tu11(ip,Nl)
        P2(ip,Nl)=Tu21(ip,Nl)
*        type*,'ip,Nl,P1(ip,Nl),P2(ip,Nl)=',ip,Nl,P1(ip,Nl),P2(ip,Nl)

99999 continue

      nlm1=Nl-1
      DO 99998 IL=NLM1,1,-1
	    ilp1=il+1
        ioff=Np4*(il-1)+Np2
        DO 99997 IP=1,NP
          ip1=ioff+ip
          ip2=ip1+Np
          ip3=ip2+Np
          ip4=ip3+Np

*         Y=E(ilp1)*P(ilp1)
          Y11=ea(ip,ilp1)*P1(ip,ilp1)
          Y21=eb(ip,ilp1)*P2(ip,ilp1)

*         P=T*Y
          P1(ip,il)=T11(ip,il)*Y11+T12(ip,il)*Y21
          P2(ip,il)=T21(ip,il)*Y11+T22(ip,il)*Y21
*          type*,'ip,il,P1(ip,il),P2(ip,il)=',ip,il,P1(ip,il),P2(ip,il)

*         Q=T*Y
          Q1(ip,il)=U11(ip,il)*Y11+U12(ip,il)*Y21
          Q2(ip,il)=U21(ip,il)*Y11+U22(ip,il)*Y21
*          type*,'ip,il,Q1(ip,il),Q2(ip,il)=',ip,il,Q1(ip,il),Q2(ip,il)
 
99997 continue
99998 continue

      DO 99996 IP=1,NP
          ip1=ip
          ip2=ip+Np

*         Y=E(1)*P(1)
          Y11=ea(ip,1)*P1(ip,1)
          Y21=eb(ip,1)*P2(ip,1)

*         Q(0)=U*Y
          Q1(ip,0)=U11(ip,0)*Y11+U12(ip,0)*Y21
          Q2(ip,0)=U21(ip,0)*Y11+U22(ip,0)*Y21

*          type*,'ip,il,Q1(ip,il),Q2(ip,il)=',ip,0,Q1(ip,0),Q2(ip,0)
 
99996 continue


      endif

      call DownSweep(Np,NL,R11,R12,R21,R22,
     &     S11,S12,S21,S22,P1,P2,Q1,Q2,ea,eb,B)

*     set the reciprocal basis element BB
***********************************************
*     set RecU and RecD	
***********************************************
*     Init P and Q
      Do il=0,Nl
	    Do ip=1,Np
		  Q1(ip,il)=C0
		  Q2(ip,il)=C0
		  P1(ip,il)=C0
		  P2(ip,il)=C0
		Enddo
	  Enddo

      Ciwh=CI*w*h(0)
      Ciwx=CI*w*Xr
      Ciwz=CI*w*Zr
      Ciwhmz=CI*w*(h(0)-Zr)
 
      DO IP=1,Np
        Temp=Ciwh*etamat(ip,0)
	    Temp=Ciwh*netamat(ip,0)
		Temp=CIwz*etamat(ip,0)
		EArf=CDEXP(Temp)
		Temp=CIwz*netamat(ip,0)
		EBrf=CDEXP(Temp)		
		
		Temp=Ciwhmz*etamat(ip,0)
		EAr0=CDEXP(Temp)
		Temp=Ciwhmz*netamat(ip,0)
		EBr0=CDEXP(Temp)	
 
        TEMP=-CIwx*Pvec(ip)
        Ex=CDEXP(Temp)
		
		pi=Pvec(ip)
		etai=etamat(ip,0)
		netai=netamat(ip,0)
		
		epsa=C1/CDsqrt(2.D0*rho*etai)
		epsb=C1/CDsqrt(2.D0*rho*netai)
 
*       Z component response
        if(ir.eq.1) then
		  
		  RecPu(ip)=Ci*etai*epsa*EArf*Ex
		  SigSu(ip)=-Ci*pi*epsb*EBrf*Ex

		  RecPd(ip)=Ci*etai*epsa*EAr0*Ex
		  RecSd(ip)=-Ci*pi*epsb*EBr0*Ex
		else
*       X component response

		  RecPu(ip)=-Ci*pi*epsa*EArf*Ex
		  SigSu(ip)=-Ci*netai*epsb*EBrf*Ex

		  RecPd(ip)=-Ci*pi*epsa*EAr0*Ex
		  RecSd(ip)=Ci*netai*epsb*EBr0*Ex
		Endif
		  
		Enddo


        Do ip=1,Np
*         MB=E(0)*Ru(-1)*E(0)
          MB11=Ea(ip,0)*Ru11(ip,-1)*Ea(ip,0)
          MB12=Ea(ip,0)*Ru12(ip,-1)*Eb(ip,0)
          MB21=Eb(ip,0)*Ru21(ip,-1)*Ea(ip,0)
          MB22=Eb(ip,0)*Ru22(ip,-1)*Eb(ip,0)
*          type*,'MB11,MB12,MB21,MB22=',MB11,MB12,MB21,MB22

*         W=(I-MB*R)^(-1)
	      W11=C1-MB21*R12(ip,0)-MB22*R22(ip,0)
		  W12=MB11*R12(ip,0)+MB12*R22(ip,0)
		  W21=MB21*R11(ip,0)+MB22*R21(ip,0)
		  W22=C1-MB11*R11(ip,0)-MB12*R21(ip,0)
		  InvDet=C1/(W11*W22-W12*W21)
		  
		  
		  W11=W11*InvDet
		  W12=W12*InvDet
		  W21=W21*InvDet
		  W22=W22*InvDet


*          type*,'W11,W12,W21,W22=',W11,W12,W21,W22

          Rec11=Ru11(ip,-1)*RecPU(ip)+Ru12(ip,-1)*RecSU(ip)
          Rec21=Ru21(ip,-1)*RecPU(ip)+Ru22(ip,-1)*RecSU(ip)
          Rec11=RecPD(ip)+ea(ip,0)*Rec11
          Rec21=RecSD(ip)+eb(ip,0)*Rec21

*          type*,'Rec11,Rec21=',Rec11,Rec21

		  V11=W11*Rec11+W12*Rec21
		  V21=W21*Rec11+W22*Rec21
		  
*         D(0)=S(0)*V
		  Q1(ip,0)=S11(ip,0)*V11+S12(ip,0)*V21
		  Q2(ip,0)=S21(ip,0)*V11+S22(ip,0)*V21
*          type*,'ip,Q1(ip,0),Q2(ip,0)=',ip,Q1(ip,0),Q2(ip,0)

      Enddo


      call DownSweep(Np,NL,R11,R12,R21,R22,
     &     S11,S12,S21,S22,P1,P2,Q1,Q2,ea,eb,BB)

      return
      end

************************************************************
      SUBROUTINE Init(Nl,Nl1,Np,Np4,pvec,etamat,netamat,
     &           w,h,xleft,xright,rho0,alpha0,beta0,rho,
     &           alpha,beta,Rd11,Rd12,Rd21,Rd22,Td11,Td12,
     &           Td21,Td22,Tu11,Tu12,Tu21,Tu22,Ru11,Ru12,
     &           Ru21,Ru22,rtEa,rtEb,Ea,Eb,R11,R12,R21,R22,
     &           S11,S12,S21,S22,U11,U12,U21,U22,T11,T12,T21,T22)
	
*     set up interfacial scattering arrays and factorise propagator
      Implicit    none
      INTEGER     Nl,Nl1,Np,Np4
      Integer     Np2,Np3,ilp1,il,il2,ip
      Real*8      h(0:Nl),xleft(Nl),xright(Nl)
      Real*8      rho0(0:Nl1),rho(0:Nl1)
      Complex*16  alpha0(0:Nl1),beta0(0:Nl1)
      Complex*16  alpha(0:Nl1),beta(0:Nl1)
      Complex*16  w,pvec(Np)
      Complex*16  T11(Np,0:Nl),T12(Np,0:Nl)
      Complex*16  T21(Np,0:Nl),T22(Np,0:Nl)
      Complex*16  U11(Np,0:Nl),U12(Np,0:Nl)
      Complex*16  U21(Np,0:Nl),U22(Np,0:Nl)
      Complex*16  R11(Np,0:Nl),R12(Np,0:Nl)
      Complex*16  R21(Np,0:Nl),R22(Np,0:Nl)
      Complex*16  S11(Np,0:Nl),S12(Np,0:Nl)
      Complex*16  S21(Np,0:Nl),S22(Np,0:Nl)
 
      Complex*16  Rd11(Np,0:Nl),Rd12(Np,0:Nl)
      Complex*16  Rd21(Np,0:Nl),Rd22(Np,0:Nl)
      Complex*16  Tu11(Np,0:Nl),Tu12(Np,0:Nl)
      Complex*16  Tu21(Np,0:Nl),Tu22(Np,0:Nl)
      Complex*16  Td11(Np,0:Nl),Td12(Np,0:Nl)
      Complex*16  Td21(Np,0:Nl),Td22(Np,0:Nl)
      Complex*16  Ru11(Np,-1:Nl),Ru12(Np,-1:Nl)
      Complex*16  Ru21(Np,-1:Nl),Ru22(Np,-1:Nl)
      Complex*16  etamat(Np,0:Nl1),netamat(Np,0:Nl1)
      Complex*16  rtEa(Np,0:Nl),rtEb(Np,0:Nl),Ea(Np,0:Nl),Eb(Np,0:Nl)
 
      Complex*16  p,eta,neta,epsa,epsb,twomup,C0,C1,Ci
      Complex*16  Mu11,Mu12,Mu21,Mu22,Md11,Md12,Md21,Md22
      Complex*16  Wf11,Wf12,Wf21,Wf22
      integer     i,j
 
      Np2=2*Np
      Np3=3*Np
 
      C0=(0.D0,0.D0)
      C1=(1.D0,0.D0)	  	
      Ci=(0.D0,1.D0)
 
*     set the vertical slowness
      Do 53 il=0,Nl1
        call Vslo(Np,alpha0(il),Pvec,etamat(1,il))
        call Vslo(Np,beta0(il),Pvec,netamat(1,il))
   53 continue
 
      Do 50 il=0,Nl
      DO 99999 IP=1,NP
        rtEa(ip,il)=CDexp(0.5D0*Ci*w*h(il)*etamat(ip,il))
        rtEb(ip,il)=CDexp(0.5D0*Ci*w*h(il)*netamat(ip,il))
        Ea(ip,il)=rtEa(ip,il)*rtEa(ip,il)
        Eb(ip,il)=rtEb(ip,il)*rtEb(ip,il)
99999 continue
50    continue
 
      DO    75 IP=1,Np
        call Rfs(Pvec(ip),etamat(ip,0),netamat(ip,0),
     &       rho0(0),beta0(0),Ru11(ip,-1),Ru12(ip,-1),
     &       Ru21(ip,-1),Ru22(ip,-1))
*       EXCLUDE FREE SURFACE******************************
*        Ru11(ip,-1)=C0
*        Ru12(ip,-1)=C0
*        Ru21(ip,-1)=C0
*        Ru22(ip,-1)=C0
   75 continue
 
*     load reference medium interfacial scattering arrays
      DO 99998 IL=0,NL
	    ilp1=il+1
*       set the reference medium interfacial scattering matrices
        call IntScat(Np,pvec,etamat(1,il),netamat(1,il),
     &    rho0(il),beta0(il),etamat(1,ilp1),netamat(1,ilp1),
     &    rho0(ilp1),beta0(ilp1),Rd11(1,il),Rd12(1,il),
     &    Rd21(1,il),Rd22(1,il),Tu11(1,il),Tu12(1,il),
     &    Tu21(1,il),Tu22(1,il),Td11(1,il),Td12(1,il),
     &    Td21(1,il),Td22(1,il),Ru11(1,il),Ru12(1,il),
     &    Ru21(1,il),Ru22(1,il))
99998 continue
 

*******************************************************
*     Factor the System Matrix
      call    Factor(Nl,Np,T11,T12,T21,T22,U11,U12,U21,U22,
     &        R11,R12,R21,R22,S11,S12,S21,S22,Ea,Eb,Rd11,
     &        Rd12,Rd21,Rd22,Tu11,Tu12,Tu21,Tu22,Ru11,Ru12,
     &        Ru21,Ru22,Td11,Td12,Td21,Td22)
 
      RETURN
      END

 
 
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
 
      subroutine Vslo(Nk,C,Pvec,Vslovec)
      implicit none
      Integer  Nk,ik,ierr
*     compute the vertical slowness
      Complex*16 Pvec(Nk),Vslovec(Nk),C,Cinv
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
      T=CMPLX((Cinv+Pvec(ik))*(Cinv-Pvec(ik)))
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
 
