      program GMRC
*******************************************************************
*     The point explosion response of a Cylindrical 
*     Inclusion  centered at (Xc,Zc) of radius r, embedded within an 
*     elastic layer. Scattering arrays calculated with a fourth order 
*     implict method. The GMRES method is employed with
*     output written to RespX and RespZ
*******************************************************************
      implicit     none
      INTEGER      Nseg,Ndeg,Np,Np4,Nl,Nl1,Nl2,Nl4,Nl8,M,Qp,Q
*      Parameter    (Nseg=50,Ndeg=4)
      Parameter    (Nseg=1,Ndeg=4)
*      Parameter    (Np=Ndeg*Nseg+1,Np4=4*Np) !Np 201
      Parameter    (Np=3,Np4=4*Np) !Np 201
*      parameter    (Nl=23,NL1=Nl+1,M=Nl*Np4,Qp=20,Q=Qp-1)
      parameter    (Nl=2,NL1=Nl+1,M=Nl*Np4,Qp=12,Q=Qp-1)
      parameter    (NL2=2*NL,NL4=4*NL,NL8=8*NL)
      Real*8       h(0:Nl),xleft(Nl),xright(Nl)
      Real*8       rho0(0:NL+1),rho(0:NL+1),drho(NL)
      Complex*16   alpha0(0:NL+1),beta0(0:NL+1),dbeta(NL)
      Complex*16   alpha(0:NL+1),beta(0:NL+1)
      Complex*16   QB(0:NL+1)
      Complex*16   RespX(Np),RespZ(Np),CV
      Complex*16   RespX0(Np),RespZ0(Np)
      Real*8       rt3
      integer      il,i
      integer      i1,i2
      Real*8       vu,cpu
******************************************************************
      Integer     Maxit,iter,irst
      Real*8      L2,Tol,rmax
      Parameter   (Tol=1.D-6)
      Complex*16  X1(M),R(M),R0(M),b0(M),X(M)
      Complex*16  Pmat(M,Qp),c(Qp),Wvec(M),ek(M)
      Complex*16  Rmat(Q,Q),Jmat(2,Q),Vexact(M)
      logical     exact
******************************************************************
      Complex*16  Xwk(M),SY(M),PSY(M)
      Complex*16  dS(Np4,Np4,Nl)
      Complex*16  T11(Np,0:Nl),T12(Np,0:Nl)
      Complex*16  T21(Np,0:Nl),T22(Np,0:Nl)
      Complex*16  U11(Np,-1:Nl),U12(Np,-1:Nl)
      Complex*16  U21(Np,-1:Nl),U22(Np,-1:Nl)
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
      Complex*16  Ea(Np,0:Nl),Eb(Np,0:Nl)
      Complex*16  Df1(Np),Df2(Np),Uf1(Np),Uf2(Np)
      Complex*16  SigPu(Np),SigSu(Np),SigPd(Np),SigSd(Np)
 
*******************************************************
*     Source information
*******************************************************
*     source type
      Integer is
*     is=1 => double couple
*     is=2 => explosion
*     is=3 => Incident plane P-Wave
*     Source depth Zs
      Real*8  Zs
*     Source offset Xs
      Real*8  Xs
      Parameter(is=3,Zs=0.D0,Xs=0.D0)
*      Parameter(is=2,Zs=0.D0,Xs=0.D0)

*******************************************************
*     SLOWNESS CONTOUR DATA (Chebychev grid)
*******************************************************
      Integer     iz,iseg,ip
      Real*8      zvec(Ndeg),Kmax,k1,kval,Ap,Bp
      Complex*16  pvec(Np),dseg
*     pmax>=K1*w/beta0(0)
      Parameter   (K1=4.0)
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
*     Model geometry
*     23 scattering layer case
      Real*8     Xc,Zc,Rad,z
      Parameter  (Rad=.115D0,Xc=0.D0,Zc=Rad+0.2)
*******************************************************
	  DATA     beta0/1.01,1.51,2.51,3.51/	
      DATA     rho0/1.01,1.51,2.51,3.51/
      DATA     h/3*1.D0/
      DATA     drho/2*0.1D0/
      DATA     dbeta/2*0.1D0/
      DATA     QB/4*1.D10/

*      DATA     beta0/25*3.5D0/	
*      DATA     rho0/25*2.5D0/		
*      DATA     drho/23*0.1D0/
*      DATA     dbeta/23*0.2D0/
*      DATA     QB/25*3000.0D0/
*     Reference Layer Depths
*      DATA     h/0.2D0,23*0.01D0/


*      data     iwvec/1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,
*     &          17,18,19,20,21,22,23,24,25,26,27,28,29,30,31/
 
 
*      OPEN( UNIT=11, FILE='/home03/158/tmn158/GMRCRespX',
*     &     STATUS='UNKNOWN')
*      OPEN( UNIT=12, FILE='/home03/158/tmn158/GMRCRespZ',
*     &     STATUS='UNKNOWN')

      OPEN( UNIT=11, FILE='GMRCRespP',
     &     STATUS='UNKNOWN')
      OPEN( UNIT=12, FILE='GMRCRespS',
     &     STATUS='UNKNOWN')
 
**********************************************************
*     set left and right ends of inclusion at the
*     middle of each reference layer 
**********************************************************
       Do  500 iz=1,Nl
        z=0.205D0+Dfloat(iz-1)*0.01D0
*     left end of slab
        Xleft(iz)=Xc-Dsqrt(Rad**2-(z-Zc)**2)
*     right end of slab
        Xright(iz)=Xc+Dsqrt(Rad**2-(z-Zc)**2)
500   continue
***************************************************
*     set Slowness Grid  {-pmax,...,pmax}
***************************************************
*     set Chebchev abscissae
      do 13 iz=1,Ndeg
        zvec(iz)=-dcos(3.14159265358979324D0*DBLE(iz)
     &            /DBLE(Ndeg))
 13   Continue
 
      rt3=DSqrt(3.0D0)
      rho(0)=rho0(0)
      rho(NL+1)=rho0(NL+1)
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

*         w=dcmplx(iwvec(iw)*dw,eps)

        beta0(0)=CV(w,beta0(0),QB)
	beta(0)=beta0(0)
	beta0(Nl+1)=CV(w,beta0(Nl+1),QB)
	beta(Nl+1)=beta0(Nl+1)
	alpha0(0)=rt3*beta0(0)
	alpha(0)=alpha0(0)
	alpha0(Nl+1)=rt3*beta0(Nl+1)
	alpha(Nl+1)=alpha0(Nl+1)
        DO    99 IL=1,NL
          beta0(il)=CV(w,beta0(il),QB)
	  beta(il)=beta0(il)+CV(w,dbeta(il),QB)
	  alpha0(il)=rt3*beta0(il)
	  alpha(il)=rt3*beta(il)
   99   continue

        go to 007 !!!!!!!!!!!!!!!!!!!!!!!!!!DDDDEEEELLLLEEETTEEEEE
 
        Kmax=DBLE(K1*w/beta0(0))
        type*,'Kmax=',Kmax
        Ap=kmax/DBLE(Nseg)
        type*,'Ap=',Ap
        dseg=2.0*Ap
        ip=1
        Bp=Ap-Kmax
        Pvec(ip)=-Kmax/w
        Do 11 iseg=1,Nseg
          Do 10 iz=1,Ndeg
            ip=ip+1
            Kval=Ap*zvec(iz)+Bp
            type*,'ip,kval=',ip,kval
            pvec(ip)=Kval/w
*            type*,'ip,pvec(ip)=',ip,pvec(ip)
 10       Continue
          Bp=Bp+dseg
 11     Continue

007     continue
*!!!!!!!!!!!!DELETE!!!!!!!!!!!!
      w=Cmplx(1.00000000000000000)
	  pvec(1)=(-.1,0.)
	  pvec(2)=(0.,0.)
	  pvec(3)=(.1,0.)
*!!!!!!!!!!DELETE!!!!!!!!!!!!    

*        pause
 
*       calculate layer stack reflection response
	
*************************************************************
*     initialise Scattering operators dS, R/T coeffs and
*     factorise System matrix
*************************************************************
      call Init(Nl,Nl1,Np,Np4,pvec,etamat,netamat,
     &           w,h,xleft,xright,rho0,alpha0,beta0,rho,
     &           alpha,beta,Rd11,Rd12,Rd21,Rd22,Td11,Td12,
     &           Td21,Td22,Tu11,Tu12,Tu21,Tu22,Ru11,Ru12,
     &           Ru21,Ru22,Ea,Eb,R11,R12,R21,R22,S11,
     &           S12,S21,S22,U11,U12,U21,U22,T11,T12,T21,
     &           T22,dS)
 
 
****************************************************************
*     set incident field b0
***************************************************************

       call setinc(is,Nl,Nl1,Nl4,Np,Zs,Xs,rho0(0),
     &         alpha0(0),beta0(0),pvec,w,h,etamat,netamat,
     &           R11,R12,R21,R22,S11,S12,S21,S22,
     &           U11,U12,U21,U22,T11,T12,T21,T22,
     &           Rd11,Rd12,Rd21,Rd22,Td11,Td12,
     &           Td21,Td22,Ru11,Ru12,Ru21,Ru22,P1,P2,
     &           Q1,Q2,Df1,Df2,Uf1,Uf2,ea,eb,SigPu,
     &           SigPd,SigSu,SigSd,RespX0,RespZ0,b0)


 
      Do i=1,M
	    type*,'i,b0(i)=',i,b0(i)
      Enddo
*      stop
 
****************************************************
*     Read in exact solution if exact=.true.
****************************************************
      if(exact) then
        open(19,file='ExactDat',status='old')
      DO 99999 IL=1,M
	  Read(19,*) Vexact(il)
*	  type*,'Vexact(il)=',il,Vexact(il)
99999 continue
	close (19)
      endif
****************************************************
*     set initial approx X0
      DO 99998 I=1,M
	    X(i)=b0(i)       !BORN trial solution
*	    X(i)=Vexact(i)   !DELETE
99998 continue
 
      call  GMRESit(Np,Np4,Nl,Nl4,M,Qp,
     &           Q,iter,irst,L2,Tol,rmax,exact,
     &           R,R0,b0,Pmat,c,Wvec,ek,Rmat,
     &           Jmat,Vexact,Maxit,
     &           R11,R12,R21,R22,S11,S12,S21,S22,
     &           U11,U12,U21,U22,T11,T12,T21,T22,
     &           Ru11,Ru12,Ru21,Ru22,Tu11,Tu12,
     &           Tu21,Tu22,P1,P2,Q1,Q2,Df1,Df2,
     &           Uf1,Uf2,Xwk,dS,SY,PSY,ea,eb,X,X1)
 
        if(exact) then
          type*,'iterate,restarts,Solution:',iter,irst
      DO 99997 I=1,M
	    type*, 'CDabs(Vexact(i)-X(i))/CDabs(Vexact(i))=',
     &          i,CDabs(Vexact(i)-X(i))/CDabs(Vexact(i))
            type*
99997 continue
        else
            type*,'solution coverged after iteration',iter
            type*,'approximate L2 error=',L2
       DO 99996 I=1,M
	      type*, 'X(i)=',i,X(i)
          type*
99996 continue
        endif
      stop
 
*     set the receiver response
 
      call PSX(M,Nl,Np,Np4,ea,eb,pvec,
     &     etamat(1,0),netamat(1,0),rho0(0),beta0(0),
     &            U11,U12,U21,U22,T11,T12,T21,T22,
     &            R11,R12,R21,R22,S11,S12,S21,S22,
     &            Ru11,Ru12,Ru21,Ru22,Tu11,Tu12,
     &            Tu21,Tu22,P1,P2,Q1,Q2,Df1,Df2,
     &            Uf1,Uf2,dS,SY,X,RespX,RespZ) 
 
*       write TOTAL=REFERENCE+SCATTERED surface 
*       response at freq. iw to file
        Write(11,*)  iwvec(iw),w
 
        DO    91 I=1,Np
          Write(11,*)  i,Pvec(i)
   91   continue
 
        DO    97 I=1,Np
*          type*,'i,RespX(i)=',i,RespX(i)
          Write(11,*)  RespX0(i)+RespX(i)
   97   continue
 
        Write(12,*)  iwvec(iw),w
 
        DO    96 I=1,Np
*          type*,'i,RespZ(i)=',i,RespZ(i)
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
      subroutine GMRESit(Np,Np4,Nl,Nl4,
     &           M,Qp,Q,iter,irst,L2,Tol,rmax,exact,
     &           R,R0,b0,Pmat,c,Wvec,ek,Rmat,
     &           Jmat,Vexact,Maxit,
     &           R11,R12,R21,R22,S11,S12,S21,S22,
     &           U11,U12,U21,U22,T11,T12,T21,T22,
     &           Ru11,Ru12,Ru21,Ru22,Tu11,Tu12,
     &           Tu21,Tu22,P1,P2,Q1,Q2,Df1,Df2,
     &           Uf1,Uf2,Xwk,dS,SY,PSY,ea,eb,X,X1)
**********************************
      Implicit    none
      Integer     Np,Np4,Qp,Q,M,i,k,iter
      INTEGER     NL,NL4
******************************************************************
      Real*8      L2,Tol,rmax
      Complex*16  X1(M),R(M),R0(M),b0(M),X(M)
      Complex*16  c(Qp),Wvec(M),ek(M),Vexact(M)
      Complex*16  Pmat(M,Qp),Rmat(Q,Q),Jmat(2,Q)
      integer     Maxit,irst,irstmax
      parameter   (irstmax=2)  !only allow two restarts
      logical     sing,exact
******************************************************************
      Complex*16  Ea(Np,0:Nl),Eb(Np,0:Nl)
      Complex*16  dS(Np4,Np4,Nl),Xwk(M),SY(M),PSY(M)
      Complex*16  T11(Np,0:Nl),T12(Np,0:Nl)
      Complex*16  T21(Np,0:Nl),T22(Np,0:Nl)
      Complex*16  U11(Np,-1:Nl),U12(Np,-1:Nl)
      Complex*16  U21(Np,-1:Nl),U22(Np,-1:Nl)
      Complex*16  R11(Np,0:Nl),R12(Np,0:Nl)
      Complex*16  R21(Np,0:Nl),R22(Np,0:Nl)
      Complex*16  S11(Np,0:Nl),S12(Np,0:Nl)
      Complex*16  S21(Np,0:Nl),S22(Np,0:Nl)
      Complex*16  P1(Np,0:Nl),P2(Np,0:Nl)
      Complex*16  Q1(Np,0:Nl),Q2(Np,0:Nl)
 
      Complex*16  Tu11(Np,0:Nl),Tu12(Np,0:Nl)
      Complex*16  Tu21(Np,0:Nl),Tu22(Np,0:Nl)
      Complex*16  Ru11(Np,-1:Nl),Ru12(Np,-1:Nl)
      Complex*16  Ru21(Np,-1:Nl),Ru22(Np,-1:Nl)
      Complex*16  Df1(Np),Df2(Np),Uf1(Np),Uf2(Np)
 
      irst=0
007   continue

      type*,'in GMRESit,M=',M
************************************************************ 
*     set initial residual r0=b0-Ax
************************************************************
*       Do i=1,M
*	      type*,'check this X(i)=',i,X(i)
*      enddo

      call AtimesV(M,Nl,Nl4,Np,Np4,
     &            R11,R12,R21,R22,S11,S12,S21,S22,
     &            U11,U12,U21,U22,T11,T12,T21,T22,
     &            Ru11,Ru12,Ru21,Ru22,Tu11,Tu12,
     &            Tu21,Tu22,P1,P2,Q1,Q2,Df1,Df2,
     &            Uf1,Uf2,dS,Xwk,SY,PSY,ea,eb,X,X1)
 

	
       Do i=1,M
	    r0(i)=b0(i)-X1(i)
       Enddo
	
*       Do i=1,M
*	      type*,'r0(i)=',i,r0(i)
*      enddo
*      pause


*     set r0=P1*r0 and store Housholder vector in Pmat(.,1)
      call setPkV(Pmat,r0,M,1,M,Qp,c,sing)
*     Wvec=P1*r0
      Wvec(1)=r0(1)
      DO 99998 I=2,M
	    Wvec(i)=(0.0,0.0)
99998 continue
*
*
*     ******************************************
*     set Col k of upper triangular Rmat(M,M)
*     ******************************************
*
      Maxit=min(Q,M-1)
*      type*,'Maxit=',Maxit
      iter=Maxit
      Do 100  k=1,Maxit
*       set unit vector ek
        Do 10 i=1,M
	      ek(i)=(0.0,0.0)
10      continue
	    ek(k)=(1.0,0.0)
*
*       ******************************************
*       set ek=P(1)*....P(k)*ek
*       ******************************************
*
        call QV(Pmat,M,k,M,c,ek)

*       ******************************************
*       set r=(1-PdS)*ek
*       ******************************************
*
      call AtimesV(M,Nl,Nl4,Np,Np4,
     &            R11,R12,R21,R22,S11,S12,S21,S22,
     &            U11,U12,U21,U22,T11,T12,T21,T22,
     &            Ru11,Ru12,Ru21,Ru22,Tu11,Tu12,
     &            Tu21,Tu22,P1,P2,Q1,Q2,Df1,Df2,
     &            Uf1,Uf2,dS,Xwk,SY,PSY,ea,eb,ek,r)
 
*	  Do i=1,M
*	     type*,'i,r(i)=',i,r(i)
*	  Enddo
      

*
*       ******************************************
*       set r=P(k)*....*P(1)*r
*       ******************************************
*
        call QtV(Pmat,M,k,M,c,r)
*	    Do i=1,M
*	     type*,'i,r(i)=',i,r(i)
*	    Enddo

		rmax=0.0
		do 11 i=k+1,M
            rmax=max(rmax,CDabs(r(i)))
11      continue
        If(rmax.gt.0.0)then
		  call setPkV(Pmat,r,M,k+1,M,Qp,c,sing)
		else !zero Householder vector
		  type*,'rmax=0.0 case'
*		  iter=k-1
*		  go to 200 !restart
		  Do 12 i= k+1,M         !recomment out
		    Pmat(i,K+1)=(0.0,0.0)
12        continue		
	    Endif

        if(k.gt.1) then
*
*       ******************************************
*       set r=J(k-1)*....*J(1)*r
*       ******************************************
*
        Do 13 i=1,k-1
          call rotV(r,i,Jmat(1,i),Jmat(2,i))
13      continue
        Endif
	    if(CDabs(r(k+1)).ne.0.0)then
	       Jmat(1,k)=r(k)
	       Jmat(2,k)=r(k+1)
	    else
	       Jmat(1,k)=(1.0,0.0)
	       Jmat(2,k)=(0.0,0.0)
	    Endif
        call rotV(r,k,Jmat(1,k),Jmat(2,k))
        call rotV(Wvec,k,Jmat(1,k),Jmat(2,k))
*
*       ******************************************
*       set Rmat(.,k)=[Rmat(.,k-1),r]
*       ******************************************
*
        Do 14 i=1,k
	      Rmat(i,k)=r(i)
14      continue
*
*       ******************************************
*       Test for convergence
*       ******************************************
*
        L2=CDabs(Wvec(K+1))
        type*,'iter,irst,||r||=',k,irst,L2
        pause
        if(L2.lt.Tol)then
	  iter=k
	  go to 200
	  endif
 
100   continue
 
200   continue
*
*     ******************************************
*     solve Wvec= Rmat(.,iter)*Wvec(iter)
*     ******************************************
*
      call rsolvb(Rmat,iter,Q,Wvec)
 
      Do 250 k=1,iter
*
*       set unit vector ek
        Do 210 i=1,M
		  ek(i)=(0.0,0.0)
210     continue
	    ek(k)=(1.0,0.0)
*
*       ******************************************
*       set ek=P(1)*....P(k)*ek
*       ******************************************
*
        call QV(Pmat,M,k,M,c,ek)
	    Do 215 i=1,M
*	      type*,'i,ek(i),Wvec(i)=',i,ek(i),Wvec(i)
	      X(i)=X(i)+ek(i)*Wvec(k)
215     continue
 
250   continue
 
      if((L2.gt.Tol).and.(irst.lt.irstmax))then
	  irst=irst+1
	  goto 007
      endif
      return
      end

*************************************************************
 
      subroutine rsolvb(Rmat,dim,M,Wvec)
*     Column oriented backsub
      implicit none
      integer  M,dim
      Complex*16  Rmat(M,M),Wvec(*)
      integer  i,j
        do 12 j=dim,1,-1
          Wvec(j)=Wvec(j)/Rmat(j,j)
          do 11 i=1,j-1
            Wvec(i)=Wvec(i)-Rmat(i,j)*Wvec(j)
11        continue
12      continue
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
*     P(k)...*P(j)*..P(1)*V
      integer  n,np,k
      Complex*16  a(np,np),V(n)
      Real*8     c(n)
      integer  i,j
      Complex*16  sum,tau
      do 13 j=1,k
          sum=(0.0,0.0)
          do 11 i=j,n
            sum=sum+conjg(a(i,j))*V(i)
11        continue
          tau=sum/c(j)
          do 12 i=j,n
            V(i)=V(i)-tau*a(i,j)
12        continue
13    continue
      return
      end
 
      subroutine QV(a,n,k,np,c,V)
      implicit none
*     P(1)...*P(j)*..P(k)*V
      integer  n,np,k
      Complex*16  a(np,np),V(n)
      Real*8     c(n)
      integer  i,j
      Complex*16  sum,tau
      do 13 j=k,1,-1
          sum=(0.0,0.0)
          do 11 i=j,n
            sum=sum+conjg(a(i,j))*V(i)
11        continue
          tau=sum/c(j)
          do 12 i=j,n
            V(i)=V(i)-tau*a(i,j)
12        continue
13    continue
      return
      end
 
      SUBROUTINE setPkV(a,V,n,k,np,mp,c,sing)
      implicit none
      INTEGER  n,np,mp,k
      Complex*16  a(np,np),V(n)
      Real*8     c(n)
      LOGICAL  sing
      INTEGER  i
      Real*8     scale,L2,absakk
      Complex*16  sigma
	  if(k.lt.1) Pause'k.lt.1 in setPkV'
	  if(k.gt.Mp) Pause'k.gt.Mp in setPkV'
      sing=.false.
      scale=0.
      do 11 i=k,n
            scale=max(scale,CDabs(V(i)))
11     continue
       if(scale.lt.1.E-8)then
            c(k)=(0.0,0.0)
       else
         do 12 i=k,n
              V(i)=V(i)/scale
12       continue
         L2=0.0
         do 13 i=k,n
              L2=L2+V(i)*conjg(V(i))
13       continue
         L2=sqrt(L2)  ! L2 norm of V
         absakk=CDabs(V(k))
         if(absakk.lt.1.E-6) then
           pause 'absakk.lt.1.E-6'
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
 
*************************************************************

      SUBROUTINE setPSY(Nl,Nl4,Np,ea,eb,
     &           R11,R12,R21,R22,S11,S12,S21,S22,
     &           U11,U12,U21,U22,T11,T12,T21,T22,
     &           Ru11,Ru12,Ru21,Ru22,Tu11,Tu12,
     &           Tu21,Tu22,P1,P2,Q1,Q2,Df1,Df2,
     &           Uf1,Uf2,Xwk,SY,X)
	
*     set X=P*SY
      Implicit none
      INTEGER     Nl,Nl4,Np
      complex*16  ea(Np,0:Nl),eb(Np,0:Nl)
      Complex*16  Xwk(Nl4*Np)
      Complex*16  SY(Nl4*Np),X(Nl4*Np)
 
      Complex*16  Df1(Np),Df2(Np),Uf1(Np),Uf2(Np)
      Complex*16  T11(Np,0:Nl),T12(Np,0:Nl)
      Complex*16  T21(Np,0:Nl),T22(Np,0:Nl)
      Complex*16  U11(Np,-1:Nl),U12(Np,-1:Nl)
      Complex*16  U21(Np,-1:Nl),U22(Np,-1:Nl)
      Complex*16  R11(Np,0:Nl),R12(Np,0:Nl)
      Complex*16  R21(Np,0:Nl),R22(Np,0:Nl)
      Complex*16  S11(Np,0:Nl),S12(Np,0:Nl)
      Complex*16  S21(Np,0:Nl),S22(Np,0:Nl)
 
      Complex*16  Tu11(Np,0:Nl),Tu12(Np,0:Nl)
      Complex*16  Tu21(Np,0:Nl),Tu22(Np,0:Nl)
      Complex*16  Ru11(Np,-1:Nl),Ru12(Np,-1:Nl)
      Complex*16  Ru21(Np,-1:Nl),Ru22(Np,-1:Nl)
      Complex*16  P1(Np,0:Nl),P2(Np,0:Nl)
      Complex*16  Q1(Np,0:Nl),Q2(Np,0:Nl)
      Integer     Np2,Np3,Np6,Np7
****************************************************************
*     LOCAL variables
*****************************************************************
      Integer    ioff4,ioff8,IL,ip
*      type*,'entered setPSY'


      call  UpSweep(Np,Nl,SY,Ru11,Ru12,Ru21,Ru22,
     &           Tu11,Tu12,Tu21,Tu22,ea,eb,T11,T12,T21,
     &           T22,U11,U12,U21,U22,R11,R12,R21,R22,
     &           S11,S12,S21,S22,P1,P2,Q1,Q2)
*        Do ip=1,Np
*      Do il=0,Nl
*          type*,'il,ip,P1(ip,il),P2(ip,il)=',ip,il,P1(ip,il),P2(ip,il)
*        Enddo
*      Enddo
*      pause

*        Do ip=1,Np
*      Do il=0,Nl
*          type*,'il,ip,Q1(ip,il),Q2(ip,il)=',ip,il,Q1(ip,il),Q2(ip,il)
*        Enddo
*      Enddo
*      pause
 
*     set Df=downgoing from Free surface
 
      DO 99999 IP=1,NP
*       Uf=U(-1)^(-1)*P0
        Uf1(ip)=U11(ip,-1)*P1(ip,0)+U12(ip,-1)*P2(ip,0)
        Uf2(ip)=U21(ip,-1)*P1(ip,0)+U22(ip,-1)*P2(ip,0)
 
*       Df=Ru(-1)*Uf
        Df1(ip)=Ru11(ip,-1)*Uf1(ip)+Ru12(ip,-1)*Uf2(ip)
        Df2(ip)=Ru21(ip,-1)*Uf1(ip)+Ru22(ip,-1)*Uf2(ip)
99999 continue
 
 
      call DownSweep(Np,NL,R11,R12,R21,R22,
     &     S11,S12,S21,S22,P1,P2,Q1,Q2,Df1,Df2,X)
 
      RETURN
      END
 
      Subroutine DownSweep(Np,NL,R11,R12,R21,R22,S11,S12,S21,
     &                         S22,P1,P2,Q1,Q2,Df1,Df2,X)
      Implicit    None
      Integer     Np,NL,il,ilm1,i1,i2,ip
      Complex*16  R11(Np,0:Nl),R12(Np,0:Nl)
      Complex*16  R21(Np,0:Nl),R22(Np,0:Nl)
      Complex*16  S11(Np,0:Nl),S12(Np,0:Nl)
      Complex*16  S21(Np,0:Nl),S22(Np,0:Nl)
 
      Complex*16  P1(Np,0:Nl),P2(Np,0:Nl)
      Complex*16  Q1(Np,0:Nl),Q2(Np,0:Nl)
      Complex*16  Df1(Np),Df2(Np),X(*)
      Integer     ip1,ip2,ip3,ip4
***************************************
      DO 99999 IP=1,NP
        ip1=ip
        ip2=ip1+Np
        ip3=ip2+Np
        ip4=ip3+Np
        X(ip1)=Q1(ip,0)+S11(ip,0)*Df1(ip)+S12(ip,0)*Df2(ip)
        X(ip2)=Q2(ip,0)+S21(ip,0)*Df1(ip)+S22(ip,0)*Df2(ip)
        X(ip3)=P1(ip,1)+R11(ip,1)*X(ip1)+R12(ip,1)*X(ip2)
        X(ip4)=P2(ip,1)+R21(ip,1)*X(ip1)+R22(ip,1)*X(ip2)
99999 continue
****************************************
      DO 99998 IL=2,Nl
	    i1=(il-2)*4*Np
	    i2=(il-1)*4*Np
	    ilm1=il-1
        DO 99997 IP=1,NP
          ip1=ip
          ip2=ip1+Np
          ip3=ip2+Np
          ip4=ip3+Np
 
	  X(i2+ip1)=Q1(ip,ilm1)+S11(ip,ilm1)*X(i1+ip1)+
     &                S12(ip,ilm1)*X(i1+ip2)
	  X(i2+ip2)=Q2(ip,ilm1)+S21(ip,ilm1)*X(i1+ip1)+
     &                S22(ip,ilm1)*X(i1+ip2)
          X(i2+ip3)=P1(ip,il)+R11(ip,il)*X(i2+ip1)+
     &                R12(ip,il)*X(i2+ip2)
	  X(i2+ip4)=P2(ip,il)+R21(ip,il)*X(i2+ip1)+
     &                R22(ip,il)*X(i2+ip2)
99997   continue
99998 continue
      return
      end
	
	
      Subroutine UpSweep(Np,Nl,SY,Ru11,Ru12,Ru21,Ru22,
     &           Tu11,Tu12,Tu21,Tu22,ea,eb,T11,T12,T21,
     &           T22,U11,U12,U21,U22,R11,R12,R21,R22,
     &           S11,S12,S21,S22,P1,P2,Q1,Q2)
      Implicit None
      Integer     NL,Np,Np4,Np2,ip,NLM1,il,ilp1
      Complex*16  SY(*)
	  complex*16  ea(Np,0:nl),eb(Np,0:nl)
 
      Complex*16  T11(Np,0:Nl),T12(Np,0:Nl)
      Complex*16  T21(Np,0:Nl),T22(Np,0:Nl)
      Complex*16  U11(Np,-1:Nl),U12(Np,-1:Nl)
      Complex*16  U21(Np,-1:Nl),U22(Np,-1:Nl)

      Complex*16  R11(Np,0:Nl),R12(Np,0:Nl)
      Complex*16  R21(Np,0:Nl),R22(Np,0:Nl)
      Complex*16  S11(Np,0:Nl),S12(Np,0:Nl)
      Complex*16  S21(Np,0:Nl),S22(Np,0:Nl)
 
      Complex*16  Tu11(Np,0:Nl),Tu12(Np,0:Nl)
      Complex*16  Tu21(Np,0:Nl),Tu22(Np,0:Nl)
      Complex*16  Ru11(Np,-1:Nl),Ru12(Np,-1:Nl)
      Complex*16  Ru21(Np,-1:Nl),Ru22(Np,-1:Nl)
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

*         Q=R*SY(ip1)+T*Y
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

*         P(0)=T*Y
          P1(ip,0)=T11(ip,0)*Y11+T12(ip,0)*Y21
          P2(ip,0)=T21(ip,0)*Y11+T22(ip,0)*Y21
*        type*,'ip,il,P1(ip,il),P2(ip,il)=',ip,0,P1(ip,0),P2(ip,0)

*         Q(0)=U*Y
          Q1(ip,0)=U11(ip,0)*Y11+U12(ip,0)*Y21
          Q2(ip,0)=U21(ip,0)*Y11+U22(ip,0)*Y21

*        type*,'ip,il,Q1(ip,il),Q2(ip,il)=',ip,0,Q1(ip,0),Q2(ip,0)
 
99996 continue
      return
      end
 
************************************************************
      SUBROUTINE Init(Nl,Nl1,Np,Np4,pvec,etamat,netamat,
     &           w,h,xleft,xright,rho0,alpha0,beta0,rho,
     &           alpha,beta,Rd11,Rd12,Rd21,Rd22,Td11,Td12,
     &           Td21,Td22,Tu11,Tu12,Tu21,Tu22,Ru11,Ru12,
     &           Ru21,Ru22,Ea,Eb,R11,R12,R21,R22,S11,S12,
     &           S21,S22,U11,U12,U21,U22,T11,T12,T21,T22,dS)
	
*     set up scattering array dS and factorise propagator
      Implicit    none
      INTEGER     Nl,Nl1,Np,Np4
      Integer     Np2,Np3,ilp1,il,il2,ip
      Real*8      h(0:Nl),xleft(Nl),xright(Nl)
      Real*8      rho0(0:Nl1),rho(0:Nl1)
      Complex*16  alpha0(0:Nl1),beta0(0:Nl1)
      Complex*16  alpha(0:Nl1),beta(0:Nl1)
      Complex*16  w,pvec(Np)
      Complex*16  dS(Np4,Np4,Nl)
      Complex*16  T11(Np,0:Nl),T12(Np,0:Nl)
      Complex*16  T21(Np,0:Nl),T22(Np,0:Nl)
      Complex*16  U11(Np,-1:Nl),U12(Np,-1:Nl)
      Complex*16  U21(Np,-1:Nl),U22(Np,-1:Nl)
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
      Complex*16  Ea(Np,0:Nl),Eb(Np,0:Nl)
 
      Complex*16  C0,C1,Ci
      integer     i,j
 
      Np2=2*Np
      Np3=3*Np
 
      C0=(0.D0,0.D0)
      C1=(1.D0,0.D0)	  	
      Ci=(0.D0,1.D0)
 
*     set the vertical slowness
      Do 53 il=0,Nl+1
        call Vslo(Np,alpha0(il),Pvec,etamat(1,il))
        call Vslo(Np,beta0(il),Pvec,netamat(1,il))
   53 continue
 
      Do 50 il=0,Nl
      DO 99999 IP=1,NP
        Ea(ip,il)=CDexp(Ci*w*h(il)*etamat(ip,il))
        Eb(ip,il)=CDexp(Ci*w*h(il)*netamat(ip,il))
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
 
*     set layer scattering arrays
      DO 99997 IL=1,NL
*Born SINGLE SCATTERING 

*      call  BornScat(Np,Np2,Np4,pvec,etamat(1,il),netamat(1,il),
*     &      w,h(il),xleft(il),xright(il),rho0(il),alpha0(il),
*     &      beta0(il),rho(il),alpha(il),beta(il),dS(1,1,il))

*Analytic solution for homo. plane layer
       call SetS(Np,Np4,pvec,etamat(1,il),netamat(1,il),
     &      w,h(il),rho0(il),alpha0(il),beta0(il),rho(il),
     &      alpha(il),beta(il),Ea(1,il),Eb(1,il),dS(1,1,il))  

*      Do i=1,Np4
*        Do j=1,Np4
*          type*,'i,j,dS(i,j,il)=',i,j,dS(i,j,il)
*        enddo
*      enddo
*      pause
              

99997 continue
*******************************************************
*     Factor the System Matrix
      call    Factor(Nl,Np,T11,T12,T21,T22,U11,U12,U21,U22,
     &        R11,R12,R21,R22,S11,S12,S21,S22,Ea,Eb,Rd11,
     &        Rd12,Rd21,Rd22,Tu11,Tu12,Tu21,Tu22,Ru11,Ru12,
     &        Ru21,Ru22,Td11,Td12,Td21,Td22)
 
      RETURN
      END

************************************************************** 
************************************************************** 
************************************************************** 

      subroutine SetS(Np,Np4,pvec,etavec,netavec,w,dh,
     &           rho0,alpha0,beta0,rho,alpha,beta,Ea,Eb,dS)
	
*     set up scattering array dS and factorise propagator
      Implicit    none
      INTEGER     Np,Np2,Np3,Np4,ip
      Real*8      dh,rho0,rho
      Complex*16  alpha0,beta0
      Complex*16  alpha,beta
      Complex*16  w,pvec(Np)
      Complex*16  dS(Np4,Np4)
      Complex*16  etavec(Np),netavec(Np)
      Complex*16  Ea(Np),Eb(Np)
*
*     local variables
*      
      Integer  Np1
      parameter(Np1=20)

	  complex*16  InvDet,a,b,c,d
	  complex*16  RRt11,RRt21,RRt12,RRt22
	  complex*16  RRb11,RRb21,RRb12,RRb22 
	  complex*16  TTd11,TTd21,TTd12,TTd22 
      Complex*16  Rd11(Np1,2),Rd12(Np1,2)
      Complex*16  Rd21(Np1,2),Rd22(Np1,2)
      Complex*16  Tu11(Np1,2),Tu12(Np1,2)
      Complex*16  Tu21(Np1,2),Tu22(Np1,2)
      Complex*16  Td11(Np1,2),Td12(Np1,2)
      Complex*16  Td21(Np1,2),Td22(Np1,2)
      Complex*16  Ru11(Np1,2),Ru12(Np1,2)
      Complex*16  Ru21(Np1,2),Ru22(Np1,2)
      Complex*16  etavec1(Np1),netavec1(Np1)
      Complex*16  Ea1(Np1),Eb1(Np1)
 
      Complex*16  C0,C1,Ci
      integer     i,j,ip1,ip2,ip3,ip4

      If(Np.gt.Np1) stop 'Np.gt.Np1 in SetS'
 
      C0=(0.D0,0.D0)
      C1=(1.D0,0.D0)	  	
      Ci=(0.D0,1.D0)
      Np2=2*Np
      Np3=3*Np
 
*     set the vertical slowness
      call Vslo(Np,alpha,Pvec,etavec1)
      call Vslo(Np,beta,Pvec,netavec1)
      DO 99999 IP=1,NP
        Ea1(ip)=CDexp(Ci*w*dh*etavec1(ip))
        Eb1(ip)=CDexp(Ci*w*dh*netavec1(ip))
99999 continue
 
****************************************************************

      Do i=1,Np4
        Do j=1,Np4
          dS(i,j)=C0
        enddo
      enddo

*       top interface 
        call IntScat(Np,pvec,
     &    etavec,netavec,rho0,beta0,
     &    etavec1,netavec1,rho,beta,
     &    Rd11(1,1),Rd12(1,1),Rd21(1,1),Rd22(1,1),
     &    Tu11(1,1),Tu12(1,1),Tu21(1,1),Tu22(1,1),
     &    Td11(1,1),Td12(1,1),Td21(1,1),Td22(1,1),
     &    Ru11(1,1),Ru12(1,1),Ru21(1,1),Ru22(1,1))

*       bottom interface 
        call IntScat(Np,pvec,
     &    etavec1,netavec1,rho,beta,
     &    etavec,netavec,rho0,beta0,
     &    Rd11(1,2),Rd12(1,2),Rd21(1,2),Rd22(1,2),
     &    Tu11(1,2),Tu12(1,2),Tu21(1,2),Tu22(1,2),
     &    Td11(1,2),Td12(1,2),Td21(1,2),Td22(1,2),
     &    Ru11(1,2),Ru12(1,2),Ru21(1,2),Ru22(1,2))

      Do 99995 ip=1,Np
      
      ip1=ip
      ip2=ip+Np
      ip3=ip2+Np
      ip4=ip3+Np
	  
	  RRt11=Rd11(ip,2)*ea1(ip)*ea1(ip)
	  RRt12=Rd12(ip,2)*ea1(ip)*eb1(ip)
	  RRt21=Rd21(ip,2)*ea1(ip)*eb1(ip)
	  RRt22=Rd22(ip,2)*eb1(ip)*eb1(ip)

	  a=(C1-RRt22*Ru22(ip,1)-RRt12*Ru21(ip,1))
	  b=(RRt22*Ru12(ip,1)+RRt12*Ru11(ip,1))	  
	  c=(RRt21*Ru22(ip,1)+RRt11*Ru21(ip,1))
	  d=(C1-RRt21*Ru12(ip,1)-RRt11*Ru11(ip,1))
	  InvDet=C1/(a*d-b*c)
	  
	  a=InvDet*a
	  b=InvDet*b
	  c=InvDet*c
	  d=InvDet*d
      
	  TTd11=Td11(ip,1)*a+Td21(ip,1)*b
	  TTd12=Td12(ip,1)*a+Td22(ip,1)*b
	  TTd21=Td11(ip,1)*c+Td21(ip,1)*d
	  TTd22=Td12(ip,1)*c+Td22(ip,1)*d
	  
      
	  dS(ip1,ip1)=Rd11(ip,1)+RRt22*TTd21*Tu12(ip,1)+RRt21*TTd11*Tu12(ip,1)+
     &          RRt12*TTd21*Tu11(ip,1)+RRt11*TTd11*Tu11(ip,1)

      dS(ip2,ip1)=Rd21(ip,1)+RRt22*TTd21*Tu22(ip,1)+RRt21*TTd11*Tu22(ip,1)+
     &          RRt12*TTd21*Tu21(ip,1)+RRt11*TTd11*Tu21(ip,1)

      dS(ip2,ip2)=Rd22(ip,1)+RRt22*TTd22*Tu22(ip,1)+RRt21*TTd12*Tu22(ip,1)+
     &          RRt12*TTd22*Tu21(ip,1)+RRt11*TTd12*Tu21(ip,1)


      dS(ip3,ip1)=eb1(ip)*TTd21*Td12(ip,2)+ea1(ip)*TTd11*Td11(ip,2)-ea(ip)

      dS(ip4,ip1)=eb1(ip)*TTd21*Td22(ip,2)+ea1(ip)*TTd11*Td21(ip,2)

      dS(ip4,ip2)=eb1(ip)*TTd22*Td22(ip,2)+ea1(ip)*TTd12*Td21(ip,2)-eb(ip)
      
	  dS(ip1,ip2)=-dS(ip2,ip1)
	  dS(ip1,ip3)=dS(ip3,ip1)
	  dS(ip1,ip4)=-dS(ip4,ip1)
	  dS(ip2,ip3)=-dS(ip4,ip1)
	  dS(ip2,ip4)=dS(ip4,ip2)
	  dS(ip3,ip2)=dS(ip4,ip1)
	  dS(ip3,ip3)=dS(ip1,ip1)
	  dS(ip3,ip4)=dS(ip2,ip1)
	  dS(ip4,ip3)=-dS(ip2,ip1)
	  dS(ip4,ip4)=dS(ip2,ip2)


99995 continue     
 
      RETURN
      END

************************************************************** 
************************************************************** 
************************************************************** 
************************************************************** 
 
      Subroutine Factor(Nl,Np,T11,T12,T21,T22,U11,U12,U21,U22,
     &           R11,R12,R21,R22,S11,S12,S21,S22,Ea,Eb,Rd11,Rd12,
     &           Rd21,Rd22,Tu11,Tu12,Tu21,Tu22,Ru11,Ru12,Ru21,
     &           Ru22,Td11,Td12,Td21,Td22)
      Implicit    None
      integer     Np,ip,il,nl,ilp1,nlm1
      Complex*16  Ea(Np,0:Nl),Eb(Np,0:Nl)
      Complex*16  T11(Np,0:Nl),T12(Np,0:Nl)
      Complex*16  T21(Np,0:Nl),T22(Np,0:Nl)
      Complex*16  U11(Np,-1:Nl),U12(Np,-1:Nl)
      Complex*16  U21(Np,-1:Nl),U22(Np,-1:Nl)
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
      Complex*16  MT11,MT12,MT21,MT22,C1,InvDet
 
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

	    U11(ip,il)=W11*Ru11(ip,il)+W12*Ru21(ip,il)
	    U12(ip,il)=W11*Ru12(ip,il)+W12*Ru22(ip,il)
	    U21(ip,il)=W21*Ru11(ip,il)+W22*Ru21(ip,il)
	    U22(ip,il)=W21*Ru12(ip,il)+W22*Ru22(ip,il)
		
		
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

*       T=Tu*W

        T11(ip,il)=Tu11(ip,il)*W11+Tu12(ip,il)*W21
		T12(ip,il)=Tu11(ip,il)*W12+Tu12(ip,il)*W22
		T21(ip,il)=Tu21(ip,il)*W11+Tu22(ip,il)*W21
		T22(ip,il)=Tu21(ip,il)*W12+Tu22(ip,il)*W22

99997 continue
 
99998 continue
 
      DO 99996 IP=1,NP
*      pause'check point 4'

*         S(0)=S(0)*E
          S11(ip,0)=S11(ip,0)*Ea(ip,0)
          S12(ip,0)=S12(ip,0)*Eb(ip,0)
          S21(ip,0)=S21(ip,0)*Ea(ip,0)
          S22(ip,0)=S22(ip,0)*Eb(ip,0)
 
*         R(0)=E*R(0)*E
          R11(ip,0)=Ea(ip,0)*R11(ip,0)*Ea(ip,0)
          R12(ip,0)=Ea(ip,0)*R12(ip,0)*Eb(ip,0)
          R21(ip,0)=Eb(ip,0)*R21(ip,0)*Ea(ip,0)
          R22(ip,0)=Eb(ip,0)*R22(ip,0)*Eb(ip,0)
*         U=(I-R(0)*Ru(-1))^(-1)
	  W11=C1-R21(ip,0)*Ru12(ip,-1)-
     &        R22(ip,0)*Ru22(ip,-1)
	  W12=R11(ip,0)*Ru12(ip,-1)+
     &        R12(ip,0)*Ru22(ip,-1)
	  W21=R21(ip,0)*Ru11(ip,-1)+
     &        R22(ip,0)*Ru21(ip,-1)
	  W22=C1-R11(ip,0)*Ru11(ip,-1)-
     &        R12(ip,0)*Ru21(ip,-1)
	  InvDet=C1/(W11*W22-W12*W21)
 
*	  U11(ip,-1)=W11*InvDet*Ea(ip,0)
*	  U12(ip,-1)=W12*InvDet*Eb(ip,0)
*	  U21(ip,-1)=W21*InvDet*Ea(ip,0)
*	  U22(ip,-1)=W22*InvDet*Eb(ip,0)
 
	  U11(ip,-1)=W11*InvDet !rephased in setInc
	  U12(ip,-1)=W12*InvDet !rephased in setInc
	  U21(ip,-1)=W21*InvDet !rephased in setInc
	  U22(ip,-1)=W22*InvDet !rephased in setInc
 
99996 continue
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
	  type*,'T=',T
      T1=W0**SIGMA/DSIN(SIGMA*PI/2)
	  type*,'T1=',T1
      CW=(1.D0,0.D0)+CMPLX(0.5D0*T1/Q0,0.D0)*T
	  type*,'CW=',CW
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
 
 
*********************************************************************
      SUBROUTINE setinc(is,Nl,Nl1,Nl4,Np,
     &           Zs,Xs,rho,alpha,beta,pvec,w,h,etamat,
     &           netamat,R11,R12,R21,R22,S11,S12,S21,S22,
     &           U11,U12,U21,U22,T11,T12,T21,T22,
     &           Rd11,Rd12,Rd21,Rd22,Td11,Td12,
     &           Td21,Td22,Ru11,Ru12,Ru21,Ru22,
     &           P1,P2,Q1,Q2,Df1,Df2,Uf1,Uf2,ea,
     &           eb,SigPu,SigPd,SigSu,SigSd,
     &           RespX0,RespZ0,X)
	
*     set the incident field
      Implicit none
      INTEGER     is,Nl,Nl1,Nl2,Nl4,Np,ip
      Real*8      h(0:Nl),rho,Zs,Xs
      Complex*16  w,pvec(Np)
      Complex*16  X(Nl4*Np),alpha,beta
 
      Complex*16  Df1(Np),Df2(Np),Uf1(Np),Uf2(Np)
      Complex*16  T11(Np,0:Nl),T12(Np,0:Nl)
      Complex*16  T21(Np,0:Nl),T22(Np,0:Nl)
      Complex*16  U11(Np,-1:Nl),U12(Np,-1:Nl)
      Complex*16  U21(Np,-1:Nl),U22(Np,-1:Nl)
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
      Complex*16  Ru11(Np,-1:Nl),Ru12(Np,-1:Nl)
      Complex*16  Ru21(Np,-1:Nl),Ru22(Np,-1:Nl)
      Complex*16  etamat(Np,0:Nl1),netamat(Np,0:Nl1)
      Complex*16  RespX0(Np),RespZ0(Np),Ea(Np,0:Nl),Eb(Np,0:Nl)
      Complex*16  B11,B12,B21,B22,Temp,C1,C0,Ci
      Integer     il,i1,i2,ilm1,np2,np3,np6,np7
      Integer     ip1,ip2,ip3,ip4,ioff4,ioff8
******************************************************************
*     Source data
      Complex*16  SigPu(Np),SigSu(Np),SigPd(Np),SigSd(Np)
      Complex*16  Eas0,Ebs0,Easf,Ebsf
      Complex*16  Ndi11,Ndi21,etai,netai
      Complex*16  pi,twomup,epsa,epsb,Ciwh,Ciwx,Ciwz,Ciwhmz
      Complex*16  Ex,bsq,rbsq
******************************************************************
 
      C1=(1.D0,0.D0)	
      C0=(0.D0,0.D0)
      Ci=(0.D0,1.D0)
 
**************************************************
      Ciwh=CI*w*h(0)
      Ciwx=CI*w*Xs
      Ciwz=CI*w*Zs
      Ciwhmz=CI*w*(h(0)-Zs)
      bsq=beta*beta
      rbsq=rho*bsq
 
      DO   100 IP=1,Np
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
*         PLANE WAVE SOURCE down P(p=0.D0)
	        SigPu(ip)=C0
			SigSu(ip)=C0
			SigPd(ip)=Eas0
			SigSd(ip)=C0
        Endif
  100 continue
********************************************************
**************************************************
**************************************************
************************************************
      DO 99999 IP=1,NP
 
*       B1=Rd(0)*SD
        B11=Rd11(ip,0)*SigPD(ip)+Rd12(ip,0)*SigSD(ip)
        B12=Rd21(ip,0)*SigPD(ip)+Rd22(ip,0)*SigSD(ip)
 
*       B2=Td(0)*SD
        B21=Td11(ip,0)*SigPD(ip)+Td12(ip,0)*SigSD(ip)
        B22=Td21(ip,0)*SigPD(ip)+Td22(ip,0)*SigSD(ip)
 
*       P(0)=B1+T*B2
        P1(ip,0)=B11+T11(ip,0)*B21+T12(ip,0)*B22
        P2(ip,0)=B12+T21(ip,0)*B21+T22(ip,0)*B22
 
 
*       Q(0)=U*B2
        Q1(ip,0)=U11(ip,0)*B21+U12(ip,0)*B22
        Q2(ip,0)=U21(ip,0)*B21+U22(ip,0)*B22
 
99999 continue
**************************************************
 
 
*     set Df=downgoing from Free surface
 
      DO 99998 IP=1,NP
*       Uf=U(-1)^(-1)*(SigU+E0*P0)
 
*        direct component
        Uf1(ip)=U11(ip,-1)*SigPU(ip)+U12(ip,-1)*SigSU(ip)
        Uf2(ip)=U21(ip,-1)*SigPU(ip)+U22(ip,-1)*SigSU(ip)
 
        U11(ip,-1)=U11(ip,-1)*Ea(ip,0) !rephased for later use
        U21(ip,-1)=U21(ip,-1)*Ea(ip,0) !rephased for later use
 
        U12(ip,-1)=U12(ip,-1)*Eb(ip,0) !rephased for later use
        U22(ip,-1)=U22(ip,-1)*Eb(ip,0) !rephased for later use
 
        Uf1(ip)=Uf1(ip)+U11(ip,-1)*P1(ip,0)+U12(ip,-1)*P2(ip,0)
        Uf2(ip)=Uf2(ip)+U21(ip,-1)*P1(ip,0)+U22(ip,-1)*P2(ip,0)
 
*       Df=Ru(-1)*Uf
        Df1(ip)=Ru11(ip,-1)*Uf1(ip)+Ru12(ip,-1)*Uf2(ip)
        Df2(ip)=Ru21(ip,-1)*Uf1(ip)+Ru22(ip,-1)*Uf2(ip)
 
 
*       N.B. Direct field must be added also
*        Uf1(ip)=Uf1(ip)+SigPu(ip)
*        Uf2(ip)=Uf2(ip)+SigSu(ip)
 
99998 continue
 
*     Resp0=Mu(0)*Uf+Md(0)*Df
      call  setRec(Pvec,Np,ETAmat(1,0),NETAmat(1,0),RHO,
     &      BETA,Df1,Df2,Uf1,Uf2,RespX0,RespZ0)
 
***************************************
      DO 99997 IP=1,NP
*        pause'check point 1'
        ip1=ip
        ip2=ip1+Np
        ip3=ip2+Np
        ip4=ip3+Np
        X(ip1)=Q1(ip,0)+S11(ip,0)*Df1(ip)+S12(ip,0)*Df2(ip)
        X(ip2)=Q2(ip,0)+S21(ip,0)*Df1(ip)+S22(ip,0)*Df2(ip)
        X(ip3)=R11(ip,1)*X(ip1)+R12(ip,1)*X(ip2)
        X(ip4)=R21(ip,1)*X(ip1)+R22(ip,1)*X(ip2)
99997 continue
****************************************
      DO 99996 IL=2,NL
*        pause'check point 2'
	i1=(il-2)*4*Np
	i2=(il-1)*4*Np
	ilm1=il-1
      DO 99995 IP=1,NP
*        pause'check point 3'
          ip1=ip
          ip2=ip1+Np
          ip3=ip2+Np
          ip4=ip3+Np
	  X(i2+ip1)=S11(ip,ilm1)*X(i1+ip1)+
     &                S12(ip,ilm1)*X(i1+ip2)
	  X(i2+ip2)=S21(ip,ilm1)*X(i1+ip1)+
     &                S22(ip,ilm1)*X(i1+ip2)
      X(i2+ip3)=R11(ip,il)*X(i2+ip1)+
     &                R12(ip,il)*X(i2+ip2)
	  X(i2+ip4)=R21(ip,il)*X(i2+ip1)+
     &                R22(ip,il)*X(i2+ip2)
99995 continue
99996 continue
 
      RETURN
      END
 
***************************************************************
      subroutine AtimesV(M,Nl,Nl4,Np,Np4,
     &            R11,R12,R21,R22,S11,S12,S21,S22,
     &            U11,U12,U21,U22,T11,T12,T21,T22,
     &            Ru11,Ru12,Ru21,Ru22,Tu11,Tu12,
     &            Tu21,Tu22,P1,P2,Q1,Q2,Df1,Df2,
     &            Uf1,Uf2,dS,Xwk,SY,PSY,ea,eb,V,X)
***************************************************************
*     X=(1-P*dS)V
***************************************************************
      Implicit    none
      INTEGER     M,Nl,Nl4,Np,Np4,i
      Complex*16  ea(Np,0:Nl),eb(Np,0:Nl),dS(Np4,Np4,Nl)
      Complex*16  Xwk(M),SY(M),PSY(M),V(M),X(M)
 
      Complex*16  Df1(Np),Df2(Np),Uf1(Np),Uf2(Np)
      Complex*16  T11(Np,0:Nl),T12(Np,0:Nl)
      Complex*16  T21(Np,0:Nl),T22(Np,0:Nl)
      Complex*16  U11(Np,-1:Nl),U12(Np,-1:Nl)
      Complex*16  U21(Np,-1:Nl),U22(Np,-1:Nl)
      Complex*16  R11(Np,0:Nl),R12(Np,0:Nl)
      Complex*16  R21(Np,0:Nl),R22(Np,0:Nl)
      Complex*16  S11(Np,0:Nl),S12(Np,0:Nl)
      Complex*16  S21(Np,0:Nl),S22(Np,0:Nl)
 
      Complex*16  Tu11(Np,0:Nl),Tu12(Np,0:Nl)
      Complex*16  Tu21(Np,0:Nl),Tu22(Np,0:Nl)
      Complex*16  Ru11(Np,-1:Nl),Ru12(Np,-1:Nl)
      Complex*16  Ru21(Np,-1:Nl),Ru22(Np,-1:Nl)
      Complex*16  P1(Np,0:Nl),P2(Np,0:Nl)
      Complex*16  Q1(Np,0:Nl),Q2(Np,0:Nl)
***********************************************************
      Integer     il,ioff
      Complex*16  alf,bet

*      type*,'entered AtimesV'
*      pause
*
*     X=V
*
      call ZCOPY(M,V,1,X,1)
 
*     SY=S*V
      alf=(1.D0,0.D0)
      bet=(0.D0,0.D0)

*      Do i=1,M
*        type*,'V(i)=',i,V(i)
*      enddo

      
      DO 99999 IL=1,NL
        ioff=Np4*(il-1)
        call ZGEMV('N',Np4,Np4,alf,dS(1,1,il),
     &        Np4,V(ioff+1),1,bet,SY(ioff+1),1)
99999 continue
*      Do i=1,M
*        type*,'SY(i)=',i,SY(i)
*      enddo
*      pause

      call setPSY(Nl,Nl4,Np,ea,eb,
     &           R11,R12,R21,R22,S11,S12,S21,S22,
     &           U11,U12,U21,U22,T11,T12,T21,T22,
     &           Ru11,Ru12,Ru21,Ru22,Tu11,Tu12,
     &           Tu21,Tu22,P1,P2,Q1,Q2,Df1,Df2,
     &           Uf1,Uf2,Xwk,SY,PSY)

*      Do i=1,M
*        type*,'PSY(i)=',i,PSY(i)
*      enddo
*      pause
 
 
*     X=X-PSY
      alf=(-1.D0,0.D0)
      call ZAXPY(M,alf,PSY,1,X,1)

*      Do i=1,M
*        type*,'X(i)=',i,X(i)
*      enddo
*      pause 'leaving AtimesV'
 
      return
      end

*************************************************************
	
      REAL FUNCTION KTAPER(KS,KLT,KRT,KF,K)
*     APPLIES A TUCKEY WINDOW TO THE SLOWNESS CONTOUR
      REAL KS,PI,KLT,KRT,KF,K
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
 
*      COMPLEX*16 FUNCTION S(OMEGA,OmegaMax)
*     ***************************************************************
*     Fourier transform of Ricker wavelet
*     ****************************************************************
*      COMPLEX OMEGA,CI,Z,OmegaMax,Alfa,T0
*      CI=(0.0,1.0)
*      alfa=.12*omegamax
*      T0=5.5/(ALFA*SQRT(2.0))
*      Z=OMEGA*OMEGA/(4.*ALFA*ALFA)
*      S=Z*CDEXP(-Z+Ci*OMEGA*T0)
*      RETURN
*      END
 
 
 
 
 
 
*************************************************************
      subroutine setRec(Pvec,Np,ETAvec,NETAvec,RHO,BETA,
     &           Df1,Df2,Uf1,Uf2,RespX,RespZ)
      implicit      none
      integer       Np,ip
      Real*8        rho
      Complex*16    RespX(Np),RespZ(Np)
      Complex*16    Df1(Np),Df2(Np),Uf1(Np),Uf2(Np)
      Complex*16    Pvec(Np),etavec(Np),netavec(Np)
      Complex*16    Md11,Md12,Md21,Md22
      Complex*16    twomup,beta
      Complex*16    ci,c1,epsa,epsb
 
      Ci=Dcmplx(0.D0,1.D0)
      C1=Dcmplx(1.D0,0.D0)
 
      DO 99999 IP=1,NP
        twomup=2.D0*rho*beta*beta*pvec(ip)
        epsa=C1/CDsqrt(2.D0*rho*etavec(ip))
        epsb=C1/CDsqrt(2.D0*rho*netavec(ip))
        Md11=Ci*pvec(ip)*epsa
        Md21=Ci*etavec(ip)*epsa
        Md12=-Ci*netavec(ip)*epsb
        Md22=Ci*pvec(ip)*epsb
        RespX(ip)=Md11*(Df1(ip)+Uf1(ip))+
     &           Md12*(Df2(ip)-Uf2(ip))
        RespZ(ip)=Md21*(Df1(ip)-Uf1(ip))+
     &           Md22*(Df2(ip)+Uf2(ip))
99999 continue
      return
      end
      subroutine PSX(M,Nl,Np,Np4,ea,eb,
     &            pvec,etavec,netavec,rho,beta,
     &            U11,U12,U21,U22,T11,T12,T21,T22,
     &            R11,R12,R21,R22,S11,S12,S21,S22,
     &            Ru11,Ru12,Ru21,Ru22,Tu11,Tu12,
     &            Tu21,Tu22,P1,P2,Q1,Q2,Df1,Df2,
     &            Uf1,Uf2,dS,SY,X,RespX,RespZ)
***************************************************************
*     Resp=P*dS*X
***************************************************************
      Implicit    none
      INTEGER     M,Nl,Np,Np4,ip
      Real*8      rho
      complex*16  ea(Np,0:Nl),eb(Np,0:Nl)
      Complex*16  dS(Np4,Np4,Nl)
      Complex*16  SY(M),X(M)
      Complex*16  RespX(Np),RespZ(Np),beta
      Complex*16  Pvec(Np),etavec(Np),netavec(Np)
 
      Complex*16  Df1(Np),Df2(Np),Uf1(Np),Uf2(Np)
      Complex*16  T11(Np,0:Nl),T12(Np,0:Nl)
      Complex*16  T21(Np,0:Nl),T22(Np,0:Nl)
      Complex*16  U11(Np,-1:Nl),U12(Np,-1:Nl)
      Complex*16  U21(Np,-1:Nl),U22(Np,-1:Nl)
      Complex*16  R11(Np,0:Nl),R12(Np,0:Nl)
      Complex*16  R21(Np,0:Nl),R22(Np,0:Nl)
      Complex*16  S11(Np,0:Nl),S12(Np,0:Nl)
      Complex*16  S21(Np,0:Nl),S22(Np,0:Nl)
 
      Complex*16  Tu11(Np,0:Nl),Tu12(Np,0:Nl)
      Complex*16  Tu21(Np,0:Nl),Tu22(Np,0:Nl)
      Complex*16  Ru11(Np,-1:Nl),Ru12(Np,-1:Nl)
      Complex*16  Ru21(Np,-1:Nl),Ru22(Np,-1:Nl)
      Complex*16  P1(Np,0:Nl),P2(Np,0:Nl)
      Complex*16  Q1(Np,0:Nl),Q2(Np,0:Nl)
***********************************************************
      Integer     il,ioff
      Complex*16  alf,bet
 
*     SY=S*X
      alf=(1.D0,0.D0)
      bet=(0.D0,0.D0)
      DO 99999 IL=1,NL
        ioff=Np4*(il-1)
        call ZGEMV('N',Np4,Np4,alf,dS(1,1,il),
     &        Np4,X(ioff+1),1,bet,SY(ioff+1),1)
99999 continue

 
      call  UpSweep(Np,Nl,SY,Ru11,Ru12,Ru21,Ru22,
     &           Tu11,Tu12,Tu21,Tu22,ea,eb,T11,T12,T21,
     &           T22,U11,U12,U21,U22,R11,R12,R21,R22,
     &           S11,S12,S21,S22,P1,P2,Q1,Q2)
 
*     set Df=downgoing from Free surface
*     set Uf=upgoing at Free surface
 
      DO 99998 IP=1,NP
*       Uf=U(-1)^(-1)*P0
        Uf1(ip)=U11(ip,-1)*P1(ip,0)+U12(ip,-1)*P2(ip,0)
        Uf2(ip)=U21(ip,-1)*P1(ip,0)+U22(ip,-1)*P2(ip,0)
 
*       Df=Ru(-1)*Uf
        Df1(ip)=Ru11(ip,-1)*Uf1(ip)+Ru12(ip,-1)*Uf2(ip)
        Df2(ip)=Ru21(ip,-1)*Uf1(ip)+Ru22(ip,-1)*Uf2(ip)

*       Plane wave Response ****DELETE
*       P-wave
*        RespX(ip)=Uf1(ip)
*       S-wave
*        RespZ(ip)=Uf2(ip)
99998 continue
 
      call  setRec(Pvec,Np,ETAvec,NETAvec,RHO,BETA,
     &           Df1,Df2,Uf1,Uf2,RespX,RespZ)
      return
      end


      subroutine BornScat(Nk,Nk2,Nk4,pvec,etavec,netavec,
     & w,h,xleft,xright,rho0,alpha0,beta0,rho,alpha,beta,
     & dS)
*******************************************************************
*     set the layer single scattering operators
*******************************************************************
      Implicit   none
      Integer    Nk,Nk2,Nk4
      Complex*16 pvec(Nk),pi,po,w
      Complex*16 dS(Nk4,Nk4)
      Complex*16 etavec(Nk),netavec(Nk)
 
      Real*8     rho0,rho,h,dk
      Complex*16 alpha0,beta0,alpha,beta
      Complex*16 alphasq0,betasq0,nu10
      Complex*16 alphasq,betasq,nu1
 
      Complex*16 Ciw,Ciwh,Ciwh2
      integer    ipi,ipo
      integer    ki1,ki2,ki3,ki4
      integer    ko1,ko2,ko3,ko4
 
      Complex*16 twomup,epsa,epsb
 
      Complex*16 Mdi11,Mdi12,Mdi21,Mdi22
      Complex*16 Ndi11,Ndi12,Ndi21,Ndi22
      Complex*16 Mdo11,Mdo12,Mdo21,Mdo22
      Complex*16 Ndo11,Ndo12,Ndo21,Ndo22
 
      Complex*16 Mui11,Mui12,Mui21,Mui22
      Complex*16 Nui11,Nui12,Nui21,Nui22
      Complex*16 Muo11,Muo12,Muo21,Muo22
      Complex*16 Nuo11,Nuo12,Nuo21,Nuo22
 
      Complex*16 dA21,dA34,dA13,dA24,dA31,dA42
      Complex*16 Ci,C1,C0,eai,ebi,eao,ebo
      Complex*16 couple,formfac
      Complex*16 etai,netai,etao,netao
      Complex*16 sum21,sum24,sum13,sum42,sum31
********************************************************
      Real*8     PPi,xleft,xright,L,xmd2,xpd2,lambda
      parameter  (lambda=6.0)
      parameter  (PPi=3.14159265358979324D0)
      Real*8     e1,temp1
      Complex*16 e2

      Ci=(0.D0,1.D0)
      C0=(0.D0,0.D0)
      C1=(1.D0,0.D0)
*      xleft=0.5D0,xright=1.5D0  !input
      L=(xright-xleft)
      xmd2=0.5D0*L
      xpd2=0.5D0*(xright+xleft)
 
      Ciw=Ci*w
      Ciwh=Ciw*h
      Ciwh2=0.5D0*Ciwh
      alphasq0=alpha0*alpha0
      betasq0=beta0*beta0
      nu10=4.D0*betasq0*(1.D0-betasq0/alphasq0)
      alphasq=alpha*alpha
      betasq=beta*beta
      nu1=4.D0*betasq*(1.D0-betasq/alphasq)
      sum21=2.D0*(betasq/alphasq-betasq0/alphasq0)
      sum24=(1.D0/(rho*alphasq)-1.D0/(rho0*alphasq0))
      sum13=(1.D0/(rho*betasq)-1.D0/(rho0*betasq0))
      sum42=(rho-rho0)
      sum31=(rho*nu1-rho0*nu10)
 
      DO    87 IPI=1,NK
      Ki1=ipi
      Ki2=Ki1+Nk
      Ki3=Ki2+Nk
      Ki4=Ki3+Nk
 
      pi=Pvec(ipi)
      etai=etavec(ipi)
      netai=netavec(ipi)
 
      twomup=2.D0*rho0*betasq0*pi
      epsa=C1/CDsqrt(2.D0*rho0*etai)
      epsb=C1/CDsqrt(2.D0*rho0*netai)
 
      Mdi21=Ci*etai*epsa
      Mdi22=Ci*pi*epsb
      Mdi11=Ci*pi*epsa
      Mdi12=-Ci*netai*epsb
 
      Ndi21=-Ci*(twomup*pi-rho0)*epsa
      Ndi12=Ci*(twomup*pi-rho0)*epsb
      Ndi11=Ci*twomup*etai*epsa
      Ndi22=Ci*twomup*netai*epsb
 
      Mui21=-Mdi21
      Mui22=Mdi22
      Mui11=Mdi11
      Mui12=-Mdi12
 
      Nui21=Ndi21
      Nui22=-Ndi22
      Nui11=-Ndi11
      Nui12=Ndi12

      eai=CDExp(Ciwh2*etai)
      ebi=CDExp(Ciwh2*netai)
 
      DO    86 IPO=1,NK
      Ko1=ipo
      Ko2=Ko1+Nk
      Ko3=Ko2+Nk
      Ko4=Ko3+Nk
      po=Pvec(ipo)
      etao=etavec(ipo)
      netao=netavec(ipo)


      eao=CDExp(Ciwh2*etao)
      ebo=CDExp(Ciwh2*netao) 
      type*,'Ciwh2,etao,eao,ebo=',Ciwh2,etao,eao,ebo

*
*     **************************************************
*     set the Pi->Po coupling elements  Ciw*<Po|dA|Pi>
*     ***************************************************
*
      dk=Dble(w*(po-pi))
*
*     Compute formfactor
*
      If(Dabs(dk).gt.1.D-10)then
        e1=Dexp(0.5D0*PPi*dk/lambda)
        e2=CDexp(Dcmplx(0.D0,-dk*xpd2))
        temp1=Dsin(xmd2*dk)*e1/(e1**2-1.D0)/lambda
        couple=Dcmplx(temp1*DBLE(e2),temp1*Dimag(e2))
*        couple=Dcmplx(0.D0,0.D0) !delete
      else
*        couple=Dcmplx(L)
        couple=Dcmplx(L/(2.D0*PPi),0.D0) !delete
      Endif
 
      formfac=Ciwh*couple
 
************************************************************
 
      dA21=pi*sum21*formfac
      dA34=po*sum21*formfac
      dA24=sum24*formfac
      dA13=sum13*formfac
      dA42=sum42*formfac
      dA31=dA42-pi*po*sum31*formfac
*      type*,'pi,po=',pi,po
*      type*,'formfac,dA21,dA34,dA24,dA13,dA42,dA31=',formfac,dA21,dA34,dA24,dA13,dA42,dA31
 
*     ************************************************************
*     set the first-Born scattering elements for Pi->Po transition
*     ************************************************************
 
 
      twomup=2.D0*rho0*betasq0*po
      epsa=C1/CDsqrt(2.D0*rho0*etao)
      epsb=C1/CDsqrt(2.D0*rho0*netao)
 
      Mdo21=Ci*etao*epsa
      Mdo22=Ci*po*epsb
      Mdo11=Ci*po*epsa
      Mdo12=-Ci*netao*epsb
 
      Ndo21=-Ci*(twomup*po-rho0)*epsa
      Ndo12=Ci*(twomup*po-rho0)*epsb
      Ndo11=Ci*twomup*etao*epsa
      Ndo22=Ci*twomup*netao*epsb
 
      Muo21=-Mdo21
      Muo22=Mdo22
      Muo11=Mdo11
      Muo12=-Mdo12
 
      Nuo21=Ndo21
      Nuo22=-Ndo22
      Nuo11=-Ndo11
      Nuo12=Ndo12
 
      dS(Ko1,Ki1)=(-(Ndi21*dA34+Mdi11*dA31)*Mdo11+
     &      (-Ndi21*dA24-Mdi11*dA21)*Ndo21+Mdi21*
     &      Mdo21*dA42+Ndi11*Ndo11*dA13)*eao*eai

      dS(Ko1,Ki2)=(-(Ndi22*dA34+Mdi12*dA31)*Mdo11+(-Ndi22*dA24-
     &      Mdi12*dA21)*Ndo21+Mdi22*Mdo21*dA42+
     &      Ndi12*Ndo11*dA13)*eao*ebi

      dS(Ko1,Ki3)=(-(Nui21*dA34+Mui11*dA31)*Mdo11+(-Nui21*dA24-
     &     Mui11*dA21)*Ndo21+Mui21*Mdo21*dA42+Nui11*
     &     Ndo11*dA13+C1)*eao*eai

      dS(Ko1,Ki4)=(-(Nui22*dA34+Mui12*dA31)*Mdo11+(-Nui22*dA24-
     &      Mui12*dA21)*Ndo21+Mui22*Mdo21*dA42+Nui12*
     &      Ndo11*dA13)*eao*ebi

      dS(Ko2,Ki1)=((Ndi21*dA34+Mdi11*dA31)*Mdo12-(-Ndi21*dA24-
     &      Mdi11*dA21)*Ndo22-Mdi21*Mdo22*dA42-Ndi11*
     &      Ndo12*dA13)*ebo*eai

      dS(Ko2,Ki2)=((Ndi22*dA34+Mdi12*dA31)*Mdo12-(-Ndi22*dA24-
     &     Mdi12*dA21)*Ndo22-Mdi22*Mdo22*dA42-Ndi12*
     &     Ndo12*dA13)*ebo*ebi

      dS(Ko2,Ki3)=((Nui21*dA34+Mui11*dA31)*Mdo12-(-Nui21*dA24-
     &     Mui11*dA21)*Ndo22-Mui21*Mdo22*dA42-Nui11*
     &     Ndo12*dA13)*ebo*eai

      dS(Ko2,Ki4)=((Nui22*dA34+Mui12*dA31)*Mdo12-(-Nui22*dA24-
     &     Mui12*dA21)*Ndo22-Mui22*Mdo22*dA42-Nui12*
     &     Ndo12*dA13+C1)*ebo*ebi

      dS(Ko3,Ki1)=(-(Ndi21*dA34+Mdi11*dA31)*Muo11+(-Ndi21*dA24-
     &     Mdi11*dA21)*Nuo21+Mdi21*Muo21*dA42+Ndi11*
     &     Nuo11*dA13+C1)*eao*eai

 
      dS(Ko3,Ki2)=(-(Ndi22*dA34+Mdi12*dA31)*Muo11+(-Ndi22*dA24-
     &     Mdi12*dA21)*Nuo21+Mdi22*Muo21*dA42+Ndi12*
     &     Nuo11*dA13)*eao*ebi

 
      dS(Ko3,Ki3)=(-(Nui21*dA34+Mui11*dA31)*Muo11+(-Nui21*dA24-
     &     Mui11*dA21)*Nuo21+Mui21*Muo21*dA42+Nui11*
     &     Nuo11*dA13)*eao*eai

 
      dS(Ko3,Ki4)=(-(Nui22*dA34+Mui12*dA31)*Muo11+(-Nui22*dA24-
     &     Mui12*dA21)*Nuo21+Mui22*Muo21*dA42+Nui12*
     &     Nuo11*dA13)*eao*ebi

      dS(Ko4,Ki1)=((Ndi21*dA34+Mdi11*dA31)*Muo12-(-Ndi21*dA24-
     &    Mdi11*dA21)*Nuo22-Mdi21*Muo22*dA42-Ndi11*
     &    Nuo12*dA13)*ebo*eai

      dS(Ko4,Ki2)=((Ndi22*dA34+Mdi12*dA31)*Muo12-(-Ndi22*dA24-
     &    Mdi12*dA21)*Nuo22-Mdi22*Muo22*dA42-Ndi12*
     &    Nuo12*dA13+C1)*ebo*ebi

      dS(Ko4,Ki3)=((Nui21*dA34+Mui11*dA31)*Muo12-(-Nui21*dA24-
     &    Mui11*dA21)*Nuo22-Mui21*Muo22*dA42-Nui11*
     &    Nuo12*dA13)*ebo*eai

      dS(Ko4,Ki4)=((Nui22*dA34+Mui12*dA31)*Muo12-(-Nui22*dA24-
     &    Mui12*dA21)*Nuo22-Mui22*Muo22*dA42-Nui12*
     &    Nuo12*dA13)*ebo*ebi
 
   86 continue
   87 continue
      return
      End

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
