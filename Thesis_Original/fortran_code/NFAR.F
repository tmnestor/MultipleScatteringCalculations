
      program NFAR
      implicit none
      integer outfile,nrmax,nlp1max,nwmax,nlp1,nr,nw,nt,iw,ir,il,np
	  INTEGER NWM,IP,IT,II
	  real    DR,DS,FRL,DBR,TAU,PS,PF,PLT,PRT,DP34,F0,T,VR,DT,TSHIFT,TT
	  REAL    TPI,PI2,PI4,W0,TMIN,TMAX,EPS,FMIN,FMAX,HANN,RL,TC,BIG,AM
	  PARAMETER(outfile=11,NRMAX=1,NLP1MAX=3,NWMAX=512)
      REAL    UT(NWMAX),TIM(NWMAX),CA0(NLP1MAX),CB0(NLP1MAX)
	  REAL    QA0(NLP1MAX),QB0(NLP1MAX),H(NLP1MAX),RHO(NLP1MAX)
      COMPLEX A(NLP1MAX),B(NLP1MAX),ALFA(NLP1MAX),BETA(NLP1MAX)
      COMPLEX P2,DP,W,WSQ,DW,ETA(NLP1MAX),NETA(NLP1MAX),WMAX
      COMPLEX Ci,C0,C1,CPS,CPLT,CPRT,CPF,LAMBDAS,WFACTOR,S
      COMPLEX CSIG(NRMAX),EMPI4,E1(NRMAX),E2(NRMAX)
      COMPLEX DU,U1,U2,UU(NRMAX),CMPLXSLO,VERTICALSLO
      COMPLEX U(NWMAX/2,NRMAX),UZ(NWMAX),ZLAG
	  character *1 TAB
	  PARAMETER (Ci=(0.0,1.0),C1=(1.0,0.0),C0=(0.0,0.0))
	  open(outfile,file='NFARgraphdat',status='new')
	  TAB=char(9)
!********************************************************************
!     INPUT LAYER PARAMETERS
!********************************************************************
      NLP1=2
	  IF(NLP1 .GT. NLP1MAX) THEN
	    PAUSE 'TOO MANY LAYERS'
	  END IF
	  CA0(1)=1.5
	  CB0(1)=0.0
	  QA0(1)=1E10
	  QB0(1)=1E10
	  RHO(1)=1.0
	  H(1)=7.5
	  CA0(2)=1.6
	  CB0(2)=0.3
	  QA0(2)=1E10
	  QB0(2)=1E10
	  RHO(2)=2.0
	  H(2)=23.5
	  CA0(3)=3.3
	  CB0(3)=1.5
	  QA0(3)=1E10
	  QB0(3)=1E10
	  RHO(3)=2.0
	  H(3)=1.E10
	  
!**********************************************************************
!     INPUT THE EXPERIMENTAL GEOMETRY
!**********************************************************************
      NR=1
	  IF(NR .GT. NRMAX) THEN
	    PAUSE 'TOO MANY RECEIVERS'
	  ENDIF
      DR=0.0
      DS=0.0
      FRL=10.0
      DBR=20.0
	  TAU=2*H(1)-DS-DR
!*********************************************************************
!     INPUT SLOWNESS CONTOUR INFORMATION
!*********************************************************************
      PS=0.2
	  PF=4.0
	  PLT=1.0/3.0
	  PRT=10.0/3.0
	  DP34=.0008
	  
	  
      F0=1.0
      T=180.0
      NW=128
      NT=2*NW
      VR=2.0
      DT=T/FLOAT(NT)
      TSHIFT=0.0
      TPI=2.0*PI	  
      PI2=PI/2.
      PI4=PI/4.
      W0=TPI*F0
      DW=TPI/T
      EMPI4=CEXP(CMPLX(0.0,-PI4))
      NWM=NW-1
      TMIN=1/FLOAT(NWM)
      TMAX=T
      FMIN=1/T
	  WMAX=NW*DW
*     max frequency index NW corresponds to frequency limit of source	  
      FMAX=FLOAT(NWM)/T
*     define frequency damping factor
      eps=log(20.0)/T
*     define starting frequency (could have small pos. imag part)	  
	  
      W=cmplx(0.0,0.0)
!**********************************************************************	  
!     loop over frequency pass-band of source
!**********************************************************************
      DO 800 IW=1,NWM
      W=W+DW
      WRITE (6,*) 'FREQUENCY=',IW,W
	  WFACTOR=CMPLX(1,-eps/REAL(W))
	  CPS=PS*WFACTOR
	  CPLT=PLT*WFACTOR
	  CPRT=PRT*WFACTOR
	  CPF=PF*WFACTOR
      DO 25 IR=1,NR
*     compute CSIG(ir)=iwr for each receiver range r
      CSIG(IR)=W*CMPLX(0.0,(FRL+FLOAT(IR-1)*DBR))
      UU(IR)=C0
 25   CONTINUE
      WSQ=W*W
*     define complex frequency dependent slownesses of each layer
      A(1)=CMPLXSLO(W,CA0(1),QA0(1))
	  B(1)=C0
	  ALFA(1)=C1/A(1)
*	  TYPE*,'A(1)=',A(1)
*	  PAUSE
	  BETA(1)=C0
      DO IL=2,NLP1
        A(IL)=CMPLXSLO(W,CA0(IL),QA0(IL))
		B(IL)=CMPLXSLO(W,CB0(IL),QB0(IL))
*		TYPE *,'A(IL),B(IL)=',A(IL),B(IL)
*		PAUSE
		ALFA(IL)=C1/A(IL)
	    BETA(IL)=C1/B(IL)
	  END DO
*     DEFINE THE SLOWNESS CONTOUR INTEGRATION STEP
      P2=CPS
      NP=INT(1.5*CABS(CPF-CPS)*(cabs(W/W0)**.95)/DP34)
      IF(NP .LT. 30) NP=30
      DP=(CPF-CPS)/CMPLX(FLOAT(NP),0.0)
*     DELETE THESE THREE STATEMENTS AFTER TESTING
      P2=0.0
      CALL SLOWRESP(P2,W,ETA,NETA,U2,E2,A,B,ALFA,BETA,RHO,H,TAU,
     +	                  NR,NLP1,CPS,CPLT,CPRT,CPF,CSIG)
      UU(1)=U2
      TYPE*,'U=',UU(1)
*	  
!*********************************************************************
!     BEGIN INTEGRATION OVER THE SLOWNESS CONTOUR
!**********************************************************************
*     TYPE *,'P2=',P2
*	  PAUSE
*      CALL SLOWRESP(P2,W,ETA,NETA,U2,E2,A,B,ALFA,BETA,RHO,H,TAU,
*     +	                  NR,NLP1,CPS,CPLT,CPRT,CPF,CSIG)

*     TYPE *,'U2=',U2
*	  PAUSE
*      DO 730 IP=1,NP
*      U1=U2
*      DO 736 IR=1,NR
*      E1(IR)=E2(IR)
* 736  CONTINUE
*      P2=P2+DP
*	  TYPE *,'P2=',P2
*	  PAUSE
*      CALL SLOWRESP(P2,W,ETA,NETA,U2,E2,A,B,ALFA,BETA,RHO,H,TAU,
*     +	                  NR,NLP1,CPS,CPLT,CPRT,CPF,CSIG)
*     TYPE *,'U2=',U2
*	  PAUSE
*     ACCUMULATE THE TRANSFER FUNCTION AT EACH RECEIVER USING FILON'S RULE
*      DO 738 IR=1,NR
*	  TYPE *,'CSIG(IR),E2(',IR,')=',CSIG(IR),E2(IR)
*	  PAUSE
*      DU=U2*E2(IR)-U1*E1(IR)-(E2(IR)-E1(IR))*(U2-U1)/(CSIG(IR)*DP)
*      DU=DU/CSIG(IR)
*      UU(IR)=UU(IR)+DU
*	  TYPE*,'DU,UU=',DU,UU
*	  PAUSE
* 738  CONTINUE
* 730  CONTINUE
!*************************************************************************
!     END INTEGRATION OVER SLOWNESS CONTOUR
! ************************************************************************

*     apply slowness independent terms
	  ZLAG=0.5*S(w,wmax)*(1.0+COS(PI*FLOAT(IW)/FLOAT(NW)))
*     ZLAG=ZLAG*EMPI4*WSQ/csqrt(W)/sqrt(PI2)
*     Define bulk modulus at the source
*      LAMBDAS=CMPLX(RHO(1),0.0)/(A(1)*A(1))
*      smooth pressure response by dividing by -iw	
*      ZLAG=Ci*ZLAG/(LAMBDAS*W)
      DO 780 IR=1,NR
      RL=FRL+FLOAT(IR-1)*DBR
*     incorporate phase change to effect the reduced time scale
      U(IW,IR)=UU(IR)*ZLAG*CEXP(W*CMPLX(0.0,(-RL/VR+TSHIFT)))
*      U(IW,IR)=UU(IR)*ZLAG
*	  TYPE *,'ZLAG=',ZLAG
*	  PAUSE
*	  TC=TPI/W
*	  TYPE *,TC,NP
*	  TYPE*
 780  CONTINUE
 800  CONTINUE


!**********************************************************************
!     END FREQUENCY LOOP
!**********************************************************************
 
      TYPE *,'LEFT FREQUENCY LOOP'
*     fourier transform each transfer function to give the time series
      DO 1000 IR=1,NR
      RL=FRL+FLOAT(IR-1)*DBR
      DO 1001 IW=1,NWM
      UZ(IW+1)=U(IW,IR)
      UZ(NT-IW+1)=CONJG(U(IW,IR))
1001  CONTINUE
      UZ(1)=C0
      UZ(NW+1)=C0
      CALL FFT(UZ,NT,-1.0)
	  DO 1002 IT=1,NT
        UT(IT)=REAL(UZ(IT))
		type*,'FFT output',UT(IT)
1002  CONTINUE
      BIG=0.0
	  DO 1003 II=1,NT
	  AM=ABS(UT(II))
	  IF(AM .GT. BIG) BIG=AM
1003  CONTINUE
      BIG=2./BIG
	  RL=FRL + FLOAT(IR-1)*DBR
	  AM=BIG*((RL*RL+(DS-DR)**2)**.50)/SQRT(RL)
	  AM=AM*SQRT(FRL)/(SQRT(FRL**2+(DS-DR)**2))
      TT=RL/VR-TSHIFT
*      TT=0.0
      DO 1030 II=1,NT
      TIM(II)=FLOAT(II-1)*DT+TT
*     compensate for the frequency damping
*	  UT(II)=AM*UT(II)*exp(eps*TIM(II))
	  UT(II)=UT(II)*exp(eps*TIM(II))
	  WRITE (outfile,999) TIM(II),TAB,UT(II)
	  type*,'t,u(t)=',TIM(II),UT(II)
1030  CONTINUE    
1000  CONTINUE
 999  FORMAT(f5.1,A1,g10.4) 
      close(outfile)
      STOP
      END
	  
      SUBROUTINE SLOWRESP(P,W,ETA,NETA,U,E,A,B,ALFA,BETA,RHO,H,TAU,
     +	                  NR,NLP1,CPS,CPLT,CPRT,CPF,CSIG)
!**********************************************************************
!     CALCULATE THE HORIZONTEL SLOWNESS DEPENDENT COMPONENTS OF THE
!     REFLECTIVITY INTEGRAND
!**********************************************************************
      INTEGER NLP1,NR,IR,IL
      COMPLEX P,W,U,E2(NR),A(NLP1),B(NLP1),ALFA(NLP1),BETA(NLP1)
	  COMPLEX ETA(NLP1),NETA(NLP1),PDPU,CPS,CPLT,CPRT,CPF,CSIG(NR)
	  COMPLEX Ci,C1,C0,OB,PHASE
	  PARAMETER (Ci=(0.0,1.0),C1=(1.0,0.0),C0=(0.0,0.0))
	  REAL RHO(NLP1),H(NLP1),TAU 

*     CALCULATE THE VERTICAL SLOWNESS OF P&S WAVES (ETA,NETA RESPECTIVELY)
      ETA(1)=VERTICALSLO(A(1),P)
	  TYPE*,'ETA(',1,')=',ETA(1)
*	  PAUSE
      NETA(1)=C0
      DO IL=2,NLP1
	    ETA(IL)=VERTICALSLO(A(IL),P)
		TYPE*,'ETA(',IL,')=',ETA(IL)
*		PAUSE
		NETA(IL)=VERTICALSLO(B(IL),P)
		TYPE*,'NETA(',IL,')=',NETA(IL)
*		PAUSE
	  END DO
*     CALCULATE THE OCEAN BOTTOM REFLECTIVITY
*      PDPU=REFLEX1(ETA(1),RHO(1),ETA(2),RHO(2))
      PdPu=Riccatti_Reflex(nlp1,p,w,rho,h,eta,neta,beta)
      
	  TYPE*,'RHO(1),RHO(2),PDPU=',RHO(1),RHO(2),PDPU
*	  PAUSE
*	  U=PDPU*CSQRT(P)*CEXP(Ci*W*ETA(1)*TAU)/ETA(1)
*	  U=U*CMPLX(PTAPER(CPS,CPLT,CPRT,CPF,P),0.0)
*      DO 10 IR=1,NR
*	    E(IR)=CEXP(CSIG(IR)*P)
*10  CONTINUE
      PHASE=Ci*W*ETA(1)*TAU
	  TYPE*,'PHASE=',PHASE
	  PAUSE
      U=PDPU*CEXP(PHASE)
      RETURN
      END
	  

	  
	  REAL FUNCTION PTAPER(CPS,CPLT,CPRT,CPF,CP)
*     APPLIES A TUCKEY WINDOW TO THE SLOWNESS CONTOUR
	  COMPLEX CPS,CPLT,CPRT,CPF,CP
	  REAL AVCP,AVCPLT,AVCPS,AVCPRT,AVCPF
	  AVCP=CABS(CP)
	  AVCPS=CABS(CPS)
	  AVCPLT=CABS(CPLT)
	  AVCPRT=CABS(CPRT)
	  AVCPF=CABS(CPF)
	  IF(AVCP .GE. AVCPS .AND. AVCP .LE. AVCPLT) THEN
	    PTAPER=.5*(1.+COS(PI*(AVCPLT-AVCP)/(AVCPLT-AVCPS)))
	  ELSE IF(AVCP .GT. AVCPLT .AND. AVCP .LT. AVCPRT) THEN
	    PTAPER=1.0
	  ELSE IF(AVCP .GE. AVCPRT .AND. AVCP .LE. AVCPF )THEN
	    PTAPER=0.5*(1+COS(PI*(AVCP-AVCPRT)/(AVCPF-AVCPRT)))
	  ELSE
	    PTAPER=0.0
	  END IF
	  RETURN
	  END
	  
	  COMPLEX FUNCTION CMPLXSLO(W,C0,Q0)
*     DETERMINE THE FREQUENCY DEPENDENT RECIPROCAL PROPAGATION 
*     VELOCITY (SLOWNESS) ACCORDING TO THE POWER LAW OF STRICK
	  REAL C0,Q0,EPS,W0,SIGMA,T1
	  COMPLEX W,T,CW
	  PARAMETER(W0=1.0,SIGMA=0.1,EPS=0.001)
	  T=CEXP(-SIGMA*CLOG(CMPLX(EPS+AIMAG(W),-REAL(W))))
	  T1=W0**SIGMA/SIN(SIGMA*PI/2)
	  CW=CMPLX(0.5*T1/Q0/C0,0.0)*T
	  CW=CW+CMPLX(1.0/C0,0.0)
	  CMPLXSLO=CW
	  RETURN
	  END
	  
	  COMPLEX FUNCTION VERTICALSLO(C,P)
	  COMPLEX C,P,T
	  T=(C+P)*(C-P)
	  VERTICALSLO=csqrt(T)
	  IF(AIMAG(VERTICALSLO) .LE. 0.0)THEN 
	    VERTICALSLO=-VERTICALSLO
	  END IF
	  RETURN
	  END
	  
      
      COMPLEX FUNCTION REFLEX1(ETA1,RHO1,ETA2,RHO2)
	  REAL RHO1,RHO2
	  COMPLEX ETA1,ETA2,TEMP,TOP,BOT
	  TEMP=cmplx(RHO1/RHO2,0.0)*ETA2
	  TOP=ETA1-TEMP
	  BOT=ETA1+TEMP
	  REFLEX=TOP/BOT
	  RETURN
	  END
	  
	  
	  SUBROUTINE SCATMAT(P,ETA1,NETA1,RHO1,BETA1,ETA2,NETA2,RHO2,BETA2,
     +                   Rd,Tu,Td,Ru)
*     modified interfacial scattering matrices (see Aki&Richards pp149-51)
	  REAL RHO1,RHO2,rtrho1,rtrho2
	  COMPLEX P,ETA1,NETA1,ETA2,NETA2,BETA1,BETA2,PSQ,CRHO1,CRHO2,DRHO,DMU
	  COMPLEX PSQDMU,Rd(2,2),Td(2,2),Ru(2,2),Tu(2,2),a,b,c,d,Det,E,F,G,H
	  complex rteta1,rteta2,rtneta1,rtneta2,Ci,mCi
	  Ci=(0.0,1.0)
	  mCi=(0.0,-1.0)
	  rtrho1=sqrt(rho1)
	  rtrho2=sqrt(rho2)
	  rteta1=csqrt(eta1)
	  rteta2=csqrt(eta2)
	  psq=p*p
	  crho1=cmplx(rho1,0.0)
	  crho2=cmplx(rho2,0.0)
	  drho=crho2-crho1
	  dmu=crho2*beta2*beta2-crho1*beta1*beta1
	  psqdmu=psq*dmu
	  d=2.0*dmu
	  a=drho-psqdmu
	  b=crho2-psqdmu
	  c=crho1-psqdmu
	  E=b*eta1+c*eta2
	  F=b*neta1+c*neta2
	  G=a-d*eta1*neta2
	  H=a-d*eta2*neta1
	  Det=E*F+G*H*psq
	  Rd(1,1)=((b*eta1-c*eta2)*F-(a+d*eta1*neta2)*H*psq)/Det
	  Rd(2,1)=2*mCi*rteta1*rtneta1*(a*b+c*d*eta2*neta2)*p/Det
	  Rd(1,2)=Rd(2,1)
	  Rd(2,2)=((b*neta1-c*neta2)*E-(a+d*eta2*neta1)*G*psq)/Det
	  Td(1,1)=2*rtrho1*rtrho2*rteta1*rteta2*F/Det
	  Td(2,1)=2.0*mCi*rtrho1*rtrho2*rteta1*rtneta2*H*p/Det
	  Td(1,2)=2*mCi*rtrho1*rtrho2*rtneta1*rteta2*G*p/Det
	  Td(2,2)=-2*rtrho1*rtrho2*rtneta1*rtneta2*E/Det
	  Tu(1,1)=Td(1,1)
	  Tu(1,2)=Td(2,1)
	  Tu(2,1)=Td(1,2)
	  Tu(2,2)=Td(2,2)
	  Ru(1,1)=-((b*eta1-c*eta2)*F+(a+d*eta2*neta1)*G*psq)/Det
	  Ru(2,1)=2*mCi*rteta2*rtneta2*(a*c+b*d*eta1*neta1)*p/Det
	  Ru(1,2)=Ru(2,1)
	  Ru(2,2)=-((b*neta1-c*neta2)*E+(a+d*eta1*neta2)*H*psq)/Det
	  return
	  end
	  
	  subroutine SpecialInv(a,b,Inv)
*     Inv=(I-A*B)^-1
	  complex a(2,2),b(2,2),Inv(2,2),c1,det
	  c1=(1.0,0.0)
	  Inv(1,1)=c1-a(2,1)*b(1,2)-a(2,2)*b(2,2)
	  Inv(1,2)=a(1,1)*b(1,2)+a(1,2)*b(2,2)
	  Inv(2,1)=a(2,1)*b(1,1)+a(2,2)*b(2,1)
	  inv(2,2)=c1-a(1,1)*b(1,1)-a(1,2)*b(2,1)
	  Det=Inv(1,1)*Inv(2,2)-Inv(1,2)*Inv(2,1)
	  Inv(1,1)=Inv(1,1)/Det
	  Inv(1,2)=Inv(1,2)/Det
	  Inv(2,1)=Inv(2,1)/Det
	  Inv(2,2)=Inv(2,2)/Det
	  return
	  end
	  
	  subroutine matmult(a,u,t)
*     form complex 2x2 matrix T=A*U
      complex t(2,2),a(2,2),u(2,2),c0
	  c0=(0.0,0.0)
	  do i=1,2
	    do j=1,2
		  t(i,j)=c0
		End Do
	  End do
	  do i=1,2
	    do j=1,2
		  do k=1,2
		    t(i,j)=a(i,k)*u(k,j)+t(i,j)
		  end do
		end do
	  end do
	  return
	  end
	  
	  complex function reflex(nl,p,w,rho,h,eta,neta,beta)
*     compute the ratio of the upgoing to the downgoing wavefield
*     of a plane layer stack using Kennett's recursive algorithm
      integer i,j,nl
      real H(n),rho(n)
      complex p,w,eta(n),neta(n),beta(n),Wrk1(2,2),Wrk2(2,2),RRt(2,2)
      complex Rd(2,2),Tu(2,2),Td(2,2),Ru(2,2),RRb(2,2),c0,RRd(2,2),RR
	  complex PdPu,PuPu,PdPd,SuPu,PdSd
	  c0=(0.0,0.0)
*     impose the lower half space radiation condition
	  do i=1,2
	    do j=1,2
		  RRt(i,j)=c0
		end do
	  end do
	  do il=nl,2,-1
*     use kennet's recursive algorithm to build the reflectivity from
*     the base of the elastic layer stack to the top of layer 2 (i.e.
*     just below the ocean bottom)
	    ilp1=il+1
	    call ScatMat(p,eta(il),neta(il),rho(il),beta(il),eta(ilp1),
     +               neta(ilp1),rho(ilp1),beta(ilp1),Rd,Tu,Td,Ru)
	    call SpecialInv(Ru,RRt,Wrk1)
		call MatMult(Wrk1,Td,Wrk2)
		call MatMult(RRt,Wrk2,Wrk1)
		call MatMult(Tu,Wrk1,Wrk2)
		call MatAdd(Rd,Wrk2,RRd)
		call phasor(w,h(il),eta(il),neta(il),RRb,RRt)
      End Do
*     generate the reflectivity just above the Ocean Bottom from the
*     reflectivity just below the ocean bottom
      call OBScatMat(p,eta(1),neta(1),rho(1),eta(2),neta(2),
     +     rho(2),beta(2),PdPu,PuPu,PdPd,SuPu,PdSd,Ru)
      call SpecialInv(Ru,RRt,Wrk1)
	  call MatMult(RRt,Wrk1,RRb)
	  RR=PuPu*PdSd*RRb(1,1)+2.0*PuPu*PdSd*RRb(2,1)+SuPu*PdSd*RRb(2,2)
	  RR=RR+PdPu
	  reflex=RR
	  return
	  end
	  
	  subroutine Phasor(w,h,eta,neta,RRb,RRt)
*     include the phase terms appropriate to propagation across the layer
	  complex w,eta,neta,RRb(2,2),RRt(2,2),a,b,ci
	  real h
	    ci=(0.0,1.0)
	    a=cexp(ci*h*eta*w)
	    b=cexp(ci*h*neta*w)
	    RRt(1,1)=a*a*RRb(1,1)
	    RRt(1,2)=a*b*RRb(1,2)
	    RRt(2,1)=a*b*RRb(2,1)
	    RRt(2,2)=b*b*RRb(2,2)
	  return
	  end
	  
	  subroutine MatAdd(a,u,t)
*     matrix addition of 2x2 complex matrices t=a+u
      complex a(2,2),u(2,2),t(2,2)
	  do i=1,2
	    do j=1,2
		  t(i,j)=a(i,j)+u(i,j)
		end do
	  end do
	  return
	  end
	  
	  SUBROUTINE OBScatMat(P,ETA1,NETA1,RHO1,ETA2,NETA2,RHO2,BETA2,
     +                   PdPu,PuPu,PdPd,SuPu,PdSd,Ru)
	  REAL RHO1,RHO2,rtrho1,rtrho2
	  COMPLEX P,ETA1,NETA1,ETA2,NETA2,BETA2,PSQ,CRHO1,CRHO2
	  COMPLEX PdPu,PuPu,PdPd,SuPu,PdSd,Ru(2,2),Tu(2,2),a,b,c,d,Det
	  complex rteta1,rteta2,rtneta1,rtneta2,Ci,mCi,E,F,G,H
	  Ci=(0.0,1.0)
	  mCi=(0.0,-1.0)
	  rtrho1=sqrt(rho1)
	  rtrho2=sqrt(rho2)
	  rteta1=csqrt(eta1)
	  rteta2=csqrt(eta2)
	  psq=p*p
	  crho1=cmplx(rho1,0.0)
	  crho2=cmplx(rho2,0.0)
	  b=crho2-2.0*beta*beta*psq*crho2
	  a=b-crho1
	  d=crho2-b
	  c=crho1+d
	  E=b*eta1+c*eta2
	  F=b
	  G=a-d*eta1*neta2
	  H=-d*eta2
	  Det=E*F+G*H*psq
	  PdPu=((b*eta1-c*eta2)*F-(a+d*eta1*neta2)*H*psq)/Det
	  PdPd=2*rtrho1*rtrho2*rteta1*rteta2*F/Det
	  PdSd=2.0*mCi*rtrho1*rtrho2*rteta1*rtneta2*H*p/Det
	  PuPu=PdPd
	  SuPu=PdSd
	  Ru(1,1)=-d*eta2*G*psq/Det
	  Ru(2,1)=2*mCi*rteta2*rtneta2*b*d*eta1*p/Det
	  Ru(1,2)=Ru(2,1)
	  Ru(2,2)=-(a+d*eta1*neta2)*H*psq/Det
	  return
	  end		  
	  

	  
	  
      complex function OB(P,ETA1,RHO1,ETA2,NETA2,RHO2,BETA2)
	  REAL RHO1,RHO2
	  COMPLEX P,ETA1,ETA2,NETA2,BETA2,PSQ,CRHO1,CRHO2
	  COMPLEX a,b,c,d,Det,dpsq
	  COMPLEX E,F,G,H
*	  TYPE *,'P,ETA1,RHO1,ETA2,NETA2,RHO2,BETA2=',P,ETA1,RHO1,ETA2,
*     +        NETA2,RHO2,BETA2
	  psq=p*p
	  crho1=cmplx(rho1,0.0)
	  crho2=cmplx(rho2,0.0)
	  d=2*rho2*beta2*beta2
	  dpsq=d*psq
	  a=crho2-crho1-dpsq
	  b=rho2-dpsq
	  c=crho1+dpsq
*	  TYPE *,'a,b,c,d=',a,b,c,d
*	  PAUSE
	  E=b*eta1+c*eta2
	  F=b
	  G=a-d*eta1*neta2
	  H=-d*eta2
*	  TYPE *,'E,F,G,H=',E,F,G,H
*	  PAUSE
	  Det=E*F+G*H*psq
	  OB=((b*eta1-c*eta2)*F-(a+d*eta1*neta2)*H*psq)/Det
*	  TYPE *,'DET,OB=',DET,OB
*	  PAUSE
	  return
	  End

      SUBROUTINE FFT(CX,LX,SIGNI)
      COMPLEX CX(LX),CTEMP,CW
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
      AA=3.14159265358979*SIGNI*FLOAT((M-1))/FLOAT(L)
      CW=CMPLX(COS(AA),SIN(AA))
      DO 5 I=M,LX,ISTEP
      CTEMP=CW*CX(I+L)
      CX(I+L)=CX(I)-CTEMP
 5    CX(I)=CX(I)+CTEMP
      L=ISTEP
      IF(L.LT.LX) GO TO 4
      RETURN
      END
	  
      COMPLEX FUNCTION CMPLXRT(Z)
*     A ROBUST VERSION OF THE SQUARE ROOT OF A COMPLEX NUMBER
	  COMPLEX Z,RTZ
	  REAL A,B,X,Y,W,R,EPS
	  PARAMETER(EPS=1E-10)
	  A=CABS(Z)
	  IF(A .LE. EPS) THEN
	    CMPLXRT=(0.0,0.0)
		RETURN
	  ENDIF
	  A=REAL(Z)
	  B=AIMAG(Z)
	  X=ABS(A)
	  Y=ABS(B)
	  IF(X .GT. Y) THEN
	    R=Y/X
		W=SQRT(X)*SQRT(0.5*(1.0+SQRT(1.0+R*R)))
	  ELSE
	    R=Y/X
		W=SQRT(Y)*SQRT(0.5*(R+SQRT(1.0+R*R)))
	  ENDIF
	  IF(A .GT. 0.0) THEN
	    CMPLXRT=CMPLX(W,0.5*B/W)
	  ELSE
	    IF(B .GE. 0.0)THEN
		   CMPLXRT=CMPLX(W,.5*B/W)
	    ELSE
		   CMPLXRT=CMPLX(-W,-0.5*B/W)
	    END IF
	  ENDIF
	  RETURN
	  END
	  
	  complex function Riccatti_Reflex(nlp1,p,w,rho,h,eta,neta,beta)
************************************************************************
*     compute the ratio of the upgoing to the downgoing wavefield      *
*     of a plane layer stack using a Riccatti recursive algorithm      *
************************************************************************
      Implicit none
      integer i,j,nlp1,nl,il
      real H(nlp1),rho(nlp1),d
      complex p,w,eta(nlp1),neta(nlp1),beta(nlp1),C0,C1,Ci,psq,mu,iwd
	  parameter(C0=(0.0,0.0),C1=(1.0,0.0),Ci=(0.0,1.0))
	  complex M11,M12,M21,M22,N11,N12,N21,N22,X11,X12,X21,X22
	  complex T11,T12,T21,T22,U11,U12,U21,U22,V11,V12,V21,V22,TWOMUP
	  complex RR11,RR12,RR21,RR22,ea,eb,eaeb,invDet,a,b,t,N,M
	  psq=p*p
	  nl=nlp1-1
      RR11=C0
	  RR12=C0
	  RR21=C0
	  RR22=C0
*     set (-M+)n=M
      twomup=2.0*rho(nlp1)*beta(nlp1)*beta(nlp1)*p
      M11=Ci*eta(nlp1)
	  M12=-p
	  M21=M11*twomup
	  M22=cmplx(rho(nlp1),0.0)-twomup*p
	  N11=-M22
	  N22=Ci*neta(nlp1)
	  N12=-N22*twomup
	  N21=-p
*     compute X=-M(NLP1)*Inv(N(NLP1))
	  invDet=c1/(N11*N22-N12*N21)
	  X11=(M11*N12-M12*N21)*invDet
	  X12=(M12*N11-M11*N12)*invDet
	  X21=(M21*N22-M22*N21)*invDet
	  X22=(M22*N11-M21*N12)*invDet
	  Do il=nl,2,-1
*       SET THE LAYER EIGEN MATRICES
        twomup=2.0*rho(nlp1)*beta(nlp1)*beta(nlp1)*p
        M11=-Ci*eta(il)
	    M12=p
	    M21=M11*twomup
	    M22=twomup*p-cmplx(rho(il),0.0)
	    N11=M22
	    N22=Ci*neta(il)
	    N12=-N22*twomup
	    N21=-p
*       calculate XN=T
        T11=X11*N11+X12*N21
		T12=X11*N12+X12*N22
		T21=X21*N11+X22*N21
		T22=X21*N12+X22*N22
*       U=M-T
        U11=M11-T11
		U12=M12-T12
		U21=M21-T21
		U22=M22-T22
*       V=M+T
        V11=M11+T11
		V12=M12+T12
		V21=M21+T21
		V22=M22+T22
*       RR=Inv(u)*V
        invDet=c1/(U11*U22-U12*U21)
		RR11=(U22*V11-U12*V21)*invDet
		RR12=(U22*V12-U12*V22)*invDet
		RR21=(U11*V21-U21*V11)*invDet
		RR22=(U11*V22-U21*V12)*invDet
		d=h(il)-h(il-1)
		iwd=Ci*w*d
		ea=cexp(iwd*eta(il))
		eb=cexp(iwd*neta(il))
		eaeb=ea*eb
		RR11=ea*ea*RR11
		RR22=eb*eb*RR22
		RR12=eaeb*RR12
		RR21=eaeb*RR21
        IF (IL .GT. 2)THEN
*         T=M(RR)
          T11=M11*RR11+M12*RR21
		  T12=M11*RR12+M12*RR22
		  T21=M21*RR11+M22*RR21
		  T22=M21*RR12+M22*RR22
*         U=T-M
          U11=T11-M11
		  U12=T12-M12
		  U21=T21-M21
		  U22=T22-M22
*         T=N(RR)
          T11=N11*RR11+N12*RR21
		  T12=N11*RR12+N12*RR22
		  T21=N21*RR11+N22*RR21
		  T22=N21*RR12+N22*RR22
*         V=T+N
          V11=T11+N11
		  V12=T12+N12
		  V21=T21+N21
		  V22=T22+N22
*         X=U*Inv(V)
          invDet=c1/(V11*V22-V21*V12)
		  X11=(U11*V22-U12*V21)*invDet
		  X12=(U12*V11-U11*V12)*invDet
		  X21=(U21*V22-U22*V21)*invDet
		  X22=(U22*V11-U21*V12)*invDet
        END IF
      End Do
*     Incorporate the ocean bottom inteface
      t=(M21-M21*RR11-M22*RR21)/(M21*RR12+M22*RR22-M22)
      M=N11*(RR11+t*RR12+C1)+N12*(RR21+t*(RR22+C1)) 
	  N=M11*(RR11+t*RR12-C1)+M12*(RR21+t*(RR22-C1))
      a=rho(1)*N
	  b=Ci*eta(1)*M
	  Riccatti_Reflex=(b+a)/(b-a)
	  Return
	  End
	  
      COMPLEX FUNCTION S(OMEGA,OmegaMax)
*     ***************************************************************
*     Fourier transform of Ricker wavelet
*     ****************************************************************
	  COMPLEX OMEGA,CI,Z,OmegaMax,Alfa,T0
	  CI=(0.0,1.0)
	  alfa=.12*omegamax
	  T0=5.5/(ALFA*SQRT(2.0))
	  Z=OMEGA*OMEGA/(4.*ALFA*ALFA)
	  S=Z*CEXP(-Z+CI*OMEGA*T0)
	  RETURN
	  END
	  