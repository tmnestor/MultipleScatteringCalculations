      program RicattiSlo
	  INTEGER   OUTFILE,NLP1MAX,NWMAX,NPMAX,Maxils
	  parameter (outfile=11,NLP1MAX=4,NWMAX=256,NPMAX=1)
      COMPLEX   Ci,C0,C1,W0
	  PARAMETER (Ci=(0.0,1.0),C1=(1.0,0.0),C0=(0.0,0.0))
      COMPLEX   A(NLP1MAX),B(NLP1MAX),BETA(NLP1MAX)
      COMPLEX   W,DW,ETA(NLP1MAX),NETA(NLP1MAX)
      COMPLEX   CMPLXSLO,VERTICALSLO
      COMPLEX   U(NWMAX),Uwk(2*NWMAX),M(NWMAX),S
	  complex   M11(nlp1max),M12(nlp1max),M21(nlp1max),M22(nlp1max)
	  complex   N11(nlp1max),N12(nlp1max),N21(nlp1max),N22(nlp1max)
	  complex   Y11(nlp1max),Y12(nlp1max),Y21(nlp1max),Y22(nlp1max)
	  complex   X11(nlp1max),X12(nlp1max),X21(nlp1max),X22(nlp1max)
	  complex   rteap(nlp1max),rtebp(nlp1max),rtinceap(nlp1max)
	  complex   rteam(nlp1max),rtebm(nlp1max),rtinceam(nlp1max)
      complex   rtincebp(nlp1max),rtincebm(nlp1max)
	  complex   RR11(nlp1max,Nwmax),RR12(nlp1max,Nwmax)
	  complex   RR21(nlp1max,Nwmax),RR22(nlp1max,Nwmax)
	  complex   PuPu(Nlp1max,Nwmax),SuPu(Nlp1max,Nwmax),PdPu(Nwmax)
	  complex   SUZ(nlp1max),SUR(nlp1max),SDZ(nlp1max),SDR(nlp1max)
	  complex   PUZ(nlp1max),PUR(nlp1max),PDZ(nlp1max),PDR(nlp1max)
	  complex   GPZ(Nlp1max,Nwmax),GPR(Nlp1max,Nwmax),SourceDirectivity
	  complex   DGPZ(Nlp1max,Nwmax),DGPR(Nlp1max,Nwmax)
      complex   DPDR(Nlp1max,Nwmax),DPDA(Nlp1max,Nwmax),DPDB(Nlp1max,Nwmax)
	  complex   ea(nlp1max),eb(nlp1max),eaea(nlp1max),ebeb(nlp1max)
	  complex   eaeb(nlp1max),incea(nlp1max),inceaea(nlp1max),T1,T2
      complex   inceaeb(nlp1max),inceb(nlp1max),incebeb(nlp1max)
	  REAL      p,CA0(NLP1MAX),CB0(NLP1MAX),DD(NLP1MAX)
	  REAL      QA0(NLP1MAX),QB0(NLP1MAX),H(NLP1MAX),RHO(NLP1MAX),T,DT
	  character *1 TAB
	  TAB=char(9)
	  open(outfile,file='QRiccattigraphdat',status='new')
	  
!**********************************************************************
!     INPUT THE EXPERIMENTAL GEOMETRY
!**********************************************************************
      NLP1=2
	  NL=NLP1-1
	  IF(NLP1 .GT. NLP1MAX) THEN
	    PAUSE 'TOO MANY LAYERS'
	  END IF
      DR=0.0
      DS=DR
!********************************************************************
!     INPUT LAYER PARAMETERS
!********************************************************************
	  CA0(1)=1.5
	  CB0(1)=0.0
	  QA0(1)=1E10
	  QB0(1)=1E10
	  RHO(1)=1.0
	  H(1)  =7.5
	  DD(1) =H(1)-0.5*(Ds+Dr)
	  CA0(2)=1.6
	  CB0(2)=0.3
	  QA0(2)=1.E10
	  QB0(2)=1E10
	  RHO(2)=2.0
	  H(2)  =31.0
	  DD(2) =H(2)-H(1)
	  CA0(3)=3.0
	  CB0(3)=1.5
	  QA0(3)=1E10
	  QB0(3)=1E10
	  RHO(3)=3.0
	  H(3)  =61.0
	  DD(3)=H(3)-H(2)
	  CA0(4)=5.0
	  CB0(4)=3.0
	  QA0(4)=1E10
	  QB0(4)=1E10
	  RHO(4)=2.0
	  H(4)  =1.E20
	  

*     Input Length of time series and number of frequencies
      T=120.0
      NW=128

*     Define miscellaneous program constants
      NT=2*NW
      DT=T/FLOAT(NT)
      TPI=2.0*PI	  
      PI2=PI/2.
      PI4=PI/4.
      DW=TPI/T
	  WMAX=NW*DW
      NWM=NW-1

*     define frequency damping factor
*      eps=log(2.0)/T
      eps=0.0
*     max frequency index NW corresponds to frequency limit of source	  
      FMAX=FLOAT(NWM)/T


*     define starting frequency (could have small pos. imag part)	  
      W0=cmplx(0.0,eps)
	  
*     ****************************************************************
*     input  source spectrum vector
*     ****************************************************************
      w=W0
      Do 20 iw=1,Nwm
	    w=w+Dw
	    M(iw)=S(w,wmax)
*		type*,'w,wmax,M=',w,wmax,M(iw)
 20   continue

*     ****************************************************************
*     define complex frequency independent slownesses of each layer
*     ****************************************************************
      A(1)=CMPLXSLO(CA0(1),QA0(1))
	  B(1)=C0
	  BETA(1)=C0
      Do 30 IL=2,NLP1
        A(IL)=CMPLXSLO(CA0(IL),QA0(IL))
		B(IL)=CMPLXSLO(CB0(IL),QB0(IL))
*		TYPE *,'A(IL),B(IL)=',A(IL),B(IL)
*		PAUSE
	    BETA(IL)=C1/B(IL)
 30   continue
	  
       Do while(p.ne.1)
*     ****************************************************************
*     Input the Slowness 
*     ****************************************************************
	    Type*,'input p'
		Accept*,p
	   CALL SLOWRESP(p,W0,Nw,Dw,X11,X12,X21,X22,Y11,Y12,Y21,Y22,
     &	    NLP1,Nwmax,Nlp1max,A,B,BETA,RHO,DD,M11,M12,M21,
     &      M22,N11,N12,N21,N22,eta,neta,ea,eb,eaea,ebeb,eaeb,
     &      incea,inceb,inceaea,incebeb,inceaeb,
     &      rteap,rtebp,rtinceap,rteam,rtebm,rtinceam,rtincebp,rtincebm,
     &      RR11,RR12,RR21,RR22,PuPu,SuPu,SUZ,SUR,SDZ,SDR,PUZ,PUR,PDZ,
     &      PDR,GPZ,GPR,DGPZ,DGPR,DPDR,DPDA,DPDB,PdPu,SourceDirectivity)

*      T1=eta(1)*M22(2)*M22(2)+2*Ci*N12(2)*p*eta(2)*Beta(2)*Beta(2)
*	  T2=rho(1)*rho(2)*eta(2)
*	  Type*,'OBreflectivity=',(T1-T2)/(T1+T2)
*      Pause
*     multiply the the slowness response by the source spectrum vector
      w=w0
      Do 50 iw=1,NWM
	    w=w+dw
        U(iw)=PdPu(iw)*S(w,wmax)*SourceDirectivity/w
*		U(iw)=PdPu(iw)*M(iw)
 50   continue    
 
*     Inverse FFT transfer function to give R(Tau,p) response
      Do 60 iw=1,NWM
        Uwk(iw+1)=U(iw)
        Uwk(NT-iw+1)=CONJG(U(iw))
 60   continue
      Uwk(1)=C0
      Uwk(NW+1)=C0
      CALL FFT(Uwk,NT,-1.0)
	  TT=-Dt
      Do 70 it=1,NT
	    TT=TT+Dt
*       compensate for the frequency damping and write to plotting file
        ff=Real(Uwk(it))*exp(eps*TT)
	    WRITE (outfile,999) TT,TAB,ff
		TYPE*,TT,TAB,ff
  70  continue
      End do
 999  FORMAT(f5.1,A1,g10.4) 
      close(outfile)
      STOP
      END

      SUBROUTINE SLOWRESP(p,W0,Nw,Dw,X11,X12,X21,X22,Y11,Y12,Y21,
     &	            Y22,NLP1,Nwmax,Nlp1max,A,B,BETA,RHO,DD,M11,M12,M21,
     &              M22,N11,N12,N21,N22,eta,neta,ea,eb,eaea,ebeb,eaeb,incea,
     &              inceb,inceaea,incebeb,inceaeb,
     &              rteap,rtebp,rtinceap,rteam,rtebm,rtinceam,rtincebp,
     &              rtincebm,RR11,RR12,RR21,RR22,PuPu,SuPu,SUZ,SUR,SDZ,SDR,
     &              PUZ,PUR,PDZ,PDR,GPZ,GPR,DGPZ,DGPR,DPDR,DPDA,DPDB,PdPu,
     &              SourceDirectivity)
	  
!**********************************************************************
!     CALCULATE THE HORIZONTEL SLOWNESS DEPENDENT COMPONENTS OF THE
!     REFLECTIVITY INTEGRAND
!**********************************************************************
*	  Implicit none
	  COMPLEX Ci,C1,C0	  
	  PARAMETER (Ci=(0.0,1.0),C1=(1.0,0.0),C0=(0.0,0.0))
      INTEGER NLP1,IL,Nw,Nl,Nwmax,NLP1MAX
      COMPLEX W0,Dw,A(NLP1MAX),B(NLP1MAX),BETA(NLP1MAX)
	  COMPLEX ETA(NLP1max),NETA(NLP1max),cp,twomup,twomupsq
	  complex M11(nlp1max),M12(nlp1max),M21(nlp1max),M22(nlp1max)
	  complex N11(nlp1max),N12(nlp1max),N21(nlp1max),N22(nlp1max)
	  complex Y11(nlp1max),Y12(nlp1max),Y21(nlp1max),Y22(nlp1max)
	  complex X11(nlp1max),X12(nlp1max),X21(nlp1max),X22(nlp1max)
	  complex RR11(nlp1max,Nwmax),RR12(nlp1max,Nwmax)
	  complex RR21(nlp1max,Nwmax),RR22(nlp1max,Nwmax)
	  complex PuPu(Nlp1max,Nwmax),SuPu(Nlp1max,Nwmax),PdPu(Nwmax)
	  complex GPZ(Nlp1max,Nwmax),GPR(Nlp1max,Nwmax)
	  complex DGPZ(Nlp1max,Nwmax),DGPR(Nlp1max,Nwmax)
      complex DPDR(Nlp1max,Nwmax),DPDA(Nlp1max,Nwmax),DPDB(Nlp1max,Nwmax)
	  complex SUZ(nlp1max),SUR(nlp1max),SDZ(nlp1max),SDR(nlp1max)
	  complex PUZ(nlp1max),PUR(nlp1max),PDZ(nlp1max),PDR(nlp1max)
	  complex rteap(nlp1max),rtebp(nlp1max),rtinceap(nlp1max)
	  complex rteam(nlp1max),rtebm(nlp1max),rtinceam(nlp1max)
      complex rtincebp(nlp1max),rtincebm(nlp1max),whineta,whieta
	  complex ea(nlp1max),eb(nlp1max),eaea(nlp1max),ebeb(nlp1max)
	  complex eaeb(nlp1max),incea(nlp1max),inceaea(nlp1max)
      complex inceaeb(nlp1max),inceb(nlp1max),incebeb(nlp1max)
	  complex Dwhineta,Dwhieta,twomu,wk,SourceDirectivity
	  REAL    RHO(NLP1max),DD(NLP1max),p,psq
	  
*     Initialise frequency independent layer parameters
	  psq=p*p
	  cp=cmplx(p)
      Nl=Nlp1-1
	  IL=1
      NETA(IL)=C0
	  ETA(IL)=VERTICALSLO(A(IL),CP)
	  ea(1)=cexp(Ci*W0*DD(1)*eta(1))
	  eaea(1)=ea(1)*ea(1)
	  incea(1)=cexp(Ci*Dw*DD(1)*eta(1))
	  inceaea(1)=incea(1)*incea(1)
	  eb(1)=C0
	  inceb(1)=C0
	  eaeb(1)=ea(1)
	  inceaeb(1)=C0
	  M11(1)=Ci*eta(1)
	  M12(1)=C0
	  M21(1)=C0
	  M22(1)=C0
	  N11(1)=cmplx(rho(1),0)
	  N12(1)=C0
	  N21(1)=C0
	  N22(1)=C0
	  SourceDirectivity=Ci/(2*rho(1)*eta(1))
      DO IL=2,NL
*       set layer eigenvector matrices
	    ETA(IL)=VERTICALSLO(A(IL),CP)
		NETA(IL)=VERTICALSLO(B(IL),CP)
	    M11(il)=-Ci*eta(il)
		M12(il)=p
		twomu=2*rho(il)*beta(il)*beta(il)
		twomup=TWOMU*p
		twomupsq=twomu*psq
		M21(il)=M11(il)*twomup
		M22(il)=(twomupsq-cmplx(rho(il),0.0))
		N11(il)=M22(il)
		N22(il)=Ci*neta(il)
		N21(il)=-p
		N12(il)=-N22(il)*twomup
*       set phase terms
		wk=0.5*DD(il)*W0*Ci
		whieta=wk*eta(il)
        whineta=wk*neta(il)
        rteap(il)=cexp(whieta)
        rteam(il)=cexp(-whieta)
		rtebp(il)=cexp(whineta)
		rtebm(il)=cexp(-whineta)
		ea(il)=rteap(il)*rteap(il)
		eb(il)=rtebp(il)*rtebp(il)
		ebeb(il)=eb(il)*eb(il)
		eaeb(il)=ea(il)*eb(il)
		eaea(il)=ea(il)*ea(il)
		wk=0.5*DD(il)*Dw*Ci
		Dwhieta=wk*eta(il)
        Dwhineta=wk*neta(il)
		rtinceap(il)=cexp(Dwhieta)
		rtinceam(il)=cexp(-Dwhieta)
		rtincebp(il)=cexp(Dwhineta)
		rtincebm(il)=cexp(-Dwhineta)
		incea(il)=rtinceap(il)*rtinceap(il)
		inceb(il)=rtincebp(il)*rtincebp(il)
		inceaea(il)=incea(il)*incea(il)
		incebeb(il)=inceb(il)*inceb(il)
		inceaeb(il)=incea(il)*inceb(il)
		PUZ(il)=-eta(il)*rteam(il)
		PUR(il)=Ci*p*rteam(il)
		PDZ(il)=eta(il)*rteap(il)
		PDR(il)=-Ci*p*rteap(il)
		SUZ(il)=Ci*p*rtebm(il)
		SUR(il)=-neta(il)*rtebm(il)
		SDZ(il)=-Ci*p*rtebp(il)
		SDR(il)=-neta(il)*rtebp(il)
	  END DO
	  ETA(Nlp1)=VERTICALSLO(A(Nlp1),CP)
	  NETA(Nlp1)=VERTICALSLO(B(Nlp1),CP)
	  M11(nlp1)=-Ci*eta(nlp1)
	  M12(nlp1)=p
	  twomup=2*rho(nlp1)*beta(nlp1)*beta(nlp1)*p
	  M21(nlp1)=M11(nlp1)*twomup
	  M22(nlp1)=(twomup*p-cmplx(rho(nlp1),0.0))
	  N11(nlp1)=M22(nlp1)
	  N22(nlp1)=Ci*neta(nlp1)
	  N21(nlp1)=-p
	  N12(nlp1)=-N22(nlp1)*twomup
*     CALCULATE THE DOWNWARD REFLECTION & UPWARD TRANSMISSION ARRAYS
      call    Riccatti_Reflex(nlp1,Nlp1max,Nw,Nwmax,X11,X12,
     &        X21,X22,Y11,Y12,Y21,Y22,M11,M12,M21,M22,
     &        N11,N12,N21,N22,ea,eb,eaea,ebeb,eaeb,incea,
     &        inceb,inceaea,incebeb,inceaeb,RR11,RR12,RR21,RR22,
     &        PdPu,PuPu,SuPu)
*     CALCULATE THE GREEN's FUNCTIONS 
      call    Gfn(nlp1,Nlp1max,Nw,Nwmax,W0,Dw,eta,neta,
     &        rteap,rtebp,rtinceap,rteam,rtebm,rtinceam,rtincebp,
     &        rtincebm,RR11,RR12,RR21,RR22,PuPu,SuPu,SUZ,SUR,SDZ,SDR,
     &        PUZ,PUR,PDZ,PDR,GPZ,GPR,DGPZ,DGPR)

*     CALCULATE THE FRECHET DERIVATIVES
      call    Frechet(p,nlp1,Nlp1max,Nw,Nwmax,W0,Dw,alpha,beta,rho,
     &        GPZ,GPR,DGPZ,DGPR,DPDR,DPDA,DPDB)

      RETURN
      END
	  
      subroutine Frechet(p,nlp1,Nlp1max,Nw,Nwmax,W0,Dw,alpha,beta,rho,
     &           GPZ,GPR,DGPZ,DGPR,DPDR,DPDA,DPDB)
      INTEGER NLP1,IL,Nw,Nl,Nwmax,NLP1MAX,iw
      COMPLEX W0,Dw,Alpha(NLP1MAX),Beta(NLP1MAX),bsq,asq,wsq,w,wp,wsqpsq
	  COMPLEX DGPZSQ,DGPRGPZ,GPZSQ,DGPRSQ,DGPZGPR,GPRSQ,C1,t
	  complex GPZ(Nlp1max,Nwmax),GPR(Nlp1max,Nwmax)
	  complex DGPZ(Nlp1max,Nwmax),DGPR(Nlp1max,Nwmax)
      complex DPDR(Nlp1max,Nwmax),DPDA(Nlp1max,Nwmax),DPDB(Nlp1max,Nwmax)
	  REAL    p,RHO(NLP1MAX),psq
	  parameter(C1=(1.0,0.0))
	  psq=p*p
      nl=nlp1-1
      w=w0
	  do iw=1,Nw-1
	    w=w+dw
		wsq=w*w
		wp=w*p
		wsqpsq=wsq*psq
		do il=2,nl
	      asq=alpha(il)*alpha(il)
		  bsq=beta(il)*beta(il)
		  DGPZSQ=DGPZ(iw,il)*DGPZ(iw,il)
		  DGPRGPZ=DGPR(iw,il)*GPZ(iw,il)
		  DGPZGPR=DGPZ(iw,il)*GPR(iw,il)
		  DGPRSQ=DGPR(iw,il)*DGPR(iw,il)
		  GPZSQ=GPZ(iw,il)*GPZ(iw,il)
		  GPRSQ=GPR(iw,il)*GPR(iw,il)
		  t=asq*DGPZSQ+2*wp*bsq*DGPRGPZ+(wsqpsq*bsq-wsq)*GPZSQ
		  t=t+bsq*DGPRSQ+2*wp*(2*bsq-asq)*DGPZGPR+(wsqpsq*asq-wsq)*GPRSQ
		  DPDR(iw,il)=t
		  t=(DGPZSQ-2*wp*DGPZGPR+wsqpsq*GPRSQ)
		  DPDA(iw,il)=2.*rho(il)*alpha(il)*t
		  t=DGPRSQ+2.*wp*(DGPRGPZ+2.*DGPZGPR)+wsqpsq*GPZSQ
		  DPDB(iw,il)=2.*rho(il)*beta(il)*t
	    end do
	  end do
	  return
	  end

      Subroutine Gfn(nlp1,Nlp1max,Nw,Nwmax,W0,Dw,eta,neta,
     &           rteap,rtebp,rtinceap,rteam,rtebm,rtinceam,rtincebp,
     &           rtincebm,RR11,RR12,RR21,RR22,PuPu,SuPu,SUZ,SUR,SDZ,SDR,
     &           PUZ,PUR,PDZ,PDR,GPZ,GPR,DGPZ,DGPR)
*     ********************************************************************
*                FORM THE GREEN's FUNCTIONS
*     ********************************************************************
      Implicit none
	  Integer    il,iw,nlp1,nl,Nw,Nwmax,Nlp1max
	  complex    GPZ(Nlp1max,Nwmax),GPR(Nlp1max,Nwmax),W0,W,Dw,Ci
	  complex    DGPZ(Nlp1max,Nwmax),DGPR(Nlp1max,Nwmax),neta(Nlp1max)
	  complex    RR11(nlp1max,Nwmax),RR12(nlp1max,Nwmax)
	  complex    RR21(nlp1max,Nwmax),RR22(nlp1max,Nwmax)
	  complex    PuPu(Nlp1max,Nwmax),SuPu(Nlp1max,Nwmax)
	  complex    rteap(nlp1max),rtebp(nlp1max),rtinceap(nlp1max)
	  complex    rteam(nlp1max),rtebm(nlp1max),rtinceam(nlp1max)
      complex    rtincebp(nlp1max),rtincebm(nlp1max),eta(Nlp1max)
	  complex    SUZ(nlp1max),SUR(nlp1max),SDZ(nlp1max),SDR(nlp1max)
	  complex    PUZ(nlp1max),PUR(nlp1max),PDZ(nlp1max),PDR(nlp1max)
	  complex    PZ,PR,SZ,SR,DPZ,DPR,DSZ,DSR,d1,d2,d3,d4,t1,t2,t3,t4
	  complex    t5,t6,t7,t8,t9,t10,t11,t12
	  parameter(Ci=(0.0,1.0))
      nl=nlp1-1
      w=w0
	  do iw=1,Nw-1
	    w=w+dw
		do il=2,nl
	      PUZ(il)=PUZ(il)*rtinceam(il)
		  PUR(il)=PUR(il)*rtinceam(il)
		  PDZ(il)=PDZ(il)*rtinceap(il)
		  PDR(il)=PDR(il)*rtinceap(il)
		  SUZ(il)=SUZ(il)*rtincebm(il)
		  SUR(il)=SUR(il)*rtincebm(il)
		  SDZ(il)=SDZ(il)*rtincebp(il)
		  SDR(il)=SDR(il)*rtincebp(il)
		  t1=RR11(il,iw)*PDZ(il)
		  t2=RR12(il,iw)*SDZ(il)
		  t3=PUZ(il)
		  PZ=t1+t2+t3
		  t4=RR11(il,iw)*PDR(il)
		  t5=RR12(il,iw)*SDR(il)
		  t6=PUR(il)
		  PR=t4+t5+t6
		  t7=RR21(il,iw)*PDZ(il)
		  t8=RR22(il,iw)*SDZ(il)
		  t9=SUZ(il)
		  SZ=t7+t8+t9
		  t10=RR21(il,iw)*PDR(il)
		  t11=RR22(il,iw)*SDR(il)
		  t12=SUR(il)
		  SR=t10+t11+t12
          d1=Ci*w*eta(il)
		  d2=Ci*w*neta(il)
		  d3=-d1
		  d4=-d2
		  DPZ=t1*D3+t2*D4+D1*t3
		  DPR=t4*D3+t5*D4+D1*t6
		  DSZ=t7*D3+t8*D4+D2*t9
		  DSR=t10*D3+t11*D4+D2*t12
		  GPZ(il,iw)=PuPu(il,iw)*PZ+SuPu(il,iw)*SZ
          GPR(il,iw)=PuPu(il,iw)*PR+SuPu(il,iw)*SR
          DGPZ(il,iw)=PuPu(il,iw)*DPZ+SuPu(il,iw)*DSZ
		  DGPR(il,iw)=PuPu(il,iw)*DPR+SuPu(il,iw)*DSR
		END DO ! il
	  END DO  ! iw
      return
	  end

      Subroutine Riccatti_Reflex(nlp1,Nlp1max,Nw,Nwmax,X11,X12,
     &        X21,X22,Y11,Y12,Y21,Y22,M11,M12,M21,M22,
     &        N11,N12,N21,N22,ea,eb,eaea,ebeb,eaeb,incea,
     &        inceb,inceaea,incebeb,inceaeb,
     &        RR11,RR12,RR21,RR22,PdPu,PuPu,SuPu)
***********************************************************************	  
*     compute the Downward Reflection  & Upward Transmission matrices
*     of a plane layer stack bounded above by an acoustic halfspace
*     using a Discrete-Riccatti recursive algorithm
***********************************************************************
      Implicit none
      integer  i,j,nlp1,nl,il,Nw,iw,Nlp1max,Nwmax,ilp1
      complex  C0,C1,Ci
	  parameter(C0=(0.0,0.0),C1=(1.0,0.0),Ci=(0.0,1.0))
	  complex  RRt11,RRt12,RRt21,RRt22,U,e,g,S,PdPu(Nwmax)
	  complex  M11(nlp1max),M12(nlp1max),M21(nlp1max),M22(nlp1max)
	  complex  N11(nlp1max),N12(nlp1max),N21(nlp1max),N22(nlp1max)
	  complex  X11(nlp1max),X12(nlp1max),X21(nlp1max),X22(nlp1max)
	  complex  Y11(nlp1max),Y12(nlp1max),Y21(nlp1max),Y22(nlp1max)
	  complex  RR11(nlp1max,Nwmax),RR12(nlp1max,Nwmax)
	  complex  RR21(nlp1max,Nwmax),RR22(nlp1max,Nwmax)
	  complex  PuPu(Nlp1max,Nwmax),SuPu(Nlp1max,Nwmax)
	  complex  U11,U12,U21,U22,V11,V12,V21,V22
	  complex  Q11,Q12,Q21,Q22,W11,W12,W21,W22
	  complex  ea(nlp1max),eb(nlp1max),eaea(nlp1max),ebeb(nlp1max)
	  complex  eaeb(nlp1max),incea(nlp1max),inceaea(nlp1max)
      complex  inceaeb(nlp1max),inceb(nlp1max),incebeb(nlp1max)
	  complex  T1,T2,InvDet,Vp,fp,hp,Vs,fs,hs
!**********************************************************************	  
!              loop over frequency pass-band of source
!**********************************************************************
	  do iw=1,Nw-1
	  nl=nlp1-1
***********************************************************************
*     with an upward sweep from the base of the layer stack to just 
*     below the ocean bottom calculate the reflectivity phased wrt the
*     bottom of each layer
***********************************************************************
      RRT11=c0
	  RRT12=c0
	  RRT21=c0
	  RRT22=c0
*     compute -M(NLP1)*Inv(N(NLP1))
	  InvDet=C1/(N11(nlp1)*N22(nlp1)-N12(nlp1)*N21(nlp1))
	  X11(nl)=(-M11(nlp1)*N22(nlp1)+M12(nlp1)*N21(nlp1))*InvDet
	  X12(nl)=(-M12(nlp1)*N11(nlp1)+M11(nlp1)*N12(nlp1))*InvDet
	  X21(nl)=(-M21(nlp1)*N22(nlp1)+M22(nlp1)*N21(nlp1))*InvDet
	  X22(nl)=(-M22(nlp1)*N11(nlp1)+M21(nlp1)*N12(nlp1))*InvDet
*	  type*,'X11,X12,X21,X22=',X11(nl),X12(nl),X21(nl),X22(nl)
*	  pause
	  Do il=nl,2,-1
*       calculate W=X*N
        W11=X11(il)*N11(il)+X12(il)*N21(il)
		W12=X11(il)*N12(il)+X12(il)*N22(il)
		W21=X21(il)*N11(il)+X22(il)*N21(il)
		W22=X21(il)*N12(il)+X22(il)*N22(il)
*		type*,'W=XN W11,W12,W21,W22=',W11,W12,W21,W22
*		pause
*       Q=M-W
        Q11=M11(il)-W11
		Q12=M12(il)-W12
		Q21=M21(il)-W21
		Q22=M22(il)-W22
*		type*,'Q=M-XN Q11,Q12,Q21,Q22=',Q11,Q12,Q21,Q22
*		pause
*       Y=Inv(Q)
        InvDet=C1/(Q11*Q22-Q12*Q21)
        Y11(il)=InvDet*Q22
		Y12(il)=-InvDet*Q12
		Y21(il)=-InvDet*Q21
		Y22(il)=InvDet*Q11
*       V=M+W
        V11=M11(il)+W11
		V12=M12(il)+W12
		V21=M21(il)+W21
		V22=M22(il)+W22
*       RR=Y*V
        RR11(il,iw)=Y11(il)*V11+Y12(il)*V21
		RR12(il,iw)=Y11(il)*V12+Y12(il)*V22
		RR21(il,iw)=Y21(il)*V11+Y22(il)*V21
		RR22(il,iw)=Y21(il)*V12+Y22(il)*V22
*		type*,'RR11,RR12,RR21,RR22=',RR11(il,iw),RR12(il,iw),
*     &         RR21(il,iw),RR22(il,iw)
*		pause
*       Apply the phase terms for layer propagation
		eaea(il)=eaea(il)*inceaea(il)
		ebeb(il)=ebeb(il)*incebeb(il)
		eaeb(il)=eaeb(il)*inceaeb(il)
		RRt11=eaea(il)*RR11(il,iw)
		RRt22=ebeb(il)*RR22(il,iw)
		RRt12=eaeb(il)*RR12(il,iw)
		RRt21=eaeb(il)*RR21(il,iw)
*       Q=M(RR)
        if(il .gt.2) then
        Q11=M11(il)*RRt11+M12(il)*RRt21
		Q12=M11(il)*RRt12+M12(il)*RRt22
		Q21=M21(il)*RRt11+M22(il)*RRt21
		Q22=M21(il)*RRt12+M22(il)*RRt22
*       U=Q-M
        U11=Q11-M11(il)
		U12=Q12-M12(il)
		U21=Q21-M21(il)
		U22=Q22-M22(il)
*       Q=N(RR)
        Q11=N11(il)*RRt11+N12(il)*RRt21
		Q12=N11(il)*RRt12+N12(il)*RRt22
		Q21=N21(il)*RRt11+N22(il)*RRt21
		Q22=N21(il)*RRt12+N22(il)*RRt22
*       V=Q+N
        V11=Q11+N11(il)
		V12=Q12+N12(il)
		V21=Q21+N21(il)
		V22=Q22+N22(il)
*       X=U*Inv(V)
        InvDet=C1/(V11*V22-V21*V12)
		X11(il-1)=(U11*V22-U12*V21)*InvDet
		X12(il-1)=(U12*V11-U11*V12)*InvDet
		X21(il-1)=(U21*V22-U22*V21)*InvDet
		X22(il-1)=(U22*V11-U21*V12)*InvDet
*		type*,'X11,X12,X21,X22',X11(il-1),X12(il-1),X21(il-1),X22(il-1)
*		pause
      End if
      End Do 
*	  type*,'RRt11,RRt12,RRt21,RRt22=',RRt11,RRt12,RRt21,RRt22
*	  pause
*************************************************************************
*                     END REFLECTIVITY LOOP
*************************************************************************
**************************************************************************
*             INCORPORATE THE OCEAN BOTTOM INTERFACE
**************************************************************************
*     find RR11 (phase related to the source depth)
	  U=-(M21(2)*(RRt11-C1)+M22(2)*RRt21)/(M22(2)*(RRt22-C1)+M21(2)*RRt12)
      e=M11(2)*(RRt11+U*RRt12-C1)+M12(2)*(RRt21+U*RRt22-U)
	  g=N11(2)*(RRt11+U*RRt12+C1)+N12(2)*(RRt21+U*RRt22+U)
*	  type*,'U,e,g=',U,e,g
*	  pause
	  S=-e*N11(1)/(g*M11(1))
	  eaea(1)=eaea(1)*inceaea(1)
	  PdPu(iw)=eaea(1)*(C1-S)/(C1+S)
	  Type*,'iw,PdPu,eaea=',iw,(C1-S)/(C1+S),eaea(1)
	  pause
	  t1=C1/(M21(2)*RRt12+M22(2)*(RRt22-C1))
	  Vp=-M21(2)*t1
	  Vs=-M22(2)*t1
	  t1=(M11(2)*RRt12+M12(2)*(RRt22-C1))
	  fp=Vp*t1+M11(2)
	  fs=Vs*t1+M12(2)
	  t1=(N11(2)*RRt12+N12(2)*(RRt22+C1))
	  hp=Vp*t1+N11(2)
	  hs=Vs*t1+N12(2)
*	  type*,'V,f,h=',V,f,h
*	  Pause
	  ea(1)=ea(1)*incea(1)
	  t1=(M11(1)*g-N11(1)*e)
	  t2=-N11(1)*ea(1)/t1
	  PuPu(1,iw)=t2*(fp*g-e*hp)
	  SuPu(1,iw)=t2*(fs*g-e*hs)
*************************************************************************
*     With a downward sweep from just below the ocean bottom to the bottom
*     of the deepest layer calculate the upward transmission arrays phased
*     wrt the bottom of each layer
**************************************************************************
	  ea(2)=ea(2)*incea(2)
	  eb(2)=eb(2)*inceb(2)
	  PuPu(2,iw)=PuPu(1,iw)*ea(2)+SuPu(1,iw)*eb(2)
	  SuPu(2,iw)=PuPu(1,iw)*ea(2)+SuPu(1,iw)*eb(2)
      Do il=2,NL-1
		   ilp1=il+1
*          Q=X(il)*N(ilp1)
           Q11=X11(il)*N11(ilp1)+X12(il)*N21(ilp1)
		   Q12=X11(il)*N12(ilp1)+X12(il)*N22(ilp1)
		   Q21=X21(il)*N11(ilp1)+X22(il)*N21(ilp1)
		   Q22=X21(il)*N12(ilp1)+X22(il)*N22(ilp1)
*		   type*,'Q=X*N,Q11,Q22',Q11,Q22
*          V=M-Q
           V11=M11(ilp1)-Q11
		   V12=M12(ilp1)-Q12
		   V21=M21(ilp1)-Q21
		   V22=M22(ilp1)-Q22
*		   type*,'V=M-XN,V11,V22=',V11,V22
*          U=Y*V
           U11=Y11(il)*V11+Y12(il)*V21
		   U12=Y11(il)*V12+Y12(il)*V22
		   U21=Y21(il)*V11+Y22(il)*V21
		   U22=Y21(il)*V12+Y22(il)*V22
*		   type*,'interfacial transmissitivity U=YV U11,U12,U21,U22=',
*     &            U11,U12,U21,U22
*		   pause
		   ea(ilp1)=ea(ilp1)*incea(ilp1)
		   eb(ilp1)=eb(ilp1)*inceb(ilp1)
           PuPu(ilp1,iw)=(PuPu(il,iw)*U11+SuPu(il,iw)*U21)*ea(ilp1)
		   SuPu(ilp1,iw)=(PuPu(il,iw)*U12+SuPu(il,iw)*U22)*eb(ilp1)

		End Do
	   
	  End Do 
!**********************************************************************
!     END FREQUENCY LOOP
!**********************************************************************
	  Return
	  End
	  
	  COMPLEX FUNCTION CMPLXSLO(C,Q)
*     ************************************************************
*     DETERMINE THE FREQUENCY INDEPENDENT RECIPROCAL PROPAGATION 
*     VELOCITY (SLOWNESS) 
*     ************************************************************
	  REAL C,Q,twoQ,twoQsq,twoQsqp1byC
	  twoQ=2*Q
	  twoQsq=twoQ*twoQ
	  twoQsqp1byC=(1+twoQsq)*C
	  cmplxslo=cmplx(twoQsq/twoQsqp1byC,TwoQ/twoQsqp1byC)
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
	  
      COMPLEX FUNCTION S(OMEGA,OmegaMax)
*     ***************************************************************
*     Fourier transform of Ricker wavelet
*     ****************************************************************
	  REAL T0
	  COMPLEX OMEGA,CI,Z,OmegaMax,Alfa
	  CI=(0.0,1.0)
	  alfa=.25*omegamax
	  T0=5.5/(ALFA*SQRT(2.0))
	  Z=OMEGA*OMEGA/(4.*ALFA*ALFA)
	  S=Z*CEXP(-Z+CI*OMEGA*T0)
	  RETURN
	  END
	  
	  

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
      AA=Pi*SIGNI*FLOAT((M-1))/FLOAT(L)
      CW=CMPLX(COS(AA),SIN(AA))
      DO 5 I=M,LX,ISTEP
      CTEMP=CW*CX(I+L)
      CX(I+L)=CX(I)-CTEMP
 5    CX(I)=CX(I)+CTEMP
      L=ISTEP
      IF(L.LT.LX) GO TO 4
      RETURN
      END
