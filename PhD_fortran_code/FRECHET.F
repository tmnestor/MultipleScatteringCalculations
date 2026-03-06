      Program  Frechet
	  Implicit none

	  Integer  Nl,Nlp1,Nlm1
	  Parameter(Nl=3,Nlp1=Nl+1,Nlm1=Nl-1)
      Integer  Np,Nw
	  Parameter(Np=1,Nw=1)
	  Real     w,pvec(Np),T,Dw,twopi,w0




      COMPLEX A(0:NLP1),B(NLP1),BETA(NLP1)
	  COMPLEX ETA(0:NLP1),NETA(NLP1),U(Nw,np)
	  complex dRa(nl),dTa(nl)

	  complex Rd11(0:nl),Rd12(0:nl),Rd21(0:nl),Rd22(0:nl)
	  complex Tu11(0:nl),Tu12(0:nl),Tu21(0:nl),Tu22(0:nl)
	  complex Td11(0:nl),Td12(0:nl),Td21(0:nl),Td22(0:nl)
	  complex Ru11(0:nl),Ru12(0:nl),Ru21(0:nl),Ru22(0:nl)
	  complex ea(0:nl),eb(0:nl),eaeb(0:nl),incea(0:nl)
	  complex eaea(0:nl),ebeb(0:nl),incebeb(0:nl),inceaea(0:nl)
	  complex inceb(0:nl),inceaeb(0:nl)


	  REAL    RHO(0:NLP1),CA0(0:NLP1),CB0(NLP1),DD(0:NL)
	  REAL    QA0(0:NLP1),QB0(NLP1)


      Complex  RR0(Nw),dRR0(Nw,Nl),RRd11(Nl),RRd12(Nl),RRd21(Nl),RRd22(Nl)
	  Complex  TTd11(0:Nlm1),TTd12(Nlm1),TTd21(0:Nlm1),TTd22(Nlm1)
	  Complex  Uu11(Nw,Nl),Uu21(Nw,Nl),Ud11(Nw,Nl),Ud21(Nw,Nl)

      Integer  il,ip,iw
      Complex  cmplxslo,C1
      Complex  RRp,RRm
      Real     del

      DATA     RHO/1.0,1.0,1.20,1.30,1.20/
	  DATA     CA0/1.5,1.75,1.78,1.801,1.82/
	  DATA     CB0/1.6,1.85,1.95,1.85/

	  DATA     QA0/100000,100000,100000,100000,100000/
	  DATA     QB0/100000,100000,100000,100000/


	  DATA     DD/0.0,1.0,1.0,1.0/



	  DATA     W/1.0/
	  DATA     pvec/0.3/


      del=.001
      RRm=(0.182064407478342994,0.186711038399220645) 
      RRp=(0.182094275036855735,0.186739299386934735)

      type*,'dRR=',(RRp-RRm)/(2.*del)
      stop




      C1=(1.0,0.0)

      twoPi=2*acos(-1.0)
      T=20
*      Dw=twoPi/T
      Dw=1.0  !!!!!!!!!!!!!!!!!!!!!!!DELETE
      w0=0.0

*     ****************************************************************
*     define complex frequency independent slownesses of each layer
*     ****************************************************************
      A(0)=CMPLXSLO(CA0(0),QA0(0))
      Do 30 IL=1,NLP1
        A(IL)=CMPLXSLO(CA0(IL),QA0(IL))
		B(IL)=CMPLXSLO(CB0(IL),QB0(IL))
*		TYPE *,'A(IL),B(IL)=',A(IL),B(IL)
*		PAUSE
	    BETA(IL)=C1/B(IL)
 30   continue
	  
      Do 40 ip=1,Np
*     *****************************************************************
*     compute the plane wave respone of the medium for each slowness
*     *****************************************************************
	   CALL SLOWRESP(pvec,ip,np,W0,Nw,Dw,ea,eb,eaeb,Nl,Nlm1,Nlp1,U,
     &      A,B,BETA,RHO,DD,Rd11,Rd12,Rd21,Rd22,Tu11,Tu12,Tu21,Tu22,Ru11,
     &      Ru12,Ru21,Ru22,Td11,Td12,Td21,Td22,eta,neta,incea,inceb,
     &      eaea,ebeb,inceaea,incebeb,inceaeb,RR0,dRR0,RRd11,RRd12,RRd21,RRd22,
     &      TTd11,TTd12,TTd21,TTd22,Uu11,Uu21,Ud11,Ud21,dRa,dTa)

*	    Do iw=1,Nw
        do il=1,Nl
        type*,'il,Ud11(iw,il)=',il,Ud11(Nw,il)
        type*,'il,Ud21(iw,il)=',il,Ud21(Nw,il)
        type*,'il,Uu11(iw,il)=',il,Uu11(Nw,il)
        type*,'il,Uu21(iw,il)=',il,Uu21(Nw,il)

        enddo
*        enddo


	    Do iw=1,Nw
          type*,'iw,RR0(iw)=',iw,RR0(iw)
        enddo

	    Do iw=1,Nw
          do il=1,Nl
            type*,'iw,il,dRR0(iw,il)=',iw,il,dRR0(iw,il)
          enddo
        enddo


*     *****************************************************************
*     End Slowness Loop
*     *****************************************************************
 40   continue



	  STOP 'normal completion'
	  END


      SUBROUTINE SLOWRESP(pvec,ip,np,W0,Nw,Dw,ea,eb,eaeb,Nl,Nlm1,Nlp1,U,
     &      A,B,BETA,RHO,DD,Rd11,Rd12,Rd21,Rd22,Tu11,Tu12,Tu21,Tu22,Ru11,
     &      Ru12,Ru21,Ru22,Td11,Td12,Td21,Td22,eta,neta,incea,inceb,
     &      eaea,ebeb,inceaea,incebeb,inceaeb,RR0,dRR0,RRd11,RRd12,RRd21,RRd22,
     &      TTd11,TTd12,TTd21,TTd22,Uu11,Uu21,Ud11,Ud21,dRa,dTa)
	  
!**********************************************************************
!     CALCULATE THE HORIZONTEL SLOWNESS DEPENDENT COMPONENTS OF THE
!     REFLECTIVITY INTEGRAND
!**********************************************************************
	  Implicit none
	  Integer  Nl,Nlp1,Nlm1,il,ilp1,ilm1,ip
      Integer  Np,Nw
	  Real     w,pvec(Np),W0,Dw





      COMPLEX A(0:NLP1),B(NLP1),BETA(NLP1)
	  COMPLEX ETA(0:NLP1),NETA(NLP1),U(Nw,np)
	  complex dRa(nl),dTa(nl)

	  complex Rd11(0:nl),Rd12(0:nl),Rd21(0:nl),Rd22(0:nl)
	  complex Tu11(0:nl),Tu12(0:nl),Tu21(0:nl),Tu22(0:nl)
	  complex Td11(0:nl),Td12(0:nl),Td21(0:nl),Td22(0:nl)
	  complex Ru11(0:nl),Ru12(0:nl),Ru21(0:nl),Ru22(0:nl)
	  complex ea(0:nl),eb(0:nl),eaeb(0:nl),incea(0:nl)
	  complex eaea(0:nl),ebeb(0:nl),incebeb(0:nl),inceaea(0:nl)
	  complex inceb(0:nl),inceaeb(0:nl)


	  REAL    RHO(0:NLP1),DD(0:NL),p


      Complex  RR0(Nw),dRR0(Nw,Nl),RRd11(Nl),RRd12(Nl),RRd21(Nl),RRd22(Nl)
	  Complex  TTd11(0:Nlm1),TTd12(Nlm1),TTd21(0:Nlm1),TTd22(Nlm1)
	  Complex  Uu11(Nw,Nl),Uu21(Nw,Nl),Ud11(Nw,Nl),Ud21(Nw,Nl)
	  
	  COMPLEX  Ci,C1,C0	  
	  PARAMETER (Ci=(0.0,1.0),C1=(1.0,0.0),C0=(0.0,0.0))
	  Complex  VERTICALSLO,cp
	  
*     Initialise frequency independent layer parameters
      p=pvec(ip)
	  cp=cmplx(p)


	  ETA(0)=VERTICALSLO(A(0),CP)
	  ea(0)=cexp(Ci*W0*DD(0)*eta(0))
	  incea(0)=cexp(Ci*Dw*DD(0)*eta(0))
	  eaea(0)=ea(0)*ea(0)
	  inceaea(0)=incea(0)*incea(0)
      DO IL=1,NL
	    ETA(IL)=VERTICALSLO(A(IL),CP)
		NETA(IL)=VERTICALSLO(B(IL),CP)
        ea(il)=cexp(Ci*W0*eta(il)*DD(il))
		eb(il)=cexp(Ci*W0*neta(il)*DD(il))
		eaea(il)=ea(il)*ea(il)
		ebeb(il)=eb(il)*eb(il)
		eaeb(il)=ea(il)*eb(il)
		incea(il)=cexp(Ci*Dw*eta(il)*DD(il))
		inceb(il)=cexp(Ci*Dw*neta(il)*DD(il))
		inceaea(il)=incea(il)*incea(il)
		incebeb(il)=inceb(il)*inceb(il)
		inceaeb(il)=incea(il)*inceb(il)
        dRa(il)=(A(il)*A(il)*A(il))/(2.0*(A(il)*A(il)-p**2))
        dTa(il)=-Ci*DD(il)*(A(il)*A(il)*A(il))/eta(il)
	  END DO
	  ETA(Nlp1)=VERTICALSLO(A(Nlp1),CP)
	  NETA(Nlp1)=VERTICALSLO(B(Nlp1),CP)

*
*     *******************************************************************
*     calculate the interfacial scattering coefficients of the layerstack
*     *******************************************************************
*
      call OBSMat(p,eta(0),rho(0),eta(1),neta(1),rho(1),beta(1),Rd11(0),
     &     Td11(0),Td21(0),Tu11(0),Ru11(0),Ru21(0),Tu12(0),Ru12(0),Ru22(0))
*     Calculate the Interfacial Scattering Coefficients
	  Do il=1,nl
	    ilp1=il+1
	    call ScatMat(p,eta(il),neta(il),rho(il),beta(il),eta(ilp1),
     &  neta(ilp1),rho(ilp1),beta(ilp1),Rd11(il),Rd12(il),Rd21(il),
     &  Rd22(il),Tu11(il),Tu12(il),Tu21(il),Tu22(il),Td11(il),Td12(il),
     &  Td21(il),Td22(il),Ru11(il),Ru12(il),Ru21(il),Ru22(il))
	  End Do  

*     CALCULATE THE OCEAN CRUST REFLECTIVITY (phase related to Ds)
      call DSFS(Nl,Nw,w0,dw,inceaea,incebeb,inceaeb,eaea,ebeb,eaeb,ea,eb,incea,inceb,
     &           Rd11,Rd12,Rd21,Rd22,Td11,Td12,Td21,Td22,Tu11,Tu12,Tu21,Tu22,
     &           Ru11,Ru12,Ru21,Ru22,TTd11,TTd12,TTd21,TTd22,RR0,dRR0,RRd11,RRd12,RRd21,
     &           RRd22,Uu11,Uu21,Ud11,Ud21,dRa,dTa)

      RETURN
      END	  
	  


      Subroutine DSFS(Nl,Nw,w0,dw,inceaea,incebeb,inceaeb,eaea,ebeb,eaeb,ea,eb,incea,inceb,
     &           Rd11,Rd12,Rd21,Rd22,Td11,Td12,Td21,Td22,Tu11,Tu12,Tu21,Tu22,
     &           Ru11,Ru12,Ru21,Ru22,TTd11,TTd12,TTd21,TTd22,RR0,dRR0,RRd11,RRd12,RRd21,
     &           RRd22,Uu11,Uu21,Ud11,Ud21,dRa,dTa)
	 
*     ***************************************************************
*     doublesweep-Free surface case
*     ***************************************************************

	  Implicit None
	  Integer Nw,iw,il,jl,Nl,Nlm1,Nlp1,ilm1,ilp1
      Real    w0,dw,w
	  Complex InvDet,a,b,c,d,C1,D1,D2
	  Complex RRt11,RRt21,RRt12,RRt22,RRb11,RRb21,RRb12,RRb22
	  complex ea(0:nl),eb(0:nl),eaeb(0:nl),incea(0:nl)
	  complex eaea(0:nl),ebeb(0:nl),incebeb(0:nl),inceaea(0:nl)
	  complex inceb(0:nl),inceaeb(0:nl)

	  complex dRa(nl),dTa(nl),drda,dtda

      Complex  RR0(Nw),dRR0(Nw,Nl),RRd11(Nl),RRd12(Nl),RRd21(Nl),RRd22(Nl)
	  Complex  TTd11(0:Nlm1),TTd12(Nlm1),TTd21(0:Nlm1),TTd22(Nlm1)
      Complex  Rd11(0:Nl),Rd12(0:Nl),Rd21(0:Nl),Rd22(0:Nl)
	  Complex  Ru11(0:Nl),Ru12(0:Nl),Ru21(0:Nl),Ru22(0:Nl)
	  Complex  Tu11(0:Nl),Tu12(0:Nl),Tu21(0:Nl),Tu22(0:Nl)
	  Complex  Td11(0:Nl),Td12(0:Nl),Td21(0:Nl),Td22(0:Nl)
	  Complex  Uu11(Nw,Nl),Uu21(Nw,Nl),Ud11(Nw,Nl),Ud21(Nw,Nl)


	  C1=Cmplx(1.0,0.0)
      w=w0
	  Nlp1=Nl+1


*     ***************************************************************************
*                          UP SWEEP
*     ***************************************************************************


	  RRd11(Nl)=Rd11(Nl)
      RRd12(Nl)=Rd12(Nl)
      RRd21(Nl)=Rd21(Nl)
      RRd22(Nl)=Rd22(Nl)

      Do 1000 iw=1,Nw
      w=w+dw
      type*,'iw=',iw
      Do il= Nl-1,1,-1 
	    ilp1=il+1
*		RRt=E*RRd*E

	    eaea(ilp1)=eaea(ilp1)*inceaea(ilp1)
	    ebeb(ilp1)=ebeb(ilp1)*incebeb(ilp1)
	    eaeb(ilp1)=eaeb(ilp1)*inceaeb(ilp1)

        RRt11=RRd11(ilp1)*eaea(ilp1)
        RRt12=RRd12(ilp1)*eaeb(ilp1)
        RRt21=RRd21(ilp1)*eaeb(ilp1)
        RRt22=RRd22(ilp1)*ebeb(ilp1)

*       set (1-Ru(il)*RRt)^(-1)	  
	    a=(C1-RRt22*Ru22(il)-RRt12*Ru21(il))
        b=(RRt22*Ru12(il)+RRt12*Ru11(il))	  
        c=(RRt21*Ru22(il)+RRt11*Ru21(il))
        d=(C1-RRt21*Ru12(il)-RRt11*Ru11(il))
        InvDet=C1/(a*d-b*c)
	  
	    a=InvDet*a
	    b=InvDet*b
	    c=InvDet*c
	    d=InvDet*d

*       set TTd(il)=(1-Ru(il)*RRt)^(-1)*Td(il)	  

        TTd11(il)=Td11(il)*a+Td21(il)*b
        TTd12(il)=Td12(il)*a+Td22(il)*b
        TTd21(il)=Td11(il)*c+Td21(il)*d
        TTd22(il)=Td12(il)*c+Td22(il)*d

*       set RRd(il)=Rd(il)+Tu(il)*RRt*TTd(il)	  
      
        RRd11(il)=Rd11(il)+RRt22*TTd21(il)*Tu12(il)+RRt21*TTd11(il)*Tu12(il)+
     &          RRt12*TTd21(il)*Tu11(il)+RRt11*TTd11(il)*Tu11(il)

        RRd12(il)=Rd12(il)+RRt11*Tu11(il)*TTd12(il)+RRt12*Tu11(il)*TTd22(il)+
     &          RRt21*Tu12(il)*TTd12(il)+RRt22*Tu12(il)*TTd22(il)


        RRd21(il)=Rd21(il)+RRt22*TTd21(il)*Tu22(il)+RRt21*TTd11(il)*Tu22(il)+
     &          RRt12*TTd21(il)*Tu21(il)+RRt11*TTd11(il)*Tu21(il)

        RRd22(il)=Rd22(il)+RRt22*TTd22(il)*Tu22(il)+RRt21*TTd12(il)*Tu22(il)+
     &          RRt12*TTd22(il)*Tu21(il)+RRt11*TTd12(il)*Tu21(il)
	  End Do


*     ***************************************************************************
*                          OCEAN BOTTOM
*     ***************************************************************************
*		RRt=E*RRd*E
	    eaea(1)=eaea(1)*inceaea(1)
	    ebeb(1)=ebeb(1)*incebeb(1)
	    eaeb(1)=eaeb(1)*inceaeb(1)

        RRt11=RRd11(1)*eaea(1)
        RRt12=RRd12(1)*eaeb(1)
        RRt21=RRd21(1)*eaeb(1)
        RRt22=RRd22(1)*ebeb(1)

*       set (1-Ru(0)*RRt)^(-1)	  
	    a=(C1-RRt22*Ru22(0)-RRt12*Ru21(0))
        b=(RRt22*Ru12(0)+RRt12*Ru11(0))	  
        c=(RRt21*Ru22(0)+RRt11*Ru21(0))
        d=(C1-RRt21*Ru12(0)-RRt11*Ru11(0))
        InvDet=C1/(a*d-b*c)
	  
	    a=InvDet*a
	    b=InvDet*b
	    c=InvDet*c
	    d=InvDet*d

*       set TTd(0)=(1-Ru(0)*RRt)^(-1)*Td(0)	  

        TTd11(0)=Td11(0)*a+Td21(0)*b
        TTd21(0)=Td11(0)*c+Td21(0)*d

*       set OCEAN BOTTOM REFLECTIVITY	  
      
        RR0(iw)=Rd11(0)+Tu11(0)*(RRt12*TTd21(0)+RRt11*TTd11(0))+
     &         Tu12(0)*(RRt22*TTd21(0)+RRt21*TTd11(0))
 
        type*,'iw,RR0(iw)=',iw,RR0(iw)
*     ***************************************************************************
*                          DOWN SWEEP
*     ***************************************************************************

      Ud11(iw,1)=TTd11(0)  
      Ud21(iw,1)=TTd21(0)  
	  Do il=1,Nl-1
	    ilp1=il+1

	    ea(il)=ea(il)*incea(il)
	    eb(il)=eb(il)*inceb(il)
        
		D1=Ud11(iw,il)*ea(il)
		D2=Ud21(iw,il)*eb(il)

        Uu11(iw,il)=RRd11(il)*D1+RRd12(il)*D2
        Uu21(iw,il)=RRd21(il)*D1+RRd22(il)*D2

        drda=(C1-eaea(il))*dRa(il)
        dtda=w*ea(il)*dTa(il)

        type*,'il,drda=',il,drda
        type*,'il,dtda=',il,dtda

        DRR0(iw,il)=Uu11(iw,il)*(Ud11(iw,il)*dtda+Uu11(iw,il)*drda)+
     &              Ud11(iw,il)*(Ud11(iw,il)*drda+Uu11(iw,il)*dtda)

        Ud11(iw,ilp1)=TTd11(il)*D1+TTd12(il)*D2
        Ud21(iw,ilp1)=TTd21(il)*D1+TTd22(il)*D2
		
      End Do 

	  ea(Nl)=ea(Nl)*incea(Nl)
	  eb(Nl)=eb(Nl)*inceb(Nl)


      D1=Ud11(iw,Nl)*ea(Nl)
	  D2=Ud21(iw,Nl)*eb(Nl)

      Uu11(iw,Nl)=RRd11(Nl)*D1+RRd12(Nl)*D2
      Uu21(iw,Nl)=RRd21(Nl)*D1+RRd22(Nl)*D2

      drda=(C1-eaea(Nl))*dRa(Nl)
      dtda=w*ea(Nl)*dTa(Nl)

      DRR0(iw,Nl)=Uu11(iw,Nl)*(Ud11(iw,Nl)*dtda+Uu11(iw,Nl)*drda)+
     &            Ud11(iw,Nl)*(Ud11(iw,Nl)*drda+Uu11(iw,Nl)*dtda)

1000  Continue

      return
	  End


	  Subroutine  ScatMat(p,eta1,neta1,rho1,beta1,eta2,neta2,rho2,beta2,
     &                     Rd11,Rd12,Rd21,Rd22,Tu11,Tu12,Tu21,Tu22,
     &                     Td11,Td12,Td21,Td22,Ru11,Ru12,Ru21,Ru22)
*     ********************************************************************
*     modified interfacial scattering matrices (see Aki&Richards pp149-51)
*     ********************************************************************
	  REAL RHO1,RHO2,rtrho1,rtrho2,p
	  COMPLEX ETA1,NETA1,ETA2,NETA2,BETA1,BETA2,PSQ,CRHO1,CRHO2,DRHO,DMU
	  COMPLEX rteta1,rteta2,rtneta1,rtneta2,Ci,m2Ci,Rd11,Rd12,Rd21,Rd22
      COMPLEX Tu11,Tu12,Tu21,Tu22,Td11,Td12,Td21,Td22,Ru11,Ru12,Ru21,Ru22
	  COMPLEX PSQD,a,b,c,d,Det,E,F,G,H,Q,R,S,T,U,V,rtza1,rtza2,rtzb1,rtzb2
	  Ci=(0.0,1.0)
	  m2Ci=(0.0,-2.0)
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
	  crho1=cmplx(rho1,0.0)
	  crho2=cmplx(rho2,0.0)
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
	  Rd21=m2ci*rteta1*rtneta1*S
	  Rd12=Rd21
	  Rd22=T-U
	  Td11=2*rtza1*rtza2*F
	  Td21=m2ci*rtza1*rtzb2*H*p
	  Td12=m2ci*rtzb1*rtza2*G*p
	  Td22=2*rtzb1*rtzb2*E
	  Tu11=Td11
	  Tu21=Td12
	  Tu12=Td21
	  Tu22=Td22
	  Ru11=-(Q+U)
	  Ru21=m2ci*rteta2*rtneta2*V
	  Ru12=Ru21
	  Ru22=-(T+R)
	  
	  return
	  end	  
	  
	  SUBROUTINE OBSMat(P,ETA1,RHO1,ETA2,NETA2,RHO2,BETA2,
     +           PdPu,PdPd,PdSd,PuPu,PuPd,PuSd,SuPu,SuPd,SuSd)
*     calculate scattering matrix of the sea floor
      Implicit none
	  REAL RHO1,RHO2,rtrho1,rtrho2,p,PSQ
	  COMPLEX ETA1,ETA2,NETA2,BETA2,CRHO1,CRHO2
	  COMPLEX T1,T2,T4,T5,T6,a,b,c,d,Det,drho,dmu,psqd
	  complex rteta1,rteta2,rtneta2,Ci,mCi,E,F,G,H,rtza1,rtza2,rtzb2
	  COMPLEX PdPu,PdPd,PdSd,PuPu,PuPd,PuSd,SuPu,SuPd,SuSd
	  Ci=(0.0,1.0)
	  mCi=(0.0,-1.0)
	  rtrho1=sqrt(rho1)
	  rtrho2=sqrt(rho2)
	  rteta1=csqrt(eta1)
	  rteta2=csqrt(eta2)
	  rtneta2=csqrt(neta2)
	  rtza1=rtrho1*rteta1
	  rtza2=rtrho2*rteta2
	  rtzb2=rtrho2*rtneta2
	  psq=p*p
	  crho1=cmplx(rho1,0.0)
	  crho2=cmplx(rho2,0.0)
	  drho=crho2-crho1
	  dmu=crho2*beta2*beta2

	  d=2.0*dmu
	  psqd=psq*d
	  a=drho-psqd
	  b=crho2-psqd
	  c=crho1+psqd
	  E=b*eta1+c*eta2
	  F=b
	  G=a-d*eta1*neta2
	  H=-d*eta2
	  Det=E*F+G*H*psq
      E=E/Det
	  F=F/Det
	  G=G/Det
	  H=H/Det
	  T1=(b*eta1-c*eta2)*F
	  T2=(a+d*eta1*neta2)*H*psq
	  T4=b*E
	  T5=d*eta2*G*psq
	  T6=b*d*eta1*p/Det
	  PdPu=(T1-T2)
	  PdPd=2.0*rtza1*rtza2*F
	  PdSd=2.0*mCi*rtza1*rtzb2*H*p
	  PuPu=PdPd
	  SuPu=PdSd
	  PuPd=-(T1+T5)
	  PuSd=2*mci*rteta2*rtneta2*T6
	  SuPd=PuSd
	  SuSd=-(T2+T4)
	  return
	  end	
	  
	  
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
