      Program  setJost
	  Implicit none

	  Integer  il,ilp1,Nl,Nlp1,np,ilm1,ip
	  Parameter(np=10000,Nl=2,Nlp1=Nl+1) 

      Real     rho0(0:Nlp1),alpha0(0:Nlp1),beta0(0:Nlp1)
	  Real     w,p,h(0:Nl),dp,Det,eps
	  parameter(eps=.1)

      Complex  RRd11(0:Nlp1),RRd12(0:Nlp1),RRd21(0:Nlp1),RRd22(0:Nlp1)
	  Complex  RRu11(0:Nlp1),RRu12(0:Nlp1),RRu21(0:Nlp1),RRu22(0:Nlp1)
	  Complex  TTu11(0:Nlp1),TTu12(0:Nlp1),TTu21(0:Nlp1),TTu22(0:Nlp1)
	  Complex  TTd11(0:Nlp1),TTd12(0:Nlp1),TTd21(0:Nlp1),TTd22(0:Nlp1)
      Complex  Rd11(0:Nl),Rd12(0:Nl),Rd21(0:Nl),Rd22(0:Nl)
	  Complex  Ru11(0:Nl),Ru12(0:Nl),Ru21(0:Nl),Ru22(0:Nl)
	  Complex  Tu11(0:Nl),Tu12(0:Nl),Tu21(0:Nl),Tu22(0:Nl)
	  Complex  Td11(0:Nl),Td12(0:Nl),Td21(0:Nl),Td22(0:Nl)
	  Complex  Wu11(0:Nlp1),Wu12(0:Nlp1),Wu21(0:Nlp1),Wu22(0:Nlp1)
	  Complex  Wd11(0:Nlp1),Wd12(0:Nlp1),Wd21(0:Nlp1),Wd22(0:Nlp1)
	  Complex  Uu11(0:Nlp1),Uu12(0:Nlp1),Uu21(0:Nlp1),Uu22(0:Nlp1)
	  Complex  Ud11(0:Nlp1),Ud12(0:Nlp1),Ud21(0:Nlp1),Ud22(0:Nlp1)
	  Complex  Vu11(0:Nlp1),Vu12(0:Nlp1),Vu21(0:Nlp1),Vu22(0:Nlp1)
	  Complex  Vd11(0:Nlp1),Vd12(0:Nlp1),Vd21(0:Nlp1),Vd22(0:Nlp1)
	  Complex  Gu11(0:Nlp1),Gu12(0:Nlp1),Gu21(0:Nlp1),Gu22(0:Nlp1)
	  Complex  Gd11(0:Nlp1),Gd12(0:Nlp1),Gd21(0:Nlp1),Gd22(0:Nlp1)
      Complex  Rf11,Rf12,Rf21,Rf22,InvUv11,InvUv12,InvUv21,InvUv22,InvDet
	  Complex  Nd(2,2),Md(2,2),Nu(2,2),Mu(2,2)
	  Complex  Ci,C0,C1,Wt11,Wt12,Wt21,Wt22,t11,t12,t21,t22
	  Complex  ea0(0:Nl),eb0(0:Nl)
	  Complex  eta0(0:Nlp1),neta0(0:Nlp1)
	  Complex  Temp,VERTICALSLO,Csqrt,cbeta1,cbeta2

	  Logical  Surf,reflex,Pin,Pou



	  DATA     alpha0/1.51,2.01,2.51,3.51/
	  DATA     beta0/1.01,1.51,1.71,1.91/	
      DATA     rho0/1.01,1.21,1.31,1.41/
	  DATA     W,h/12.0,0.5,.1,.1/
      DATA     Surf,reflex,Pin,Pou/.true.,.false.,.true.,.false./
	  If(Surf.and.reflex) STOP 'Freesurface/downward reflexion unimplemented'
	  CI=(0.0,1.0)
	  C0=(0.0,0.0)
	  C1=(1.0,0.0)
	  if(.not.Surf) H(0)=0.0

	  dp=((1.0/beta0(0))-(1.0/beta0(Nlp1)))/float(np-1)

	  type*,'dp=',dp
	  p=(1.0/beta0(Nlp1))-dp
	  Do  ip=1,np
	  p=p+dp
	  p=0.679779541947369299
	  do il=0,Nlp1
	     eta0(il)=VERTICALSLO(alpha0(il),p)
	     neta0(il)=VERTICALSLO(beta0(il),p)
      end do !il
	  
      Do il=0,Nl
	    Temp=CI*w*h(il)*eta0(il)
        EA0(il)=CEXP(Temp)
	    Temp=CI*w*h(il)*neta0(il)
        EB0(il)=CEXP(Temp)		
      End Do

*     *******************************************************
*     set free surface reflection array Rf11,Rf12,Rf21,Rf22
*     and free-surface amplitude factor
*     *******************************************************
        call Emat(p,eta0(0),neta0(0),rho0(0),beta0(0),Md,Nd,Mu,Nu)
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
*     set (Wf)^(#)= transpose of the free-space amplification = Ci*Nd^(-1)
		Wt11=Ci*InvDet*Nd(2,2)
		Wt22=Ci*InvDet*Nd(1,1)
		Wt12=-Ci*InvDet*Nd(1,2)
		Wt21=-Ci*InvDet*Nd(2,1)

*     *******************************************************
*     set reference medium Jost solutions
	  do il=0,Nl
	    ilp1=il+1
	  	cbeta1=cmplx(beta0(il),0.0)
	  	cbeta2=cmplx(beta0(ilp1),0.0)
*       set the reference medium interfacial scattering matrices
		call ScatMat(p,eta0(il),neta0(il),rho0(il),CBeta1,eta0(ilp1),
     &     neta0(ilp1),rho0(ilp1),CBeta2,Rd11(il),Rd12(il),Rd21(il),
     &     Rd22(il),Tu11(il),Tu12(il),Tu21(il),Tu22(il),Td11(il),
     &     Td12(il),Td21(il),Td22(il),Ru11(il),Ru12(il),Ru21(il),Ru22(il))
	  end do
*     Set the Jost solutions Ud,Uu,Vd,Vu in each of the Nl layers
      call DSFS(Det,Nl,ea0,eb0,Rd11,Rd12,Rd21,Rd22,Td11,Td12,Td21,Td22,
     &     Tu11,Tu12,Tu21,Tu22,Ru11,Ru12,Ru21,Ru22,TTu11,TTu12,TTu21,
     &     TTu22,TTd11,TTd12,TTd21,TTd22,RRd11,RRd12,RRd21,RRd22,RRu11,
     &     RRu12,RRu21,RRu22,Uu11,Uu12,Uu21,Uu22,Ud11,Ud12,Ud21,Ud22,
     &     Vu11,Vu12,Vu21,Vu22,Vd11,Vd12,Vd21,Vd22,RF11,RF12,RF21,RF22,
     &     Wt11,Wt12,Wt21,Wt22)



*       SET inverse propagation invariant
*        t11=-Ud11(Nlp1)
*        t22=-Ud22(Nlp1)
*        t21=Ud12(Nlp1)
*        t12=Ud21(Nlp1)
*	    Det=Cabs(t11*t22-t12*t21)
		If(Det.lt.eps) then
	       type*,'p,Det=',p,Det
		Endif
      Enddo !ip

	  do il=0,Nl
        type*,'il,Vu11(il)=',il,Vu11(il)
        type*,'il,Vu12(il)=',il,Vu12(il)
        type*,'il,Vu21(il)=',il,Vu21(il)
        type*,'il,Vu22(il)=',il,Vu22(il)

        type*,'il,Vd11(il)=',il,Vd11(il)
        type*,'il,Vd12(il)=',il,Vd12(il)
        type*,'il,Vd21(il)=',il,Vd21(il)
        type*,'il,Vd22(il)=',il,Vd22(il)

        enddo
		
	    do il=0,Nl
        type*,'il,Uu11(il)=',il,Uu11(il)
        type*,'il,Uu12(il)=',il,Uu12(il)
        type*,'il,Uu21(il)=',il,Uu21(il)
        type*,'il,Uu22(il)=',il,Uu22(il)

        type*,'il,Ud11(il)=',il,Ud11(il)
        type*,'il,Ud12(il)=',il,Ud12(il)
        type*,'il,Ud21(il)=',il,Ud21(il)
        type*,'il,Ud22(il)=',il,Ud22(il)

        enddo
	  do il=1,Nl
        type*,'il,Uu11(il)/Vu11(il)=',il,Uu11(il)/Vu11(il)
        type*,'il,Uu21(il)/Vu21(il)=',il,Uu21(il)/Vu21(il)
        type*,'il,Ud11(il)/Vd11(il)=',il,Ud11(il)/Vd11(il)
        type*,'il,Ud21(il)/Vd21(il)=',il,Ud21(il)/Vd21(il)


        type*,'il,Uu12(il)/Vu12(il)=',il,Uu12(il)/Vu12(il)
        type*,'il,Uu22(il)/Vu22(il)=',il,Uu22(il)/Vu22(il)
        type*,'il,Ud12(il)/Vd12(il)=',il,Ud12(il)/Vd12(il)
        type*,'il,Ud22(il)/Vd22(il)=',il,Ud22(il)/Vd22(il)

      enddo

	  STOP 'normal completion'
	  END

	  
	  
      Subroutine DSFS(Det,Nl,ea,eb,Rd11,Rd12,Rd21,Rd22,Td11,Td12,Td21,Td22,
     &           Tu11,Tu12,Tu21,Tu22,Ru11,Ru12,Ru21,Ru22,TTu11,TTu12,TTu21,
     &           TTu22,TTd11,TTd12,TTd21,TTd22,RRd11,RRd12,RRd21,RRd22,RRu11,
     &           RRu12,RRu21,RRu22,Uu11,Uu12,Uu21,Uu22,Ud11,Ud12,Ud21,Ud22,
     &           Vu11,Vu12,Vu21,Vu22,Vd11,Vd12,Vd21,Vd22,RF11,RF12,RF21,RF22,
     &           Wt11,Wt12,Wt21,Wt22)
	 
*     ***************************************************************
*     doublesweep-Free surface case
*     ***************************************************************

	  Implicit None
	  Integer il,jl,Nl,Nlm1,Nlp1,ilm1,ilp1
	  Real    Det
	  Complex InvDet,a,b,c,d,C1,Wt11,Wt12,Wt21,Wt22
	  Complex t11,t12,t21,t22
	  Complex RRt11,RRt21,RRt12,RRt22,RRb11,RRb21,RRb12,RRb22
	  Complex ea(0:Nl),eb(0:Nl)
	  Complex RRd11(0:Nl+1),RRd12(0:Nl+1),RRd21(0:Nl+1),RRd22(0:Nl+1)
	  Complex RRu11(0:Nl+1),RRu12(0:Nl+1),RRu21(0:Nl+1),RRu22(0:Nl+1)
	  Complex TTu11(0:Nl+1),TTu12(0:Nl+1),TTu21(0:Nl+1),TTu22(0:Nl+1)
	  Complex TTd11(0:Nl+1),TTd12(0:Nl+1),TTd21(0:Nl+1),TTd22(0:Nl+1)
	  Complex Tu11(0:Nl),Tu12(0:Nl),Tu21(0:Nl),Tu22(0:Nl)
	  Complex Td11(0:Nl),Td12(0:Nl),Td21(0:Nl),Td22(0:Nl)
	  Complex Rd11(0:Nl),Rd12(0:Nl),Rd21(0:Nl),Rd22(0:Nl)
	  Complex Ru11(0:Nl),Ru12(0:Nl),Ru21(0:Nl),Ru22(0:Nl)
      Complex RF11,RF12,RF21,RF22
	  Complex Uu11(0:Nl+1),Uu12(0:Nl+1),Uu21(0:Nl+1),Uu22(0:Nl+1)
	  Complex Ud11(0:Nl+1),Ud12(0:Nl+1),Ud21(0:Nl+1),Ud22(0:Nl+1)
	  Complex Vu11(0:Nl),Vu12(0:Nl),Vu21(0:Nl),Vu22(0:Nl)
	  Complex Vd11(0:Nl),Vd12(0:Nl),Vd21(0:Nl),Vd22(0:Nl)
      Complex R11,R12,R21,R22

	  C1=Cmplx(1.0,0.0)
	  Nlp1=Nl+1
	  RRd11(Nl)=Rd11(Nl)
      RRd12(Nl)=Rd12(Nl)
      RRd21(Nl)=Rd21(Nl)
      RRd22(Nl)=Rd22(Nl)

	  TTd11(Nl)=Td11(Nl)
      TTd12(Nl)=Td12(Nl)
      TTd21(Nl)=Td21(Nl)
      TTd22(Nl)=Td22(Nl)
       
      Do il= Nl-1,0,-1 
	    ilp1=il+1
*		RRt=E*RRd*E
        RRt11=RRd11(ilp1)*ea(ilp1)*ea(ilp1)
        RRt12=RRd12(ilp1)*ea(ilp1)*eb(ilp1)
        RRt21=RRd21(ilp1)*ea(ilp1)*eb(ilp1)
        RRt22=RRd22(ilp1)*eb(ilp1)*eb(ilp1)

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

*     free-surface interaction
*		RRt=E*RRd*E
        RRt11=RRd11(0)*ea(0)*ea(0)
        RRt12=RRd12(0)*ea(0)*eb(0)
        RRt21=RRd21(0)*ea(0)*eb(0)
        RRt22=RRd22(0)*eb(0)*eb(0)


	    a=(C1-RRt22*Rf22-RRt12*Rf21)
        b=(RRt22*Rf12+RRt12*Rf11)	  
        c=(RRt21*Rf22+RRt11*Rf21)
        d=(C1-RRt21*Rf12-RRt11*Rf11)
		Det=Cabs(a*d-b*c)
*		type*,'check this, Det=',Det
        InvDet=C1/(a*d-b*c)
	  
	    a=InvDet*a
	    b=InvDet*b
	    c=InvDet*c
	    d=InvDet*d
        Ud11(0)=Wt11*a+Wt21*b
        Ud12(0)=Wt12*a+Wt22*b
        Ud21(0)=Wt11*c+Wt21*d
        Ud22(0)=Wt12*c+Wt22*d
	  
	  Do il=1,Nl
	    ilm1=il-1
        Ud11(il)=TTd11(ilm1)*Ud11(ilm1)*ea(ilm1)+TTd12(ilm1)*Ud21(ilm1)*eb(ilm1)
        Ud12(il)=TTd11(ilm1)*Ud12(ilm1)*ea(ilm1)+TTd12(ilm1)*Ud22(ilm1)*eb(ilm1)
        Ud21(il)=TTd21(ilm1)*Ud11(ilm1)*ea(ilm1)+TTd22(ilm1)*Ud21(ilm1)*eb(ilm1)
        Ud22(il)=TTd21(ilm1)*Ud12(ilm1)*ea(ilm1)+TTd22(ilm1)*Ud22(ilm1)*eb(ilm1)
        
        Uu11(il)=RRd11(il)*Ud11(il)*ea(il)+RRd12(il)*Ud21(il)*eb(il)
        Uu12(il)=RRd11(il)*Ud12(il)*ea(il)+RRd12(il)*Ud22(il)*eb(il)
        Uu21(il)=RRd21(il)*Ud11(il)*ea(il)+RRd22(il)*Ud21(il)*eb(il)
        Uu22(il)=RRd21(il)*Ud12(il)*ea(il)+RRd22(il)*Ud22(il)*eb(il)
		
      End Do 

*     set Ud,Uu in lower halfspace
      Ud11(Nlp1)=Td11(Nl)*Ud11(Nl)*ea(Nl)+Td12(Nl)*Ud21(Nl)*eb(Nl)
      Ud12(Nlp1)=Td11(Nl)*Ud12(Nl)*ea(Nl)+Td12(Nl)*Ud22(Nl)*eb(Nl)
      Ud21(Nlp1)=Td21(Nl)*Ud11(Nl)*ea(Nl)+Td22(Nl)*Ud21(Nl)*eb(Nl)
      Ud22(Nlp1)=Td21(Nl)*Ud12(Nl)*ea(Nl)+Td22(Nl)*Ud22(Nl)*eb(Nl)
        
      Uu11(Nlp1)=(0.0,0.0)
      Uu12(Nlp1)=(0.0,0.0)
      Uu21(Nlp1)=(0.0,0.0)
      Uu22(Nlp1)=(0.0,0.0)
      		
	  RRu11(0)=Rf11
      RRu12(0)=Rf12
      RRu21(0)=Rf21
      RRu22(0)=Rf22
*	  RRu11(0)=(0.0,0.0)
*      RRu12(0)=(0.0,0.0)
*      RRu21(0)=(0.0,0.0)
*      RRu22(0)=(0.0,0.0)
	   Do il=0,Nl
	    ilp1=il+1
        RRb11=RRu11(il)*ea(il)*ea(il)
        RRb12=RRu12(il)*ea(il)*eb(il)
        RRb21=RRu21(il)*ea(il)*eb(il)
        RRb22=RRu22(il)*eb(il)*eb(il)

*       set (1-Rd(il)*RRb)^(-1)
        a=C1-RRb22*Rd22(il)-RRb12*Rd21(il)
        b=RRb22*Rd12(il)+RRb12*Rd11(il)
        c=RRb21*Rd22(il)+RRb11*Rd21(il)
        d=C1-RRb21*Rd12(il)-RRb11*Rd11(il)
        InvDet=C1/(a*d-b*c)
        a=a*InvDet
        b=b*InvDet
        c=c*InvDet
        d=d*InvDet

*       set TTu(il)=(1-Rd(il)*RRb)^(-1)*Tu(il)

        TTu11(il)=Tu11(il)*a+Tu21(il)*b
        TTu12(il)=Tu12(il)*a+Tu22(il)*b
        TTu21(il)=Tu11(il)*c+Tu21(il)*d
        TTu22(il)=Tu12(il)*c+Tu22(il)*d
		

*       set RRu(ilp1)=Ru(il)+Td(il)*RRb*TTu(il)

        RRu11(ilp1)=Ru11(il)+RRb22*TTu21(il)*Td12(il)+RRb21*TTu11(il)*Td12(il)+
     &            RRb12*TTu21(il)*Td11(il)+RRb11*TTu11(il)*Td11(il)

        RRu12(ilp1)=Ru12(il)+Td11(il)*RRb11*TTu12(il)+Td11(il)*RRb12*TTu22(il)+
     &				  Td12(il)*RRb21*TTu12(il)+Td12(il)*RRb22*TTu22(il)

        RRu21(ilp1)=Ru21(il)+RRb22*TTu21(il)*Td22(il)+RRb21*TTu11(il)*Td22(il)+
     &            RRb12*TTu21(il)*Td21(il)+RRb11*TTu11(il)*Td21(il)

        RRu22(ilp1)=Ru22(il)+RRb22*TTu22(il)*Td22(il)+RRb21*TTu12(il)*Td22(il)+
     &            RRb12*TTu22(il)*Td21(il)+RRb11*TTu12(il)*Td21(il)
	 
      End Do
	  
	  Nlm1=Nl-1

	  Vu11(Nl)=TTu11(Nl)

	  Vu21(Nl)=TTu21(Nl)

	  Vu12(Nl)=TTu12(Nl)
	  Vu22(Nl)=TTu22(Nl)
	  
      Vd11(Nl)=RRu11(Nl)*Vu11(Nl)*ea(Nl)+RRu12(Nl)*Vu21(Nl)*eb(Nl)
      Vd12(Nl)=RRu11(Nl)*Vu12(Nl)*ea(Nl)+RRu12(Nl)*Vu22(Nl)*eb(Nl)
      Vd21(Nl)=RRu21(Nl)*Vu11(Nl)*ea(Nl)+RRu22(Nl)*Vu21(Nl)*eb(Nl)
      Vd22(Nl)=RRu21(Nl)*Vu12(Nl)*ea(Nl)+RRu22(Nl)*Vu22(Nl)*eb(Nl)

	  Do il=Nl-1,0,-1
	    ilp1=il+1

		Vu11(il)=TTu11(il)*Vu11(ilp1)*ea(ilp1)+TTu12(il)*Vu21(ilp1)*eb(ilp1)
		Vu12(il)=TTu11(il)*Vu12(ilp1)*ea(ilp1)+TTu12(il)*Vu22(ilp1)*eb(ilp1)
		Vu21(il)=TTu21(il)*Vu11(ilp1)*ea(ilp1)+TTu22(il)*Vu21(ilp1)*eb(ilp1)
		Vu22(il)=TTu21(il)*Vu12(ilp1)*ea(ilp1)+TTu22(il)*Vu22(ilp1)*eb(ilp1)
		
		Vd11(il)=RRu11(il)*Vu11(il)*ea(il)+RRu12(il)*Vu21(il)*eb(il)
        Vd12(il)=RRu11(il)*Vu12(il)*ea(il)+RRu12(il)*Vu22(il)*eb(il)
        Vd21(il)=RRu21(il)*Vu11(il)*ea(il)+RRu22(il)*Vu21(il)*eb(il)
        Vd22(il)=RRu21(il)*Vu12(il)*ea(il)+RRu22(il)*Vu22(il)*eb(il)

      End Do

	  
      return
	  End

	   
      subroutine Emat(P,ETA,NETA,RHO,BETA,Md,Nd,Mu,Nu)
*********************************************************
*     form the layer eigen arrays Md,Nd
*********************************************************
      implicit none
      real    rho,twomup,beta,p
      complex eta,neta,Nd(2,2),Md(2,2),Nu(2,2),Mu(2,2),ci,epsa,epsb,c1
	  Ci=cmplx(0.0,1.0)
	  C1=cmplx(1.0,0.0)
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


	  Subroutine  ScatMat(p,eta1,neta1,rho1,beta1,eta2,neta2,rho2,beta2,
     &                     Rd11,Rd12,Rd21,Rd22,Tu11,Tu12,Tu21,Tu22,
     &                     Td11,Td12,Td21,Td22,Ru11,Ru12,Ru21,Ru22)
*     ********************************************************************
*     plane-wave interfacial scattering matrices  
*     modified from Aki&Richards pp149-51
*     ********************************************************************
	  REAL    RHO1,RHO2,rtrho1,rtrho2,p,m2
	  COMPLEX ETA1,NETA1,ETA2,NETA2,BETA1,BETA2,PSQ,CRHO1,CRHO2,DRHO,Dmu
	  COMPLEX rteta1,rteta2,rtneta1,rtneta2,Rd11,Rd12,Rd21,Rd22
      COMPLEX Tu11,Tu12,Tu21,Tu22,Td11,Td12,Td21,Td22,Ru11,Ru12,Ru21,Ru22
	  COMPLEX PSQD,a,b,c,d,Det,E,F,G,H,Q,R,S,T,U,V,rtza1,rtza2,rtzb1,rtzb2

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


	  COMPLEX FUNCTION VERTICALSLO(C,P)
	  REAL C,P,oneoverc,cmin
	  parameter(cmin=1.0E-6)
	  COMPLEX T
	  if(abs(c).le.cmin)then
	    PAUSE 'propagation velocity too small'
	  else
	    oneoverc=1.0/c
	  end if
	  T=CMPLX((oneoverC+P)*(oneoverC-P),0.0)
	  VERTICALSLO=csqrt(T)
	  IF(AIMAG(VERTICALSLO) .LT. 0.0)THEN 
	    VERTICALSLO=-VERTICALSLO
	  END IF
	  RETURN
	  END


