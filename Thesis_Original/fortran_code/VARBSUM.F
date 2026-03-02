      Program  VarBSum
	  Implicit none

	  Integer  il,jl,ilp1,Nl,Nlp1,Pdim,MaxBorn,imom,jmom
	  Integer  i,j,k,MomDim,iborn,ib,jb,jbp1,MaxMom
	  integer  nmax,bs,np,Maxbs,ilm1,ioff,joff,ibp1
	  Parameter(np=1,bs=2,Maxbs=2,Nl=2,Nlp1=Nl+1,Pdim=4*Nl) 
	  Parameter(Maxborn=5,MomDim=bs*MaxBorn,MaxMom=2*MaxBorn-1)
	  INTEGER  IPIV(MomDim),INDXR(MomDim),INDXC(MomDim)

      Real     rho0(0:Nlp1),alpha0(0:Nlp1),beta0(0:Nlp1)
      Real     rho(0:Nlp1),alpha(0:Nlp1),beta(0:Nlp1)
      Real     drho(Nl),dalpha(Nl),dbeta(Nl)
	  Real     w,p(np),h(0:Nl)
	  Complex  B11,B12,B21,B22,R11old,R21old,R12old,R22old
	  Complex  R11new,R12new,R21new,R22new,sumC,sumD
	  Complex  T11old,T21old,T12old,T22old,T11new,T21new,T12new,T22new
	  Complex  MomMat(MomDim,MomDim),MomInv(MomDim,MomDim)
	  Complex  MomVec(MomDim,bs),MomVect(bs,MomDim)
	  Complex  C(MomDim,bs),D(bs,MomDim)
	  Complex  Mom11(0:MaxMom),Mom12(0:MaxMom)
	  Complex  Mom21(0:MaxMom),Mom22(0:MaxMom)

	  Complex  S11(Nl),S21(Nl),S22(Nl)
	  Complex  S31(Nl),S41(Nl),S42(Nl)
	  Complex  Pmat(Pdim,Pdim),PmatRT(Pdim,Pdim),PmatJS(Pdim,Pdim),sum
	  Complex  PmatT(Pdim,Pdim)
      Complex  RRd11(0:Nlp1),RRd12(0:Nlp1),RRd21(0:Nlp1),RRd22(0:Nlp1)
	  Complex  RRu11(0:Nlp1),RRu12(0:Nlp1),RRu21(0:Nlp1),RRu22(0:Nlp1)
	  Complex  TTu11(0:Nlp1),TTu12(0:Nlp1),TTu21(0:Nlp1),TTu22(0:Nlp1)
	  Complex  TTd11(0:Nlp1),TTd12(0:Nlp1),TTd21(0:Nlp1),TTd22(0:Nlp1)
      Complex  Rd11(0:Nl),Rd12(0:Nl),Rd21(0:Nl),Rd22(0:Nl)
	  Complex  Ru11(0:Nl),Ru12(0:Nl),Ru21(0:Nl),Ru22(0:Nl)
	  Complex  Tu11(0:Nl),Tu12(0:Nl),Tu21(0:Nl),Tu22(0:Nl)
	  Complex  Td11(0:Nl),Td12(0:Nl),Td21(0:Nl),Td22(0:Nl)
	  Complex  Uu11(0:Nlp1),Uu12(0:Nlp1),Uu21(0:Nlp1),Uu22(0:Nlp1)
	  Complex  Ud11(0:Nlp1),Ud12(0:Nlp1),Ud21(0:Nlp1),Ud22(0:Nlp1)
	  Complex  Vu11(0:Nlp1),Vu12(0:Nlp1),Vu21(0:Nlp1),Vu22(0:Nlp1)
	  Complex  Vd11(0:Nlp1),Vd12(0:Nlp1),Vd21(0:Nlp1),Vd22(0:Nlp1)
	  Complex  Bd11(0:Nlp1),Bd21(0:Nlp1),Bu11(0:Nlp1),Bu21(0:Nlp1)
      Complex  Rf11,Rf12,Rf21,Rf22,InvUv11,InvUv12,InvUv21,InvUv22,InvDet
	  Complex  Nd(2,2),Md(2,2),Nu(2,2),Mu(2,2)
	  Complex  Ci,C0,C1,Wt11,Wt12,Wt21,Wt22,t11,t12,t21,t22
	  Complex  ea0(0:Nl),eb0(0:Nl),eta(0:Nlp1),neta(0:Nlp1)
	  Complex  ea(0:Nl),eb(0:Nl),eta0(0:Nlp1),neta0(0:Nlp1)
	  Complex  Temp,VERTICALSLO,Csqrt,cbeta1,cbeta2

	  Complex  SY11(Nl),SY12(Nl),SY21(Nl),SY22(Nl)
	  Complex  SY31(Nl),SY32(Nl),SY41(Nl),SY42(Nl)	  
	  Complex  XS11(Nl),XS12(Nl),XS21(Nl),XS22(Nl)
	  Complex  XS31(Nl),XS32(Nl),XS41(Nl),XS42(Nl)	  
	  Complex  Xd11(Nl,MaxBorn),Xd12(Nl,MaxBorn)
	  Complex  Xd21(Nl,MaxBorn),Xd22(Nl,MaxBorn)
	  Complex  Xu11(Nl,MaxBorn),Xu12(Nl,MaxBorn)
	  Complex  Xu21(Nl,MaxBorn),Xu22(Nl,MaxBorn)
	  Complex  Yd11(Nl,MaxBorn+1),Yd12(Nl,MaxBorn+1)
	  Complex  Yd21(Nl,MaxBorn+1),Yd22(Nl,MaxBorn+1)
	  Complex  Yu11(Nl,MaxBorn+1),Yu12(Nl,MaxBorn+1)
	  Complex  Yu21(Nl,MaxBorn+1),Yu22(Nl,MaxBorn+1)

	  Complex  Refresp11,Refresp12,Refresp21,Refresp22
	  Complex  Resresp11,Resresp12,Resresp21,Resresp22
	  Complex  Totresp11,Totresp12,Totresp21,Totresp22
	  Logical  Surf,reflex


	  DATA     alpha0/1.51,2.01,2.51,3.51/
	  DATA     beta0/1.01,1.51,2.51,3.01/	
      DATA     rho0/1.01,1.51,1.81,2.01/
      DATA     drho/2*(.1)/
	  DATA     dalpha/2*(.1)/
	  DATA     dbeta/2*(.1)/

	  DATA     p,W,h/0.3,1.0,0.5,.1,.1/

*	  DATA     alpha0/1.6,2.1,1.6/
*	  DATA     beta0/1.0,1.5,1.0/	
*      DATA     rho0/1.1,1.6,1.1/
*      DATA     drho/0.1/
*	  DATA     dalpha/0.1/
*	  DATA     dbeta/0.1/
*	  DATA     p,W,h/.3,1.0,0.0,1.0/
      DATA     Surf,reflex/.false.,.true./
	  If(Surf.and.reflex) STOP 'Freesurface/downward reflexion unimplemented'
	  CI=(0.0,1.0)
	  C0=(0.0,0.0)
	  C1=(1.0,0.0)
	  if(.not.Surf) H(0)=0.0
	  
	  rho(0)=rho0(0)
	  rho(NL+1)=rho0(NL+1)

	  alpha(0)=alpha0(0)
	  alpha(NL+1)=alpha0(NL+1)

	  beta(0)=beta0(0)
	  beta(NL+1)=beta0(NL+1)
	  Do il=1,Nl
		alpha(il)=alpha0(il)+dalpha(il)
		beta(il)=beta0(il)+dbeta(il)
		rho(il)=rho0(il)+drho(il)
	  Enddo
	  
	  do il=0,Nlp1
	     eta0(il)=VERTICALSLO(alpha0(il),P)
	     neta0(il)=VERTICALSLO(beta0(il),P)
		 
		 eta(il)=VERTICALSLO(alpha(il),P)
	     neta(il)=VERTICALSLO(beta(il),P)

      end do !il
      Do il=0,Nl
	    Temp=CI*w*h(il)*eta0(il)
        EA0(il)=CEXP(Temp)
	    Temp=CI*w*h(il)*neta0(il)
        EB0(il)=CEXP(Temp)		
	    Temp=CI*w*h(il)*eta(il)
        EA(il)=CEXP(Temp)
	    Temp=CI*w*h(il)*neta(il)
        EB(il)=CEXP(Temp)
*		Type*,'il,ea0(il)=',il,ea0(il)
*		Type*,'il,eb0(il)=',il,eb0(il)
      End Do

*     *******************************************************
*     set free surface reflection array Rf11,Rf12,Rf21,Rf22
*     and free-surface amplitude factor
*     *******************************************************
      If(surf) then
        call Emat(P,eta0(0),neta0(0),rho0(0),beta0(0),Md,Nd,Mu,Nu)
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
*		Type*,'Rf11=',Rf11
*		Type*,'Rf12=',Rf12
*		Type*,'Rf21=',Rf21
*		Type*,'Rf22=',Rf22
*     set (Wf)^(#)= transpose of the free-space amplification = Ci*Nd^(-1)
		Wt11=Ci*InvDet*Nd(2,2)
		Wt22=Ci*InvDet*Nd(1,1)
		Wt12=-Ci*InvDet*Nd(1,2)
		Wt21=-Ci*InvDet*Nd(2,1)
*		Type*,'Wt11,Wt22,Wt12,Wt12=',Wt11,Wt22,Wt12,Wt12
      else !no-freesurface
        Rf11=(0.0,0.0)
		Rf12=(0.0,0.0)
		Rf21=(0.0,0.0)
		Rf22=(0.0,0.0)
		Wt11=(1.0,0.0)
		Wt22=(1.0,0.0)
		Wt12=(0.0,0.0)
		Wt21=(0.0,0.0)
      Endif !free-surface test	  

	  Do il=1,Nl
	    ilp1=il+1
		ilm1=il-1
		cbeta1=cmplx(beta0(il),0.0)
		cbeta2=cmplx(beta(il),0.0)

	    call ScatMat(p,eta0(il),neta0(il),rho0(il),cbeta1,eta(il),
     &  neta(il),rho(il),cbeta2,Rd11(ilm1),Rd12(ilm1),Rd21(ilm1),Rd22(ilm1),
     &  Tu11(ilm1),Tu12(ilm1),Tu21(ilm1),Tu22(ilm1),Td11(ilm1),Td12(ilm1),
     &  Td21(ilm1),Td22(ilm1),Ru11(ilm1),Ru12(ilm1),Ru21(ilm1),Ru22(ilm1))

		cbeta1=cmplx(beta(il),0.0)
		cbeta2=cmplx(beta0(il),0.0)

	    call ScatMat(p,eta(il),neta(il),rho(il),cbeta1,eta0(il),
     &  neta0(il),rho0(il),cbeta2,Rd11(il),Rd12(il),Rd21(il),
     &  Rd22(il),Tu11(il),Tu12(il),Tu21(il),Tu22(il),Td11(il),Td12(il),
     &  Td21(il),Td22(il),Ru11(il),Ru12(il),Ru21(il),Ru22(il))

*       *************************************************************
*		set the Layer S matrices
*       *************************************************************
	  	call SetS(il,Nl,ea(il),eb(il),Rd11,Rd12,Rd21,Rd22,Td11,Td12,Td21,Td22,
     &  Tu11,Tu12,Tu21,Tu22,Ru11,Ru12,Ru21,Ru22,S11(il),S21(il),
     &   S22(il),S31(il),S41(il),S42(il))
	    S31(il)=S31(il)-ea0(il)  
	    S42(il)=S42(il)-eb0(il)
*	    TYpe*,'il,S11=',il,S11(il)
*	    TYpe*,'il,S21=',il,S21(il)
*	    TYpe*,'il,S22=',il,S22(il)
*	    TYpe*,'il,S31=',il,S31(il)
*	    TYpe*,'il,S41=',il,S41(il)
*	    TYpe*,'il,S42=',il,S42(il)
	  enddo
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
*	    Type*,'il=',il
*		Type*,'Rd11(il)=',Rd11(il)
*		Type*,'Rd12(il)=',Rd12(il)
*		Type*,'Rd21(il)=',Rd21(il)
*		Type*,'Rd22(il)=',Rd22(il)
*		Type*,'Td11(il)=',Td11(il)
*		Type*,'Td12(il)=',Td12(il)
*		Type*,'Td21(il)=',Td21(il)
*		Type*,'Td22(il)=',Td22(il)
*		Type*,'Ru11(il)=',Ru11(il)
*		Type*,'Ru12(il)=',Ru12(il)
*		Type*,'Ru21(il)=',Ru21(il)
*		Type*,'Ru22(il)=',Ru22(il)
*		Type*,'Tu11(il)=',Tu11(il)
*		Type*,'Tu12(il)=',Tu12(il)
*		Type*,'Tu21(il)=',Tu21(il)
*		Type*,'Tu22(il)=',Tu22(il)
	  end do
*     Set the Jost solutions Ud,Uu,Vd,Vu in each of the Nl layers
      call DSFS(Nl,ea0,eb0,Rd11,Rd12,Rd21,Rd22,Td11,Td12,Td21,Td22,
     &     Tu11,Tu12,Tu21,Tu22,Ru11,Ru12,Ru21,Ru22,TTu11,TTu12,TTu21,
     &     TTu22,TTd11,TTd12,TTd21,TTd22,RRd11,RRd12,RRd21,RRd22,RRu11,
     &     RRu12,RRu21,RRu22,Uu11,Uu12,Uu21,Uu22,Ud11,Ud12,Ud21,Ud22,
     &     Vu11,Vu12,Vu21,Vu22,Vd11,Vd12,Vd21,Vd22,RF11,RF12,RF21,RF22,
     &     Wt11,Wt12,Wt21,Wt22)
	 
       If(reflex) then
         R11old=RRd11(0)
		 R12old=RRd12(0)
		 R21old=RRd21(0)
		 R22old=RRd22(0)
		 Type*,'R11old=',R11old	 
		 Type*,'R12old=',R12old	 
		 Type*,'R21old=',R21old	 
		 Type*,'R22old=',R22old
	   else
         T11old=-Ud11(Nlp1)
		 T12old=Ud21(Nlp1)
		 T21old=Ud12(Nlp1)
		 T22old=-Ud22(Nlp1)
		 Type*,'T11old=',T11old	 
		 Type*,'T12old=',T12old	 
		 Type*,'T21old=',T21old	 
		 Type*,'T22old=',T22old
	   endif

*     SET inverse propagation invariant
      t11=-Ud11(Nlp1)
      t22=-Ud22(Nlp1)
      t21=Ud12(Nlp1)
      t12=Ud21(Nlp1)
	  InvDet=C1/(t11*t22-t12*t21)	  
	  InvUv11=InvDet*t22
	  InvUv12=-InvDet*t12
	  InvUv21=-InvDet*t21
	  InvUv22=InvDet*t11
*      type*,'InvUv11=',InvUv11
*      type*,'InvUv12=',InvUv12
*      type*,'InvUv21=',InvUv21
*      type*,'InvUv22=',InvUv22


*     set the scattered field Scat
******************************************
      If(reflex.and.(.not.surf)) then
	    Do il=1,Nl
	      Xd11(il,1)=Ud11(il)
		  Xd12(il,1)=-Ud12(il)
		  Xd21(il,1)=-Ud21(il)
		  Xd22(il,1)=Ud22(il)
		  Xu11(il,1)=Uu11(il)
		  Xu12(il,1)=-Uu12(il)
		  Xu21(il,1)=-Uu21(il)
		  Xu22(il,1)=Uu22(il)
	    Enddo
	  else
	    Do il=1,Nl
	      Xd11(il,1)=-Ud11(il)
		  Xd12(il,1)=Ud12(il)
		  Xd21(il,1)=Ud21(il)
		  Xd22(il,1)=-Ud22(il)
		  Xu11(il,1)=-Uu11(il)
		  Xu12(il,1)=Uu12(il)
		  Xu21(il,1)=Uu21(il)
		  Xu22(il,1)=-Uu22(il)
	    Enddo
	  endif
	   
*      set the incident field 
**********************************************
       if(reflex) then
*        set incident field for reflection geometry
         Do il=1,Nl
		   Yd11(il,1)=Ud11(il)
		   Yd12(il,1)=Ud12(il)
		   Yd21(il,1)=Ud21(il)
		   Yd22(il,1)=Ud22(il)
		   Yu11(il,1)=Uu11(il)
		   Yu12(il,1)=Uu12(il)
		   Yu21(il,1)=Uu21(il)
		   Yu22(il,1)=Uu22(il)
*		   Type*,'il,Ud11(il)=',il,Ud11(il)
*		   Type*,'il,Ud12(il)=',il,Ud12(il)
*		   Type*,'il,Ud21(il)=',il,Ud21(il)
*		   Type*,'il,Ud22(il)=',il,Ud22(il)
*		   Type*,'il,Uu11(il)=',il,Uu11(il)
*		   Type*,'il,Uu12(il)=',il,Uu12(il)
*		   Type*,'il,Uu21(il)=',il,Uu21(il)
*		   Type*,'il,Uu22(il)=',il,Uu22(il)
         enddo
	   else !(upward transmission geometry)
         Do il=1,Nl
		   Yd11(il,1)=Vd11(il)
		   Yd12(il,1)=Vd12(il)
		   Yd21(il,1)=Vd21(il)
		   Yd22(il,1)=Vd22(il)
		   Yu11(il,1)=Vu11(il)
		   Yu12(il,1)=Vu12(il)
		   Yu21(il,1)=Vu21(il)
		   Yu22(il,1)=Vu22(il)
		   
*		   Type*,'il,Vd11(il)=',il,Vd11(il)
*		   Type*,'il,Vd12(il)=',il,Vd12(il)
*		   Type*,'il,Vd21(il)=',il,Vd21(il)
*		   Type*,'il,Vd22(il)=',il,Vd22(il)
*		   Type*,'il,Vu11(il)=',il,Vu11(il)
*		   Type*,'il,Vu12(il)=',il,Vu12(il)
*		   Type*,'il,Vu21(il)=',il,Vu21(il)
*		   Type*,'il,Vu22(il)=',il,Vu22(il)
         enddo
	  endif !incident test

*      Load incident field
       Do il=1,Nl
	     Bd11(il)=Vd11(il)
	     Bd21(il)=Vd21(il)
	     Bu11(il)=Vu11(il)
	     Bu21(il)=Vu21(il)
	   Enddo

**********************************************
*     set reference medium propagator	  
	  call setPmatJS(Nl,Pdim,Ud11,Ud12,Ud21,Ud22,Uu11,Uu12,Uu21,Uu22,
     &     Vd11,Vd12,Vd21,Vd22,Vu11,Vu12,Vu21,Vu22,InvUv11,
     &     InvUv12,InvUv21,InvUv22,PmatJS)


*     Set the trasnsposed Propagator	  
	  Do il=1,Pdim
	    Do jl=1,Pdim
		  PmatT(il,jl)=PmatJS(jl,il)
		Enddo
	  Enddo
	  
*************************
      call setSY(Nl,S11,S21,S22,S31,S41,S42,Yd11(1,1),Yd12(1,1),
     &     Yd21(1,1),Yd22(1,1),Yu11(1,1),Yu12(1,1),Yu21(1,1),Yu22(1,1),
     &     SY11,SY12,SY21,SY22,SY31,SY32,SY41,SY42)
*      Type*,'Check this'
*	  Type*,'SY11(1)=',SY11(1)
*	  Type*,'SY21(1)=',SY21(1)
*	  Type*,'SY31(1)=',SY31(1)
*	  Type*,'SY41(1)=',SY41(1)
*	  Type*,'SY11(2)=',SY11(2)
*	  Type*,'SY21(2)=',SY21(2)
*	  Type*,'SY31(2)=',SY31(2)
*	  Type*,'SY41(2)=',SY41(2)
      call setXS(Nl,S11,S21,S22,S31,S41,S42,Xd11(1,1),Xd12(1,1),
     &     Xd21(1,1),Xd22(1,1),Xu11(1,1),Xu12(1,1),Xu21(1,1),
     &     Xu22(1,1),XS11,XS12,XS21,XS22,XS31,XS32,XS41,XS42)

*     set Mom(0)
      Mom11(0)=C0
      Mom21(0)=C0
      Mom12(0)=C0
      Mom22(0)=C0
      Do il=1,Nl
        Mom11(0)=Mom11(0)+Xd21(il,1)*SY21(il)+Xd11(il,1)*SY11(il)+
     &       Xu21(il,1)*SY41(il)+Xu11(il,1)*SY31(il)
	    Mom12(0)=Mom12(0)+Xd21(il,1)*SY22(il)+Xd11(il,1)*SY12(il)+
     &       Xu21(il,1)*SY42(il)+Xu11(il,1)*SY32(il)
	    Mom21(0)=Mom21(0)+Xd22(il,1)*SY21(il)+Xd12(il,1)*SY11(il)+
     &       Xu22(il,1)*SY41(il)+Xu12(il,1)*SY31(il)
	    Mom22(0)=Mom22(0)+Xd22(il,1)*SY22(il)+Xd12(il,1)*SY12(il)+
     &       Xu22(il,1)*SY42(il)+Xu12(il,1)*SY32(il)	  
	  enddo
*	  Type*,'Mom11(0)=',Mom11(0)
*	  Type*,'Mom12(0)=',Mom12(0)
*	  Type*,'Mom21(0)=',Mom21(0)
*	  Type*,'Mom22(0)=',Mom22(0)
****************************
      Do ib=2,Maxborn

*     set right Krylov basis Y=P*SY 
      call setPSY(Nl,Pdim,PmatJS,SY11,SY12,SY21,SY22,SY31,SY32,SY41,SY42,
     &     Yd11(1,ib),Yd12(1,ib),Yd21(1,ib),Yd22(1,ib),Yu11(1,ib),
     &     Yu12(1,ib),Yu21(1,ib),Yu22(1,ib))
*      Type*,'Check this,ib=',ib
*	  TYpe*,'Yd11(1,ib)=',Yd11(1,ib)
*	  TYpe*,'Yd21(1,ib)=',Yd21(1,ib)
*	  TYpe*,'Yu11(1,ib)=',Yu11(1,ib)
*	  TYpe*,'Yu21(1,ib)=',Yu21(1,ib)
*	  TYpe*,'Yd11(2,ib)=',Yd11(2,ib)
*	  TYpe*,'Yd21(2,ib)=',Yd21(2,ib)
*	  TYpe*,'Yu11(2,ib)=',Yu11(2,ib)
*	  TYpe*,'Yu21(2,ib)=',Yu21(2,ib)
*	  stop
*      update Bornsum
       Do il=1,Nl
	     Bd11(il)=Bd11(il)+Yd11(il,ib)
	     Bd21(il)=Bd21(il)+Yd21(il,ib)
	     Bu11(il)=Bu11(il)+Yu11(il,ib)
	     Bu21(il)=Bu21(il)+Yu21(il,ib)
	   Enddo

*     Set left Krylov basis X=P^t*XS
      call setPSY(Nl,Pdim,PmatT,XS11,XS12,XS21,XS22,XS31,XS32,XS41,XS42,
     &     Xd11(1,ib),Xd12(1,ib),Xd21(1,ib),Xd22(1,ib),Xu11(1,ib),
     &     Xu12(1,ib),Xu21(1,ib),Xu22(1,ib))

*     set SY=S*Y
      call setSY(Nl,S11,S21,S22,S31,S41,S42,Yd11(1,ib),Yd12(1,ib),
     &     Yd21(1,ib),Yd22(1,ib),Yu11(1,ib),Yu12(1,ib),Yu21(1,ib),
     &     Yu22(1,ib),SY11,SY12,SY21,SY22,SY31,SY32,SY41,SY42)
	 

*     set XS=X^t*S
      call setXS(Nl,S11,S21,S22,S31,S41,S42,Xd11(1,ib),Xd12(1,ib),
     &     Xd21(1,ib),Xd22(1,ib),Xu11(1,ib),Xu12(1,ib),Xu21(1,ib),
     &     Xu22(1,ib),XS11,XS12,XS21,XS22,XS31,XS32,XS41,XS42)

*     set Mom(ib-1)
      Mom11(ib-1)=C0
      Mom21(ib-1)=C0
      Mom12(ib-1)=C0
      Mom22(ib-1)=C0
      Do il=1,Nl
        Mom11(ib-1)=Mom11(ib-1)+Xd21(il,1)*SY21(il)+Xd11(il,1)*SY11(il)+
     &       Xu21(il,1)*SY41(il)+Xu11(il,1)*SY31(il)
	    Mom12(ib-1)=Mom12(ib-1)+Xd21(il,1)*SY22(il)+Xd11(il,1)*SY12(il)+
     &       Xu21(il,1)*SY42(il)+Xu11(il,1)*SY32(il)
	    Mom21(ib-1)=Mom21(ib-1)+Xd22(il,1)*SY21(il)+Xd12(il,1)*SY11(il)+
     &       Xu22(il,1)*SY41(il)+Xu12(il,1)*SY31(il)
	    Mom22(ib-1)=Mom22(ib-1)+Xd22(il,1)*SY22(il)+Xd12(il,1)*SY12(il)+
     &       Xu22(il,1)*SY42(il)+Xu12(il,1)*SY32(il)	  
	  enddo

	  enddo !ib loop
	  

*     set the MaxBorn+1 right Krylov basis vector Y=P*SY 
      call setPSY(Nl,Pdim,PmatJS,SY11,SY12,SY21,SY22,SY31,SY32,SY41,SY42,
     &     Yd11(1,MaxBorn+1),Yd12(1,MaxBorn+1),Yd21(1,MaxBorn+1),
     &     Yd22(1,MaxBorn+1),Yu11(1,MaxBorn+1),Yu12(1,MaxBorn+1),
     &     Yu21(1,MaxBorn+1),Yu22(1,MaxBorn+1))
*      update Bornsum
       ib=MaxBorn+1
       Do il=1,Nl
	     Bd11(il)=Bd11(il)+Yd11(il,ib)
	     Bd21(il)=Bd21(il)+Yd21(il,ib)
	     Bu11(il)=Bu11(il)+Yu11(il,ib)
	     Bu21(il)=Bu21(il)+Yu21(il,ib)
	   Enddo

*     set SY=S*Y
      call setSY(Nl,S11,S21,S22,S31,S41,S42,Yd11(1,MaxBorn+1),
     &     Yd12(1,MaxBorn+1),Yd21(1,MaxBorn+1),Yd22(1,MaxBorn+1),
     &     Yu11(1,MaxBorn+1),Yu12(1,MaxBorn+1),Yu21(1,MaxBorn+1),
     &     Yu22(1,MaxBorn+1),SY11,SY12,SY21,SY22,SY31,SY32,SY41,SY42)

*     Set the remaining moments
      ib=1
      Do imom=MaxBorn,MaxMom
*       set Mom(imom)
        Mom11(imom)=C0
        Mom21(imom)=C0
        Mom12(imom)=C0
        Mom22(imom)=C0
        Do il=1,Nl
          Mom11(imom)=Mom11(imom)+Xd21(il,ib)*SY21(il)+Xd11(il,ib)*SY11(il)+
     &       Xu21(il,ib)*SY41(il)+Xu11(il,ib)*SY31(il)
	      Mom12(imom)=Mom12(imom)+Xd21(il,ib)*SY22(il)+Xd11(il,ib)*SY12(il)+
     &       Xu21(il,ib)*SY42(il)+Xu11(il,ib)*SY32(il)
	      Mom21(imom)=Mom21(imom)+Xd22(il,ib)*SY21(il)+Xd12(il,ib)*SY11(il)+
     &       Xu22(il,ib)*SY41(il)+Xu12(il,ib)*SY31(il)
	      Mom22(imom)=Mom22(imom)+Xd22(il,ib)*SY22(il)+Xd12(il,ib)*SY12(il)+
     &       Xu22(il,ib)*SY42(il)+Xu12(il,ib)*SY32(il)
	    enddo

	    Type*,'imom,Mom11(imom)=',imom,Mom11(imom)
		Type*,'imom,Mom12(imom)=',imom,Mom12(imom)
		Type*,'imom,Mom21(imom)=',imom,Mom21(imom)
		Type*,'imom,Mom22(imom)=',imom,Mom22(imom)

	    ib=ib+1
      Enddo !imom-loop


      Do il=1,MomDim
	    Do jl=1,MomDim
		  MomMat(il,jl)=(0.0,0.0)
		Enddo
	  Enddo
	  
*      set up the moment array
	   Do imom=0,MaxBorn-1
	     Do jmom=0,MaxBorn-1
		 
		   MomMat(imom*bs+1,jmom*bs+1)=Mom11(imom+jmom)-Mom11(imom+jmom+1)
		   if (bs .eq.2) then
		     MomMat(imom*bs+2,jmom*bs+1)=Mom21(imom+jmom)-Mom21(imom+jmom+1)
			 MomMat(imom*bs+1,jmom*bs+2)=Mom12(imom+jmom)-Mom12(imom+jmom+1)
			 MomMat(imom*bs+2,jmom*bs+2)=Mom22(imom+jmom)-Mom22(imom+jmom+1)
		   end if
         End do !jmom loop
		 
		 MomVec(imom*bs+1,1)=Mom11(imom)
		 MomVect(1,1+imom*bs)=Mom11(imom)
		 if (bs .eq.2) then
		   MomVec(imom*bs+1,2)=Mom12(imom)
		   MomVec(imom*bs+2,1)=Mom21(imom)
		   MomVec(imom*bs+2,2)=Mom22(imom)
		   MomVect(1,2+imom*bs)=Mom12(imom)
		   MomVect(2,1+imom*bs)=Mom21(imom)
		   MomVect(2,2+imom*bs)=Mom22(imom)
         end if
	   End do !imom loop


	   Do i=1,MomDim
	     do j= 1,MomDim
		   MomInv(i,j)=(0.0,0.0)
*		   type*,'i,j,MomMat(i,j)=',i,j
*		   Type*,MomMat(i,j)
*		   Type*
		 End do !j loop
		 MomInv(i,i)=(1.0,0.0)
	   End do !i loop


	   call CMPLXGJ(MomMat,MomDim,MomDim,MomInv,MomDim,MomDim,
     &             IPIV,INDXR,INDXC,MomDim)

	   Do i=1,MomDim
	     do j= 1,bs
		   sumC=(0.0,0.0)
		   sumD=(0.0,0.0)
		   Do k=1,MomDim
		     sumC=sumC+MomInv(i,k)*MomVec(k,j)
		     sumD=sumD+MomVect(j,k)*MomInv(k,i)
		   End do ! k-loop
		   C(i,j)=sumC
		   D(j,i)=sumD
		 End do !j loop
	  End do !i loop
********************************************
*     form the iterated Matrix functional
	  Resresp11=(0.0,0.0)
	  Resresp12=(0.0,0.0)
	  Resresp21=(0.0,0.0)
	  Resresp22=(0.0,0.0)

	  Do imom=1,MomDim
		   Resresp11=Resresp11+MomVect(1,imom)*C(imom,1)
		   Type*,'imom,Resresp11=',imom,Resresp11
		   if (bs.eq.2) then
		   Resresp12=Resresp12+MomVect(1,imom)*C(imom,2)
		   Resresp21=Resresp21+MomVect(2,imom)*C(imom,1)
		   Resresp22=Resresp22+MomVect(2,imom)*C(imom,2)
		   end if
		   
*		   Resresp11=Resresp11+D(1,imom)*MomVec(imom,1)
*		   Type*,'imom,Resresp11=',imom,Resresp11
*		   if (bs.eq.2) then
*		   Resresp12=Resresp12+D(1,imom)*MomVec(imom,2)
*		   Resresp21=Resresp21+D(2,imom)*MomVec(imom,1)
*		   Resresp22=Resresp22+D(2,imom)*MomVec(imom,2)
*		   end if

      End do ! imom loop
********************************************
	  do il=0,Nl

	    ilp1=il+1
		cbeta1=cmplx(beta(il),0.0)
		cbeta2=cmplx(beta(ilp1),0.0)

	    call ScatMat(p,eta(il),neta(il),rho(il),cbeta1,eta(ilp1),
     &  neta(ilp1),rho(ilp1),cbeta2,Rd11(il),Rd12(il),Rd21(il),
     &  Rd22(il),Tu11(il),Tu12(il),Tu21(il),Tu22(il),Td11(il),Td12(il),
     &  Td21(il),Td22(il),Ru11(il),Ru12(il),Ru21(il),Ru22(il))

	  end do

      call DSFS(Nl,ea,eb,Rd11,Rd12,Rd21,Rd22,Td11,Td12,Td21,Td22,
     &           Tu11,Tu12,Tu21,Tu22,Ru11,Ru12,Ru21,Ru22,TTu11,TTu12,TTu21,
     &           TTu22,TTd11,TTd12,TTd21,TTd22,RRd11,RRd12,RRd21,RRd22,RRu11,
     &           RRu12,RRu21,RRu22,Uu11,Uu12,Uu21,Uu22,Ud11,Ud12,Ud21,Ud22,
     &           Vu11,Vu12,Vu21,Vu22,Vd11,Vd12,Vd21,Vd22,RF11,RF12,RF21,RF22,
     &           Wt11,Wt12,Wt21,Wt22)
*      Load incident field
*      Test accuracy of BornSum
       Do il=1,Nl
	     Type*,'Cabs(Bd11(il)-Vd11(il))/Cabs(Vd11(il))=',il,Cabs(Bd11(il)-Vd11(il))/Cabs(Vd11(il))
	     Type*,'Cabs(Bd21(il)-Vd21(il))/Cabs(Vd21(il))=',il,Cabs(Bd21(il)-Vd21(il))/Cabs(Vd21(il))
	     Type*,'Cabs(Bu11(il)-Vu11(il))/Cabs(Vu11(il))=',il,Cabs(Bu11(il)-Vu11(il))/Cabs(Vu11(il))
	     Type*,'Cabs(Bu21(il)-Vu21(il))/Cabs(Vu21(il))=',il,Cabs(Bu21(il)-Vu21(il))/Cabs(Vu21(il))
	   Enddo

       If(reflex) then
         R11new=RRd11(0)
		 R12new=RRd12(0)
		 R21new=RRd21(0)
		 R22new=RRd22(0)
		 Type*,'R11new=',R11new	 
		 Type*,'R21new=',R21new	 
		 Type*,'R22new=',R22new
       else
         T11new=-Ud11(Nlp1)
		 T12new=Ud21(Nlp1)
		 T21new=Ud12(Nlp1)
		 T22new=-Ud22(Nlp1)
		 Type*,'T11new=',T11new	 
		 Type*,'T12new=',T12new	 
		 Type*,'T21new=',T21new	 
		 Type*,'T22new=',T22new
		endif

*      calculate the high order Born approximation
	   B11=C0
	   B21=C0
	   B12=C0
	   B22=C0
       Do imom=0,Maxmom
	     Type*,'imom,Mom11(imom)=',imom,Mom11(imom)
		 Type*,'imom,Mom12(imom)=',imom,Mom12(imom)
		 Type*,'imom,Mom21(imom)=',imom,Mom21(imom)
		 Type*,'imom,Mom22(imom)=',imom,Mom22(imom)
         B11=B11+Mom11(imom)
         B21=B21+Mom21(imom)
         B12=B12+Mom12(imom)
         B22=B22+Mom22(imom)
       enddo

***************************************
*     Test accuracy of the result
      If(reflex) then
	    Type*,'B11/(R11new-R11old)=',B11/(R11new-R11old)
		Type*,'B12/(R12new-R12old)=',B12/(R12new-R12old)
		Type*,'B21/(R21new-R21old)=',B21/(R21new-R21old)
		Type*,'B22/(R22new-R22old)=',B22/(R22new-R22old)

	    Type*,'ResResp11/(R11new-R11old)=',ResResp11/(R11new-R11old)
		Type*,'ResResp12/(R12new-R12old)=',ResResp12/(R12new-R12old)
		Type*,'ResResp21/(R21new-R21old)=',ResResp21/(R21new-R21old)
		Type*,'ResResp22/(R22new-R22old)=',ResResp22/(R22new-R22old)
      else
	    Type*,'B11/(T11new-T11old)=',B11/(T11new-T11old)
		Type*,'B12/(T12new-T12old)=',B12/(T12new-T12old)
		Type*,'B21/(T21new-T21old)=',B21/(T21new-T21old)
		Type*,'B22/(T22new-T22old)=',B22/(T22new-T22old)

	    Type*,'ResResp11/(T11new-T11old)=',ResResp11/(T11new-T11old)
		Type*,'ResResp12/(T12new-T12old)=',ResResp12/(T12new-T12old)
		Type*,'ResResp21/(T21new-T21old)=',ResResp21/(T21new-T21old)
		Type*,'ResResp22/(T22new-T22old)=',ResResp22/(T22new-T22old)
      Endif
	  STOP 'normal completion'
	  END

      Subroutine setPSY(Nl,Pdim,Pmat,SY11,SY12,SY21,SY22,SY31,SY32,
     &     SY41,SY42,Yd11,Yd12,Yd21,Yd22,Yu11,Yu12,Yu21,Yu22)
*     set Y=P*SY
      Implicit none
	  Integer  Nl,Pdim,il,jl,ioff,joff
	  Complex  Pmat(Pdim,Pdim)
	  Complex  SY11(Nl),SY12(Nl),SY21(Nl),SY22(Nl)
	  Complex  SY31(Nl),SY32(Nl),SY41(Nl),SY42(Nl)	  
	  Complex  Yd11(Nl),Yd12(Nl),Yd21(Nl),Yd22(Nl)
	  Complex  Yu11(Nl),Yu12(Nl),Yu21(Nl),Yu22(Nl)
	  Type*,'entered setPSY'
      Do il=1,Nl
	    Yd11(il)=(0.0,0.0)
	    Yd12(il)=(0.0,0.0)
	    Yd21(il)=(0.0,0.0)
	    Yd22(il)=(0.0,0.0)
	    Yu11(il)=(0.0,0.0)
	    Yu12(il)=(0.0,0.0)
	    Yu21(il)=(0.0,0.0)
	    Yu22(il)=(0.0,0.0)
		
		Type*,'il,SY11(il)=',il,SY11(il)
		Type*,'il,SY21(il)=',il,SY21(il)
		Type*,'il,SY31(il)=',il,SY31(il)
		Type*,'il,SY41(il)=',il,SY41(il)
		ioff=4*(il-1)
		Do jl=1,Nl
		  joff=4*(jl-1)
          Yd11(il)=Yd11(il)+Pmat(ioff+1,joff+4)*SY41(jl)+Pmat(ioff+1,joff+3)*
     &    SY31(jl)+Pmat(ioff+1,joff+2)*SY21(jl)+Pmat(ioff+1,joff+1)*SY11(jl)
		  Yd12(il)=Yd12(il)+Pmat(ioff+1,joff+4)*SY42(jl)+Pmat(ioff+1,joff+3)*
     &    SY32(jl)+Pmat(ioff+1,joff+2)*SY22(jl)+Pmat(ioff+1,joff+1)*SY12(jl)
		  Yd21(il)=Yd21(il)+Pmat(ioff+2,joff+4)*SY41(jl)+Pmat(ioff+2,joff+3)*
     &    SY31(jl)+Pmat(ioff+2,joff+2)*SY21(jl)+Pmat(ioff+2,joff+1)*SY11(jl)
		  Yd22(il)=Yd22(il)+Pmat(ioff+2,joff+4)*SY42(jl)+Pmat(ioff+2,joff+3)*
     &    SY32(jl)+Pmat(ioff+2,joff+2)*SY22(jl)+Pmat(ioff+2,joff+1)*SY12(jl)
		  Yu11(il)=Yu11(il)+Pmat(ioff+3,joff+4)*SY41(jl)+Pmat(ioff+3,joff+3)*
     &    SY31(jl)+Pmat(ioff+3,joff+2)*SY21(jl)+Pmat(ioff+3,joff+1)*SY11(jl)
		  Yu12(il)=Yu12(il)+Pmat(ioff+3,joff+4)*SY42(jl)+Pmat(ioff+3,joff+3)*
     &    SY32(jl)+Pmat(ioff+3,joff+2)*SY22(jl)+Pmat(ioff+3,joff+1)*SY12(jl)
		  Yu21(il)=Yu21(il)+Pmat(ioff+4,joff+4)*SY41(jl)+Pmat(ioff+4,joff+3)*
     &    SY31(jl)+Pmat(ioff+4,joff+2)*SY21(jl)+Pmat(ioff+4,joff+1)*SY11(jl)
		  Yu22(il)=Yu22(il)+Pmat(ioff+4,joff+4)*SY42(jl)+Pmat(ioff+4,joff+3)*
     &    SY32(jl)+Pmat(ioff+4,joff+2)*SY22(jl)+Pmat(ioff+4,joff+1)*SY12(jl)

		Enddo !jl-loop
		Type*,'il,Yd11(il)=',il,Yd11(il)
		Type*,'il,Yd21(il)=',il,Yd21(il)
		Type*,'il,Yu11(il)=',il,Yu11(il)
		Type*,'il,Yu21(il)=',il,Yu21(il)
      Enddo !il-loop
	  pause 'stop here to check'
      return
	  end

	  subroutine setPmatJS(Nl,Pdim,Ud11,Ud12,Ud21,Ud22,Uu11,Uu12,Uu21,Uu22,Vd11,
     & Vd12,Vd21,Vd22,Vu11,Vu12,Vu21,Vu22,InvUv11,InvUv12,InvUv21,InvUv22,Pmat)
	  Implicit none
	  Integer  Nl,Pdim,il,jl,ioff,joff
	  Complex  Uu11(0:Nl),Uu12(0:Nl),Uu21(0:Nl),Uu22(0:Nl)
	  Complex  Ud11(0:Nl),Ud12(0:Nl),Ud21(0:Nl),Ud22(0:Nl)
	  Complex  Vu11(0:Nl),Vu12(0:Nl),Vu21(0:Nl),Vu22(0:Nl)
	  Complex  Vd11(0:Nl),Vd12(0:Nl),Vd21(0:Nl),Vd22(0:Nl)
      Complex  Pmat(Pdim,Pdim),InvUv11,InvUv12,InvUv21,InvUv22
	  Complex  C1
	  C1=(1.0,0.0)
	  
*************************
*     set diagonal Blocks P_ii
      Do il=1,Nl
	    ioff=4*(il-1)
        Pmat(ioff+1,ioff+1)=(InvUv22*Vd12(il)+InvUv12*Vd11(il))*Ud12(il)+
     &    (-InvUv21*Vd12(il)-InvUv11*Vd11(il))*Ud11(il)
        Pmat(ioff+2,ioff+1)=(InvUv22*Vd12(il)+InvUv12*Vd11(il))*Ud22(il)+
     &    (-InvUv21*Vd12(il)-InvUv11*Vd11(il))*Ud21(il)
        Pmat(ioff+2,ioff+2)=(-InvUv22*Vd22(il)-InvUv12*Vd21(il))*Ud22(il)+
     &    (InvUv21*Vd22(il)+InvUv11*Vd21(il))*Ud21(il)
        Pmat(ioff+3,ioff+1)=(InvUv22*Vd12(il)+InvUv12*Vd11(il))*Uu12(il)+
     &    (-InvUv21*Vd12(il)-InvUv11*Vd11(il))*Uu11(il)
        Pmat(ioff+3,ioff+2)=(-InvUv22*Vd22(il)-InvUv12*Vd21(il))*Uu12(il)+
     &    (InvUv21*Vd22(il)+InvUv11*Vd21(il))*Uu11(il)
        Pmat(ioff+3,ioff+3)=(InvUv22*Vu12(il)+InvUv12*Vu11(il))*Uu12(il)+
     &    (-InvUv21*Vu12(il)-InvUv11*Vu11(il))*Uu11(il)
        Pmat(ioff+4,ioff+1)=(InvUv22*Vd12(il)+InvUv12*Vd11(il))*Uu22(il)+
     &    (-InvUv21*Vd12(il)-InvUv11*Vd11(il))*Uu21(il)
        Pmat(ioff+4,ioff+2)=(-InvUv22*Vd22(il)-InvUv12*Vd21(il))*Uu22(il)+
     &    (InvUv21*Vd22(il)+InvUv11*Vd21(il))*Uu21(il)
        Pmat(ioff+4,ioff+3)=(InvUv22*Vu12(il)+InvUv12*Vu11(il))*Uu22(il)+
     &    (-InvUv21*Vu12(il)-InvUv11*Vu11(il))*Uu21(il)
        Pmat(ioff+4,ioff+4)=(-InvUv22*Vu22(il)-InvUv12*Vu21(il))*Uu22(il)+
     &    (InvUv21*Vu22(il)+InvUv11*Vu21(il))*Uu21(il)
	 
	  Enddo !il-loop
	 

***************************************
      Do jl=1,Nl
	    joff=4*(jl-1)
	    Do il=jl+1,Nl
	      ioff=4*(il-1)
          Pmat(ioff+1,joff+1)=(InvUv22*Vd12(jl)+InvUv12*Vd11(jl))*Ud12(il)+
     &     (-InvUv21*Vd12(jl)-InvUv11*Vd11(jl))*Ud11(il)
          Pmat(ioff+1,joff+2)=(-InvUv22*Vd22(jl)-InvUv12*Vd21(jl))*Ud12(il)+
     &     (InvUv21*Vd22(jl)+InvUv11*Vd21(jl))*Ud11(il)
          Pmat(ioff+1,joff+3)=(InvUv22*Vu12(jl)+InvUv12*Vu11(jl))*Ud12(il)+
     &     (-InvUv21*Vu12(jl)-InvUv11*Vu11(jl))*Ud11(il)
          Pmat(ioff+1,joff+4)=(-InvUv22*Vu22(jl)-InvUv12*Vu21(jl))*Ud12(il)+
     &     (InvUv21*Vu22(jl)+InvUv11*Vu21(jl))*Ud11(il)
          Pmat(ioff+2,joff+1)=(InvUv22*Vd12(jl)+InvUv12*Vd11(jl))*Ud22(il)+
     &     (-InvUv21*Vd12(jl)-InvUv11*Vd11(jl))*Ud21(il)
          Pmat(ioff+2,joff+2)=(-InvUv22*Vd22(jl)-InvUv12*Vd21(jl))*Ud22(il)+
     &     (InvUv21*Vd22(jl)+InvUv11*Vd21(jl))*Ud21(il)
          Pmat(ioff+2,joff+3)=(InvUv22*Vu12(jl)+InvUv12*Vu11(jl))*Ud22(il)+
     &     (-InvUv21*Vu12(jl)-InvUv11*Vu11(jl))*Ud21(il)
          Pmat(ioff+2,joff+4)=(-InvUv22*Vu22(jl)-InvUv12*Vu21(jl))*Ud22(il)+
     &     (InvUv21*Vu22(jl)+InvUv11*Vu21(jl))*Ud21(il)
          Pmat(ioff+3,joff+1)=(InvUv22*Vd12(jl)+InvUv12*Vd11(jl))*Uu12(il)+
     &     (-InvUv21*Vd12(jl)-InvUv11*Vd11(jl))*Uu11(il)
          Pmat(ioff+3,joff+2)=(-InvUv22*Vd22(jl)-InvUv12*Vd21(jl))*Uu12(il)+
     &     (InvUv21*Vd22(jl)+InvUv11*Vd21(jl))*Uu11(il)
          Pmat(ioff+3,joff+3)=(InvUv22*Vu12(jl)+InvUv12*Vu11(jl))*Uu12(il)+
     &     (-InvUv21*Vu12(jl)-InvUv11*Vu11(jl))*Uu11(il)
          Pmat(ioff+3,joff+4)=(-InvUv22*Vu22(jl)-InvUv12*Vu21(jl))*Uu12(il)+
     &     (InvUv21*Vu22(jl)+InvUv11*Vu21(jl))*Uu11(il)
          Pmat(ioff+4,joff+1)=(InvUv22*Vd12(jl)+InvUv12*Vd11(jl))*Uu22(il)+
     &     (-InvUv21*Vd12(jl)-InvUv11*Vd11(jl))*Uu21(il)
          Pmat(ioff+4,joff+2)=(-InvUv22*Vd22(jl)-InvUv12*Vd21(jl))*Uu22(il)+
     &     (InvUv21*Vd22(jl)+InvUv11*Vd21(jl))*Uu21(il)
          Pmat(ioff+4,joff+3)=(InvUv22*Vu12(jl)+InvUv12*Vu11(jl))*Uu22(il)+
     &     (-InvUv21*Vu12(jl)-InvUv11*Vu11(jl))*Uu21(il)
          Pmat(ioff+4,joff+4)=(-InvUv22*Vu22(jl)-InvUv12*Vu21(jl))*Uu22(il)+
     &     (InvUv21*Vu22(jl)+InvUv11*Vu21(jl))*Uu21(il)
	    enddo !il-loop
	  enddo !jl-loop

*     Calulate Upper triangle of Pmat
	  Do il=1,Pdim
	    Do jl=il+1,Pdim
		  Pmat(il,jl)=Pmat(jl,il)*((-1)**(il-jl))
		enddo !jl
	  enddo !il
      
	  TYpe*,'Pmat(1,1)=',Pmat(1,1)
	  TYpe*,'Pmat(1,2)=',Pmat(1,2)
	  TYpe*,'Pmat(2,1)=',Pmat(2,1)
	  TYpe*,'Pmat(2,2)=',Pmat(2,2)
	  TYpe*,'Pmat(1,3)=',Pmat(1,3)
	  TYpe*,'Pmat(1,4)=',Pmat(1,4)
	  TYpe*,'Pmat(2,3)=',Pmat(2,3)
	  TYpe*,'Pmat(2,4)=',Pmat(2,4)
	  TYpe*,'Pmat(3,3)=',Pmat(3,3)
	  TYpe*,'Pmat(3,4)=',Pmat(3,4)
	  TYpe*,'Pmat(4,3)=',Pmat(4,3)
	  TYpe*,'Pmat(4,4)=',Pmat(4,4)

	  TYpe*,'Pmat(3,1)=',Pmat(3,1)
	  TYpe*,'Pmat(3,2)=',Pmat(3,2)
	  TYpe*,'Pmat(4,1)=',Pmat(4,1)
	  TYpe*,'Pmat(4,2)=',Pmat(4,2)

	  pause
      return
	  end
	  
	  Subroutine setSY(Nl,S11,S21,S22,S31,S41,S42,Yd11,Yd12,Yd21,Yd22,
     &           Yu11,Yu12,Yu21,Yu22,SY11,SY12,SY21,SY22,SY31,SY32,SY41,SY42)
	  Implicit none
	  integer  Nl,il
	  Complex  SY11(Nl),SY12(Nl),SY21(Nl),SY22(Nl)
	  Complex  SY31(Nl),SY32(Nl),SY41(Nl),SY42(Nl)
	  Complex  S11(Nl),S21(Nl),S22(Nl)
	  Complex  S31(Nl),S41(Nl),S42(Nl)	  
	  Complex  Yd11(Nl),Yd12(Nl),Yd21(Nl),Yd22(Nl)
	  Complex  Yu11(Nl),Yu12(Nl),Yu21(Nl),Yu22(Nl)

*     set SY
      Do il=1,Nl
        SY11(il)=-S21(il)*Yd21(il)+S11(il)*Yd11(il)-S41(il)*Yu21(il)+
     &        S31(il)*Yu11(il)
	    SY12(il)=-S21(il)*Yd22(il)+S11(il)*Yd12(il)-S41(il)*Yu22(il)+
     &        S31(il)*Yu12(il)
	    SY21(il)=S22(il)*Yd21(il)+S21(il)*Yd11(il)+S42(il)*Yu21(il)-
     &        S41(il)*Yu11(il)
	    SY22(il)=S22(il)*Yd22(il)+S21(il)*Yd12(il)+S42(il)*Yu22(il)-
     &        S41(il)*Yu12(il)
	    SY31(il)=S41(il)*Yd21(il)+S31(il)*Yd11(il)+S21(il)*Yu21(il)+
     &        S11(il)*Yu11(il)
	    SY32(il)=S41(il)*Yd22(il)+S31(il)*Yd12(il)+S21(il)*Yu22(il)+
     &        S11(il)*Yu12(il)
	    SY41(il)=S42(il)*Yd21(il)+S41(il)*Yd11(il)+S22(il)*Yu21(il)-
     &        S21(il)*Yu11(il)
	    SY42(il)=S42(il)*Yd22(il)+S41(il)*Yd12(il)+S22(il)*Yu22(il)-
     &        S21(il)*Yu12(il)
	  End do
      return
	  end
	  

	  
      Subroutine DSFS(Nl,ea,eb,Rd11,Rd12,Rd21,Rd22,Td11,Td12,Td21,Td22,
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
	  Complex InvDet,a,b,c,d,C1,Wt11,Wt12,Wt21,Wt22
	  Complex t11,t12,t21,t22
	  Complex RRt11,RRt21,RRt12,RRt22,RRb11,RRb21,RRb12,RRb22
	  Complex ea(0:Nl),eb(0:Nl),RRd11(0:Nl),RRd12(0:Nl)
	  Complex RRd21(0:Nl),RRd22(0:Nl)
	  Complex RRu11(0:Nl),RRu12(0:Nl),RRu21(0:Nl),RRu22(0:Nl)
	  Complex TTu11(0:Nl),TTu12(0:Nl),TTu21(0:Nl),TTu22(0:Nl)
	  Complex TTd11(0:Nl),TTd12(0:Nl),TTd21(0:Nl),TTd22(0:Nl)
	  Complex Tu11(0:Nl),Tu12(0:Nl),Tu21(0:Nl),Tu22(0:Nl)
	  Complex Td11(0:Nl),Td12(0:Nl),Td21(0:Nl),Td22(0:Nl)
	  Complex Rd11(0:Nl),Rd12(0:Nl),Rd21(0:Nl),Rd22(0:Nl)
	  Complex Ru11(0:Nl),Ru12(0:Nl),Ru21(0:Nl),Ru22(0:Nl)
      Complex RF11,RF12,RF21,RF22
	  Complex Uu11(0:Nl),Uu12(0:Nl),Uu21(0:Nl),Uu22(0:Nl)
	  Complex Ud11(0:Nl),Ud12(0:Nl),Ud21(0:Nl),Ud22(0:Nl)
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
*	    Type*,'il,TTd11(il),TTd21(il),TTd22(il)='
*	    Type*,il,TTd11(il),TTd21(il),TTd22(il)

*       set RRd(il)=Rd(il)+Tu(il)*RRt*TTd(il)	  
      
        RRd11(il)=Rd11(il)+RRt22*TTd21(il)*Tu12(il)+RRt21*TTd11(il)*Tu12(il)+
     &          RRt12*TTd21(il)*Tu11(il)+RRt11*TTd11(il)*Tu11(il)

        RRd12(il)=Rd12(il)+RRt11*Tu11(il)*TTd12(il)+RRt12*Tu11(il)*TTd22(il)+
     &          RRt21*Tu12(il)*TTd12(il)+RRt22*Tu12(il)*TTd22(il)


        RRd21(il)=Rd21(il)+RRt22*TTd21(il)*Tu22(il)+RRt21*TTd11(il)*Tu22(il)+
     &          RRt12*TTd21(il)*Tu21(il)+RRt11*TTd11(il)*Tu21(il)

        RRd22(il)=Rd22(il)+RRt22*TTd22(il)*Tu22(il)+RRt21*TTd12(il)*Tu22(il)+
     &          RRt12*TTd22(il)*Tu21(il)+RRt11*TTd12(il)*Tu21(il)
*	    Type*,'il,RRd11(il),RRd12(il),RRd21(il),RRd22(il)='
*	    Type*,il,RRd11(il),RRd12(il),RRd21(il),RRd22(il)
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
        InvDet=C1/(a*d-b*c)
	  
	    a=InvDet*a
	    b=InvDet*b
	    c=InvDet*c
	    d=InvDet*d
*        Type*,'a,b,c,d=',a,b,c,d
*		pause
        Ud11(0)=Wt11*a+Wt21*b
        Ud12(0)=Wt12*a+Wt22*b
        Ud21(0)=Wt11*c+Wt21*d
        Ud22(0)=Wt12*c+Wt22*d
*		type*,'Ud11(0)=',Ud11(0)
*		type*,'Ud12(0)=',Ud12(0)
*		type*,'Ud21(0)=',Ud21(0)
*		type*,'Ud22(0)=',Ud22(0)
	  
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
		
*		Type*,'il,Ud11(il),Ud12(il),Ud21(il),Ud22(il)='
*		Type*,il,Ud11(il),Ud12(il),Ud21(il),Ud22(il)
*		Type*,'il,Uu11(il),Uu12(il),Uu21(il),Uu22(il)='
*		Type*,il,Uu11(il),Uu12(il),Uu21(il),Uu22(il)
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
      		
*      Type*,'Nlp1,Ud11(Nlp1),Ud12(Nlp1),Ud21(Nlp1),Ud22(Nlp1)='
*	  Type*,Nlp1,Ud11(Nlp1),Ud12(Nlp1),Ud21(Nlp1),Ud22(Nlp1)
*	  Type*,'Nlp1,Uu11(Nlp1),Uu12(Nlp1),Uu21(Nlp1),Uu22(Nlp1)='
*	  Type*,Nlp1,Uu11(Nlp1),Uu12(Nlp1),Uu21(Nlp1),Uu22(Nlp1)
	  

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
*		type*,'a,b,c,d=',a,b,c,d

*       set TTu(il)=(1-Rd(il)*RRb)^(-1)*Tu(il)

        TTu11(il)=Tu11(il)*a+Tu21(il)*b
        TTu12(il)=Tu12(il)*a+Tu22(il)*b
        TTu21(il)=Tu11(il)*c+Tu21(il)*d
        TTu22(il)=Tu12(il)*c+Tu22(il)*d
		
*	    Type*,'il,TTu11(il),TTu21(il),TTu22(il)='
*	    Type*,il,TTu11(il),TTu21(il),TTu22(il)

*       set RRu(ilp1)=Ru(il)+Td(il)*RRb*TTu(il)

        RRu11(ilp1)=Ru11(il)+RRb22*TTu21(il)*Td12(il)+RRb21*TTu11(il)*Td12(il)+
     &            RRb12*TTu21(il)*Td11(il)+RRb11*TTu11(il)*Td11(il)

        RRu12(ilp1)=Ru12(il)+Td11(il)*RRb11*TTu12(il)+Td11(il)*RRb12*TTu22(il)+
     &				  Td12(il)*RRb21*TTu12(il)+Td12(il)*RRb22*TTu22(il)

        RRu21(ilp1)=Ru21(il)+RRb22*TTu21(il)*Td22(il)+RRb21*TTu11(il)*Td22(il)+
     &            RRb12*TTu21(il)*Td21(il)+RRb11*TTu11(il)*Td21(il)

        RRu22(ilp1)=Ru22(il)+RRb22*TTu22(il)*Td22(il)+RRb21*TTu12(il)*Td22(il)+
     &            RRb12*TTu22(il)*Td21(il)+RRb11*TTu12(il)*Td21(il)
*	    Type*,'ilp1,RRu11(ilp1),RRu21(ilp1),RRu22(ilp1)='
*	    Type*,ilp1,RRu11(ilp1),RRu21(ilp1),RRu22(ilp1)
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

*	  Type*,'il,Vd11(il),Vd12(il),Vd21(il),Vd22(il)='
*	  Type*,Nl,Vd11(Nl),Vd12(Nl),Vd21(Nl),Vd22(Nl)
*	  Type*,'il,Vu11(il),Vu12(il),Vu21(il),Vu22(il)='
*	  Type*,Nl,Vu11(Nl),Vu12(Nl),Vu21(Nl),Vu22(Nl)
	  
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
*		Type*,'il,Vd11(il),Vd12(il),Vd21(il),Vd22(il)='
*		Type*,il,Vd11(il),Vd12(il),Vd21(il),Vd22(il)
*		Type*,'il,Vu11(il),Vu12(il),Vu21(il),Vu22(il)='
*		Type*,il,Vu11(il),Vu12(il),Vu21(il),Vu22(il)

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


      Subroutine SetS(il,Nl,ea,eb,Rd11,Rd12,Rd21,Rd22,Td11,Td12,Td21,Td22,
     &    Tu11,Tu12,Tu21,Tu22,Ru11,Ru12,Ru21,Ru22,S11,S21,S22,S31,S41,S42)
	  Implicit None
	  Integer il,ilm1,Nl
	  Complex InvDet,a,b,c,d,C1
	  Complex RRt11,RRt21,RRt12,RRt22,RRb11,RRb21,RRb12,RRb22
	  Complex ea,eb,RRd11,RRd12,RRd21,RRd22
	  Complex TTd11,TTd12,TTd21,TTd22
	  Complex Tu11(0:Nl),Tu12(0:Nl),Tu21(0:Nl),Tu22(0:Nl)
	  Complex Td11(0:Nl),Td12(0:Nl),Td21(0:Nl),Td22(0:Nl)
	  Complex Rd11(0:Nl),Rd12(0:Nl),Rd21(0:Nl),Rd22(0:Nl)
      Complex Ru11(0:Nl),Ru12(0:Nl),Ru21(0:Nl),Ru22(0:Nl)
	  Complex S11,S21,S22,S31,S41,S42

	  C1=Cmplx(1.0,0.0)
	  
	  ilm1=il-1
	  RRd11=Rd11(il)
	  RRd12=Rd12(il)
	  RRd21=Rd21(il)
	  RRd22=Rd22(il)
	  
	  RRt11=RRd11*ea*ea
	  RRt12=RRd12*ea*eb
	  RRt21=RRd21*ea*eb
	  RRt22=RRd22*eb*eb
	  
	  a=(C1-RRt22*Ru22(ilm1)-RRt12*Ru21(ilm1))
	  b=(RRt22*Ru12(ilm1)+RRt12*Ru11(ilm1))	  
	  c=(RRt21*Ru22(ilm1)+RRt11*Ru21(ilm1))
	  d=(C1-RRt21*Ru12(ilm1)-RRt11*Ru11(ilm1))
	  InvDet=C1/(a*d-b*c)
	  
	  a=InvDet*a
	  b=InvDet*b
	  c=InvDet*c
	  d=InvDet*d
	  
	  TTd11=Td11(ilm1)*a+Td21(ilm1)*b
	  TTd12=Td12(ilm1)*a+Td22(ilm1)*b
	  TTd21=Td11(ilm1)*c+Td21(ilm1)*d
	  TTd22=Td12(ilm1)*c+Td22(ilm1)*d
	  
      
	  S11=Rd11(ilm1)+RRt22*TTd21*Tu12(ilm1)+RRt21*TTd11*Tu12(ilm1)+
     &          RRt12*TTd21*Tu11(ilm1)+RRt11*TTd11*Tu11(ilm1)

      S21=Rd21(ilm1)+RRt22*TTd21*Tu22(ilm1)+RRt21*TTd11*Tu22(ilm1)+
     &          RRt12*TTd21*Tu21(ilm1)+RRt11*TTd11*Tu21(ilm1)

      S22=Rd22(ilm1)+RRt22*TTd22*Tu22(ilm1)+RRt21*TTd12*Tu22(ilm1)+
     &          RRt12*TTd22*Tu21(ilm1)+RRt11*TTd12*Tu21(ilm1)


      S31=eb*TTd21*Td12(il)+ea*TTd11*Td11(il)

      S41=eb*TTd21*Td22(il)+ea*TTd11*Td21(il)

      S42=eb*TTd22*Td22(il)+ea*TTd12*Td21(il)
	  return
	  end

	  Subroutine setXS(Nl,S11,S21,S22,S31,S41,S42,Xd11,Xd12,Xd21,Xd22,
     &           Xu11,Xu12,Xu21,Xu22,XS11,XS12,XS21,XS22,XS31,XS32,XS41,XS42)
	  Implicit none
	  integer  Nl,il
	  Complex  XS11(Nl),XS12(Nl),XS21(Nl),XS22(Nl)
	  Complex  XS31(Nl),XS32(Nl),XS41(Nl),XS42(Nl)
	  Complex  S11(Nl),S21(Nl),S22(Nl)
	  Complex  S31(Nl),S41(Nl),S42(Nl)	  
	  Complex  Xd11(Nl),Xd12(Nl),Xd21(Nl),Xd22(Nl)
	  Complex  Xu11(Nl),Xu12(Nl),Xu21(Nl),Xu22(Nl)

*     set X^t*S
      Do il=1,Nl
        XS11(il)=Xu21(il)*S41(il)+Xu11(il)*S31(il)+Xd21(il)*S21(il)+
     &         Xd11(il)*S11(il)
	    XS12(il)=Xu22(il)*S41(il)+Xu12(il)*S31(il)+Xd22(il)*S21(il)+
     &         Xd12(il)*S11(il)
	    XS21(il)=Xu21(il)*S42(il)+Xu11(il)*S41(il)+Xd21(il)*S22(il)-
     &         Xd11(il)*S21(il)
	    XS22(il)=Xu22(il)*S42(il)+Xu12(il)*S41(il)+Xd22(il)*S22(il)-
     &         Xd12(il)*S21(il)
	    XS31(il)=-Xd21(il)*S41(il)+Xd11(il)*S31(il)-Xu21(il)*S21(il)+
     &         Xu11(il)*S11(il)
	    XS32(il)=-Xd22(il)*S41(il)+Xd12(il)*S31(il)-Xu22(il)*S21(il)+
     &         Xu12(il)*S11(il)
	    XS41(il)=Xd21(il)*S42(il)-Xd11(il)*S41(il)+Xu21(il)*S22(il)+
     &         Xu11(il)*S21(il)
	    XS42(il)=Xd22(il)*S42(il)-Xd12(il)*S41(il)+Xu22(il)*S22(il)+
     &         Xu12(il)*S21(il)

	  End do
      return
	  end

      Subroutine setXSP(Nl,Pdim,Pmat,XS11,XS12,XS21,XS22,XS31,XS32,
     &     XS41,XS42,Xd11,Xd12,Xd21,Xd22,Xu11,Xu12,Xu21,Xu22)
*     set X=XS^t*P
      Implicit none
	  Integer  Nl,Pdim,il,jl,ioff,joff
	  Complex  Pmat(Pdim,Pdim)
	  Complex  XS11(Nl),XS12(Nl),XS21(Nl),XS22(Nl)
	  Complex  XS31(Nl),XS32(Nl),XS41(Nl),XS42(Nl)	  
	  Complex  Xd11(Nl),Xd12(Nl),Xd21(Nl),Xd22(Nl)
	  Complex  Xu11(Nl),Xu12(Nl),Xu21(Nl),Xu22(Nl)
	  
      Do il=1,Nl
	    Xd11(il)=(0.0,0.0)
	    Xd12(il)=(0.0,0.0)
	    Xd21(il)=(0.0,0.0)
	    Xd22(il)=(0.0,0.0)
	    Xu11(il)=(0.0,0.0)
	    Xu12(il)=(0.0,0.0)
	    Xu21(il)=(0.0,0.0)
	    Xu22(il)=(0.0,0.0)
		ioff=4*(il-1)
		Do jl=1,Nl
		  joff=4*(jl-1)
          Xd11(il)=Pmat(4+joff,ioff+1)*XS41(jl)+Pmat(3+joff,ioff+1)*XS31(jl)+
     &         Pmat(2+joff,ioff+1)*XS21(jl)+Pmat(1+joff,ioff+1)*XS11(jl)
          Xd12(il)=Pmat(4+joff,ioff+1)*XS42(jl)+Pmat(3+joff,ioff+1)*XS32(jl)+
     &         Pmat(2+joff,ioff+1)*XS22(jl)+Pmat(1+joff,ioff+1)*XS12(jl)
          Xd21(il)=Pmat(4+joff,ioff+2)*XS41(jl)+Pmat(3+joff,ioff+2)*XS31(jl)+
     &         Pmat(2+joff,ioff+2)*XS21(jl)+Pmat(1+joff,ioff+2)*XS11(jl)
          Xd22(il)=Pmat(4+joff,ioff+2)*XS42(jl)+Pmat(3+joff,ioff+2)*XS32(jl)+
     &         Pmat(2+joff,ioff+2)*XS22(jl)+Pmat(1+joff,ioff+2)*XS12(jl)
          Xu11(il)=Pmat(4+joff,ioff+3)*XS41(jl)+Pmat(3+joff,ioff+3)*XS31(jl)+
     &         Pmat(2+joff,ioff+3)*XS21(jl)+Pmat(1+joff,ioff+3)*XS11(jl)
          Xu12(il)=Pmat(4+joff,ioff+3)*XS42(jl)+Pmat(3+joff,ioff+3)*XS32(jl)+
     &         Pmat(2+joff,ioff+3)*XS22(jl)+Pmat(1+joff,ioff+3)*XS12(jl)
          Xu21(il)=Pmat(4+joff,ioff+4)*XS41(jl)+Pmat(3+joff,ioff+4)*XS31(jl)+
     &         Pmat(2+joff,ioff+4)*XS21(jl)+Pmat(1+joff,ioff+4)*XS11(jl)
          Xu22(il)=Pmat(4+joff,ioff+4)*XS42(jl)+Pmat(3+joff,ioff+4)*XS32(jl)+
     &         Pmat(2+joff,ioff+4)*XS22(jl)+Pmat(1+joff,ioff+4)*XS12(jl)
		Enddo !jl-loop
      Enddo !il-loop
      return
	  end

      SUBROUTINE CMPLXGJ(A,N,NP,B,M,MP,IPIV,INDXR,INDXC,NMAX)	
	  IMPLICIT NONE
	  INTEGER  NMAX,NP,N,MP,M,IPIV(NMAX),INDXR(NMAX),INDXC(NMAX)
	  INTEGER  IROW,ICOL,I,J,K,L,LL
	  REAL     BIG,EPSMIN
	  xcomplex  A(NP,NP),B(NP,MP),DUM,C1,PIVINV
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
