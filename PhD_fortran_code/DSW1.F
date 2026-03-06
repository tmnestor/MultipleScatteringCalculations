       Implicit none
       Real    w,rho0,rho1,ca0,ca1,cb0,cb1
       Complex P1,P2,P3,P4,P5,S1,S2,S4,S5
       Complex SigP1,SigP2,SigP3,SigP4,SigP5
       Complex SigS1,SigS2,SigS4,SigS5

       data w/3.0/

       data rho0,rho1/3.,3.15/
       data cb0,cb1/3.,3.15/
       data ca0,ca1/5.0,5.25/

       data P1,P2,P3,P4,P5/(0.0,1.0),(1.0,0.0),(1.0,1.0),(0.0,3.0),(0.0,5.0)/
       data S1,S2,S4,S5/(0.0,1.0),(1.0,1.0),(1.0,0.0),(2.0,1.0)/

       call TC(w,rho0,rho1,ca0,ca1,cb0,cb1,P1,P2,P3,P4,P5,
     &      S1,S2,S4,S5,SigP1,SigP2,SigP3,SigP4,SigP5,SigS1,
     &      SigS2,SigS4,SigS5)

      type*,'SigP1=',SigP1
      type*,'SigP2=',SigP2
      type*,'SigP3=',SigP3
      type*,'SigP4=',SigP4
      type*,'SigP5=',SigP5


      type*,'SigS1=',SigS1
      type*,'SigS2=',SigS2
      type*,'SigS4=',SigS4
      type*,'SigS5=',SigS5


       stop
       end

**********************************************************************
       Subroutine TC(w,rho0,rho1,ca0,ca1,cb0,cb1,P1,P2,P3,P4,P5,
     &      S1,S2,S4,S5,SigP1,SigP2,SigP3,SigP4,SigP5,SigS1,
     &      SigS2,SigS4,SigS5)
       Implicit none
       Real    w,rho0,rho1,ca0,ca1,cb0,cb1
       Complex P1,P2,P3,P4,P5,S1,S2,S4,S5
       Complex SigP1,SigP2,SigP3,SigP4,SigP5
       Complex SigS1,SigS2,SigS4,SigS5


       Real Cb12,Ca12,ca02,cb02,ca04,cb04
       Real mu,lam,w2,mu2,lam2
       Real drho,da,db,gam1,gam0,zet1,zet0,dgam,dzet

       Complex PP11,PP22,PP31,PP33,PP51
       Complex PS11,PS51,PS22,PS31
       Complex SS11,SS51,SS22,facP,facS,facPS

**********************************************************************
       Cb12=cb1*cb1
       Ca12=ca1*ca1
       ca02=ca0*ca0
       cb02=cb0*cb0
       ca04=ca02*ca02
       cb04=cb02*cb02
       mu=rho0*cb02
       lam=rho0*(Ca02-2.0*Cb02)
       w2=w*w
       mu2=mu*mu
       lam2=lam*lam



       drho=rho1-rho0
       da=1.0/(rho1*Ca12)-1.0/(rho0*Ca02)
       db=1.0/(rho1*Cb12)-1.0/(rho0*Cb02)
       gam1=1.0-2.0*cb12/ca12
       gam0=1.0-2.0*cb02/ca02
       zet1=4.0*rho1*cb12*(1.0-cb12/ca12)
       zet0=4.0*rho0*Cb02*(1.0-cb02/ca02)
       dgam=gam1-gam0
       dzet=zet1-zet0

       type*,'drho=',drho
       type*,'da=',da
       type*,'db=',db
       type*,'dzet=',dzet
       type*,'dgam=',dgam


*************************************************************************
	PP11=Cmplx(0.,-.015625)*(dzet - 4.*dgam*mu - 4.*da*mu2 - 
     -       4.*db*mu2)*w2/(ca04*rho0)
	  
	PP22=Cmplx(0.,.125)*drho*w2/(ca02*rho0)

	PP31=Cmplx(0.,-.03125)*(dzet + 2*dgam*lam + 4.*da*lam*mu + 
     -       4.*da*mu2)*w2/(ca04*rho0)
	  
	PP33=Cmplx(0.,-1./16)*(dzet + 4.*dgam*lam - 4.*da*lam2 + 
     -   4.*dgam*mu - 8*da*lam*mu - 4.*da*mu2)*w2/(ca04*rho0)
	  
	  
	PP51=Cmplx(0.,-.015625)*(dzet - 4.*dgam*mu - 4.*da*mu2 + 
     -       4.*db*mu2)*w2/(ca04*rho0)
	  
	type*,'PP11=',PP11
	type*,'PP22=',PP22
	type*,'PP31=',PP31
	type*,'PP33=',PP33
	type*,'PP51=',PP51

	PS11=(-dzet + 4.*dgam*mu + 4.*da*mu2 + 4.*db*mu2)*
     -     w2/(64*ca02*cb02*rho0)
	  
	PS22=drho*w2/(8*ca0*cb0*rho0)

	PS31=-((dzet + 2*dgam*lam + 4.*da*lam*mu + 4.*da*mu2)*
     -       w2)/(32*ca02*cb02*rho0)
	  
	PS51=-((dzet - 4.*dgam*mu - 4.*da*mu2 + 4.*db*mu2)*
     -       w2)/(64*ca02*cb02*rho0)
	
	type*,'PS11=',PS11
	type*,'PS22=',PS22
	type*,'PS31=',PS31
	type*,'PS51=',PS51
  
	SS11=Cmplx(0.,-.015625)*(dzet - 4.*dgam*mu - 4.*da*mu2 - 
     -       4.*db*mu2)*w2/(cb04*rho0)
	  
	SS22=Cmplx(0.,.125)*drho*w2/(cb02*rho0)

	SS51=Cmplx(0.,.015625)*(dzet - 4.*dgam*mu - 4.*da*mu2 + 
     -       4.*db*mu2)*w2/(cb04*rho0)

	type*,'SS11=',SS11
	type*,'SS22=',SS22
	type*,'SS51=',SS51

*********************************************************************
	SigP1=P5*PP51+P3*PP31+P1*PP11-PS51*S5+PS11*S1
	SigP2=P2*PP22+PS22*S2
	SigP3=P5*PP31+P1*PP31+P3*PP33-PS31*S5+PS31*S1
	SigP4=P4*PP22-PS22*S4
	SigP5=P1*PP51+P3*PP31+P5*PP11-PS11*S5+PS51*S1
	SigS1=-P5*PS51-P3*PS31-P1*PS11+S5*SS51+S1*SS11
	SigS2=-P2*PS22+S2*SS22
	SigS4=P4*PS22+S4*SS22
	SigS5=P1*PS51+P3*PS31+P5*PS11+S1*SS51+S5*SS11
      type*,'PP11,PP51,PP22,PP31,PP33=',PP11,PP51,PP22,PP31,PP33
      type*,'PS11,PS51,PS22,PS31=',PS11,PS51,PS22,PS31
      type*,'SS11=',SS11
      type*,'SS22=',SS22
      type*,'SS51=',SS51
*********************************************************************

      return
      end
	
