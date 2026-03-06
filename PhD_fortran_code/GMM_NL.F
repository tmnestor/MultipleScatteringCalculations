      Program  GMMNL
	  Implicit none
	  Integer  Nl0,rfN1,Nl,Nlp1,Pdim,i,j,Np,outfile
	  Parameter(Np=1,Nl0=2,Nl=3*Nl0,outfile=7)
	  Parameter(Nlp1=Nl+1,Pdim=4*Nlp1+2,rfN1=Nl0+1) 
      Complex  Pmat(Pdim,Pdim),B(Pdim),X(4*Nl0*Np)
	  INTEGER  IPIV(PDim),INDXR(PDim),INDXC(PDim)
      Real     rho0(0:rfN1),alpha0(0:rfN1),beta0(0:rfN1)
      Real     rho(0:Nlp1),alpha(0:Nlp1),beta(0:Nlp1)
      Real     drho(1:Nl0),dalpha(1:Nl0),dbeta(1:Nl0)
	  Real     w,p(Np),h0(0:Nl0),h(0:Nl),pmax,dp,rt3
	  Integer  ip,il,ioff
*******************************************************
	  Complex Md(2,2),Nd(2,2),Mu(2,2),Nu(2,2)
	  Complex ea,eb
	  Complex VERTICALSLO,eta,neta
******************************************************

*	  DATA     alpha0/1.51,2.01,3.01,4.01/
*	  DATA     beta0/1.01,1.51,2.51,3.51/	
*      DATA     rho0/1.01,1.51,2.51,3.51/

	  DATA     beta0/1.51,1.51,1.51,1.51/	
      DATA     rho0/1.51,1.51,1.51,1.51/



	  DATA     dalpha/2*0.0/
      DATA     drho/2*0.1/
	  DATA     dbeta/2*0.1/

*      DATA     drho/2*0.0/
*	  DATA     dbeta/2*0.0/
	  
	  DATA     W/1.0/
	  DATA     h0/3*1.0/
	  Data     pmax/.1/

      open(outfile,file='GMRESDat',status='unknown')
      dp=2.0*Pmax/float(Np-1)
	  Do ip=1,Np
	    P(ip)=-pmax+(ip-1)*dp
		Type*,'ip,p(ip)=',ip,p(ip)
	  Enddo
	  If(np.eq.1)then
	    p(1)=-.1
	  endif

	  rt3=Sqrt(3.0)
*     load model
      rho(0)=rho0(0)
      alpha(0)=rt3*beta0(0)
      beta(0)=beta0(0)
      h(0)=h0(0)
      Do il=1,Nl0
	    ioff=3*(il-1)
	    rho(ioff+1)=rho0(il)
		rho(ioff+2)=rho0(il)+drho(il)
	    rho(ioff+3)=rho0(il)
	    alpha(ioff+1)=rt3*beta0(il)
		alpha(ioff+2)=rt3*beta0(il)+rt3*dbeta(il)
	    alpha(ioff+3)=rt3*beta0(il)
	    beta(ioff+1)=beta0(il)
		beta(ioff+2)=beta0(il)+dbeta(il)
	    beta(ioff+3)=beta0(il)
		h(ioff+1)=0.0
		h(ioff+2)=h0(il)
		h(ioff+3)=0.0
	  Enddo
      rho(Nlp1)=rho0(rfN1)
      alpha(Nlp1)=rt3*beta0(rfN1)
      beta(Nlp1)=beta0(rfN1)


      Do il=0,Nl+1
		type*,'rho(',il,')=',rho(il)
	  Enddo

      Do il=0,Nl+1
		type*,'alpha(',il,')=',alpha(il)
	  Enddo

      Do il=0,Nl+1
		type*,'beta(',il,')=',beta(il)
	  Enddo


      Do ip=1,Np
        call setGM(Pdim,Nl,Pmat,B,rho,alpha,beta,p(ip),W,h)

*	  DO I=1,Pdim
*		DO J=1,Pdim
*	      type*,'i,j,Pmat(I,J)=',i,j
*	      type*,Pmat(I,J)
*	      type*
*	    END DO 
*	  END DO
        
	    CALL CMPLXGJ(Pmat,Pdim,Pdim,B,1,1,IPIV,INDXR,INDXC,Pdim)

	    ETA=VERTICALSLO(ALPHA(0),P)
	    NETA=VERTICALSLO(Beta(0),P)
	    CALL Emat(P,ETA,NETA,rho(0),BETA(0),MD,ND,MU,NU)
	    ea=CEXP((0.0,1.0)*W*H(0)*ETA)
	    eb=CEXP((0.0,1.0)*W*H(0)*NETA)
		
        TYpe*,'Md(1,1)*B(1)+Md(1,2)*B(2)+Mu(1,1)*ea*B(3)+Mu(1,2)*eb*B(4)='
        TYpe*,Md(1,1)*B(1)+Md(1,2)*B(2)+Mu(1,1)*ea*B(3)+Mu(1,2)*eb*B(4)

        TYpe*,'Md(2,1)*B(1)+Md(2,2)*B(2)+Mu(2,1)*ea*B(3)+Mu(2,2)*eb*B(4)='
        TYpe*,Md(2,1)*B(1)+Md(2,2)*B(2)+Mu(2,1)*ea*B(3)+Mu(2,2)*eb*B(4)
		
	    Do il=1,Nl0
	      ioff=4+12*(il-1)
*	      Type*,'B(i)=',B(ioff+1)
*	      Type*,'B(i)=',B(ioff+2)
*	      Type*,'B(i)=',B(ioff+11)
*	      Type*,'B(i)=',B(ioff+12)
          X(4*Np*(il-1)+ip)=B(ioff+1)
          X(4*Np*(il-1)+Np+ip)=B(ioff+2)
          X(4*Np*(il-1)+Np*2+ip)=B(ioff+11)
          X(4*Np*(il-1)+Np*3+ip)=B(ioff+12)
        Enddo!il
      Enddo!ip
	  Do i=1,4*Nl0*Np
	    Type*,'Vexact(',i,')=',X(i)
	  Write(outfile,*), X(i)
	  Enddo
	  close (outfile)
	  stop'normal completion'
	  end

	  Subroutine setGM(Pdim,Nl,Pmat,B,rho0,alpha0,beta0,p,W,h)
	  Implicit none
	  Integer Nl,Nlp1,Pdim,il,ioff,joff,koff,i,j
      Complex Pmat(Pdim,Pdim),B(Pdim)
      Real     rho0(0:*),alpha0(0:*),beta0(0:*)
	  Real     w,p,h(0:Nl),Zs

	  
	  Complex Md1(2,2),Nd1(2,2),Mu1(2,2),Nu1(2,2)
	  Complex Md(2,2),Nd(2,2),Mu(2,2),Nu(2,2)
	  Complex ea1,ea,eb1,eb,easf,ebsf,eas0,ebs0
	  Complex SigPU,SigSU,SigPD,SigSD
	  Complex VERTICALSLO,T(2,2),eta1,neta1,eta,neta
	  Complex Ci,C0,C1
*     source directivity
      SigPU=(1.0,0.0)
      SigSU=(0.0,0.0)
      SigPD=(1.0,0.0)
      SigSD=(0.0,0.0)
	  Zs=0.5*h(0)
      Nlp1=Nl+1
	  CI=(0.0,1.0)
	  C0=(0.0,0.0)
	  C1=(1.0,0.0)
      type*,'PDIM=',PDIM
	  DO I=1,Pdim
		B(I)=C0
		DO J=1,Pdim
	      Pmat(I,J)=C0
	    END DO 
	  END DO

******************************
*     set the free-surface 
	  ETA=VERTICALSLO(ALPHA0(0),P)
	  NETA=VERTICALSLO(Beta0(0),P)
	  ea=CEXP(CI*W*H(0)*ETA)
	  eb=CEXP(CI*W*H(0)*NETA)
	  easf=CEXP(CI*W*zs*ETA)
	  ebsf=CEXP(CI*W*zs*NETA)
	  eas0=CEXP(CI*W*(H(0)-zs)*ETA)
	  ebs0=CEXP(CI*W*(H(0)-zs)*NETA)
	  CALL Emat(P,ETA,NETA,rho0(0),BETA0(0),MD,ND,MU,NU)
      T(1,1)=Nd(1,1)
	  T(2,1)=Nd(2,1)
	  T(1,2)=Nd(1,2)
	  T(2,2)=Nd(2,2)
      CALL ASSEMBLE(Pmat,Pdim,Pdim,T,1,1)

      T(1,1)=Nu(1,1)*ea
	  T(2,1)=Nu(2,1)*ea
	  T(1,2)=Nu(1,2)*eb
	  T(2,2)=Nu(2,2)*eb
*      CALL ASSEMBLE(Pmat,Pdim,Pdim,T,1,3)  !uncomment to include Free Surface

*      B(1)=-Nu(1,1)*easf*SigPU
*	  B(2)=-Nu(2,1)*easf*SigPU


*	  B(1)=Nu(1,2)*ebsf*SigSU
*	  B(2)=Nu(2,2)*ebsf*SigSU
*********************************************
*     set interface 0
	  Do i=1,2
	    Do j=1,2
		  Md1(i,j)=Md(i,j)
		  Mu1(i,j)=Mu(i,j)
		  Nd1(i,j)=Nd(i,j)
		  Nu1(i,j)=Nu(i,j)
		Enddo
	  Enddo

	  ETA1=eta
	  NETA1=neta
	  EA1=ea
	  EB1=eb


	  ETA=VERTICALSLO(ALPHA0(1),P)
	  NETA=VERTICALSLO(Beta0(1),P)
	  ea=CEXP(CI*W*H(1)*ETA)
	  eb=CEXP(CI*W*H(1)*NETA)
	  CALL Emat(P,ETA,NETA,rho0(1),BETA0(1),MD,ND,MU,NU)
      ioff=2
	  joff=0

      T(1,1)=Md1(1,1)*EA1
	  T(2,1)=Md1(2,1)*EA1
	  T(1,2)=Md1(1,2)*EB1
	  T(2,2)=Md1(2,2)*EB1
      CALL ASSEMBLE(Pmat,Pdim,Pdim,T,ioff+1,joff+1)

      T(1,1)=Nd1(1,1)*EA1
	  T(2,1)=Nd1(2,1)*EA1
	  T(1,2)=Nd1(1,2)*EB1
	  T(2,2)=Nd1(2,2)*EB1
      CALL ASSEMBLE(Pmat,Pdim,Pdim,T,ioff+3,joff+1)
***********************
*      B(1)=(0.0,1.0)

      B(3)=-Md1(1,1)*EAs0*SigPD
	  B(4)=-Md1(2,1)*EAs0*SigPD
      B(5)=-Nd1(1,1)*EAs0*SigPD
	  B(6)=-Nd1(2,1)*EAs0*SigPD

*      B(3)=-Md1(1,1)
*	  B(4)=-Md1(2,1)
*      B(5)=-Nd1(1,1)
*	  B(6)=-Nd1(2,1)



*      B(3)=Mu(1,1)
*	  B(4)=Mu(2,1)
*      B(5)=Nu(1,1)
*	  B(6)=Nu(2,1)

*      B(3)=Mu(1,2)
*	  B(4)=Mu(2,2)
*      B(5)=Nu(1,2)
*	  B(6)=Nu(2,2)
	  
***********************
      T(1,1)=Mu1(1,1)
	  T(2,1)=Mu1(2,1)
	  T(1,2)=Mu1(1,2)
	  T(2,2)=Mu1(2,2)
      CALL ASSEMBLE(Pmat,Pdim,Pdim,T,ioff+1,joff+3)

      T(1,1)=Nu1(1,1)
	  T(2,1)=Nu1(2,1)
	  T(1,2)=Nu1(1,2)
	  T(2,2)=Nu1(2,2)
      CALL ASSEMBLE(Pmat,Pdim,Pdim,T,ioff+3,joff+3)

      T(1,1)=-Md(1,1)
	  T(2,1)=-Md(2,1)
	  T(1,2)=-Md(1,2)
	  T(2,2)=-Md(2,2)
      CALL ASSEMBLE(Pmat,Pdim,Pdim,T,ioff+1,joff+5)

      T(1,1)=-Nd(1,1)
	  T(2,1)=-Nd(2,1)
	  T(1,2)=-Nd(1,2)
	  T(2,2)=-Nd(2,2)
      CALL ASSEMBLE(Pmat,Pdim,Pdim,T,ioff+3,joff+5)


      T(1,1)=-Mu(1,1)*ea
	  T(2,1)=-Mu(2,1)*ea
	  T(1,2)=-Mu(1,2)*eb
	  T(2,2)=-Mu(2,2)*eb
      CALL ASSEMBLE(Pmat,Pdim,Pdim,T,ioff+1,joff+7)

      T(1,1)=-Nu(1,1)*ea
	  T(2,1)=-Nu(2,1)*ea
	  T(1,2)=-Nu(1,2)*eb
	  T(2,2)=-Nu(2,2)*eb
      CALL ASSEMBLE(Pmat,Pdim,Pdim,T,ioff+3,joff+7)

**************************************************
*     set interface il : il=1...Nl-1

      Do il=1,Nl-1

	    Do i=1,2
	      Do j=1,2
		    Md1(i,j)=Md(i,j)
		    Mu1(i,j)=Mu(i,j)
		    Nd1(i,j)=Nd(i,j)
		    Nu1(i,j)=Nu(i,j)
		  Enddo
	    Enddo

	    ETA1=eta
		NETA1=neta
		EA1=ea
		EB1=eb
		
		
		ETA=VERTICALSLO(ALPHA0(il+1),P)
		NETA=VERTICALSLO(Beta0(il+1),P)
		ea=CEXP(CI*W*H(il+1)*ETA)
		eb=CEXP(CI*W*H(il+1)*NETA)
		CALL Emat(P,ETA,NETA,rho0(il+1),BETA0(il+1),MD,ND,MU,NU)
		ioff=6+(il-1)*4
		joff=ioff-2
		
		T(1,1)=Md1(1,1)*EA1
		T(2,1)=Md1(2,1)*EA1
		T(1,2)=Md1(1,2)*EB1
		T(2,2)=Md1(2,2)*EB1
		CALL ASSEMBLE(Pmat,Pdim,Pdim,T,ioff+1,joff+1)
		
		T(1,1)=Nd1(1,1)*EA1
		T(2,1)=Nd1(2,1)*EA1
		T(1,2)=Nd1(1,2)*EB1
		T(2,2)=Nd1(2,2)*EB1
		CALL ASSEMBLE(Pmat,Pdim,Pdim,T,ioff+3,joff+1)
		
		T(1,1)=Mu1(1,1)
		T(2,1)=Mu1(2,1)
		T(1,2)=Mu1(1,2)
		T(2,2)=Mu1(2,2)
		CALL ASSEMBLE(Pmat,Pdim,Pdim,T,ioff+1,joff+3)
		
		T(1,1)=Nu1(1,1)
		T(2,1)=Nu1(2,1)
		T(1,2)=Nu1(1,2)
		T(2,2)=Nu1(2,2)
		CALL ASSEMBLE(Pmat,Pdim,Pdim,T,ioff+3,joff+3)
		
		T(1,1)=-Md(1,1)
		T(2,1)=-Md(2,1)
		T(1,2)=-Md(1,2)
		T(2,2)=-Md(2,2)
		CALL ASSEMBLE(Pmat,Pdim,Pdim,T,ioff+1,joff+5)
		
		T(1,1)=-Nd(1,1)
		T(2,1)=-Nd(2,1)
		T(1,2)=-Nd(1,2)
		T(2,2)=-Nd(2,2)
		CALL ASSEMBLE(Pmat,Pdim,Pdim,T,ioff+3,joff+5)
		
		
		T(1,1)=-Mu(1,1)*ea
		T(2,1)=-Mu(2,1)*ea
		T(1,2)=-Mu(1,2)*eb
		T(2,2)=-Mu(2,2)*eb
		CALL ASSEMBLE(Pmat,Pdim,Pdim,T,ioff+1,joff+7)
		
		T(1,1)=-Nu(1,1)*ea
		T(2,1)=-Nu(2,1)*ea
		T(1,2)=-Nu(1,2)*eb
		T(2,2)=-Nu(2,2)*eb
		CALL ASSEMBLE(Pmat,Pdim,Pdim,T,ioff+3,joff+7)
	  Enddo !il

**************************************************
*     set interface Nl
      il=Nl
	  Do i=1,2
	    Do j=1,2
		  Md1(i,j)=Md(i,j)
		  Mu1(i,j)=Mu(i,j)
		  Nd1(i,j)=Nd(i,j)
		  Nu1(i,j)=Nu(i,j)
		Enddo
	  Enddo

	  ETA1=eta
	  NETA1=neta
	  EA1=ea
	  EB1=eb


	  ETA=VERTICALSLO(ALPHA0(Nl+1),P)
	  NETA=VERTICALSLO(Beta0(Nl+1),P)
	  CALL Emat(P,ETA,NETA,rho0(Nl+1),BETA0(Nl+1),MD,ND,MU,NU)
      ioff=6+(Nl-1)*4
	  joff=ioff-2

      T(1,1)=Md1(1,1)*EA1
	  T(2,1)=Md1(2,1)*EA1
	  T(1,2)=Md1(1,2)*EB1
	  T(2,2)=Md1(2,2)*EB1
      CALL ASSEMBLE(Pmat,Pdim,Pdim,T,ioff+1,joff+1)

      T(1,1)=Nd1(1,1)*EA1
	  T(2,1)=Nd1(2,1)*EA1
	  T(1,2)=Nd1(1,2)*EB1
	  T(2,2)=Nd1(2,2)*EB1
      CALL ASSEMBLE(Pmat,Pdim,Pdim,T,ioff+3,joff+1)

      T(1,1)=Mu1(1,1)
	  T(2,1)=Mu1(2,1)
	  T(1,2)=Mu1(1,2)
	  T(2,2)=Mu1(2,2)
      CALL ASSEMBLE(Pmat,Pdim,Pdim,T,ioff+1,joff+3)

      T(1,1)=Nu1(1,1)
	  T(2,1)=Nu1(2,1)
	  T(1,2)=Nu1(1,2)
	  T(2,2)=Nu1(2,2)
      CALL ASSEMBLE(Pmat,Pdim,Pdim,T,ioff+3,joff+3)

      T(1,1)=-Md(1,1)
	  T(2,1)=-Md(2,1)
	  T(1,2)=-Md(1,2)
	  T(2,2)=-Md(2,2)
      CALL ASSEMBLE(Pmat,Pdim,Pdim,T,ioff+1,joff+5)

      T(1,1)=-Nd(1,1)
	  T(2,1)=-Nd(2,1)
	  T(1,2)=-Nd(1,2)
	  T(2,2)=-Nd(2,2)
      CALL ASSEMBLE(Pmat,Pdim,Pdim,T,ioff+3,joff+5)
	  
*	  B(Pdim-3)=-Md1(1,2)
*	  B(Pdim-2)=-Md1(2,2)
*	  B(Pdim-1)=-Nd1(1,2)
*	  B(Pdim)=-Nd1(2,2)
*     Incident P+SV-wave from the lower halfspace
*      B(Pdim-3)=Mu(1,1)+Mu(1,2)
*	  B(Pdim-2)=Mu(2,1)+Mu(2,2)

*      B(Pdim-1)=Nu(1,1)+Nu(1,2)
*	  B(Pdim)=Nu(2,1)+Nu(2,2)

*     Incident P-wave from the lower halfspace
*      B(Pdim-3)=Mu(1,1)
*	  B(Pdim-2)=Mu(2,1)

*      B(Pdim-1)=Nu(1,1)
*	  B(Pdim)=Nu(2,1)

*     Incident S-wave from the lower halfspace
*      B(Pdim-3)=Mu(1,2)
*	  B(Pdim-2)=Mu(2,2)

*      B(Pdim-1)=Nu(1,2)
*	  B(Pdim)=Nu(2,2)

	  return
	  end
	  
      SUBROUTINE ASSEMBLE(A,NP,MP,T,I,J)
      IMPLICIT NONE
      INTEGER NP,MP,I,J,IP1,JP1
	  COMPLEX A(NP,MP),T(2,2)
      IP1=I+1
	  JP1=J+1
      IF(IP1 .GT. NP) PAUSE 'IP1 OUT OF BOUNDS'
      IF(JP1 .GT. MP) PAUSE 'JP1 OUT OF BOUNDS'
	  A(I,J)=T(1,1)
	  A(I,JP1)=T(1,2)
	  A(IP1,J)=T(2,1)
	  A(IP1,JP1)=T(2,2)
	  RETURN
	  END

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
      
      SUBROUTINE CMPLXGJ(A,N,NP,B,M,MP,IPIV,INDXR,INDXC,NMAX)	
	  IMPLICIT NONE
	  INTEGER  NMAX,NP,N,MP,M,IPIV(NMAX),INDXR(NMAX),INDXC(NMAX)
	  INTEGER  IROW,ICOL,I,J,K,L,LL
	  REAL     BIG,EPSMIN
	  COMPLEX  A(NP,NP),B(NP,MP),DUM,C1,PIVINV
	  PARAMETER (C1=(1.0,0.0),EPSMIN=1.0E-04)
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
			    PAUSE 'MATRIX SINGULAR'
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
		
	    
