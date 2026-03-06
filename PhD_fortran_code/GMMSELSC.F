	  IMPLICIT NONE
	  INTEGER  NMAX,NP,MP,IL
      PARAMETER(NMAX=8,NP=8,MP=4)
	  INTEGER  IPIV(NMAX),INDXR(NMAX),INDXC(NMAX),I,J

C	  
	  COMPLEX  A(NP,NP),B(NP,MP),CI,C0,C1,EA2,EB2,rtea,rteb
	  COMPLEX  sigu11,sigu12,sigu21,sigu22
	  COMPLEX  sigd11,sigd12,sigd21,sigd22
      REAL     RHO(3),ALPHA(3),BETA(3),P,W,H2
      COMPLEX  Md1(2,2),Nd1(2,2),Nd2(2,2),Md2(2,2),Md3(2,2),Nd3(2,2)
      COMPLEX  Mu1(2,2),Nu1(2,2),Nu2(2,2),Mu2(2,2),Mu3(2,2),Nu3(2,2)
      COMPLEX  VERTICALSLO,T(2,2),ETA(3),NETA(3)

      DATA     rho/1.51,1.61,1.51/
	  DATA     alpha/2.6,2.7,2.6/
	  DATA     beta/1.5,1.6,1.5/
	  DATA     p,W/100.0,1.0/

	  CI=(0.0,1.0)
	  C0=(0.0,0.0)
	  C1=(1.0,0.0)

      h2=w/beta(2)/100000
	  DO il=1,3
	    ETA(il)=VERTICALSLO(ALPHA(il),P)
	    NETA(il)=VERTICALSLO(Beta(il),P)
*		type*,'eta(il),neta(il)=',eta(il),neta(il)
      end do
	  EA2=CEXP(CI*W*H2*ETA(2))
	  EB2=CEXP(CI*W*H2*NETA(2))
	  rtea=CEXP(0.5*CI*W*H2*ETA(2))
	  rteb=CEXP(0.5*CI*W*H2*NETA(2))
	  CALL Emat(P,ETA(1),NETA(1),RHO(1),BETA(1),MD1,ND1,MU1,NU1)
	  CALL Emat(P,ETA(2),NETA(2),RHO(2),BETA(2),MD2,ND2,MU2,NU2)
	  CALL Emat(P,ETA(3),NETA(3),RHO(3),BETA(3),MD3,ND3,MU3,NU3)
	  DO I=1,NP
	    DO J=1,MP
		  B(I,J)=C0
		END DO
		DO J=1,NP
	      A(I,J)=C0
	    END DO 
	  END DO
      T(1,1)=Mu1(1,1)
	  T(2,1)=Mu1(2,1)
	  T(1,2)=Mu1(1,2)
	  T(2,2)=Mu1(2,2)
      CALL ASSEMBLE(A,8,8,T,1,1)
      T(1,1)=Nu1(1,1)
	  T(2,1)=Nu1(2,1)
	  T(1,2)=Nu1(1,2)
	  T(2,2)=Nu1(2,2)
      CALL ASSEMBLE(A,8,8,T,3,1)

      T(1,1)=-Md2(1,1)
	  T(2,1)=-Md2(2,1)
	  T(1,2)=-Md2(1,2)
	  T(2,2)=-Md2(2,2)
      CALL ASSEMBLE(A,8,8,T,1,3)
      T(1,1)=-Nd2(1,1)
	  T(2,1)=-Nd2(2,1)
	  T(1,2)=-Nd2(1,2)
	  T(2,2)=-Nd2(2,2)
      CALL ASSEMBLE(A,8,8,T,3,3)

      T(1,1)=-Mu2(1,1)*EA2
	  T(2,1)=-Mu2(2,1)*EA2
	  T(1,2)=-Mu2(1,2)*EB2
	  T(2,2)=-Mu2(2,2)*EB2
      CALL ASSEMBLE(A,8,8,T,1,5)

      T(1,1)=-Nu2(1,1)*EA2
	  T(2,1)=-Nu2(2,1)*EA2
	  T(1,2)=-Nu2(1,2)*EB2
	  T(2,2)=-Nu2(2,2)*EB2
      CALL ASSEMBLE(A,8,8,T,3,5)
	  
      T(1,1)=Md2(1,1)*EA2
	  T(2,1)=Md2(2,1)*EA2
	  T(1,2)=Md2(1,2)*EB2
	  T(2,2)=Md2(2,2)*EB2
      CALL ASSEMBLE(A,8,8,T,5,3)

      T(1,1)=Nd2(1,1)*EA2
	  T(2,1)=Nd2(2,1)*EA2
	  T(1,2)=Nd2(1,2)*EB2
	  T(2,2)=Nd2(2,2)*EB2
      CALL ASSEMBLE(A,8,8,T,7,3)

      T(1,1)=Mu2(1,1)
	  T(2,1)=Mu2(2,1)
	  T(1,2)=Mu2(1,2)
	  T(2,2)=Mu2(2,2)
      CALL ASSEMBLE(A,8,8,T,5,5)

      T(1,1)=Nu2(1,1)
	  T(2,1)=Nu2(2,1)
	  T(1,2)=Nu2(1,2)
	  T(2,2)=Nu2(2,2)
      CALL ASSEMBLE(A,8,8,T,7,5)

      T(1,1)=-Md3(1,1)
	  T(2,1)=-Md3(2,1)
	  T(1,2)=-Md3(1,2)
	  T(2,2)=-Md3(2,2)
      CALL ASSEMBLE(A,8,8,T,5,7)

      T(1,1)=-Nd3(1,1)
	  T(2,1)=-Nd3(2,1)
	  T(1,2)=-Nd3(1,2)
	  T(2,2)=-Nd3(2,2)
      CALL ASSEMBLE(A,8,8,T,7,7)

*     set the incident field from the bounding subspaces
      B(1,1)=-Md1(1,1)
	  B(2,1)=-Md1(2,1)
	  B(3,1)=-Nd1(1,1)
	  B(4,1)=-Nd1(2,1)
      B(1,2)=-Md1(1,2)
	  B(2,2)=-Md1(2,2)
	  B(3,2)=-Nd1(1,2)
	  B(4,2)=-Nd1(2,2)


      B(5,3)=Mu3(1,1)
	  B(6,3)=Mu3(2,1)
	  B(7,3)=Nu3(1,1)
	  B(8,3)=Nu3(2,1)
      B(5,4)=Mu3(1,2)
	  B(6,4)=Mu3(2,2)
	  B(7,4)=Nu3(1,2)
	  B(8,4)=Nu3(2,2)



	  CALL CMPLXGJ(A,8,NP,B,4,MP,IPIV,INDXR,INDXC,NMAX)
	  

	  EA2=CEXP(CI*W*H2*ETA(1))
	  EB2=CEXP(CI*W*H2*NETA(1))

*      B(1,3)=B(1,3)-ea2
*      B(2,4)=B(2,4)-eb2
*      B(7,1)=B(7,1)-ea2
*      B(8,2)=B(8,2)-eb2
       type*,'ea2*ea2=',ea2*ea2
      type*,'PuPd=',B(1,1)
      type*,'PuSd=',B(1,2)
      type*,'SuPd=',B(2,1)
      type*,'SuSd=',B(2,2)

      type*,'PdPd=',B(7,1)
      type*,'SdPd=',B(8,1)
      type*,'PdSd=',B(7,2)
      type*,'SdSd=',B(8,2)

      type*,'PuPu=',B(1,3)
      type*,'PuSu=',B(1,4)
      type*,'SuPu=',B(2,3)
      type*,'SuSu=',B(2,4)


      type*,'PdPu=',B(7,3)
      type*,'PdSu=',B(7,4)
      type*,'SdPu=',B(8,3)
      type*,'SdSu=',B(8,4)
	  
	  type*,'Md1=',Md1
	  type*,'Md2=',Md2
	  type*,'Nd1=',Nd1
	  type*,'Nd2=',Nd2
	  STOP
	  END
		
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
		
		    
