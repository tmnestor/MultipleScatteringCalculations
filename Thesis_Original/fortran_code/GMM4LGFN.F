	  IMPLICIT NONE
	  INTEGER  NMAX,NP,MP,IL
      PARAMETER(NMAX=15,NP=15,MP=1)
	  INTEGER  IPIV(NMAX),INDXR(NMAX),INDXC(NMAX),I,J
	  COMPLEX  A(NP,NP),B(NP,MP),CI,C0,C1,EA1,EA2,EB2,EA3,EB3,EA4,EB4,rtea2,rteb2
      REAL RHO(5),ALPHA(5),BETA(5),P,W,H1,H2,H3,h4
      COMPLEX M4(2,2),N4(2,2),N2(2,2),M2(2,2),M3(2,2),N3(2,2),ETA(5),NETA(5)
      COMPLEX VERTICALSLO,T(2,2),W11,W12,W21,W22,EPSA,EPSB,RTEA,RTEB,M5(2,2),N5(2,2)
      DATA     RHO/1.0,1.0,1.0,1.20,1.35/
	  DATA     alpha/1.5,1.75,1.78,1.8,1.82/
	  DATA     beta/1.5,1.6,1.7,1.85,1.9/
	  DATA     W,h1,h2,h3,h4/1.0,0.0,1.0,1.0,1.0/
	  CI=(0.0,1.0)
	  C0=(0.0,0.0)
	  C1=(1.0,0.0)
	  Type*,'input p'
	  Accept*,p

	  ETA(1)=VERTICALSLO(ALPHA(1),P)
*	  TYPE*,'ETA(1)=',ETA(1)
*	  PAUSE
      DO il=2,5
	  ETA(il)=VERTICALSLO(ALPHA(il),P)
	  NETA(il)=VERTICALSLO(Beta(il),P)
      end do
      EA1=CEXP(CI*W*H1*ETA(1))
      EA2=CEXP(CI*W*H2*ETA(2))
	  EB2=CEXP(CI*W*H2*NETA(2))
      EA3=CEXP(CI*W*H3*ETA(3))
	  EB3=CEXP(CI*W*H3*NETA(3))
      EA4=CEXP(CI*W*H4*ETA(4))
	  EB4=CEXP(CI*W*H4*NETA(4))
	  RTEA2=CEXP(CI*W*H2*ETA(2)*0.5)
	  RTEB2=CEXP(CI*W*H2*NETA(2)*0.5)

*	  TYPE*,'EAEA1,EAEA2,EbEb2',EA1*EA1,EA2*EA2,Eb2*Eb2
*	  TYPE*,'EAEA3,EbEb3',EA3*EA3,Eb3*Eb3
*	  PAUSE
	  W12=CI*csqrt(ETA(1))/sqrt(2.0*rho(1))
*	  W12=CI*ETA(1)
	  W11=-W12
*	  W21=-RHO(1)
      W21=-sqrt(RHO(1)/2.0)/csqrt(ETA(1))
	  W22=W21
*	  type*,'w11,w21=',w11,w21
*	  pause
      CALL Emat(P,ETA(2),NETA(2),RHO(2),BETA(2),M2,N2)
      CALL Emat(P,ETA(3),NETA(3),RHO(3),BETA(3),M3,N3)
      CALL Emat(P,ETA(4),NETA(4),RHO(4),BETA(4),M4,N4)
      CALL Emat(P,ETA(5),NETA(5),RHO(5),BETA(5),M5,N5)
	  DO I=1,NP
	    DO J=1,MP
		  B(I,J)=C0
		END DO
		DO J=1,NP
	      A(I,J)=C0
	    END DO 
	  END DO
      A(1,1)=W21
	  A(2,1)=W11
      DO I=1,2
	    DO J=1,2
		  T(I,J)=-M2(I,J)
		END DO
      END DO
      CALL ASSEMBLE(A,15,15,T,2,2)
      CALL ASSEMBLE(A,15,15,M2,2,4)
      A(1,2)=-N2(1,1)
	  A(1,3)=-N2(1,2)
	  A(1,4)=-N2(1,1)
	  A(1,5)=-N2(1,2)
      T(1,1)=M2(1,1)/EA2
	  T(2,1)=M2(2,1)/EA2
	  T(1,2)=M2(1,2)/EB2
	  T(2,2)=M2(2,2)/EB2
      CALL ASSEMBLE(A,15,15,T,4,2)
      T(1,1)=-M2(1,1)*EA2
	  T(2,1)=-M2(2,1)*EA2
	  T(1,2)=-M2(1,2)*EB2
	  T(2,2)=-M2(2,2)*EB2
      CALL ASSEMBLE(A,15,15,T,4,4)
      T(1,1)=-M3(1,1)
	  T(2,1)=-M3(2,1)
	  T(1,2)=-M3(1,2)
	  T(2,2)=-M3(2,2)
      CALL ASSEMBLE(A,15,15,T,4,6)
      CALL ASSEMBLE(A,15,15,M3,4,8)
      T(1,1)=N2(1,1)/EA2
	  T(2,1)=N2(2,1)/EA2
	  T(1,2)=N2(1,2)/EB2
	  T(2,2)=N2(2,2)/EB2
      CALL ASSEMBLE(A,15,15,T,6,2)
      T(1,1)=N2(1,1)*EA2
	  T(2,1)=N2(2,1)*EA2
	  T(1,2)=N2(1,2)*EB2
	  T(2,2)=N2(2,2)*EB2
      CALL ASSEMBLE(A,15,15,T,6,4)
      T(1,1)=-N3(1,1)
	  T(2,1)=-N3(2,1)
	  T(1,2)=-N3(1,2)
	  T(2,2)=-N3(2,2)
      CALL ASSEMBLE(A,15,15,T,6,6)
      CALL ASSEMBLE(A,15,15,T,6,8)
********ACROSS THE SOURCE DEPTH***********************
      T(1,1)=M3(1,1)/EA3
	  T(2,1)=M3(2,1)/EA3
	  T(1,2)=M3(1,2)/EB3
	  T(2,2)=M3(2,2)/EB3
      CALL ASSEMBLE(A,15,15,T,8,6)
      T(1,1)=-M3(1,1)*EA3
	  T(2,1)=-M3(2,1)*EA3
	  T(1,2)=-M3(1,2)*EB3
	  T(2,2)=-M3(2,2)*EB3
      CALL ASSEMBLE(A,15,15,T,8,8)
      T(1,1)=N3(1,1)/EA3
	  T(2,1)=N3(2,1)/EA3
	  T(1,2)=N3(1,2)/EB3
	  T(2,2)=N3(2,2)/EB3
      CALL ASSEMBLE(A,15,15,T,10,6)
      T(1,1)=N3(1,1)*EA3
	  T(2,1)=N3(2,1)*EA3
	  T(1,2)=N3(1,2)*EB3
	  T(2,2)=N3(2,2)*EB3
      CALL ASSEMBLE(A,15,15,T,10,8)      
     
	  
	  T(1,1)=-N4(1,1)
	  T(2,1)=-N4(2,1)
	  T(1,2)=-N4(1,2)
	  T(2,2)=-N4(2,2)
      CALL ASSEMBLE(A,15,15,T,10,10)
      CALL ASSEMBLE(A,15,15,T,10,12)
      CALL ASSEMBLE(A,15,15,M4,8,12)
	  T(1,1)=-M4(1,1)
	  T(2,1)=-M4(2,1)
	  T(1,2)=-M4(1,2)
	  T(2,2)=-M4(2,2)
      CALL ASSEMBLE(A,15,15,T,8,10)
     
********ACROSS THE LAYER 4***********************
      T(1,1)=M4(1,1)/EA4
	  T(2,1)=M4(2,1)/EA4
	  T(1,2)=M4(1,2)/EB4
	  T(2,2)=M4(2,2)/EB4
      CALL ASSEMBLE(A,15,15,T,12,10)
      T(1,1)=-M4(1,1)*EA4
	  T(2,1)=-M4(2,1)*EA4
	  T(1,2)=-M4(1,2)*EB4
	  T(2,2)=-M4(2,2)*EB4
      CALL ASSEMBLE(A,15,15,T,12,12)
      T(1,1)=N4(1,1)/EA4
	  T(2,1)=N4(2,1)/EA4
	  T(1,2)=N4(1,2)/EB4
	  T(2,2)=N4(2,2)/EB4
      CALL ASSEMBLE(A,15,15,T,14,10)
      T(1,1)=N4(1,1)*EA4
	  T(2,1)=N4(2,1)*EA4
	  T(1,2)=N4(1,2)*EB4
	  T(2,2)=N4(2,2)*EB4
      CALL ASSEMBLE(A,15,15,T,14,12)      
     
	  
	  T(1,1)=-N5(1,1)
	  T(2,1)=-N5(2,1)
	  T(1,2)=-N5(1,2)
	  T(2,2)=-N5(2,2)
      CALL ASSEMBLE(A,15,15,T,14,14)
      CALL ASSEMBLE(A,15,15,M5,12,14)
	  
*     unit down going P at bottom of layer 1	  
      B(1,1)=-W22
	  B(2,1)=-W12
	  
*      B(9,1)=cmplx(1.0,0.0)  ! saltus of trz between layers 3&4
*      B(4,1)=cmplx(1.0,0.0)  ! saltus of Uz between layers 2&3
*     unit upgoing P wave at the top of layer 5	  
*	  B(12,1)=M5(1,1)
*	  B(13,1)=M5(2,1)
*	  B(14,1)=N5(1,1)
*	  B(15,1)=N5(2,1)
	  CALL CMPLXGJ(A,15,NP,B,1,MP,IPIV,INDXR,INDXC,NMAX)
      TYPE*,'SOLUTION='
      DO I=1,15
	    TYPE*,'B(',I,')=',B(I,1)
	  END DO
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

      subroutine Emat(P,ETA,NETA,RHO,BETA,M,N)
*     form the layer eigen arrays
      implicit none
      real    rho,twomup,beta,p
      complex eta,neta,N(2,2),M(2,2),ci,epsa,epsb,c1
	  Ci=(0.0,1.0)
	  C1=(1.0,0.0)
	  twomup=2*rho*beta*beta*p
*	  type*,'eta,neta=',eta,neta
	  epsa=C1/Csqrt(2.0*rho*eta)
	  epsb=C1/Csqrt(2.0*rho*neta)
*	  epsa=C1
*	  epsb=C1
*	  type*,'epsa,epsb=',epsa,epsb
*	  pause
	  M(1,1)=-Ci*eta*epsa
	  M(1,2)=p*epsb
	  M(2,1)=M(1,1)*twomup
	  M(2,2)=twomup*p-cmplx(rho,0.0)
	  N(1,1)=M(2,2)*epsa
	  M(2,2)=M(2,2)*epsb
	  N(2,2)=Ci*neta*epsb
	  N(2,1)=-p*epsa
	  N(1,2)=-N(2,2)*twomup
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
