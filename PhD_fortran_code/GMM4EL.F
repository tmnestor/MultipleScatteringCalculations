

	  CI=(0.0,1.0)
	  C0=(0.0,0.0)
	  C1=(1.0,0.0)


      p2=p*p
	  p4=p2*p2
	  ETA(1)=VERTICALSLO(ALPHA(1),P)

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

*     Set Eigen Array for an Acoustic Ocean
	  W11=-CI*csqrt(ETA(1))/sqrt(2.0*rho(1))
	  W21=-sqrt(RHO(1)/2.0)/csqrt(ETA(1))
      W12=-W11
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
	  
*     SET RHS for a unit down going P at bottom of layer 1	  
      B(1,1)=-W12
	  B(2,1)=-W22
	  
	  
*     OCEAN BOTOM BALANCE
      A(1,1)=W11
	  A(2,1)=W21
	  A(1,2)=-M2(1,1)
	  A(1,3)=-M2(1,2)
	  A(1,4)=M2(1,1)*ea2
	  A(1,5)=-M2(1,2)*eb2
      T(1,1)=-N2(1,1)
	  T(2,1)=-N2(2,1)
	  T(1,2)=-N2(1,2)
	  T(2,2)=-N2(2,2)
      CALL ASSEMBLE(A,15,15,T,2,2)
      T(1,1)=-N2(1,1)*ea2
	  T(2,1)=N2(2,1)*ea2
	  T(1,2)=N2(1,2)*eb2
	  T(2,2)=-N2(2,2)*eb2
      CALL ASSEMBLE(A,15,15,T,2,4)
	  
*     Interface 2 BALANCE	  
      T(1,1)=M2(1,1)*ea2
	  T(2,1)=M2(2,1)*ea2
	  T(1,2)=M2(1,2)*eb2
	  T(2,2)=M2(2,2)*eb2
      CALL ASSEMBLE(A,15,15,T,4,2)
      T(1,1)=N2(1,1)*ea2
	  T(2,1)=N2(2,1)*ea2
	  T(1,2)=N2(1,2)*eb2
	  T(2,2)=N2(2,2)*eb2
      CALL ASSEMBLE(A,15,15,T,6,2)
      T(1,1)=-M2(1,1)
	  T(2,1)=M2(2,1)
	  T(1,2)=M2(1,2)
	  T(2,2)=-M2(2,2)
      CALL ASSEMBLE(A,15,15,T,4,4)
      T(1,1)=N2(1,1)
	  T(2,1)=-N2(2,1)
	  T(1,2)=-N2(1,2)
	  T(2,2)=N2(2,2)
      CALL ASSEMBLE(A,15,15,T,6,4)
      T(1,1)=-M3(1,1)
	  T(2,1)=-M3(2,1)
	  T(1,2)=-M3(1,2)
	  T(2,2)=-M3(2,2)
      CALL ASSEMBLE(A,15,15,T,4,6)
      T(1,1)=-N3(1,1)
	  T(2,1)=-N3(2,1)
	  T(1,2)=-N3(1,2)
	  T(2,2)=-N3(2,2)
      CALL ASSEMBLE(A,15,15,T,6,6)
      T(1,1)=M3(1,1)*ea3
	  T(2,1)=-M3(2,1)*ea3
	  T(1,2)=-M3(1,2)*eb3
	  T(2,2)=M3(2,2)*eb3
      CALL ASSEMBLE(A,15,15,T,4,8)
      T(1,1)=-N3(1,1)*ea3
	  T(2,1)=N3(2,1)*ea3
	  T(1,2)=N3(1,2)*eb3
	  T(2,2)=-N3(2,2)*eb3
      CALL ASSEMBLE(A,15,15,T,6,8)

*     Interface 3 BALANCE	  
      T(1,1)=M3(1,1)*ea3
	  T(2,1)=M3(2,1)*ea3
	  T(1,2)=M3(1,2)*eb3
	  T(2,2)=M3(2,2)*eb3
      CALL ASSEMBLE(A,15,15,T,8,6)
      T(1,1)=N3(1,1)*ea3
	  T(2,1)=N3(2,1)*ea3
	  T(1,2)=N3(1,2)*eb3
	  T(2,2)=N3(2,2)*eb3
      CALL ASSEMBLE(A,15,15,T,10,6)
      T(1,1)=-M3(1,1)
	  T(2,1)=M3(2,1)
	  T(1,2)=M3(1,2)
	  T(2,2)=-M3(2,2)
      CALL ASSEMBLE(A,15,15,T,8,8)
      T(1,1)=N3(1,1)
	  T(2,1)=-N3(2,1)
	  T(1,2)=-N3(1,2)
	  T(2,2)=N3(2,2)
      CALL ASSEMBLE(A,15,15,T,10,8)
      T(1,1)=-M4(1,1)
	  T(2,1)=-M4(2,1)
	  T(1,2)=-M4(1,2)
	  T(2,2)=-M4(2,2)
      CALL ASSEMBLE(A,15,15,T,8,10)
      T(1,1)=-N4(1,1)
	  T(2,1)=-N4(2,1)
	  T(1,2)=-N4(1,2)
	  T(2,2)=-N4(2,2)
      CALL ASSEMBLE(A,15,15,T,10,10)
      T(1,1)=M4(1,1)*ea4
	  T(2,1)=-M4(2,1)*ea4
	  T(1,2)=-M4(1,2)*eb4
	  T(2,2)=M4(2,2)*eb4
      CALL ASSEMBLE(A,15,15,T,8,12)
      T(1,1)=-N4(1,1)*ea4
	  T(2,1)=N4(2,1)*ea4
	  T(1,2)=N4(1,2)*eb4
	  T(2,2)=-N4(2,2)*eb4
      CALL ASSEMBLE(A,15,15,T,10,12)

*     Interface 4 BALANCE	  
      T(1,1)=M4(1,1)*ea4
	  T(2,1)=M4(2,1)*ea4
	  T(1,2)=M4(1,2)*eb4
	  T(2,2)=M4(2,2)*eb4
      CALL ASSEMBLE(A,15,15,T,12,10)
      T(1,1)=N4(1,1)*ea4
	  T(2,1)=N4(2,1)*ea4
	  T(1,2)=N4(1,2)*eb4
	  T(2,2)=N4(2,2)*eb4
      CALL ASSEMBLE(A,15,15,T,14,10)
      T(1,1)=-M4(1,1)
	  T(2,1)=M4(2,1)
	  T(1,2)=M4(1,2)
	  T(2,2)=-M4(2,2)
      CALL ASSEMBLE(A,15,15,T,12,12)
      T(1,1)=N4(1,1)
	  T(2,1)=-N4(2,1)
	  T(1,2)=-N4(1,2)
	  T(2,2)=N4(2,2)
      CALL ASSEMBLE(A,15,15,T,14,12)
      T(1,1)=-M5(1,1)
	  T(2,1)=-M5(2,1)
	  T(1,2)=-M5(1,2)
	  T(2,2)=-M5(2,2)
      CALL ASSEMBLE(A,15,15,T,12,14)
      T(1,1)=-N5(1,1)
	  T(2,1)=-N5(2,1)
	  T(1,2)=-N5(1,2)
	  T(2,2)=-N5(2,2)
      CALL ASSEMBLE(A,15,15,T,14,14)

	  
	  


	  CALL CMPLXGJ(A,15,NP,B,1,MP,IPIV,INDXR,INDXC,NMAX)
      TYPE*,'SOLUTION='
      DO I=1,15
	    TYPE*,'B(',I,')=',B(I,1)
		X(i)=B(i,1)
	  END DO

	  STOP
	  END





      SUBROUTiNE ASSEMBLE(A,NP,MP,T,i,J)
      iMPLiCiT NONE
      iNTEGER NP,MP,i,J,iP1,JP1
	  COMPLEX A(NP,MP),T(2,2)
      iP1=i+1
	  JP1=J+1
      iF(iP1 .GT. NP) PAUSE 'iP1 OUT OF BOUNDS'
      iF(JP1 .GT. MP) PAUSE 'JP1 OUT OF BOUNDS'
	  A(i,J)=T(1,1)
	  A(i,JP1)=T(1,2)
	  A(iP1,J)=T(2,1)
	  A(iP1,JP1)=T(2,2)
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
	  epsa=C1/Csqrt(2.0*rho*eta)
	  epsb=C1/Csqrt(2.0*rho*neta)
	  M(1,1)=Ci*eta*epsa
	  M(1,2)=p*epsb
	  M(2,1)=p*epsa
	  M(2,2)=Ci*neta*epsb
	  N(1,1)=(twomup*p-rho)*epsa
	  N(2,2)=(twomup*p-rho)*epsb
	  N(2,1)=Ci*twomup*eta*epsa
	  N(1,2)=Ci*twomup*neta*epsb
	  return
	  end

	  COMPLEX FUNCTiON VERTiCALSLO(C,P)
	  REAL C,P,oneoverc,cmin
	  parameter(cmin=1.0E-6)
	  COMPLEX T
	  if(abs(c).le.cmin)then
	    PAUSE 'propagation velocity too small'
	  else
	    oneoverc=1.0/c
	  end if
	  T=CMPLX((oneoverC+P)*(oneoverC-P),0.0)
	  VERTiCALSLO=csqrt(T)
	  iF(AiMAG(VERTiCALSLO) .LT. 0.0)THEN 
	    VERTiCALSLO=-VERTiCALSLO
	  END iF
	  RETURN
	  END
      
      SUBROUTiNE CMPLXGJ(A,N,NP,B,M,MP,iPiV,iNDXR,iNDXC,NMAX)	
	  iMPLiCiT NONE
	  iNTEGER  NMAX,NP,N,MP,M,iPiV(NMAX),iNDXR(NMAX),iNDXC(NMAX)
	  iNTEGER  iROW,iCOL,i,J,K,L,LL
	  REAL     BiG,EPSMiN
	  COMPLEX  A(NP,NP),B(NP,MP),DUM,C1,PiViNV
	  PARAMETER (C1=(1.0,0.0),EPSMiN=1.0E-04)
	  DO J=1,N
	    iPiV(J)=0
	  END DO
	  DO i=1,N
	    BiG=0.0
		DO J=1,N
		  iF(iPiV(J) .NE. 1)THEN
		    DO K=1,N
			  iF(iPiV(K) .EQ. 0)THEN
			    iF(CABS(A(J,K)).GE.BiG)THEN
				  BiG=CABS(A(J,K))
				  iROW=J
				  iCOL=K
				END iF
			  ELSE iF (iPiV(K) .GT. 1) THEN
			    PAUSE 'MATRiX SiNGULAR'
			  END iF
			END DO
		  END iF
		END DO
		iPiV(iCOL)=iPiV(iCOL)+1
		iF(iROW .NE. iCOL)THEN
		  DO L=1,N
		    DUM=A(iROW,L)
			A(iROW,L)=A(iCOL,L)
			A(iCOL,L)=DUM
		  END DO
		  DO L=1,M
		    DUM=B(iROW,L)
			B(iROW,L)=B(iCOL,L)
			B(iCOL,L)=DUM
		  END DO
		END iF
		iNDXR(i)=iROW
		iNDXC(i)=iCOL
		iF(CABS(A(iCOL,iCOL)) .LT. EPSMiN) PAUSE 'SiNGULAR MATRiX'
		PiViNV=C1/A(iCOL,iCOL)
		A(iCOL,iCOL)=C1
		DO L=1,N
		  A(iCOL,L)=A(iCOL,L)*PiViNV
		END DO
		DO L=1,M
		  B(iCOL,L)=B(iCOL,L)*PiViNV
		END DO
		DO LL=1,N
		  iF(LL .NE. iCOL)THEN
		    DUM=A(LL,iCOL)
			A(LL,iCOL)=(0.0,0.0)
			DO L=1,N
			  A(LL,L)=A(LL,L)-A(iCOL,L)*DUM
			END DO
			DO L=1,M
			  B(LL,L)=B(LL,L)-B(iCOL,L)*DUM
			END DO
		  ENDiF
		END DO
	  END DO
	  DO L=N,1,-1
	    iF(iNDXR(L).NE.iNDXC(L))THEN
		  DO K=1,N
		    DUM=A(K,iNDXR(L))
			A(K,iNDXR(L))=A(K,iNDXC(L))
			A(K,iNDXC(L))=DUM
		  END DO
		END IF
	  END DO
	RETURN
	END

      
