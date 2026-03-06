      Program 	  fftprop
*
      Implicit    none
	  Integer 	  Nk,Nscatx,Nscatz,Nh,Nm,Nhm1,nscat,jskip
	  Parameter	  (Nk=4096,Nh=Nk/2,Nscatz=2,Nscatx=81,jskip=8)
	  Complex     SY(Nscatx,-2:2,2,Nscatz),PSY(Nscatx,-2:2,2,Nscatz)
	  Complex     PC(Nk,-2:2,2,Nscatz)
	  Complex     Mu1(Nk),SigPU(Nk),Md1(Nk),SigPD(Nk)
	  Complex     Mu2(Nk),SigSU(Nk),Md2(Nk),SigSD(Nk)
	  Complex     PU(Nk),SU(Nk),PD(Nk),SD(Nk)
	  Complex     Rpp(Nk),Rsp(Nk),Rss(Nk)	
	  Complex     W11(Nk),W12(Nk),W21(Nk),W22(Nk)
	  Complex     Eavec(Nk),Ebvec(Nk),rtEa(Nk),rtEb(Nk)
	  Complex 	  kxvec(Nk),kzavec(Nk),kzbvec(Nk)
	  Complex     Svec(Nscatx,-2:2,2,Nscatz),Rvec(Nscatx,-2:2,2,Nscatz)


	  Real    	  Pi,intfac,x,dkx,Kxmax,L,kx,dx,z
	  Parameter   (Pi=3.14159265358979324)
	  Real        rho,alpha,Q,ai,Xs,Xr
	  Integer     is,ir
	  Real        freq,InvN,signi,atten,lambda,Gmin,fac,zh
	  Complex     Ka0,Kb0,a0,ca0,cb0,w,twortqa,twortqb
	  Complex     R1,R12,R2,Rayleigh,p,p2,cb02,cb04,qa,qb
	  Complex     VERTICALSLO,C1,C0,Ci,im,temp,kz
	  Integer     iscat,jscat,ik,i,j,m

	  equivalence (Mu1(1),Md1(1))
	  equivalence (Mu2(1),Md2(1))
	  equivalence (SigPU(1),SigPD(1))
	  equivalence (SigSU(1),SigSD(1))
	  
	  data        alpha,rho/5.0,3.0/
	  data        Xs,Xr/10.0,0.0/
	  data        is,ir/1,1/
      data 		  Q,freq/50.0,2.0/
      
	  Ci=(0.0,1.0)
	  C0=(0.0,0.0)
	  C1=(1.0,0.0)

	  InvN=1.0/float(Nk)
	  Nm=Nk-1
	  Nhm1=Nh-1

	  ai=alpha/(2.0*Q)
	  a0=Cmplx(alpha,ai)
	  ca0=C1/a0
	  cb0=ca0/sqrt(3.0)
	  w=Cmplx(2.0*Pi*freq,.2)
	  ka0=w*a0
	  kb0=ka0*sqrt(3.0)
	  
	  type*,'w=',w
	  type*,'ka0=',ka0
	  type*,'kb0=',kb0
	  
*     set the  phase space dimensions

	  lambda=2*Pi/Abs(Ka0)
	  
	  type*,'lambda=',lambda
	  
************************************************************
	  kxmax=32.*Real(ka0)
	  dkx=2.*kxmax/float(Nk)
	  intfac=Sqrt(dkx/(2.0*Pi))
	  dx=Pi/kxmax
	  z=float(jskip)*dx
	  type*,'del=',z
*************************************************************	  
	  
	  
	  Type*,'kxmax=',kxmax
	  type*,'kxmax/real(ka0)=',kxmax/real(ka0)
	  
	  type*,'dx=',dx
	  
	  type*,'dkx=',dkx
	  
	  type*,'lambda/z=',lambda/z
      type*,'z/dx=',z/dx
	  type*,'lambda/dx=',lambda/dx

	  
	  type*,'G=',2.0*Pi/(z*Real(Ka0))
	  
	  Do iscat=1,Nscatz	  
	  Do jscat=1,Nscatx
	  Do m=-2,2
	    Svec(jscat,m,1,iscat)=C0
	    Svec(jscat,m,2,iscat)=C0
	    Rvec(jscat,m,1,iscat)=C0
	    Rvec(jscat,m,2,iscat)=C0
        PSY(jscat,m,1,iscat)=C0 !P
        PSY(jscat,m,2,iscat)=C0 !S
	  Enddo
	  Enddo
	  Enddo


	  Do iscat=1,Nscatz	  
	  Do jscat=1,Nscatx
	  Do m=-2,2
	    SY(jscat,m,1,iscat)=C0  !P
	    SY(jscat,m,2,iscat)=C0  !S
	  Enddo
	  Enddo
	  Enddo

*     Vertical Array Source vector Nscatz deep	  
*	    SY(1,-1,1,1)=C1  !top left corner
*	    SY(Nscatx,-2,2,1)=C1  !top right corner
	    SY(Nscatx,-2,1,Nscatz)=C1  !bottom right corner

	  
*     FFT ordered Kx
	  Do i=1,Nhm1
		Kxvec(i+1)=float(i)*dkx
		Kxvec(i+1+Nh)=float(i-Nh)*dkx
      Enddo
	  Kxvec(1)=0.0
	  Kxvec(Nh+1)=-Nh*dkx

      zh=0.5*z
	  Do ik=1,Nk
		kzavec(ik)=VerticalSlo(ka0,kxvec(ik))
		kzbvec(ik)=VerticalSlo(kb0,kxvec(ik))
		rtEa(ik)=Cexp(Ci*kzavec(ik)*Abs(zh))
		rtEb(ik)=Cexp(Ci*kzbvec(ik)*Abs(zh))
      Enddo
	  
*     set the free surface reflection array
	  cb02=cb0**2
	  cb04=cb02**2
	  Do ik=1,Nk
	    p=kxvec(ik)/w
		p2=p*p
		qa=kzavec(ik)/w
		qb=kzbvec(ik)/w
		twortqa=CSqrt(2.0*qa)
		twortqb=CSqrt(2.0*qb)
	    R1=(1.0 - 2*cb02*p2)
	    R12=R1**2
	    R2=4.0*cb04*p2*qa*qb
	    Rayleigh=(R12 + R2)
	    Rpp(ik)=(-R12 + R2)/Rayleigh
		Rsp(ik)=-4.0*cb04*p*R1*Csqrt(qa*qb)*Eavec(ik)*Ebvec(ik)/Rayleigh
		Rss(ik)=Ebvec(ik)*Rpp(ik)
		Rpp(ik)=Eavec(ik)*Rpp(ik)
		W11(ik)=-Ci*twortqa*R1/(Rayleigh*Sqrt(rho))
		W12(ik)=-2*Ci*cb02*twortqa*qb*p/(Rayleigh*Sqrt(rho))
		W21(ik)=-2*Ci*cb02*qa*twortqb*p/(Rayleigh*Sqrt(rho))
		W22(ik)=Ci*twortqb*R1/(Rayleigh*Sqrt(rho))
*		type*,'ik,W11(ik),W12(ik),W21(ik),W22(ik)=',ik,W11(ik),W12(ik),W21(ik),W22(ik)
	  enddo
  
*     set PW-Harmonic transformation arrays
      Do i=1,Nscatz
*P
	  Do ik=1,Nk
	      kx=Kxvec(ik)
		  kz=kzavec(ik)
		  PC(ik,0,1,i)=Csqrt(dkx/(Pi*kz))
		  temp=((kx/ka0)+Ci*(kz/ka0))
		  PC(ik,1,1,i)=PC(ik,0,1,i)*temp
	      PC(ik,2,1,i)=PC(ik,1,1,i)*temp
		  temp=((kx/ka0)-Ci*(kz/ka0))
		  PC(ik,-1,1,i)=PC(ik,0,1,i)*temp
	      PC(ik,-2,1,i)=PC(ik,-1,1,i)*temp
	  Enddo

*SV	  
	  Do ik=1,Nk
	      kx=Kxvec(ik)
		  kz=kzbvec(ik)
		  PC(ik,0,2,i)=Csqrt(dkx/(Pi*kz))
		  temp=((kx/kb0)+Ci*(kz/kb0))
		  PC(ik,1,2,i)=PC(ik,0,2,i)*temp
	      PC(ik,2,2,i)=PC(ik,1,2,i)*temp
		  temp=((kx/kb0)-Ci*(kz/kb0))
		  PC(ik,-1,2,i)=PC(ik,0,2,i)*temp
	      PC(ik,-2,2,i)=PC(ik,-1,2,i)*temp
	  Enddo
	  Enddo



	  
	  Do i=1,Nk
		Eavec(ik)=rtEa(ik)*rtEa(ik)
		Ebvec(ik)=rtEb(ik)*rtEb(ik)
	  Enddo

	  
*     *************************************************************
*      set the incident and scattered basis states svec and rvec
*     *************************************************************

      if(is.eq.1) then
	    Do ik=1,Nk
		  temp=Cexp(-Ci*kxvec(ik)*Xs)
          Pd(ik)=Csqrt(dkx/(Pi*kzavec(ik)))*rtea(ik)*W11(ik)*temp
          Sd(ik)=Csqrt(dkx/(Pi*kzbvec(ik)))*rteb(ik)*W21(ik)*temp
	    Enddo
	  else
	    Do ik=1,Nk
		  temp=Cexp(-Ci*kxvec(ik)*Xs)
          Pd(ik)=Csqrt(dkx/(Pi*kzavec(ik)))*rtea(ik)*W12(ik)*temp
          Sd(ik)=Csqrt(dkx/(Pi*kzbvec(ik)))*rteb(ik)*W22(ik)*temp
	    Enddo
	  endif

      Do 6000 iscat=1,Nscatz
	  
	    type*,'iscat=',iscat


        Do 700 m=-2,2

		  Do ik=1,Nk
            Md1(ik)=PC(ik,m,1,iscat)*PD(ik)
            Md2(ik)=PC(ik,m,2,iscat)*SD(ik)
		  Enddo
		  
		  signi=1.0
		  call FFT(Md1,Nk,signi) !k->x
		  call FFT(Md2,Nk,signi) !k->x
          
		  j=1
		  Do jscat=1,Nscatx	  
            Svec(jscat,m,1,iscat)=Svec(jscat,m,1,iscat)+Md1(j)
            Svec(jscat,m,2,iscat)=Svec(jscat,m,2,iscat)+Md2(j)
			j=j+jskip
		  Enddo
		  
700		continue
	  
		Do ik=1,Nk	  
		  PD(ik)=Eavec(ik)*PD(ik)
		  SD(ik)=Ebvec(ik)*SD(ik)
		Enddo


6000  continue  !end downsweep


      if(ir.eq.1) then
	    Do ik=1,Nk
		  temp=Cexp(-Ci*kxvec(ik)*Xr)
          Pd(ik)=Csqrt(dkx/(Pi*kzavec(ik)))*rtea(ik)*W11(ik)*temp
          Sd(ik)=Csqrt(dkx/(Pi*kzbvec(ik)))*rteb(ik)*W21(ik)*temp
	    Enddo
	  else
	    Do ik=1,Nk
		  temp=Cexp(-Ci*kxvec(ik)*Xr)
          Pd(ik)=Csqrt(dkx/(Pi*kzavec(ik)))*rtea(ik)*W12(ik)*temp
          Sd(ik)=Csqrt(dkx/(Pi*kzbvec(ik)))*rteb(ik)*W22(ik)*temp
	    Enddo
	  endif

      Do 8000 iscat=1,Nscatz
	  
	    type*,'iscat=',iscat


        Do 1400 m=-2,2

		  Do ik=1,Nk
            Md1(ik)=PC(ik,m,1,iscat)*PD(ik)
            Md2(ik)=PC(ik,m,2,iscat)*SD(ik)
		  Enddo
		  
		  signi=1.0
		  call FFT(Md1,Nk,signi) !k->x
		  call FFT(Md2,Nk,signi) !k->x
          
		  j=1
		  Do jscat=1,Nscatx	  
            Rvec(jscat,m,1,iscat)=Rvec(jscat,m,1,iscat)+Md1(j)
            Rvec(jscat,m,2,iscat)=Rvec(jscat,m,2,iscat)+Md2(j)
			j=j+jskip
		  Enddo
		  
1400		continue
	  
		Do ik=1,Nk	  
		  PD(ik)=Eavec(ik)*PD(ik)
		  SD(ik)=Ebvec(ik)*SD(ik)
		Enddo


8000  continue  !end downsweep
*     *************************************************************
*     *************************************************************
*     *************************************************************
*     *************************************************************


*     *************************************************************
*     begin up\down sweep
*     *************************************************************
	  
	  Do i=1,Nk
	    PU(i)=C0
	    SU(i)=C0
	  Enddo
	  
      Do 1000 iscat=Nscatz,1,-1
	  
	    type*,'iscat=',iscat

		Do ik=1,Nk	  
		  PU(ik)=Eavec(ik)*PU(ik)
		  SU(ik)=Ebvec(ik)*SU(ik)
		Enddo

        Do 50 m=-2,2
		
		  Do ik=1,Nk
            Mu1(ik)=PC(ik,-m,1,iscat)*PU(ik)
            Mu2(ik)=PC(ik,-m,2,iscat)*SU(ik)
		  Enddo
		  
		  signi=1.0
		  call FFT(Mu1,Nk,signi) !k->x
		  call FFT(Mu2,Nk,signi) !k->x
          
		  j=1
		  Do jscat=1,Nscatx	  
            PSY(jscat,m,1,iscat)=PSY(jscat,m,1,iscat)+Mu1(j)
            PSY(jscat,m,2,iscat)=PSY(jscat,m,2,iscat)+Mu2(j)
			j=j+jskip
		  Enddo
		  
50		continue
	  
		
		Do 60 m=-2,2
		
	      Do ik=1,Nk
		    SigPU(ik)=C0
		    SigSU(ik)=C0
		  Enddo

          j=1
		  Do jscat=1,Nscatx
	        SigPU(j)=SY(jscat,m,1,iscat)
	        SigSU(j)=SY(jscat,m,2,iscat)
	        j=j+jskip
		  Enddo
		
		  signi=-1.0
		  call FFT(SigPU,Nk,signi) !x->k
		  call FFT(SigSU,Nk,signi) !x->k
		  
		  Do ik=1,Nk	  
            PU(ik)=PU(ik)+PC(ik,m,1,iscat)*SigPU(ik)
            SU(ik)=SU(ik)+PC(ik,m,2,iscat)*SigSU(ik)
		  Enddo
		
60      continue		

1000  continue  !end depth loop


	  Do ik=1,Nk
	    PD(ik)=Rpp(ik)*PU(ik)-Rsp(ik)*SU(ik)
	    SD(ik)=Rsp(ik)*PU(ik)+Rss(ik)*SU(ik)
	  Enddo
	  
      Do 2000 iscat=1,Nscatz
	  
	    type*,'iscat=',iscat


        Do 70 m=-2,2

		  Do ik=1,Nk
            Md1(ik)=PC(ik,m,1,iscat)*PD(ik)
            Md2(ik)=PC(ik,m,2,iscat)*SD(ik)
		  Enddo
		  
		  signi=1.0
		  call FFT(Md1,Nk,signi) !k->x
		  call FFT(Md2,Nk,signi) !k->x
          
		  j=1
		  Do jscat=1,Nscatx	  
            PSY(jscat,m,1,iscat)=PSY(jscat,m,1,iscat)+Md1(j)
            PSY(jscat,m,2,iscat)=PSY(jscat,m,2,iscat)+Md2(j)
			j=j+jskip
		  Enddo
		  
70		continue
	  
		
		Do 80 m=-2,2
		
	      Do ik=1,Nk
		    SigPD(ik)=C0
		    SigSD(ik)=C0
		  Enddo

          j=1
		  Do jscat=1,Nscatx
	        SigPD(j)=SY(jscat,m,1,iscat)
	        SigSD(j)=SY(jscat,m,2,iscat)
	        j=j+jskip
		  Enddo
		
		  signi=-1.0
		  call FFT(SigPD,Nk,signi) !x->k
		  call FFT(SigSD,Nk,signi) !x->k
		  
		  Do ik=1,Nk	  
            PD(ik)=PD(ik)+PC(ik,-m,1,iscat)*SigPD(ik)
            SD(ik)=SD(ik)+PC(ik,-m,2,iscat)*SigSD(ik)
		  Enddo
		
80      continue

		Do ik=1,Nk	  
		  PD(ik)=Eavec(ik)*PD(ik)
		  SD(ik)=Ebvec(ik)*SD(ik)
		Enddo


2000  continue  !end downsweep

*      type*,'Top left corner PSY(1,2,1,1)=',PSY(1,2,1,1)
*       stop

      Do 5000 iscat=Nscatz,1,-1
	  
	    type*,'iscat=',iscat

	    Do i=1,Nk,2
	      PU(i)=C0
	      SU(i)=C0
	    Enddo


*       begin right sweep
		Do 3000 jscat=1,Nscatx
		  type*,'jscat=',jscat
		  Do ik=1,Nk,2	  
		    PU(ik)=Eavec(ik)*PU(ik)
		    SU(ik)=Ebvec(ik)*SU(ik)
		  Enddo

*k->x
        Do 90 m=-2,2
		  im=2.0*((Ci)**m)
		  Do ik=1,Nk,2
            PSY(jscat,m,1,iscat)=PSY(jscat,m,1,iscat)+im*PC(ik,-m,1,iscat)*PU(ik)
            PSY(jscat,m,2,iscat)=PSY(jscat,m,2,iscat)+im*PC(ik,-m,2,iscat)*SU(ik)
		  Enddo
90		continue
		
		Do 30 m=-2,2
		  im=(-Ci)**m
		  Do ik=1,Nk,2	  
            PU(ik)=PU(ik)+im*PC(ik,m,1,iscat)*SY(jscat,m,1,iscat)
            SU(ik)=SU(ik)+im*PC(ik,m,2,iscat)*SY(jscat,m,2,iscat)
		  Enddo
30      continue		

3000  continue  !end right sweep

	  Do ik=1,Nk
	    PD(ik)=C0
	    SD(ik)=C0
	  Enddo
      
*     begin left sweep
      Do 4000 jscat=Nscatx,1,-1
	  
	    type*,'jscat=',jscat


        Do 110 m=-2,2
		
          im=2.*((Ci)**m)
		  Do ik=1,Nk,2
            PSY(jscat,m,1,iscat)=PSY(jscat,m,1,iscat)+im*PC(ik,m,1,iscat)*PD(ik)
            PSY(jscat,m,2,iscat)=PSY(jscat,m,2,iscat)+im*PC(ik,m,2,iscat)*SD(ik)
		  Enddo
110		continue
	  
		
		Do 120 m=-2,2
          im=(-Ci)**m
		  Do ik=1,Nk,2	  
            PD(ik)=PD(ik)+im*PC(ik,-m,1,iscat)*SY(jscat,m,1,iscat)
            SD(ik)=SD(ik)+im*PC(ik,-m,2,iscat)*SY(jscat,m,2,iscat)
		  Enddo
120      continue

		Do ik=1,Nk,2	  
		  PD(ik)=Eavec(ik)*PD(ik)
		  SD(ik)=Ebvec(ik)*SD(ik)
		Enddo


4000  continue  !end leftsweep


5000  continue  !end depth loop


*	  type*,'PSY(1,2,2,1)=',PSY(1,2,2,1)
*	  type*,'PSY(Nscatx,2,2,1)=',PSY(Nscatx,2,2,1)

      j=0
	  Do iscat=1,Nscatz	  
	  Do jscat=1,Nscatx
	  Do i = 1,2
	  Do m=-2,2
	    j=j+1
*        type*,'PSY(jscat,m,i,iscat)=',j,PSY(jscat,m,i,iscat)
        type*,'Rvec(jscat,m,i,iscat)=',j,Rvec(jscat,m,i,iscat)
        type*,'Svec(jscat,m,i,iscat)=',j,Svec(jscat,m,i,iscat)
	  Enddo
 	  Enddo
	  Enddo
	  Enddo
      stop 'normal completion'
	  end

      complex  	function  fPC(Kx,Kz,dkx,Ka0,m)
      Implicit 	none
	  Integer  	m
      real 		dkx,kx,Pi
	  Parameter (Pi=     3.14159265358979324)
	  complex  	ka0,kz,Ci
	  Ci=(0.0,1.0)
      fPC=Csqrt(dkx/(Pi*kz))*((kx/ka0)+Ci*(kz/ka0))**m
      return
      end

	  COMPLEX FUNCTION VERTICALSLO(K,Kx)
	  REAL Kx
	  COMPLEX T,K
	  T=CMPLX((K+Kx)*(K-Kx))
	  VERTICALSLO=csqrt(T)
	  IF(AIMAG(VERTICALSLO) .LT. 0.0)THEN 
	    VERTICALSLO=-VERTICALSLO
	  END IF
	  RETURN
	  END

      SUBROUTINE FFT(CX,LX,SIGNI)
      COMPLEX    CX(LX),CTEMP,CW
	  Real       Pi
	  Parameter (Pi=3.14159265358979324)
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

	  real function Atten(t)
	  real   t
      If(t.eq.0.0) then
		Atten=1.0
	   else
*		Atten=60.*((Sin(t)/t)**6)/ (11.* Sin(t)**2+3.* Sin(3.*t)*Sin(t)+  
*     &     55.*Cos(t)**2 + 5.* Cos(3.*t)*Cos(t) )
*		Atten=((Sin(t)/t)**2)
 		Atten=3.0*((Sin(t)/t)**4)/(2.0+Cos(2.0*t))
	  endif
	  return
	  end
	  

