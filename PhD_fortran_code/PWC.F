      program   PwC
*****************************************************************     
	  Implicit  none
      Integer   Nl
      Parameter (Nl=1)

      Integer 	Lmax,DL,Nscat,Nscatx,Nscaty,BLKS
      Integer	Tdim,GTdim,iscatx,iscaty,iscat
      Parameter	(Lmax=2,DL=2*Lmax,BLKS=4*Lmax+2,Nscatx=6,Nscaty=6)
      Parameter	(Nscat=Nscatx*Nscaty,Tdim=Nscat*BLKS,GTdim=Nl*Tdim)
      Real*8	H,Xmin,Ymin,D,radius,Xc,Yc
      Parameter	(d=1.D0,radius=0.499999D0*d,H=Nscaty*d)
      Parameter (Xmin=0.D0,Ymin=0.5D0*d)

      Real*8 	 w,rho0(Nl),alpha0(Nl),beta0(Nl)
      Real*8     rho1(Nl),alpha1(Nl),beta1(Nl)

      Integer    Np,Np4
      parameter  (Np=2,Np4=4*Np)
      Real*8     pvec(Np)
      Real*8	 XcYc(Nscat,2,Nl)
      Complex*16 Smat(Np4,Np4)
****************************************************************
      Integer    IPIV(Tdim)
      Complex*16 PCoutS(Np4,Tdim,Nl)
      Complex*16 PCin(Tdim,Np4,Nl),PCout(Np4,Tdim)
****************************************************************
	  
      Complex*16 J0a(0:Lmax),H0a(0:Lmax)
      Complex*16 J1a(0:Lmax)
      Complex*16 Haij(0:DL)
	  
      Complex*16 J0b(0:Lmax),H0b(0:Lmax)
      Complex*16 J1b(0:Lmax)
      Complex*16 Hbij(0:DL)	  
	  
      Complex*16 InvT(Tdim,Tdim),Tmat(2,2,-Lmax:Lmax,Nscat)
      Complex*16 SPath(Tdim,Tdim)

      Integer    ipo,ipi,il

****************************************************************      
      data pvec/-0.1D0,0.1D0/

      data w/2.D0/
      data rho0,alpha0,beta0/1.D0,2.D0,1.D0/
      data rho1,alpha1,beta1/1.5D0,2.8D0,1.2D0/

*     set the coordinates of the scattering centres
*     assumed arranged in a lattice with all x coordinates 
*     assinged for each fixed the y coordinate (i.e. stored by rows)
      Do il=1,Nl
      iscat=0
      Do iscaty=1,Nscaty
	Yc=ymin+(iscaty-1)*D
	Do iscatx=1,Nscatx
	  Xc=xmin+(iscatx-1)*D
	  iscat=iscat+1
	  XcYc(iscat,1,il)=Xc
	  XcYc(iscat,2,il)=Yc
          type*,'iscat,XcYc(iscat,1),XcYc(iscat,2)=',iscat
          type*,XcYc(iscat,1,il),XcYc(iscat,2,il)
          type*
	Enddo
      Enddo


      call  setPwC(Np,Np4,BLKS,Tdim,Nscat,Lmax,DL,w,XcYc(1,1,il),H,
     &      radius,pvec,rho0,alpha0,beta0,rho1,alpha1,beta1,
     &      J0a,H0a,J1a,Haij,J0b,H0b,J1b,Hbij,IPIV,PCout,
     &      SPath,InvT,Tmat,PCoutS(1,1,il),PCin(1,1,il))

*
*     SMAT=PCoutS*PCin
*
      call ZGEMM('N','N',Np4,Np4,Tdim,(1.D0,0.D0),PCoutS,Np4,
     &          PCin,Tdim,(0.D0,0.D0),SMAT,Np4)


      Do ipo=1,Np4
        Do ipi=1,Np4
          Type*,'ipo,ipi,Smat(ipo,ipi)=',ipo,ipi
          Type*,Smat(ipo,ipi)
          Type*
        Enddo
      Enddo

      Enddo !il
*     layer loop

      stop
      END



      Subroutine setPwC(Np,Np4,BLKS,Tdim,Nscat,Lmax,DL,w,XcYc,H,
     &      radius,pvec,rho0,alpha0,beta0,rho1,alpha1,beta1,
     &      J0a,H0a,J1a,Haij,J0b,H0b,J1b,Hbij,IPIV,PCout,SPath,
     &      InvT,Tmat,PCoutS,PCin)
****************************************************************
*     set the plane wave Smat for a collection of cylindrical
*     scatterers
****************************************************************
      Implicit 	 none
      Integer    Np,Np4,Tdim

      Real*8 	 w,H,rho0,alpha0,beta0,rho1,alpha1,beta1
      Real*8     pvec(Np),XcYc(Nscat,2),radius

****************************************************************
      Integer    IPIV(Tdim)
      Complex*16 PCoutS(Np4,Tdim)
      Complex*16 PCin(Tdim,Np4),PCout(Np4,Tdim)
****************************************************************
      Integer 	 Lmax,DL,Nscat,BLKS,i
	  
      Complex*16 J0a(0:Lmax),H0a(0:Lmax)
      Complex*16 J1a(0:Lmax)
      Complex*16 Haij(0:DL)
	  
      Complex*16 J0b(0:Lmax),H0b(0:Lmax)
      Complex*16 J1b(0:Lmax)
      Complex*16 Hbij(0:DL)
	  
	  
      Complex*16 InvT(Tdim,Tdim),Tmat(2,2,-Lmax:Lmax,Nscat)
      Complex*16 SPath(Tdim,Tdim)
      integer    ioff,iscati,mi,i,j
****************************************************************
      Integer    info
      Complex*16 alf,bet
****************************************************************


*     set isolated T matrices
      call SetT(Tdim,Nscat,J0a,H0a,J1a,J0b,H0b,J1b,
     &          rho0,alpha0,beta0,rho1,alpha1,beta1,w,radius,
     &          Lmax,DL,BLKS,XcYc,Tmat)

      Do 20 i=1,Tdim
	Do 10 j=1,Tdim
          Spath(i,j)=(0.D0,0.D0)
10	Continue
20    Continue

*     Spath=T
      Do 40 iscati=1,Nscat
	Do 30 mi=-Lmax,Lmax
          ioff=BLKS*(iscati-1)+2*(Lmax+mi)
          SPath(ioff+1,ioff+1)=Tmat(1,1,mi,iscati)
          SPath(ioff+1,ioff+2)=Tmat(1,2,mi,iscati)
          SPath(ioff+2,ioff+1)=Tmat(2,1,mi,iscati)
          SPath(ioff+2,ioff+2)=Tmat(2,2,mi,iscati)
30	Continue
40    Continue

*      Do i=1,Tdim
*        Do j=1,Tdim
*          Type*,'i,j,T(i,j)=',i,j
*          TYpe*,Spath(i,j)
*          Type*
*        Enddo
*      Enddo
      If(nscat.gt.1) then
*       set  (1-T*P)
        call SetInvT(Tdim,Nscat,Haij,Hbij,rho0,alpha0,
     &  beta0,rho1,alpha1,beta1,w,radius,Lmax,DL,BLKS,
     &  XcYc,Tmat,InvT)

        open(unit=9,file='T.m',status='unknown')

        write(9,'(A4)') 'tr=['
        Do j=1,Tdim
          Do i=1,Tdim
             write(9,'(f9.6)')  DREAL(InvT(i,j))
          Enddo
        Enddo
        write(9,'(A2)')'];'

        write(9,'(A4)') 'ti=['
        Do j=1,Tdim
          Do i=1,Tdim
             write(9,'(f9.6)')  DImag(InvT(i,j))
          Enddo
        Enddo
        write(9,'(A2)')'];'
        write(9,'(A)') 'A=reshape(tr + i*ti,sqrt(length(tr)),
     &                  sqrt(length(tr)));'
        close(9)

*       set  SPath=(1-T*P)^(-1)*Spath
        CALL ZGETRF(Tdim,Tdim,InvT,Tdim,IPIV,INFO)
        if(info.ne.0) then
             print*,'InvT is singular'
             stop
        endif
        CALL ZGETRS('N',Tdim,Tdim,InvT,Tdim,IPIV,SPath,
     &         Tdim,INFO)

*        Do i=1,Tdim
*          Do j=1,Tdim
*            Type*,'i,j,SPath(i,j)=',i,j
*            TYpe*,Spath(i,j)
*            Type*
*          Enddo
*        Enddo
      Endif
      call PwCyl(alpha0,beta0,Nscat,XcYc,
     &     H,Np,Tdim,w,Lmax,Pvec,Spath,PCin,PCout)

*      Do ipo=1,Tdim
*        Do ipi=1,4*Np
*          Type*,'i,j,PCin(i,j)=',ipo,ipi
*          Type*,PCin(ipo,ipi)
*          Type*
*        Enddo
*      Enddo

*       Do ipo=1,Tdim
*       Do ipi=1,4*Np
*          Type*,'i,j,PCout(i,j)=',ipo,ipi
*          Type*,PCout(ipo,ipi)
*          Type*
*        Enddo
*      Enddo

*
*         PCoutS=PCout*SPATH
*
          alf=(1.D0,0.D0)
          bet=(0.D0,0.D0)
          call ZGEMM('N','N',Np4,Tdim,Tdim,alf,PCout,Np4,
     &          SPATH,Tdim,bet,PCoutS,Np4)

      return
      end

      Subroutine SetT(Tdim,Nscat,J0a,H0a,J1a,J0b,H0b,J1b,
     &          rho0,alpha0,beta0,rho1,alpha1,beta1,w,radius,
     &          Lmax,DL,BLKS,XcYc,Tmat)

      Implicit 	none
      Integer	Tdim,Lmax,Nscat,BLKS,DL
      Real*8 	rho0,alpha0,beta0,rho1,alpha1
      Real*8    beta1,w,radius,fl,lsq,wsq
      Real*8    Ka0,Xa0,Ka1,Xa1,Kb0,Xb0,Kb1,Xb1

      Complex*16 Tmat(2,2,-Lmax:Lmax,Nscat)
	  
      Complex*16 J0a(0:Lmax)
      Complex*16 J1a(0:Lmax)
      Complex*16 H0a(0:Lmax)
	  
      Complex*16 J0b(0:Lmax)
      Complex*16 J1b(0:Lmax)
      Complex*16 H0b(0:Lmax)
	  
      Complex*16  Ci,C1
      Real*8	  temp,Pi	
      Parameter   (Pi=3.14159265358979324D0)
      Real*8	  XcYc(Nscat,2)
      Integer	  i,l,lm1
      Complex*16  MJ011,MJ012,MJ021,MJ022
      Complex*16  MH011,MH012,MH021,MH022
      Complex*16  MJ111,MJ112,MJ121,MJ122
      Complex*16  NJ011,NJ012,NJ021,NJ022
      Complex*16  NH011,NH012,NH021,NH022
      Complex*16  NJ111,NJ112,NJ121,NJ122
      Complex*16  INVRE11,INVRE12,INVRE21,INVRE22
      Complex*16  INVOu11,INVOu12,INVOu21,INVOu22,InvDet
      Real*8      rsqm2,mu0,mu1

********************************************************
      integer     Nz,ifail
      EXTERNAL 	  s17def,s17dlf	
********************************************************
      
      wsq=w*w
*     set the waveno. of host
      Ka0=w/alpha0
      Kb0=w/beta0

      Ci=(0.D0,1.D0)
      C1=(1.D0,0.D0)
	  
      rsqm2=1.D0/radius/radius
      Temp=DSqrt(Pi*radius/(2.D0))/w

      mu0=rho0*beta0*beta0
      mu1=rho1*beta1*beta1
      Xa0=Ka0*radius
      Xb0=Kb0*radius

      Ka1=w/alpha1
      Kb1=w/beta1
      Xa1=ka1*radius
      Xb1=kb1*radius


*     set exterior eigenfunctions of the scatterer
      CALL s17def (0.D0, Dcmplx(Xa0), Lmax+1,'u', 
     &       J0a(0),Nz,ifail)	  	  
      CALL s17def (0.D0, Dcmplx(Xb0), Lmax+1,'u', 
     &       J0b(0),Nz,ifail)
      CALL s17dlf (1,0.D0, Dcmplx(Xa0), Lmax+1,'u', 
     &       H0a(0),Nz,ifail)	  	  
      CALL s17dlf (1,0.D0, Dcmplx(Xb0), Lmax+1,'u', 
     &       H0b(0),Nz,ifail)
	  	  

*     set interior eigenfunctions of the scatterer
      CALL s17def (0.D0, Dcmplx(Xa1), Lmax+1,'u', 
     &       J1a(0),Nz,ifail)	  	  
      CALL s17def (0.D0, Dcmplx(Xb1), Lmax+1,'u', 
     &       J1b(0),Nz,ifail)

********************************************************
*     Loop over the Scatteres
*******************************************************
      Do 1000 i=1,Nscat

	Do 999 L=1,Lmax
	  lm1=l-1
          fL=float(L)
	  lsq=fl*fl
*         set the Tmatrix


*         Regular array for host 
*         reg P-wave
          MJ011=Temp*(ka0*J0a(lm1)-fl*J0a(l)/radius)
          MJ021=-Ci*Temp*fl*J0a(l)/radius
          NJ011=2.D0*Temp*mu0*rsqm2*(((lsq+l)-0.5D0*Xb0*Xb0)*J0a(l)-
     &          Xa0*J0a(lm1))
	  NJ021=-Ci*2.D0*Temp*mu0*rsqm2*(Xa0*J0a(lm1)-(fl+1)*J0a(l))

*         reg S-wave
	  MJ012=-Ci*Temp*fl*J0b(l)/radius
	  MJ022=Temp*(fl*J0b(l)/radius-kb0*J0b(lm1))
	  NJ012=-Ci*2.D0*Temp*mu0*rsqm2*(Xb0*J0b(lm1)-(fl+1)*J0b(l))
	  NJ022=2.D0*Temp*mu0*rsqm2*((0.5D0*Xb0*Xb0-(lsq+l))*J0b(l)+
     &          Xb0*J0b(lm1))


*         Irregular array for host
*         irreg. P-wave
          MH011=Temp*(ka0*H0a(lm1)-fl*H0a(l)/radius)
          MH021=-Ci*Temp*fl*H0a(l)/radius
          NH011=2.D0*Temp*mu0*rsqm2*(((lsq+l)-0.5D0*Xb0*Xb0)*H0a(l)-
     &          Xa0*H0a(lm1))
	  NH021=-Ci*2.D0*Temp*mu0*rsqm2*(Xa0*H0a(lm1)-(fl+1)*H0a(l))

*         irreg. S-wave
	  MH012=-Ci*Temp*fl*H0b(l)/radius
	  MH022=Temp*(fl*H0b(l)/radius-kb0*H0b(lm1))
	  NH012=-Ci*2.D0*Temp*mu0*rsqm2*(Xb0*H0b(lm1)-(fl+1)*H0b(l))
	  NH022=2.D0*Temp*mu0*rsqm2*((0.5D0*Xb0*Xb0-(lsq+l))*H0b(l)+
     &          Xb0*H0b(lm1))


*         Regular array for scatterer
*         reg P-wave
          MJ111=Temp*(ka1*J1a(lm1)-fl*J1a(l)/radius)
          MJ121=Ci*Temp*fl*J1a(l)/radius
          NJ111=2.D0*Temp*mu1*rsqm2*(((lsq+l)-0.5D0*Xb1*Xb1)*J1a(l)-
     &          Xa1*J1a(lm1))
	  NJ121=Ci*2.D0*Temp*mu1*rsqm2*(Xa1*J1a(lm1)-(fl+1)*J1a(l))

*         reg S-wave
	  MJ112=Ci*Temp*fl*J1b(l)/radius
	  MJ122=Temp*(fl*J1b(l)/radius-kb1*J1b(lm1))
	  NJ112=Ci*2.D0*Temp*mu1*rsqm2*(Xb1*J1b(lm1)-(fl+1)*J1b(l))
	  NJ122=2.D0*Temp*mu1*rsqm2*((0.5D0*Xb1*Xb1-(lsq+l))*J1b(l)+
     &          Xb1*J1b(lm1))



      INVRE11=-NJ021*MJ121-NJ011*MJ111+MJ021*NJ121+MJ011*NJ111
	  INVRE12=-NJ021*MJ122-NJ011*MJ112+MJ021*NJ122+MJ011*NJ112
	  INVRE21=-NJ022*MJ121-NJ012*MJ111+MJ022*NJ121+MJ012*NJ111
	  INVRE22=-NJ022*MJ122-NJ012*MJ112+MJ022*NJ122+MJ012*NJ112
		
	  INVOU11=-NH021*MJ121-NH011*MJ111+MH021*NJ121+MH011*NJ111
	  INVOU12=-NH021*MJ122-NH011*MJ112+MH021*NJ122+MH011*NJ112
	  INVOU21=-NH022*MJ121-NH012*MJ111+MH022*NJ121+MH012*NJ111
	  INVOU22=-NH022*MJ122-NH012*MJ112+MH022*NJ122+MH012*NJ112
		
	  INVDET=C1/(INVOU11*INVOU22-INVOU21*INVOU12)
		
	  Tmat(1,1,l,i)=-INVDET*(INVRE11*INVOU22-INVRE12*INVOU21)
	  Tmat(1,2,l,i)=-INVDET*(-INVRE11*INVOU12+INVRE12*INVOU11)
	  Tmat(2,1,l,i)=-INVDET*(INVRE21*INVOU22-INVRE22*INVOU21)
	  Tmat(2,2,l,i)=-INVDET*(-INVRE21*INVOU12+INVRE22*INVOU11)

          Tmat(1,1,-l,i)=Tmat(1,1,l,i)
          Tmat(1,2,-l,i)=-Tmat(1,2,l,i)
          Tmat(2,1,-l,i)=-Tmat(2,1,l,i)
          Tmat(2,2,-l,i)=Tmat(2,2,l,i)

999     continue 

*       l=0 case
        l=0
*       reg P-wave
        MJ011=-Temp*ka0*J0a(1)
        NJ011=2.D0*Temp*mu0*rsqm2*(-0.5D0*Xb0*Xb0*J0a(0)+
     &          Xa0*J0a(1))

*       reg S-wave
	MJ022=Temp*kb0*J0b(1)
	NJ022=2.D0*Temp*mu0*rsqm2*(0.5D0*Xb0*Xb0*J0b(0)-
     &          Xb0*J0b(1))

*       irreg P-wave
        MH011=-Temp*ka0*H0a(1)
        NH011=2.D0*Temp*mu0*rsqm2*(-0.5D0*Xb0*Xb0*H0a(0)+
     &          Xa0*H0a(1))

*       irreg S-wave
	MH022=Temp*kb0*H0b(1)
	NH022=2.D0*Temp*mu0*rsqm2*(0.5D0*Xb0*Xb0*H0b(0)-
     &          Xb0*H0b(1))

*       reg P-wave in scatterer
        MJ111=-Temp*ka1*J1a(1)
        NJ111=2.D0*Temp*mu1*rsqm2*(-0.5D0*Xb1*Xb1*J1a(0)+
     &          Xa1*J1a(1))

*       reg S-wave
	MJ122=Temp*kb1*J1b(1)
	NJ122=2.D0*Temp*mu1*rsqm2*(0.5D0*Xb1*Xb1*J1b(0)-
     &          Xb1*J1b(1))



        INVRE11=-NJ011*MJ111+MJ011*NJ111
	INVRE22=-NJ022*MJ122+MJ022*NJ122
		
	INVOU11=-NH011*MJ111+MH011*NJ111
	INVOU22=-NH022*MJ122+MH022*NJ122
		
	INVDET=C1/(INVOU11*INVOU22)
		
	Tmat(1,1,0,i)=-INVDET*(INVRE11*INVOU22)
	Tmat(1,2,0,i)=(0.D0,0.D0)
	Tmat(2,1,0,i)=(0.D0,0.D0)
	Tmat(2,2,0,i)=-INVDET*(INVRE22*INVOU11)
	type*,'Tmat(1,1,0,i)=',i,Tmat(1,1,0,i)
	type*,'Tmat(2,2,0,i)=',i,Tmat(2,2,0,i)

1000  continue 
********************************************************
*     End Loop over the Scatteres
*******************************************************
      Return
      End	

      Subroutine SetInvT(Tdim,Nscat,Haij,Hbij,rho0,alpha0,
     &           beta0,rho1,alpha1,beta1,w,radius,Lmax,
     &       	 DL,BLKS,XcYc,Tmat,InvT)

      Implicit 	 none
      Integer	 Tdim,Lmax,Nscat,BLKS,DL
      Real*8 	 rho0,alpha0,beta0,rho1,alpha1
      Real*8     beta1,w,radius,fl,flsq,wsq
      Complex*16 Tmat(2,2,-Lmax:Lmax,Nscat)	  
      Complex*16 Haij(0:DL),Hbij(0:DL)
	  
	  
      Complex*16  InvT(Tdim,Tdim),Emij,Ci,C1
      Complex*16  TPP,TPS,TSP,TSS,Paij,Pbij
      Real*8	  dij,thetaij,sign	
      Real*8	  XcYc(Nscat,2),Xci,Yci,Xcj,Ycj
      Real*8 	  Ka0,Kb0,Xaij,Xbij
      Integer	  i,j,ioff,joff,gioff,gjoff,L,N,iabsL,N1,lmn

********************************************************
      integer     Nz,ifail
      EXTERNAL 	  s17def,s17dlf	
********************************************************
       
      Ci=(0.D0,1.D0)
      C1=(1.D0,0.D0)
      wsq=w*w
*     set the waveno. of host
      Ka0=w/alpha0
      Kb0=w/beta0

      Do i=1,Nscat
	ioff=Blks*(i-1)

	Xci=XcYc(i,1)
	Yci=XcYc(i,2)

	Do j=1,Nscat
	  joff=Blks*(j-1)
	  If(i.ne.j) then

	    Xcj=XcYc(j,1)
	    Ycj=XcYc(j,2)
*           get coordinates of j RELATIVE to i
	    call c2p(Xci,Yci,Xcj,Ycj,dij,thetaij)

*	    generate translation matrix coeffs
	    Xaij=Ka0*dij
	    Xbij=Kb0*dij

            CALL s17dlf (1,0.D0, Dcmplx(Xaij), DL+1,'u', 
     &          Haij(0),Nz,ifail)	  	  
            CALL s17dlf (1,0.D0, Dcmplx(Xbij), DL+1,'u', 
     &          Hbij(0),Nz,ifail)	  	  

	    Do L=-Lmax,Lmax
		iabsL=iabs(L)
		gioff=ioff+2*(L+Lmax)
		fl=Dfloat(L)
		flsq=fl*fl
					
*               get Tmatrix for scatterer-i
	        TPP=Tmat(1,1,l,i)
	        TPS=Tmat(1,2,l,i)
		TSP=Tmat(2,1,l,i)
	        TSS=Tmat(2,2,l,i)

		Do N= -Lmax,Lmax
		  gjoff=joff+2*(N+Lmax)
		  lmn=l-n
		  Emij=CDexp(DCmplx(0.D0,-thetaij*lmn))

		  sign=1.0
		  if(lmn.lt.0) then
		    lmn=-lmn
		    n1=lmn-int(lmn/2)*2
		    if(n1.eq.1) then
			sign=-1.0
		    Endif
		  endif
		  Paij=sign*Emij*Haij(lmn)
		  Pbij=sign*Emij*Hbij(lmn)
*       type*,'iscat,jscat=',i,j
*       type*,'l,n,Paij,Pbij=',l,n
*       type*,Paij
*       type*,Pbij
*       type*

*                       -Ti*Pij
		  InvT(gioff+1,gjoff+1)=-TPP*Paij
		  InvT(gioff+1,gjoff+2)=-TPS*Pbij
		  InvT(gioff+2,gjoff+1)=-TSP*Paij
		  InvT(gioff+2,gjoff+2)=-TSS*Pbij
		Enddo                                  !n loop
	      Enddo				       !l loop

	    Else				!(iscat.eq.jscat)
                                !set the identity diagonal BLOCK

	      Do l=-Lmax,Lmax
		gioff=ioff+2*(L+Lmax)

		Do n= -Lmax,Lmax
		  gjoff=joff+2*(N+Lmax)
		  InvT(gioff+1,gjoff+1)=(0.D0,0.D0)
		  InvT(gioff+1,gjoff+2)=(0.D0,0.D0)
		  InvT(gioff+2,gjoff+1)=(0.D0,0.D0)
		  InvT(gioff+2,gjoff+2)=(0.D0,0.D0)

		Enddo 			               !n loop
		InvT(gioff+1,gioff+1)=(1.D0,0.D0)
		InvT(gioff+2,gioff+2)=(1.D0,0.D0)

	      Enddo 				       !l loop
	    Endif 				       !(i.eq.j)
	  Enddo 				       !j loop	
	Enddo 					       !i loop
      Return
      End		

      Subroutine PwCyl(alpha0,beta0,Nscat,XcYc,
     &     H,Np,Tdim,w,Lmax,Pvec,Spath,PCin,PCout)
      Implicit none
      Integer    Nscat,Tdim,Lmax,Np
      Real*8     XcYc(Nscat,2)
      Real*8     w,H,alpha0,beta0,Pvec(Np)
      Complex*16 Spath(Tdim,Tdim),PCin(Tdim,4*Np),PCout(4*Np,Tdim)

      Integer    ipi,ipi1,ipi2,ipi3,ipo,ipo1,ipo2,ipo3
      Integer    loci,loco,mi,mo,Np2,Np4
      Integer    ioffTi,ioffTo
      Real*8     pi,po,xi,xo,zi,zo
      Complex*16 facp,facs
      Complex*16 etai,etao,netai,netao,Exi,Exo
      Complex*16 eadi,ebdi,eaui,ebui
      Complex*16 eado,ebdo,eauo,ebuo
      Complex*16 VERTICALSLO,Ci
      Complex*16 rtetai,rtnetai,rtetao,rtnetao      
      Complex*16 phidai,phiuai,phidao,phiuao    
      Complex*16 phidbi,phiubi,phidbo,phiubo    

      Ci=(0.D0,1.D0)
      Np2=2*Np

      Np4=4*Np

      Do 1000 ipi=1,Np
        ipi1=ipi+Np
        ipi2=ipi+2*Np
        ipi3=ipi+3*Np
        pi=pvec(ipi)
        etai=VERTICALSLO(alpha0,pi)
        netai=VERTICALSLO(beta0,pi)
        rtetai=CDsqrt(etai)
        rtnetai=CDsqrt(netai)

*        type*,'pi=',pi

        Do 998 loci=1,Nscat
          xi=XcYc(loci,1)
          zi=XcYc(loci,2)
          Exi=CDexp(Ci*w*pi*xi)
          eadi=CDexp(Ci*w*etai*zi)
          ebdi=CDexp(Ci*w*netai*zi)
          eaui=CDexp(Ci*w*etai*(H-zi))
          ebui=CDexp(Ci*w*netai*(H-zi))
          facp=Sqrt(2.0)*Exi/(w*rtetai)
          facs=Sqrt(2.0)*Exi/(w*rtnetai)

          Do 994 mi=-Lmax,Lmax
            ioffTi=(4*Lmax+2)*(loci-1)+2*(Lmax+mi)
            phidai=facp*eadi*(alpha0*(pi+Ci*etai))**mi
            phidbi=facs*ebdi*(beta0*(pi+Ci*netai))**mi


            phiuai=facp*eaui*(alpha0*(pi-Ci*etai))**mi
            phiubi=facs*ebui*(beta0*(pi-Ci*netai))**mi

            PCin(ioffti+1,ipi)=phidai
            PCin(ioffti+2,ipi)=(0.D0,0.D0)
            PCin(ioffti+1,ipi1)=(0.D0,0.D0)
            PCin(ioffti+2,ipi1)=phidbi

            PCin(ioffti+1,ipi2)=phiuai
            PCin(ioffti+2,ipi2)=(0.D0,0.D0)
            PCin(ioffti+1,ipi3)=(0.D0,0.D0)
            PCin(ioffti+2,ipi3)=phiubi


994	  Continue

998	Continue

1000  Continue

      Do 999  ipo=1,Np
        ipo1=ipo+Np
        ipo2=ipo+2*Np
        ipo3=ipo+3*Np
        po=pvec(ipo)
        etao=VERTICALSLO(alpha0,po)
        netao=VERTICALSLO(beta0,po)
        rtetao=CDsqrt(etao)
        rtnetao=CDsqrt(netao)

*        type*,'po=',po

        Do 997 loco=1,Nscat
          xo=XcYc(loco,1)
          zo=XcYc(loco,2)
          Exo=CDexp(-Ci*w*po*xo)
          eado=CDexp(Ci*w*etao*zo)
          ebdo=CDexp(Ci*w*netao*zo)
          eauo=CDexp(Ci*w*etao*(H-zo))
          ebuo=CDexp(Ci*w*netao*(H-zo))
          facp=Sqrt(2.0)*Exo/(w*rtetai)
          facs=Sqrt(2.0)*Exo/(w*rtnetai)

	  Do 993 mo=-Lmax,Lmax
            ioffTo=(4*Lmax+2)*(loco-1)+2*(Lmax+mo)
            phidao=facp*eado*(alpha0*(po+Ci*etao))**mo
            phidbo=facs*ebdo*(beta0*(po+Ci*netao))**mo


            phiuao=facp*eauo*(alpha0*(po-Ci*etao))**mo
            phiubo=facs*ebuo*(beta0*(po-Ci*netao))**mo

            PCout(ipo,ioffto+1)=phidao
            PCout(ipo,ioffto+2)=(0.D0,0.D0)
            PCout(ipo1,ioffto+1)=(0.D0,0.D0)
            PCout(ipo1,ioffto+2)=phidbo

            PCout(ipo2,ioffto+1)=phiuao
            PCout(ipo2,ioffto+2)=(0.D0,0.D0)
            PCout(ipo3,ioffto+1)=(0.D0,0.D0)
            PCout(ipo3,ioffto+2)=phiubo
993	  Continue

997	Continue

999   Continue

      return
      end


      COMPLEX*16 FUNCTION VERTICALSLO(C,P)
      Implicit none
      REAL*8 C,P,oneoverc,cmin
	  parameter(cmin=1.0D-6)
	  COMPLEX*16 T
	  if(Dabs(c).le.cmin)then
	    PAUSE 'propagation velocity too small'
	  else
	    oneoverc=1.D0/c
	  end if
	  T=DCMPLX((oneoverC+P)*(oneoverC-P),0.D0)
	  VERTICALSLO=cDsqrt(T)
	  IF(DIMAG(VERTICALSLO) .LT. 0.0)THEN 
	    VERTICALSLO=-VERTICALSLO
	  END IF
	  RETURN
	  END

	Subroutine c2p(xi,yi,xj,yj,dij,thetaij)
*       return polar angle 0<= thetaij < 2 Pi	
*       of (xj,yj) relative to (xi,yi) and dist
*       dij of (xj,yj) relative to (xi,yi)
	Implicit none
	real*8 xi,yi,xj,yj,dij,thetaij,x,y,arg
	x=xj-xi
	y=yj-yi
	dij=DSqrt(x**2.D0+y**2.D0)
*
*       get angle from the depth axis y
*
	thetaij=arg(y,x)
	return
	end

	Real*8 function arg(x,y)
*       return polar angle 0<=arg<2 Pi	of (xj,yj) relative to (xi,yi)
	Implicit none 
	Real*8 	x,y,Pi,Pid2,ratio,ans
	Pi=4.0*Datan(1.D0)
	Pid2=Pi/2.D0
	if(x.eq. 0.D0) then
	  if(y.gt.0.D0) then
	    arg= Pid2
	    return
	  elseif(y.lt.0.D0) then 	
	    arg= 3.0*Pid2
	    return
	  else
	    STOP '(0.,0.) arg'
	  endif
	  Endif
	  ratio=y/x
	  ans=Datan(ratio)
	  if(ratio .ge. 0.D0) then
	    if(x.gt. 0.D0)then
	     arg=ans
	     return
	    else
	     arg=ans+pi
	     return
	    endif
	  Endif
	  if(x.gt.0.D0) then
	    arg=ans+2.D0*Pi
	  return
	  else
	   arg=ans+Pi
	   return
	  endif
	  Stop'you shouldnt be here'
	  END

