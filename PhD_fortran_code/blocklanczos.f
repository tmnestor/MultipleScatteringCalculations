      Program  BlockLanczos1
      Implicit none
	  Integer  Nl,Np,Mp,N,M,N1,il,ilp1,i,j,k,l,ip,ioff,itype,iter,dim
	  Parameter(Np=3) !must be odd
	  Parameter(Nl=2,N1=Nl+1,N=4*Nl*Np)
      real     rho0(0:N1),alpha0(0:N1),beta0(0:N1)
      real     rho(0:N1),alpha(0:N1),beta(0:N1)
      real     drho(Nl),dalpha(Nl),dbeta(Nl)
	  real     w,p(Np),h(0:Nl),Xs,Xr,pmax,dp
	  Integer  is  !is=1=>Exp; is=2=>Doub
	  Integer  ir  !ir=1=>X-disp; ir=2=>Z-disp
	  logical  adj,sing

*******************************************************
      Complex  Smat(4*Np,4*Np,Nl)
	  Complex  Bmat(N,Np),Rmat(Np,Np)
	  complex  X(N),Resp0(Np)
	  Complex  XSY,C0,C1,temp,zdotu

******************************************************


	  DATA     alpha0/1.51,2.51,3.51,4.51/
	  DATA     beta0/1.01,2.01,3.01,4.01/	
      DATA     rho0/1.01,2.01,3.01,4.01/
      DATA     drho/1.0,1.0/
	  DATA     dalpha/1.0,1.0/
	  DATA     dbeta/1.0,1.0/
      DATA     Xs,Xr/0.0,10.0/
      DATA     is,ir/2,2/
	  DATA     pmax/.1/
	  DATA     W,h/1.0,0.5,1.0,1.0/
	  Data     adj/.false./
	  
	  C0=(0.0,0.0)
	  C1=(1.0,0.0)
      if(Np.gt.1)then
        dp=2.0*Pmax/float(Np-1)
	    Do ip=1,Np
	      P(ip)=-pmax+(ip-1)*dp
		  Type*,'ip,p(ip)=',ip,p(ip)
	    Enddo
	  else
	    TYpe*,'input a single p value'
	    Accept*,p
	  Endif

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
		TYpe*,'il,alpha(il)=',il,alpha(il)
		TYpe*,'il,beta(il)=',il,beta(il)
		TYpe*,'il,rho(il)=',il,rho(il)
	  Enddo

*********************************************
*     set Global scattering array
*********************************************
	  
	  call setSmat(Nl,Np,w,p,h,rho0,alpha0,
     &                 beta0,rho,alpha,beta,Smat)

*********************************************
*     set Bmat
*********************************************

	  Do i=1,N
	    Do ip=1,Np
		  Bmat(i,ip)=(0.0,0.0)
		Enddo
      Enddo

      Do ip=1,Np
	  
        call setInc(Nl,Xs,is,ir,w,p(ip),h,rho0,alpha0,beta0,X,Resp0(ip))
	 
        Do il=1,Nl
          ioff=4*(il-1)
           Do itype=1,4
             Bmat(4*Np*(il-1)+Np*(itype-1)+ip,ip)=X(ioff+itype)
           Enddo !itype
         Enddo !il
*        Type*,'Resp=',Resp(ip)
      Enddo !ip
	  
	  temp=XSY(Nl,Np,SMAT,Bmat(1,1),Bmat(1,3))
	  temp=temp-XSY(Nl,Np,SMAT,Bmat(1,3),Bmat(1,1))
	  TYpe*,'<1,3>-<3,1>=',temp
      pause
	  

	  Do i=1,N
        Do ip=1,Np
	    Type*,'i,Bmat(i,ip)=',i,ip,Bmat(i,ip)
        Enddo
	  Enddo
      pause


      call setQR(N,Nl,Np,Np,Smat,Bmat,Rmat)

	  Do i=1,N
        Do ip=1,Np
	    Type*,'i,Bmat(i,ip)=',i,ip,Bmat(i,ip)
        Enddo
	  Enddo

	  Do i=1,Np
	    Do j=1,Np
		  TYpe*,'Rmat(i,j)=',i,j,Rmat(i,j)
		Enddo
      Enddo
	  
	  Do i=1,Np
	    Do j=1,Np
*	      TYpe*,'<i,j>=',i,j, XSY(Nl,Np,SMAT,Bmat(1,i),Bmat(1,j))
	      TYpe*,'<i,j>=',i,j, zdotu(N,Bmat(1,i),1,Bmat(1,j),1)
	    Enddo
	  Enddo

      stop'normal completion'
	  end
*************************************
      subroutine setSmat(Nl,Np,w,p,h,rho0,alpha0,
     &                 beta0,rho,alpha,beta,Smat)
      Implicit none
      integer  Nl,Np,il,ipi,ipo,ioffi,ioffo,Np2,Np3
      real     h(0:Nl),rho0(0:Nl),rho(0:Nl)
      real     alpha0(0:Nl),beta0(0:Nl)
      real     alpha(0:Nl),beta(0:Nl),w,p(Np)
      complex  S(4,4),Smat(4*Np,4*Np,Nl)

*     set SMAT
      Np2=2*Np
      Np3=3*Np

      Do il=1,Nl
      Do ipo=1,Np !loop over scattered slownesses
      ioffo=4*Np*(il-1)+ipo
      Do ipi=1,Np !loop over incident slownesses
      ioffi=4*Np*(il-1)+ipi

          If(ipi.eq.ipo) then
	  	    call SetS(il,Nl,w,p(ipi),h,rho0,alpha0,beta0,
     &           rho,alpha,beta,S)
          else !offdiagonal coupling
            S(1,1)=(0.0,0.0)
            S(1,2)=(0.0,0.0)
            S(1,3)=(0.0,0.0)
            S(1,4)=(0.0,0.0)
            S(2,1)=(0.0,0.0)
            S(2,2)=(0.0,0.0)
            S(2,3)=(0.0,0.0)
            S(2,4)=(0.0,0.0)
            S(3,1)=(0.0,0.0)
            S(3,2)=(0.0,0.0)
            S(3,3)=(0.0,0.0)
            S(3,4)=(0.0,0.0)
            S(4,1)=(0.0,0.0)
            S(4,2)=(0.0,0.0)
            S(4,3)=(0.0,0.0)
            S(4,4)=(0.0,0.0)
 		  Endif
		  Smat(ipo,ipi,il)=S(1,1)
		  Smat(ipo,ipi+Np,il)=S(1,2)
		  Smat(ipo,ipi+Np2,il)=S(1,3)
		  Smat(ipo,ipi+Np3,il)=S(1,4)

		  Smat(ipo+Np,ipi,il)=S(2,1)
		  Smat(ipo+Np,ipi+Np,il)=S(2,2)
		  Smat(ipo+Np,ipi+Np2,il)=S(2,3)
		  Smat(ipo+Np,ipi+Np3,il)=S(2,4)

		  Smat(ipo+Np2,ipi,il)=S(3,1)
		  Smat(ipo+Np2,ipi+Np,il)=S(3,2)
		  Smat(ipo+Np2,ipi+Np2,il)=S(3,3)
		  Smat(ipo+Np2,ipi+Np3,il)=S(3,4)
	  

		  Smat(ipo+Np3,ipi,il)=S(4,1)
		  Smat(ipo+Np3,ipi+Np,il)=S(4,2)
		  Smat(ipo+Np3,ipi+Np2,il)=S(4,3)
		  Smat(ipo+Np3,ipi+Np3,il)=S(4,4)
	  
	  Enddo !ipi
	  Enddo !ipo
	  End do !il
	  return
	  end

      subroutine setQR(N,Nl,Np,M,Smat,V,R)
      implicit    none
      integer     N,Nl,Np,M,j,k
      Complex     Smat(4*Np,4*Np,Nl)
      complex     V(N,M),R(M,M),zdotu,sk,XSY

      Do k=1,M
        Do j=1,M
          R(k,j)=(0.0,0.0)
        enddo
      enddo

      Do 100 k=1,M
        Do j=1,N
          type*,'j,V(j,k)=',j,V(j,k)
        enddo
*        R(k,k)=Csqrt(XSY(Nl,Np,SMAT,V(1,k),V(1,k)))
	    R(k,k)=Csqrt(zdotu(N,V(1,k),1,V(1,k),1))
        type*,'k,R(k,k)=',k,R(k,k)
        pause
        sk=(1.0,0.0)/R(k,k)
        call zscal(N,sk,V(1,k),1)
        Do 90 j=k+1,m
*          R(k,j)=XSY(Nl,Np,SMAT,V(1,k),V(1,j))
          R(k,j)=zdotu(N,V(1,k),1,V(1,j),1)
          call zaxpy(n,-R(k,j),V(1,k),1,V(1,j),1)
90      continue
100   continue
      return
      end

      complex function XSY(Nl,Np,SMAT,X,Y)
      Implicit none
      integer  Nl,Np,il,ioff,Np4,N
      complex  Y(4*Nl*Np),X(4*Nl*Np)
      Complex  Smat(4*Np,4*Np,Nl),zdotu
*     local variables
      complex  S(4,4),C0,C1 
	  complex  SY(100)
      integer  i,j,k,Mp
      complex  dum

      C1=(1.0,0.0)
      C0=(0.0,0.0)
	  
	  N=4*Np*Nl
	  if(N.gt.100)stop 'N.gt.100 in XSY'
	  
	  Np4=4*Np


      DO 99999 IL=1,NL
        ioff=Np4*(il-1)
        call ZGEMV('N',Np4,Np4,C1,SMAT(1,1,il),
     &        Np4,Y(ioff+1),1,C0,SY(ioff+1),1)
99999 continue

	  Mp=(Np+1)/2
	  Do i=1,4*Nl
		  k=(i-1)*Np
		  Do j=1,Mp-1
		    Dum=SY(k+Mp+j)
			SY(k+Mp+j)=SY(k+Mp-j)
			SY(k+Mp-j)=Dum
		  Enddo !j
      Enddo!i
      XSY=zdotu(N,X,1,SY,1)
	  return
	  end
	  
      subroutine swapper1(y,Np,Nl)
      implicit none
      integer  i,j,Np,Mp,Nl,k
      complex  Dum,y(*) 

      Mp=(Np+1)/2
      Do i=1,4*Nl
        k=(i-1)*Np
        Do j=1,Mp-1
          Dum=Y(k+Mp+j)
          Y(k+Mp+j)=Y(k+Mp-j)
          Y(k+Mp-j)=Dum
        Enddo !j
       Enddo!i
       return
       end

*************************************************************
      Subroutine atimes(Nl,Np,Adj,w,p,h,rho0,alpha0,beta0,
     &                  rho,alpha,beta,Y,X)
*     ********************************************************
*     set X=(1-P*dS)*Y	or  X= [I(*)K](1-P*dS)[I(*)K]*Y
*     ********************************************************
      Implicit none
	  Integer   Nl,Np,i,k,j,Mp,ip,il
      real      rho0(0:Nl),alpha0(0:Nl),beta0(0:Nl),h(0:Nl)
      real      rho(0:Nl),alpha(0:Nl),beta(0:Nl),w,p(Np)
	  complex   X(4*Nl*Np),Y(4*Nl*Np)
	  Integer   Nlmax,Npmax,NLM4
	  logical   Adj
*     local variables !ensure sufficient space has been allocated
	  parameter (Nlmax=2,Npmax=45,NLM4=4*Nlmax*Npmax)
	  complex   b(4*Nlmax),c(4*Nlmax),SY(4*Nlmax*Npmax),Dum
      type*,'entered atimes'
	  Do i=1,4*Nl*Np
	    Type*,'i,X(i)=',i,X(i)
	  Enddo
*	  pause
	  if(Nl.gt.NLmax) then
	    stop 'Nl.gt.NLmax in atimes'
	  Endif

	  if(Np.gt.Npmax) then
	    stop 'Np.gt.Npmax in atimes'
	  Endif
      

      If(Adj)then !set [I(*)K] Y
	    if(Np.eq.1) pause 'Adj.and.Np=1 not implemented'
	    Mp=(Np+1)/2
	    Do i=1,4*Nl
		  k=(i-1)*Np
		  Do j=1,Mp-1
		    Dum=Y(k+Mp+j)
			Y(k+Mp+j)=Y(k+Mp-j)
			Y(k+Mp-j)=Dum
		  Enddo !j
		Enddo!i
	  Endif !Adj
*     SY=dS*Y
      call setSY(Nl,Np,w,p,h,rho0,alpha0,beta0,rho,alpha,beta,Y,SY)
      type*,'left setSy'
	  Do i =1, 4*Nl*Np
	    Type*,'SY(i)=',i,SY(i)
	  Enddo
*	  pause
*     X=P*SY
      Do ip=1,Np
	    Do il=1,Nl
		  Do j=1,4
		    b((il-1)*4+j)=SY(4*Np*(il-1)+Np*(j-1)+ip)
		  Enddo!j
		Enddo !il
	    call setPSY(Nl,p(ip),w,p,h,rho0,alpha0,beta0,b,c)
      type*,'left setPSY'
	  Do i =1, 4*Nl
	    Type*,'PSY(i)=',i,c(i)
	  Enddo
	    Do il=1,Nl
		  Do j=1,4
		    X(4*Np*(il-1)+Np*(j-1)+ip)=c((il-1)*4+j)
		  Enddo!j
		Enddo !il
	  Enddo !ip

	  Do i=1,4*Nl*Np
	    X(i)=Y(i)-X(i)
	  Enddo !i

      If(Adj)then !set [I(*)K] X
	    Mp=(Np+1)/2
	    Do i=1,4*Nl
		  k=(i-1)*Np
		  Do j=1,Mp-1
		    Dum=X(k+Mp+j)
			X(k+Mp+j)=X(k+Mp-j)
			X(k+Mp-j)=Dum
*           unswapp Y also because it is needed again
		    Dum=Y(k+Mp+j)
			Y(k+Mp+j)=Y(k+Mp-j)
			Y(k+Mp-j)=Dum

		  Enddo !j
		Enddo!i
	  Endif !Adj

*      Type*,'Leaving amat'
*	  Do i=1,4*Nl*Np
*	    Type*,'i,X(i)=',i,X(i)
*	  Enddo !i
	  return
	  end

      SUBROUTINE setPSY(Nl,p,w,p,h,rho,alpha,beta,SY,X)
	  
*     set X=P*SY
	  Implicit none
      INTEGER N1,IL,Nl,ioff
	  Integer  Nlm,NLM4
	  parameter(Nlm=2,NLM4=4*Nlm,N1=Nlm+1)
	  Integer ilp1,ilm1
      real    w,p,rho(0:N1),alpha(0:N1),beta(0:N1),h(0:Nl)
	  complex SY(NLM4),X(NLM4)

*******************************************************************
      complex  Rf11,Rf12,Rf21,Rf22,InvUv11,InvUv12,InvUv21,InvUv22
	  complex  Nd(2,2),Md(2,2),Nu(2,2),Mu(2,2),VERTICALSLO
	  complex  cbeta1,cbeta2,temp,Ci,C1
	  complex T11(0:NLm),T12(0:NLm),T21(0:NLm),T22(0:NLm)
	  complex U11(0:NLm),U12(0:NLm),U21(0:NLm),U22(0:NLm)
	  complex P11(0:NLm),P21(0:NLm)
	  complex Q11(0:NLm),Q21(0:NLm)
	  complex Bone1(0:NLm),Bone2(0:NLm)
	  complex Btwo1(0:NLm),Btwo2(0:NLm)
	  complex R11(0:NLm),R12(0:NLm),R21(0:NLm),R22(0:NLm)
	  complex S11(0:NLm),S12(0:NLm),S21(0:NLm),S22(0:NLm)
	  complex Rd11(0:NLm),Rd12(0:NLm),Rd21(0:NLm),Rd22(0:NLm)
	  complex Tu11(0:NLm),Tu12(0:NLm),Tu21(0:NLm),Tu22(0:NLm)
	  complex Td11(0:NLm),Td12(0:NLm),Td21(0:NLm),Td22(0:NLm)
	  complex Ru11(0:NLm),Ru12(0:NLm),Ru21(0:NLm),Ru22(0:NLm)
	  complex ea(0:NLm),eb(0:NLm),eta(0:N1),neta(0:N1)
	  complex W11,W12,W21,W22,V11,V12,V21,V22,InvDet,Df1,Df2


	  if(Nl.gt.NLm) then
	    stop 'Nl.gt.NLm in setPSY'
	  Endif


      C1=(1.0,0.0)	  
      Ci=(0.0,1.0)
*     *******************************************************
*     set free surface reflection array Rf11,Rf12,Rf21,Rf22
*     and free-surface amplitude factor
*     *******************************************************
	  do il=0,N1
	     eta(il)=VERTICALSLO(alpha(il),P)
	     neta(il)=VERTICALSLO(beta(il),P)
      end do !il
      Do il=0,Nl
	    Temp=CI*w*h(il)*eta(il)
        EA(il)=CEXP(Temp)
	    Temp=CI*w*h(il)*neta(il)
        EB(il)=CEXP(Temp)		
      End Do

      call Emat(P,eta(0),neta(0),rho(0),beta(0),Md,Nd,Mu,Nu)
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

*     set reference medium interfacial scattering arrays
	  do il=0,Nl
	    ilp1=il+1
	  	cbeta1=cmplx(beta(il),0.0)
	  	cbeta2=cmplx(beta(ilp1),0.0)
*       set the reference medium interfacial scattering matrices
		call ScatMat(p,eta(il),neta(il),rho(il),CBeta1,eta(ilp1),
     &     neta(ilp1),rho(ilp1),CBeta2,Rd11(il),Rd12(il),Rd21(il),
     &     Rd22(il),Tu11(il),Tu12(il),Tu21(il),Tu22(il),Td11(il),
     &     Td12(il),Td21(il),Td22(il),Ru11(il),Ru12(il),Ru21(il),
     &     Ru22(il))
	  end do
*******************************************************	  
*     Factor the System Matrix
      call    Factor(Nl,ea,eb,T11,T12,T21,T22,U11,U12,U21,U22,
     &        R11,R12,R21,R22,S11,S12,S21,S22,Rd11,Rd12,Rd21,Rd22,
     &        Tu11,Tu12,Tu21,Tu22,Ru11,Ru12,Ru21,Ru22,Td11,Td12,
     &        Td21,Td22)


	  call  RHS(NL,ea,eb,Rd11,Rd12,Rd21,Rd22,Td11,Td12,Td21,Td22,
     &          Ru11,Ru12,Ru21,Ru22,Tu11,Tu12,Tu21,Tu22,
     &      SY,Bone1,Bone2,Btwo1,Btwo2)
*      Do il=0,Nl
*	    Type*,'i,Bone1(i)=',il,Bone1(il)
*	    Type*,'i,Btwo1(i)=',il,Btwo1(il)
*	    Type*,'i,Bone2(i)=',il,Bone2(il)
*	    Type*,'i,Btwo2(i)=',il,Btwo2(il)
*	  Enddo
      call  UpwardSweep(NL,Bone1,Bone2,Btwo1,Btwo2,Ru11,Ru12,Ru21,
     &      Ru22,Tu11,Tu12,Tu21,Tu22,ea,eb,T11,T12,T21,T22,U11,U12,
     &      U21,U22,P11,P21,Q11,Q21)
      
*     set Df=downgoing from Free surface
*     Df=Rf*E0*(1-R(0)*Rf*E0)^(-1)*P0
      Rf11=Rf11*ea(0)
      Rf12=Rf12*eb(0)
      Rf21=Rf21*ea(0)
      Rf22=Rf22*eb(0)

*     U=(I-Ru*Rf)^(-1)
	  W11=C1-R21(0)*Rf12-R22(0)*Rf22
	  W12=R11(0)*Rf12+R12(0)*Rf22
	  W21=R21(0)*Rf11+R22(0)*Rf21
	  W22=C1-R11(0)*Rf11-R12(0)*Rf21
	  InvDet=C1/(W11*W22-W12*W21)
	  V11=W11*InvDet
	  V12=W12*InvDet
	  V21=W21*InvDet
	  V22=W22*InvDet
	  
*     W=Rf*V
      W11=Rf11*V11+Rf12*V21
	  W12=Rf11*V12+Rf12*V22
	  W21=Rf21*V11+Rf22*V21
	  W22=Rf21*V12+Rf22*V22

*     Df=W*P(0) !downgoing from the free-surface
      Df1=W11*P11(0)+W12*P21(0)
      Df2=W21*P11(0)+W22*P21(0)

	  
	  call DownwardSweep(NL,R11,R12,R21,R22,S11,S12,S21,S22,
     &                   P11,P21,Q11,Q21,Df1,Df2,X)

	  RETURN
      END

      Subroutine Factor(Nl,ea,eb,T11,T12,T21,T22,U11,U12,U21,U22,
     &           R11,R12,R21,R22,S11,S12,S21,S22,Rd11,Rd12,Rd21,
     &           Rd22,Tu11,Tu12,Tu21,Tu22,Ru11,Ru12,Ru21,Ru22,Td11,
     &           Td12,Td21,Td22)
      Implicit None
	  integer il,nl,ilp1,nlm1
	  complex T11(0:Nl),T12(0:Nl),T21(0:Nl),T22(0:Nl)
	  complex U11(0:Nl),U12(0:Nl),U21(0:Nl),U22(0:Nl)
	  complex R11(0:Nl),R12(0:Nl),R21(0:Nl),R22(0:Nl)
	  complex S11(0:Nl),S12(0:Nl),S21(0:Nl),S22(0:Nl)
	  complex ea(0:Nl),eb(0:Nl)
	  complex W11,W12,W21,W22,V11,V12,V21,V22,C1,InvDet
	  complex Rd11(0:Nl),Rd12(0:Nl),Rd21(0:Nl),Rd22(0:Nl)
	  complex Tu11(0:Nl),Tu12(0:Nl),Tu21(0:Nl),Tu22(0:Nl)
	  complex Td11(0:Nl),Td12(0:Nl),Td21(0:Nl),Td22(0:Nl)
	  complex Ru11(0:Nl),Ru12(0:Nl),Ru21(0:Nl),Ru22(0:Nl)
	  complex MT11,MT12,MT21,MT22
	  C1=(1.0,0.0)
*     R=Rd(nl)*E(nl)
      R11(nl)=ea(nl)*Rd11(nl)
	  R12(nl)=eb(nl)*Rd12(nl)
	  R21(nl)=ea(nl)*Rd21(nl)
	  R22(nl)=eb(nl)*Rd22(nl)
*	  type*,'Rd11(nl),ea(nl),R11(nl)=',Rd11(nl),ea(nl),R11(nl)

*     S=Td(nl)*E(nl)
      S11(nl)=ea(nl)*Td11(nl)
	  S12(nl)=eb(nl)*Td12(nl)
	  S21(nl)=ea(nl)*Td21(nl)
	  S22(nl)=eb(nl)*Td22(nl)
*	  type*,'S11(nl),S22(nl)=',S11(nl),S22(nl)
      nlm1=nl-1
      do il=Nlm1,0,-1
        ilp1=il+1
*       MT=E(ilp1)*R(ilp1)
        MT11=Ea(ilp1)*R11(ilp1)
        MT12=Ea(ilp1)*R12(ilp1)
        MT21=Eb(ilp1)*R21(ilp1)
        MT22=Eb(ilp1)*R22(ilp1)
*        type*,'MT11,MT22,il=',ilp1,MT11,MT22
*       U=(I-Ru*MT)^(-1)
	    W11=C1-Ru21(il)*MT12-Ru22(il)*MT22
	    W12=Ru11(il)*MT12+Ru12(il)*MT22
	    W21=Ru21(il)*MT11+Ru22(il)*MT21
	    W22=C1-Ru11(il)*MT11-Ru12(il)*MT21
	    InvDet=C1/(W11*W22-W12*W21)
	    U11(il)=W11*InvDet
	    U12(il)=W12*InvDet
	    U21(il)=W21*InvDet
	    U22(il)=W22*InvDet
*		type*,'il,U11,U12,U21,U22=',il
*		type*,U11(il),U12(il),U21(il),U22(il)
*        type*
*       W=MT*U
        W11=MT11*U11(il)+MT12*U21(il)
		W12=MT11*U12(il)+MT12*U22(il)
		W21=MT21*U11(il)+MT22*U21(il)
		W22=MT21*U12(il)+MT22*U22(il)

*       T=Tu*W
        T11(il)=Tu11(il)*W11+Tu12(il)*W21
		T12(il)=Tu11(il)*W12+Tu12(il)*W22
		T21(il)=Tu21(il)*W11+Tu22(il)*W21
		T22(il)=Tu21(il)*W12+Tu22(il)*W22
*		type*,'il,T11,T12,T21,T22=',il
*		type*,T11(il),T12(il),T21(il),T22(il)
*        type*


*       V=Td(il)*E(il)
        V11=ea(il)*Td11(il)
		V12=eb(il)*Td12(il)
		V21=ea(il)*Td21(il)
		V22=eb(il)*Td22(il)

*       S=U*V
        S11(il)=U11(il)*V11+U12(il)*V21
		S12(il)=U11(il)*V12+U12(il)*V22
		S21(il)=U21(il)*V11+U22(il)*V21
		S22(il)=U21(il)*V12+U22(il)*V22
        
*		type*,'il,S11,S12,S21,S22=',il
*		type*,S11(il),S12(il),S21(il),S22(il)
*        type*

*       W=T*V
        W11=T11(il)*V11+T12(il)*V21
		W12=T11(il)*V12+T12(il)*V22
		W21=T21(il)*V11+T22(il)*V21
		W22=T21(il)*V12+T22(il)*V22

*       R=Rd*E(il)+W
        R11(il)=Rd11(il)*ea(il)+W11
		R12(il)=Rd12(il)*eb(il)+W12
		R21(il)=Rd21(il)*ea(il)+W21
		R22(il)=Rd22(il)*eb(il)+W22

*		type*,'il,R11,R12,R21,R22=',il
*		type*,R11(il),R12(il),R21(il),R22(il)
*        type*
      end do !il

      return
	  end
	  
	  Subroutine  RHS(NL,ea,eb,Rd11,Rd12,Rd21,Rd22,Td11,Td12,
     &            Td21,Td22,Ru11,Ru12,Ru21,Ru22,Tu11,Tu12,Tu21,
     &            Tu22,Y,Bone1,Bone2,Btwo1,Btwo2)
	  Implicit None
      Integer Nl,Nlm1,il,ioff,ioffp
	  complex Bone1(0:Nl),Bone2(0:Nl)
	  complex Btwo1(0:Nl),Btwo2(0:Nl)
	  complex Y(4*Nl)
	  complex Rd11(0:Nl),Rd12(0:Nl),Rd21(0:Nl),Rd22(0:Nl)
	  complex Td11(0:Nl),Td12(0:Nl),Td21(0:Nl),Td22(0:Nl)
	  complex Ru11(0:Nl),Ru12(0:Nl),Ru21(0:Nl),Ru22(0:Nl)
	  complex Tu11(0:Nl),Tu12(0:Nl),Tu21(0:Nl),Tu22(0:Nl)
	  complex ea(0:Nl),eb(0:Nl),C0

*     Bone(0)=Tu(0)*SU(1)
      Bone1(0)=Tu11(0)*Y(1)+Tu12(0)*Y(2)
      Bone2(0)=Tu21(0)*Y(1)+Tu22(0)*Y(2)

*     Btwo(0)=Ru(0)*SU(1)
      Btwo1(0)=Ru11(0)*Y(1)+Ru12(0)*Y(2)
      Btwo2(0)=Ru21(0)*Y(1)+Ru22(0)*Y(2)

      nlm1=nl-1
      do il=1,nlm1 !interface loop
	      ioff=4*(il-1)
		  ioffp=ioff+4

*         Bone(il)=Rd(il)*SD(il)+Tu(il)*SU(il+1)
          Bone1(il)=Rd11(il)*Y(ioff+3)+Rd12(il)*Y(ioff+4)+
     &              Tu11(il)*Y(ioffp+1)+Tu12(il)*Y(ioffp+2)
          Bone2(il)=Rd21(il)*Y(ioff+3)+Rd22(il)*Y(ioff+4)+
     &              Tu21(il)*Y(ioffp+1)+Tu22(il)*Y(ioffp+2)

*         Btwo(il)=Td(il)*SD(il)+Ru(il)*SU(il+1)
          Btwo1(il)=Td11(il)*Y(ioff+3)+Td12(il)*Y(ioff+4)+
     &              Ru11(il)*Y(ioffp+1)+Ru12(il)*Y(ioffp+2)
          Btwo2(il)=Td21(il)*Y(ioff+3)+Td22(il)*Y(ioff+4)+
     &              Ru21(il)*Y(ioffp+1)+Ru22(il)*Y(ioffp+2)
      end do !il
	  ioff=4*nlm1

*     Bone(Nl)=Rd(Nl)*SD(Nl)
      Bone1(Nl)=Rd11(Nl)*Y(ioff+3)+Rd12(Nl)*Y(ioff+4)
      Bone2(Nl)=Rd21(Nl)*Y(ioff+3)+Rd22(Nl)*Y(ioff+4)

*     Btwo(Nl)=Td(Nl)*SD(Nl)
      Btwo1(Nl)=Td11(Nl)*Y(ioff+3)+Td12(Nl)*Y(ioff+4)
      Btwo2(Nl)=Td21(Nl)*Y(ioff+3)+Td22(Nl)*Y(ioff+4)
      return
	  end


	  
	  Subroutine DownwardSweep(NL,R11,R12,R21,R22,S11,S12,
     &           S21,S22,P11,P21,Q11,Q21,Df1,Df2,X)
	  Implicit None
	  Integer NL,il,ilm1,i1,i2
	  complex P11(0:Nl),P21(0:Nl),Q11(0:Nl),Q21(0:Nl)
	  complex X(4*Nl)
	  complex R11(0:Nl),R12(0:Nl),R21(0:Nl),R22(0:Nl),Df1,Df2
	  complex S11(0:Nl),S12(0:Nl),S21(0:Nl),S22(0:Nl)

***************************************
	  
	  X(1)=Q11(0)+S11(0)*Df1+S12(0)*Df2
	  X(2)=Q21(0)+S21(0)*Df1+S22(0)*Df2
      X(3)=P11(1)+R11(1)*X(1)+R12(1)*X(2)
	  X(4)=P21(1)+R21(1)*X(1)+R22(1)*X(2)
****************************************
	  do il=2,nl
	    i1=(il-2)*4
	    i2=i1+4
		ilm1=il-1
	    X(i2+1)=Q11(ilm1)+S11(ilm1)*X(i1+1)+S12(ilm1)*X(i1+2)
	    X(i2+2)=Q21(ilm1)+S21(ilm1)*X(i1+1)+S22(ilm1)*X(i1+2)
        X(i2+3)=P11(il)+R11(il)*X(i2+1)+R12(il)*X(i2+2)
		X(i2+4)=P21(il)+R21(il)*X(i2+1)+R22(il)*X(i2+2)
	  end do !il
	  return
	  end
	  
	  
      Subroutine UpwardSweep(NL,Bone1,Bone2,Btwo1,Btwo2,Ru11,
     &           Ru12,Ru21,Ru22,Tu11,Tu12,Tu21,Tu22,ea,eb,T11,
     &           T12,T21,T22,U11,U12,U21,U22,P11,P21,Q11,Q21)
	  Implicit None
	  Integer NL,NLM1,il,ilp1
	  complex T11(0:Nl),T12(0:Nl),T21(0:Nl),T22(0:Nl)
	  complex U11(0:Nl),U12(0:Nl),U21(0:Nl),U22(0:Nl)
	  complex P11(0:Nl),P21(0:Nl)
	  complex Q11(0:Nl),Q21(0:Nl)
	  complex Bone1(0:Nl),Bone2(0:Nl)
	  complex Btwo1(0:Nl),Btwo2(0:Nl)
	  complex Tu11(0:Nl),Tu12(0:Nl),Tu21(0:Nl),Tu22(0:Nl)
	  complex Ru11(0:Nl),Ru12(0:Nl),Ru21(0:Nl),Ru22(0:Nl)
	  complex ea(0:nl),eb(0:nl)
	  complex W11,W21,V11,V21,M11,M21
***********************************************
*     set P and Q	 
***********************************************
	  P11(nl)=Bone1(nl)
	  P21(nl)=Bone2(nl)
	  Q11(nl)=Btwo1(nl)
	  Q21(nl)=Btwo2(nl)
	  nlm1=nl-1
	  do il=nlm1,0,-1
	    ilp1=il+1
*       V=E(ilp1)*P(ilp1)
        V11=Ea(ilp1)*P11(ilp1)
        V21=Eb(ilp1)*P21(ilp1)

*       W=Tu*V
        W11=Tu11(il)*V11+Tu12(il)*V21
		W21=Tu21(il)*V11+Tu22(il)*V21

*       M=Ru*V
        M11=Ru11(il)*V11+Ru12(il)*V21
		M21=Ru21(il)*V11+Ru22(il)*V21

*       V=Bone+W
        V11=Bone1(il)+W11
        V21=Bone2(il)+W21

*       W=Btwo+M
        W11=Btwo1(il)+M11
        W21=Btwo2(il)+M21
    
*	    P(il)=V+T(il)*W
        P11(il)=V11+T11(il)*W11+T12(il)*W21
		P21(il)=V21+T21(il)*W11+T22(il)*W21
*		type*,'il,P11,P21,V11,W11=',il,P11(il),P21(il),V11,W11

*       Q(il)=U(il)*W
        Q11(il)=U11(il)*W11+U12(il)*W21
		Q21(il)=U21(il)*W11+U22(il)*W21
	  end do !il
      return
	  end

      subroutine Emat(P,ETA,NETA,RHO,BETA,Md,Nd,Mu,Nu)
*********************************************************
*     form the layer eigen arrays Md,Nd
*********************************************************
      implicit none
      real    rho,twomup,beta,p
      complex eta,neta,Nd(2,2),Md(2,2),Nu(2,2),Mu(2,2)
	  complex ci,epsa,epsb,c1
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


	  Subroutine  ScatMat(p,eta1,neta1,rho1,beta1,eta2,neta2,rho2,
     &            beta2,Rd11,Rd12,Rd21,Rd22,Tu11,Tu12,Tu21,Tu22,
     &            Td11,Td12,Td21,Td22,Ru11,Ru12,Ru21,Ru22)
*     ********************************************************************
*     plane-wave interfacial scattering matrices  
*     modified from Aki&Richards pp149-51
*     ********************************************************************
	  real    RHO1,RHO2,rtrho1,rtrho2,p,m2
	  complex ETA1,NETA1,ETA2,NETA2,BETA1,BETA2
	  complex PSQ,CRHO1,CRHO2,DRHO,Dmu
	  complex rteta1,rteta2,rtneta1,rtneta2
	  complex Rd11,Rd12,Rd21,Rd22
      complex Tu11,Tu12,Tu21,Tu22,Td11,Td12,Td21
	  complex Td22,Ru11,Ru12,Ru21,Ru22
	  complex PSQD,a,b,c,d,Det,E,F,G,H,Q,R,S,T,U,V
	  complex rtza1,rtza2,rtzb1,rtzb2

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
	  If(Cabs(Det).lt.0.000001) then
	    Type*,'hit Stonley pole in ScatMat, p=',p
		Pause
	  Endif
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


	  complex FUNCTION VERTICALSLO(C,P)
	  real C,P,oneoverc,cmin
	  parameter(cmin=1.0E-6)
	  complex T
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
	  If(Cabs(VERTICALSLO).lt. 0.000001) then
	    Pause'Cabs(VERTICALSLO).lt. 0.000001'
	  Endif
	  RETURN
	  END

************************************
	  Subroutine setSY(Nl,Np,w,p,h,rho0,alpha0,
     &                 beta0,rho,alpha,beta,Y,SY)
	  Implicit none
	  integer  Nl,Np,il,ioffo,ioffi,ipi,ipo,jo,ji
	  complex  SY(4*Nl*Np),Y(4*Nl*Np)
      real     rho0(0:Nl),alpha0(0:Nl),beta0(0:Nl),h(0:Nl)
      real     rho(0:Nl),alpha(0:Nl),beta(0:Nl),w,p(Np)
	  complex  S(4,4),sum  
*      Type*,'entered setSY'
*	  pause
	  Do il=1,4*Np*Nl
	    SY(il)=(0.0,0.0)
	  Enddo
*     set SY=S*Y
      Do il=1,Nl
        Do ipo=1,Np !loop over scattered slownesses
		  ioffo=4*Np*(il-1)+ipo
          Do ipi=1,Np !loop over incident slownesses
		  ioffi=4*Np*(il-1)+ipi

          If(ipi.eq.ipo) then
	  	    call SetS(il,Nl,w,p(ipi),h,rho0,alpha0,beta0,
     &           rho,alpha,beta,S)
          else !offdiagonal coupling
            S(1,1)=(0.0,0.0)
            S(1,2)=(0.0,0.0)
            S(1,3)=(0.0,0.0)
            S(1,4)=(0.0,0.0)
            S(2,1)=(0.0,0.0)
            S(2,2)=(0.0,0.0)
            S(2,3)=(0.0,0.0)
            S(2,4)=(0.0,0.0)
            S(3,1)=(0.0,0.0)
            S(3,2)=(0.0,0.0)
            S(3,3)=(0.0,0.0)
            S(3,4)=(0.0,0.0)
            S(4,1)=(0.0,0.0)
            S(4,2)=(0.0,0.0)
            S(4,3)=(0.0,0.0)
            S(4,4)=(0.0,0.0)
 		  Endif

          Do jo=1,4	
		     sum=(0.0,0.0)
		     Do ji=1,4
			   sum=sum+S(jo,ji)*Y(ioffi+Np*(ji-1))
*			   TYpe*,'jo,ji,S(jo,ji)=',jo,ji,S(jo,ji)
*			   TYpe*,'ioffi+Np*(ji-1),Y(ioffi+Np*(ji-1))=',
*     &                ioffi+Np*(ji-1),Y(ioffi+Np*(ji-1))
*			   Type*,'sum=',sum
             Enddo !ji
*			 Type*,'il,ipo,ipi,jo,p(ipi)=',il,ipo,ipi,jo,p(ipi)
			 SY(ioffo+Np*(jo-1))=SY(ioffo+Np*(jo-1))+sum
*			 Type*,'sum=',sum
*			 Type*,'ioffo+Np*(jo-1),SY(ioffo+Np*(jo-1))'
*			 Type*,ioffo+Np*(jo-1),SY(ioffo+Np*(jo-1))
*			 Type*,'*************************************'
*			 Type*
		  Enddo !jo

	      Enddo !ipi
*		  pause
	    Enddo !ipo
*		pause
	  End do !il
      return
	  end

	  
	  
	  
      subroutine SetD(P,ETA,NETA,RHO,BETA,Md,Nd,Mu,Nu)
*********************************************************
*     form the layer eigen arrays Md,Nd
*********************************************************
      implicit none
      real    rho,twomup,beta,p
      complex eta,neta,Nd(2,2),Md(2,2),Nu(2,2),Mu(2,2)
	  complex ci,epsa,epsb,c1
	  Ci=cmplx(0.0,1.0)
	  C1=cmplx(1.0,0.0)
	  twomup=2.0*rho*beta*beta*p
	  epsa=C1/Csqrt(2.0*rho*eta)
	  epsb=C1/Csqrt(2.0*rho*neta)

	  Md(2,1)=Ci*eta*epsa
	  Md(2,2)=Ci*p*epsb
	  Md(1,1)=Ci*p*epsa
	  Md(1,2)=-Ci*neta*epsb

	  Nd(2,1)=-Ci*(twomup*p-rho)*epsa
	  Nd(1,2)=Ci*(twomup*p-rho)*epsb
	  Nd(1,1)=Ci*twomup*eta*epsa
	  Nd(2,2)=Ci*twomup*neta*epsb

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
      Subroutine setInc(Nl,Xs,is,ir,w,p,h,rho,alpha,beta,V,Resp0)
	 
*     ***********************************************************
*     sweep-Free surface case
*     ***********************************************************

	  Implicit None
	  Integer  Nl,is,ir
      real     rho(0:Nl),alpha(0:Nl),beta(0:Nl),Xs,w,p,h(0:Nl)
	  complex  V(Nl),Resp0

	  Integer  NLM,NL1,io,io1
	  Parameter(NLM=2,NL1=NLM+1)
	  Integer il,jl,Nlm1,Nlp1,ilm1,ilp1
	  complex InvDet,a,b,c,d,Wt11,Wt12,Wt21,Wt22
	  complex t11,t12,t21,t22
	  complex RRt11,RRt21,RRt12,RRt22,RRb11,RRb21,RRb12,RRb22
	  complex eta(0:NL1),neta(0:NL1),ea(0:NLM),eb(0:NLM)
	  complex RRd11(0:NL1),RRd12(0:NL1),RRd21(0:NL1),RRd22(0:NL1)
	  complex RRu11(0:NL1),RRu12(0:NL1),RRu21(0:NL1),RRu22(0:NL1)
	  complex TTu11(0:NL1),TTu12(0:NL1),TTu21(0:NL1),TTu22(0:NL1)
	  complex TTd11(0:NL1),TTd12(0:NL1),TTd21(0:NL1),TTd22(0:NL1)
	  complex Tu11(0:NLM),Tu12(0:NLM),Tu21(0:NLM),Tu22(0:NLM)
	  complex Td11(0:NLM),Td12(0:NLM),Td21(0:NLM),Td22(0:NLM)
	  complex Rd11(0:NLM),Rd12(0:NLM),Rd21(0:NLM),Rd22(0:NLM)
	  complex Ru11(0:NLM),Ru12(0:NLM),Ru21(0:NLM),Ru22(0:NLM)
      complex S1,S2,RF11,RF12,RF21,RF22,VERTICALSLO
	  complex Md(2,2),Mu(2,2),Nd(2,2),Nu(2,2)
	  complex Vu11,Vu21,Vd11,Vd21
      complex R11,R12,R21,R22,C1,Ci,temp,CBeta1,CBeta2


      If(NL.gt.NLM) then
	    STOP 'NL.gt.NLM in setIncSc'
	  Endif
	  C1=Cmplx(1.0,0.0)
	  Ci=Cmplx(0.0,1.0)
      Nlp1=Nl+1
	  do il=0,Nlp1
	     eta(il)=VERTICALSLO(alpha(il),P)
	     neta(il)=VERTICALSLO(beta(il),P)
      end do !il
      Do il=0,Nl
	    Temp=CI*w*h(il)*eta(il)
        EA(il)=CEXP(Temp)
	    Temp=CI*w*h(il)*neta(il)
        EB(il)=CEXP(Temp)		
*		Type*,'il,ea(il)=',il,ea(il)
*		Type*,'il,eb(il)=',il,eb(il)
      End Do

*     set reference medium interfacial scattering arrays
	  do il=0,Nl
	    ilp1=il+1
	  	cbeta1=cmplx(beta(il),0.0)
	  	cbeta2=cmplx(beta(ilp1),0.0)
*       set the reference medium interfacial scattering matrices
		call ScatMat(p,eta(il),neta(il),rho(il),CBeta1,eta(ilp1),
     &     neta(ilp1),rho(ilp1),CBeta2,Rd11(il),Rd12(il),Rd21(il),
     &     Rd22(il),Tu11(il),Tu12(il),Tu21(il),Tu22(il),Td11(il),
     &     Td12(il),Td21(il),Td22(il),Ru11(il),Ru12(il),Ru21(il),
     &     Ru22(il))
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

******************************************************
*     free-surface interaction
******************************************************

	  call Emat(P,eta(0),neta(0),rho(0),beta(0),Md,Nd,Mu,Nu)
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
*	  Type*,'Rf11=',Rf11
*	  Type*,'Rf12=',Rf12
*	  Type*,'Rf21=',Rf21
*	  Type*,'Rf22=',Rf22

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

        RRu11(ilp1)=Ru11(il)+RRb22*TTu21(il)*Td12(il)+
     &            RRb21*TTu11(il)*Td12(il)+RRb12*TTu21(il)
     &            *Td11(il)+RRb11*TTu11(il)*Td11(il)

        RRu12(ilp1)=Ru12(il)+Td11(il)*RRb11*TTu12(il)+Td11(il)*
     &            RRb12*TTu22(il)+Td12(il)*RRb21*TTu12(il)+Td12(il)*
     &			  RRb22*TTu22(il)

        RRu21(ilp1)=Ru21(il)+RRb22*TTu21(il)*Td22(il)+RRb21*
     &            TTu11(il)*Td22(il)+RRb12*TTu21(il)*Td21(il)+
     &            RRb11*TTu11(il)*Td21(il)

        RRu22(ilp1)=Ru22(il)+RRb22*TTu22(il)*Td22(il)+RRb21*
     &            TTu12(il)*Td22(il)+RRb12*TTu22(il)*Td21(il)+
     &            RRb11*TTu12(il)*Td21(il)
*	    Type*,'ilp1,RRu11(ilp1),RRu21(ilp1),RRu22(ilp1)='
*	    Type*,ilp1,RRu11(ilp1),RRu21(ilp1),RRu22(ilp1)
      End Do
	  Nlm1=Nl-1
*****************************************************
*     set Vu(Nl)=TTu(NL)*Sig
*****************************************************
	  call Emat(P,eta(Nl+1),neta(Nl+1),rho(Nl+1),beta(Nl+1),
     &     Md,Nd,Mu,Nu)
      temp=Cexp(-Ci*w*p*Xs)/(rho(Nl+1)*beta(Nl+1)*beta(Nl+1))
      If(is.eq.2) then
*	    S1=temp*Nd(1,1)
*	    S2=temp*Nd(2,1)
        S1=(1.0,0.0) !Up P-wave
        S2=(0.0,0.0)
	  else
	    Pause 'is out of bounds'
	  Endif
	  
	  io=4*(Nl-1)
	  V(io+3)=TTu11(Nl)*S1+TTu12(Nl)*S2
	  V(io+4)=TTu21(Nl)*S1+TTu22(Nl)*S2
	  
      V(io+1)=RRu11(Nl)*V(io+3)*ea(Nl)+RRu12(Nl)*V(io+4)*eb(Nl)
      V(io+2)=RRu21(Nl)*V(io+3)*ea(Nl)+RRu22(Nl)*V(io+4)*eb(Nl)
*      Type*,'io+1,V(io+1)=',io+1,V(io+1)
*      Type*,'io+2,V(io+2)=',io+2,V(io+2)
*      Type*,'io+3,V(io+3)=',io+3,V(io+3)
*      Type*,'io+4,V(io+4)=',io+4,V(io+4)
	  Do il=Nl-1,1,-1
	    ilp1=il+1
	    io=4*(il-1)
	    io1=4*il

		V(io+3)=TTu11(il)*V(io1+3)*ea(ilp1)+TTu12(il)*
     &          V(io1+4)*eb(ilp1)
		V(io+4)=TTu21(il)*V(io1+3)*ea(ilp1)+TTu22(il)*
     &          V(io1+4)*eb(ilp1)

		V(io+1)=RRu11(il)*V(io+3)*ea(il)+RRu12(il)*V(io+4)*eb(il)
        V(io+2)=RRu21(il)*V(io+3)*ea(il)+RRu22(il)*V(io+4)*eb(il)
*      Type*,'io+1,V(io+1)=',io+1,V(io+1)
*      Type*,'io+2,V(io+2)=',io+2,V(io+2)
*      Type*,'io+3,V(io+3)=',io+3,V(io+3)
*      Type*,'io+4,V(io+4)=',io+4,V(io+4)

      End Do

	  Vu11=TTu11(0)*V(3)*ea(1)+TTu12(0)*V(4)*eb(1)
	  Vu21=TTu21(0)*V(3)*ea(1)+TTu22(0)*V(4)*eb(1)

	  Vd11=RRu11(0)*V(3)*ea(1)+RRu12(0)*V(4)*eb(1)
      Vd21=RRu21(0)*V(3)*ea(1)+RRu22(0)*V(4)*eb(1)

*     set the free surface displacement
*     Uv(0+)=Mu(0)*E*Vu(0)+Md(0)*Vd(0)
	  call Emat(P,eta(0),neta(0),rho(0),beta(0),Md,Nd,Mu,Nu)
      If(ir.eq.1) then
	    Resp0=Mu(1,1)*ea(0)*Vu11+Mu(1,2)*eb(0)*Vu21+
     &       Md(1,1)*Vd11+Md(1,2)*Vd21
	  elseif(ir.eq.2) then
	    Resp0=Mu(2,1)*ea(0)*Vu11+Mu(2,2)*eb(0)*Vu21+
     &       Md(2,1)*Vd11+Md(2,2)*Vd21
	  Else
	    Pause 'ir out of bounds in setInc'
	  Endif
      return
	  End
	  
********************************
      Subroutine SetS(il,Nl,w,p,h,rho0,alpha0,beta0,
     &           rho,alpha,beta,S)
	  Implicit None
	  Integer il,Nl,i,j
      real     rho0(0:Nl),alpha0(0:Nl),beta0(0:Nl),h(0:Nl)
      real     rho(0:Nl),alpha(0:Nl),beta(0:Nl),w,p
	  complex  S(4,4)

	  complex InvDet,a,b,c,d,C1,Ci
	  complex RRt11,RRt21,RRt12,RRt22
	  complex RRb11,RRb21,RRb12,RRb22
	  complex ea,eb,ea0,eb0,eta0,neta0,eta,neta
	  complex RRd11,RRd12,RRd21,RRd22
	  complex TTd11,TTd12,TTd21,TTd22
	  complex Tu11t,Tu12t,Tu21t,Tu22t
	  complex Td11t,Td12t,Td21t,Td22t
	  complex Rd11t,Rd12t,Rd21t,Rd22t
      complex Ru11t,Ru12t,Ru21t,Ru22t
	  complex Tu11b,Tu12b,Tu21b,Tu22b
	  complex Td11b,Td12b,Td21b,Td22b
	  complex Rd11b,Rd12b,Rd21b,Rd22b
      complex Ru11b,Ru12b,Ru21b,Ru22b
	  complex VERTICALSLO,cbeta1,cbeta2

	  C1=Cmplx(1.0,0.0)
	  Ci=(0.0,1.0)
      
	  eta0=VERTICALSLO(alpha0(il),P)
	  neta0=VERTICALSLO(beta0(il),P)

	  eta=VERTICALSLO(alpha(il),P)
	  neta=VERTICALSLO(beta(il),P)

      Ea0=CEXP(CI*w*h(il)*eta0)
      Eb0=CEXP(CI*w*h(il)*neta0)

      Ea=CEXP(CI*w*h(il)*eta)
      Eb=CEXP(CI*w*h(il)*neta)

****************************************************************
*     set top interface
		cbeta1=cmplx(beta0(il),0.0)
		cbeta2=cmplx(beta(il),0.0)

	    call ScatMat(p,eta0,neta0,rho0(il),cbeta1,eta,
     &  neta,rho(il),cbeta2,Rd11t,Rd12t,Rd21t,Rd22t,
     &  Tu11t,Tu12t,Tu21t,Tu22t,Td11t,Td12t,
     &  Td21t,Td22t,Ru11t,Ru12t,Ru21t,Ru22t)

****************************************************************
*     set bottom interface

		cbeta1=cmplx(beta(il),0.0)
		cbeta2=cmplx(beta0(il),0.0)

	    call ScatMat(p,eta,neta,rho(il),cbeta1,eta0,
     &  neta0,rho0(il),cbeta2,Rd11b,Rd12b,Rd21b,
     &  Rd22b,Tu11b,Tu12b,Tu21b,Tu22b,Td11b,Td12b,
     &  Td21b,Td22b,Ru11b,Ru12b,Ru21b,Ru22b)

****************************************************************
	  
	  RRd11=Rd11b
	  RRd12=Rd12b
	  RRd21=Rd21b
	  RRd22=Rd22b
	  
	  RRt11=RRd11*ea*ea
	  RRt12=RRd12*ea*eb
	  RRt21=RRd21*ea*eb
	  RRt22=RRd22*eb*eb

	  a=(C1-RRt22*Ru22t-RRt12*Ru21t)
	  b=(RRt22*Ru12t+RRt12*Ru11t)	  
	  c=(RRt21*Ru22t+RRt11*Ru21t)
	  d=(C1-RRt21*Ru12t-RRt11*Ru11t)
	  InvDet=C1/(a*d-b*c)
	  
	  a=InvDet*a
	  b=InvDet*b
	  c=InvDet*c
	  d=InvDet*d
      
	  TTd11=Td11t*a+Td21t*b
	  TTd12=Td12t*a+Td22t*b
	  TTd21=Td11t*c+Td21t*d
	  TTd22=Td12t*c+Td22t*d
	  
      
	  S(1,1)=Rd11t+RRt22*TTd21*Tu12t+RRt21*TTd11*Tu12t+
     &          RRt12*TTd21*Tu11t+RRt11*TTd11*Tu11t

      S(2,1)=Rd21t+RRt22*TTd21*Tu22t+RRt21*TTd11*Tu22t+
     &          RRt12*TTd21*Tu21t+RRt11*TTd11*Tu21t

      S(2,2)=Rd22t+RRt22*TTd22*Tu22t+RRt21*TTd12*Tu22t+
     &          RRt12*TTd22*Tu21t+RRt11*TTd12*Tu21t


      S(3,1)=eb*TTd21*Td12b+ea*TTd11*Td11b-ea0

      S(4,1)=eb*TTd21*Td22b+ea*TTd11*Td21b

      S(4,2)=eb*TTd22*Td22b+ea*TTd12*Td21b-eb0
      
	  S(1,2)=-S(2,1)
	  S(1,3)=S(3,1)
	  S(1,4)=-S(4,1)
	  S(2,3)=-S(4,1)
	  S(2,4)=S(4,2)
	  S(3,2)=S(4,1)
	  S(3,3)=S(1,1)
	  S(3,4)=S(2,1)
	  S(4,3)=-S(2,1)
	  S(4,4)=S(2,2)
	  
*	  Do i=1,4
*	    Do j=1,4
*		  Type*,'i,j,S(i,j)=',i,j,S(i,j)
*		Enddo
*	  Enddo
	  return
	  end

*******************************************************
      subroutine zaxpy(n,za,zx,incx,zy,incy)
c
c     constant times a vector plus a vector.
c     jack dongarra, 3/11/78.
c     modified 12/3/93, array(1) declarations changed to array(*)
c
      complex zx(*),zy(*),za
      integer i,incx,incy,ix,iy,n
      real dcabs1
      if(n.le.0)return
      if (dcabs1(za) .eq. 0.0d0) return
      if (incx.eq.1.and.incy.eq.1)go to 20
c
c        code for unequal increments or equal increments
c          not equal to 1
c
      ix = 1
      iy = 1
      if(incx.lt.0)ix = (-n+1)*incx + 1
      if(incy.lt.0)iy = (-n+1)*incy + 1
      do 10 i = 1,n
        zy(iy) = zy(iy) + za*zx(ix)
        ix = ix + incx
        iy = iy + incy
   10 continue
      return
c
c        code for both increments equal to 1
c
   20 do 30 i = 1,n
        zy(i) = zy(i) + za*zx(i)
   30 continue
      return
      end
****************************************************

      subroutine  zscal(n,za,zx,incx)
c
c     scales a vector by a constant.
c     jack dongarra, 3/11/78.
c     modified 3/93 to return if incx .le. 0.
c     modified 12/3/93, array(1) declarations changed to array(*)
c
      complex za,zx(*)
      integer i,incx,ix,n
c
      if( n.le.0 .or. incx.le.0 )return
      if(incx.eq.1)go to 20
c
c        code for increment not equal to 1
c
      ix = 1
      do 10 i = 1,n
        zx(ix) = za*zx(ix)
        ix = ix + incx
   10 continue
      return
c
c        code for increment equal to 1
c
   20 do 30 i = 1,n
        zx(i) = za*zx(i)
   30 continue
      return
      end
*******************************************************
      real function dcabs1(z)
       complex z,zz
      real t(2)
      equivalence (zz,t(1))
      zz = z
      dcabs1 = dabs(t(1)) + dabs(t(2))
      return
      end

      complex function zdotu(n,zx,incx,zy,incy)
c
c     forms the dot product of two vectors.
c     jack dongarra, 3/11/78.
c     modified 12/3/93, array(1) declarations changed to array(*)
c
      complex zx(*),zy(*),ztemp
      integer i,incx,incy,ix,iy,n
      ztemp = (0.0,0.0)
      zdotu = (0.0,0.0)
      if(n.le.0)return
      if(incx.eq.1.and.incy.eq.1)go to 20
c
c        code for unequal increments or equal increments
c          not equal to 1
c
      ix = 1
      iy = 1
      if(incx.lt.0)ix = (-n+1)*incx + 1
      if(incy.lt.0)iy = (-n+1)*incy + 1
      do 10 i = 1,n
        ztemp = ztemp + zx(ix)*zy(iy)
        ix = ix + incx
        iy = iy + incy
   10 continue
      zdotu = ztemp
      return
c
c        code for both increments equal to 1
c
   20 do 30 i = 1,n
        ztemp = ztemp + zx(i)*zy(i)
   30 continue
      zdotu = ztemp
      return
      end

*******************************************************
*****************************************************

      SUBROUTINE ZGEMV ( TRANS, M, N, ALPHA, A, LDA, X, INCX,
     $                   BETA, Y, INCY )
*     .. Scalar Arguments ..
      Complex         ALPHA, BETA
      INTEGER            INCX, INCY, LDA, M, N
      CHARACTER*1        TRANS
*     .. Array Arguments ..
      Complex         A( LDA, * ), X( * ), Y( * )
*     ..
*
*  Purpose
*  =======
*
*  ZGEMV  performs one of the matrix-vector operations
*
*     y := alpha*A*x + beta*y,   or   y := alpha*A'*x + beta*y,   or
*
*     y := alpha*conjg( A' )*x + beta*y,
*
*  where alpha and beta are scalars, x and y are vectors and A is an
*  m by n matrix.
*
*  Parameters
*  ==========
*
*  TRANS  - CHARACTER*1.
*           On entry, TRANS specifies the operation to be performed as
*           follows:
*
*              TRANS = 'N' or 'n'   y := alpha*A*x + beta*y.
*
*              TRANS = 'T' or 't'   y := alpha*A'*x + beta*y.
*
*              TRANS = 'C' or 'c'   y := alpha*conjg( A' )*x + beta*y.
*
*           Unchanged on exit.
*
*  M      - INTEGER.
*           On entry, M specifies the number of rows of the matrix A.
*           M must be at least zero.
*           Unchanged on exit.
*
*  N      - INTEGER.
*           On entry, N specifies the number of columns of the matrix A.
*           N must be at least zero.
*           Unchanged on exit.
*
*  ALPHA  - Complex      .
*           On entry, ALPHA specifies the scalar alpha.
*           Unchanged on exit.
*
*  A      - Complex       array of DIMENSION ( LDA, n ).
*           Before entry, the leading m by n part of the array A must
*           contain the matrix of coefficients.
*           Unchanged on exit.
*
*  LDA    - INTEGER.
*           On entry, LDA specifies the first dimension of A as declared
*           in the calling (sub) program. LDA must be at least
*           max( 1, m ).
*           Unchanged on exit.
*
*  X      - Complex       array of DIMENSION at least
*           ( 1 + ( n - 1 )*abs( INCX ) ) when TRANS = 'N' or 'n'
*           and at least
*           ( 1 + ( m - 1 )*abs( INCX ) ) otherwise.
*           Before entry, the incremented array X must contain the
*           vector x.
*           Unchanged on exit.
*
*  INCX   - INTEGER.
*           On entry, INCX specifies the increment for the elements of
*           X. INCX must not be zero.
*           Unchanged on exit.
*
*  BETA   - Complex      .
*           On entry, BETA specifies the scalar beta. When BETA is
*           supplied as zero then Y need not be set on input.
*           Unchanged on exit.
*
*  Y      - Complex       array of DIMENSION at least
*           ( 1 + ( m - 1 )*abs( INCY ) ) when TRANS = 'N' or 'n'
*           and at least
*           ( 1 + ( n - 1 )*abs( INCY ) ) otherwise.
*           Before entry with BETA non-zero, the incremented array Y
*           must contain the vector y. On exit, Y is overwritten by the
*           updated vector y.
*
*  INCY   - INTEGER.
*           On entry, INCY specifies the increment for the elements of
*           Y. INCY must not be zero.
*           Unchanged on exit.
*
*
*  Level 2 Blas routine.
*
*  -- Written on 22-October-1986.
*     Jack Dongarra, Argonne National Lab.
*     Jeremy Du Croz, Nag Central Office.
*     Sven Hammarling, Nag Central Office.
*     Richard Hanson, Sandia National Labs.
*
*
*     .. Parameters ..
      Complex          ONE
      PARAMETER        ( ONE  = ( 1.00, 0.00 ) )
      Complex          ZERO
      PARAMETER        ( ZERO = ( 0.00, 0.00 ) )
*     .. Local Scalars ..
      Complex         TEMP
      INTEGER            I, INFO, IX, IY, J, JX, JY, KX, KY, LENX, LENY
      LOGICAL            NOCONJ
*     .. External Functions ..
      LOGICAL            LSAME
      EXTERNAL           LSAME
*     .. External Subroutines ..
      EXTERNAL           XERBLA
*     .. Intrinsic Functions ..
      INTRINSIC          DCONJG, MAX
*     ..
*     .. Executable Statements ..
*
*     Test the input parameters.
*
      INFO = 0
      IF     ( .NOT.LSAME( TRANS, 'N' ).AND.
     $         .NOT.LSAME( TRANS, 'T' ).AND.
     $         .NOT.LSAME( TRANS, 'C' )      )THEN
         INFO = 1
      ELSE IF( M.LT.0 )THEN
         INFO = 2
      ELSE IF( N.LT.0 )THEN
         INFO = 3
      ELSE IF( LDA.LT.MAX( 1, M ) )THEN
         INFO = 6
      ELSE IF( INCX.EQ.0 )THEN
         INFO = 8
      ELSE IF( INCY.EQ.0 )THEN
         INFO = 11
      END IF
      IF( INFO.NE.0 )THEN
         CALL XERBLA( 'ZGEMV ', INFO )
         RETURN
      END IF
*
*     Quick return if possible.
*
      IF( ( M.EQ.0 ).OR.( N.EQ.0 ).OR.
     $    ( ( ALPHA.EQ.ZERO ).AND.( BETA.EQ.ONE ) ) )
     $   RETURN
*
      NOCONJ = LSAME( TRANS, 'T' )
*
*     Set  LENX  and  LENY, the lengths of the vectors x and y, and set
*     up the start points in  X  and  Y.
*
      IF( LSAME( TRANS, 'N' ) )THEN
         LENX = N
         LENY = M
      ELSE
         LENX = M
         LENY = N
      END IF
      IF( INCX.GT.0 )THEN
         KX = 1
      ELSE
         KX = 1 - ( LENX - 1 )*INCX
      END IF
      IF( INCY.GT.0 )THEN
         KY = 1
      ELSE
         KY = 1 - ( LENY - 1 )*INCY
      END IF
*
*     Start the operations. In this version the elements of A are
*     accessed sequentially with one pass through A.
*
*     First form  y := beta*y.
*
      IF( BETA.NE.ONE )THEN
         IF( INCY.EQ.1 )THEN
            IF( BETA.EQ.ZERO )THEN
               DO 10, I = 1, LENY
                  Y( I ) = ZERO
   10          CONTINUE
            ELSE
               DO 20, I = 1, LENY
                  Y( I ) = BETA*Y( I )
   20          CONTINUE
            END IF
         ELSE
            IY = KY
            IF( BETA.EQ.ZERO )THEN
               DO 30, I = 1, LENY
                  Y( IY ) = ZERO
                  IY      = IY   + INCY
   30          CONTINUE
            ELSE
               DO 40, I = 1, LENY
                  Y( IY ) = BETA*Y( IY )
                  IY      = IY           + INCY
   40          CONTINUE
            END IF
         END IF
      END IF
      IF( ALPHA.EQ.ZERO )
     $   RETURN
      IF( LSAME( TRANS, 'N' ) )THEN
*
*        Form  y := alpha*A*x + y.
*
         JX = KX
         IF( INCY.EQ.1 )THEN
            DO 60, J = 1, N
               IF( X( JX ).NE.ZERO )THEN
                  TEMP = ALPHA*X( JX )
                  DO 50, I = 1, M
                     Y( I ) = Y( I ) + TEMP*A( I, J )
   50             CONTINUE
               END IF
               JX = JX + INCX
   60       CONTINUE
         ELSE
            DO 80, J = 1, N
               IF( X( JX ).NE.ZERO )THEN
                  TEMP = ALPHA*X( JX )
                  IY   = KY
                  DO 70, I = 1, M
                     Y( IY ) = Y( IY ) + TEMP*A( I, J )
                     IY      = IY      + INCY
   70             CONTINUE
               END IF
               JX = JX + INCX
   80       CONTINUE
         END IF
      ELSE
*
*        Form  y := alpha*A'*x + y  or  y := alpha*conjg( A' )*x + y.
*
         JY = KY
         IF( INCX.EQ.1 )THEN
            DO 110, J = 1, N
               TEMP = ZERO
               IF( NOCONJ )THEN
                  DO 90, I = 1, M
                     TEMP = TEMP + A( I, J )*X( I )
   90             CONTINUE
               ELSE
                  DO 100, I = 1, M
                     TEMP = TEMP + CONJG( A( I, J ) )*X( I )
  100             CONTINUE
               END IF
               Y( JY ) = Y( JY ) + ALPHA*TEMP
               JY      = JY      + INCY
  110       CONTINUE
         ELSE
            DO 140, J = 1, N
               TEMP = ZERO
               IX   = KX
               IF( NOCONJ )THEN
                  DO 120, I = 1, M
                     TEMP = TEMP + A( I, J )*X( IX )
                     IX   = IX   + INCX
  120             CONTINUE
               ELSE
                  DO 130, I = 1, M
                     TEMP = TEMP + CONJG( A( I, J ) )*X( IX )
                     IX   = IX   + INCX
  130             CONTINUE
               END IF
               Y( JY ) = Y( JY ) + ALPHA*TEMP
               JY      = JY      + INCY
  140       CONTINUE
         END IF
      END IF
*
      RETURN
*
*     End of ZGEMV .
*
      END
**************************************************
*************************************************
      LOGICAL          FUNCTION LSAME( CA, CB )
*
*  -- LAPACK auxiliary routine (version 2.0) --
*     Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
*     Courant Institute, Argonne National Lab, and Rice University
*     January 31, 1994
*
*     .. Scalar Arguments ..
      CHARACTER          CA, CB
*     ..
*
*  Purpose
*  =======
*
*  LSAME returns .TRUE. if CA is the same letter as CB regardless of
*  case.
*
*  Arguments
*  =========
*
*  CA      (input) CHARACTER*1
*  CB      (input) CHARACTER*1
*          CA and CB specify the single characters to be compared.
*
* =====================================================================
*
*     .. Intrinsic Functions ..
      INTRINSIC          ICHAR
*     ..
*     .. Local Scalars ..
      INTEGER            INTA, INTB, ZCODE
*     ..
*     .. Executable Statements ..
*
*     Test if the characters are equal
*
      LSAME = CA.EQ.CB
      IF( LSAME )
     $   RETURN
*
*     Now test for equivalence if both characters are alphabetic.
*
      ZCODE = ICHAR( 'Z' )
*
*     Use 'Z' rather than 'A' so that ASCII can be detected on Prime
*     machines, on which ICHAR returns a value with bit 8 set.
*     ICHAR('A') on Prime machines returns 193 which is the same as
*     ICHAR('A') on an EBCDIC machine.
*
      INTA = ICHAR( CA )
      INTB = ICHAR( CB )
*
      IF( ZCODE.EQ.90 .OR. ZCODE.EQ.122 ) THEN
*
*        ASCII is assumed - ZCODE is the ASCII code of either lower or
*        upper case 'Z'.
*
         IF( INTA.GE.97 .AND. INTA.LE.122 ) INTA = INTA - 32
         IF( INTB.GE.97 .AND. INTB.LE.122 ) INTB = INTB - 32
*
      ELSE IF( ZCODE.EQ.233 .OR. ZCODE.EQ.169 ) THEN
*
*        EBCDIC is assumed - ZCODE is the EBCDIC code of either lower or
*        upper case 'Z'.
*
         IF( INTA.GE.129 .AND. INTA.LE.137 .OR.
     $       INTA.GE.145 .AND. INTA.LE.153 .OR.
     $       INTA.GE.162 .AND. INTA.LE.169 ) INTA = INTA + 64
         IF( INTB.GE.129 .AND. INTB.LE.137 .OR.
     $       INTB.GE.145 .AND. INTB.LE.153 .OR.
     $       INTB.GE.162 .AND. INTB.LE.169 ) INTB = INTB + 64
*
      ELSE IF( ZCODE.EQ.218 .OR. ZCODE.EQ.250 ) THEN
*
*        ASCII is assumed, on Prime machines - ZCODE is the ASCII code
*        plus 128 of either lower or upper case 'Z'.
*
         IF( INTA.GE.225 .AND. INTA.LE.250 ) INTA = INTA - 32
         IF( INTB.GE.225 .AND. INTB.LE.250 ) INTB = INTB - 32
      END IF
      LSAME = INTA.EQ.INTB
*
*     RETURN
*
*     End of LSAME
*
      END
****************************************************

      SUBROUTINE XERBLA( SRNAME, INFO )
*
*  -- LAPACK auxiliary routine (preliminary version) --
*     Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
*     Courant Institute, Argonne National Lab, and Rice University
*     February 29, 1992
*
*     .. Scalar Arguments ..
      CHARACTER*6        SRNAME
      INTEGER            INFO
*     ..
*
*  Purpose
*  =======
*
*  XERBLA  is an error handler for the LAPACK routines.
*  It is called by an LAPACK routine if an input parameter has an
*  invalid value.  A message is printed and execution stops.
*
*  Installers may consider modifying the STOP statement in order to
*  call system-specific exception-handling facilities.
*
*  Arguments
*  =========
*
*  SRNAME  (input) CHARACTER*6
*          The name of the routine which called XERBLA.
*
*  INFO    (input) INTEGER
*          The position of the invalid parameter in the parameter list
*          of the calling routine.
*
*
      WRITE( *, FMT = 9999 )SRNAME, INFO
*
      STOP
*
 9999 FORMAT( ' ** On entry to ', A6, ' parameter number ', I2, ' had ',
     $      'an illegal value' )
*
*     End of XERBLA
*
      END
*******************************************************
