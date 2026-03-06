      Program  GMRES_P

      Implicit none
	  Integer  Nl,Np,Mp,N,M,N1,il,ilp1,i,j,k,l,ip,ioff,itype,iter,dim
	  Parameter(Np=1) !must be odd
	  Parameter(Nl=2,N1=Nl+1,N=4*Nl*Np,Mp=66,M=Mp-1)
      real     rho0(0:N1),alpha0(0:N1),beta0(0:N1)
      real     rho(0:N1),alpha(0:N1),beta(0:N1)
      real     drho(Nl),dalpha(Nl),dbeta(Nl)
	  real     L2,Tol,L2r0,w,p(Np),h(0:Nl),Xs,Xr,pmax,dp,rmax,rt3
	  Parameter(Tol=1.E-5)
	  Integer  is  !is=1=>Exp; is=2=>Doub
	  Integer  ir  !ir=1=>X-disp; ir=2=>Z-disp
	  complex  X0(N),X1(N),R(N),R0(N),b0(N),V(N),X(N),Resp0(Np)
	  complex  Pmat(N,Mp),c(Mp),Wvec(N),ek(N)
	  complex  Rmat(M,M),Jmat(2,M),Vexact(N),sum
	  integer  outfile,Maxit,irst
	  parameter(outfile=7)
	  logical  adj,sing

*******************************************************
	  Complex  Idvec(N),Amat(N,N),Bmat(N,N),GMAT(N,N),C0,C1 !DELETE
	  INTEGER  IPIV(N),INDXR(N),INDXC(N)

******************************************************

*	  DATA     alpha0/1.51,2.01,3.01,4.01/
*	  DATA     beta0/1.01,1.51,2.51,3.51/	
*      DATA     rho0/1.01,1.51,2.51,3.51/

	  DATA     beta0/1.01,1.01,1.01,1.01/	
      DATA     rho0/1.01,1.01,1.01,1.01/


      DATA     drho/2*0.1/
	  DATA     dalpha/2*0.0/
	  DATA     dbeta/2*0.1/
	  DATA     W/1.0/
	  DATA     h/3*1.0/
	  Data     pmax/.1/

      DATA     Xs,Xr/0.0,10.0/
      DATA     is,ir/2,2/
	  Data     adj/.false./
	  
	  C0=(0.0,0.0)
	  C1=(1.0,0.0)
      if(Np.gt.1)then
        dp=2.0*Pmax/float(Np-1)
	    Do ip=1,Np
	      P(ip)=-pmax+(ip-1)*dp
		  Type*,'ip,p(ip)=',ip,p(ip)
	    Enddo
		pause
	  else
	    TYpe*,'input a single p value'
	    Accept*,p
	  Endif

      rt3=sqrt(3.0)
	  rho(0)=rho0(0)
	  rho(NL+1)=rho0(NL+1)

	  alpha0(0)=rt3*beta0(0)
	  alpha0(NL+1)=rt3*beta0(NL+1)

	  alpha(0)=alpha0(0)
	  alpha(NL+1)=alpha0(NL+1)

	  beta(0)=beta0(0)
	  beta(NL+1)=beta0(NL+1)


	  Do il=1,Nl
		alpha0(il)=rt3*beta0(il)
		alpha(il)=alpha0(il)+sqrt(3.0)*dbeta(il)
		beta(il)=beta0(il)+dbeta(il)
		rho(il)=rho0(il)+drho(il)
	  Enddo

      
	  Do il=0,Nl+1
		TYpe*,'alpha0(',il,')=Cmplx(',alpha0(il)
		TYpe*,'beta0(',il,')=Cmplx(',beta0(il)
		TYpe*,'rho0(',il,')=',rho0(il)
	  Enddo

	  Do il=0,Nl+1
		TYpe*,'alpha(',il,')=Cmplx(',alpha(il)
		TYpe*,'beta(',il,')=Cmplx(',beta(il)
		TYpe*,'rho(',il,')=',rho(il)
	  Enddo
	  pause

*********************************************
*     set b0, the reference medium field
*********************************************

	  Do ip=1,Np
        call setInc(Nl,Xs,is,ir,w,p(ip),h,rho0,alpha0,beta0,X,Resp0(ip))
		Do il=1,Nl
		  ioff=4*(il-1)
		  Do itype=1,4
		    b0(4*Np*(il-1)+Np*(itype-1)+ip)=X(ioff+itype)
		  Enddo !itype
		Enddo !il
*        Type*,'Resp0=',Resp0(ip)
	  Enddo !ip
      TYpe*,'incident field='
	  Do i=1,N
	    Type*,'b0(',i,')=',b0(i) !DELETE
	  Enddo

*********************************************
*     generate exact solution
**********************************************
*      go to 111
*     set Amat(N,N),
      Do j=1,N
	    Do i=1,N
	     Idvec(i)=(0.0,0.0)
	    Enddo !i
	    Idvec(j)=(1.0,0.0)
        call atimes(Nl,Np,Adj,w,p,h,rho0,alpha0,beta0,
     &             rho,alpha,beta,Idvec,Amat(1,j))
	  Enddo !j


	  Do i=1,N
		  Vexact(i)=B0(i)
	  Enddo !i
      
	  call CMPLXGJ(Amat,N,N,Vexact,1,1,IPIV,INDXR,INDXC,N)

	  Do i=1,N
		  Type*,'i,Vexact(i)=',i,Vexact(i)
	  Enddo !i
	  
*111    continue

**********************************
*      open(outfile,file='GMRESDat',status='old')
	  
*	  Do i=1,4*Nl*Np
*	    Read(outfile,*), Vexact(i)
*	    Type*,'Vexact(i)=',i,Vexact(i)
*	  Enddo
*	  close (outfile)


********************************************************
*     set initial approx X0
	  Do i=1,N
*	    X(i)=(0.0,0.0)
	    X(i)=b0(i)

*		X(i)=Vexact(i)   !DELETE
	  Enddo

**********************************
      irst=0
007   continue
*     set initial residual r0

	  Do i=1,N
		TYpE*,'check this X(i)=',i,X(i)
      enddo

      call atimes(Nl,Np,Adj,w,p,h,rho0,alpha0,beta0,
     &             rho,alpha,beta,X,X1)

       Do i=1,M
	      type*,'X(i)=',i,X(i)
      enddo
	 
	  Do i=1,N
	    r0(i)=b0(i)-X1(i)
	  Enddo
	  
	  Do i=1,N
		TYpE*,'r0(i)=',i,r0(i)
      enddo
*     set r0=P1*r0 and store Housholder vector in Pmat(.,1)
      call setPkV(Pmat,r0,n,1,n,mp,c,sing)
*     Wvec=P1*r0
      Wvec(1)=r0(1)
      Do i=2,N
	    Wvec(i)=(0.0,0.0)
	  Enddo
*
*
*       ******************************************
*       set Col k of upper triangular Rmat(M,M)
*       ******************************************
*
      Maxit=min(M,N-1)
	  Type*,'Maxit=',Maxit
	  dim=Maxit
	  Do 100  k=1,Maxit
*       set unit vector ek
        Do 10 i=1,N
		  ek(i)=(0.0,0.0)
10      continue
		ek(k)=(1.0,0.0) 
*
*       ******************************************
*       set ek=P(1)*....P(k)*ek
*       ******************************************
*
       call QV(Pmat,n,k,n,c,ek)
*	   Do i=1,N
*	     Type*,'i,ek(i)=',i,ek(i)
*	   Enddo
*
*       ******************************************
*       set r=(1-PdS)*ek
*       ******************************************
*
        call atimes(Nl,Np,Adj,w,p,h,rho0,alpha0,
     &              beta0,rho,alpha,beta,ek,r)
	   Do i=1,N
	     Type*,'i,r(i)=',i,r(i)
	   Enddo
	   

*
*       ******************************************
*       set r=P(k)*....*P(1)*r
*       ******************************************
*
        call QtV(Pmat,n,k,n,c,r)
*	   Do i=1,N
*	     Type*,'i,r(i)=',i,r(i)
*	   Enddo
		rmax=0.0
		do 11 i=k+1,n
            rmax=max(rmax,cabs(r(i)))
11      continue
        If(rmax.gt.0.0)then
		  call setPkV(Pmat,r,n,k+1,n,mp,c,sing)
		else !zero Householder vector
		  Type*,'rmax=0.0 case'
		  dim=k-1
		  go to 200 !restart
*		  Do 12 i= k+1,N         !recomment out
*		    Pmat(i,K+1)=(0.0,0.0)
*12        continue		
		Endif
        if(k.gt.1) then
*
*       ******************************************
*       set r=J(k-1)*....*J(1)*r
*       ******************************************
*
          Do 13 i=1,k-1
            call rotV(r,i,Jmat(1,i),Jmat(2,i))
13        continue
        Endif
		if(Cabs(r(k+1)).ne.0.0)then
		    Jmat(1,k)=r(k)
		    Jmat(2,k)=r(k+1)
		else
		    Jmat(1,k)=(1.0,0.0)
		    Jmat(2,k)=(0.0,0.0)
		Endif
        call rotV(r,k,Jmat(1,k),Jmat(2,k))
        call rotV(Wvec,k,Jmat(1,k),Jmat(2,k))
*
*       ******************************************
*       set Rmat(.,k)=[Rmat(.,k-1),r]
*       ******************************************
*
        Do 14 i=1,k
		  Rmat(i,k)=r(i)
14      continue
*
*       ******************************************
*       Test for convergence
*       ******************************************
*
        L2=Cabs(Wvec(K+1))
        Type*,'iter,irst,||r||=',k,irst,L2
        if(L2.lt.Tol)then
		  dim=k
		  go to 200
		endif

100   continue

200   continue
*
*       ******************************************
*       solve Wvec= Rmat(.,dim)*Wvec(dim)
*       ******************************************
*
      call rsolvb(Rmat,dim,M,Wvec)

	  Do 250 k=1,dim
*
*       set unit vector ek
        Do 210 i=1,N
		  ek(i)=(0.0,0.0)
210     continue
	    ek(k)=(1.0,0.0) 
*
*       ******************************************
*       set ek=P(1)*....P(k)*ek
*       ******************************************
*
        call QV(Pmat,n,k,n,c,ek)
	    Do 215 i=1,N
*	      Type*,'i,ek(i),Wvec(i)=',i,ek(i),Wvec(i)
	      X(i)=X(i)+ek(i)*Wvec(k)
215     continue

250   continue

      if(L2.lt.Tol)then
          Type*,'iterate,restarts,Solution:',dim,irst
          Do i=1,N
		        TYpe*, 'Cabs(Vexact(i)-X(i))/Cabs(Vexact(i))=',
     &          i,Cabs(Vexact(i)-X(i))/Cabs(Vexact(i))
		        TYpe*
	      Enddo !i
	  else !restart
	      irst=irst+1
		  goto 007
	  endif


      stop'normal completion'
	  end
*************************************
      subroutine rsolvb(Rmat,dim,M,Wvec)
*     Column oriented backsub
      implicit none
      integer  M,dim
      complex  Rmat(M,M),Wvec(*)
      integer  i,j
        do 12 j=dim,1,-1
          Wvec(j)=Wvec(j)/Rmat(j,j)
          do 11 i=1,j-1
            Wvec(i)=Wvec(i)-Rmat(i,j)*Wvec(j)
11        continue
12      continue
      return
      end
*************************************
      subroutine rotV(V,i,a11,a21)
      implicit none
      integer  n,np,i
      complex  a11,a21,V(*)
      real     fact,cabsa,cabsb,m
      complex  c,s,w,y
      cabsa=Cabs(a11)
      cabsb=Cabs(a21)
      m=sqrt(cabsa**2+cabsb**2)
      c=a11/m
      s=a21/m
	  y=V(i)
	  w=V(i+1)
	  V(i)=conjg(c)*y+conjg(s)*w
	  V(i+1)=c*w-s*y
      return
      end
***********************************

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
*      type*,'entered atimes'
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
	  Do i =1, 4*Nl*Np
	    Type*,'entering setSY, Y(i)=',i,Y(i)
	  Enddo

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
	  
*     set RF=C0 !!!!!!!!!!!!!!!!!DELETE
      Rf11=(0.0,0.0)
      Rf12=(0.0,0.0)
      Rf21=(0.0,0.0)
      Rf22=(0.0,0.0)

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
      type*,'*************************************************'
      Do il=0,Nl
	    Type*,'p,il,P11(il)=',p,il,P11(il)
	    Type*,'il,P21(il)=',il,P21(il)
	    Type*,'il,Q11(il)=',il,Q11(il)
	    Type*,'il,Q21(il)=',il,Q21(il)
	  Enddo
      type*,'*************************************************'
      
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
        type*,'P11(1)=',P11(1)
        type*,'R11(1)=',R11(1)
        type*,'X(1)=',X(1)
        type*,'R12(1)=',R12(1)
        type*,'X(2)=',X(2)
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
		type*,'il,P11,P21=',nl,P11(nl),P21(nl)
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

		type*,'il,Btwo1(il),Btwo2(il)=',il,Btwo1(il),Btwo2(il)
    
*	    P(il)=V+T(il)*W
        P11(il)=V11+T11(il)*W11+T12(il)*W21
		P21(il)=V21+T21(il)*W11+T22(il)*W21
		type*,'il,P11,P21=',il,P11(il),P21(il)
		type*,'***************************************************'
		type*,'il,T11(il)*W11+T12(il)*W21=',il,T11(il)*W11+T12(il)*W21
		type*,'il,T21(il)*W11+T22(il)*W21=',il,T21(il)*W11+T22(il)*W21
		type*,'***************************************************'

*       Q(il)=U(il)*W
        Q11(il)=U11(il)*W11+U12(il)*W21
		Q21(il)=U21(il)*W11+U22(il)*W21
		type*,'il,Q11,Q21=',il,Q11(il),Q21(il)
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
*            call setSLV(il,Nl,w,p(ipi),p(ipo),h,rho0,alpha0,
*     &         beta0,rho,alpha,beta,S)
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

	  
	  complex function couple(contrast,dk,x1,x2)
	  Implicit none
	  real     contrast,dk,x1,x2
	  complex  Ci,e1,e2
	  Ci=(0.0,1.0)
	  If(dk.lt.1E-10)then
	    couple=cmplx(contrast*(x2-x1))
	  else
	    e1=cexp(-Ci*dk*x1)
	    e2=cexp(-Ci*dk*x2)
		couple=cmplx(0.0,contrast/dk)*(e2-e1)
	  Endif
	  return
	  end
	  
      subroutine setSLV(il,Nl,w,pi,po,h,rho0,alpha0,beta0,
     &           rho,alpha,beta,S)
      Implicit none
	  Integer il,Nl,i,j,jp1
	  real    w,x1,x2,contrast,Eps,recipN
	  Parameter(Eps=1.0E-10)
	  complex S(4,4)
      complex dS11,dS12,dS21,dS22,dS31,dS32,dS41,dS42
      complex dS13,dS14,dS23,dS24,dS33,dS34,dS43,dS44
      real     rho0(0:Nl),alpha0(0:Nl),beta0(0:Nl),h(0:Nl)
      real     rho(0:Nl),alpha(0:Nl),beta(0:Nl)
      real    pi,po,dk,nu1,nu10
      real    alphasq0,betasq0,alphasq,betasq
	  complex Ndi(2,2),Mdi(2,2),Nui(2,2),Mui(2,2)
	  complex Ndo(2,2),Mdo(2,2),Nuo(2,2),Muo(2,2)
	  complex dA21,dA34,dA13,dA24,dA31,dA42
      complex Ci,C1,C0
      complex Verticalslo,couple
	  complex etai,netai,etao,netao,eai,ebi,eao,ebo
      complex sum21,sum24,sum13,sum42,sum31,dewp,ewp
	  
	  
	  If(w.eq.0.0) stop'static scattering case not yet implemented'
*	  recipN=1.0/float(Nx)
      Ci=(0.0,1.0)
      C0=(0.0,0.0)
      C1=(1.0,0.0)
      etai=VERTICALSLO(alpha0,pi)
      netai=VERTICALSLO(beta0,pi)
      etao=VERTICALSLO(alpha0,po)
      netao=VERTICALSLO(beta0,po)
      EAi=CEXP(0.5*Ci*w*h(il)*etai)
      EBi=CEXP(0.5*Ci*w*h(il)*netai)
      EAo=CEXP(0.5*Ci*w*h(il)*etao)
      EBo=CEXP(0.5*Ci*w*h(il)*netao)
*
*     **************************************************
*     set the Pi->Po coupling elements  Ci*w*<Po|dA|Pi>
*     ***************************************************
*
      alphasq0=alpha0(il)*alpha0(il)
      betasq0=beta0(il)*beta0(il)
      nu10=4.0*betasq0*(1.0-betasq0/alphasq0)
*      sum21=(0.0,0.0)
*      sum24=(0.0,0.0)
*      sum13=(0.0,0.0)
*      sum42=(0.0,0.0)
*      sum31=(0.0,0.0)
*	  dewp=Cexp(Ci*w*(pi-po)*dx)
*	  ewp=(1.0,0.0)
*	  Do j=0,Nx-1
*	    jp1=j+1
*lateral variation case
*        alphasq=alpha(jp1)*alpha(jp1)
*        betasq=beta(jp1)*beta(jp1)
*        nu1=4.0*betasq*(1.0-betasq/alphasq)
*        sum21=sum21+ewp*(betasq/alphasq-betasq0/alphasq0)
*        sum24=sum24+ewp*(1.0/(rho(jp1)*alphasq)-1.0/(rho0*alphasq0))
*        sum13=sum13+ewp*(1.0/(rho(jp1)*betasq)-1.0/(rho0*betasq0))
*        sum42=sum42+ewp*(rho(jp1)-rho0)
*        sum31=sum31+ewp*(rho(jp1)*nu1-rho0*nu10)
* 
*	    ewp=ewp*dewp
*      Enddo
*lateral invariant case
        dk=w*(po-pi)
		x1=0.5
		x2=1.5
        alphasq=alpha(il)*alpha(il)
        betasq=beta(il)*beta(il)
        nu1=4.0*betasq*(1.0-betasq/alphasq)
		contrast=betasq/alphasq-betasq0/alphasq0
        sum21=couple(contrast,dk,x1,x2)
		contrast=(1.0/(rho(il)*alphasq)-1.0/(rho0(il)*alphasq0))
        sum24=couple(contrast,dk,x1,x2)
		contrast=(1.0/(rho(il)*betasq)-1.0/(rho0(il)*betasq0))
        sum13=couple(contrast,dk,x1,x2)
		contrast=(rho(il)-rho0(il))
        sum42=couple(contrast,dk,x1,x2)
		contrast=(rho(il)*nu1-rho0(il)*nu10)
        sum31=couple(contrast,dk,x1,x2)

      dA21=Ci*w*pi*2.0*sum21
      dA34=Ci*w*po*2.0*sum21
      dA24=Ci*w*sum24
      dA13=Ci*w*sum13
      dA42=Ci*w*sum42
*     ************************************************************
*     set the first-Born scattering elements for Pi->Po transition
*     ************************************************************
      call SetD(Pi,ETAi,NETAi,rho0,beta0,Mdi,Ndi,Mui,Nui)
      call SetD(Po,ETAo,NETAo,rho0,beta0,Mdo,Ndo,Muo,Nuo)

      dS11=-(Ndi(2,1)*dA34+Mdi(1,1)*dA31)*Mdo(1,1)+
     &      (-Ndi(2,1)*dA24-Mdi(1,1)*dA21)*Ndo(2,1)+Mdi(2,1)*
     &      Mdo(2,1)*dA42+Ndi(1,1)*Ndo(1,1)*dA13
	  dS12=-(Ndi(2,2)*dA34+Mdi(1,2)*dA31)*Mdo(1,1)+(-Ndi(2,2)*dA24-
     &      Mdi(1,2)*dA21)*Ndo(2,1)+Mdi(2,2)*Mdo(2,1)*dA42+
     &      Ndi(1,2)*Ndo(1,1)*dA13
	  dS13=-(Nui(2,1)*dA34+Mui(1,1)*dA31)*Mdo(1,1)+(-Nui(2,1)*dA24-
     &     Mui(1,1)*dA21)*Ndo(2,1)+Mui(2,1)*Mdo(2,1)*dA42+Nui(1,1)*
     &     Ndo(1,1)*dA13
	  dS14=-(Nui(2,2)*dA34+Mui(1,2)*dA31)*Mdo(1,1)+(-Nui(2,2)*dA24-
     &      Mui(1,2)*dA21)*Ndo(2,1)+Mui(2,2)*Mdo(2,1)*dA42+Nui(1,2)*
     &      Ndo(1,1)*dA13
	  dS21=(Ndi(2,1)*dA34+Mdi(1,1)*dA31)*Mdo(1,2)-(-Ndi(2,1)*dA24-
     &      Mdi(1,1)*dA21)*Ndo(2,2)-Mdi(2,1)*Mdo(2,2)*dA42-Ndi(1,1)*
     &      Ndo(1,2)*dA13
	  dS22=(Ndi(2,2)*dA34+Mdi(1,2)*dA31)*Mdo(1,2)-(-Ndi(2,2)*dA24-
     &     Mdi(1,2)*dA21)*Ndo(2,2)-Mdi(2,2)*Mdo(2,2)*dA42-Ndi(1,2)*
     &     Ndo(1,2)*dA13
	  dS23=(Nui(2,1)*dA34+Mui(1,1)*dA31)*Mdo(1,2)-(-Nui(2,1)*dA24-
     &     Mui(1,1)*dA21)*Ndo(2,2)-Mui(2,1)*Mdo(2,2)*dA42-Nui(1,1)*
     &     Ndo(1,2)*dA13
	  dS24=(Nui(2,2)*dA34+Mui(1,2)*dA31)*Mdo(1,2)-(-Nui(2,2)*dA24-
     &     Mui(1,2)*dA21)*Ndo(2,2)-Mui(2,2)*Mdo(2,2)*dA42-Nui(1,2)*
     &     Ndo(1,2)*dA13
	  dS31=-(Ndi(2,1)*dA34+Mdi(1,1)*dA31)*Muo(1,1)+(-Ndi(2,1)*dA24-
     &     Mdi(1,1)*dA21)*Nuo(2,1)+Mdi(2,1)*Muo(2,1)*dA42+Ndi(1,1)*
     &     Nuo(1,1)*dA13
	  dS32=-(Ndi(2,2)*dA34+Mdi(1,2)*dA31)*Muo(1,1)+(-Ndi(2,2)*dA24-
     &     Mdi(1,2)*dA21)*Nuo(2,1)+Mdi(2,2)*Muo(2,1)*dA42+Ndi(1,2)*
     &     Nuo(1,1)*dA13
	  dS33=-(Nui(2,1)*dA34+Mui(1,1)*dA31)*Muo(1,1)+(-Nui(2,1)*dA24-
     &     Mui(1,1)*dA21)*Nuo(2,1)+Mui(2,1)*Muo(2,1)*dA42+Nui(1,1)*
     &     Nuo(1,1)*dA13

	  dS34=-(Nui(2,2)*dA34+Mui(1,2)*dA31)*Muo(1,1)+(-Nui(2,2)*dA24-
     &     Mui(1,2)*dA21)*Nuo(2,1)+Mui(2,2)*Muo(2,1)*dA42+Nui(1,2)*
     &     Nuo(1,1)*dA13
	  dS41=(Ndi(2,1)*dA34+Mdi(1,1)*dA31)*Muo(1,2)-(-Ndi(2,1)*dA24-
     &    Mdi(1,1)*dA21)*Nuo(2,2)-Mdi(2,1)*Muo(2,2)*dA42-Ndi(1,1)*
     &    Nuo(1,2)*dA13
	  dS42=(Ndi(2,2)*dA34+Mdi(1,2)*dA31)*Muo(1,2)-(-Ndi(2,2)*dA24-
     &    Mdi(1,2)*dA21)*Nuo(2,2)-Mdi(2,2)*Muo(2,2)*dA42-Ndi(1,2)*
     &    Nuo(1,2)*dA13
	  dS43=(Nui(2,1)*dA34+Mui(1,1)*dA31)*Muo(1,2)-(-Nui(2,1)*dA24-
     &    Mui(1,1)*dA21)*Nuo(2,2)-Mui(2,1)*Muo(2,2)*dA42-Nui(1,1)*
     &    Nuo(1,2)*dA13
	  dS44=(Nui(2,2)*dA34+Mui(1,2)*dA31)*Muo(1,2)-(-Nui(2,2)*dA24-
     &    Mui(1,2)*dA21)*Nuo(2,2)-Mui(2,2)*Muo(2,2)*dA42-Nui(1,2)*
     &    Nuo(1,2)*dA13
      
	  S(1,1)=h(il)*eao*eai*dS11
	  S(1,2)=h(il)*eao*ebi*dS12
	  S(1,3)=h(il)*eao*eai*dS13
	  S(1,4)=h(il)*eao*ebi*dS14

	  S(2,1)=h(il)*ebo*eai*dS21
	  S(2,2)=h(il)*ebo*ebi*dS22
	  S(2,3)=h(il)*ebo*eai*dS23
	  S(2,4)=h(il)*ebo*ebi*dS24

	  S(3,1)=h(il)*eao*eai*dS31
	  S(3,2)=h(il)*eao*ebi*dS32
	  S(3,3)=h(il)*eao*eai*dS33
	  S(3,4)=h(il)*eao*ebi*dS34

	  S(4,1)=h(il)*ebo*eai*dS41
	  S(4,2)=h(il)*ebo*ebi*dS42
	  S(4,3)=h(il)*ebo*eai*dS43
	  S(4,4)=h(il)*ebo*ebi*dS44

      return
	  End
	  
	  
	  
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
*     doublesweep-Free surface case
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
        S1=(1.0,0.0) !DELETE
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
	  
      subroutine QtV(a,n,k,np,c,V)
      implicit none
*     P(k)...*P(j)*..P(1)*V
      integer  n,np,k
      complex  a(np,np),V(n)
      real     c(n)
      integer  i,j
      complex  sum,tau
        do 13 j=1,k
          sum=(0.0,0.0)
          do 11 i=j,n 
            sum=sum+conjg(a(i,j))*V(i)
11        continue
          tau=sum/c(j)
          do 12 i=j,n
            V(i)=V(i)-tau*a(i,j)
12        continue
13      continue
      return
      end

      subroutine QV(a,n,k,np,c,V)
      implicit none
*     P(1)...*P(j)*..P(k)*V
      integer  n,np,k
      complex  a(np,np),V(n)
      real     c(n)
      integer  i,j
      complex  sum,tau
        do 13 j=k,1,-1
          sum=(0.0,0.0)
          do 11 i=j,n 
            sum=sum+conjg(a(i,j))*V(i)
11        continue
          tau=sum/c(j)
          do 12 i=j,n
            V(i)=V(i)-tau*a(i,j)
12        continue
13      continue
      return
      end

      SUBROUTINE setPkV(a,V,n,k,np,mp,c,sing)
      implicit none
      INTEGER  n,np,mp,k
      complex  a(np,np),V(n)
      real     c(n)
      LOGICAL  sing
      INTEGER  i,j
      real     scale,L2,absakk
      complex  sigma,sum,tau
	  if(k.lt.1) Pause'k.lt.1 in setPkV'
	  if(k.gt.Mp) Pause'k.gt.Mp in setPkV'
      sing=.false.
      scale=0.
          do 11 i=k,n
            scale=max(scale,cabs(V(i)))
11        continue
          if(scale.lt.1.E-6)then
            sing=.true.
            c(k)=(0.0,0.0)
          else
            do 12 i=k,n
              V(i)=V(i)/scale
12          continue
            L2=0.0
            do 13 i=k,n
              L2=L2+V(i)*conjg(V(i))
13          continue
            L2=sqrt(L2)  ! L2 norm of V
*            TYpe*,'k,L2=',k,L2
            absakk=cabs(V(k))
            if(absakk.lt.1.E-6) then
              pause 'absakk.lt.1.E-6'
            endif
*            Type*,'k,V(k)=',k,V(k)
            sigma=L2*V(k)/absakk
            a(k,k)=V(k)+sigma
            c(k)=L2*(L2+absakk)
            V(k)=-scale*sigma
*            Type*,'c(k)=',c(k)
*            Type*,'V(k)=',V(k)
             do 15 i=k+1,n
                a(i,k)=V(i)
				V(i)=(0.0,0.0)
15           continue
      endif
      return
      END
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

********************************************************
      Subroutine atimes1(Nl,Np,Adj,w,p,h,rho0,alpha0,beta0,
     &                  rho,alpha,beta,Y,X)
*     ********************************************************
*     set X=(1+P*dS)*Y
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
	    call setPSY(Nl,p(ip),w,p,h,rho,alpha,beta,b,c)
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
	    X(i)=Y(i)+X(i)
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
