      Program WT
      Implicit None

      Integer outfile,N,Nmax,i,j,isign,icount
      Parameter(Nmax=256,outfile=11)
      Real eps,dn,c,x,ff,biggest,ahold(Nmax,Nmax),del
	  Real tempd,w,lambda,f,kf,ki,dk,kmax
      Complex a(Nmax,Nmax),tempc
      Parameter(eps=1.E-4)
      character *1 TAB
      external daub4
      external pwt
      integer nn(2)/256,256/
      TAB=char(9)
      open(outfile,file='WTGraphDat',status='new')
	  
	  lambda=1.0/sqrt(2.0)
	  type*,'lambda=',lambda
	  stop
		 lambda=3.0
		 w=2.0
		 kmax=4*pi
      	 N=256
		 del=2.0*kmax/float(N)
         call pwtset(4)
     	 Do i= 0,N-1
*		   kf=-kmax+i*del
		   do j=0,N-1
*		     ki=-kmax+j*del
*			 dk=kf-ki
             if(i.eq.j) then
*         	 a(i+1,j+1)=Cmplx(f(dk,w,lambda),0.0)
         	 a(i+1,j+1)=(1.0,0.0)
			 else
			 a(i+1,j+1)=(0.0,0.0)
			 endif
			 ahold(i+1,j+1)=Real(a(i+1,j+1))
*			 type*,'i+1,j+1,ahold(i+1,j+1)=',i+1,j+1,ahold(i+1,j+1)
*	         type*,'i,a(i+1)',i,a(i+1)
          enddo
     	end do


		isign=1
*		call wt1(a,n,isign,daub4)
*	call wt1(a,n,isign,pwt)
*       call wtn(a,nn,ndim,isign,pwt)
      write(*,*) 'calling wtn'
      call wtn(a,nn,2,1,pwt)
      write(*,*) 'left wtn'

	  biggest=0.0
	  do i=0,N-1
	    do j=0,N-1
	      if(Cabs(a(i+1,j+1)).gt.biggest) biggest=Cabs(a(i+1,j+1))
		enddo
      End do

       icount=0
	   do i=0,N-1
	   do j=0,N-1
	    if(Cabs(a(i+1,j+1)).lt.biggest*1.0E-4) then
	      a(i+1,j+1)=(0.0,0.0)
	      icount=icount+1
	    end if
*	    WRITE (outfile,999) i,TAB,temp(i+1)
*	    TYPE*,i,TAB,temp(i+1)
        enddo
      	End do
       type*,'no. of vanishing coeffs=',icount
        pause
		isign=-1
*		call wt1(a,n,isign,daub4)
*      call wt1(a,n,isign,pwt)
*      call wtn(a,nn,ndim,isign,pwt)
      write(*,*) 'entering inverse transform'
      call wtn(a,nn,2,-1,pwt)
      pause 'left wt1 again'
      biggest=0.0
      Do i= 0,N-1
      Do j= 0,N-1
	    tempd=(a(i+1,j+1)-ahold(i+1,j+1))/ahold(i+1,j+1)
        Type*,i+1,j+1,Real(a(i+1,j+1)),tempd
	    if(Abs(tempd).gt.biggest) biggest=Abs(tempd)
	  enddo
      end do
      type*,'largest relative error=',biggest
 999  FORMAT(I4,A1,g10.4) 
      close(outfile)
      STOP 'normal completion'
      END
      
	  real function f(k,w,lambda)
	  real k,w,lambda,mk
	  mk=abs(k)
	  if(mk.lt.1.e-8)then
	    f=w
	  else
	    f=pi*sin(w*mk)*exp(-0.5*pi*mk/lambda)/(1.0-exp(-pi*mk/lambda))/lambda
	  end if
	  return
	  end
	  
	  
      Subroutine WT1(a,n,isign,wtstep)
      implicit none
      Integer isign,n,nn
      Complex a(n)
      external wtstep
      if(n.lt.4) return
      if(isign.ge.0) then
        nn=n
 1      if(nn.ge.4) then
          call wtstep(a,nn,isign)
	   nn=nn/2
	 goto 1
	 endif
       else
	  nn=4
 2       if (nn.le.n) then
           call wtstep(a,nn,isign)
           nn=nn*2
	    goto 2
	  endif
	endif
	return
	end
	
	subroutine daub4(a,n,isign)
	implicit none
	integer n, isign,nmax
	real C0,C1,C2,C3
	parameter(C0=0.4829629131445341,C1=0.8365163037378079,
     &           C2=0.2241438680420134,C3=-0.1294095225512604,Nmax=1024)
       Complex a(n),wksp(Nmax)
	integer nh,nh1,i,j
	if(n.lt.4) return
	if(n.gt. Nmax) Pause'wksp too small in daub4'
	nh=n/2
	nh1=nh+1
	if(isign.ge.0) then
	  i=1
	  do j=1,n-3,2
	    wksp(i)=C0*a(j)+C1*a(j+1)+C2*a(j+2)+C3*a(j+3)
	    wksp(i+nh)=C3*a(j)-C2*a(j+1)+C1*a(j+2)-C0*a(j+3)
	    i=i+1
	  end do
	  wksp(i)=C0*a(n-1)+C1*a(n)+C2*a(1)+C3*a(2)
	  wksp(i+nh)=C3*a(n-1)-C2*a(n)+C1*a(1)-C0*a(2)
	else
	  wksp(1)=C2*a(nh)+C1*a(n)+C0*a(1)+C3*a(nh1)
	  wksp(2)=C3*a(nh)-C0*a(n)+C1*a(1)-C2*a(nh1)
	  j=3
	  do i=1,nh-1
	    wksp(j)=C2*a(i)+C1*a(i+nh)+C0*a(i+1)+C3*a(i+nh1)
	    wksp(j+1)=C3*a(i)-C0*a(i+nh)+C1*a(i+1)-C2*a(i+nh1)
	    j=j+2
	  end do
	endif
	do i=1,n
	  a(i)=wksp(i)
*	  type*,'i,a(i)=',i,a(i)
	enddo
	return
	end

      Subroutine pwtset(n)
      implicit none
      integer n,ncmax,ncof,ioff,joff
      parameter(ncmax=50)
      real cc(ncmax),cr(ncmax)
      common /pwtcom/cc,cr,ncof,ioff,joff
      integer k
      real sig,c4(4),c12(12),c20(20)
      save c4,c12,c20,/pwtcom/
      
      data c4/0.4829629131445341,0.8365163037378079,
     &     0.2241438680420134,-0.1294095225512604/
     
      data c12/0.111540743350,0.494623890398,0.751133908021,
     &     0.315250351709,-0.226264693965,-0.129766867567,
     &     0.097501605587,0.027522865530,-0.031582039318,
     &     0.000553842201,0.004777257511,-0.001077301085/
     
      data c20/0.026670057901,0.188176800078,0.527201188932,
     &     0.688459039454,0.281172343661,-0.249846424327,
     &     -0.195946274377,0.127369340336,0.093057364604,
     &     -0.071394147166,-0.029457536822,0.033212674059,
     &     0.003606553567,-0.010733175483,0.001395351747,
     &     0.001992405295,-0.000685856695,-0.000116466855,
     &     0.000093588670,-0.000013264203/
     
      ncof=n
      sig=-1.0
      do k=1,n
        if(n.eq.4) then
	   cc(k)=c4(k)
	 else if(n.eq.12) then
	   cc(k)=c12(k)
	 else if(n.eq.20) then
	   cc(k)=c20(k)
        else
	   pause'unimplemented value of n in pwtset'
	 endif
	 cr(ncof+1-k)=sig*cc(k)
	 sig=-sig
       enddo
	ioff=-n/2
	joff=-n/2
	return
	end
	
	subroutine pwt(a,n,isign)
	integer isign,n,nmax,ncmax,ncof,ioff,joff
	parameter (nmax=2048,ncmax=50)
	real cc(Ncmax),cr(Ncmax)
       common /pwtcom/cc,cr,ncof,ioff,joff
	integer i,ii,j,jf,jr,k,n1,ni,nj,nh,nmod
	Complex a(n),wksp(nmax),ai,ai1
	if (n.lt.4) return
	nmod=ncof*n
	n1=n-1
	nh=n/2
	do j=1,n
	  wksp(j)=0.0
	enddo
	if(isign.ge.0) then
	  ii=1
	  do i=1,n,2
	    ni=i+nmod+ioff
	    nj=i+nmod+joff
	    do k=1,ncof
	      jf=iand(n1,ni+k)
	      jr=iand(n1,nj+k)
	      wksp(ii)=wksp(ii)+cc(k)*a(jf+1)
	      wksp(ii+nh)=wksp(ii+nh)+cr(k)*a(jr+1)
	    enddo
	    ii=ii+1
	  enddo
	else
	  ii=1
	  do i=1,n,2
	    ai=a(ii)
	    ai1=a(ii+nh)
	    ni=i+nmod+ioff
	    nj=i+nmod+joff
	    do k=1,ncof
	      jf=iand(n1,ni+k)+1
	      jr=iand(n1,nj+k)+1
	      wksp(jf)=wksp(jf)+cc(k)*ai
	      wksp(jr)=wksp(jr)+cr(k)*ai1
	    enddo
	    ii=ii+1
	  enddo
	endif
	do i=1,n
	  a(i)=wksp(i)
*	  type*,'i,a(i)=',i,a(i)
	enddo
	return
	end

      Subroutine wtn(a,nn,ndim,isign,wtstep)
      implicit none
      integer   isign,ndim,nn(ndim),nmax
      parameter (nmax=1024)
      complex   a(*),wksp(nmax)
      external  wtstep
      integer   i1,i2,i3,idim,k,n,nnew,nprev,nt,ntot
      
      ntot=1
      do idim=1,ndim
        ntot=ntot*nn(idim)
      enddo
      nprev=1
      do idim=1,ndim
        n=nn(idim)
	 nnew=n*nprev
	 if(n.gt.4) then
	   do i2=0,ntot-1,nnew
	     do i1=1,nprev
	       i3=i1+i2
		do k=1,n
		  wksp(k)=a(i3)
		  i3=i3+nprev
		enddo
              if(isign.ge.0) then
                nt=n
 1              if(nt.ge.4) then
                call wtstep(wksp,nt,isign)
	         nt=nt/2
	         goto 1
	       endif
            else
	       nt=4
 2            if (nt.le.n) then
              call wtstep(wksp,nt,isign)
              nt=nt*2
	       goto 2
	     endif
	   endif
	   i3=i1+i2
	   do k=1,n
	     a(i3)=wksp(k)
	     i3=i3+nprev
	   enddo
	 enddo
        enddo
	 endif
	 nprev=nnew
      enddo
      return
      end
