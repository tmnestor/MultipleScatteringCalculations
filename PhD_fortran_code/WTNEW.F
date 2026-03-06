      Program WTnew
      Implicit None
      integer  n,isign,i
	  Parameter(n=64)
      Real a(n)
      data a/64*(1.0)/
		isign=1
		call wt(a,n,isign)

		isign=-1
		call wt(a,n,isign)

	  do i=1,n
	    type*,'i,a(i)=',i,a(i)
	  enddo

      STOP 'normal completion'
      END
      
	  
	  
      Subroutine WT(a,n,isign)
      implicit none
      Integer isign,n,nn
      Real a(n)
      if(n.lt.2) return
      if(isign.ge.0) then
        nn=n
 1      if(nn.ge.2) then
          call Haar(a,nn,isign)
	   nn=nn/2
	   goto 1
	   endif
       else
	    nn=2
 2       if (nn.le.n) then
           call Haar(a,nn,isign)
           nn=nn*2
	    goto 2
	  endif
	endif
	return
	end

	  subroutine Haar(a,n,isign)
	  implicit none
	  integer n, isign,nmax
	  real C0
	  parameter(C0=0.707106781186547524,Nmax=1024)
	  Real  a(n),wksp(Nmax)
	  integer nh,nh1,i,j
	  if(n.lt.2) return
	  if(n.gt. Nmax) Pause'wksp too small in Haar'
	  nh=n/2
	  nh1=nh+1
	  if(isign.ge.0) then
	    i=1
	    do j=1,n-1,2
	    wksp(i)=C0*(a(j)+a(j+1))
	    wksp(i+nh)=C0*(a(j)-a(j+1))
	    i=i+1
	    end do
	  else
	    j=1
	    do i=1,nh
	      wksp(j)=C0*(a(i)+a(i+nh))
	      wksp(j+1)=C0*(a(i)-a(i+nh))
	      j=j+2
	    end do
	  endif
	  do i=1,n
	    a(i)=wksp(i)
*	    type*,'i,a(i)=',i,a(i)
	  enddo
	  return
	  end

