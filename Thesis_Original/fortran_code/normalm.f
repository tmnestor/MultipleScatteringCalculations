      PROGRAM SFROID
      PARAMETER(NE=3,M=41,NB=1,NCI=NE,NCJ=NE-NB+1,NCK=M+1,NSI=NE,
     *    NSJ=2*NE+1,NYJ=NE,NYK=M)
      COMMON X(M),H,MM,N,C2,ANORM
      DIMENSION SCALV(NE),INDEXV(NE),Y(NE,M),C(NCI,NCJ,NCK),S(NSI,NSJ)
      ITMAX=100
      CONV=5.E-6
      SLOWC=1.
      H=1./(M-1)
      C2=0.
      WRITE(*,*)'ENTER M,N'
      READ(*,*)MM,N
      IF(MOD(N+MM,2).EQ.1)THEN
        INDEXV(1)=1
        INDEXV(2)=2
        INDEXV(3)=3
      ELSE
        INDEXV(1)=2
        INDEXV(2)=1
        INDEXV(3)=3
      ENDIF
      ANORM=1.
      IF(MM.NE.0)THEN
        Q1=N
        DO 11 I=1,MM
          ANORM=-.5*ANORM*(N+I)*(Q1/I)
          Q1=Q1-1.
11      CONTINUE
      ENDIF
      DO 12 K=1,M-1
        X(K)=(K-1)*H
        FAC1=1.-X(K)**2
        FAC2=FAC1**(-MM/2.)
        Y(1,K)=PLGNDR(N,MM,X(K))*FAC2
        DERIV=-((N-MM+1)*PLGNDR(N+1,MM,X(K))-(N+1)*
     *      X(K)*PLGNDR(N,MM,X(K)))/FAC1
        Y(2,K)=MM*X(K)*Y(1,K)/FAC1+DERIV*FAC2
        Y(3,K)=N*(N+1)-MM*(MM+1)
12    CONTINUE
      X(M)=1.
      Y(1,M)=ANORM
      Y(3,M)=N*(N+1)-MM*(MM+1)
      Y(2,M)=(Y(3,M)-C2)*Y(1,M)/(2.*(MM+1.))
      SCALV(1)=ABS(ANORM)
      SCALV(2)=MAX(ABS(ANORM),Y(2,M))
      SCALV(3)=MAX(1.,Y(3,M))
1     CONTINUE
      WRITE (*,*) 'ENTER C**2 OR 999 TO END'
      READ (*,*) C2
      IF(C2.EQ.999.)STOP
      CALL SOLVDE(ITMAX,CONV,SLOWC,SCALV,INDEXV,NE,NB,M,Y,NYJ,NYK,
     *    C,NCI,NCJ,NCK,S,NSI,NSJ)
      WRITE (*,*) ' M = ',MM,'  N = ',N,
     *    '  C**2 = ',C2,'  LAMBDA = ',Y(3,1)+MM*(MM+1)
      GO TO 1
      END


      SUBROUTINE solvde(itmax,conv,slowc,scalv,indexv,ne,nb,m,y,nyj,nyk,
     *c,nci,ncj,nck,s,nsi,nsj)
      INTEGER itmax,m,nb,nci,ncj,nck,ne,nsi,nsj,nyj,nyk,indexv(nyj),NMAX
      REAL conv,slowc,c(nci,ncj,nck),s(nsi,nsj),scalv(nyj),y(nyj,nyk)
      PARAMETER (NMAX=10)
CU    USES bksub,difeq,pinvs,red
      INTEGER ic1,ic2,ic3,ic4,it,j,j1,j2,j3,j4,j5,j6,j7,j8,j9,jc1,jcf,
     *jv,k,k1,k2,km,kp,nvars,kmax(NMAX)
      REAL err,errj,fac,vmax,vz,ermax(NMAX)
      k1=1
      k2=m
      nvars=ne*m
      j1=1
      j2=nb
      j3=nb+1
      j4=ne
      j5=j4+j1
      j6=j4+j2
      j7=j4+j3
      j8=j4+j4
      j9=j8+j1
      ic1=1
      ic2=ne-nb
      ic3=ic2+1
      ic4=ne
      jc1=1
      jcf=ic3
      do 16 it=1,itmax
        k=k1
        call difeq(k,k1,k2,j9,ic3,ic4,indexv,ne,s,nsi,nsj,y,nyj,nyk)
        call pinvs(ic3,ic4,j5,j9,jc1,k1,c,nci,ncj,nck,s,nsi,nsj)
        do 11 k=k1+1,k2
          kp=k-1
          call difeq(k,k1,k2,j9,ic1,ic4,indexv,ne,s,nsi,nsj,y,nyj,nyk)
          call red(ic1,ic4,j1,j2,j3,j4,j9,ic3,jc1,jcf,kp,c,nci,ncj,nck,
     *s,nsi,nsj)
          call pinvs(ic1,ic4,j3,j9,jc1,k,c,nci,ncj,nck,s,nsi,nsj)
11      continue
        k=k2+1
        call difeq(k,k1,k2,j9,ic1,ic2,indexv,ne,s,nsi,nsj,y,nyj,nyk)
        call red(ic1,ic2,j5,j6,j7,j8,j9,ic3,jc1,jcf,k2,c,nci,ncj,nck,s,
     *nsi,nsj)
        call pinvs(ic1,ic2,j7,j9,jcf,k2+1,c,nci,ncj,nck,s,nsi,nsj)
        call bksub(ne,nb,jcf,k1,k2,c,nci,ncj,nck)
        err=0.
        do 13 j=1,ne
          jv=indexv(j)
          errj=0.
          km=0
          vmax=0.
          do 12 k=k1,k2
            vz=abs(c(jv,1,k))
            if(vz.gt.vmax) then
               vmax=vz
               km=k
            endif
            errj=errj+vz
12        continue
          err=err+errj/scalv(j)
          ermax(j)=c(jv,1,km)/scalv(j)
          kmax(j)=km
13      continue
        err=err/nvars
        fac=slowc/max(slowc,err)
        do 15 j=1,ne
          jv=indexv(j)
          do 14 k=k1,k2
            y(j,k)=y(j,k)-fac*c(jv,1,k)
14        continue
15      continue
        write(*,100) 'it,err,fac=',it,err,fac
        if(err.lt.conv) return
16    continue
      pause 'itmax exceeded in solvde'
100   format(1x,A11,i4,2f12.6)
      return
      END


      SUBROUTINE red(iz1,iz2,jz1,jz2,jm1,jm2,jmf,ic1,jc1,jcf,kc,c,nci,
     *ncj,nck,s,nsi,nsj)
      INTEGER ic1,iz1,iz2,jc1,jcf,jm1,jm2,jmf,jz1,jz2,kc,nci,ncj,nck,
     *nsi,nsj
      REAL c(nci,ncj,nck),s(nsi,nsj)
      INTEGER i,ic,j,l,loff
      REAL vx
      loff=jc1-jm1
      ic=ic1
      do 14 j=jz1,jz2
        do 12 l=jm1,jm2
          vx=c(ic,l+loff,kc)
          do 11 i=iz1,iz2
            s(i,l)=s(i,l)-s(i,j)*vx
11        continue
12      continue
        vx=c(ic,jcf,kc)
        do 13 i=iz1,iz2
          s(i,jmf)=s(i,jmf)-s(i,j)*vx
13      continue
        ic=ic+1
14    continue
      return
      END


      SUBROUTINE pinvs(ie1,ie2,je1,jsf,jc1,k,c,nci,ncj,nck,s,nsi,nsj)
      INTEGER ie1,ie2,jc1,je1,jsf,k,nci,ncj,nck,nsi,nsj,NMAX
      REAL c(nci,ncj,nck),s(nsi,nsj)
      PARAMETER (NMAX=10)
      INTEGER i,icoff,id,ipiv,irow,j,jcoff,je2,jp,jpiv,js1,indxr(NMAX)
      REAL big,dum,piv,pivinv,pscl(NMAX)
      je2=je1+ie2-ie1
      js1=je2+1
      do 12 i=ie1,ie2
        big=0.
        do 11 j=je1,je2
          if(abs(s(i,j)).gt.big) big=abs(s(i,j))
11      continue
        if(big.eq.0.) pause 'singular matrix, row all 0 in pinvs'
        pscl(i)=1./big
        indxr(i)=0
12    continue
      do 18 id=ie1,ie2
        piv=0.
        do 14 i=ie1,ie2
          if(indxr(i).eq.0) then
            big=0.
            do 13 j=je1,je2
              if(abs(s(i,j)).gt.big) then
                jp=j
                big=abs(s(i,j))
              endif
13          continue
            if(big*pscl(i).gt.piv) then
              ipiv=i
              jpiv=jp
              piv=big*pscl(i)
            endif
          endif
14      continue
        if(s(ipiv,jpiv).eq.0.) pause 'singular matrix in pinvs'
        indxr(ipiv)=jpiv
        pivinv=1./s(ipiv,jpiv)
        do 15 j=je1,jsf
          s(ipiv,j)=s(ipiv,j)*pivinv
15      continue
        s(ipiv,jpiv)=1.
        do 17 i=ie1,ie2
          if(indxr(i).ne.jpiv) then
            if(s(i,jpiv).ne.0.) then
              dum=s(i,jpiv)
              do 16 j=je1,jsf
                s(i,j)=s(i,j)-dum*s(ipiv,j)
16            continue
              s(i,jpiv)=0.
            endif
          endif
17      continue
18    continue
      jcoff=jc1-js1
      icoff=ie1-je1
      do 21 i=ie1,ie2
        irow=indxr(i)+icoff
        do 19 j=js1,jsf
          c(irow,j+jcoff,k)=s(i,j)
19      continue
21    continue
      return
      END





      SUBROUTINE bksub(ne,nb,jf,k1,k2,c,nci,ncj,nck)
      INTEGER jf,k1,k2,nb,nci,ncj,nck,ne
      REAL c(nci,ncj,nck)
      INTEGER i,im,j,k,kp,nbf
      REAL xx
      nbf=ne-nb
      im=1
      do 13 k=k2,k1,-1
        if (k.eq.k1) im=nbf+1
        kp=k+1
        do 12 j=1,nbf
          xx=c(j,jf,kp)
          do 11 i=im,ne
            c(i,jf,k)=c(i,jf,k)-c(i,j,k)*xx
11        continue
12      continue
13    continue
      do 16 k=k1,k2
        kp=k+1
        do 14 i=1,nb
          c(i,1,k)=c(i+nbf,jf,k)
14      continue
        do 15 i=1,nbf
          c(i+nb,1,k)=c(i,jf,kp)
15      continue
16    continue
      return
      END



      SUBROUTINE DIFEQ(K,K1,K2,JSF,IS1,ISF,INDEXV,NE,S,NSI,NSJ,Y,NYJ,NYK)
      PARAMETER(M=41)
      COMMON X(M),H,MM,N,C2,ANORM
      DIMENSION Y(NYJ,NYK),S(NSI,NSJ),INDEXV(NYJ)
      IF(K.EQ.K1) THEN
        IF(MOD(N+MM,2).EQ.1)THEN
          S(3,3+INDEXV(1))=1.
          S(3,3+INDEXV(2))=0.
          S(3,3+INDEXV(3))=0.
          S(3,JSF)=Y(1,1)
        ELSE
          S(3,3+INDEXV(1))=0.
          S(3,3+INDEXV(2))=1.
          S(3,3+INDEXV(3))=0.
          S(3,JSF)=Y(2,1)
        ENDIF
      ELSE IF(K.GT.K2) THEN
        S(1,3+INDEXV(1))=-(Y(3,M)-C2)/(2.*(MM+1.))
        S(1,3+INDEXV(2))=1.
        S(1,3+INDEXV(3))=-Y(1,M)/(2.*(MM+1.))
        S(1,JSF)=Y(2,M)-(Y(3,M)-C2)*Y(1,M)/(2.*(MM+1.))
        S(2,3+INDEXV(1))=1.
        S(2,3+INDEXV(2))=0.
        S(2,3+INDEXV(3))=0.
        S(2,JSF)=Y(1,M)-ANORM
      ELSE
        S(1,INDEXV(1))=-1.
        S(1,INDEXV(2))=-.5*H
        S(1,INDEXV(3))=0.
        S(1,3+INDEXV(1))=1.
        S(1,3+INDEXV(2))=-.5*H
        S(1,3+INDEXV(3))=0.
        TEMP=H/(1.-(X(K)+X(K-1))**2*.25)
        TEMP2=.5*(Y(3,K)+Y(3,K-1))-C2*.25*(X(K)+X(K-1))**2
        S(2,INDEXV(1))=TEMP*TEMP2*.5
        S(2,INDEXV(2))=-1.-.5*TEMP*(MM+1.)*(X(K)+X(K-1))
        S(2,INDEXV(3))=.25*TEMP*(Y(1,K)+Y(1,K-1))
        S(2,3+INDEXV(1))=S(2,INDEXV(1))
        S(2,3+INDEXV(2))=2.+S(2,INDEXV(2))
        S(2,3+INDEXV(3))=S(2,INDEXV(3))
        S(3,INDEXV(1))=0.
        S(3,INDEXV(2))=0.
        S(3,INDEXV(3))=-1.
        S(3,3+INDEXV(1))=0.
        S(3,3+INDEXV(2))=0.
        S(3,3+INDEXV(3))=1.
        S(1,JSF)=Y(1,K)-Y(1,K-1)-.5*H*(Y(2,K)+Y(2,K-1))
        S(2,JSF)=Y(2,K)-Y(2,K-1)-TEMP*((X(K)+X(K-1))
     *      *.5*(MM+1.)*(Y(2,K)+Y(2,K-1))-TEMP2*
     *      .5*(Y(1,K)+Y(1,K-1)))
        S(3,JSF)=Y(3,K)-Y(3,K-1)
      ENDIF
      RETURN
      END


      FUNCTION plgndr(l,m,x)
      INTEGER l,m
      REAL plgndr,x
      INTEGER i,ll
      REAL fact,pll,pmm,pmmp1,somx2
      if(m.lt.0.or.m.gt.l.or.abs(x).gt.1.)pause
     *'bad arguments in plgndr'
      pmm=1.
      if(m.gt.0) then
        somx2=sqrt((1.-x)*(1.+x))
        fact=1.
        do 11 i=1,m
          pmm=-pmm*fact*somx2
          fact=fact+2.
11      continue
      endif
      if(l.eq.m) then
        plgndr=pmm
      else
        pmmp1=x*(2*m+1)*pmm
        if(l.eq.m+1) then
          plgndr=pmmp1
        else
          do 12 ll=m+2,l
            pll=(x*(2*ll-1)*pmmp1-(ll+m-1)*pmm)/(ll-m)
            pmm=pmmp1
            pmmp1=pll
12        continue
          plgndr=pll
        endif
      endif
      return
      END

