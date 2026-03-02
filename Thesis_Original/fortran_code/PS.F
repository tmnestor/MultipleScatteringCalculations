	  program ps
*
*	2-D Wave solution by the pseudo-spectral method
*
	  parameter (npx=256,npz=256,npx2=2*npx,npz2=2*npz)
	  dimension ud(npx,npz),un(npx,npz),uf(npx,npz)
	  dimension wd(npx,npz),wn(npx,npz),wf(npx,npz)
	  dimension tmp1(npx2,npz),tmp2(npx2,npz),tmp3(npx2,npz)
	  dimension tmpx(npx2,npz),tmpz(npx2,npz),tmpp(npx2,npz)
	  dimension w(npx,npz),nn(2),vp2(npx,npz),vs2(npx,npz)

      data dxz,dt,nstep/20.,0.001,20/
      data isx,isz,t0,alpha/33,33,0.15,1500.0/


*	  open(8,file='psin',form='unformatted',status='old')
*	  read(8,*) dxz,dt,nx,nz,nstep
*
*	read source position and time function parameters
*
*	  read(8,*) isx,isz,t0,alpha


	  nn(1)=npx
	  nn(2)=npz
	  do 80 j=1,npz
	  do 80 i=1,npx
	  vp2(i,j)=0.0
	  vs2(i,j)=0.0
80	  w(i,j)=1.0
*
*	generate absorbing boundaries array
*
	  call weight(npx,npz,w)
*
*	read velocity model
*
	  call vmodel(npx,npz,vp2,vs2)
      pause'left vmodel'
*
*	dfx:inerval of wavenumber in kx-component 
*	dfz:inerval of wavenumber in kz-component 
*
	  dist=dxz*float(npx)
	  dfx=1./dist
	  dist=dxz*float(npz)
	  dfz=1./dist

	  call inplane(npx,npz,npx2,npz2,nn,w,vp2,vs2,dfx,dfz,
     *  isx,isz,t0,alpha,dt,ud,un,uf,wd,wn,wf,tmp1,tmp2,tmp3,
     * tmpx,tmpz,tmpp,nstep)

	  stop
	  end
	  
	  subroutine weight (npx,npz,w)
	  dimension w (npx,npz)
	  do 78 j=1,npz
	  do 78 i=1,npx
	  absorb=0
	  if (i.lt.17) absorb= (0.025*(16-i))**2
	  if (i.gt.(npx-17)) absorb=(0.025*(i-npx+16))**2
	  w(i,j)=exp(-absorb)*w(i,j)
78	  continue

      do 79 j=1,npz
	  do 79 i=1,npx
	  absorb=0
	  if (j.lt.17) absorb= (0.025*(16-j))**2
	  if (j.gt.(npz-17)) absorb= (0.025*(j-npz+16))**2
	  w(i,j)=exp(-absorb)*w(i,j)
79	  continue
	  return
	  end
	  
	  subroutine vmodel(npx,npz,vp2,vs2)
	  dimension vp2(npx,npz),vs2(npx,npz)
*
*	nb : number of media block to be read
*	ilx,ilz: low corner positon of marked block
* 	iux, Iuz: upper corner position of marked block
*	vpb: p-wave velocity of marked block (m/sec)
*	vsb: s-wave velocity of marked block (m/sec)
* 
	  
      vpb=2700.
      vsb=1500.
	  vsb2=(vsb*vsb)
	  vpb2=(vpb*vpb)
	  do 210 j=1,npz
	  do 210 i=1,npx
	  vs2(i,j)=vsb2
	  vp2(i,j)=vpb2
210   continue 


      vpb=4000.
      vsb=2200.
	  vsb2=(vsb*vsb)
	  vpb2=(vpb*vpb)
	  do 220 j=1,30
	  do 220 i=1,30
	  vs2(i,j)=vsb2
	  vp2(i,j)=vpb2
220   continue 
	  return
	  end
*
*	In-plane (P-SV) and Rayleigh waves) solution
*
	  subroutine inplane(npx,npz,npx2,npz2,nn,w,vp2,vs2,dfx,dfz,
     * isx,isz,t0,alpha,dt,ud,un,uf,wd,wn,wf,tmp1,tmp2,tmp3,
     * tmpx,tmpz,tmpp,nstep)
	  dimension ud(npx,npz),un(npx,npz),uf(npx,npz)
	  dimension wd(npx,npz),wn(npx,npz),wf(npx,npz)
	  dimension tmp1(npx2,npz),tmp2(npx2,npz),tmp3(npx2,npz)
	  dimension tmpx(npx2,npz),vp2(npx,npz),vs2(npx,npz)
	  dimension w(npx,npz),nn(2),tmpz(npx2,npz),tmpp(npx2,npz)
*
*	snap,pzv : output file for snapshots data
*	semi.psx, semi.psz : output files of seismic traces for selected
*	profiles
*

	  open(39,FILE='snap.psv',form='unformatted')
	  open(49,FILE='semi.psz',form='unformatted')
	  open(69,FILE='semi.psx',form='unformatted')
	  do 333 j=1,npz
	  do 333 i=1,npx
	  ud(i,j)=0
	  un(i,j)=0
	  uf(k,j)=0
	  wd(i,j)=0
	  wn(i,j)=0
333   wf(i,j)=0
	  do 789 istep=1,nstep
	  time=dt*istep

*
*	source input (displacement source)
*
	  call r1(amp,time,t0,alpha)
	  do 558 j=1,npz
	  do 558 i=1,npx
	  dj=(float(i-isx)**2+float(j-isz)**2)*0.1
	  wn(i,j)=amp*(float(j-isz))*exp(-dj)+wn(i,j)
	  un(i,j)=amp*(float(j-isx))*exp(-dj)+un(i,j)
558	  continue
	  do 402 j=1,npz
	  do 402 i=1,npx
	  i1=i*2-1
	  i2=i*2
	  tmpx(i2,j)=0.0
	  tmpx(i1,j)=un(i,j)
	  tmpz(i2,j)=0.0
402	  tmpz(i1,j)=wn(i,j)
*
*	spatial derivative computation
*
	  call four2(tmpx,nn,1,npx2)
	  call four2(tmpz,nn,1,npx2)
*
*	process for x-component
*
	  call iwmx2(dfx,dfz,npx,npz,tmpx,tmp1,npx2)
	  call iwmz2(dfx,dfz,npx,npz,tmpx,tmp2,npx2)
	  call iwmxz(dfx,dfz,npx,npz,tmpz,tmp3,npx2)
	  call four2(tmp1,nn,-1,npx2)
	  call four2(tmp2,nn,-1,npx2)
	  call four2(tmp3,nn,-1,npx2)
	  do 302 j=1,npz
	  do 302 i=1,npx
	  ii=i*2-1
302	  tmpp(ii,j)=tmp1(ii,j)*vp2(i,j)+tmp2(ii,j)*vs2(i,j)+
     *  tmp3(ii,j)*(vp2(i,j)-vs2(i,j))
*
* process for z-component
*
	  call iwmx2(dfx,dfz,npx,npz,tmpz,tmp2,npx2)
	  call iwmz2(dfx,dfz,npx,npz,tmpz,tmp1,npx2)
	  call iwmxz(dfx,dfz,npx,npz,tmpx,tmp3,npx2)
	  call four2(tmp1,nn,-1,npx2)
	  call four2(tmp2,nn,-1,npx2)
	  call four2(tmp3,nn,-1,npx2)
	  do 322 j=1,npz
	  do 322 i=1,npx
	  ii=i*2-1
322	  tmpz(ii,j)=tmp1(ii,j)*vp2(i,j)+tmp2(ii,j)*vs2(i,j)+
     *           tmp3(ii,j)*(vp2(i,j)-vs2(i,j))
*
*	time marching computation and absorbing boundaries
*
	  do 505 j=1,npz
	  do 505 i=1,npx
	  ii=i*2-1
	  ud(i,j)=tmpp(ii,j)*dt+ud(i,j)
	  wd(i,j)=tmpz(ii,j)*dt+wd(i,j)
505   continue
	  do 349 j=1,npz
	  do 349 i=1,npx
	  uf(i,j)=ud(i,j)*dt+un(i,j)
	  wf(i,j)=wd(i,j)*dt+wn(i,j)
349	  continue
	  do 459 j=1,npz
	  do 459 i=1,npx
	  un(i,j)=uf(i,j)*w(i,j)
459	  wn(i,j)=wf(i,j)*w(i,j)
*
*	output time traces for selected profiles
*
	  write(69) (un(i,71),wn(i,71),i=1,64)
	  write(49) (un(51,j),wn(51,j),j=1,64)
*
*	output snapshot wavefield at selected time steps
*
	  if(istep.lt.150) go to 779
	  if(mod(istep,50).ne.0) go to 779
	  do 589 j=1,npz
	  do 589 i=1,npx
	  write(39) un(i,j)
589   write(39) wn(i,j)
779   print *,'passed step ',istep
789	  continue
	  print *,'pass in-plane case'
	  return 
	  end

*
*	First deivative of a Gaussian function
*
	  subroutine r1(f,t,t0,alpha)
	  alp1=-1.0*alpha*(t-t0)*(t-t0)
	  ep=exp(alp1)
	  f=(t-t0)*ep
	  return
	  end
	  

	  subroutine iwmx2z2(ffx,ffz,nx,nz,tmp,npx2)	
	  dimension tmp(npx2,*)
	  pi=3.1415926
	  nzh=nz/2
	  nxh=nx/2
	  wx=2.*pi*ffx
	  wz=2.*pi*ffz
	  do 1 j=1,nz
	  j1=j
	  if(j.gt.nzh+1) j1=nz-j+2
	  akz2=(wz*float(j1-1))**2
	  do 1 i=1,nx
	  i1=i
	  i3=i*2-1
	  i4=i*2
	  if(i.gt.nxh+1)i1=nx-i+2
	  akx2=(wx+float(i1-1)**2)
	  tmp(i3,j)=-tmp(i3,j)*(akx2+akz2)
	  tmp(i4,j)=-tmp(i4,j)*(akx2+akz2)
1	  continue
	  return
	  end
	  

	  subroutine iwmx2(ffx,ffz,nx,nz,tmpi,tmpo,npx2)
	  dimension tmpi(npx2,*),tmpo(npx2,*)
	  pi=3.1415926
	  nxh=nx/2
	  wx=2.*pi*ffx
	  do 1 j=1,nz
	  do 1 i=1,nx
	  i1=i
	  i3=i*2-1
	  i4=i*2
	  if(i.gt.nxh+1)i1=nx-i+2
	  akx2=(wx*float(i1-1))**2
	  tmpo(i3,j)=-tmpi(i3,j)*(akx2)
	  tmpo(i4,j)=-tmpi(i4,j)*(akx2)
1	  continue
	  return
	  end
	 

	  subroutine iwmz2(ffx,ffz,nx,nz,tmpi,tmpo,npx2)
	  dimension tmpi(npx2,*),tmpo(npx2,*)
	  pi=3.1415926
	  nzh=nz/2
	  wz=2.*pi*ffz
	  do 1 j=1,nz
	  j1=j
	  if(j.gt.nzh+1) j1=nz-j+2
	  akz2=(wz*float(j1-1))**2
	  do 1 i=1,nx
	  i3=i*2-1	
	  i4=i*2
	  tmpo(i3,j)=-tmpi(i3,j)*(akz2)
	  tmpo(i4,j)=-tmpi(i4,j)*(akz2)
1	  continue
	  return
	  end
	 
	  
	  subroutine iwmxz(ffx,ffz,nx,nz,tmpi,tmpo,npx2)
	  dimension tmpi(npx2,*),tmpo(npx2,*)
	  pi=3.1415926
	  nzh=nz/2
	  nxh=nx/2
	  wx=2.*pi*ffx
	  wz=2.*pi*ffz
	  do 1 j=1, nz
	  j1=j
	  if(j.gt.nzh+1) j1=nz-j+2
	  akz=(wz*float(j1-1))
	  do 1 i=1,nx
	  i1=i
	  i3=i*2-1
	  i4=ii*2
	  if(i.gt.nxh+1)i1=nx-i+2
	  akx=(wx*float(i1-1))
	  tmpo(i3,j)=-tmpi(i3,j)*(akx*akz)
	  tmpo(i4,j)=-tmpi(i4,j)*(akx*akz)
	  if(i.gt.nxh+1.and.j.le.nzh+1) tmpo(i3,j)= tmpi(i3,j)*(akx*akz)
	  if(i.le.nxh+1.and.j.gt.nzh+1) tmpo(i3,j)= tmpi(i3,j)*(akx*akz)
	  if(i.gt.nxh+1.and.j.le.nzh+1) tmpo(i4,j)= tmpi(i4,j)*(akx*akz)
	  if(i.le.nxh+1.and.j.gt.nzh+1) tmpo(i4,j)= tmpi(i4,j)*(akx*akz)
1 	  continue
	  return
	  end
*
*	2-D FFT with respect to x and z component assembled by 1-D FFT
* 	(subroutine stock)
* 	
	  subroutine four2(tmp,nn,ii,npx2)
	  parameter(npx=64,npz=64)
	  complex work(npx),tmpx(npx),tmpz(npx)
	  dimension tmp(npx2,*),nn(2)
	  do 40 j=1,nn(2)
	  do 30 i=1,nn(1)
	  i1=i*2-1
	  i2=i*2
	  tmpx(i)=cmplx(tmp(i1,j),tmp(i2,j))
30 	  continue
	  call stock(ii,nn(1),tmpx,work)
	  do 50 i=1,nn(1)
	  i1=i*2-1
	  i2=i*2
	  tmp(i1,j)=real(tmpx(i))
	  tmp(i2,j)=aimag(tmpx(i))
50	  continue
40	  continue
	  do 70 i=1,nn(1)
	  i1=i*2-1
	  i2=i*2
	  do 60 j=1,nn(2)
	  tmpz(j)=cmplx(tmp(i1,j),tmp(i2,j))
60	  continue
	  call stock(ii,nn(2),tmpz,work)
	  do 80 j=1,nn(2)
	  tmp(i1,j)=real(tmpz(j))
	  tmp(i2,j)=aimag(tmpz(j))
80 	  continue
70	  continue
	  return
	  end
*
* 	FFT subroutine modified by Jiunyih Chen to swap the work area with 
* 	vector
*
	  subroutine stock(is,lc,c,work)
*
* 	is= +1/-1 for forward/backward transform
* 	lc= 2**n n should be an integer
* 	c(lc) = array to be transformed (inpput) and transformed (output)
* 	work(lc) = work area for internal use in stock
*
	  complex c(1),work(1)
	  ls=1
	  isw=0
100	  continue
	  if (ls.ge.lc) go to 31
	  ns=lc/(ls+ls)
	  if(isw.ne.0) go to 41
	  call stocks(is,ls,ns,c,work)
	  isw=1
	  go to 51
41	  continue
	  call stocks(is,ls,ns,work,c)
	  isw=0
51	  continue
	  ls=ls+ls
	  go to 100
31	  continue
	  if(isw.ne.1) go to 21
	  do 11 i=1,lc
	  c(i)=work(i)
11	  continue
21	  continue
	  if(is.ge.0) go to 22
	  ss=1./float(ls)
	  do 12 i=1,lc
	  c(i)=c(i)*ss
12	  continue
22	  continue
	  return
	  end
	  

	  subroutine stocks(is,ls,ns,c,ch)
	  complex omega,omegk,c(ns,2,ls),ch(ns,ls,2)
	  data pi/3.1415926/
	  angle=-float(is)*pi/float(ls)
	  omega=cmplx(cos(angle),sin(angle))
	  if(ls.le.ns.or.ns.ge.32) go to 21
	  do 13 j=1,ns
	  omegk=1.
	  do 14 i=1,ls
	  ch(j,i,1)=c(j,1,i)+omegk*c(j,2,i)
	  ch(j,i,2)=c(j,1,i)-omegk*c(j,2,i)
	  omegk=omega*omegk
14	  continue
13	  continue
	  return
21	  continue
	  omegk=1.
	  do 15 i=1,ls
	  do 16 j=1,ns
	  ch(j,i,1)=c(j,1,i)+omegk*c(j,2,i)
	  ch(j,i,2)=c(j,1,i)-omegk*c(j,2,i)
16	  continue
	  omegk=omega*omegk
15	  continue
	  return
	  end
 
	

	

	







	
	
 
	
