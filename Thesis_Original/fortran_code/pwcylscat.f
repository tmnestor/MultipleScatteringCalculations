      Subroutine PwCylScat(Np,M,Bs,w,L,alpha0,beta0,pvec,etavec,
     &           netavec,rtEa,rtEb,Tpp,Tps,Tsp,Tss,Xc,Y,Ytemp,SYtemp,SY)
      Implicit none
      Integer    M,Bs,Np
      Real*8     Xc,L,PPi,dk,kmin
      parameter  (PPi=3.14159265358979324D0)

      Complex*16 Y(Np,4),SY(Np,4),Ytemp(Bs,2),SYtemp(Bs,2)
      Complex*16 w,dp,alpha0,beta0,Pvec(Np)
      Complex*16 etavec(Np),netavec(np),rtea(np),rteb(np)
      Complex*16 Tpp(Bs),Tps(Bs),Tsp(Bs),Tss(Bs)
      
      
      
      Integer    il,ipi,ipo,mi,mo
      Complex*16 fac,facp,facs,pi,po
      Complex*16 etai,etao,netai,netao,dEx,Ex0,Exi,Exo
      Complex*16 eadi,ebdi,eaui,ebui
      Complex*16 eado,ebdo,eauo,ebuo
      Complex*16 Ci,C0,sump,sums,sumPU,sumSU,sumPD,sumSD
      Complex*16 rtetai,rtnetai,rtetao,rtnetao      
      Complex*16 phidai,phiuai,phidao,phiuao    
      Complex*16 phidbi,phiubi,phidbo,phiubo    

      Ci=(0.D0,1.D0)
      C0=(0.D0,0.D0)

       
      dk=2.D0*PPi/L
      kmin=Dble(w*pvec(1))
      fac=CDsqrt(2.D0/(w*L))
      
      
      il=0
      Do 1000 mi=-M,M
      il=il+1
      sump=C0
      sums=C0

      dEx=CDexp(Ci*dk*Xc)
      Exi=CDexp(Ci*kmin*Xc)
      Do 100 ipi=1,Np
        pi=pvec(ipi)
        etai=etavec(ipi)
        netai=netavec(ipi)
        rtetai=CDsqrt(etai)
        rtnetai=CDsqrt(netai)
        eadi=rtea(ipi)
        ebdi=rteb(ipi)
        eaui=rtea(ipi)
        ebui=rteb(ipi)
        facp=fac*Exi/(rtetai)
        facs=fac*Exi/(rtnetai)

        phidai=facp*eadi*(alpha0*(pi+Ci*etai))**mi
        phiuai=facp*eaui*(alpha0*(pi-Ci*etai))**mi
        sump=sump+phidai*Y(ipi,1)+phiuai*Y(ipi,3)
        
        phidbi=facs*ebdi*(beta0*(pi+Ci*netai))**mi
        phiubi=facs*ebui*(beta0*(pi-Ci*netai))**mi
        sums=sums+phidbi*Y(ipi,2)+phiubi*Y(ipi,4)
        Exi=dEx*Exi
100	  Continue
      Ytemp(il,1)=sump
      Ytemp(il,2)=sums

1000  Continue

      Do 1100 il=1,M
        SYtemp(il,1) = Tpp(il)*Y(il,1) + Tps(il)*Y(il,2)
        SYtemp(il,2) = Tsp(il)*Y(il,1) + Tss(il)*Y(il,2)
1100  Continue

      dEx=CDexp(-Ci*dk*Xc)
      Exo=CDexp(-Ci*kmin*Xc)
      Do 999  ipo=1,Np
	  po=pvec(ipo)
	  etao=etavec(ipo)
	  netao=netavec(ipo)
	  rtetao=CDsqrt(etao)
	  rtnetao=CDsqrt(netao)
	  
	  eado=rtea(ipo)
	  ebdo=rteb(ipo)
	  eauo=rtea(ipo)
	  ebuo=rteb(ipo)
	  facp=fac*Exo/(rtetao)
	  facs=fac*Exo/(rtnetao)
          
      sumPU=C0
      sumSU=C0
      sumPD=C0
      sumSD=C0
      il=0
	  Do 993 mo=-M,M
	    il=il+1
		phidao=facp*eado*(alpha0*(po+Ci*etao))**mo
		SumPU=SumPU+phidao*SYtemp(il,1)
		
		phidbo=facs*ebdo*(beta0*(po+Ci*netao))**mo
		SumSU=SumSU+phidbo*SYtemp(il,2)
		
		phiuao=facp*eauo*(alpha0*(po-Ci*etao))**mo
		SumPD=SumPD+phiuao*SYtemp(il,1)
		
		phiubo=facs*ebuo*(beta0*(po-Ci*netao))**mo
		SumSD=SumSD+phiubo*SYtemp(il,2)

993	  Continue
      SY(ipo,1)=SumPU
      SY(ipo,2)=SumSU
      SY(ipo,3)=SumPD
      SY(ipo,4)=SumSD
      Exo=dEx*Exo
999   Continue

      return
      end
