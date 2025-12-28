import util as yu
from util import *

#!============== General ==============#

#!============== Renomalization ==============#
if True:
    j2j1={
        'jq':[[1,'j+'],[1,'js'],[1,'jc']],
        'jv1':[[1,'j-']],
        'jv2':[[1,'j+'],[-2,'js']],
        'jv3':[[1,'j+'],[1,'js'],[-3,'jc']]
    }
    fla2iso={
        'u':[(1/4,'q'),(1/2,'v1'),(1/6,'v2'),(1/12,'v3')],
        'd':[(1/4,'q'),(-1/2,'v1'),(1/6,'v2'),(1/12,'v3')],
        's':[(1/4,'q'),(-1/3,'v2'),(1/12,'v3')],
        'c':[(1/4,'q'),(-1/4,'v3')]
    }
    fla2iso_conn={
        'u':[(1/2,'q'),(1/2,'v1')],
        'd':[(1/2,'q'),(-1/2,'v1')],
    }

    lat_a2s_plt=np.arange(0,0.009,0.001)

    def extendBare_avgx(key2bare):
        keys=list(key2bare)
        enss=yu.removeDuplicates([ens for ens,j in keys])
        stouts=yu.removeDuplicates([int(j.split('stout')[-1]) for ens,j in keys if 'stout' in j])
        
        for ens in enss:
            ZERO=key2bare[(ens,'j+;conn')]*0
            key2bare[(ens,'js;conn')]=ZERO
            key2bare[(ens,'jc;conn')]=ZERO
            key2bare[(ens,'j-;disc')]=ZERO
            for cd in ['conn','disc']:
                for j in j2j1.keys():
                    key2bare[(ens,f'{j};{cd}')]=np.sum([factor*key2bare[(ens,f'{j1};{cd}')] for factor,j1 in j2j1[j]],axis=0)
            
    def bareRC2phy_avgx(key2bare,ens2RCs,mn_conn='mu=nu',mn_disc='mu!=nu',mn_g='mu!=nu'):
        keys=list(key2bare)
        enss=yu.removeDuplicates([ens for ens,j in keys])
        stouts=yu.removeDuplicates([int(j.split('stout')[-1]) for ens,j in keys if 'stout' in j])
        
        key2phy={}
        for ens in enss:
            for j in ['jv1','jv2','jv3']:
                key2phy[(ens,f'{j};conn')]=ens2RCs[ens][f'Zqq({mn_conn})']*key2bare[(ens,f'{j};conn')]
                key2phy[(ens,f'{j};disc')]=ens2RCs[ens][f'Zqq({mn_disc})']*key2bare[(ens,f'{j};disc')]
                key2phy[(ens,f'{j}')]=key2phy[(ens,f'{j};conn')]+key2phy[(ens,f'{j};disc')]
                
            for stout in stouts:
                key2phy[(ens,f'jq;conn;stout{stout}')]=ens2RCs[ens][f'Zqq^s({mn_conn})']*key2bare[(ens,'jq;conn')]
                key2phy[(ens,f'jq;disc;stout{stout}')]=ens2RCs[ens][f'Zqq^s({mn_disc})']*key2bare[(ens,'jq;disc')]
                key2phy[(ens,f'jq;0mix;stout{stout}')]=key2phy[(ens,f'jq;conn;stout{stout}')]+key2phy[(ens,f'jq;disc;stout{stout}')]
                
                key2phy[(ens,f'jq;mix;stout{stout}')]=ens2RCs[ens][f'Zqg({mn_g})']*key2bare[(ens,f'jg;stout{stout}')]
                key2phy[(ens,f'jq;stout{stout}')]=key2phy[(ens,f'jq;0mix;stout{stout}')]+key2phy[(ens,f'jq;mix;stout{stout}')]
                
                key2phy[(ens,f'jg;0mix;stout{stout}')]=ens2RCs[ens][f'Zgg^{stout}({mn_g})']*key2bare[(ens,f'jg;stout{stout}')]
                key2phy[(ens,f'jg;mix;stout{stout}')]=ens2RCs[ens][f'Zgq({mn_conn})']*key2bare[(ens,'jq;conn')]+ens2RCs[ens][f'Zgq({mn_disc})']*key2bare[(ens,'jq;disc')]
                key2phy[(ens,f'jg;stout{stout}')]=key2phy[(ens,f'jg;mix;stout{stout}')]+key2phy[(ens,f'jg;0mix;stout{stout}')]
                
                key2phy[(ens,f'jtot;stout{stout}')]=key2phy[(ens,f'jq;stout{stout}')]+key2phy[(ens,f'jg;stout{stout}')]
                
                for fla in fla2iso.keys():
                    key2phy[(ens,f'j{fla};stout{stout}')]=np.sum([factor*(key2phy[(ens,f'j{iso};stout{stout}')] if iso in ['q'] else key2phy[(ens,f'j{iso}')])  for factor,iso in fla2iso[fla]],axis=0)
        
        for j in ['jv1','jv2','jv3']+[f'jq;stout{stout}' for stout in stouts]+[f'jg;stout{stout}' for stout in stouts]:
            ens2dat={ens:key2phy[(ens,j)] for ens in enss}
            fits=yu.doFits_continuumExtrapolation(ens2dat,lat_a2s_plt=lat_a2s_plt)
            for fit in fits:
                fitlabel,pars_jk,chi2_jk,Ndof=fit
                key2phy[(f'a=#_{fitlabel}',j)]=pars_jk
            pars_jk,probs_jk=yu.jackMA(fits)
            key2phy[('a=#_MA',j)]=pars_jk
        
        for stout in stouts:
            for fitlabel in ['const','linear','MA']:
                for fla in fla2iso.keys():
                    key2phy[(f'a=#_{fitlabel}',f'j{fla};stout{stout}')]=np.sum([factor*(key2phy[(f'a=#_{fitlabel}',f'j{iso};stout{stout}')] if iso in ['q'] else key2phy[(f'a=#_{fitlabel}',f'j{iso}')]) for factor,iso in fla2iso[fla]],axis=0)
                key2phy[(f'a=#_{fitlabel}',f'jtot;stout{stout}')]=key2phy[(f'a=#_{fitlabel}',f'jq;stout{stout}')]+key2phy[(f'a=#_{fitlabel}',f'jg;stout{stout}')]
        
        return key2phy

    def bareRC2phy_avgx_pre(key2bare,ens2RCs,ens2RCs_pre,mn_conn='mu=nu',mn_disc='mu!=nu',mn_g='mu!=nu'):
        keys=list(key2bare)
        enss=yu.removeDuplicates([ens for ens,j in keys]); enss=['b']
        stouts=yu.removeDuplicates([int(j.split('stout')[-1]) for ens,j in keys if 'stout' in j])
        
        key2phy={}
        for ens in enss:
            for j in ['jv1','jv2','jv3']:
                key2phy[(ens,f'{j};conn')]=ens2RCs[ens][f'Zqq({mn_conn})']*key2bare[(ens,f'{j};conn')]
                key2phy[(ens,f'{j};disc')]=ens2RCs[ens][f'Zqq({mn_disc})']*key2bare[(ens,f'{j};disc')]
                key2phy[(ens,f'{j}')]=key2phy[(ens,f'{j};conn')]+key2phy[(ens,f'{j};disc')]
                
            for stout in stouts:
                key2phy[(ens,f'jq;conn;stout{stout}')]=ens2RCs[ens][f'Zqq^s({mn_conn})']*key2bare[(ens,'jq;conn')]
                key2phy[(ens,f'jq;disc;stout{stout}')]=ens2RCs_pre[ens][f'Zqq^s^{stout}({mn_disc})']*key2bare[(ens,'jq;disc')]
                key2phy[(ens,f'jq;0mix;stout{stout}')]=key2phy[(ens,f'jq;conn;stout{stout}')]+key2phy[(ens,f'jq;disc;stout{stout}')]
                
                key2phy[(ens,f'jq;mix;stout{stout}')]=ens2RCs_pre[ens][f'Zqg^{stout}({mn_g})']*key2bare[(ens,f'jg;stout{stout}')]
                key2phy[(ens,f'jq;stout{stout}')]=key2phy[(ens,f'jq;0mix;stout{stout}')]+key2phy[(ens,f'jq;mix;stout{stout}')]
                
                key2phy[(ens,f'jg;0mix;stout{stout}')]=ens2RCs_pre[ens][f'Zgg^{stout}({mn_g})']*key2bare[(ens,f'jg;stout{stout}')]
                key2phy[(ens,f'jg;mix;stout{stout}')]=ens2RCs[ens][f'Zgq({mn_conn})']*key2bare[(ens,'jq;conn')]+ens2RCs_pre[ens][f'Zgq^{stout}({mn_disc})']*key2bare[(ens,'jq;disc')]
                key2phy[(ens,f'jg;stout{stout}')]=key2phy[(ens,f'jg;mix;stout{stout}')]+key2phy[(ens,f'jg;0mix;stout{stout}')]
                
                key2phy[(ens,f'jtot;stout{stout}')]=key2phy[(ens,f'jq;stout{stout}')]+key2phy[(ens,f'jg;stout{stout}')]
                
                for fla in fla2iso.keys():
                    key2phy[(ens,f'j{fla};stout{stout}')]=np.sum([factor*(key2phy[(ens,f'j{iso};stout{stout}')] if iso in ['q'] else key2phy[(ens,f'j{iso}')])  for factor,iso in fla2iso[fla]],axis=0)        
        
        return key2phy
    
#!============== Plot ==============#
if True:
    def convert_key2phy_stout(key2phy_old,stout):
        key2phy={}
        for key in key2phy_old.keys():
            ens,j=key
            if j.endswith(f';stout{stout}'):
                key2phy[(ens,j.removesuffix(f';stout{stout}'))]=key2phy_old[key]
            if 'stout' not in j:
                key2phy[key]=key2phy_old[key]
        return key2phy
    def convert_key2phy_stouts(key2phy_old,stouts):
        key2phy={}
        for key in key2phy_old.keys():
            ens,j=key
            if j.endswith(f';stout{stouts[0]}'):
                j0=j.removesuffix(f';stout{stouts[0]}')
                key2phy[(ens,j0)]=np.mean([key2phy_old[(ens,f'{j0};stout{stout}')] for stout in stouts],axis=0)
            if 'stout' not in j:
                key2phy[key]=key2phy_old[key]
        return key2phy                

    def makePlot_a2dependence_avgx(list_dic):
        Ncol=len(list_dic)
        fig, axs = yu.getFigAxs(2,Ncol,Lrow=4,Lcol=6,sharex=True,sharey='row')
        ax=axs[0,0]
        ax.set_xlim([0,lat_a2s_plt[-1]])
        ax.set_xticks([0,0.003,0.006])
        ax.set_ylim([0.2,1.4])
        ax.set_yticks([0.4,0.6,0.8,1.0,1.2])
        ax.set_ylabel(r'$\langle \mathrm{x} \rangle_{q,g}$')
        ax=axs[1,0]
        ax.set_ylim([-0.1,0.5])
        ax.set_ylabel(r'$\langle \mathrm{x}\rangle_{q}$')
        for icol in range(Ncol):
            axs[0,icol].axhline(1,color='black',ls='--',marker='')
            axs[0,icol].axvline(0,color='black',ls='dotted',marker='')
            axs[1,icol].set_xlabel(r'$a^2$ [fm$^2$]')
            axs[1,icol].axhline(0,color='black',ls='--',marker='')
            
        for icol in range(Ncol):
            dic=list_dic[icol]
            def setParameter(default,key):
                return dic[key] if key in dic else default
            
            key2phy=dic['key2phy']
            key2phy_pre=setParameter(None,'key2phy_pre')
            
            keys=list(key2phy)
            enss=yu.removeDuplicates([ens for ens,j in keys if 'a=#' not in ens])
            
            def get(ens,j):
                return key2phy[(ens,j)]
            def get_pre(ens,j):
                return key2phy_pre[(ens,j)]
            
            j2color={'jq':'purple','jg':'cyan','jtot':'grey','ju':'r','jd':'g','js':'b','jc':'orange'}
            j2label={'jq':'q','jg':'g','jtot':'N','ju':'u','jd':'d','js':'s','jc':'c'}
            j2fmt={'jq':'d','jg':'s','jtot':'o','ju':'^','jd':'v','js':'<','jc':'>'}
            
            
            ax=axs[0,icol]
            js=['jq','jtot','jg']
            for ij,j in enumerate(js):
                color=j2color[j]
                mean,err=yu.jackme(get('a=#_MA',j))
                label=rf'$\langle x\rangle _{{{j2label[j]}}}=$'+yu.un2str(mean[0],err[0],forceResult=1)
                for iens,ens in enumerate(enss):
                    plt_x=yu.ens2a[ens]**2+(ij-len(js)/2)*5e-5; plt_y,plt_yerr=yu.jackme(get(ens,j))
                    ax.errorbar(plt_x,plt_y,plt_yerr,color=color,fmt=j2fmt[j],label=label if iens==0 else None)
                    
                    if key2phy_pre is None or ens not in ['b']:
                        continue
                    plt_x=yu.ens2a[ens]**2+0.0001; plt_y,plt_yerr=yu.jackme(get_pre(ens,j))
                    ax.errorbar(plt_x,plt_y,plt_yerr,color=color,fmt=j2fmt[j],mfc='white')
                        
                mean,err=yu.jackme(get('a=#_MA',j))
                x=lat_a2s_plt; ymin=mean-err; ymax=mean+err
                ax.plot(x,mean,color=color,linestyle='--',marker='')
                ax.fill_between(x, ymin, ymax, color=color, alpha=0.1)
            ax.legend(fontsize=10,ncol=2)

            ax=axs[1,icol]
            js=['ju','jd','js','jc']
            for ij,j in enumerate(js):
                color=j2color[j]
                mean,err=yu.jackme(get('a=#_MA',j))
                label=rf'$\langle x\rangle _{{{j2label[j]}}}=$'+yu.un2str(mean[0],err[0],forceResult=1)
                for iens,ens in enumerate(enss):
                    plt_x=yu.ens2a[ens]**2+(ij-len(js)/2)*5e-5; plt_y,plt_yerr=yu.jackme(get(ens,j))
                    ax.errorbar(plt_x,plt_y,plt_yerr,color=color,fmt=j2fmt[j],label=label if iens==0 else None)
                    if key2phy_pre is None or ens not in ['b']:
                        continue
                    plt_x=yu.ens2a[ens]**2+0.0001; plt_y,plt_yerr=yu.jackme(get_pre(ens,j))
                    ax.errorbar(plt_x,plt_y,plt_yerr,color=color,fmt=j2fmt[j],mfc='white')

                mean,err=yu.jackme(get('a=#_MA',j))
                x=lat_a2s_plt; ymin=mean-err; ymax=mean+err
                ax.plot(x,mean,color=color,linestyle='--',marker='')
                ax.fill_between(x, ymin, ymax, color=color, alpha=0.1)
            ax.legend(fontsize=10,ncol=2)
        
        return fig,axs

#!============== Rotation ==============#
if True:
    from itertools import permutations
    
    elements=[(sx,sy,sz,xyz) for sx in [1,-1] for sy in [1,-1] for sz in [1,-1] for xyz in permutations([0, 1, 2], 3)] # Permute first Flip next    
    def rotate_mom(e,mom):
        sx,sy,sz,xyz=e; ix,iy,iz=xyz; iix,iiy,iiz=tuple([ix,iy,iz].index(i) for i in range(3))
        return [sx*mom[iix],sy*mom[iiy],sz*mom[iiz],sx*mom[iix+3],sy*mom[iiy+3],sz*mom[iiz+3]]
    def mom2moms(mom):
        moms=list(set([tuple(rotate_mom(e,mom)) for e in elements])) 
        moms.sort()
        moms=np.array(moms)
        return moms
    def mom2standard(mom):
        moms=list(set([tuple(rotate_mom(e,mom)) for e in elements])) 
        return list(max(moms, key=lambda x: x[::-1]))
    def mom_exchangeSourceSink(mom):
        p1x,p1y,p1z,qx,qy,qz=mom
        px,py,pz=p1x+qx,p1y+qy,p1z+qz
        return [px,py,pz,-qx,-qy,-qz]
    def mom2standard_exchangeSourceSink(mom):
        momE=mom_exchangeSourceSink(mom)
        moms=list(set([tuple(rotate_mom(e,mom)) for e in elements]+[tuple(rotate_mom(e,momE)) for e in elements])) 
        return list(max(moms, key=lambda x: x[::-1]))
    
#!============== Form factor decomposition ==============#
if True:
    import sympy as sp
    from sympy import sqrt
    from itertools import permutations
    
    projs=['P0', 'Px', 'Py', 'Pz']
    inserts=['tt', 'tx', 'ty', 'tz', 'xx', 'xy', 'xz', 'yy', 'yz', 'zz']
    
    funcs_ri=[np.real,np.imag]

    id=np.eye(4)
    g1=np.array([[0, 0, 0, 1j],
                [0, 0, 1j, 0],
                [0, -1j, 0, 0],
                [-1j, 0, 0, 0]])

    g2=np.array([[0, 0, 0, 1],
                [0, 0, -1, 0],
                [0, -1, 0, 0],
                [1, 0, 0, 0]])

    g3=np.array([[0, 0, 1j, 0],
                [0, 0, 0, -1j],
                [-1j, 0, 0, 0],
                [0, 1j, 0, 0]])

    g4=np.array([[1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, -1, 0],
                [0, 0, 0, -1]])

    g5 = g1@g2@g3@g4
    gm = np.array([g1, g2, g3, g4])
    sgm = np.array([[(gm[mu]@gm[nu] - gm[nu]@gm[mu])/2 for nu in range(4)] for mu in range(4)])

    G0 = (id + g4) / 4
    G1 = 1j * g5 @ g1 @ G0
    G2 = 1j * g5 @ g2 @ G0
    G3 = 1j * g5 @ g3 @ G0
    G = [G1, G2, G3, G0]

    insert2ind={'x':0,'y':1,'z':2,'t':3}
    def ME2FF(m,pvec,pvec1,proj,insert):
        Gn={'P0':G0,'Px':G1,'Py':G2,'Pz':G3}[proj]
        mu,nu=insert
        mu=insert2ind[mu]; nu=insert2ind[nu]
        
        px,py,pz=pvec
        p1x,p1y,p1z=pvec1
        
        if m==sp.symbols('m'):
            pt=sp.symbols('pt')
            p1t=sp.symbols('p1t')
        else:
            pt=1j*np.sqrt(px**2+py**2+pz**2+m**2)
            p1t=1j*np.sqrt(p1x**2+p1y**2+p1z**2+m**2)
            
        p=np.array([px,py,pz,pt])
        p1=np.array([p1x,p1y,p1z,p1t])

        pS=np.sum(gm*p[:,None,None],axis=0)
        p1S=np.sum(gm*p1[:,None,None],axis=0)
        Px, Py, Pz, Pt = p + p1
        qx, qy, qz, qt = p - p1
        P=np.array([Px,Py,Pz,Pt])
        q=np.array([qx,qy,qz,qt])
        Q2 = -2*m**2 - 2*p1.dot(p)
        
        #==============================
        factorA= 1j; factorB= -1j; factorC=1
        factorBase=sp.symbols('K')/(4*m**2); factorSgm=1
        
        if m!=sp.symbols('m'):
            xE=pt/1j; xE1=p1t/1j
            factorBase=1/np.sqrt(2*xE1*(xE1+m)*2*xE*(xE+m))
        
        la=(gm[mu]*P[nu]/2+gm[nu]*P[mu]/2)/2-(np.sum(gm*P[:,None,None]/2,axis=0))*id[mu,nu]/4
        lb=(1j/(2*m))*((np.einsum('rab,r->ab',sgm[mu],q)*P[nu]/2+np.einsum('rab,r->ab',sgm[nu],q)*P[mu]/2)/2-np.einsum('srab,r,s->ab',sgm,q,P/2)*id[mu,nu]/4)*factorSgm
        lc=(id/m)*(q[mu]*q[nu]-Q2/4*id[mu,nu])
        
        res=np.array([factorBase*factor*np.trace(Gn@(-1j*p1S+m*id)@Lambda@(-1j*pS+m*id)) for Lambda,factor in zip([la,lb,lc],[factorA,factorB,factorC])])
        
        if m==sp.symbols('m'):
            xE = sp.symbols('E')
            xE1 = sp.symbols('E1')
            xE1=xE=sqrt(px**2+py**2+pz**2+m**2)
            for t in res:
                # t=t.subs({p1x:0,p1y:0,p1z:0,p1t:1j*m,pt:1j*xE})
                # t=t.subs({px:sqrt(xE**2-m**2-py**2-pz**2)})            
                t=t.subs({p1t:1j*xE1,pt:1j*xE})
                if px==p1x and py==p1y and pz==p1z:
                    t=t.subs({sp.symbols('K'):(4*m**2)/(2*xE*(xE+m))})
                t=sp.expand(sp.sympify(t))
                print(t)
            print()
            return
        
        return res

    def nonzeroQ(mom,proj,insert):
        n1vec=np.array(mom[:3]); nqvec=np.array(mom[3:6])
        nvec=n1vec+nqvec
        
        m=0.5; L=64
        pvec=nvec*(2*np.pi/L); p1vec=n1vec*(2*np.pi/L)
        
        res=ME2FF(m,pvec,p1vec,proj,insert)
        tr=np.sum(np.abs(np.real(res))); ti=np.sum(np.abs(np.imag(res)))
        threshold=1e-8
        return (tr>threshold,ti>threshold)

    def rotateMPI(rot,mom,proj,insert):
        sx,sy,sz,xyz=rot; signs=[sx,sy,sz,1]
        ix,iy,iz=xyz; iix,iiy,iiz=tuple([ix,iy,iz].index(i) for i in range(3))
        xyzt=['x','y','z','t']
        xyzt2={'x':xyzt[ix],'y':xyzt[iy],'z':xyzt[iz],'t':'t'}
        
        mom1=[sx*mom[iix],sy*mom[iiy],sz*mom[iiz],sx*mom[iix+3],sy*mom[iiy+3],sz*mom[iiz+3]]
        proj1='P0' if proj=='P0' else f'P{xyzt2[proj[1]]}'
        insert1=f'{xyzt2[insert[0]]}{xyzt2[insert[1]]}'
        insert1=insert1 if insert1 in inserts else insert1[1]+insert1[0]
        return [mom1,proj1,insert1]
    
    def mpi2standard_pi(mom,proj,insert):
        assert(np.all(mom==mom2standard(mom)))
        mpis=[rotateMPI(e,mom,proj,insert) for e in elements]
        pis=[(p,i) for m,p,i in mpis if np.all(m==mom)]
        pis.sort(key=lambda mpi:''.join(mpi))
        return pis[-1]

    def useQ(mom,proj,insert):
        r,i=nonzeroQ(mom,proj,insert)
        if (r,i)==(False,False):
            return (False,False)
        if insert == 'tt': # traceless makes tt=-xx-yy-zz
            return (False,False)
        
        if (proj,insert) != mpi2standard_pi(mom,proj,insert):
            return (False,False)
        return (r,i)

    def useQ_noAvg(mom,proj,insert):
        r,i=nonzeroQ(mom,proj,insert)
        if (r,i)==(False,False):
            return (False,False)
        if insert == 'tt': # traceless makes tt=-xx-yy-zz
            return (False,False)
        
        return (r,i)