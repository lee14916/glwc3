import os,h5py,warnings,pickle,functools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.backends.backend_pdf import PdfPages
from math import floor, log10
from scipy.optimize import leastsq, fsolve
from scipy.linalg import cholesky
from scipy.stats import chi2 as chi2_dist
from IPython.display import display,HTML
from itertools import product
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

np.seterr(invalid=['ignore','warn'][0])
np.set_printoptions(legacy='1.25')

mpl.style.use('default')
mpl.rcParams['figure.facecolor'] = 'white'
mpl.rcParams['figure.titlesize'] = 20
mpl.rcParams['figure.figsize'] = [6.4*1.2,4.8*1.2]
mpl.rcParams['axes.labelsize'] = 24
mpl.rcParams['axes.titlesize'] = 24
mpl.rcParams['lines.marker'] = 's'
mpl.rcParams['lines.linestyle'] = ''
mpl.rcParams['lines.markersize'] = 7
mpl.rcParams['errorbar.capsize'] = 6
mpl.rcParams['xtick.labelsize'] = mpl.rcParams['ytick.labelsize'] = 22
mpl.rcParams['xtick.major.size'] = mpl.rcParams['ytick.major.size'] = 10
mpl.rcParams['xtick.top']=mpl.rcParams['ytick.right']=True
mpl.rcParams['xtick.direction']=mpl.rcParams['ytick.direction']='in'
mpl.rcParams['legend.fontsize'] = 16
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"

__all__ = ['np','os','mpl','plt','h5py','pickle','pd','display','product','defaultdict']

#!============== Initialization ==============#
if True:
    flag_fast=False # If True, certain functions will be speeded up using approximations.
    path_fig_internal=None; path_pkl_internal=None; path_fig=None; path_pkl=None
    def setpath(scriptname):
        global path_fig_internal, path_pkl_internal, path_fig, path_pkl
        path_fig_internal=f'fig/{scriptname}/internal_ignore/'
        path_pkl_internal=f'pkl/{scriptname}/internal_ignore/'
        path_fig=f'fig/{scriptname}/reg_ignore/'
        path_pkl=f'pkl/{scriptname}/reg_ignore/'
        
        for path in [path_fig_internal, path_pkl_internal, path_fig, path_pkl]:
            os.makedirs(path,exist_ok=True)

#!============== small functions ==============#
if True:
    deepKey=lambda dic,n: dic if n==0 else deepKey(dic[list(dic.keys())[0]],n-1)
    npRound=lambda dat,n:np.round(np.array(dat).astype(float),n)
    
    # c2pt2meff=lambda c2pt:np.log(np.roll(c2pt,1,axis=0)/c2pt) # use this one should also change the definition for func_meff_nst
    c2pt2meff=lambda c2pt:np.log(c2pt/np.roll(c2pt,-1,axis=0))
    nvec2n2= lambda nvec:nvec[0]**2+nvec[1]**2+nvec[2]**2
    
    def c2pt2meff_pion(C2pt,totalT,tau=1):
        tfunc_C=lambda m,t:np.exp(-m*t)+np.exp(-m*(totalT-t))
        t=C2pt # dim=time
        t=(np.roll(t,-tau,axis=0))/(t)
        t=np.array([fsolve2(lambda m:((tfunc_C(m,xt+tau))/(tfunc_C(m,xt)))-ele, 0.0593) for xt,ele in enumerate(t)])
        return t
    
    def symmetrizeRatio(tf2ratio):
        for tf in tf2ratio.keys():
            tf2ratio[tf]=(tf2ratio[tf]+tf2ratio[tf][:,::-1])/2
        return tf2ratio

    def fsolve2(func,x0):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res=fsolve(func, x0)[0]
        return res if res!=x0 else np.NaN
    
    cfg2old=lambda cfg: cfg[1:]+'_r'+{'a':'0','b':'1','c':'2','d':'3'}[cfg[0]]
    cfg2new=lambda cfg: {'0':'a','1':'b','2':'c','3':'d'}[cfg[-1]] + cfg[:4]
        
    def decodeList(l):
        return [ele.decode() for ele in l]
    def removeDuplicates(l):
        return list(set(l))
    def list2dic(l):
        dic={}
        for i,ele in enumerate(l):
            dic[ele]=i
        return dic
    
    def formatList(l,fmt='>8.4g',toStringQ=True):
        l=[format(ele,fmt) for ele in l]
        if toStringQ:
            l=f"[{','.join(l)}]"
        return l
    def formatMatrix(m,fmt='>8.4g',toStringQ=True):
        m=[[format(ele,fmt) for ele in row] for row in m]
        if toStringQ:
            m='\n'.join([(' '.join(row)) for row in m])
        return m
    
    def any2filename(t):
        if type(t)==str:
            return t
        if type(t)==list:
            return ';'.join(t)
        1/0
    def save_pkl(file,res):
        with open(file,'wb') as f:
            pickle.dump(res,f)
    def save_txt(file,txt):
        if type(txt)==list:
            txt='\n'.join(txt)
        with open(file,'w') as f:
            f.write(txt)
    def load_pkl(file):
        if not os.path.isfile(file):
            return None        
        with open(file,'rb') as f:
            res=pickle.load(f)
        return res
    def save_pkl_internal(label,res):
        if path_pkl_internal is None:
            print('path_pkl_internal is None, stop saving')
            return False
        save_pkl(f'{path_pkl_internal}{any2filename(label)}.pkl',res)
        return True
    def save_txt_internal(label,txt):
        save_txt(f'{path_pkl_internal}{any2filename(label)}.txt',txt)
    def load_pkl_internal(file):
        if path_pkl_internal is None:
            print('path_pkl_internal is None, stop loading')
            return None
        res=load_pkl(f'{path_pkl_internal}{any2filename(file)}.pkl')
        return res
    def save_pkl_reg(label,res,mkdirQ=False):
        file=f'{path_pkl}{any2filename(label)}.pkl'
        if mkdirQ:
            os.makedirs(os.path.dirname(file), exist_ok=True)
        save_pkl(file,res)
    def save_txt_reg(label,txt,mkdirQ=False):
        file=f'{path_pkl}{any2filename(label)}.txt'
        if mkdirQ:
            os.makedirs(os.path.dirname(file), exist_ok=True)
        save_txt(file,txt)
    def load_pkl_reg(label):
        return load_pkl(f'{path_pkl}{label}.pkl')
    def clear_pkl_internal(file):
        if os.path.isfile(f'{path_pkl_internal}{any2filename(file)}.pkl'):
            os.remove(f'{path_pkl_internal}{any2filename(file)}.pkl')
    
    def cut_tf2ratio(tf2ratio,tfmax,tfmin=0):
        return {tf:tf2ratio[tf] for tf in tf2ratio.keys() if tfmin<=tf<=tfmax}
    
    def removeError(dat):
        t=np.mean(dat,axis=0)
        return dat*0 + t[None,...]
            
    def print_hdf5_structure(file):
        with h5py.File(file, "r") as f:
            def show(tf):                
                keys=list(tf.keys()); keys.sort()
                keys_dataset=[]; keys_group=[]
                for key in keys:
                    if isinstance(tf[key], h5py.Dataset):
                        keys_dataset.append(key)
                    else:
                        keys_group.append(key)

                show_group=f'{len(keys_group)} groups, {keys_group[:4]}; ' if len(keys_group)!=0 else ''
                show_dataset=f'{len(keys_dataset)} datasets, {keys_dataset[:4]}; ' if len(keys_dataset)!=0 else ''
                print(f'{tf.name}: {show_group}{show_dataset}')
                
                keys_dataset=keys_dataset[:1] if len(keys_dataset)>4 else keys_dataset
                for key in keys_dataset:
                    show_key=f'; {tf[key][:]}' if ( (tf[key].ndim==1 and len(tf[key])<20) or (tf[key].ndim==0) ) else ''
                    if key in ['notes','inserts']:
                        print(f'{key}={decodeList(tf[key][:])}')
                    else:
                        print(f'{key}: {tf[key].shape}{show_key}')

                keys_group=keys_group[:1] if len(keys_group)>4 else keys_group
                for key in keys_group:
                    show(tf[key])

            show(f)

#!============== Parallelization ==============#
if True:
    def parallelizeTasks(run, tasks, max_workers=12):
        results = [None] * len(tasks)
        
        if max_workers == 1:
            for idx, task in enumerate(tasks):
                results[idx] = run(task)
                print(f"{idx+1}/{len(tasks)} tasks completed", end='          \r', flush=True)
        else:
            with ProcessPoolExecutor(max_workers=max_workers) as ex:
                futures = [ex.submit(run, task) for task in tasks]
                fut_to_idx = {fut: idx for idx, fut in enumerate(futures)}
                for i, fut in enumerate(as_completed(futures), 1):
                    idx = fut_to_idx[fut]
                    results[idx] = fut.result()
                    print(f"{i}/{len(tasks)} tasks completed", end='          \r', flush=True)
        print()
        return results
    
#!============== mom opearations ==============#
if True:
    def removeDuplicates_moms(moms):
        return [list(mom) for mom in {tuple(mom) for mom in moms}]
    
    def moms2dic(moms):
        dic={}
        for i,mom in enumerate(moms):
            dic[tuple(mom)]=i
        return dic
    def moms2list(moms):
        return [list(mom) for mom in moms]
    def mom2str(mom):
        return ','.join([str(ele) for ele in mom])
    def str2mom(momstr):
        return [int(ele) for ele in momstr.split(',')]
    
    def mom2msq(mom):
        assert(len(mom)==3)
        return mom[0]**2+mom[1]**2+mom[2]**2
    def getSortkey_mom(mom): # @mom2standard
        msq=mom2msq(mom)
        return (msq,-mom[2],-mom[1],-mom[0])
    def mom2standard(mom): # @getSortkey_mom
        return sorted([abs(mom[0]),abs(mom[1]),abs(mom[2])]) 
    
    elements_rot48=[(sx,sy,sz,xyz) for sx in [1,-1] for sy in [1,-1] for sz in [1,-1] for xyz in [(0,1,2),(0,2,1),(1,0,2),(1,2,0),(2,0,1),(2,1,0)]]
    def rotate_mom(e,mom):
        if len(mom)!=3:
            return rotate_mom(e,mom[:3])+rotate_mom(e,mom[3:])
        sx,sy,sz,xyz=e; ix,iy,iz=xyz; iix,iiy,iiz=tuple([ix,iy,iz].index(i) for i in range(3))
        return [sx*mom[iix],sy*mom[iiy],sz*mom[iiz]]
    def mom2moms(mom):
        return [list(mom) for mom in {tuple(rotate_mom(e,mom)) for e in elements_rot48}] 
    
    #==== for mom3pt=[p1,q] ====#
    def get_moms(max_mom2_pc,max_mom2_pf):
        range_xyz=range(0,int(np.sqrt(max_mom2_pc))+2)
        moms_pc=[[x,y,z] for x in range_xyz for y in range_xyz for z in range_xyz if x**2+y**2+z**2<=max_mom2_pc]

        range_xyz=range(-int(np.sqrt(max_mom2_pf))-1,int(np.sqrt(max_mom2_pf))+2)
        moms_pf=[[x,y,z] for x in range_xyz for y in range_xyz for z in range_xyz if x**2+y**2+z**2<=max_mom2_pf]
        
        moms=[pf+pc for pf in moms_pf for pc in moms_pc]
        moms=[list(ele) for ele in set([tuple(mom3pt2standard(mom)) for mom in moms])]
        moms=sorted(moms,key=getSortkey_mom3pt)
        return moms

    def n2qpp12str(n2qpp1):
        return ','.join([str(ele) for ele in n2qpp1])
    def str2n2qpp1(n2qpp1_str):
        return [int(ele) for ele in n2qpp1_str.split(',')]
    def mom2n2qpp1(mom):
        p1x,p1y,p1z,qx,qy,qz=mom
        px,py,pz=p1x+qx,p1y+qy,p1z+qz
        
        q2=qx**2+qy**2+qz**2
        p2=px**2+py**2+pz**2
        p12=p1x**2+p1y**2+p1z**2
        return (q2,p2,p12)
    def mom2n2qpp1_sym(mom):
        (q2,p2,p12)=mom2n2qpp1(mom)
        return (q2,p2,p12) if p2>=p12 else (q2,p12,p2)
        
    def mom_exchangeSourceSink(mom):
        p1x,p1y,p1z,qx,qy,qz=mom
        px,py,pz=p1x+qx,p1y+qy,p1z+qz
        return [px,py,pz,-qx,-qy,-qz]
    
    def getSortkey_mom3pt_internal(mom):
        return (-mom[2+3],-mom[1+3],-mom[0+3],-mom[2],-mom[1],-mom[0])
    def mom3pt2standard_internal(mom):
        return min(mom2moms(mom),key=getSortkey_mom3pt_internal)
    def getSortkey_mom3pt(mom):
        p1x,p1y,p1z,qx,qy,qz=mom
        px,py,pz=p1x+qx,p1y+qy,p1z+qz
        
        q2=qx**2+qy**2+qz**2
        p2=px**2+py**2+pz**2
        p12=p1x**2+p1y**2+p1z**2
        
        if p2>p12:
            exchangeQ=0
            return (q2,-p2,p12,exchangeQ,getSortkey_mom3pt_internal(mom))
        momE=mom_exchangeSourceSink(mom)
        if p2<p12:
            exchangeQ=1
            return (q2,-p12,p2,exchangeQ,getSortkey_mom3pt_internal(momE))
        
        mom_std=mom3pt2standard_internal(mom)
        momE_std=mom3pt2standard_internal(momE)
        if getSortkey_mom3pt_internal(mom_std)<=getSortkey_mom3pt_internal(momE_std):
            exchangeQ=0
            return (q2,-p2,p12,exchangeQ,getSortkey_mom3pt_internal(mom))
        else:
            exchangeQ=1
            return (q2,-p12,p2,exchangeQ,getSortkey_mom3pt_internal(momE))
    def mom3pt2standard(mom):
        moms=mom2moms(mom)
        return min(moms,key=getSortkey_mom3pt)
    def mom3pt2standard_sym(mom):
        moms=mom2moms(mom)+mom2moms(mom_exchangeSourceSink(mom))
        return min(moms,key=getSortkey_mom3pt)
        
#!============== constants ==============#
if True:
    hbarc = 1/197.3
    m_proton,m_neutron=938.2721,939.5654
    m_avgpn=(m_proton+m_neutron)/2

#!============== error analysis ==============#
if True:
    def propagateError(func,mean,cov):
        '''
        Linear propagation of uncertainty
        y = func(x) = func(mean) + A(x-mean) + nonlinear terms; x~(mean,cov)
        '''
        y_mean=func(mean)
        AT=[]
        for j in range(len(mean)):
            unit=np.sqrt(cov[j,j])/1000
            ej=np.zeros(len(mean)); ej[j]=unit
            AT.append((np.array(func(mean+ej))-y_mean)/unit)
        AT=np.array(AT)
        A=AT.T
        y_cov=A@cov@AT
        return (np.array(y_mean),np.sqrt(np.diag(y_cov)),y_cov)
    def cov2correlation(cov):
        errs=np.sqrt(np.diag(cov))
        rho=cov/np.outer(errs, errs)
        return rho
    def getCDR(cov):
        rho=cov2correlation(cov)
        evals=np.linalg.eigvalsh(rho)
        eval_max=np.max(evals); eval_min=np.min(evals)
        kappa=eval_max/eval_min
        CDR=10*np.log10(kappa)
        return CDR
    def jackknife(dat,d:int=0,nmin:int=6000):
        n=len(dat)
        if flag_fast:
            d=n//300
        elif d==0:
            d=n//nmin
        d=max(d,1)
        
        if d!=1:
            def tfunc(dat):
                shape=dat.shape
                nLeft=(shape[0]//d)*d
                shape_new=(shape[0]//d,d)+shape[1:]
                return dat[:nLeft].reshape(shape_new).mean(axis=1)
            dat_run=tfunc(dat)
        else:
            dat_run=dat
        n=len(dat_run)
        return np.array([np.mean(np.delete(dat_run,i,axis=0),axis=0) for i in range(n)])
    def jackme(dat_jk):
        n=len(dat_jk)
        dat_mean=np.mean(dat_jk,axis=0)
        dat_err=np.sqrt(np.var(dat_jk,axis=0,ddof=0)*(n-1))
        return (dat_mean,dat_err)
    def jackme_un2str(dat_jk,toStringQ=True,**kargs):
        mean,err=jackme(dat_jk)
        if len(dat_jk.shape)==1:
            return un2str(mean,err,**kargs)
        elif len(dat_jk.shape)==2:
            t=[un2str(m,e,**kargs) for m,e in zip(mean,err)]
            if toStringQ:
                return f"[{', '.join(t)}]"
            else:
                return t
        1/0
    def jackmec(dat_jk):
        n=len(dat_jk)
        dat_mean=np.mean(dat_jk,axis=0)
        dat_cov=np.atleast_2d(np.cov(np.array(dat_jk).T,ddof=0)*(n-1))
        dat_err=np.sqrt(np.diag(dat_cov))
        return (dat_mean,dat_err,dat_cov)
    def jackmap(func,dat_jk,*args,**kargs):
        t=[func(dat,*args,**kargs) for dat in dat_jk]
        if type(t[0]) is tuple:
            return tuple([np.array([t[i][ind] for i in range(len(t))]) for ind in range(len(t[0]))])
        return np.array(t)
    def jackknife_pseudo(mean,cov,n):
        mean=np.array(mean); cov=np.array(cov)
        if len(mean.shape)==len(cov.shape)==0:
            mean=mean[None]; cov=cov[None,None]**2
        if len(mean.shape)==len(cov.shape)==1:
            cov=np.diag(cov**2)
        dat_ens=np.random.multivariate_normal(mean,cov*n,n)
        dat_jk=jackknife(dat_ens)
        # do transformation [pars_jk -> A pars_jk + B] to force pseudo mean and err exactly the same
        mean1,_,cov1=jackmec(dat_jk)
        A=np.sqrt(np.diag(cov)/np.diag(cov1))
        B=mean-A*mean1
        dat_jk=A[None,:]*dat_jk+B[None,:]
        return dat_jk
    
    def jackknife2(in_dat,in_func=lambda dat:np.mean(np.real(dat),axis=0),minNcfg:int=600,d:int=0):
        '''
        - in_dat: any-dimensional list of ndarrays. Each ndarray in the list has 0-axis for cfgs
        - in_func: dat -> estimator
        - Estimator: number or 1d-list/array or 2d-list/array or 1d-list of 1d-arrays
        - d: jackknife delete parameter
        ### return: mean,err,cov
        - mean,err: estimator's format reformatted to 1d-list of 1d-arrays
        - cov: 2d-list of 2d-arrays
        '''  
        getNcfg=lambda dat: len(dat) if type(dat)==np.ndarray else getNcfg(dat[0])
        n=getNcfg(in_dat)
        if flag_fast:
            d=n//300
        elif d==0:
            d=n//minNcfg
        d=max(d,1)
        
        # average ${d} cfgs
        if d!=1:
            def tfunc(dat):
                if type(dat)==list:
                    return [tfunc(ele) for ele in dat]
                shape=dat.shape
                nLeft=(shape[0]//d)*d
                shape_new=(shape[0]//d,d)+shape[1:]
                return dat[:nLeft].reshape(shape_new).mean(axis=1)
            dat=tfunc(in_dat)
        else:
            dat=in_dat
        
        # reformat output of in_func
        t=in_func(dat)
        if type(t) in [list,np.ndarray] and type(t[0]) in [list,np.ndarray]:
            lenList=[len(ele) for ele in t]
            func=lambda dat:np.concatenate(in_func(dat))
        elif type(t) in [list,np.ndarray] and type(t[0]) not in [list,np.ndarray]:
            lenList=[len(t)]
            func=lambda dat:np.array(in_func(dat))
        elif type(t) not in [list,np.ndarray]:
            lenList=[1]
            func=lambda dat:np.array([in_func(dat)])
        else:
            1/0
            
        # delete i 
        delete= lambda dat,i: np.delete(dat,i,axis=0) if type(dat)==np.ndarray else [delete(ele,i) for ele in dat]
        
        # jackknife     
        n=getNcfg(dat)
        Tn1=np.array([func(delete(dat,i)) for i in range(n)])
        # print(np.mean(func()))
        # print(np.sqrt(np.var(Tn1)*n))
        TnBar=np.mean(Tn1,axis=0)
        (tMean,tCov)=(TnBar, np.atleast_2d(np.cov(Tn1.T,ddof=0)*(n-1)))
        # Tn=func(dat); (mean,cov)=(n*Tn-(n-1)*TnBar, np.atleast_2d(np.cov(Tn1.T,ddof)*(n-1))) # bias improvement (not suitable for fit)
        tErr=np.sqrt(np.diag(tCov))
        
        # reformat
        mean=[];err=[];t=0
        for i in lenList:
            mean.append(tMean[t:t+i])
            err.append(tErr[t:t+i])
            t+=i
        cov=[];t=0
        for i in lenList:
            covI=[];tI=0
            for j in lenList:
                covI.append(tCov[t:t+i,tI:tI+j]);tI+=j
            cov.append(covI);t+=i
            
        ret=(mean,err,cov)
        return ret
    def jackknife2_ErrErr(dat,func_mean_err=lambda mean,err:err):
        def func(dat):
            Ncfg,Ntime=dat.shape
            mean=np.mean(dat,axis=0)
            err=np.sqrt(np.var(dat,axis=0,ddof=1)/Ncfg)    
            return func_mean_err(mean,err)
        return jackknife2(dat,func)
    def get_autocorrelation(dat,normalizeQ=True):
        Ncfg, Ntime = dat.shape
        dat = dat - dat.mean(axis=0, keepdims=True)
        C = np.empty((Ncfg, Ntime))
        for tau in range(Ncfg):
            C[tau] = np.mean(dat[:Ncfg - tau] * dat[tau:], axis=0)
        if normalizeQ:
            C /= C[0]
        return np.array(C)
    def jackknife2_autocorrelation(dat,normalizeQ=True):
        Ncfg=len(dat)
        mean,err,_=jackknife2(dat,lambda dat:get_autocorrelation(dat,normalizeQ)[:Ncfg-1])
        return np.array(mean),np.array(err)

    def superjackknife(dats_jk):
        Nens=len(dats_jk)
        Ncfgss=[len(dat) for dat in dats_jk]
        dats_jkmean=[np.mean(dat_jk,axis=0) for dat_jk in dats_jk]
        t=[[dats_jk[i] if i==j else np.repeat(dats_jkmean[j][None,:],Ncfgss[i],axis=0) for j in range(Nens)] for i in range(Nens)]
        return np.block(t)

    def jackMA(fits,propagateChi2=True):
        ''' 
        fits=[fit]; fit=(fit_label,pars_jk,chi2_jk,Ndof)
        '''
        if propagateChi2:
            temp=[(pars_jk
                ,np.exp(-chi2_jk/2+Ndof) # weights_jk
                ) for fit_label,pars_jk,chi2_jk,Ndof in fits]
        else:
            temp=[(pars_jk
                ,np.exp(-np.mean(chi2_jk,axis=0)[:,None]/2+Ndof) # weights_jk
                ) for fit_label,pars_jk,chi2_jk,Ndof in fits]
        # print([weights_jk[0,0] for pars_jk,weights_jk in temp])
        weightsSum_jk=np.sum([weights_jk for _,weights_jk in temp],axis=0)
        pars_jk=np.sum([pars_jk*weights_jk/weightsSum_jk for pars_jk,weights_jk in temp],axis=0)
        probs_jk=np.transpose([weights_jk[:,0]/weightsSum_jk[:,0] for _,weights_jk in temp])
        return pars_jk,probs_jk

    def modelAvg(fits):
        '''
        fits=[fit]; fit=(pars_mean,pars_err,chi2,Ndof)
        '''
        weights=np.exp([-chi2/2+Ndof for pars_mean,pars_err,chi2,Ndof in fits])
        probs=weights/np.sum(weights)
        pars_mean_MA=np.sum(np.array([pars_mean for pars_mean,pars_err,chi2,Ndof in fits])*probs[:,None],axis=0)
        pars_err_MA=np.sqrt(np.sum(np.array([pars_err**2+pars_mean**2 for pars_mean,pars_err,chi2,Ndof in fits])*probs[:,None],axis=0)-pars_mean_MA**2)
        return (pars_mean_MA,pars_err_MA,probs)
    def jackMA2(fits): # doing model average after jackknife
        temp=[]
        for fit_label,pars_jk,chi2_jk,Ndof in fits:
            pars_mean,pars_err=jackme(pars_jk)
            chi2_mean,chi2_err=jackme(chi2_jk)
            temp.append((pars_mean,pars_err,chi2_mean[0],Ndof))
        # print([np.exp(-chi2/2+Ndof) for pars_mean,pars_err,chi2,Ndof in temp])
        return modelAvg(temp)

    # uncertainty to string: taken from https://stackoverflow.com/questions/6671053/python-pretty-print-errorbars
    def un2str(x, xe, precision=2, forceResult = 1):
        if type(x) in [list,np.ndarray]:
            return [un2str(m,e,precision=precision,forceResult=forceResult) for m,e in zip(x,xe)]
        
        """pretty print nominal value and uncertainty

        x  - nominal value
        xe - uncertainty
        precision - number of significant digits in uncertainty

        returns shortest string representation of `x +- xe` either as
            x.xx(ee)e+xx
        or as
            xxx.xx(ee)"""
        # base 10 exponents
        x_exp = int(floor(log10(np.abs(x))))
        xe_exp = int(floor(log10(xe)))

        # uncertainty
        un_exp = xe_exp-precision+1
        un_int = round(xe*10**(-un_exp))

        # nominal value
        no_exp = un_exp
        no_int = round(x*10**(-no_exp))

        # format - nom(unc)exp
        fieldw = x_exp - no_exp
        
        if fieldw<0 and forceResult!=1:
            return un2str(x, xe, precision+1,forceResult=forceResult)
        if fieldw>=0:
            fmt = '%%.%df' % fieldw
            result1 = (fmt + '(%.0f)e%d') % (no_int*10**(-fieldw), un_int, x_exp)
        else:
            result1 = None

        # format - nom(unc)
        fieldw = max(0, -no_exp)
        fmt = '%%.%df' % fieldw
        result2 = (fmt + '(%.0f)') % (no_int*10**no_exp, un_int*10**max(0, un_exp))
        if un_exp<0 and un_int*10**un_exp>=1:
            fmt2= '(%%.%df)' % (-un_exp)
            result2 = (fmt + fmt2) % (no_int*10**no_exp, un_int*10**un_exp)
        
        if forceResult is not None:
            return [result1,result2][forceResult]

        # return shortest representation
        if len(result2) <= len(result1):
            return result2
        else:
            return result1
        
    def chi2Ndof2pval(chi2, Ndof):
        pval = 1 - chi2_dist.cdf(chi2, Ndof)
        return pval

#!============== auto-correlation ==============#
if False:
    import puwr
    def random_autocorrelation(Ncfg,Nm,tau=0.5):
        return np.transpose([puwr.correlated_data(tau, Ncfg)[0][0] for i in range(Nm)])

    def tauint(dat,func=lambda x:x): 
        #! It seems puwr.tauint has at least one bug: the output for mean is wrong when effective mass function is involved.
        '''
        dat.shape=Ncfg,Nm; m for measurement
        func: Nm->No; o for observable
        '''
        Ncfg,Nm=dat.shape
        No=len(func(dat[0]))
        dat_tauint=[[dat[:,i]] for i in range(Nm)]

        mdtds=np.array([puwr.tauint(dat_tauint,lambda *x:func(np.array(x))[i]) for i in range(No)])
        mean,delta,tint,d_tint = mdtds.T
        mean=func(np.mean(dat,axis=0))
        return mean,delta,tint,d_tint

#!============== fit (basic) ==============#
if True:
    def jackfit(fitfunc,y_jk,pars0,mask=None,parsExtra_jk=None,priors=[],getFilterInfoQ=False,**kargs):
        '''
        return pars_jk,chi2_jk,Ndof,Nwarning \\
        priors=[(ind of par, mean, width)]
        '''
        with warnings.catch_warnings(record=True) as list_warnings:
            warnings.simplefilter("always")

            y_mean,_,y_cov=jackmec(y_jk)
            Ndata=len(y_mean); Npar=len(pars0); Ndof=Ndata-Npar
            if mask is not None:
                if mask == 'uncorrelated':
                    y_cov=np.diag(np.diag(y_cov))
                else:
                    y_cov=y_cov*mask
                
            cho_L_Inv = np.linalg.inv(cholesky(y_cov, lower=True)) # y_cov^{-1}=cho_L_Inv^T@cho_L_Inv
            if parsExtra_jk is None:
                if len(priors)==0:
                    fitfunc_wrapper=lambda pars: cho_L_Inv@(fitfunc(pars)-y_mean)
                else:
                    fitfunc_wrapper=lambda pars: np.concatenate([cho_L_Inv@(fitfunc(pars)-y_mean),[(pars[ind]-mean)/width for ind,mean,width in priors]])
            else:
                parsExtra_mean=list(np.mean(parsExtra_jk,axis=0))
                if len(priors)==0:
                    fitfunc_wrapper=lambda pars: cho_L_Inv@(fitfunc(list(pars)+parsExtra_mean)-y_mean)
                else:
                    fitfunc_wrapper=lambda pars: np.concatenate([cho_L_Inv@(fitfunc(list(pars)+parsExtra_mean)-y_mean),[(pars[ind]-mean)/width for ind,mean,width in priors]])
            pars_mean,pars_cov=leastsq(fitfunc_wrapper,pars0,full_output=True,**kargs)[:2]
            if getFilterInfoQ:
                chi2=np.sum(fitfunc_wrapper(pars_mean)**2)
                return np.array([pars_mean]),np.array([chi2]),Ndof,'getFilterInfo=True'

            if flag_fast == "FastFit": # Generate pseudo jackknife resamples from the single fit rather than doing lots of fits
                n=len(y_jk)
                pars_jk=jackknife_pseudo(pars_mean,pars_cov,n)
            else:    
                def func(yp):
                    if parsExtra_jk is None:
                        if len(priors)==0:
                            fitfunc_wrapper2=lambda pars: cho_L_Inv@(fitfunc(pars)-yp)
                        else:
                            fitfunc_wrapper2=lambda pars: np.concatenate([cho_L_Inv@(fitfunc(pars)-yp),[(pars[ind]-mean)/width for ind,mean,width in priors]])
                    else:
                        y,p=yp
                        if len(priors)==0:
                            fitfunc_wrapper2=lambda pars: cho_L_Inv@(fitfunc(list(pars)+list(p))-y)
                        else:
                            fitfunc_wrapper2=lambda pars: np.concatenate([cho_L_Inv@(fitfunc(list(pars)+list(p))-y),[(pars[ind]-mean)/width for ind,mean,width in priors]])
                    pars=leastsq(fitfunc_wrapper2,pars_mean,**kargs)[0]
                    return pars
                if parsExtra_jk is not None:
                    y_jk=zip(y_jk,parsExtra_jk)
                    
                pars_jk=jackmap(func,y_jk)
                
            chi2_jk=np.array([[np.sum(fitfunc_wrapper(pars)**2)] for pars in pars_jk])
            
            Nwarning = len(list_warnings)
            for w in list_warnings:
                warnings.showwarning(message=w.message,category=w.category,filename=w.filename,lineno=w.lineno,file=w.file,line=w.line)
        return pars_jk,chi2_jk,Ndof,Nwarning
    
    def find_fitmax(dat,threshold=0.2):
        mean,err=jackme(dat)
        rela=np.abs(err/mean)
        temp=[(i,rela) for i,rela in enumerate(rela) if rela>0.2 and i!=0]
        fitmax=temp[0][0]-1 if len(temp)!=0 else len(mean)-1
        return fitmax
    
    def doFit_const(y_jk,corrQ=True,**kargs):
        '''
        return pars_jk,chi2_jk,Ndof
        '''
        Ndata=y_jk.shape[1]
        if Ndata==1:
            return y_jk,np.zeros((len(y_jk),1)),0
        def fitfunc(pars):
            return list(pars)*Ndata
        pars_jk,chi2_jk,Ndof,Nwarning=jackfit(fitfunc,y_jk,[np.mean(y_jk)],mask=None if corrQ else 'uncorrelated',**kargs)
        return pars_jk,chi2_jk,Ndof
    
    def doFit_linear(xs,y_jk,corrQ=True,**kargs):
        '''
        return pars_jk,chi2_jk,Ndof
        '''
        def fitfunc(pars):
            c0,c1=pars
            return c1*xs+c0
        pars_jk,chi2_jk,Ndof,Nwarning=jackfit(fitfunc,y_jk,[np.mean(y_jk),0],mask=None if corrQ else 'uncorrelated',**kargs)
        return pars_jk,chi2_jk,Ndof
    
    def fits2text(fits,toStringQ=False):
        text=[]
        pars_jk,probs_jk=jackMA(fits)
        Npar=pars_jk.shape[1]
        Ncut=Npar if Npar<20 else 1
        means,errs=jackme(pars_jk[:,:Ncut])
        probs_mean,probs_err=jackme(probs_jk)
        ind_mpf=np.argmax(probs_mean)
        fitlabel,pars_jk,chi2_jk,Ndof = fits[ind_mpf]
        chi2=np.mean(chi2_jk)
        text.append(f'Model average: pars={[un2str(mean,err) for mean,err in zip(means,errs)]}, most probable fitlabel: {fits[ind_mpf][0]}, prob={int(probs_mean[ind_mpf]*100)}%, chi2/Ndof={int(chi2*10)/10}/{Ndof}={int(chi2/Ndof*10)/10};')
        for i,fit in enumerate(fits):
            fitlabel,pars_jk,chi2_jk,Ndof = fit
            means,errs=jackme(pars_jk[:,:Ncut]); mean_chi2,err_chi2=jackme(chi2_jk)
            if Ndof!=0:
                text.append(f'fitlabel={fitlabel}, pars={[un2str(mean,err) for mean,err in zip(means,errs)]}, prob={int(probs_mean[i]*100)}%, chi2/Ndof={int(mean_chi2[0]*10)/10}/{Ndof}={int(mean_chi2[0]/Ndof*10)/10};')
            else:
                text.append(f'fitlabel={fitlabel}, pars={[un2str(mean,err) for mean,err in zip(means,errs)]}, prob={int(probs_mean[i]*100)}%, Ndof=0;')
        if toStringQ:
            text='\n'.join(text)
        return text
    def printFits(fits):
        print(fits2text(fits,toStringQ=True))
        
    def findFit(fits,fitlabel):
        for fit in fits:
            if fit[0]==fitlabel:
                return fit
    
    def decorator_fits(func):
        @functools.wraps(func)
        def wrapper(*args, label=None, overwrite=False, **kwargs):
            if label is not None and not overwrite:
                res=load_pkl_internal(label)
                if res is not None:
                    return res
            res = func(*args, **kwargs)
            if label is not None:
                if save_pkl_internal(label,res):
                    text=fits2text(res,toStringQ=False)
                    save_txt_internal(label,text)
            return res
        return wrapper

    def getFits(label):
        return load_pkl_internal(label)
    
    @decorator_fits
    def doFits_const(y_jk,xmins,xmaxs,corrQ=True,**kargs):
        fits=[]
        for xmin in xmins:
            for xmax in xmaxs:
                pars_jk,chi2_jk,Ndof=doFit_const(y_jk[:,xmin:xmax+1],corrQ=corrQ,**kargs)
                fits.append([(xmin,xmax),pars_jk,chi2_jk,Ndof])
        return fits
    
    @decorator_fits
    def doFits_linear(xs,y_jk,xmins,xmaxs,corrQ=True,**kargs):
        fits=[]
        for xmin in xmins:
            for xmax in xmaxs:
                inds=[i for i,x in enumerate(xs) if xmin<=x<=xmax]
                pars_jk,chi2_jk,Ndof=doFit_linear(xs[inds],y_jk[:,inds],corrQ=corrQ,**kargs)
                fits.append([(xmin,xmax),pars_jk,chi2_jk,Ndof])
        return fits

    @decorator_fits
    def doFits_continuumExtrapolation(ens2dat,lat_a2s_plt=None):
        enss=list(ens2dat.keys()); enss.sort()
        lat_a2s=[ens2a[ens]**2 for ens in enss]
        dat=[ens2dat[ens][:,None] for ens in enss]
        
        t=superjackknife(dat)
        fits=[]
        fitlabel='const'
        pars_jk,chi2_jk,Ndof=doFit_const(t)
        if lat_a2s_plt is not None:
            pars_jk=np.array([0*np.array(lat_a2s_plt)+pars[0] for pars in pars_jk])
        fits.append([fitlabel,pars_jk,chi2_jk,Ndof])
        fitlabel='linear'
        pars_jk,chi2_jk,Ndof=doFit_linear(np.array(lat_a2s),t)
        if lat_a2s_plt is not None:
            pars_jk=np.array([pars[1]*np.array(lat_a2s_plt)+pars[0] for pars in pars_jk])
        fits.append([fitlabel,pars_jk,chi2_jk,Ndof])
        return fits
#!============== fit (2pt) ==============#
if True:    
    def doFits_2pt_pion(dat,tmins,tmax,func=lambda ts,m:[m]*len(ts),pars0=[1],corrQ=True,debugQ=False):
        fits=[]
        for tmin in tmins:
            ts=np.arange(tmin,tmax)
            def fitfunc(pars):
                return func(ts,*pars)
            y_jk=dat[:,ts]
            pars_jk,chi2_jk,Ndof,Nwarning=jackfit(fitfunc,y_jk,pars0,mask=None if corrQ else 'uncorrelated')
            if Nwarning or debugQ:
                print(f'[Nwarning={Nwarning}] Npar={len(pars0)}, tmin={tmin}')
            pars0=np.mean(pars_jk,axis=0)
            fits.append([tmin,pars_jk,chi2_jk,Ndof])
        return fits
    
    @decorator_fits
    def doFits_2pt(dat,tmins,func,pars0,downSampling=1,corrQ=True,debugQ=False):
        tmax=find_fitmax(dat)
        fits=[]
        for tmin in tmins:
            ts=np.arange(tmin,tmax,downSampling)
            if len(ts)<2:
                continue
            def fitfunc(pars):
                return func(ts,*pars)
            y_jk=dat[:,ts]
            pars_jk,chi2_jk,Ndof,Nwarning=jackfit(fitfunc,y_jk,pars0,mask=None if corrQ else 'uncorrelated')
            if Nwarning or debugQ:
                print(f'[Nwarning={Nwarning}] Npar={len(pars0)}, tmin={tmin}')
            pars0=np.mean(pars_jk,axis=0)
            fits.append([tmin,pars_jk,chi2_jk,Ndof])
        return fits
    
    def decorator_fits_2pt(func):
        @functools.wraps(func)
        def wrapper(*args, label=None, overwrite=False, **kwargs):
            if label is not None and not overwrite:
                res=load_pkl_internal(label)
                if res is not None:
                    return res
            res = func(*args, **kwargs)
            if label is not None:
                if save_pkl_internal(label,res):
                    text=[]
                    for i,fits in enumerate(res):
                        text.append(f'{i+1} state fit')
                        text+=fits2text(fits,toStringQ=False)
                        text.append('\n')
                    save_txt_internal(label,text)
            return res
        return wrapper

    func_c2pt_1st=lambda t,E0,c0: c0*np.exp(-E0*t)
    func_c2pt_2st=lambda t,E0,c0,dE1,rc1: c0*np.exp(-E0*t)*(1 + rc1*np.exp(-dE1*t))
    func_c2pt_3st=lambda t,E0,c0,dE1,rc1,dE2,rc2: c0*np.exp(-E0*t)*(1 + rc1*np.exp(-dE1*t) + rc2*np.exp(-dE2*t))
    func_meff_1st=lambda t,E0: np.log(func_c2pt_1st(t,E0,1)/func_c2pt_1st(t+1,E0,1))
    func_meff_2st=lambda t,E0,dE1,rc1: np.log(func_c2pt_2st(t,E0,1,dE1,rc1)/func_c2pt_2st(t+1,E0,1,dE1,rc1))
    func_meff_3st=lambda t,E0,dE1,rc1,dE2,rc2: np.log(func_c2pt_3st(t,E0,1,dE1,rc1,dE2,rc2)/func_c2pt_3st(t+1,E0,1,dE1,rc1,dE2,rc2))
    @decorator_fits_2pt
    def doFits_meff_nst(meff,tminss,pars0,downSampling=1,corrQ=True,debugQ=False):
        Nst=len(tminss)
        fits_1st=doFits_2pt(meff,tminss[0],func_meff_1st,pars0[:1],downSampling=downSampling,corrQ=corrQ,debugQ=debugQ)
        if Nst==1:
            return [fits_1st]
        
        pars_jk,probs_jk=jackMA(fits_1st)
        pars0[:1]=np.mean(pars_jk,axis=0)
        fits_2st=doFits_2pt(meff,tminss[1],func_meff_2st,pars0[:3],downSampling=downSampling,corrQ=corrQ,debugQ=debugQ)
        if Nst==2:
            return [fits_1st,fits_2st]
        
        pars_jk,probs_jk=jackMA(fits_2st)
        pars0[:3]=np.mean(pars_jk,axis=0)
        fits_3st=doFits_2pt(meff,tminss[2],func_meff_3st,pars0[:5],downSampling=downSampling,corrQ=corrQ,debugQ=debugQ)
        if Nst==3:
            return [fits_1st,fits_2st,fits_3st]
#!============== fit (3pt) ==============#
if True:
    def cutExtraDiff2tcmins(cutExtras,cutDiffs,cutBase=1):
        return [(cutBase+(cutExtra+cutDiff)//2,cutBase+(cutExtra-cutDiff)//2) for cutExtra in cutExtras for cutDiff in cutDiffs if (cutExtra+cutDiff)%2==0 and cutExtra>=abs(cutDiff)]

    @decorator_fits
    def doFits_3pt_band(tf2ratio_para,tcmins,tf2tcmins=None,tfmax=None,symmetrizeQ=False,downSampling=1,unicutQ=False,corrQ=True,verbose=0):
        symQ = isinstance(tcmins[0], int)
        tf2ratio=tf2ratio_para.copy()
        if symmetrizeQ:
            assert(symQ)
            symmetrizeRatio(tf2ratio)
        
        if tf2tcmins is not None:
            tfs=sorted(tf2tcmins)
            tcmins=tf2tcmins[tfs[0]] 
        else:
            tfs=sorted(tf2ratio)
        if tfmax is not None:
            tfs=[tf for tf in tfs if tf<=tfmax]    
        
        fits=[]
        for tf in tfs: 
            ratio=tf2ratio[tf]
            if tf2tcmins is not None:
                tcmins=tf2tcmins[tf]
            for tcmin in tcmins:
                if verbose>0:
                    print(f'[verbose1] tf={tf}, tcmin={tcmin};')
                tcs_fit=np.arange(tcmin,tf-tcmin+1,downSampling) if symQ else np.arange(tcmin[0],tf-tcmin[1]+1,downSampling)
                if symmetrizeQ:
                    tcs_fit=np.arange(tcmin,tf//2+1,downSampling)
                if len(tcs_fit)==0:
                    continue
                y_jk=ratio[:,tcs_fit]
                pars_jk,chi2_jk,Ndof=doFit_const(y_jk,corrQ=corrQ,getFilterInfoQ=(unicutQ is not False))
                fits.append([(tf,tcmin),pars_jk,chi2_jk,Ndof])
                
        if unicutQ is not False:
            assert(not symQ)
            fits=filterFits_3ptasy_unicut(fits,Nmin = 3 if unicutQ==True else unicutQ)
            tf2tcmins=defaultdict(list)
            for ((tf,tcmin),pars_jk,chi2_jk,Ndof) in fits:
                tf2tcmins[tf].append(tcmin)
            if verbose>0:
                print('[verbose1] ============================ FULL RUNS FROM HERE ============================')
            return doFits_3pt_band(tf2ratio,None,tf2tcmins=tf2tcmins,downSampling=downSampling,unicutQ=False,corrQ=corrQ,verbose=verbose)
        
        return fits
    def doWA_band(fits,tf_min=None,tf_max=None,tcmin=None,corrQ=True):
        '''
        return: pars_jk,chi2_jk,Ndof,fits
        '''
        symQ = isinstance(fits[0][0][1], int)
        tfs=[fit[0][0] for fit in fits]
        tf_min = min(tfs) if tf_min is None else tf_min
        tf_max = max(tfs) if tf_max is None else tf_max
        if symQ or (isinstance(tcmin,tuple)):
            tcmin = min([tcmin if symQ else tcmin[0]+tcmin[1] for (tf,tcmin),pars_jk,chi2_jk,Ndof in fits]) if tcmin is None else tcmin
            fits=[fit for fit in fits if tf_min<=fit[0][0]<=tf_max and fit[0][1]==tcmin]
        else:
            tcmin= min([tcmin[0]+tcmin[1] for (tf,tcmin),pars_jk,chi2_jk,Ndof in fits]) if tcmin is None else tcmin
            fits=[fit for fit in fits if tf_min<=fit[0][0]<=tf_max and fit[0][1][0]+fit[0][1][1]==tcmin]
        if len(fits)==0:
            print('[doWA_band] no fits left.')
        y_jk=np.transpose([fit[1][:,0] for fit in fits])
        pars_jk,chi2_jk,Ndof=doFit_const(y_jk,corrQ)
        return pars_jk,chi2_jk,Ndof,fits
    
    def filterFits_3ptasy_unicut(fits, Nmin=1):
        tfmins, cuts, chi2s, Ndofs = map(np.array, zip(*[(tfmin, tcmina+tcminb, np.mean(chi2_jk),Ndof) for ((tfmin,(tcmina,tcminb)),pars_jk,chi2_jk,Ndof) in fits]))
        
        tfmins_unique=removeDuplicates(tfmins); tfmins_unique.sort()
        cuts_unique=removeDuplicates(cuts); cuts_unique.sort()
        fits_new=[]
        for tfmin_now in tfmins_unique:
            for cut_now in cuts_unique:
                inds = np.where((tfmins==tfmin_now) & (cuts==cut_now) & (Ndofs>=Nmin))[0]
                if len(inds)==0:
                    continue
                ind = inds[np.argmin(chi2s[inds])]
                fits_new.append(fits[ind])
        if len(fits_new)==0:
            print('[filterFits_3ptasy_unicut] no fits left.')
        return fits_new
        
    def doMA_3pt(fits,tfmin_min=None,tfmin_max=None,tcmin_min=None,tcmin_max=None,fitlabels=None,probThreshold=None,chi2RThreshold=None):
        '''
        return: pars_jk,probs_jk,fits
        '''
        if fitlabels is not None:
            if type(fitlabels)!=list:
                fitlabels=[fitlabels]
            fits=[fit for fit in fits if fit[0] in fitlabels]
        symQ = isinstance(fits[0][0][1], int)
        convert_tcmin = (lambda tcmin:tcmin) if symQ else (lambda tcmin:tcmin[0]+tcmin[1])
        convert_tcmins = (lambda tcmins:tcmins) if symQ else (lambda tcmins:[tcmina+tcminb for tcmina,tcminb in tcmins])
        
        tfmins=[fit[0][0] for fit in fits]
        tcmins=convert_tcmins([fit[0][1] for fit in fits])
        tfmin_min=min(tfmins) if tfmin_min is None else tfmin_min
        tfmin_max=max(tfmins) if tfmin_max is None else tfmin_max
        tcmin_min=min(tcmins) if tcmin_min is None else tcmin_min
        tcmin_max=max(tcmins) if tcmin_max is None else tcmin_max
        
        fits=[fit for fit in fits if tfmin_min<=fit[0][0]<=tfmin_max and tcmin_min<=convert_tcmin(fit[0][1])<=tcmin_max]
        if len(fits)==0:
            print('[doMA_3pt] no fits left.')
        pars_jk,probs_jk=jackMA(fits)
        if chi2RThreshold is not None:
            fits=[fit for fit in fits if np.mean(fit[1]/fit[2])<=chi2RThreshold]
            pars_jk,probs_jk=jackMA(fits)
        if probThreshold is not None:
            probs=np.mean(probs_jk,axis=0)
            fits=[fits[i] for i in range(len(probs)) if probs[i]>=probThreshold]
            pars_jk,probs_jk=jackMA(fits)
        return pars_jk,probs_jk,fits


    func_c3pt_2st=lambda tf,tc,E0a,E0b,a00,dE1a,dE1b,a01,a10,a11: np.exp(-E0a*(tf-tc))*np.exp(-E0b*tc)*(a00 + a01*np.exp(-dE1b*tc) + a10*np.exp(-dE1a*(tf-tc)) + a11*np.exp(-dE1a*(tf-tc))*np.exp(-dE1b*tc))
    func_c3pt_2st_0a00=lambda tf,tc,E0a,E0b,dE1a,dE1b,a01,a10,a11:np.exp(-E0a*(tf-tc))*np.exp(-E0b*tc)*(a01*np.exp(-dE1b*tc) + a10*np.exp(-dE1a*(tf-tc)) + a11*np.exp(-dE1a*(tf-tc))*np.exp(-dE1b*tc))
    
    func_ratioSYM_2st=lambda tf,tc,g,dE1,ra01,ra11, dE1_2pt,rc1_2pt:func_c3pt_2st(tf,tc,0,0,g,dE1,dE1,ra01,ra01,ra11)/func_c2pt_2st(tf,0,1,dE1_2pt,rc1_2pt)
    func_ratioSYMshare_2st=lambda tf,tc,g,ra01,ra11, dE1_2pt,rc1_2pt:func_ratioSYM_2st(tf,tc,g,dE1_2pt,ra01,ra11, dE1_2pt,rc1_2pt)
    
    func_ratioSQRT_2st=lambda tf,tc,g,dE1a,dE1b,ra01,ra10,ra11, dE1_2pta,rc1_2pta, dE1_2ptb,rc1_2ptb: \
        func_c3pt_2st(tf,tc,0,0,g,dE1a,dE1b,ra01,ra10,ra11)/np.sqrt( \
            func_c2pt_2st(tf,0,1,dE1_2pta,rc1_2pta)*func_c2pt_2st(tf,0,1,dE1_2ptb,rc1_2ptb) *\
                func_c2pt_2st(tf-tc,0,1,dE1_2pta,rc1_2pta)/func_c2pt_2st(tc,0,1,dE1_2pta,rc1_2pta) *\
                func_c2pt_2st(tc,0,1,dE1_2ptb,rc1_2ptb)/func_c2pt_2st(tf-tc,0,1,dE1_2ptb,rc1_2ptb) )
    func_ratioSQRTshare_2st=lambda tf,tc,g,ra01,ra10,ra11, dE1_2pta,rc1_2pta, dE1_2ptb,rc1_2ptb:func_ratioSQRT_2st(tf,tc,g,dE1_2pta,dE1_2ptb,ra01,ra10,ra11, dE1_2pta,rc1_2pta, dE1_2ptb,rc1_2ptb)
    
    func_ratioEFIT_2st=lambda tf,tc,g,dE1a,dE1b,ra01,ra10,ra11, dE1_2pta,rc1_2pta, dE1_2ptb,rc1_2ptb: \
        func_c3pt_2st(tf,tc,0,0,g,dE1a,dE1b,ra01,ra10,ra11)/np.sqrt( \
            func_c2pt_2st(tf,0,1,dE1_2pta,rc1_2pta)*func_c2pt_2st(tf,0,1,dE1_2ptb,rc1_2ptb) )
    func_ratioEFITshare_2st=lambda tf,tc,g,ra01,ra10,ra11, dE1_2pta,rc1_2pta, dE1_2ptb,rc1_2ptb:func_ratioEFIT_2st(tf,tc,g,dE1_2pta,dE1_2ptb,ra01,ra10,ra11, dE1_2pta,rc1_2pta, dE1_2ptb,rc1_2ptb)
    
    fittype2func={
        '2st2step_SYMshare':func_ratioSYMshare_2st,
        '2st2step_SQRTshare':func_ratioSQRTshare_2st,
        '2st2step_EFITshare':func_ratioEFITshare_2st
    }
    
    @decorator_fits
    def doFits_3pt(fittype,tf2ratio_para,tfmins,tcmins,tfmin2tcmins=None,pars_jk_meff2st=None,pars0=None,downSampling=[1,1],symmetrizeQ=False,unicutQ=False,corrQ=True,fastQ=False,verbose=0):
        '''
        fittype in ['const','sum','2st2step_SYMshare','2st2step_SQRTshare','2st2step_EFITshare'] \\
        fit = [(tfmin,tcmin),pars_jk,chi2_jk,Ndof]
        '''
        assert(fittype in ['const','sum','2st2step_SYMshare','2st2step_SQRTshare','2st2step_EFITshare'])
        
        fastFlag = (unicutQ is not False) or (fastQ)
        
        if tfmin2tcmins is not None:
            tfmins=list(tfmin2tcmins.keys()); tfmins.sort()
            tcmins=tfmin2tcmins[tfmins[0]] 

        symQ = isinstance(tcmins[0], int)

        tf2ratio=tf2ratio_para.copy()
        tfs=list(tf2ratio.keys()); tfs.sort()
        if symmetrizeQ:
            assert(symQ)
            symmetrizeRatio(tf2ratio)
        pars0Initial=pars0
        if pars0 is None:
            tfmin=min(tfs); tf=max([tf for tf in tfs if tf<tfmin+7]) 
            tc=min(tcmins) if isinstance(tcmins[0],int) else min([tcmina for tcmina,tcminb in tcmins])
            
            ratio=np.mean(tf2ratio[tf],axis=0)
            g=ratio[tf//2]
            ra01= 1 if g<ratio[tc] else -1
            ra10= 1 if g<ratio[tf-tc] else -1
            ra11=0.1
            pars0={
                'const':[g], 'sum':[g,0], '2st2step_SYMshare':[g,ra01,ra11],
                '2st2step_SQRTshare':[g,ra01,ra10,ra11], '2st2step_EFITshare':[g,ra01,ra10,ra11]
            }[fittype]
        if verbose>=3:
            print(f'[verbose3] pars0={formatList(pars0,".2f")}')
                
        fits=[]
        for tfmin in tfmins:
            if verbose==1:
                print(f'[verbose1] tfmin={tfmin};')
            if tfmin2tcmins is not None:
                tcmins=tfmin2tcmins[tfmin]
            for tcmin in tcmins:
                if ((tfmin<tcmin*2) if symQ else (tfmin<tcmin[0]+tcmin[1])):
                    continue
                if verbose>=2:
                    print(f'[verbose2] tfmin={tfmin}, tcmin={tcmin};')
                if fittype in ['sum']:                    
                    downSampling=1 if not isinstance(downSampling,int) else downSampling
                    tfs_fit=np.array([tf for tf in tfs if tfmin<=tf and tf%downSampling==tfmin%downSampling])
                    if len(tfs_fit)==0:
                        continue
                    y_jk=np.transpose([np.sum(tf2ratio[tf][:,tcmin:tf+1-tcmin],axis=1) for tf in tfs_fit]) if symQ else \
                        np.transpose([np.sum(tf2ratio[tf][:,tcmin[0]:tf+1-tcmin[1]],axis=1) for tf in tfs_fit])

                    m,e,c=jackmec(y_jk)
                else:
                    tfs_fit=[tf for tf in tfs if tcmin*2<=tf and tfmin<=tf and tf%downSampling[0]==tfmin%downSampling[0]] if symQ else \
                        [tf for tf in tfs if tcmin[0]+tcmin[1]<=tf and tfmin<=tf and tf%downSampling[0]==tfmin%downSampling[0]]
                    if len(tfs_fit)==0:
                        continue
                    tf2tcs_fit={tf:np.arange(tcmin,tf//2+1,downSampling[1]) if symmetrizeQ else np.arange(tcmin,tf-tcmin+1,downSampling[1])  for tf in tfs_fit} if symQ else \
                        {tf:np.arange(tcmin[0],tf-tcmin[1]+1,downSampling[1])  for tf in tfs_fit} 
                    y_jk=np.concatenate([tf2ratio[tf][:,tf2tcs_fit[tf]] for tf in tfs_fit],axis=1)

                if fittype in ['const']:
                    Ndata=y_jk.shape[1]
                    def fitfunc(pars):
                        return list(pars)*Ndata
                elif fittype in ['sum']:
                    def fitfunc(pars):
                        g,c=pars
                        return g*tfs_fit+c
                elif fittype in ['2st2step_SYMshare','2st2step_SQRTshare','2st2step_EFITshare']:
                    func=fittype2func[fittype]
                    def fitfunc(pars):
                        t=np.concatenate([func(tf,tf2tcs_fit[tf],*pars) for tf in tfs_fit])
                        return t

                if type(pars_jk_meff2st)==list:
                    pars_jk_meff2st=np.concatenate(pars_jk_meff2st,axis=1)
                if fittype in ['2st2step_SYMshare'] and pars_jk_meff2st.shape[1]==3:
                    pars_jk_meff2st=pars_jk_meff2st[:,[1,2]]
                if fittype in ['2st2step_SQRTshare','2st2step_EFITshare'] and pars_jk_meff2st.shape[1]==6:
                    pars_jk_meff2st=pars_jk_meff2st[:,[1,2,4,5]]
                
                pars_jk,chi2_jk,Ndof,Nwarning=jackfit(fitfunc,y_jk,pars0,parsExtra_jk=pars_jk_meff2st,mask=None if corrQ else 'uncorrelated',getFilterInfoQ=fastFlag)
                if isinstance(Nwarning,int) and Nwarning>0:
                    print(f'[Nwarning={Nwarning}] tfmin={tfmin}, tcmin={tcmin};')
                pars0=np.mean(pars_jk,axis=0)
                if verbose>=3:
                    print(f'[verbose3] pars={formatList(pars0,".2f")}')
                fits.append([(tfmin,tcmin),pars_jk,chi2_jk,Ndof])
                
        if unicutQ is not False:
            assert(not symQ)
            if len(fits)==0:
                return []
            fits=filterFits_3ptasy_unicut(fits,Nmin = 3 if unicutQ==True else unicutQ)
            tfmin2tcmins=defaultdict(list)
            for ((tfmin,tcmin),pars_jk,chi2_jk,Ndof) in fits:
                tfmin2tcmins[tfmin].append(tcmin)
            if verbose>0:
                print('[verbose] ============================ FULL RUNS FROM HERE ============================')
            return doFits_3pt(fittype,tf2ratio_para,tfmins,None,tfmin2tcmins=tfmin2tcmins,pars_jk_meff2st=pars_jk_meff2st,pars0=pars0Initial,downSampling=downSampling,symmetrizeQ=symmetrizeQ,unicutQ=False,corrQ=corrQ,fastQ=fastQ,verbose=verbose)
        
        return fits

#!============== table ==============#
if True:
    def dfs2html(dfs,titles=None):
        if type(dfs)==pd.core.frame.DataFrame:
            dfs=[dfs]; titles=[titles]
        if titles is None:
            titles=[None]*len(dfs)
        
        blocks = []
        for df,title in zip(dfs,titles):
            blocks.append(f"""
                <div>
                    <h3 style="text-align:center;">{title}</h3>
                    {df.to_html()}
                </div>
            """)

        html = f"""
        <div style="display:flex; gap:20px;">
            {''.join(blocks)}
        </div>
        """

        return HTML(html)

#!============== plot (basic) ==============#
if True:
    colors8=['r','g','b','orange','purple','brown','magenta','olive']
    fmts8=['s','o','d','^','v','<','>','*']
    
    colors16=['blue','orange','green','red','purple','brown','darkblue','olive','darkgreen','darkred','grey','tan','peru','magenta','gold','skyblue']
    fmts16=['o','v','^','<','>','d','s','h','*','H','p','8','X','P','D','.']
    
    def jitter_duplicate_x(x, fraction=0.2):
        x = np.asarray(x, float)
        out = x.copy()
        ux = np.unique(x)

        for s in ux:
            idx = np.where(x == s)[0]
            k = idx.size
            if k < 2:
                continue

            d = np.min(np.abs(ux[ux != s] - s)) if ux.size > 1 else 1.0
            out[idx] = s + (np.arange(k) - (k - 1) / 2) * d * fraction

        return out
    
    def getFigAxs(Nrow,Ncol,Lrow=None,Lcol=None,scale=1,**kwargs):
        if (Lrow,Lcol)==(None,None):
            Lcol,Lrow=mpl.rcParams['figure.figsize']
            Lrow*=scale; Lcol*=scale
            # if (Nrow,Ncol)==(1,1):
            #     Lcol*=1.5; Lrow*=1.5
        fig, axs = plt.subplots(Nrow, Ncol, figsize=(Lcol*Ncol, Lrow*Nrow), squeeze=False,**kwargs)
        fig.align_ylabels()
        return fig, axs

    def addRowHeader(axs,rows,fontsize='xx-large',**kargs):
        pad=5
        for ax, row in zip(axs[:,0], rows):
            ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                    xycoords=ax.yaxis.label, textcoords='offset points', ha='right', va='center', fontsize=fontsize, **kargs)
            
    def addColHeader(axs,cols,fontsize='xx-large',**kargs):
        pad=5
        for ax, col in zip(axs[0,:], cols):
            ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                    xycoords='axes fraction', textcoords='offset points', ha='center', va='baseline', fontsize=fontsize, **kargs)
            
    def addRefLine(ax,val,hv='h',color='grey',ls='--',marker='',label=None):
        axline={'h':ax.axhline,'v':ax.axvline}[hv]
        axline(val,color=color,ls=ls,marker=marker,label=label)

    def finalizePlot(file=None,closeQ=None,mkdirQ=False):
        if closeQ is None:
            closeQ=False if file is None else True
        plt.tight_layout()
        if file!=None:
            if path_fig_internal is None:
                print('path_fig_internal is None')
                return
            file=f'{path_fig_internal}{any2filename(file)}.pdf'
            if mkdirQ:
                os.makedirs(os.path.dirname(file), exist_ok=True)
            plt.savefig(file,bbox_inches="tight")
        if closeQ:
            plt.close()
    
    def makePDF(file,figs,mkdirQ=False):
        assert(path_fig_internal is not None)
        file=f'{path_fig_internal}{any2filename(file)}.pdf'
        if mkdirQ:
            os.makedirs(os.path.dirname(file), exist_ok=True)
        pdf = PdfPages(file)
        for fig in figs:
            pdf.savefig(fig,bbox_inches="tight")
        pdf.close()
            
    def makePlot_simpleComparison(xs,y_jk,xticklabels=None):
        fig, axs = getFigAxs(1,1)
        ax=axs[0,0]
        mean,err=jackme(y_jk)
        ax.errorbar(xs,mean,err,color='r',fmt='s')
        ax.set_xticks(xs)
        if xticklabels is not None:
            ax.set_xticklabels(xticklabels)
        return fig,axs
    
    def makePlot_continuumExtrapolation(matrix_dic,shows=['MA']):
        if type(matrix_dic)==dict:
            matrix_dic=[[matrix_dic]]
        elif type(matrix_dic[0])==dict:
            matrix_dic=[matrix_dic]
        
        Nrow,Ncol=len(matrix_dic),len(matrix_dic[0])
        fig,axs=getFigAxs(Nrow,Ncol,sharex='col',sharey='row')
        for icol in range(Ncol):
            ax=axs[-1,icol]
            ax.set_xlabel(r'$a^2$ [fm$^2$]')
            
        for irow in range(Nrow):
            for icol in range(Ncol):
                ax=axs[irow,icol]
                dic=matrix_dic[irow][icol]
                ens2dat=dic['ens2dat']
                fits,lat_a2s_plt=dic['fit:[fits,lat_a2s_plt]']
                lat_a2s_plt=np.array(lat_a2s_plt)
                enss=list(ens2dat.keys())
                
                lat_a2s=np.array([ens2a[ens]**2 for ens in enss])

                mes=[jackme(ens2dat[ens]) for ens in enss]
                plt_x=lat_a2s; plt_y=[me[0] for me in mes]; plt_yerr=[me[1] for me in mes]
                ax.errorbar(plt_x,plt_y,plt_yerr,color='g')
                
                colors={'const':'r','linear':'b','MA':'orange'}
                
                for fit in fits:
                    fitlabel,pars_jk,chi2_jk,Ndof=fit
                    if fitlabel not in shows:
                        continue
                    mean,err=jackme(pars_jk)
                    x=lat_a2s_plt; ymin=mean-err; ymax=mean+err
                    ax.plot(x,mean,color=colors[fitlabel],linestyle='--',marker='')
                    ax.fill_between(x, ymin, ymax, color=colors[fitlabel], alpha=0.1,label=f'{fitlabel}={un2str(mean[0],err[0])}')
                for fitlabel in ['MA']:
                    if fitlabel not in shows:
                        continue
                    pars_jk,probs_jk=jackMA(fits)
                    mean,err=jackme(pars_jk)
                    x=lat_a2s_plt; ymin=mean-err; ymax=mean+err
                    ax.plot(x,mean,color=colors[fitlabel],linestyle='--',marker='')
                    ax.fill_between(x, ymin, ymax, color=colors[fitlabel], alpha=0.1,label=f'{fitlabel}={un2str(mean[0],err[0])}')
                
                ax.legend(fontsize=16)
        return fig,axs
#!============== plot (2pt) ==============#
if True:
    def makePlot_2pt_pion(meff,fits,xunit=1,yunit=1,xlim='half',ylim='auto',selection=None):
        Ncfg,Nt=meff.shape
        if xlim=='half':
            xlim=[0,(Nt//2)*xunit]
        if ylim=='auto':
            mean,err=jackme(meff[:,Nt//3]*yunit)
            ylim=[mean-err*10,mean+err*13]
        
        result=None
        fig, axs = getFigAxs(1,2,Lrow=4,Lcol=8,sharey=True,gridspec_kw={'width_ratios': [3,2]})
        ax=axs[0,0]
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel(r'$t$ [fm]')
        ax.set_ylabel(r'$m_{\rm eff}$ [MeV]')
        
        ts=np.arange(len(meff[0]))
        mean,err=jackme(meff)
        plt_x=ts*xunit; plt_y=mean[ts]*yunit; plt_yerr=err[ts]*yunit
        ax.errorbar(plt_x,plt_y,plt_yerr,color='r',fmt='s')
        
        ax=axs[0,1]
        ax.set_xlabel(r'$t_{\mathrm{min}}$ [fm]')
        for fit in fits:
            tmin,pars_jk,chi2_jk,Ndof=fit
            mean,err=jackme(pars_jk)
            
            mfc=None
            if tmin==selection:
                mfc='white'
                result=pars_jk[:,0]
                plt_y=mean[0]*yunit; plt_yerr=err[0]*yunit
                ax.axhspan(plt_y-plt_yerr,plt_y+plt_yerr,color='r',alpha=0.2,label=un2str(plt_y,plt_yerr))
                ax.legend()
                axs[0,0].axhspan(plt_y-plt_yerr,plt_y+plt_yerr,color='r',alpha=0.2)
            
            plt_x=tmin*xunit; plt_y=mean[0]*yunit; plt_yerr=err[0]*yunit
            ax.errorbar(plt_x,plt_y,plt_yerr,fmt='s',color='r',mfc=mfc)

        return fig, axs, result
    
    def makePlot_2pt_SimoneStyle(meff,fitss,xunit=1,yunit=1,E0_ref=None,selection={},ylims='auto',labelType='mN'):
        for _ in [0]:
            result={}
            fig, axd = plt.subplot_mosaic([['f1','f1','f1'],['f2','f2','f3']],figsize=(24,10))
            (ax1,ax2,ax3)=(axd[key] for key in ['f1','f2','f3'])
            label_fm=' [fm]' if xunit!=1 else None
            label_GeV=' [GeV]' if yunit!=1 else None
            ax1.set_xlabel(r'$t$'+label_fm)
            ax2.set_xlabel(r'$t_{\mathrm{min}}$'+label_fm)
            ax3.set_xlabel(r'$t_{\mathrm{min}}$'+label_fm)
            ax1.set_ylabel(r'$E_0^{\mathrm{eff}}$'+label_GeV)
            ax2.set_ylabel(r'$E_0$'+label_GeV)
            ax3.set_ylabel(r'$E_1$'+label_GeV)
            if ylims=='std_N':
                ylims=[[0.86,1.11],[0.86,1.11],[0,3.9]]
            if ylims!='auto':
                ax1.set_ylim(ylims[0]); ax2.set_ylim(ylims[1]); ax3.set_ylim(ylims[2])

            mean,err=jackme(meff)
            fitmax=find_fitmax(meff)
            
            tmin=1; tmax=fitmax+1
            plt_x=np.arange(tmin,tmax)*xunit; plt_y=mean[tmin:tmax]*yunit; plt_yerr=err[tmin:tmax]*yunit
            ax1.errorbar(plt_x,plt_y,plt_yerr,color='black',fmt='s')
            
            if E0_ref is not None:
                ax1.axhline(y=E0_ref,color='black',linestyle = '--', marker='')
                ax2.axhline(y=E0_ref,color='black',linestyle = '--', marker='', label=r'$E_0^{\mathrm{ref}}=$'+f'{E0_ref:0.3f}')
            
            Nst=len(fitss)
            probThreshold=0.1
            chi2Size=12
            DNpar=1 # DNpar=1 if meffQ else 0
            percentage_shiftMultiplier=1.5
            
            if Nst==0:
                continue
            
            fitcase='1st'
            color='r'
            fits=fitss[0]    
            fitmins=[fit[0] for fit in fits]
            pars_jk,probs_jk=jackMA(fits)
            probs_mean=np.mean(probs_jk,axis=0)
            ind_mpf=np.argmax(np.mean(probs_jk,axis=0))  
            if fitcase in selection:
                ind_select=fitmins.index(selection[fitcase])
                result[fitcase]=fits[ind_select][1]
            else:
                result[fitcase]=pars_jk
            pars_mean,pars_err=jackme(result[fitcase])
            plt_x=np.array([fitmins[0]-0.5,fitmins[-1]+0.5])*xunit; plt_y=pars_mean[0]*yunit; plt_yerr=pars_err[0]*yunit
            ax2.fill_between(plt_x,plt_y-plt_yerr,plt_y+plt_yerr,color=color,alpha=0.2,label=r'$E_0^{\mathrm{1st}}=$'+un2str(plt_y,plt_yerr))
            if ylims=='auto':
                ax1.set_ylim([plt_y-plt_yerr*20,plt_y+plt_yerr*40])
                ax2.set_ylim([plt_y-plt_yerr*20,plt_y+plt_yerr*30])
            for i,fit in enumerate(fits):
                fitmin,pars_jk,chi2_jk,Ndof=fit; prob=probs_mean[i]
                (pars_mean,pars_err)=jackme(pars_jk)
                chi2R=np.mean(chi2_jk)/Ndof
                if fitcase in selection:
                    showQ = i==ind_select
                else:
                    showQ = i==ind_mpf if probThreshold is None else prob>probThreshold
                plt_x=fitmin*xunit; plt_y=pars_mean[0]*yunit; plt_yerr=pars_err[0]*yunit
                ax2.errorbar(plt_x,plt_y,plt_yerr,fmt='s',color=color,mfc='white' if showQ else None)
                ylim=ax2.get_ylim(); chi2_shift=(ylim[1]-ylim[0])/12
                ax2.annotate("%0.1f" %chi2R,(plt_x,plt_y-plt_yerr-chi2_shift),color=color,size=chi2Size,ha='center')
                if probThreshold is not None and prob>probThreshold and (fitcase not in selection):
                    ax2.annotate(f"{int(prob*100)}%",(plt_x,plt_y-plt_yerr-chi2_shift*percentage_shiftMultiplier),color=color,size=chi2Size,ha='center')
            
            if Nst==1:
                continue
            
            fitcase='2st'
            color='g'
            fits=fitss[1]    
            fitmins=[fit[0] for fit in fits]
            pars_jk,probs_jk=jackMA(fits)
            probs_mean=np.mean(probs_jk,axis=0)
            ind_mpf=np.argmax(np.mean(probs_jk,axis=0)) 
            if fitcase in selection:
                ind_select=fitmins.index(selection[fitcase])
                result[fitcase]=fits[ind_select][1]
            else:
                result[fitcase]=pars_jk
            t=np.transpose([result[fitcase][:,0],result[fitcase][:,0]+result[fitcase][:,2-DNpar]])
            pars_mean,pars_err=jackme(t)
            plt_x=np.array([fitmins[0]-0.5,fitmins[-1]+0.5])*xunit; plt_y=pars_mean[0]*yunit; plt_yerr=pars_err[0]*yunit
            ax2.fill_between(plt_x,plt_y-plt_yerr,plt_y+plt_yerr,color=color,alpha=0.2, label=r'$E_0^{\mathrm{2st}}=$'+un2str(plt_y,plt_yerr))
            plt_x=np.array([fitmins[0]-0.5,fitmins[-1]+0.5])*xunit; plt_y=pars_mean[1]*yunit; plt_yerr=pars_err[1]*yunit
            ax3.fill_between(plt_x,plt_y-plt_yerr,plt_y+plt_yerr,color=color,alpha=0.2, label=r'$E_1^{\mathrm{2st}}=$'+un2str(plt_y,plt_yerr))
            if ylims=='auto':
                ax3.set_ylim([plt_y-plt_yerr*20,plt_y+plt_yerr*30])
            for i,fit in enumerate(fits):
                fitmin,pars_jk,chi2_jk,Ndof=fit; prob=probs_mean[i]
                t=pars_jk.copy()
                t[:,1]=pars_jk[:,0]+pars_jk[:,2-DNpar]
                (pars_mean,pars_err)=jackme(t)
                chi2R=np.mean(chi2_jk)/Ndof
                if fitcase in selection:
                    showQ = i==ind_select
                else:
                    showQ = i==ind_mpf if probThreshold is None else prob>probThreshold
                plt_x=fitmin*xunit; plt_y=pars_mean[0]*yunit; plt_yerr=pars_err[0]*yunit
                ax2.errorbar(plt_x,plt_y,plt_yerr,fmt='o',color=color,mfc='white' if showQ else None)
                ylim=ax2.get_ylim(); chi2_shift=(ylim[1]-ylim[0])/12
                ax2.annotate("%0.1f" %chi2R,(plt_x,plt_y-plt_yerr-chi2_shift),color=color,size=chi2Size,ha='center')
                if probThreshold is not None and prob>probThreshold and (fitcase not in selection):
                    ax2.annotate(f"{int(prob*100)}%",(plt_x,plt_y-plt_yerr-chi2_shift*percentage_shiftMultiplier),color=color,size=chi2Size,ha='center')
                
                plt_x=fitmin*xunit; plt_y=pars_mean[1]*yunit; plt_yerr=pars_err[1]*yunit
                ax3.errorbar(plt_x,plt_y,plt_yerr,fmt='o',color=color,mfc='white' if showQ else None)
                ylim=ax3.get_ylim(); chi2_shift=(ylim[1]-ylim[0])/12
                ax3.annotate("%0.1f" %chi2R,(plt_x,plt_y-plt_yerr-chi2_shift),color=color,size=chi2Size,ha='center')
                if probThreshold is not None and prob>probThreshold and (fitcase not in selection):
                    ax3.annotate(f"{int(prob*100)}%",(plt_x,plt_y-plt_yerr-chi2_shift*percentage_shiftMultiplier),color=color,size=chi2Size,ha='center')
                    
            if Nst==2:
                continue
            
            fitcase='3st'
            color='b'
            fits=fitss[2]    
            fitmins=[fit[0] for fit in fits]
            pars_jk,probs_jk=jackMA(fits)
            probs_mean=np.mean(probs_jk,axis=0)
            ind_mpf=np.argmax(np.mean(probs_jk,axis=0))    
            if fitcase in selection:
                ind_select=fitmins.index(selection[fitcase])
                result[fitcase]=fits[ind_select][1]
            else:
                result[fitcase]=pars_jk
            t=np.transpose([result[fitcase][:,0],result[fitcase][:,0]+result[fitcase][:,2-DNpar]])
            pars_mean,pars_err=jackme(t)
            plt_x=np.array([fitmins[0]-0.5,fitmins[-1]+0.5])*xunit; plt_y=pars_mean[0]*yunit; plt_yerr=pars_err[0]*yunit
            ax2.fill_between(plt_x,plt_y-plt_yerr,plt_y+plt_yerr,color=color,alpha=0.2, label=r'$E_0^{\mathrm{3st}}=$'+un2str(plt_y,plt_yerr))
            plt_x=np.array([fitmins[0]-0.5,fitmins[-1]+0.5])*xunit; plt_y=pars_mean[1]*yunit; plt_yerr=pars_err[1]*yunit
            ax3.fill_between(plt_x,plt_y-plt_yerr,plt_y+plt_yerr,color=color,alpha=0.2, label=r'$E_1^{\mathrm{3st}}=$'+un2str(plt_y,plt_yerr))    
            for i,fit in enumerate(fits):
                fitmin,pars_jk,chi2_jk,Ndof=fit; prob=probs_mean[i]
                t=pars_jk.copy()
                t[:,1]=pars_jk[:,0]+pars_jk[:,2-DNpar]
                (pars_mean,pars_err)=jackme(t)
                chi2R=np.mean(chi2_jk)/Ndof
                if fitcase in selection:
                    showQ = i==ind_select
                else:
                    showQ = i==ind_mpf if probThreshold is None else prob>probThreshold
                plt_x=fitmin*xunit; plt_y=pars_mean[0]*yunit; plt_yerr=pars_err[0]*yunit
                ax2.errorbar(plt_x,plt_y,plt_yerr,fmt='d',color=color,mfc='white' if showQ else None)
                ylim=ax2.get_ylim(); chi2_shift=(ylim[1]-ylim[0])/12
                ax2.annotate("%0.1f" %chi2R,(plt_x,plt_y-plt_yerr-chi2_shift),color=color,size=chi2Size,ha='center')
                if probThreshold is not None and prob>probThreshold and (fitcase not in selection):
                    ax2.annotate(f"{int(prob*100)}%",(plt_x,plt_y-plt_yerr-chi2_shift*percentage_shiftMultiplier),color=color,size=chi2Size,ha='center')
                
                plt_x=fitmin*xunit; plt_y=pars_mean[1]*yunit; plt_yerr=pars_err[1]*yunit
                ax3.errorbar(plt_x,plt_y,plt_yerr,fmt='d',color=color,mfc='white' if showQ else None)
                ylim=ax3.get_ylim(); chi2_shift=(ylim[1]-ylim[0])/12
                ax3.annotate("%0.1f" %chi2R,(plt_x,plt_y-plt_yerr-chi2_shift),color=color,size=chi2Size,ha='center') 
                if probThreshold is not None and prob>probThreshold and (fitcase not in selection):
                    ax3.annotate(f"{int(prob*100)}%",(plt_x,plt_y-plt_yerr-chi2_shift*percentage_shiftMultiplier),color=color,size=chi2Size,ha='center')
        
        ax2.legend(fontsize=16)
        ax3.legend(fontsize=16)
        return fig,axd,result           
#!============== plot (3pt) ==============#
if True:
    def plot_rainbow(ax,tf2ratio,tfmin=None,tfmax=None,tcmin=1,xunit=1,yunit=1,shift=0,mid_tfshift=0,colors=colors16,mfc=None,ax_mid=None):
        tfs=list(tf2ratio.keys())
        tfmin = min(tfs) if tfmin is None else tfmin
        tfmax = max(tfs) if tfmax is None else tfmax
        tfs=[tf for tf in tfs if tfmin<=tf<=tfmax]
        for itf,tf in enumerate(tfs):
            mean,err=jackme(tf2ratio[tf])
            tcs=np.arange(tcmin,tf-tcmin+1)
            plt_x=(tcs-tf/2+0.05*(itf-len(tfs)/2)+shift*0.1)*xunit; plt_y=mean[tcs]*yunit; plt_yerr=err[tcs]*yunit
            itf_color=tfs.index(tf)
            ax.errorbar(plt_x,plt_y,plt_yerr,color=colors[itf_color%16],fmt=fmts16[itf_color%16],mfc=mfc)
            
            if ax_mid is not None:
                plt_x=(tf+mid_tfshift+shift*0.1)*xunit; plt_y=mean[tf//2]*yunit; plt_yerr=err[tf//2]*yunit
                ax_mid.errorbar(plt_x,plt_y,plt_yerr,color=colors[itf_color%16],fmt=fmts16[itf_color%16],mfc=mfc)
    
    def makePlot_3pt(list_dic,shows=['rainbow','fit_band','fit_const','fit_sum','fit_2st'],Lrow=4,Lcol=6,colHeaders='auto',colors_rainbow=colors16,colors_fit=colors8,sharey='row',indicativeErrorBandQ=True,figAxs=None,**kwargs):
        '''
        show in ['rainbow','midpoint','fit_#','fit_#_prob'] \\
        base:[tf2ratio,fits_band,fits_const,fits_sum,fits_2st] \\
        WAMA:[fit_band_WA,fit_const_MA,fit_sum_MA,fit_2st_MA] \\
        rainbow:[tfmin,tfmax,tcmin,dt] \\
        fit_band:[tfmin,tfmax,tcmin_min,tcmin_max,dtf,dtc] \\
        fit_#:[tfmin_min,tfmin_max,tcmin_min,tcmin_max,dtf,dtc] \\
        
        mfc:[global] \\
        shift:[rainbow,midpoint,fit] \\
        '''
        if type(list_dic)==dict:
            list_dic=[list_dic]
        width_ratios=[3 if show in ['rainbow'] else 2 for show in shows]
        Nrow=len(list_dic); Ncol=len(shows)
        if figAxs is None:
            fig, axs = getFigAxs(Nrow,Ncol,Lrow=Lrow,Lcol=Lcol,sharex='col',sharey=sharey, gridspec_kw={'width_ratios': width_ratios},**kwargs)
        else:
            fig, axs = figAxs
        if colHeaders is not None:
            show2Header={'rainbow':r'ratio','midpoint':r'mid point','fit_band':r'const fit to each $t_s$',
                    'fit_const':r'const fit', 'fit_2st':r'2st fit', 'fit_sum':r'summation method',
                    'fit_const_prob':r'Fit Prob. (const)','fit_sum_prob':r'Fit Prob. (sum)','fit_2st_prob':r'Fit Prob. (2st)'}
            addColHeader(axs,[show2Header[show] for show in shows] if colHeaders=='auto' else colHeaders)
            
        show2xlabel={'rainbow':r'$t_{\rm ins}-t_{s}/2$ [fm]','midpoint':r'$t_{s}^{\rm}$ [fm]','fit_band':r'$t_{s}^{\rm}$ [fm]',
                    'fit_const':r'$t_{s}^{\rm low}$ [fm]', 'fit_2st':r'$t_{s}^{\rm low}$ [fm]', 'fit_sum':r'$t_{s}^{\rm low}$ [fm]',
                    'fit_const_prob':r'Fit Prob.','fit_sum_prob':r'Fit Prob.','fit_2st_prob':r'Fit Prob.'}
        for i in range(Ncol):
            axs[-1,i].set_xlabel(show2xlabel[shows[i]])
        
        for show in shows:
            if show.endswith('_prob'):
                ax=axs[-1,shows.index(show)]  
                xticks=[1,3,10,30,90]
                ax.set_xlim([xticks[0]/2,150])
                ax.set_xscale('log')
                ax.set_xticks(xticks)
                ax.set_xticklabels([f'{x}%' for x in xticks])
                
        # determine fit ranges
        tfs_mids_phy=[]; symQs=[]
        for irow in range(Nrow):
            dic=list_dic[irow]
            if len(dic)==0:
                continue
            def setParameter(default,key):
                if type(default) is not list:
                    return dic[key] if key in dic else default                
                if key in dic:
                    res=dic[key]
                    for i in range(len(res)):
                        if res[i] is None:
                            res[i]=default[i]
                    return res
                return default

            # general
            xunit,yunit=setParameter([1,1],'xyunit')
            xunit=setParameter(xunit,'xunit'); yunit=setParameter(yunit,'yunit')
            (tf2ratio,fits_band,fits_const,fits_sum,fits_2st)=setParameter([None,None,None,None,None],'base:[tf2ratio,fits_band,fits_const,fits_sum,fits_2st]')
            tf2ratio=setParameter(tf2ratio,'tf2ratio')
            if tf2ratio is None:
                tfs_all=[]
            else:
                tfs_all=list(tf2ratio.keys()); tfs_all.sort()
            # determine symQ
            symQ=False
            for fits in [fits_band,fits_const,fits_sum,fits_2st]:
                if fits is None:
                    continue
                symQ=isinstance(fits[0][0][1],int)
                break
            symQs.append(symQ)
            # rainbow
            (tfmin,tfmax,tcmin,dt)=setParameter([0,np.inf,1,1],'rainbow:[tfmin,tfmax,tcmin,dt]')
            tfs_rainbow=[tf for tf in tfs_all if tfmin<=tf<=tfmax and tf%dt==0 and tf>=tcmin*2]
            tcmin_rainbow=tcmin
            tfs_mid=setParameter(tfs_rainbow,'tfs_mid')
            if len(tfs_mid)>0:
                tfs_mids_phy+=[(min(tfs_mid)-1)*xunit,(max(tfs_mid)+1)*xunit]
        if 'midpoint' in shows and len(tfs_mids_phy)!=0:
            ax=axs[-1,shows.index('midpoint')]
            ax.set_xlim([min(tfs_mids_phy),max(tfs_mids_phy)])
                
        for irow in range(Nrow):
            dic=list_dic[irow]
            if len(dic)==0:
                continue
            def setParameter(default,key):
                if type(default) is not list:
                    return dic[key] if key in dic else default                
                if key in dic:
                    res=dic[key]
                    for i in range(len(res)):
                        if res[i] is None:
                            res[i]=default[i]
                    return res
                return default
            
            # general
            xunit,yunit=setParameter([1,1],'xyunit')
            xunit=setParameter(xunit,'xunit'); yunit=setParameter(yunit,'yunit')
            (tf2ratio,fits_band,fits_const,fits_sum,fits_2st)=setParameter([None,None,None,None,None],'base:[tf2ratio,fits_band,fits_const,fits_sum,fits_2st]')
            tf2ratio=setParameter(tf2ratio,'tf2ratio')
            (fit_band_WA,fit_const_MA,fit_sum_MA,fit_2st_MA)=setParameter([None,None,None,None],'WAMA:[fit_band_WA,fit_const_MA,fit_sum_MA,fit_2st_MA]')
            if tf2ratio is None:
                tfs_all=[]
            else:
                tfs_all=list(tf2ratio.keys()); tfs_all.sort()
            symQ=symQs[irow]
            convert_tcmin = (lambda tcmin:tcmin) if symQ else (lambda tcmin:tcmin[0]+tcmin[1])
            convert_tcmins = (lambda tcmins:tcmins) if symQ else (lambda tcmins:[tcmina+tcminb for tcmina,tcminb in tcmins])
            # rainbow
            (tfmin,tfmax,tcmin,dt)=setParameter([0,np.inf,1,1],'rainbow:[tfmin,tfmax,tcmin,dt]')
            tfs_rainbow=[tf for tf in tfs_all if tfmin<=tf<=tfmax and tf%dt==0 and tf>=tcmin*2]
            tcmin_rainbow=tcmin
            tfs_mid=setParameter(tfs_rainbow,'tfs_mid')
            # fit_band
            if fits_band is not None:
                tfs=removeDuplicates([fit[0][0] for fit in fits_band])
                tcmins=removeDuplicates(convert_tcmins([fit[0][1] for fit in fits_band]))
                (tfmin,tfmax,tcmin_min,tcmin_max,dtf,dtc)=setParameter([min(tfs),max(tfs),min(tcmins),max(tcmins),1,1],'fit_band:[tfmin,tfmax,tcmin_min,tcmin_max,dtf,dtc]')
                tfs_band=[tf for tf in tfs if tfmin<=tf<=tfmax and tf%dtf==0]
                tcmins_band=[tcmin for tcmin in tcmins if tcmin_min<=tcmin<=tcmin_max and tcmin%dtc==0]
            else:
                tfs_band=[]
                
            def process_fits(fits,name):
                if fits is not None:
                    tfmins=removeDuplicates([fit[0][0] for fit in fits])
                    tcmins=removeDuplicates(convert_tcmins([fit[0][1] for fit in fits]))
                    (tfmin_min,tfmin_max,tcmin_min,tcmin_max,dtf,dtc)=setParameter([min(tfmins),max(tfmins),min(tcmins),max(tcmins),1,1],f'{name}:[tfmin_min,tfmin_max,tcmin_min,tcmin_max,dtf,dtc]')
                    tfmins=[tfmin for tfmin in tfmins if tfmin_min<=tfmin<=tfmin_max and tfmin%dtf==0]
                    tcmins=[tcmin for tcmin in tcmins if tcmin_min<=tcmin<=tcmin_max and tcmin%dtc==0]
                    tfmins.sort(); tcmins.sort()
                    return tfmins,tcmins
                return None,None
            tfmins_const,tcmins_const=process_fits(fits_const,'fit_const')
            tfmins_sum,tcmins_sum=process_fits(fits_sum,'fit_sum')
            tfmins_2st,tcmins_2st=process_fits(fits_2st,'fit_2st')

            tfs_color=removeDuplicates(tfs_rainbow+tfs_band+tfs_mid)
            
            [mfc_global]=setParameter(['not set'],'mfc:[global]')
            mfc_global=mfc_global if mfc_global!='None' else None
            [shift_rainbow,shift_midpoint,shift_fit]=setParameter([0,0,0],'shift:[rainbow,midpoint,fit]')
            
            show='rainbow'
            mfc=mfc_global if mfc_global!='not set' else None
            if show in shows:
                ax=axs[irow,shows.index(show)]                
                for itf,tf in enumerate(tfs_rainbow):
                    mean,err=jackme(tf2ratio[tf])
                    tcs=np.arange(tcmin_rainbow,tf-tcmin_rainbow+1)
                    plt_x=(tcs-tf/2+0.05*(itf-len(tfs_rainbow)/2)+shift_rainbow)*xunit; plt_y=mean[tcs]*yunit; plt_yerr=err[tcs]*yunit
                    itf_color=tfs_color.index(tf)
                    ax.errorbar(plt_x,plt_y,plt_yerr,color=colors_rainbow[itf_color%16],fmt=fmts16[itf_color%16],mfc=mfc)
                    
            show='midpoint'
            mfc=mfc_global if mfc_global!='not set' else None
            if show in shows:
                ax=axs[irow,shows.index(show)]   
                for itf,tf in enumerate(tfs_mid):
                    if tf%2!=0:
                        continue
                    mean,err=jackme(tf2ratio[tf][:,tf//2])
                    plt_x=(tf+shift_midpoint)*xunit; plt_y=mean*yunit; plt_yerr=err*yunit
                    itf_color=tfs_color.index(tf)
                    ax.errorbar(plt_x,plt_y,plt_yerr,color=colors_rainbow[itf_color%16],fmt=fmts16[itf_color%16],mfc=mfc)
                    
            show='fit_band'
            if show in shows and fits_band is not None:
                ax=axs[irow,shows.index(show)]   
                fits=fits_band
                if fit_band_WA is not None:
                    pars_jk,chi2_jk,Ndof,fits_WA=fit_band_WA
                    fitlabels=[fit[0] for fit in fits_WA]
                    mean,err=jackme(pars_jk[:,0])
                    plt_y=mean*yunit; plt_yerr=err*yunit
                    if indicativeErrorBandQ:
                        temp_tfs=[tf for (tf,tcmin),*_ in fits_WA]
                        plt_x=[min(temp_tfs)*xunit,max(temp_tfs)*xunit];
                        ax.fill_between(plt_x,plt_y-plt_yerr,plt_y+plt_yerr,color='r',alpha=0.2,label=un2str(plt_y,plt_yerr))
                        ax.axhspan(plt_y-plt_yerr,plt_y+plt_yerr,color='r',alpha=0.1)
                    else:
                        ax.axhspan(plt_y-plt_yerr,plt_y+plt_yerr,color='r',alpha=0.2,label=un2str(plt_y,plt_yerr))
                    ax.legend()
                    
                tcmins=removeDuplicates(convert_tcmins([fit[0][1] for fit in fits])); tcmins.sort()
                for fit in fits:
                    (tf,tcmin),pars_jk,chi2_jk,Ndof=fit
                    if tf not in tfs_band or convert_tcmin(tcmin) not in tcmins_band:
                        continue
                    itf=tfs_color.index(tf)
                    itcmin=tcmins.index(convert_tcmin(tcmin))
                    mfc=None
                    if fit_band_WA is not None and (tf,tcmin) in fitlabels:
                        mfc='white'
                    if mfc_global!='not set':
                        mfc=mfc_global
                    mean,err=jackme(pars_jk[:,0])
                    plt_x=(tf+itcmin*0.1+shift_fit)*xunit; plt_y=mean*yunit; plt_yerr=err*yunit
                    ax.errorbar(plt_x,plt_y,plt_yerr,color=colors_rainbow[itf%16],fmt=fmts16[itf%16],mfc=mfc)
                    
            def plot_fits(show,fits,tfmins,tcmins,fit_MA):
                show_prob=show+'_prob'
                if show in shows and fits is not None:
                    ax=axs[irow,shows.index(show)]
                    if fit_MA is not None:
                        pars_jk,probs_jk,fits_MA=fit_MA
                        fitlabels=[fit[0] for fit in fits_MA]
                        probs=np.mean(probs_jk,axis=0)
                        tfs=removeDuplicates([fitlabel[0] for fitlabel in fitlabels])
                        mean,err=jackme(pars_jk[:,0])
                        plt_y=mean*yunit; plt_yerr=err*yunit
                        if indicativeErrorBandQ:
                            temp_tfmins=[tfmin for (tfmin,_),*_ in fits_MA]
                            plt_x=[min(temp_tfmins)*xunit,max(temp_tfmins)*xunit]
                            ax.fill_between(plt_x,plt_y-plt_yerr,plt_y+plt_yerr,color='r',alpha=0.2,label=un2str(plt_y,plt_yerr))
                            ax.axhspan(plt_y-plt_yerr,plt_y+plt_yerr,color='r',alpha=0.1)
                        else:
                            ax.axhspan(plt_y-plt_yerr,plt_y+plt_yerr,color='r',alpha=0.2,label=un2str(plt_y,plt_yerr))
                        ax.legend()
                        if show_prob in shows:
                            axp=axs[irow,shows.index(show_prob)]
                            axp.axhspan(plt_y-plt_yerr,plt_y+plt_yerr,color='r',alpha=0.2)
                    for fit in fits:
                        (tfmin,tcmin),pars_jk,chi2_jk,Ndof=fit
                        if tfmin not in tfmins or convert_tcmin(tcmin) not in tcmins:
                            continue
                        itcmin=tcmins.index(convert_tcmin(tcmin))
                        mfc=None
                        if fit_MA is not None and (tfmin,tcmin) in fitlabels:
                            mfc='white'
                        if mfc_global!='not set':
                            mfc=mfc_global
                        mean,err=jackme(pars_jk[:,0])
                        plt_x=(tfmin+itcmin*0.1+shift_fit)*xunit; plt_y=mean*yunit; plt_yerr=err*yunit
                        ax.errorbar(plt_x,plt_y,plt_yerr,color=colors_fit[itcmin%8],fmt=fmts8[itcmin%8],mfc=mfc)
                        
                        if show_prob in shows and (tfmin,tcmin) in fitlabels:
                            ind=fitlabels.index((tfmin,tcmin))
                            prob=probs[ind]
                            if prob<1/100:
                                continue
                            mean,err=jackme(pars_jk)
                            plt_x=(prob)*100; plt_y=mean[0]*yunit; plt_yerr=err[0]*yunit
                            axp.errorbar(plt_x,plt_y,plt_yerr,color=colors_fit[itcmin%8],fmt=fmts8[itcmin%8],mfc=mfc)

                    if show=='fit_2st' and 'fit_2st_rainbow_midpoint:[fittype,pars_jk_meff2st]' in dic:
                        fittype,pars_jk_meff2st=dic['fit_2st_rainbow_midpoint:[fittype,pars_jk_meff2st]']
                        func=fittype2func[fittype]
                        ind_mpf=np.argmax(probs)
                        (tfmin,tcmin),pars_jk,*_=fits_MA[ind_mpf]
                        if pars_jk_meff2st is not None:
                            if type(pars_jk_meff2st)==list:
                                pars_jk_meff2st=np.concatenate(pars_jk_meff2st,axis=1)
                            if fittype in ['2st2step_SYMshare'] and pars_jk_meff2st.shape[1]==3:
                                pars_jk_meff2st=pars_jk_meff2st[:,[1,2]]
                            if fittype in ['2st2step_SQRTshare','2st2step_EFITshare'] and pars_jk_meff2st.shape[1]==6:
                                pars_jk_meff2st=pars_jk_meff2st[:,[1,2,4,5]]
                            pars_jk=np.concatenate([pars_jk,pars_jk_meff2st],axis=1)
                        ax=axs[irow,shows.index('rainbow')]
                        for itf,tf in enumerate(tfs_rainbow):
                            if tf<tfmin:
                                continue
                            tcs=np.arange(tcmin,tf-tcmin,0.1) if symQ else np.arange(tcmin[0],tf-tcmin[1],0.1)
                            t=np.array([func(tf,tcs,*pars) for pars in pars_jk])
                            mean,err=jackme(t)
                            plt_x=(tcs-tf//2)*xunit; plt_y=mean*yunit; plt_yerr=err*yunit
                            itf_color=tfs_color.index(tf)
                            ax.fill_between(plt_x,plt_y-plt_yerr,plt_y+plt_yerr,color=colors_rainbow[itf_color%16],alpha=0.2)   
                        if 'midpoint' in shows:
                            ax=axs[irow,shows.index('midpoint')]
                            xlim=ax.get_xlim()
                            tfs=np.arange(xlim[0]/xunit,xlim[-1]/xunit,0.1)
                            t=np.array([func(tfs,tfs/2,*pars) for pars in pars_jk])
                            mean,err=jackme(t)
                            plt_x=tfs*xunit; plt_y=mean*yunit; plt_yerr=err*yunit
                            ax.fill_between(plt_x,plt_y-plt_yerr,plt_y+plt_yerr,color='grey',alpha=0.2)   

                            tfs=np.arange(tfmin,tfs_mid[-1],0.1)
                            t=np.array([func(tfs,tfs/2,*pars) for pars in pars_jk])
                            mean,err=jackme(t)
                            plt_x=tfs*xunit; plt_y=mean*yunit; plt_yerr=err*yunit
                            ax.fill_between(plt_x,plt_y-plt_yerr,plt_y+plt_yerr,color='grey',alpha=0.7) 
                
            plot_fits('fit_const',fits_const,tfmins_const,tcmins_const,fit_const_MA)
            plot_fits('fit_sum',fits_sum,tfmins_sum,tcmins_sum,fit_sum_MA)
            plot_fits('fit_2st',fits_2st,tfmins_2st,tcmins_2st,fit_2st_MA)
                                  
        return fig,axs

    def makePlot_3pt_rainbow(list_tf2ratio,tfmin=None,tfmax=None,tcmin=None,dt=None,xunit=1,yunit=1,**kwargs):
        if type(list_tf2ratio)==dict:
            list_tf2ratio=[list_tf2ratio]
        list_dic=[{
            'tf2ratio':tf2ratio,
            'rainbow:[tfmin,tfmax,tcmin,dt]':[tfmin,tfmax,tcmin,dt],
            'xunit':xunit, 'yunit':yunit,
        } for tf2ratio in list_tf2ratio]
        return makePlot_3pt(list_dic,shows=['rainbow','midpoint'],**kwargs)
    
    def makePlot_3pt_rainbow_compare(tf2ratio,tf2ratio2,xunit=1,yunit=1,tfmin=None,tfmax=None,tcmin=None,tcmin2=1,dt=1,shift2=0,mid_tfshift2=0,**kwargs):
        list_dic=[{
            'tf2ratio':tf2ratio,
            'rainbow:[tfmin,tfmax,tcmin,dt]':[tfmin,tfmax,tcmin,dt],
            'xunit':xunit, 'yunit':yunit,
        }]
        fig,axs=makePlot_3pt(list_dic,shows=['rainbow','midpoint'],**kwargs)
        plot_rainbow(axs[0,0],tf2ratio2,xunit=xunit,yunit=yunit,mfc='white',tcmin=tcmin2,shift=shift2,mid_tfshift=mid_tfshift2,ax_mid=axs[0,1])
        return fig,axs 
        
#!============== GEVP ==============#
if True:
    def GEVP(Ct,t0List,tList=None,tvList=None):
        '''
        Ct: indexing from t=0
        t0List>=0: t0=t0List
        t0List<0: t0=t-|t0List|
        tv: reference time for getting wave function (Not return wave function if tv is None) 
        # Return #
        eVecs (the one to combine source operators): return (time,n,i) but (time,i,n) in the middle
        wave function returns if tv is not None
        '''
        Ct=Ct.astype(complex)
        (t_total,N_op,N_op)=Ct.shape
        if tList is None:
            tList=range(t_total)
        tList=np.array(tList)
        if type(t0List)==int:
            if t0List>=0:
                t0List=[t0List for t in tList]
            else:
                t0List=[t+t0List if t+t0List>0 else 1 for t in tList]
        elif type(t0List)==str:
            if t0List=='t/2':
                t0List=[(t+1)//2 for t in tList]
        t0List=[t0 if t!=t0 else 0 if t!=0 else 1 for t,t0 in zip(tList,t0List)] # we would never use t==t0 case, this is meant to avoid some warning msg
        t0List=np.array(t0List)
        Ct0=Ct[t0List]
        choL=np.linalg.cholesky(Ct0) # Ct0=choL@choL.H
        choLInv=np.linalg.inv(choL)
        choLInvDag=np.conj(np.transpose(choLInv,[0,2,1]))
        w_Ct=choLInv@Ct[tList]@choLInvDag
        (eVals,w_eVecs)=np.linalg.eig(w_Ct)
        eVals=np.real(eVals)
        
        for ind,t in enumerate(tList):
            t0=t0List[ind]
            sortList=np.argsort(-eVals[ind]) if t0<t else np.argsort(eVals[ind]) 
            (eVals[ind],w_eVecs[ind])=(eVals[ind][sortList],w_eVecs[ind][:,sortList])

        eVecs=choLInvDag@w_eVecs
        
        if tvList is not None:
            if type(tvList)==str:
                if tvList=='t0':
                    tvList=t0List
                elif tvList=='t':
                    tvList=tList
            tvList=np.array(tvList)
            tmp=np.conj(np.transpose(eVecs,[0,2,1]))@Ct[tvList]@eVecs
            tmp=np.real(tmp[:,range(N_op),range(N_op)])
            powers=np.array([ tv/(t-t0) if t!=t0 else 0 for t,t0,tv in zip(tList,t0List,tvList)])
            fn=np.sqrt( tmp / (eVals**powers[:,None]))
            eVecs_normalized=eVecs/fn[:,None,:]
            eVecs_normalized=np.transpose(eVecs_normalized,[0,2,1]) # v^n_i
            Zin=np.linalg.inv(eVecs_normalized) # (Z)_in
            
            return (eVals,eVecs_normalized,Zin)

        return (eVals,np.transpose(eVecs,[0,2,1]))

#!============== ensemble info ==============#
if True:
    ens2full={'a24':'cA211.53.24','a':'cA2.09.48','b':'cB211.072.64','c':'cC211.060.80','d':'cD211.054.96','e':'cE211.044.112'}
    ens2label={'a24':'A24','a':'A48','b':'B64','c':'C80','d':'D96','e':'E112'}
    ens2a={'a24':0.0908,'a':0.0938,'b':0.07957,'c':0.06821,'d':0.05692,'e':0.04892} # fm
    ens2NL={'a24':24,'a':48,'b':64,'c':80,'d':96,'e':112}
    ens2NT={'a24':24*2,'a':48*2,'b':64*2,'c':80*2,'d':96*2,'e':112*2}
    ens2amul={'a24':0.0053,'a':0.0009,'b':0.00072,'c':0.00060,'d':0.00054,'e':0.00044}
    
    ens2amul_iso={'b':0.0006669,'c':0.0005864,'d':0.0004934,'e': 0.0004306}
    ens2amul_iso_err={'b':0.0000028,'c':0.0000034,'d':0.0000024,'e': 0.0000023}

    ens2aInv={ens:1/(ens2a[ens]*hbarc) for ens in ens2a.keys()} # MeV

#!============== obsolete  ==============#
if False:
    app_init=[['pi0i',{'pib'}],['pi0f',{'pia'}],['j',{'j'}],['P',{'pia','pib'}],\
        ['jPi',{'j','pib'}],['jPf',{'pia','j'}],['PJP',{'pia','j','pib'}]]
    diag_init=[
        [['N','N_bw'],{'N'},\
            [[],['pi0i'],['pi0f'],['P'],['pi0f','pi0i'], ['j'],['jPi'],['j','pi0i'],['jPf'],['pi0f','j']]],
        [['T','T_bw'],{'N','pib'},\
            [[],['pi0f'],['j']]],
        [['B2pt','W2pt','Z2pt','B2pt_bw','W2pt_bw','Z2pt_bw'],{'N','pia','pib'},\
            [[]]],
        [['NJN'],{'N','j'},\
            [[],['pi0i'],['pi0f']]],
        [['B3pt','W3pt','Z3pt'],{'N','j','pib'},\
            [[]]],
        # [['NpiJNpi'],{'N','pia','j','pib'},\
        #     [[]]],
    ]

    diags_all=set(); diags_pi0Loopful=set(); diags_jLoopful=set()

    diag2baps={}; diag2dgtp={} # baps=base+apps; dgtp=diagram type
    for app,dgtp in app_init:
        diag2dgtp[app]=dgtp
    for bases,base_dgtp,appss in diag_init:
        for base in bases:
            if base.endswith('_bw'):
                continue
            for apps in appss:
                diag='-'.join([base]+apps)
                diag2baps[diag]=(base,apps)
                diag2dgtp[diag]=set.union(*([base_dgtp]+[diag2dgtp[app] for app in apps]))
                # if diag2dgtp[diag]=={'N','pia'}:
                #     continue
                # if diag2dgtp[diag]=={'N','pia','j'}:
                #     continue

                diags_all.add(diag)
                if 'pi0i' in apps or 'pi0f' in apps:
                    diags_pi0Loopful.add(diag)
                if 'j' in apps:
                    diags_jLoopful.add(diag)
                
    diags_loopful = diags_pi0Loopful | diags_jLoopful
    diags_loopless = diags_all - diags_loopful
    diags_jLoopless = diags_all - diags_jLoopful
    diags_pi0Loopless = diags_all - diags_pi0Loopful

    diag='P'
    diags_all.add(diag); diags_loopless.add(diag); diags_jLoopless.add(diag); diags_pi0Loopless.add(diag)
    diag='pi0f-pi0i'
    diags_all.add(diag); diags_loopful.add(diag); diags_jLoopless.add(diag)

    def load(path,d=0,nmin=6000):
        print('loading: '+path)
        data_load={}
        with h5py.File(path) as f:
            cfgs=[cfg.decode() for cfg in f['cfgs']]
            Ncfg=len(cfgs); Njk=len(jackknife(np.zeros(Ncfg),d=d,nmin=nmin))
            
            datasets=[]
            def visit_function(name,node):
                if isinstance(node, h5py.Dataset):
                    datasets.append(name)
                    # print(len(datasets),name,end='\r')
            f.visititems(visit_function)
                
            N=len(datasets)
            for i,dataset in enumerate(datasets):
                if 'data' in dataset:
                    data_load[dataset]=jackknife(f[dataset][()],d=d,nmin=nmin)
                else:
                    data_load[dataset]=f[dataset][()]
                print(str(i+1)+'/'+str(N)+': '+dataset,end='                           \r')
            print()

        def op_new(op,fla):
            t=op.split(';')
            t[-1]=fla
            return ';'.join(t)
        gjList=['id','gx','gy','gz','gt','g5','g5gx','g5gy','g5gz','g5gt','sgmyz','sgmzx','sgmxy','sgmtx','sgmty','sgmtz']
        diags=set([dataset.split('/')[1] for dataset in list(data_load.keys()) if 'diags' in dataset])
        opabsDic={}
        for diag in diags:
            opabsDic[diag]=[opab.decode() for opab in data_load['/'.join(['diags',diag,'opabs'])]]
            
        data={'2pt':{},'3pt':{},'VEV':{},'cfgs':[cfgs,Ncfg,Njk]}
        for dataset in data_load.keys():
            if not (dataset.startswith('diags') and 'data' in dataset):
                continue
            _,diag,_,fla=dataset.split('/')
            opabs=opabsDic[diag]
            
            npt='3pt' if '_deltat_' in dataset else '2pt'
            if npt =='2pt':
                for i,opab in enumerate(opabs):
                    opa,opb=str(opab).split('_')
                    flaa,flab=str(fla).split('_')
                    opa=op_new(opa,flaa); opb=op_new(opb,flab)
                    opab=opa+'_'+opb
                    if opab not in data[npt].keys():
                        data[npt][opab]={}
                    data[npt][opab][diag]=data_load[dataset][:,:,i]
            else:
                for i,opab in enumerate(opabs):
                    opa,opb=str(opab).split('_')
                    flaa,j,flab,_,tf=str(fla).split('_')
                    opa=op_new(opa,flaa); opb=op_new(opb,flab)
                    opab=opa+'_'+opb
                    if opab not in data[npt].keys():
                        data[npt][opab]={}
                    for i_gm,gm in enumerate(gjList):
                        insert='_'.join([gm,j,tf])
                        if insert not in data[npt][opab]:
                            data[npt][opab][insert]={}
                        data[npt][opab][insert][diag]=data_load[dataset][:,:,i,i_gm]   

        data['VEV']['j']={}
        for dataset in data_load.keys():
            if not (dataset.startswith('VEV') and 'data' in dataset):
                continue
            npt='VEV'
            _,diag,_,fla=dataset.split('/')
            if diag=='j':
                for i_gm,gm in enumerate(gjList):
                    insert='_'.join([gm,fla])
                    data[npt][diag][insert]=data_load[dataset][:,i_gm]
            elif diag=='pi0f':
                # print(dataset)
                data[npt][diag]={'sgm':data_load[dataset]}
            
        return data

    def getNpar(op):
        return {'p':1,'n,pi+':2,'p,pi0':2,'12':2}[op.split(';')[-1]]

    def getNpars(opa,opb):
        return (getNpar(opa),getNpar(opb))

    def pt2irrep(pt):
        return {'0,0,0':'G1g','0,0,1':'G1','0,0,-1':'G1','0,1,0':'G1','0,-1,0':'G1','1,0,0':'G1','-1,0,0':'G1'}[pt]
    def getop(pt,l,of):
        occ,fla=of
        return ';'.join(['g',pt,pt2irrep(pt),occ,l,fla])
    def getopab(pt,l,ofa,ofb):
        return getop(pt,l,ofa),getop(pt,l,ofb)
    def getops(pt,l,ofs):
        return [getop(pt,l,of) for of in ofs]    
    def op_getl_sgn(op):
        return {'l1':-1,'l2':1}[op.split(';')[-2]]
    def op_flipl(op):
        t=op.split(';')
        t[-2]={'l1':'l2','l2':'l1'}[t[-2]]
        return ';'.join(t)

    gtCj={'id':1,'gx':-1,'gy':-1,'gz':-1,'gt':1,'g5':-1,'g5gx':-1,'g5gy':-1,'g5gz':-1,'g5gt':1,
        'sgmxy':-1,'sgmyz':-1,'sgmzx':-1,'sgmtx':1,'sgmty':1,'sgmtz':1} # gt G^dag gt = (gtCj) G

    fourCPTstar={'id':1,'gx':-1,'gy':-1,'gz':-1,'gt':1,'g5':-1,'g5gx':1,'g5gy':1,'g5gz':1,'g5gt':-1,
            'sgmxy':1,'sgmyz':1,'sgmzx':1,'sgmtx':-1,'sgmty':-1,'sgmtz':-1} # g4CPT G^* g4CPT = (fourCPTstar) G
