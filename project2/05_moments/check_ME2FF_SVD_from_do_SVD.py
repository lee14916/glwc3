import util as yu
from util import *
import util_moments as yum

#============================= input start

cd='disc'
cd_2pt=cd

basepath_2pt_conn=f'/p/project1/ngff/li47/code/projectData/05_moments/'
basepath_2pt_disc=basepath_2pt_conn

basepath_3pt_conn=basepath_2pt_conn
basepath_3pt_disc=f'/p/project1/ngff/li47/code/scratch/run/05_moments_run5/'
basepath_3pt={'conn':basepath_3pt_conn,'disc':basepath_3pt_disc}[cd]

basepath_output=f'{basepath_3pt}doSVD/'

stouts=[10]
js_conn=['j+;conn','j-;conn']
js_disc=[f'{j};disc' for j in ['j+']] + [f'jg;stout{stout}' for stout in stouts]
js={'conn':js_conn,'disc':js_disc}[cd]

cases_munu_conn=['all']
cases_munu_disc=['unequal']
cases_munu={'conn':cases_munu_conn,'disc':cases_munu_disc}[cd]
if 'all' in cases_munu:
    ens2RCs_me=yu.load_pkl('data_aux/RCs.pkl')

cases_SVD_conn=['err']
cases_SVD_disc=['err']
cases_SVD={'conn':cases_SVD_conn,'disc':cases_SVD_disc}[cd]
cases=[(c1,c2) for c1 in cases_munu for c2 in cases_SVD]

ratiotype=['sqrt','Efit'][1]

ens2msq2pars_jk=yu.load_pkl('pkl/analysis_c2pt/reg_ignore/ens2msq2pars_jk.pkl')  

#============================= input end

assert(cd in ['conn','disc'])
assert(cd_2pt in ['conn','disc'])

projs=['P0', 'Px', 'Py', 'Pz']
inserts=['tt', 'tx', 'ty', 'tz', 'xx', 'xy', 'xz', 'yy', 'yz', 'zz']

os.makedirs(basepath_output,exist_ok=True)

def mom2num(mom):
    moms=yu.mom2moms(list(mom))
    return len(moms)
def extract2pt_conn(ens,n2qpp1):
    inpath=f'{basepath_2pt_conn}{yu.ens2full[ens]}/data_merge/'
    n2q,n2p,n2p1=n2qpp1
    path=f'{inpath}conn_2pt.h5'
    
    tf2c2pta,tf2c2ptb={},{}
    with h5py.File(path) as f:
        moms=yu.moms2list(f['moms'])
        msqs=[yu.mom2msq(mom) for mom in moms]
        
        # sink
        inds=np.where(np.array(msqs)==n2p1)[0]
        Ns=[mom2num(moms[ind]) for ind in inds]
        weights=np.array(Ns)/np.sum(Ns)
        for tfstr in f['data'].keys():
            tf=int(tfstr) 
            t=np.real(f['data'][tfstr][:])
            t=np.sum([t[:,:,ind]*weights[i] for i,ind in enumerate(inds)],axis=0)
            t=yu.jackknife(t)
            tf2c2pta[tf]=t
        
        # source
        inds=np.where(np.array(msqs)==n2p)[0]
        Ns=[mom2num(moms[ind]) for ind in inds]
        weights=np.array(Ns)/np.sum(Ns)
        for tfstr in f['data'].keys():
            tf=int(tfstr) 
            t=np.real(f['data'][tfstr][:])
            t=np.sum([t[:,:,ind]*weights[i] for i,ind in enumerate(inds)],axis=0)
            t=yu.jackknife(t)
            tf2c2ptb[tf]=t

    return tf2c2pta,tf2c2ptb
def extract2pt_disc(ens,n2qpp1):
    inpath=f'{basepath_2pt_disc}{yu.ens2full[ens]}/data_merge/'
    n2q,n2p,n2p1=n2qpp1
    path=f'{inpath}disc_2pt.h5'
    
    with h5py.File(path) as f:
        moms=yu.moms2list(f['moms'])
        msqs=[yu.mom2msq(mom) for mom in moms]
        
        # sink
        inds=np.where(np.array(msqs)==n2p1)[0]
        Ns=[mom2num(moms[ind]) for ind in inds]
        weights=np.array(Ns)/np.sum(Ns)
        t=np.real(f['data/N_N'][:])
        t=np.sum([t[:,:,ind]*weights[i] for i,ind in enumerate(inds)],axis=0)
        t=yu.jackknife(t)
        c2pta=t
        
        # source
        inds=np.where(np.array(msqs)==n2p)[0]
        Ns=[mom2num(moms[ind]) for ind in inds]
        weights=np.array(Ns)/np.sum(Ns)
        t=np.real(f['data/N_N'][:])
        t=np.sum([t[:,:,ind]*weights[i] for i,ind in enumerate(inds)],axis=0)
        t=yu.jackknife(t)
        c2ptb=t

    return c2pta,c2ptb

conj_sgns=np.array([1,-1,-1,-1,1, 1,1,1,1,1])[None,None,None,:]
def extract3pt(ens,moms):
    inpath=f'{basepath_3pt}{yu.ens2full[ens]}/data_merge/'
    n2qpp1=yu.mom2n2qpp1_sym(moms[0])
    
    j2mom2tf2c3pt={j:{} for j in js}
    for mom in moms:
        path=f'{inpath}{cd}_{yu.mom2str(mom)}.h5'
        for j in js:
            j2mom2tf2c3pt[j][tuple(mom)]={}
        with h5py.File(path) as f:
            for key in f['data'].keys():
                j,tf=key.split('_'); tf=int(tf)
                if j not in js:
                    continue
                c3pt=f['data'][key][:,:,0,:,:]
                c3pt=yu.jackknife(c3pt)
                j2mom2tf2c3pt[j][tuple(mom)][tf]=c3pt
    
    # symmetrize using conjugation
    for j in js:
        for mom in moms:
            assert(np.all(mom==yu.mom3pt2standard(mom)))
            t_mom=yu.mom3pt2standard_sym(mom)
            if np.all(mom==t_mom):
                continue
            
            for tf in j2mom2tf2c3pt[j][tuple(t_mom)].keys():
                j2mom2tf2c3pt[j][tuple(t_mom)][tf] = ( j2mom2tf2c3pt[j][tuple(t_mom)][tf] + np.conj(j2mom2tf2c3pt[j][tuple(t_mom)][tf]*conj_sgns) )/2
            del j2mom2tf2c3pt[j][tuple(mom)]
            
    return j2mom2tf2c3pt
    
def extractRatio(ens,moms):
    n2qpp1=yu.mom2n2qpp1_sym(moms[0])
    n2q,n2p,n2p1=n2qpp1
    
    if cd_2pt=='conn':
        tf2c2pta,tf2c2ptb=extract2pt_conn(ens,n2qpp1)
    else:
        c2pta,c2ptb=extract2pt_disc(ens,n2qpp1)
        
    j2mom2tf2c3pt=extract3pt(ens,moms)
    
    if ratiotype in ['Efit']:
        E0a=ens2msq2pars_jk[ens][n2p1][:,0][:,None,None,None]
        E0b=ens2msq2pars_jk[ens][n2p][:,0][:,None,None,None]
    
    j2mom2tf2ratio={j:{} for j in js}
    for j in js:
        for mom in j2mom2tf2c3pt[j].keys():
            j2mom2tf2ratio[j][mom]={}
            for tf in j2mom2tf2c3pt[j][mom].keys():
                c3pt=j2mom2tf2c3pt[j][mom][tf]
                if cd_2pt=='conn':
                    c2pta=tf2c2pta[tf]; c2ptb=tf2c2ptb[tf]
                
                if ratiotype in ['Efit']:
                    tcs_tfby2=(np.arange(tf+1)-tf/2)[None,:,None,None]
                    ratio=c3pt / np.sqrt(c2pta[:,tf:tf+1,None,None]*c2ptb[:,tf:tf+1,None,None]) / np.sqrt(np.exp(+E0a*tcs_tfby2)*np.exp(-E0b*tcs_tfby2))
                elif ratiotype in ['sqrt']:
                    ratio=c3pt/np.sqrt(
                        c2pta[:,tf:tf+1]*c2ptb[:,tf:tf+1]*\
                        c2pta[:,:tf+1][:,::-1]/c2pta[:,:tf+1]*\
                        c2ptb[:,:tf+1]/c2ptb[:,:tf+1][:,::-1]
                    )[:,:,None,None]
                else:
                    1/0
                    
                j2mom2tf2ratio[j][mom][tf]=ratio
    
    return j2mom2tf2ratio

def cov2covIsq(cov):
    singleQ=False
    if cov.ndim == 2:
        cov=cov[None,:,:]
        singleQ=True
    eigvals, eigvecs = np.linalg.eigh(cov)
    sqrt_eigvals = 1/np.sqrt(eigvals)
    covIsq = np.einsum('...ik,...k,...jk->...ij', eigvecs, sqrt_eigvals, eigvecs)
    if singleQ:
        covIsq=covIsq[0]
    return covIsq

def doSVD_cov(G,M,covIsq):
    Gt=np.einsum('Tma,Caf->CTmf',covIsq,G)
    U,S,VT=np.linalg.svd(Gt,full_matrices=False) # U=CTmf, S=CTf, VT=CTff
    F=np.einsum('CTbf,CTb,CTcb,Tcd,CTd->CTf',VT,1/S,U,covIsq,M)
    return F
def doSVD_err(G,M,errI):
    Gt=np.einsum('Tm,Cmf->CTmf',errI,G)
    U,S,VT=np.linalg.svd(Gt,full_matrices=False) # U=CTmf, S=CTf, VT=CTff
    F=np.einsum('CTbf,CTb,CTcb,Tc,CTc->CTf',VT,1/S,U,errI,M)
    return F

def doSVD_cov1(G,M,covIsq):
    Gt=np.einsum('ma,Caf->Cmf',covIsq,G)
    U,S,VT=np.linalg.svd(Gt,full_matrices=False) # U=Cmf, S=Cf, VT=Cff
    F=np.einsum('Cbf,Cb,Ccb,cd,CTd->CTf',VT,1/S,U,covIsq,M)
    return F
def doSVD_err1(G,M,errI):
    Gt=np.einsum('m,Cmf->Cmf',errI,G)
    U,S,VT=np.linalg.svd(Gt,full_matrices=False) # U=Cmf, S=Cf, VT=Cff
    F=np.einsum('Cbf,Cb,Ccb,c,CTc->CTf',VT,1/S,U,errI,M)
    return F

funcs_ri=[np.real,np.imag]
def get_tf2ratio_SVD(ens,mom2tf2ratio,case,extra=None):
    (case_munu,case_SVD)=case
    
    moms=[list(mom) for mom in mom2tf2ratio.keys()]    
    if case_munu=='unequal':
        mpirs=[(mom,proj,insert,ri) for mom in moms for proj in projs for insert in inserts for ri in [0,1] if insert[0]!=insert[1] and yum.useQ(mom,proj,insert)[ri]]
    elif case_munu=='equal':
        mpirs=[(mom,proj,insert,ri) for mom in moms for proj in projs for insert in inserts for ri in [0,1] if insert[0]==insert[1] and yum.useQ(mom,proj,insert)[ri]]
    elif case_munu=='all':
        mpirs=[(mom,proj,insert,ri) for mom in moms for proj in projs for insert in inserts for ri in [0,1] if yum.useQ(mom,proj,insert)[ri]]
    else:
        1/0
        
    mN=ens2msq2pars_jk[ens][0][:,0]
    Njk=len(mN)
    def mom2pvec(mom):
        return (np.array(mom[:3])+np.array(mom[3:]))*(2*np.pi/yu.ens2NL[ens])
    def mom2pvec1(mom):
        return np.array(mom[:3])*(2*np.pi/yu.ens2NL[ens])
    G=np.array([[funcs_ri[ri](yum.ME2FF(m,mom2pvec(mom),mom2pvec1(mom),proj,insert)) for mom,proj,insert,ri in mpirs] for m in mN])
    if len(G[0])==0:
        rank=0
    else:
        U, S, VT = np.linalg.svd(G[0])
        tol = 1e-10
        rank = np.sum(S > tol)
    # if rank!=3:
    #     return None
    
    tfs=list(mom2tf2ratio[tuple(moms[0])].keys())
    tfs.sort()
    
    tf2Mall={}
    for tf in tfs:
        M_all=np.transpose([funcs_ri[ri](mom2tf2ratio[tuple(mom)][tf][:,:,projs.index(proj),inserts.index(insert)]) for mom,proj,insert,ri in mpirs],[1,2,0])
        if case_munu=='all':
            inds=[i for i,mpir in enumerate(mpirs) if mpir[2][0]!=mpir[2][1]]
            M_all[:,:,inds]*=extra
        tf2Mall[tf]=M_all
    return tf2Mall
        
    
    if case_SVD in ['err1c','cov1c']:
        tf=max(tfs); tc=tf//2
        M_all=np.transpose([funcs_ri[ri](mom2tf2ratio[tuple(mom)][tf][:,tc,projs.index(proj),inserts.index(insert)]) for mom,proj,insert,ri in mpirs],[1,0])
        _,errUSE,covUSE=yu.jackmec(M_all)
        errIUSE=1/errUSE
        
        if case_SVD in ['cov1c']:
            covIsqUSE=cov2covIsq(covUSE)
    
    res_tf2ratio={}
    for tf in tfs:
        M_all=np.transpose([funcs_ri[ri](mom2tf2ratio[tuple(mom)][tf][:,:,projs.index(proj),inserts.index(insert)]) for mom,proj,insert,ri in mpirs],[1,2,0])
        if case_munu=='all':
            inds=[i for i,mpir in enumerate(mpirs) if mpir[2][0]!=mpir[2][1]]
            M_all[:,:,inds]*=extra
            
        if case_SVD=='err':
            err=np.array([yu.jackme(M_all[:,tc])[-1] for tc in range(tf+1)]); errI=1/err
            F=doSVD_err(G,M_all,errI)
        elif case_SVD=='cov':
            cov=np.array([yu.jackmec(M_all[:,tc])[-1] for tc in range(tf+1)]); covIsq=cov2covIsq(cov)
            F=doSVD_cov(G,M_all,covIsq)
        elif case_SVD in ['err1c']:
            F=doSVD_err1(G,M_all,errIUSE)
        elif case_SVD in ['cov1c']:
            F=doSVD_cov1(G,M_all,covIsqUSE)
        res_tf2ratio[tf]=F
    
    return res_tf2ratio


ens2RCs_me=yu.load_pkl('data_aux/RCs.pkl')


def run(ens_n2qpp1):
    ens,n2qpp1=ens_n2qpp1.split('_')
    n2qpp1=tuple([int(ele) for ele in n2qpp1.split(',')])
    n2q,n2p,n2p1=n2qpp1
    assert(n2q!=0 and n2p>=n2p1)
    
    inpath=f'{basepath_3pt}{yu.ens2full[ens]}/data_merge/'
    moms=[[int(e) for e in file[5:-3].split(',')] for file in os.listdir(inpath) \
        if file.startswith(cd) and file.endswith('.h5') and not file.endswith('n.h5') and not file.endswith('t.h5')]
    moms=[mom for mom in moms if yu.mom2n2qpp1_sym(mom)==n2qpp1]
    
    j2mom2tf2ratio=extractRatio(ens,moms)
    
    assert(len(cases)==1)
    case=cases[0]
    
    j2tf2Mall={}
    for j in js:
        extra=None
        if case[0] in ['all']:
            Zqq={'j+;conn':'Zqq^s','j-;conn':'Zqq'}[j]
            rescale=ens2RCs_me[ens][f'{Zqq}(mu!=nu)']/ens2RCs_me[ens][f'{Zqq}(mu=nu)']
            extra=rescale
        tf2Mall=get_tf2ratio_SVD(ens,j2mom2tf2ratio[j],case,extra=extra)
        j2tf2Mall[j]=tf2Mall
    
    return j2tf2Mall
    
    # for j in js:
    #     for case in cases:
    #         case_str='_'.join(case)
    #         extra=None
    #         if case[0] in ['all']:
    #             Zqq={'j+;conn':'Zqq^s','j-;conn':'Zqq'}[j]
    #             rescale=ens2RCs_me[ens][f'{Zqq}(mu!=nu)']/ens2RCs_me[ens][f'{Zqq}(mu=nu)']
    #             extra=rescale
    #         tf2ratio=get_tf2ratio_SVD(ens,j2mom2tf2ratio[j],case,extra=extra)
    #         if tf2ratio is None:
    #             continue                