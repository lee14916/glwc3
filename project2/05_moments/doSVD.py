'''
cat data_aux/ens_n2qpp1_run | xargs -I @ -P 10 python3 -u doSVD.py -e @ > log/doSVD.out & 
'''
import util as yu
from util import *
import util_moments as yum
import click
from scipy.linalg import sqrtm

#============================= input start

cd='conn'
cd_2pt=cd

basepath_2pt_conn=f'/p/project1/ngff/li47/code/projectData/05_moments/'
basepath_2pt_disc=basepath_2pt_conn

basepath_3pt_conn=basepath_2pt_conn
basepath_3pt_disc=f'/p/project1/ngff/li47/code/projectData/05_moments/'
basepath_3pt={'conn':basepath_3pt_conn,'disc':basepath_3pt_disc}[cd]

basepath_output=f'{basepath_3pt}doSVD/'

ens2msq2pars_jk=yu.load_pkl('pkl/analysis_c2pt/reg_ignore/ens2msq2pars_jk.pkl')
ens2RCs_me=yu.load_pkl('data_aux/RCs.pkl')

js_conn=['j+;conn','j-;conn']
js_disc=['jg;stout20']
js={'conn':js_conn,'disc':js_disc}[cd]

cases_munu=['unequal','equal','all']
cases_SVD=['err','cov','errTopMid','covTopMid']
cases=[(c1,c2) for c1 in cases_munu for c2 in cases_SVD]

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
        path=f'{inpath}conn_{yu.mom2str(mom)}.h5'
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
                tcs_tfby2=(np.arange(tf+1)-tf/2)[None,:,None,None]
                ratio=c3pt / np.sqrt(c2pta[:,tf:tf+1,None,None]*c2ptb[:,tf:tf+1,None,None]) / np.sqrt(np.exp(+E0a*tcs_tfby2)*np.exp(-E0b*tcs_tfby2))
                # ratio=c3pt/np.sqrt(
                #     c2pta[:,tf:tf+1]*c2ptb[:,tf:tf+1]*\
                #     c2pta[:,:tf+1][:,::-1]/c2pta[:,:tf+1]*\
                #     c2ptb[:,:tf+1]/c2ptb[:,:tf+1][:,::-1]
                # )[:,:,None,None]
                j2mom2tf2ratio[j][mom][tf]=ratio
    
    return j2mom2tf2ratio

def doSVD_err(G,M):
    err=yu.jackme(M)[-1]
    covIsq=np.diag(1/err)
    def get(g,m):
        gt=covIsq@g
        u,s,vT=np.linalg.svd(gt)
        sI=np.zeros(gt.T.shape)
        np.fill_diagonal(sI,1/s)
        return vT.T@sI@(u.T)@covIsq@m
    F=np.array([get(g,m) for g,m in zip(G,M)])
    return F
def doSVD_cov(G,M):
    cov=yu.jackmec(M)[-1]
    cov=np.diag(np.diag(cov))
    covI=np.linalg.inv(cov)
    covIsq = sqrtm(covI)
    def get(g,m):
        gt=covIsq@g
        u,s,vT=np.linalg.svd(gt)
        sI=np.zeros(gt.T.shape)
        np.fill_diagonal(sI,1/s)
        return vT.T@sI@(u.T)@covIsq@m
    F=np.array([get(g,m) for g,m in zip(G,M)])
    return F

funcs_ri=[np.real,np.imag]
def get_tf2ratio_SVD(ens,mom2tf2ratio,case_munu,extra=None):
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
    if rank!=3:
        return None
    
    tfs=mom2tf2ratio[tuple(moms[0])].keys()
    
    res_tf2ratio={}
    for tf in tfs:
        M_all=np.transpose([funcs_ri[ri](mom2tf2ratio[tuple(mom)][tf][:,:,projs.index(proj),inserts.index(insert)]) for mom,proj,insert,ri in mpirs],[1,2,0])
        
        if case_munu=='all':
            inds=[i for i,mpir in enumerate(mpirs) if mpir[2][0]!=mpir[2][1]]
            M_all[:,:,inds]*=extra
        
        t=np.zeros([Njk,tf+1,rank])
        for tc in range(tf+1):
            F=doSVD_err(G,M_all[:,tc])
            t[:,tc,:]=F
        res_tf2ratio[tf]=t
    
    return res_tf2ratio


ens2RCs_me=yu.load_pkl('data_aux/RCs.pkl')

@click.command()
@click.option('-e','--ens_n2qpp1')
def run(ens_n2qpp1):
    outfile=f'{basepath_output}conn_{ens_n2qpp1}.h5'
    outfile_flag=outfile+'_flag'
    if os.path.isfile(outfile) and (not os.path.isfile(outfile_flag)):
        print('flag_skip: ' + ens_n2qpp1)
        return
    with open(outfile_flag,'w') as f:
        pass
    
    ens,n2qpp1=ens_n2qpp1.split('_')
    n2qpp1=tuple([int(ele) for ele in n2qpp1.split(',')])
    n2q,n2p,n2p1=n2qpp1
    assert(n2q!=0 and n2p>=n2p1)
    
    inpath=f'{basepath_3pt}{yu.ens2full[ens]}/data_merge/'
    moms=[[int(e) for e in file[5:-3].split(',')] for file in os.listdir(inpath) \
        if file.startswith('conn') and file.endswith('.h5') and not file.endswith('n.h5') and not file.endswith('t.h5')]
    moms=[mom for mom in moms if yu.mom2n2qpp1_sym(mom)==n2qpp1]
    
    j2mom2tf2ratio=extractRatio(ens,moms)
    
    
    with h5py.File(outfile,'w') as f:
        for j in js:
            for case_munu in cases_munu:
                extra=None
                if case_munu in ['all']:
                    Zqq={'j+;conn':'Zqq^s','j-;conn':'Zqq'}[j]
                    rescale=ens2RCs_me[ens][f'{Zqq}(mu!=nu)']/ens2RCs_me[ens][f'{Zqq}(mu=nu)']
                    extra=rescale
                tf2ratio=get_tf2ratio_SVD(ens,j2mom2tf2ratio[j],case_munu,extra=extra)
                if tf2ratio is None:
                    continue                
                for tf in tf2ratio.keys():
                    for i,ff in enumerate(['A20','B20','C20']):
                        f.create_dataset(f'{case_munu}/{ff}_{j}_{tf}',data=tf2ratio[tf][:,:,i])
            
    os.remove(outfile_flag)
    print('flag_done: ' + ens_n2qpp1)

run()