'''
cat data_aux/ens_n2qpp1_conn_run | xargs -I @ -P 10 python3 -u doSVD_conn.py -e @ > log/doSVD_conn.out & 
'''
import util as yu
from util import *
import util_moments as yum
import click

basepath=f'/p/project1/ngff/li47/code/projectData/05_moments/'
ens2msq2pars_jk=yu.load_pkl('pkl/analysis_c2pt/reg_ignore/ens2msq2pars_jk.pkl')

projs=['P0', 'Px', 'Py', 'Pz']
inserts=['tt', 'tx', 'ty', 'tz', 'xx', 'xy', 'xz', 'yy', 'yz', 'zz']
js=['j+;conn','j-;conn']

def mom2num(mom):
    moms=yu.mom2moms(list(mom))
    return len(moms)
def extract2pt(ens,n2qpp1):
    inpath=f'{basepath}{yu.ens2full[ens]}/data_merge/'
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

conj_sgns=np.array([1,-1,-1,-1,1, 1,1,1,1,1])[None,None,None,:]
def extract3pt(ens,moms):
    inpath=f'{basepath}{yu.ens2full[ens]}/data_merge/'
    n2qpp1=yu.mom2n2qpp1_sym(moms[0])
    
    j2mom2tf2c3pt={j:{} for j in js}
    for mom in moms:
        path=f'{inpath}conn_{yu.mom2str(mom)}.h5'
        for j in js:
            j2mom2tf2c3pt[j][tuple(mom)]={}
        with h5py.File(path) as f:
            for key in f['data'].keys():
                j,tf=key.split('_'); tf=int(tf)
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
    inpath=f'{basepath}{yu.ens2full[ens]}/data_merge/'
    n2qpp1=yu.mom2n2qpp1_sym(moms[0])
    n2q,n2p,n2p1=n2qpp1
    
    tf2c2pta,tf2c2ptb=extract2pt(ens,n2qpp1)
    j2mom2tf2c3pt=extract3pt(ens,moms)
    
    E0a=ens2msq2pars_jk[ens][n2p1][:,0][:,None,None,None]
    E0b=ens2msq2pars_jk[ens][n2p][:,0][:,None,None,None]
    
    j2mom2tf2ratio={j:{} for j in js}
    for j in js:
        for mom in j2mom2tf2c3pt[j].keys():
            j2mom2tf2ratio[j][mom]={}
            for tf in j2mom2tf2c3pt[j][mom].keys():
                c3pt=j2mom2tf2c3pt[j][mom][tf]
                c2pta=tf2c2pta[tf]; c2ptb=tf2c2ptb[tf]
                tcs_tfby2=(np.arange(tf+1)-tf/2)[None,:,None,None]
                ratio=c3pt / np.sqrt(c2pta[:,tf:tf+1,None,None]*c2ptb[:,tf:tf+1,None,None]) / np.sqrt(np.exp(+E0a*tcs_tfby2)*np.exp(-E0b*tcs_tfby2))
                j2mom2tf2ratio[j][mom][tf]=ratio
    
    return j2mom2tf2ratio

def doSVD_err(G,M):
    err=yu.jackme(M)[-1]
    covIsq=np.diag(1/np.sqrt(err))
    def get(g,m):
        gt=covIsq@g
        u,s,vT=np.linalg.svd(gt)
        sI=np.zeros(gt.T.shape)
        np.fill_diagonal(sI,1/s)
        return vT.T@sI@(u.T)@covIsq@m
    F=np.array([get(g,m) for g,m in zip(G,M)])
    return F

funcs_ri=[np.real,np.imag]
def get_tf2ratio_SVD(ens,mom2tf2ratio,case,extra=None):
    moms=[list(mom) for mom in mom2tf2ratio.keys()]
    
    if case=='unequal':
        mpirs=[(mom,proj,insert,ri) for mom in moms for proj in projs for insert in inserts for ri in [0,1] if insert[0]!=insert[1] and yum.useQ(mom,proj,insert)[ri]]
    elif case=='equal':
        mpirs=[(mom,proj,insert,ri) for mom in moms for proj in projs for insert in inserts for ri in [0,1] if insert[0]==insert[1] and yum.useQ(mom,proj,insert)[ri]]
    elif case=='all':
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
        
        if case=='all':
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
    outfile=f'/p/project1/ngff/li47/code/projectData/05_moments/doSVD_conn/{ens_n2qpp1}.h5'
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
    
    inpath=f'{basepath}{yu.ens2full[ens]}/data_merge/'
    moms=[[int(e) for e in file[5:-3].split(',')] for file in os.listdir(inpath) \
        if file.startswith('conn') and not file.endswith('n.h5') and not file.endswith('t.h5')]
    moms=[mom for mom in moms if yu.mom2n2qpp1_sym(mom)==n2qpp1]
    
    j2mom2tf2ratio=extractRatio(ens,moms)
    
    rescale=ens2RCs_me[ens]['Zqq^s(mu!=nu)']/ens2RCs_me[ens]['Zqq^s(mu=nu)']
    
    with h5py.File(outfile,'w') as f:
        for j in js:
            for case,extra in zip(['unequal','equal','all'],[None,None,rescale]):
                tf2ratio=get_tf2ratio_SVD(ens,j2mom2tf2ratio[j],case,extra=rescale)
                if tf2ratio is None:
                    continue                
                for tf in tf2ratio.keys():
                    for i,ff in enumerate(['A20','B20','C20']):
                        f.create_dataset(f'{case}/{ff}_{j}_{tf}',data=tf2ratio[tf][:,:,i])
            
    os.remove(outfile_flag)
    print('flag_done: ' + ens_n2qpp1)

run()