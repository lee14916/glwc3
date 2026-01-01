'''
cat data_aux/cfgs_run | xargs -I @ -P 10 python3 -u post2avgmore.py -c @ > log/post2avgmore.out & 
'''
import h5py,os,re,click
import numpy as np
from itertools import permutations
from sympy.combinatorics import Permutation

projs=['P0', 'Px', 'Py', 'Pz']
inserts=['tt', 'tx', 'ty', 'tz', 'xx', 'xy', 'xz', 'yy', 'yz', 'zz']
inds_trace=[i for i,ins in enumerate(inserts) if ins[0]==ins[1]]

xyzt2xyz0=lambda x: x if x!='t' else '0'
t=[insert for insert in inserts]; inserts_key=[f'=der:g{xyzt2xyz0(insert[1])}D{xyzt2xyz0(insert[0])}:sym=' for insert in t]

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



elements=[(sx,sy,sz,xyz) for sx in [1,-1] for sy in [1,-1] for sz in [1,-1] for xyz in permutations([0, 1, 2], 3)] # Permute first Flip next
def rotate_vec3(e,vec3): #xyzt=0123
    if vec3 in ['t']:
        return (1,vec3)
    sx,sy,sz,xyz=e; ivec3={'x':0,'y':1,'z':2}[vec3]; ivec3_new=xyz[ivec3]; vec3_new=['x','y','z'][ivec3_new]
    sign=[sx,sy,sz][ivec3_new]
    return (sign,vec3_new)
def rotate_proj(e,proj):
    if proj=='P0':
        return (1,proj)
    sx,sy,sz,xyz=e; det=sx*sy*sz*(1 if Permutation(xyz).is_even else -1)
    (sign,proj_new)=rotate_vec3(e,proj[-1])    
    return (sign*det,proj_new)
def rotate_insert(e,insert):
    s1,i1=rotate_vec3(e,insert[0]); s2,i2=rotate_vec3(e,insert[1])
    return (s1*s2,i1+i2 if i1+i2 in inserts else i2+i1)
def mom2name(mom):
    assert(np.all(mom==mom3pt2standard(mom)))
    return ','.join([str(ele) for ele in mom])

id=np.eye(4)
gamma_1=gamma_x=np.array([[0.,0.,0.,1j],[0.,0.,1j,0.],[0.,-1j,0.,0.],[-1j,0.,0.,0.]])
gamma_2=gamma_y=np.array([[0.,0.,0.,1.],[0.,0.,-1.,0.],[0.,-1.,0.,0.],[1.,0.,0.,0.]])
gamma_3=gamma_z=np.array([[0.,0.,1j,0.],[0.,0.,0.,-1j],[-1j,0.,0.,0.],[0.,1j,0.,0.]])
gamma_4=gamma_t=np.array([[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,-1.,0.],[0.,0.,0.,-1.]])
gamma_5=(gamma_1@gamma_2@gamma_3@gamma_4)
#
P0=(id+gamma_4)/4; Px=1j*gamma_5@gamma_1@P0; Py=1j*gamma_5@gamma_2@P0; Pz=1j*gamma_5@gamma_3@P0
P0n=(id-gamma_4)/4; Pxn=1j*gamma_5@gamma_1@P0n; Pyn=1j*gamma_5@gamma_2@P0n; Pzn=1j*gamma_5@gamma_3@P0n
#
# coeff_{i} C_{i=4a+b} = Tr[proj@C] = proj_{ba} C_{ab} = proj_{ba} C_{4a+b} 
# coeff_{i=4a+b} = proj_{ba}
dirac2proj=np.array([[complex(ele) for row in proj.T for ele in row] for proj in [P0,Px,Py,Pz]])[:,[0,1,4,5]]
dirac2proj_bw=np.array([[complex(ele) for row in proj.T for ele in row] for proj in [P0n,Pxn,Pyn,Pzn]])[:,[10,11,14,15]]


#=============== Input ==================#
max_mom2=4
range_xyz=range(-int(np.sqrt(max_mom2))-1,int(np.sqrt(max_mom2))+2)
moms=[[x,y,z,0,0,0] for x in range_xyz for y in range_xyz for z in range_xyz if x**2+y**2+z**2<=max_mom2]       
moms=list(set([tuple(mom3pt2standard(mom)) for mom in moms]))
moms=[list(mom[:]) for mom in moms]; moms.sort()
moms_target=moms

jqs=['j+','js','jc'] # disc
stouts=range(40+1) # gluon

ens='cB211.072.64'
lat_L={'cB211.072.64':64,'cC211.060.80':80,'cD211.054.96':96,'cE211.044.112':112}[ens]
tfs={'cB211.072.64':range(2,22+1),'cC211.060.80':range(2,26+1),'cD211.054.96':range(2,30+1),'cE211.044.112':range(2,32+1)}[ens]
#========================================#

def extract2pt(paths,mom):
    moms=mom2moms(mom)
    srcs_all=[]
    data=[]; data_bw=[]
    for path in paths:
        with h5py.File(path) as f:
            moms_old=f['moms'][:]
            dic={}
            for i,m in enumerate(moms_old):
                dic[tuple(m)]=i
            inds_moms=[dic[tuple(m[:3])] for m in moms]
            srcs=list(f['data'].keys())
            srcs_all += srcs
            
            t1=np.array([f['data'][src]['N1_N1'][:]  for src in srcs]); t2=np.array([f['data'][src]['N2_N2'][:] for src in srcs])
            t1=t1[:,:,inds_moms]; t2=t2[:,:,inds_moms]
            t=(t1+t2)/2
            data.append(t)
            
            t1=np.array([f['data_bw'][src]['N1_N1'][:]  for src in srcs]); t2=np.array([f['data_bw'][src]['N2_N2'][:] for src in srcs])
            t1=t1[:,:,inds_moms]; t2=t2[:,:,inds_moms]
            t=(t1+t2)/2
            data_bw.append(t)
            
    data=np.concatenate(data,axis=0)
    data=np.einsum('pd,stmd->stmp',dirac2proj,data)
    data_bw=np.concatenate(data_bw,axis=0)
    data_bw=np.einsum('pd,stmd->stmp',dirac2proj_bw,data_bw)
    return srcs_all,data,data_bw

def extractLoop(basepath,mom):
    moms=mom2moms(mom)
    j2data={}
    
    txyz=['t','x','y','z']
    Dmus=['d3','d0','d1','d2']
    gnus=['gt','gx','gy','gz']
    
    path=f'{basepath}/j.h5'
    with h5py.File(path) as f:
        for j in jqs:
            gms=[gm.decode() for gm in f['inserts'][:]]
            moms_old=f['moms'][:]
            dic={}
            for i,m in enumerate(moms_old):
                dic[tuple(m)]=i
            inds_moms=[dic[tuple(m[3:])] for m in moms]
            
            t=np.array([[f[f'data/{j};{Dmu}'][:,:,gms.index(gnu)] for Dmu in Dmus] for gnu in gnus])
            t=t[:,:,:,inds_moms]
            t=(t+np.transpose(t,[1,0,2,3]))/2
            t=t - np.eye(4)[:,:,None,None]*np.trace(t,axis1=0,axis2=1)[None,None,:,:]/4
            t=np.transpose([t[txyz.index(m),txyz.index(n)] for m,n in inserts],[1,2,0])
            
            j2data[f'{j};disc']=t.copy()

    path=f'{basepath}/jg.h5'
    with h5py.File(path) as f:
        gms=[gm.decode() for gm in f['inserts'][:]]
        moms_old=f['moms'][:]
        dic={}
        for i,m in enumerate(moms_old):
            dic[tuple(m)]=i
        inds_moms=[dic[tuple(m[3:])] for m in moms]
        for stout in stouts:
            j=f'jg;stout{stout}'
            
            t=f[f'data/{j}'][:]
            t=t[:,inds_moms]
            j2data[j]=t.copy()
            
    return j2data

def src2ints(src):
    (sx,sy,sz,st)=re.search('sx([0-9]*)sy([0-9]*)sz([0-9]*)st([0-9]*)',src).groups()
    (sx,sy,sz,st)=(int(sx),int(sy),int(sz),int(st))
    return (sx,sy,sz,st)

def get_phase(src,mom):
    (sx,sy,sz,st)=src2ints(src)
    return np.exp(1j*(2*np.pi/lat_L)*(np.array([sx,sy,sz])@mom))

def correlate(srcs_all,dat2pt,dat2pt_bw,j2datLoop,mom):
    moms=mom2moms(mom)
    dic={}
    for i,m in enumerate(moms):
        dic[tuple(m)]=i
    inds_negmom=[dic[tuple(-np.array(m))] for m in moms]
    signs=(-1)*np.array([1,-1,-1,-1])[None,None,:,None]
    
    phases=np.array([[get_phase(src,m[3:]) for m in moms] for src in srcs_all])[:,None,:,None]
    sts=[src2ints(src)[-1] for src in srcs_all]
    
    jtf2dat3pt={}
    for j in j2datLoop.keys():
        datLoop=j2datLoop[j]
        datLoop=datLoop[None,:,:,:] * phases
        datLoop_fw=np.array([np.roll(datLoop[i],-st,axis=0) for i,st in enumerate(sts)])[:,:,:,None,:]
        datLoop_bw=np.array([np.roll(datLoop[i],-st-1,axis=0)[::-1] for i,st in enumerate(sts)])[:,:,:,None,:]
        
        for tf in tfs:
            t2pt=dat2pt[:,tf:tf+1,:,:,None]
            tj=datLoop_fw[:,:tf+1]
            t=np.mean(t2pt*tj,axis=0)
            
            t2pt=dat2pt_bw[:,-tf:-tf+1,:,:,None]
            tj=datLoop_bw[:,:tf+1]
            tbw=np.mean(t2pt*tj,axis=0)
            
            tbw=tbw[:,inds_negmom]
            tbw=tbw*signs
            t=(t+tbw)/2
            
            jtf2dat3pt[f'{j}_{tf}']=t.copy()

    return jtf2dat3pt
    
def avgmore(jtf2dat3pt,mom):
    moms=mom2moms(mom)
    dic={}
    for i,m in enumerate(moms):
        dic[tuple(m)]=i

    e2ind_mom={}; e2inds_proj={}; e2signs_proj={}; e2inds_insert={}; e2signs_insert={}
    for e in elements:
        e2ind_mom[e]=dic[tuple(rotate_mom(e,mom))]
        
        sx,sy,sz,xyz=e; signs=[sx,sy,sz,1]
        ix,iy,iz=xyz; iix,iiy,iiz=tuple([ix,iy,iz].index(i) for i in range(3))
        xyzt=['x','y','z','t']
        xyzt2={'x':xyzt[ix],'y':xyzt[iy],'z':xyzt[iz],'t':'t'}
        det=sx*sy*sz*(1 if Permutation(xyz).is_even else -1)
        
        e2signs_proj[e]=np.array([1,sx*det,sy*det,sz*det])
        e2inds_proj[e]=[0,ix+1,iy+1,iz+1]
        
        e2signs_insert[e]=np.array([signs[xyzt.index(insert[0])]*signs[xyzt.index(insert[1])] for insert in inserts])
        e2inds_insert[e]=[xyzt2[insert[0]]+xyzt2[insert[1]] for insert in inserts]
        e2inds_insert[e]=[inserts.index(ele) if ele in inserts else inserts.index(ele[1]+ele[0]) for ele in e2inds_insert[e]]
    
    jtf2dat3pt_new={}
    for key in jtf2dat3pt.keys():
        dat=jtf2dat3pt[key]
        def get(e):
            t=dat[:,e2ind_mom[e]]
            
            t=t*e2signs_proj[e][None,:,None]
            t=t[:,e2inds_proj[e]]
            
            t=t*e2signs_insert[e][None,None,:]
            t=t[:,:,e2inds_insert[e]]

            return t
        jtf2dat3pt_new[key]=np.mean([get(e) for e in elements],axis=0)[:,None,:,:]
        
    return jtf2dat3pt_new

@click.command()
@click.option('-c','--cfg')
def run(cfg):
    basepath=f'/p/project1/ngff/li47/code/projectData/02_discNJN_1D/{ens}/data_post_hold/{cfg}/'
    files=os.listdir(basepath)
    paths_2pt=[f'{basepath}/{file}' for file in files if file.startswith('N.h5')]
    
    path_avgsrc=f'/p/project1/ngff/li47/code/scratch/run/05_moments_run5/{ens}/data_avgsrc/{cfg}/'; os.makedirs(path_avgsrc,exist_ok=True)
    path_avgmore=f'/p/project1/ngff/li47/code/scratch/run/05_moments_run5/{ens}/data_avgmore/{cfg}/'; os.makedirs(path_avgmore,exist_ok=True)
    
    for mom in moms_target:
        assert(np.all(mom==mom3pt2standard(mom)))
        
        outfile_avgsrc=f'{path_avgsrc}/disc_{mom2name(mom)}.h5'
        outfile_avgmore=f'{path_avgmore}/disc_{mom2name(mom)}.h5'
        
        outfile=outfile_avgmore
        outfile_flag=outfile+'_flag'
        if os.path.isfile(outfile) and (not os.path.isfile(outfile_flag)):
            continue
        with open(outfile_flag,'w') as f:
            pass
        
        srcs_all,dat2pt,dat2pt_bw=extract2pt(paths_2pt,mom)
        j2datLoop=extractLoop(basepath,mom)
        
        jtf2dat3pt=correlate(srcs_all,dat2pt,dat2pt_bw,j2datLoop,mom)
        
        with h5py.File(outfile_avgsrc,'w') as f:
            f.create_dataset('notes',data=['time,mom,proj,insert','mom=[sink,ins]; sink+ins=src','proj=[P0,Px,Py,Pz]'])
            f.create_dataset('inserts',data=inserts)
            f.create_dataset('moms',data=mom2moms(mom))
            for jtf in jtf2dat3pt.keys():
                f.create_dataset(f'data/{jtf}',data=jtf2dat3pt[jtf])
                
        jtf2dat3pt=avgmore(jtf2dat3pt,mom)
                
        with h5py.File(outfile_avgmore,'w') as f:
            f.create_dataset('notes',data=['time,mom,proj,insert','mom=[sink,ins]; sink+ins=src','proj=[P0,Px,Py,Pz]'])
            f.create_dataset('inserts',data=inserts)
            f.create_dataset('moms',data=[mom])
            for jtf in jtf2dat3pt.keys():
                f.create_dataset(f'data/{jtf}',data=jtf2dat3pt[jtf])
                
        os.remove(outfile_flag)
    
    print('flag_cfg_done: '+cfg)
    
run()