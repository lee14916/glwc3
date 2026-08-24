'''
nohup python3 -u analysis_Nsgm_lbd.py > log/analysis_Nsgm_lbd.out & 
'''
import os,sys
parent_dir = os.path.abspath('..')
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
import util as yu
from util import *
import util_Nsgm as yu2

yu.setpath('analysis_Nsgm_lbd')

ens='b'
tfs=[8,10,12,14,16,18,20]

yunit_mul=yu.ens2amul_iso[ens]*yu.ens2aInv[ens]

[c2ptM,tf2c3ptM,c2ptCorrDic_NJN]=yu.load_pkl_reg('data',pathlabel='processData')
tf2c3ptM={tf:np.real(tf2c3ptM[tf]) for tf in tf2c3ptM.keys()}

lower=-0.1; upper=0.2
def transform(z):
    return lower + (upper - lower) / (1.0 + np.exp(-z))
def inverse_transform(p):
    if not lower < p < upper:
        raise ValueError("p must be strictly between lower and upper.")
    return np.log((p - lower) / (upper - p))

corrQ=False
def v2tf2ratio(v):
    v=transform(v)
    return {tf: np.array([(c3ptM[:,0,0] + v*(c3ptM[:,0,1]+c3ptM[:,1,0]) ) /  \
        (c2pt + v*(c2ptM[tf,0,1]+c2ptM[tf,1,0]) )[None]    \
        for c3ptM,c2ptM,c2pt in zip(tf2c3ptM[tf],c2ptM,c2ptCorrDic_NJN[tf])]) \
        for tf in tfs}

tfmins=[8,10,12,14,16]
tcmins=range(1,6+1)
tfmin2tcmins={tfmin:tcmins for tfmin in tfmins}
tfmin2tcmins[16]=[1,2,3,4]
fits=yu.doFits_3pt_lbd(v2tf2ratio,tfmins,tcmins,tfmin2tcmins=tfmin2tcmins,symmetrizeQ=True,pars0=[20,inverse_transform(0.1)],label=f'lbd_{corrQ}',verbose=3,corrQ=corrQ)