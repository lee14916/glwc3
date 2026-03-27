import util as yu
from util import *

lat_a2s_plt=np.arange(0,0.009,0.001)

def load(path,symmetrizeQ=True):
    with h5py.File(path) as f:
        c2pt_disc=yu.jackknife(np.real(f['disc_c2pt'][:]))
        tfs_conn=[int(tf) for tf in f['conn_c2pt'].keys()]; tfs_conn.sort()
        tfs_disc=[int(tf) for tf in f['disc_gS+'].keys()]; tfs_disc.sort()
        tf2c2pt_conn={tf:yu.jackknife(np.real(f[f'conn_c2pt/{tf}'][:])) for tf in tfs_conn}

        key2tf2c3pt={}
        for key in f.keys():
            if key.startswith('conn') or key.startswith('disc'):
                if key.endswith('c2pt'):
                    continue
                cd,g=key.split('_')
                
                tf2c3pt={}
                for tf in f[key].keys():
                    tf=int(tf)
                    c3pt=yu.jackknife(np.real(f[f'{key}/{tf}'][:]))
                    
                    if symmetrizeQ:
                        c3pt=(c3pt+c3pt[:,::-1])/2
                    
                    if cd=='disc' and g.startswith('gS'):
                        vev=yu.jackknife(np.real(f[f'vev_{g}'][:]))
                        c3pt -= c2pt_disc[:,tf:tf+1] * vev[:,None]
                    
                    tf2c3pt[int(tf)]=c3pt
                
                key2tf2c3pt[f'{g};{cd}']=tf2c3pt
        
        keys=list(key2tf2c3pt.keys())
        for key in keys:
            g,cd=key.split(';')
            if cd!='conn':
                continue
            tf2c3pt={}
            for tf in tfs_conn:
                tf2c3pt[tf]=key2tf2c3pt[f'{g};conn'][tf]+key2tf2c3pt[f'{g};disc'][tf]
            key2tf2c3pt[g]=tf2c3pt

        key2tf2ratio={}
        for key in key2tf2c3pt.keys():
            tf2ratio={}
            for tf in key2tf2c3pt[key].keys():
                if 'conn' in key:
                    tf2ratio[tf] = key2tf2c3pt[key][tf]/tf2c2pt_conn[tf][:,tf:tf+1]
                elif 'disc' in key:
                    tf2ratio[tf] = key2tf2c3pt[key][tf]/c2pt_disc[:,tf:tf+1]
                else:
                    tf2ratio[tf] = key2tf2c3pt[f'{key};conn'][tf]/tf2c2pt_conn[tf][:,tf:tf+1] + key2tf2c3pt[f'{key};disc'][tf]/c2pt_disc[:,tf:tf+1]
            key2tf2ratio[key]=tf2ratio
    return [c2pt_disc,tfs_conn,tfs_disc,tf2c2pt_conn,key2tf2c3pt,key2tf2ratio]