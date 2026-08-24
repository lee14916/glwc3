import util as yu
from util import *

# ChPT
if True:
    diff_LO={'b':(-4.30,0.27),'c':(-1.25,0.32),'d':(-5.10,0.32),'e':(-1.18,0.30)}
    diff_NLO={'b':(-3.6,0.4),'c':(-1.1,0.5),'d':(-5.10,0.32),'e':(-1.18,0.30)}
    diff_NNLO={'b':(-3.5,0.8),'c':(-1.25,0.32),'d':(-5.10,0.32),'e':(-1.18,0.30)}


#!============== Load (old) ==============#
if True:
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
            Ncfg=len(cfgs); Njk=len(yu.jackknife(np.zeros(Ncfg),d=d,nmin=nmin))
            
            datasets=[]
            def visit_function(name,node):
                if isinstance(node, h5py.Dataset):
                    datasets.append(name)
                    # print(len(datasets),name,end='\r')
            f.visititems(visit_function)
                
            N=len(datasets)
            for i,dataset in enumerate(datasets):
                if 'data' in dataset:
                    data_load[dataset]=yu.jackknife(f[dataset][()],d=d,nmin=nmin)
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

if True:
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