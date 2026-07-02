'''
cat data_aux/dat_ignore/analysis_B20_conn_run | xargs -I @ -P 10 python3 -u plot_paper_B20_conn.py -t @ > log/plot_paper_B20_conn.out & 
python3 -u plot_paper_B20_conn.py -t 1,1,0_B20_j-,conn & python3 -u plot_paper_B20_conn.py -t 2,2,0_B20_j+,conn & 
'''
import util as yu
from util import *
import util_moments as yum
import click

mpl.rcParams['lines.markersize'] = mpl.rcParams['errorbar.capsize'] = 13
mpl.rcParams['axes.labelsize'] = mpl.rcParams['axes.titlesize'] = mpl.rcParams['xtick.labelsize'] = mpl.rcParams['ytick.labelsize'] = 40
mpl.rcParams['font.size'] = 20
yu.mpl_global_elinewidth=yu.mpl_global_capthick=3

yu.setpath('analysis_B20')

tftcphy_A20_conn=tftcphy_B20_conn=(0.8,0.3)
tftcphy_A20_discq=tftcphy_B20_discq=(0.6,0.2)
tftcphy_A20_gluon=tftcphy_B20_gluon=(0.6,0.3)

def encodeTask(task):
    n2qpp1,ff,j=task
    n2q,n2p,n2p1=n2qpp1
    return f"{n2q},{n2p},{n2p1}_{ff}_{j.replace(';',',')}"
def decodeTask(task):
    n2qpp1,ff,j=task.split('_')
    j=j.replace(',',';')
    n2qpp1=tuple([int(ele) for ele in n2qpp1.split(',')])
    return (n2qpp1,ff,j)

enss_all=['b','c','d','e']

ens2msq2pars_jk=yu.load_pkl_reg('ens2msq2pars_jk',pathlabel='analysis_c2pt')

#====================
overwrite=False

enss=enss_all
ens2Njk={'b':725,'c':400,'d':493,'e':516}
path='data_aux/RCs.pkl'
with open(path,'rb') as f:
    ens2RCs_me=pickle.load(f)
ens2RCs={ens:{} for ens in enss}
for ens in enss:
    for key in ens2RCs_me[ens]:
        if key.endswith('err'):
            continue
        ens2RCs[ens][key]=yu.jackknife_pseudo(ens2RCs_me[ens][key],ens2RCs_me[ens][f'{key}_err']*0+1e-10,ens2Njk[ens])[:,0]
        
path='data_aux/RCs_np.pkl'
with open(path,'rb') as f:
    ens2RCs_np_me=pickle.load(f)
ens2RCs_np={ens:{} for ens in enss}
for ens in enss:
    for key in ens2RCs_np_me[ens]:
        if key.endswith('err'):
            continue
        ens2RCs_np[ens][key]=yu.jackknife_pseudo(ens2RCs_np_me[ens][key],ens2RCs_np_me[ens][f'{key}_err']+1e-10,ens2Njk[ens])[:,0]


@click.command()
@click.option('-t','--task')
def run(task):
    res={}
    n2qpp1,ff,j=decodeTask(task)
    
    if j in ['j-;conn']:
        c1s=['all']
    else:
        c1s=['unequal']
    c2s=['err']
    cases_todo=['_'.join([c1,c2]) for c1,c2 in product(c1s,c2s)]
    
    enss=enss_all
    n2q,n2p,n2p1=n2qpp1
    list_dic=[]
    for ens in enss:
        case2tf2ratio={}
        path=f'/p/project1/ngff/li47/code/projectData/05_moments/doSVD/conn_{ens}_{yu.n2qpp12str(n2qpp1)}.h5'
        
        yunit=ens2RCs_np_me[ens]['Zqq(mu=nu)']/ens2RCs_np_me[ens]['Zqq(mu!=nu)'] if j in ['j-;conn'] else 1
        
        tfs_conn=set()
        with h5py.File(path) as f:
            for case in f.keys():
                if case not in cases_todo:
                    continue
                if case not in case2tf2ratio.keys():
                    case2tf2ratio[case]={}
                for key in f[case].keys():
                    tff,tj,tf=key.split('_'); tf=int(tf); tfs_conn.add(tf)
                    if (tff,tj) != (ff,j):
                        continue
                    case2tf2ratio[case][tf]=f[case][key][:]
        tfs_conn=list(tfs_conn); tfs_conn.sort()
                    
        cases_do=['_'.join([c1,c2]) for c1,c2 in product(c1s,c2s) if f'{c1}_{c2}' in case2tf2ratio and len(case2tf2ratio[f'{c1}_{c2}'])!=0]
        eqs=[{'all':'=','unequal':'!=','equal':'='}[case.split('_')[0]] for case in cases_do]
        
        def createDic(ind,case):
            # print(ens,case)
            gett=lambda t:round(t/yu.ens2a[ens])
            def get_tfs(tmin,tmax,dt=1):
                return range(gett(tmin),gett(tmax),dt)
            def cutExtraDiff2tcmins(tmaxExtra,tmaxDiff):
                return yu.cutExtraDiff2tcmins(range(0,gett(tmaxExtra)),range(-gett(tmaxDiff),gett(tmaxDiff)+1),cutBase=2)

            tfmins_1st=tfs_conn
            tcmins_1st=cutExtraDiff2tcmins(0.8,0.4)
            
            tfmins_2st=tfs_conn[:-2]
            tcmins_2st=cutExtraDiff2tcmins(0.6,0.2)
            
            tfmins_2st_sum=tfmins_2st
            tcmins_2st_sum=[(2,2)]
            
            fittype='2st2step_EFITshare'
            pars_jk_meff2st=[ens2msq2pars_jk[ens][n2p1],ens2msq2pars_jk[ens][n2p]]
                            
            tf2ratio=case2tf2ratio[case]
            fits_band=yu.doFits_3pt_band(tf2ratio,tcmins_1st,corrQ=False,unicutQ=True,label=f'{n2qpp1}_{ff}_{j}_{ens}_{case}_band',overwrite=overwrite)
            fit_band_WA=yu.doWA_band(fits_band,tf_min=gett(0.9),tcmin=gett(0.2)*2,corrQ=False)
            fits_const=yu.doFits_3pt('const',tf2ratio,tfmins_1st,tcmins_1st,unicutQ=True,label=f'{n2qpp1}_{ff}_{j}_{ens}_{case}_const',overwrite=overwrite)
            fit_const_MA=yu.doMA_3pt(fits_const,tfmin_min=gett(0.9),tcmin_min=gett(0.2)*2)
            fits_sum=yu.doFits_3pt('sum',tf2ratio,tfmins_2st_sum,tcmins_2st_sum,label=f'{n2qpp1}_{ff}_{j}_{ens}_{case}_sum',overwrite=True)
            fits_sum=[fit for fit in fits_sum if fit[0][1]==(2,2)]
            # fit_sum_MA=yu.doMA_3pt(fits_sum,tcmin_min=gett(0.2)*2)
            fits_2st=yu.doFits_3pt(fittype,tf2ratio,tfmins_2st,tcmins_2st,pars_jk_meff2st=pars_jk_meff2st,unicutQ=True,label=f'{n2qpp1}_{ff}_{j}_{ens}_{case}_2st',overwrite=overwrite)
            tfphy,tcphy=tftcphy_B20_conn
            ind=np.argmin([np.abs(tfmin*yu.ens2a[ens] - tfphy) + np.abs((tcmin[0]+tcmin[1])/2*yu.ens2a[ens] - tcphy) for (tfmin,tcmin),*_ in fits_2st])
            fit_2st_MA=yu.doMA_3pt(fits_2st[ind:ind+1])
            fits_2st=[fit for fit in fits_2st if sum(fit[0][1])==sum(fits_2st[ind][0][1])]
            
            res[(case,'bandfit_WA',ens)]=fit_band_WA[0][:,0]
            res[(case,'const_MA',ens)]=fit_const_MA[0][:,0]
            # res[(case,'sum_MA',ens)]=fit_sum_MA[0][:,0]
            res[(case,'2st_MA',ens)]=fit_2st_MA[0][:,0]
                  
            dic={
                'base:[tf2ratio,fits_band,fits_const,fits_sum,fits_2st]':[tf2ratio,fits_band,fits_const,fits_sum,fits_2st],
                'WAMA:[fit_band_WA,fit_const_MA,fit_sum_MA,fit_2st_MA]':[None,None,None,fit_2st_MA],
                'rainbow:[tfmin,tfmax,tcmin,dt]':[None,None,2,None],
                'fit_2st_rainbow_midpoint:[fittype,pars_jk_meff2st]':[fittype,pars_jk_meff2st],
                'xunit':yu.ens2a[ens],
                'yunit':yunit,
            }
            dic_sum={
                'base:[tf2ratio,fits_band,fits_const,fits_sum,fits_2st]':[None,None,None,fits_sum,None],
                'fit_sum:[tfmin_min,tfmin_max,tcmin_min,tcmin_max,dtf,dtc]':[None,None,None,None,None,None],
                'xunit':yu.ens2a[ens],'yunit':yunit,
                'shift:[rainbow,midpoint,fit]':[0,0,-0.2],
            }
            return dic,dic_sum
        list_dic+=[createDic(ind,case) for ind,case in enumerate(cases_do)]
        
    t=list_dic
    list_dic=[ele[0] for ele in t]
    list_dic_sum=[ele[1] for ele in t]

    fig,axs=yu.makePlot_3pt(list_dic,shows=['rainbow','midpoint','fit_2st'],sharey=True,noLegendQ=True,colHeaders=None,fullband='fit_2st',oddmidQ=True)
    fig,axs=yu.makePlot_3pt(list_dic_sum,shows=['rainbow','rainbow','fit_sum'],figAxs=(fig,axs),colors_fit=['g'],fmts_fit=['o'],colHeaders=None)
    
    for i in range(len(enss)):
        if j in ['j-;conn']:
            axs[i,0].set_ylabel(r'${B}_{20}^{u-d}(Q_1^2)$')
        else:
            axs[i,0].set_ylabel(r'${B}_{20}^{u+d}(Q_2^2)$')
        
    # axs[-1,2].set_xlim([0.35,1.45])
    # axs[-1,2].set_xticks(np.arange(0.4,1.5,0.2))
    # fig.suptitle(rf'{yu.ens2label[ens]}; n2qpp1={n2qpp1}; $Q^2$={yum.n2qpp12Q2(n2qpp1,ens):.4f} GeV$^2$')
    
    yu.setpath('plot_paper')
    yu.finalizePlot(f'rainbow_B20/{j}_{yu.n2qpp12str(n2qpp1)}_{ff}',mkdirQ=True,closeQ=True)
    yu.setpath('analysis_B20')
    
    print('flag_done: ' + task)
    return res

run()