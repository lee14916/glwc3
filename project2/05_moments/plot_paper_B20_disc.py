'''
cat data_aux/dat_ignore/analysis_B20_disc_run | xargs -I @ -P 3 python3 -u plot_paper_B20_disc.py -t @ > log/plot_paper_B20_disc.out & 
python3 -u plot_paper_B20_disc.py -t 2,2,0_B20_j+,disc 
python3 -u plot_paper_B20_disc.py -t 2,2,0_B20_j+,disc & python3 -u plot_paper_B20_disc.py -t 2,2,0_B20_js,disc & python3 -u plot_paper_B20_disc.py -t 2,2,0_B20_jc,disc & python3 -u plot_paper_B20_disc.py -t 2,2,0_B20_jg,stout10 & 
'''
import util as yu
from util import *
import util_moments as yum
import click

mpl.rcParams['lines.markersize'] = mpl.rcParams['errorbar.capsize'] = 13
mpl.rcParams['axes.labelsize'] = mpl.rcParams['axes.titlesize'] = mpl.rcParams['xtick.labelsize'] = mpl.rcParams['ytick.labelsize'] = 40
mpl.rcParams['font.size'] = 20
yu.mpl_global_elinewidth=yu.mpl_global_capthick=3

yu.setpath('analysis_B20_2')

tftcphy_A20_conn=tftcphy_B20_conn=(0.6,0.2)
tftcphy_A20_discq=tftcphy_A20_gluon=tftcphy_B20_discq=tftcphy_B20_gluon=(0.7,0.3)

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

#====================
overwrite=False

c1s=['unequal']; c2s=['err']
cases_todo=['_'.join([c1,c2]) for c1,c2 in product(c1s,c2s)]
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
        path=f'/p/project1/ngff/li47/code/scratch/run/05_moments_run5_1DV/doSVD/disc_{ens}_{yu.n2qpp12str(n2qpp1)}.h5'
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
        if len(cases_do)==0:
            print('flag_skip: ' + task)
            return None
        eqs=[{'all':'=','unequal':'!=','equal':'='}[case.split('_')[0]] for case in cases_do]
        
        # print(f'doing {task} {ens}')
        
        def createDic(ind,case):
            rainbow_tfphy_min=0.5
            rainbow_tfphy_max=1.1
            sum_tfphy_max=0.7
            dt=2
            gett=lambda t:round(t/yu.ens2a[ens])
            def get_tfs(tmin,tmax,dt=1):
                return range(gett(tmin),gett(tmax),dt)
            def cutExtraDiff2tcmins(tmaxExtra,tmaxDiff):
                return yu.cutExtraDiff2tcmins(range(0,gett(tmaxExtra)),range(-gett(tmaxDiff),gett(tmaxDiff)+1),cutBase=2)

            # fit
            tfmins_1st=get_tfs(0.3,1.25)
            tcmins_1st=cutExtraDiff2tcmins(1,0.4)
            tfmins_2st_sum=get_tfs(0.2,1.25)
            tcmins_2st_sum=[(2,2)]
            # WAMA & display
            tf_max=gett(1.3)
            tfmin_max=gett(1.2); tcmin_max=gett(0.6)
            
            tf2ratio=case2tf2ratio[case]
            tf2ratio=yu.cut_tf2ratio(tf2ratio,gett(1.3))
            
            fits_const=yu.doFits_3pt('const',tf2ratio,tfmins_1st,tcmins_1st,unicutQ=True,label=f'{n2qpp1}_{ff}_{j}_{ens}_{case}_const',overwrite=overwrite)
            tfphy,tcphy=tftcphy_B20_discq
            ind=np.argmin([np.abs(tfmin*yu.ens2a[ens] - tfphy) + np.abs((tcmin[0]+tcmin[1])/2*yu.ens2a[ens] - tcphy) for (tfmin,tcmin),*_ in fits_const])
            fit_const_MA=yu.doMA_3pt(fits_const[ind:ind+1])
            tcmin=fits_const[ind][0][1]
            fits_const=[fit for fit in fits_const if sum(fit[0][1])==sum(tcmin)]
            
            fits_band=yu.doFits_3pt_band(tf2ratio,tcmins_1st,corrQ=False,unicutQ=True,label=f'{n2qpp1}_{ff}_{j}_{ens}_{case}_band',overwrite=True)
            # fit_band_WA=yu.doWA_band(fits_band,tf_min=gett(0.9),tf_max=tf_max,tcmin=gett(0.2)*2,corrQ=False)
            fits_band=[fit for fit in fits_band if sum(fit[0][1])==sum(tcmin)]
            
            fits_sum=yu.doFits_3pt('sum',tf2ratio,tfmins_2st_sum,tcmins_2st_sum,label=f'{n2qpp1}_{ff}_{j}_{ens}_{case}_sum',overwrite=overwrite)
            ind=np.argmin([np.abs(tfmin*yu.ens2a[ens] - tfphy) + np.abs((tcmin[0]+tcmin[1])/2*yu.ens2a[ens] - tcphy) for (tfmin,tcmin),*_ in fits_sum])
            fit_sum_MA=yu.doMA_3pt(fits_sum[ind:ind+1])
            
            # res[(case,'bandfit_WA',ens)]=fit_band_WA[0][:,0]
            res[(case,'const_MA',ens)]=fit_const_MA[0][:,0]
            res[(case,'sum_MA',ens)]=fit_sum_MA[0][:,0]
            
            dic={
                'base:[tf2ratio,fits_band,fits_const,fits_sum,fits_2st]':[tf2ratio,fits_band,fits_const,fits_sum,None],
                'WAMA:[fit_band_WA,fit_const_MA,fit_sum_MA,fit_2st_MA]':[None,fit_const_MA,fit_sum_MA,None],
                'rainbow:[tfmin,tfmax,tcmin,dt]':[gett(rainbow_tfphy_min),gett(rainbow_tfphy_max),2,dt],
                'fit_band:[tfmin,tfmax,tcmin_min,tcmin_max,dtf,dtc]':[gett(rainbow_tfphy_min),gett(rainbow_tfphy_max),None,None,dt,None],
                'fit_const:[tfmin_min,tfmin_max,tcmin_min,tcmin_max,dtf,dtc]':[None,tfmin_max,None,tcmin_max,None,None],
                'fit_sum:[tfmin_min,tfmin_max,tcmin_min,tcmin_max,dtf,dtc]':[None,tfmin_max,None,tcmin_max,None,None],
                'xunit':yu.ens2a[ens]
            }
            dic_sum={
                'base:[tf2ratio,fits_band,fits_const,fits_sum,fits_2st]':[None,None,None,fits_sum,None],
                'fit_sum:[tfmin_min,tfmin_max,tcmin_min,tcmin_max,dtf,dtc]':[gett(rainbow_tfphy_min),gett(sum_tfphy_max),None,None,None,None],
                'xunit':yu.ens2a[ens],
                'shift:[rainbow,midpoint,fit]':[0,0,-0.2],
            }
            return dic,dic_sum
        
        def createDic2(ind,case):
            rainbow_tfphy_min=0.5
            rainbow_tfphy_max=1.1       
            sum_tfphy_max=0.7
            dt=2
            # print(ens,case)
            gett=lambda t:round(t/yu.ens2a[ens])
            def get_tfs(tmin,tmax,dt=1):
                return range(gett(tmin),gett(tmax),dt)
            def cutExtraDiff2tcmins(tmaxExtra,tmaxDiff):
                return yu.cutExtraDiff2tcmins(range(0,gett(tmaxExtra)),range(-gett(tmaxDiff),gett(tmaxDiff)+1),cutBase=2)

            # fit
            tfmins_1st=get_tfs(0.3,1.25)
            tcmins_1st=cutExtraDiff2tcmins(1,0.4)
            tfmins_2st_sum=get_tfs(0.2,1.25)
            tcmins_2st_sum=[(2,2)]
            # WAMA & display
            tf_max=gett(1.3)
            tfmin_max=gett(1.2); tcmin_max=gett(0.6)
            
            tf2ratio=case2tf2ratio[case]
            tf2ratio=yu.cut_tf2ratio(tf2ratio,gett(1.3))
            
            fits_const=yu.doFits_3pt('const',tf2ratio,tfmins_1st,tcmins_1st,unicutQ=True,label=f'{n2qpp1}_{ff}_{j}_{ens}_{case}_const',overwrite=overwrite)
            tfphy,tcphy=tftcphy_B20_gluon
            ind=np.argmin([np.abs(tfmin*yu.ens2a[ens] - tfphy) + np.abs((tcmin[0]+tcmin[1])/2*yu.ens2a[ens] - tcphy) for (tfmin,tcmin),*_ in fits_const])
            fit_const_MA=yu.doMA_3pt(fits_const[ind:ind+1])
            tcmin=fits_const[ind][0][1]
            fits_const=[fit for fit in fits_const if sum(fit[0][1])==sum(tcmin)]
            
            fits_band=yu.doFits_3pt_band(tf2ratio,tcmins_1st,corrQ=False,unicutQ=True,label=f'{n2qpp1}_{ff}_{j}_{ens}_{case}_band',overwrite=True)
            # fit_band_WA=yu.doWA_band(fits_band,tf_min=gett(0.8),tf_max=tf_max,tcmin=gett(0.2)*2,corrQ=False)
            fits_band=[fit for fit in fits_band if sum(fit[0][1])==sum(tcmin)]
            
            fits_sum=yu.doFits_3pt('sum',tf2ratio,tfmins_2st_sum,tcmins_2st_sum,label=f'{n2qpp1}_{ff}_{j}_{ens}_{case}_sum',overwrite=overwrite)
            ind=np.argmin([np.abs(tfmin*yu.ens2a[ens] - tfphy) + np.abs((tcmin[0]+tcmin[1])/2*yu.ens2a[ens] - tcphy) for (tfmin,tcmin),*_ in fits_sum])
            fit_sum_MA=yu.doMA_3pt(fits_sum[ind:ind+1])
            
            # res[(case,'bandfit_WA',ens)]=fit_band_WA[0][:,0]
            res[(case,'const_MA',ens)]=fit_const_MA[0][:,0]
            res[(case,'sum_MA',ens)]=fit_sum_MA[0][:,0]

            dic={
                'base:[tf2ratio,fits_band,fits_const,fits_sum,fits_2st]':[tf2ratio,fits_band,fits_const,fits_sum,None],
                'WAMA:[fit_band_WA,fit_const_MA,fit_sum_MA,fit_2st_MA]':[None,fit_const_MA,fit_sum_MA,None],
                'rainbow:[tfmin,tfmax,tcmin,dt]':[gett(rainbow_tfphy_min),gett(rainbow_tfphy_max),2,dt],
                'fit_band:[tfmin,tfmax,tcmin_min,tcmin_max,dtf,dtc]':[gett(rainbow_tfphy_min),gett(rainbow_tfphy_max),None,None,dt,None],
                'fit_const:[tfmin_min,tfmin_max,tcmin_min,tcmin_max,dtf,dtc]':[None,tfmin_max,None,tcmin_max,None,None],
                'fit_sum:[tfmin_min,tfmin_max,tcmin_min,tcmin_max,dtf,dtc]':[None,tfmin_max,None,tcmin_max,None,None],
                'xunit':yu.ens2a[ens]
            }
            dic_sum={
                'base:[tf2ratio,fits_band,fits_const,fits_sum,fits_2st]':[None,None,None,fits_sum,None],
                'fit_sum:[tfmin_min,tfmin_max,tcmin_min,tcmin_max,dtf,dtc]':[gett(rainbow_tfphy_min),gett(sum_tfphy_max),None,None,None,None],
                'xunit':yu.ens2a[ens],
                'shift:[rainbow,midpoint,fit]':[0,0,-0.2],
            }
            return dic,dic_sum
          
        list_dic+=[createDic(ind,case) if not j.startswith('jg') else createDic2(ind,case) for ind,case in enumerate(cases_do)]
        
    t=list_dic
    list_dic=[ele[0] for ele in t]
    list_dic_sum=[ele[1] for ele in t]

    fig,axs=yu.makePlot_3pt(list_dic,shows=['rainbow','fit_band','fit_const'],noLegendQ=True,colHeaders=None,fullband='fit_const')
    jstr=j[1]
    jstr='u+d' if jstr=='+' else jstr
    for i in range(len(enss)):
        axs[i,0].set_ylabel(rf'$\tilde{{B}}_{{20}}^{{{jstr}}}(Q_2^2)$')
        axs[i,0].set_ylim(axs[i,0].get_ylim())
    fig,axs=yu.makePlot_3pt(list_dic_sum,shows=['rainbow','rainbow','fit_sum'],figAxs=(fig,axs),colors_fit=['g'],fmts_fit=['o'],colHeaders=None)
    axs[-1,0].set_xticks([-0.3,0,0.3])
    axs[-1,1].set_xticks([0.7,1.0])
    axs[-1,2].set_xticks([0.7,1.0])
    # fig.suptitle(rf'{yu.ens2label[ens]}; n2qpp1={n2qpp1}; $Q^2$={yum.n2qpp12Q2(n2qpp1,ens):.4f} GeV$^2$')
    
    # axs[-1,0].set_xlim([-0.55,0.55])
    
    yu.setpath('plot_paper')
    if j in ['j-;conn']:
        yu.finalizePlot(f'rainbow_B20/{j}_{yu.n2qpp12str(n2qpp1)}_{ff}_tilde',mkdirQ=True,closeQ=True)
    else:
        yu.finalizePlot(f'rainbow_B20/{j}_{yu.n2qpp12str(n2qpp1)}_{ff}_tilde',mkdirQ=True,closeQ=True)
    yu.setpath('analysis_B20_2')

    print('flag_done: ' + task)
    return res

run()