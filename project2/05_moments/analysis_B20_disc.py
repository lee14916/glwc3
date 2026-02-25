'''
cat data_aux/dat_ignore/analysis_B20_disc_run | xargs -I @ -P 3 python3 -u analysis_B20_disc.py -t @ > log/analysis_B20_disc.out & 
'''
import util as yu
from util import *
import util_moments as yum
import click

yu.setpath('analysis_B20_2')

def encodeTask(task):
    n2qpp1,ff,j=task
    n2q,n2p,n2p1=n2qpp1
    return f"{n2q},{n2p},{n2p1}_{ff}_{j.replace(';',',')}"
def decodeTask(task):
    n2qpp1,ff,j=task.split('_')
    j=j.replace(',',';')
    n2qpp1=tuple([int(ele) for ele in n2qpp1.split(',')])
    return (n2qpp1,ff,j)


enss_all=['b','c','d']

#====================
overwrite=True

c1s=['unequal']; c2s=['err']
cases_todo=['_'.join([c1,c2]) for c1,c2 in product(c1s,c2s)]
@click.command()
@click.option('-t','--task')
def run(task):
    res={}
    n2qpp1,ff,j=decodeTask(task)
    
    enss=enss_all
    n2q,n2p,n2p1=n2qpp1
    figs=[]
    for ens in enss:
        case2tf2ratio={}
        path=f'/p/project1/ngff/li47/code/scratch/run/05_moments_run5/doSVD/disc_{ens}_{yu.n2qpp12str(n2qpp1)}.h5'
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
            gett=lambda t:round(t/yu.ens2a[ens])
            def get_tfs(tmin,tmax,dt=1):
                return range(gett(tmin),gett(tmax),dt)
            def cutExtraDiff2tcmins(tmaxExtra,tmaxDiff):
                return yu.cutExtraDiff2tcmins(range(0,gett(tmaxExtra)),range(-gett(tmaxDiff),gett(tmaxDiff)+1),cutBase=2)

            # fit
            tfmins_1st=get_tfs(0.3,1.25)
            tcmins_1st=cutExtraDiff2tcmins(1,0.4)
            tfmins_2st_sum=get_tfs(0.2,1.0)
            tcmins_2st_sum=[(2,2)]
            # WAMA & display
            tf_max=gett(1.3)
            tfmin_max=gett(1.2); tcmin_max=gett(0.6)
            
            tf2ratio=case2tf2ratio[case]
            tf2ratio=yu.cut_tf2ratio(tf2ratio,gett(1.3))
            
            fits_band=yu.doFits_3pt_band(tf2ratio,tcmins_1st,corrQ=False,unicutQ=True,label=f'{n2qpp1}_{ff}_{j}_{ens}_{case}_band',overwrite=overwrite)
            fit_band_WA=yu.doWA_band(fits_band,tf_min=gett(0.9),tf_max=tf_max,tcmin=gett(0.2)*2,corrQ=False)
            fits_const=yu.doFits_3pt('const',tf2ratio,tfmins_1st,tcmins_1st,unicutQ=True,label=f'{n2qpp1}_{ff}_{j}_{ens}_{case}_const',overwrite=overwrite)
            tfmin=gett(0.6); tcmin=gett(0.3)
            tcmin=[tc for (tf,tc),*_ in fits_const if tf==tfmin and tc[0]+tc[1]==2*tcmin][0]
            fit_const_MA=yu.doMA_3pt(fits_const,fitlabels=(tfmin,tcmin))
            fits_sum=yu.doFits_3pt('sum',tf2ratio,tfmins_2st_sum,tcmins_2st_sum,unicutQ=2,label=f'{n2qpp1}_{ff}_{j}_{ens}_{case}_sum',overwrite=overwrite)
            tfmin=gett(0.35); tcmin=(2,2)
            fit_sum_MA=yu.doMA_3pt(fits_sum,fitlabels=(tfmin,tcmin))
            
            res[(case,'bandfit_WA',ens)]=fit_band_WA[0][:,0]
            res[(case,'const_MA',ens)]=fit_const_MA[0][:,0]
            res[(case,'sum_MA',ens)]=fit_sum_MA[0][:,0]
            
            dic={
                'base:[tf2ratio,fits_band,fits_const,fits_sum,fits_2st]':[tf2ratio,fits_band,fits_const,fits_sum,None],
                'WAMA:[fit_band_WA,fit_const_MA,fit_sum_MA,fit_2st_MA]':[fit_band_WA,fit_const_MA,fit_sum_MA,None],
                'rainbow:[tfmin,tfmax,tcmin,dt]':[None,gett(1.2),2,2],
                'fit_band:[tfmin,tfmax,tcmin_min,tcmin_max,dtf,dtc]':[None,tf_max,None,tcmin_max,None,None],
                'fit_const:[tfmin_min,tfmin_max,tcmin_min,tcmin_max,dtf,dtc]':[None,tfmin_max,None,tcmin_max,None,None],
                'fit_sum:[tfmin_min,tfmin_max,tcmin_min,tcmin_max,dtf,dtc]':[None,tfmin_max,None,tcmin_max,None,None],
                'xunit':yu.ens2a[ens]
            }
            return dic
        
        def createDic2(ind,case):
            # print(ens,case)
            gett=lambda t:round(t/yu.ens2a[ens])
            def get_tfs(tmin,tmax,dt=1):
                return range(gett(tmin),gett(tmax),dt)
            def cutExtraDiff2tcmins(tmaxExtra,tmaxDiff):
                return yu.cutExtraDiff2tcmins(range(0,gett(tmaxExtra)),range(-gett(tmaxDiff),gett(tmaxDiff)+1),cutBase=2)

            # fit
            tfmins_1st=get_tfs(0.3,1.25)
            tcmins_1st=cutExtraDiff2tcmins(1,0.4)
            tfmins_2st_sum=get_tfs(0.2,1.0)
            tcmins_2st_sum=[(2,2)]
            # WAMA & display
            tf_max=gett(1.3)
            tfmin_max=gett(1.2); tcmin_max=gett(0.6)
            
            tf2ratio=case2tf2ratio[case]
            tf2ratio=yu.cut_tf2ratio(tf2ratio,gett(1.3))
            
            fits_band=yu.doFits_3pt_band(tf2ratio,tcmins_1st,corrQ=False,unicutQ=True,label=f'{n2qpp1}_{ff}_{j}_{ens}_{case}_band',overwrite=overwrite)
            fit_band_WA=yu.doWA_band(fits_band,tf_min=gett(0.8),tf_max=tf_max,tcmin=gett(0.2)*2,corrQ=False)
            fits_const=yu.doFits_3pt('const',tf2ratio,tfmins_1st,tcmins_1st,unicutQ=True,label=f'{n2qpp1}_{ff}_{j}_{ens}_{case}_const',overwrite=overwrite)
            tfmin=gett(0.6); tcmin=gett(0.3)
            tcmin=[tc for (tf,tc),*_ in fits_const if tf==tfmin and tc[0]+tc[1]==2*tcmin][0]
            fit_const_MA=yu.doMA_3pt(fits_const,fitlabels=(tfmin,tcmin))
            fits_sum=yu.doFits_3pt('sum',tf2ratio,tfmins_2st_sum,tcmins_2st_sum,unicutQ=2,label=f'{n2qpp1}_{ff}_{j}_{ens}_{case}_sum',overwrite=overwrite)
            tfmin=gett(0.35); tcmin=(2,2)
            fit_sum_MA=yu.doMA_3pt(fits_sum,fitlabels=(tfmin,tcmin))
            
            res[(case,'bandfit_WA',ens)]=fit_band_WA[0][:,0]
            res[(case,'const_MA',ens)]=fit_const_MA[0][:,0]
            res[(case,'sum_MA',ens)]=fit_sum_MA[0][:,0]

            dic={
                'base:[tf2ratio,fits_band,fits_const,fits_sum,fits_2st]':[tf2ratio,fits_band,fits_const,fits_sum,None],
                'WAMA:[fit_band_WA,fit_const_MA,fit_sum_MA,fit_2st_MA]':[fit_band_WA,fit_const_MA,fit_sum_MA,None],
                'rainbow:[tfmin,tfmax,tcmin,dt]':[None,gett(1.2),2,2],
                'fit_band:[tfmin,tfmax,tcmin_min,tcmin_max,dtf,dtc]':[None,tf_max,None,tcmin_max,None,None],
                'fit_const:[tfmin_min,tfmin_max,tcmin_min,tcmin_max,dtf,dtc]':[None,tfmin_max,None,tcmin_max,None,None],
                'fit_sum:[tfmin_min,tfmin_max,tcmin_min,tcmin_max,dtf,dtc]':[None,tfmin_max,None,tcmin_max,None,None],
                'xunit':yu.ens2a[ens]
            }
            return dic
          
        list_dic=[createDic(ind,case) if not j.startswith('jg') else createDic2(ind,case) for ind,case in enumerate(cases_do)]

        fig,axs=yu.makePlot_3pt(list_dic,shows=['rainbow','fit_band','fit_const','fit_sum'])
        fig.suptitle(rf'{yu.ens2label[ens]}; n2qpp1={n2qpp1}; $Q^2$={yum.n2qpp12Q2(n2qpp1,ens):.4f} GeV$^2$')
        
        for i in range(len(axs)):
            axs[i,0].set_ylabel(cases_do[i])
        yu.finalizePlot(closeQ=True)
        figs.append(fig)

    yu.makePDF(f'{j}/{yu.n2qpp12str(n2qpp1)}_{ff}',figs,mkdirQ=True)
    
    print('flag_done: ' + task)
    return res

run()