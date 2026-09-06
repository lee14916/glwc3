for cfg in `cat /capstor/store/cscs/userlab/lp139/lyan/code/glwc3/project2/01_Nsgm/dataPrepare/cB211.072.64_base/data_aux/cfgs_run`
do
    echo ${cfg}
    
    inpath=/capstor/store/cscs/userlab/lp139/lyan/code/projectData_old/01_Nsgm/cB211.072.64_base/data_post/${cfg}/
    outpath=/capstor/store/cscs/userlab/lp139/lyan/code/projectData_old/01_Nsgm/cB211.072.64_base_strange_charm/data_post/${cfg}/
    mkdir -p ${outpath}
    for file in `ls ${inpath}`
    do
        echo ${file}
        ln -s ${inpath}${file} ${outpath}${file}
    done

    # break
done