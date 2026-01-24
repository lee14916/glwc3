case=Nrun2
path_pre=/p/project1/ngff/li47/code/projectData/05_moments/cE211.044.112/data_pre/${case}_127_arch/

echo -n Nfiles= > log/load_${case}.out
ls ${path_pre}* | wc -l >> log/load_${case}.out
ls ${path_pre}* | xargs -I @ -P 10 head -c1 @ >> log/load_${case}.out &