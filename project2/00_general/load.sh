path=/p/arch1/ecy00/ecy002/hadstruct/Corr/cB211.072.64/ave/hdf5/h5out/

echo -n Nfiles= > log/load.out
ls ${path}* | wc -l >> log/load.out
ls ${path}* | xargs -I @ -P 10 head -c1 @ >> log/load.out &
