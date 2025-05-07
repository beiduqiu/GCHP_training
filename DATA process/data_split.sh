#python3 new.py --input_dirs /storage1/guerin/Active/geos-chem/ZifanRunDir/Run/OutputDir /storage1/guerin/Active/geos-chem/ZifanRunDir/dataset/Restarts --output_dir /storage1/guerin/Active/geos-chem/ZifanRunDir/dataset/KppSteps

python3 data_split.py --input_dirs /storage1/guerin/Active/geos-chem/ZifanRunDir/Run/OutputDir /storage1/guerin/Active/geos-chem/ZifanRunDir/dataset/Restarts --answer_dir /storage1/guerin/Active/geos-chem/ZifanRunDir/dataset/KppSteps --do_split

#python3 new.py --input_dirs /storage1/guerin/Active/geos-chem/ZifanRunDir/Run/OutputDir /storage1/guerin/Active/geos-chem/ZifanRunDir/dataset/Restarts --answer_dir /storage1/guerin/Active/geos-chem/ZifanRunDir/dataset/KppSteps
#python3 new_eliminate.py --input_dirs /storage1/guerin/Active/geos-chem/ZifanRunDir/Run/OutputDir /storage1/guerin/Active/geos-chem/ZifanRunDir/dataset/Restarts --answer_dir /storage1/guerin/Active/geos-chem/ZifanRunDir/dataset/KppSteps

