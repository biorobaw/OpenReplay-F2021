#!/bin/bash

#SBATCH --time=1:00:00
#SBATCH --cpus-per-task 2
#SBATCH --mem=2000M
##SBATCH -p mri2016

baseDir=$1
sample_rate=$2
[ -z $SLURM_ARRAY_TASK_ID ] && SLURM_ARRAY_TASK_ID=$3
TASK_ID=$SLURM_ARRAY_TASK_ID

echo "TASK_ID " $TASK_ID

module add apps/python/3.7.3

PYTHONUSERBASE=/home/p/pablos/work/pythonlibs
# module list
# echo ""
# ls /usr/lib64/ | grep libffi
# echo $LD_LIBRARY_PATH
# which python
# echo $PATH
# echo $PYTHONPATH
# echo $PYTHONSTARTUP
# echo $PYTHONCASEOK
# echo $PYTHONHOME


config_range=`python ./scripts/utils/map_core_to_configs.py $baseDir $TASK_ID`
range=(${config_range//-/ })
echo "CONFIG_RANGE: " $config_range  


for i in $(seq ${range[0]} ${range[1]}); do 

    echo "python ./scripts/log_processing/processConfig.py $baseDir c$i $sample_rate"
    python ./scripts/log_processing/processConfig.py $baseDir c$i $sample_rate

    if [ $? -eq 0 ]; then
        echo "SUCCESS $baseDir $configId" 
    else
        echo "FAIL $baseDir $configId"
    fi

done



