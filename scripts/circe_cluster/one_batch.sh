#!/bin/bash

#SBATCH --time=4:00:00
#SBATCH --cpus-per-task 2 
#SBATCH --mem=1000M
##SBATCH -p mri2016

echo 'in script'
configFile=$1
baseLogFolder=$2

CMD_ARGS="-cp OpenReplay-F2021-1.0.0-SNAPSHOT-jar-with-dependencies.jar -Xmx1500m com.github.biorobaw.scs.Main"

module add apps/jdk/15.0.2


FAILED_IDS=""
if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
  IDS=$3
  IFS=,
  
  for configId in $IDS;
  do
    IFS=$' \t\n'
    echo "---java $CMD_ARGS $configFile $configId $baseLogFolder"
    java $CMD_ARGS $configFile $configId $baseLogFolder
  
    if [ $? -eq 0 ]; then
        echo SUCCESS
    else
        FAILED_IDS="$configId, $FAILED_IDS"
    fi
  done
  

else
  batchSize=$3
  min_indiv=$4
  max_indiv=$5
  do_missing=$6
  batchID=$SLURM_ARRAY_TASK_ID
  baseID=`expr $batchSize \* $batchID`
  maxID=`expr $batchSize - 1`
  
  
  for i in $(seq 0 $maxID)
  do
    configId=`expr $baseID + $i`

    
    if [ \( -z "$min_indiv" -o "$min_indiv" -le "$configId" \) -a \( -z "$max_indiv" -o "$configId" -le "$max_indiv" \) ]; then

      [ $do_missing == 'DO_MISSING' ] && configId=`cat $baseLogFolder/missing.csv | cut -d "," -f $(expr $configId + 1)`
      echo "---java $CMD_ARGS $configFile $configId $baseLogFolder"
      java $CMD_ARGS $configFile $configId $baseLogFolder
      
      
      if [ $? -eq 0 ]; then
          echo SUCCESS
      else
          FAILED_IDS="$configId, $FAILED_IDS"
      fi

    else

      echo "skipping individual $configId"

    fi
    
  done

fi


if [ -z $FAILED_IDS ]; then
  echo "FAILED IDS: $FAILED_IDS"
fi









