#!/bin/sh
#in git root folder execute the following commands:

source "experiments/replayF2021_experiments/run_set_variables.sh"



module add apps/python/3.7.3

for E in ${RUN[*]}; do
	echo "python3 $SCRIPT_5_PLOT_EXPERIMENT $(map $E LOG_FOLDER)/ "
	python3 $SCRIPT_5_PLOT_EXPERIMENT $(map $E LOG_FOLDER)/ 
done

