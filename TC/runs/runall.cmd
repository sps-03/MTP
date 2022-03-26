#!/bin/bash
#PBS -e errors.err
#PBS -o logs.log
#PBS -q rupesh_gpuq
#PBS -l select=1:ncpus=1:ngpus=1

JOB_ID=`echo $PBS_JOBID | cut -f 1 -d .`
TEMP_DIR=$HOME/scratch/job$JOB_ID
FILE="../inputGraphs/inputs/usaRoadNet.txt"
FILE_NAME=$(basename $FILE)
IS_DIRECTED=0

mkdir -p $TEMP_DIR
cd $TEMP_DIR
cp -R $PBS_O_WORKDIR/* .

module load cuda10.1
module load gcc640
nvcc main.cu -o tc

rm outputs/*

for UPDATE_PERC in {1..20}
do
./tc ../inputGraphs/inputs/$FILE_NAME ../inputGraphs/updates/update_$FILE_NAME $UPDATE_PERC $IS_DIRECTED >> outputs/print_$FILE_NAME
done

cp outputs/print_$FILE_NAME $PBS_O_WORKDIR/outputs/

rm -rf $TEMP_DIR