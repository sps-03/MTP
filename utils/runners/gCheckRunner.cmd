#!/bin/bash
#PBS -e errors.err
#PBS -o logs.log
#PBS -q rupesh_gpuq
#PBS -l select=1:ncpus=1:ngpus=1

JOB_ID=`echo $PBS_JOBID | cut -f 1 -d .`
tempdir=$HOME/scratch/job$JOB_ID
mkdir -p $tempdir
cd $tempdir
cp -R $PBS_O_WORKDIR/* .
cp $PBS_O_WORKDIR/../utils/graphChecker.cpp .

module load cuda10.1
module load gcc640
g++ graphChecker.cpp -o graphChecker
./graphChecker ../inputGraphs/inputs/twitter.txt output.txt undirected

cp output.txt $PBS_O_WORKDIR/
rm -rf $tempdir