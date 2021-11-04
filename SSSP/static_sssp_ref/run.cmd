#!/bin/bash
#PBS -e errors.err
#PBS -o exec.log
#PBS -q rupesh_gpuq
#PBS -l select=1:ncpus=1:ngpus=1

tpdir=`echo $PBS_JOBID | cut -f 1 -d .`
tempdir=$HOME/scratch/job$tpdir
mkdir -p $tempdir
cd $tempdir
cp -R $PBS_O_WORKDIR/* .

module load cuda10.1
module load gcc640
make
./sssp inputs/inputGraph.txt outputs/output.txt > out.txt

cp outputs/output.txt $PBS_O_WORKDIR/outputs/
cp out.txt $PBS_O_WORKDIR/
rm -rf $tempdir
