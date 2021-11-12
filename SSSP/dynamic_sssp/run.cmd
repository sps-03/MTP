#!/bin/bash
#PBS -e errors.err
#PBS -o logs.log
#PBS -q rupesh_gpuq
#PBS -l select=1:ncpus=1:ngpus=1

tpdir=`echo $PBS_JOBID | cut -f 1 -d .`
tempdir=$HOME/scratch/job$tpdir
mkdir -p $tempdir
cd $tempdir
cp -R $PBS_O_WORKDIR/* .

module load cuda10.1
module load gcc640
nvcc main.cu -o sssp
./sssp inputs/input_sample1.txt inputs/update_sample1.txt outputs/output_sample.txt > outputs/print.txt

cp outputs/output_sample.txt $PBS_O_WORKDIR/outputs/
cp outputs/print.txt $PBS_O_WORKDIR/outputs/
rm -rf $tempdir
