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
./sssp inputs/slashdot0811.txt inputs/update_slashdot0811.txt outputs/output_slashdot0811.txt > outputs/print_slashdot0811.txt

cp outputs/output_slashdot0811.txt $PBS_O_WORKDIR/outputs/
cp outputs/print_slashdot0811.txt $PBS_O_WORKDIR/outputs/
rm -rf $tempdir
