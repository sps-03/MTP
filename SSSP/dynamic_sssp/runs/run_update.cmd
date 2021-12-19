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

module load gcc640
g++ utils/update_generator.cpp -o generator
./generator ../liveJournal.txt ../update_liveJournal.txt > outputs/print_update.txt

cp ../update_liveJournal.txt $PBS_O_WORKDIR/inputs/
cp outputs/print_update.txt $PBS_O_WORKDIR/outputs/
rm -rf $tempdir