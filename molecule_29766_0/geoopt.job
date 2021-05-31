#!/bin/bash

#SBATCH -J go_16
#SBATCH -p batch
#SBATCH -t 0-02:00:00
#SBATCH --constraint=[skl]
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=1000

export OMP_PROC_BIND=true
export PSI_SCRATCH=/tmp/

source ~/programs/psi4env/bin/activate
module load iomklc/triton-2017a
module load cmake/3.12.1

srun python geoopt.py


