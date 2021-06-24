#!/bin/bash

#SBATCH -J go_13
#SBATCH -p batch
#SBATCH -t 0-01:00:00
#SBATCH --constraint=skl
#SBATCH --cpus-per-task=10

export OMP_PROC_BIND=true
export PSI_SCRATCH=/tmp/

module load anaconda3
module load iomklc/triton-2017a
module load cmake/3.12.1

srun python geoopt.py


