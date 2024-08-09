#!/bin/bash
#SBATCH -J mpi_gpu_job
#SBATCH -p batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH -o %x.o%j
#SBATCH -e %x.e%j
#SBATCH --time 00:30:00
#SBATCH --gres=gpu:2

module purge
module load nvhpc/23.7

make clean
make all
make exe