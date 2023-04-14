#!/bin/sh
#SBATCH -p F4cpu
#SBATCH -J 4s-ucc-shot
#SBATCH -o stdout.%J
#SBATCH -e stderr.%J
#SBATCH -N 4
#SBATCH -n 64
#SBATCH -c 8
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sakurairihito@gmail.com
srun julia --project=@. 4site.jl >> output-np$SLURM_NTASKS-7
