#!/bin/bash
#SBATCH -p defq
#SBATCH -n 10
#SBATCH -J uccgsd_hubbard
#SBATCH -o stdout.%J
#SBATCH -e stderr.%J

module load openmpi/3.1.5/gcc-9.3.0

echo $OMP_NUM_THREADS > output-np$SLURM_NTASKS
mpirun -np $SLURM_NTASKS julia --project=@. run.jl >> output-np$SLURM_NTASKS
