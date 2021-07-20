#!/bin/bash
#SBATCH -p defq
#SBATCH -n 32
#SBATCH -J uccgsd_dimer_tau_minus
#SBATCH -o stdout.%J
#SBATCH -e stderr.%J

module load openmpi/3.1.5/gcc-9.3.0

export OMP_NUM_THREADS=1
echo $OMP_NUM_THREADS > output-np$SLURM_NTASKS
echo $SLURM_NTASKS >> output-np$SLURM_NTASKS
julia --version >> output-np$SLURM_NTASKS
mpirun -np $SLURM_NTASKS julia --project=@. 3bath_1imp_tau_minus.jl >> output-np$SLURM_NTASKS-p=2
