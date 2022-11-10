#!/bin/bash
#SBATCH -p defq
#SBATCH -n 50
#SBATCH -J sparse_uccgsd
#SBATCH -o stdout.%J
#SBATCH -e stderr.%J

module load openmpi/3.1.5/gcc-9.3.0
export OMP_NUM_THREADS=1
echo $OMP_NUM_THREADS > output-np$SLURM_NTASKS
echo $SLURM_NTASKS >> output-np$SLURM_NTAsSKS
julia --version >> output-np$SLURM_NTASKS
mpirun -np $SLURM_NTASKS julia --project=@. 8site_sparse_ansatz.jl >> output-np$SLURM_NTASKS-sparse-V-00-debug7-seed120


