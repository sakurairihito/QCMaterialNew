#!/bin/bash
#SBATCH -p defq
#SBATCH -n 32
#SBATCH -J uccgsd_dimer_tau_plus
#SBATCH -o stdout.%J
#SBATCH -e stderr.%J

module load openmpi/3.1.5/gcc-9.3.0

export OMP_NUM_THREADS=1
echo $OMP_NUM_THREADS > output-np$SLURM_NTASKS
echo $SLURM_NTASKS >> output-np$SLURM_NTASKS
julia --version >> output-np$SLURM_NTASKS
mpirun -np $SLURM_NTASKS julia --project=@. 4site_plus_minus.jl plus_true minus_false sp_tau_plus2_p_139.txt >> output-np$SLURM_NTASKS-recursivetaus

