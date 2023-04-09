#!/bin/bash
#SBATCH -p defq
#SBATCH -n 30
#SBATCH -J kucj-6site
#SBATCH -o stdout.%J
#SBATCH -e stderr.%J

module load openmpi/3.1.5/gcc-9.3.0
export OMP_NUM_THREADS=1
echo $OMP_NUM_THREADS > output-np$SLURM_NTASKS
echo $SLURM_NTASKS >> output-np$SLURM_NTAsSKS
julia --version >> output-np$SLURM_NTASKS
#mpirun -np $SLURM_NTASKS julia --project=@. 6site_kucj.jl opt_params_k3_seed150.txt >> output-np$SLURM_NTASKS-kucj-k4-optparams1-seed150
mpirun -np $SLURM_NTASKS julia --project=@. 6site_kucj.jl opt_params_kucjsparse_k1_seed90.txt >> output-np$SLURM_NTASKS-kucj-sp