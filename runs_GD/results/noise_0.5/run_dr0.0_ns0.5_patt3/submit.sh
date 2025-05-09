#!/bin/bash
#
#SBATCH --job-name=GD_noise_0.25
#SBATCH --output=res.txt
#SBATCH --partition=batch
#
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=100


module purge
module load SciPy-bundle
module load Python/3.11.3-GCCcore-12.3.0

python CL_gradient_descent.py 
