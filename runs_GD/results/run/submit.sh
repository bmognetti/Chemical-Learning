#!/bin/bash
#
#SBATCH --job-name=test
#SBATCH --output=res.txt
#SBATCH --partition=batch
#
#SBATCH --time=50:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=100

module purge
module load releases/2023a
module load Python/3.11.3-GCCcore-12.3.0
module load SciPy-bundle/2023.07-gfbf-2023a
module load matplotlib/3.7.2-gfbf-2023a

python CL_gradient_descent.py 
