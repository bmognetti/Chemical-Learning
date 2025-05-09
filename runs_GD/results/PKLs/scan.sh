#!/bin/bash
#SBATCH --account=cfn3
#SBATCH --nodes=1
#SBATCH --ntasks=48
#SBATCH --time=10:00:00
#SBATCH --partition=cfn
#SBATCH --qos=cfn
#SBATCH --job-name=CL_scan
#SBATCH --mail-user=oleksiyt@bnl.gov
#SBATCH --mail-type=ALL

module load python

export OMP_NUM_THREADS=1

# Loop over indices 1 to 48.
for index_run in {1..48}; do
    srun --exclusive -N1 -n1 python CL_EL_scan.py $index_run >> "scan.out" 2>&1 &
done

wait
exit
