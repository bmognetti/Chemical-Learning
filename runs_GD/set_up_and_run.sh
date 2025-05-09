#!/bin/bash

noise=0.0
#drift=-0.002
#drift=-0.004
#drift=-0.08

for i in `seq 1 42`
do
    for drift in 0.0 -0.0001 -0.001 -0.005
    do
	cp -r run run_dr${drift}_ns${noise}_patt${i}
	cd run_dr${drift}_ns${noise}_patt${i}

	# prepare input file 
	echo '{' > patterns_run_config_GD.json
	echo '    "sigma2_0": 2.0,' >> patterns_run_config_GD.json
	echo '    "mean_0": 0.0,' >> patterns_run_config_GD.json
	echo '    "sigma2":' ${noise}',' >> patterns_run_config_GD.json
	echo '    "loss": 3,' >> patterns_run_config_GD.json
	echo '    "unpert_pattern": "C",' >> patterns_run_config_GD.json
	echo '    "precision": 1e-8,' >> patterns_run_config_GD.json
	echo '    "rate_k":' ${drift}',' >> patterns_run_config_GD.json
	echo '    "learning_rate": 0.005,' >> patterns_run_config_GD.json
	echo '    "N_batches_passed": 10000,' >> patterns_run_config_GD.json
	echo '    "N_patt": 48' >> patterns_run_config_GD.json
	echo '}' >> patterns_run_config_GD.json

	cp ../patterns_0/patterns_0_${i}.npy ./patterns_0.npy

	sbatch submit.sh 
  
	cd ..
	echo $i $noise
    done
done

