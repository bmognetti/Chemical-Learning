#!/bin/bash

drift=0.3
noise=0.1

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
