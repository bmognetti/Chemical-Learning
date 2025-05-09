#!/bin/bash

noise=0.125
#drift=-0.002
#drift=-0.004
#drift=-0.08

for i in `seq 1 48`
do
    for drift in 0.0 0.00001 0.00005 0.0001 0.001 0.005
    do
	tail -3 noise_${noise}/run_dr${drift}_ns${noise}_patt${i}/running_log.txt
	echo ${i} ${drift}
	echo ' '
    done
done




