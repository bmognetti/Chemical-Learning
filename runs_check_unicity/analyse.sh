#!/bin/bash

noise=0.125
#drift=-0.002
#drift=-0.004
#drift=-0.08

for noise in 0.02 0.045 0.08 0.125 0.18 0.245 0.32 0.405 0.605 0.5 0.845 1.445 2.0 2.88
do
    
    for i in `seq 1 48`
    do
	for drift in 0.0 0.00001 0.00005 0.0001 0.001 0.005
	do

	    echo 'processing noise, patt, drift=' $noise $i $drift 

	    cp run/base.svg run/prepare_figures.ipynb noise_${noise}/run_dr${drift}_ns${noise}_patt${i}/
	    cd noise_${noise}/run_dr${drift}_ns${noise}_patt${i}
	
	    jupyter nbconvert --to html --execute prepare_figures.ipynb
	
  
	    cd ../../
	done
    done
done

