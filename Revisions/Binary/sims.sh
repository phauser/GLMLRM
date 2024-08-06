#!/bin/bash

# module unload r 
module load r/4.1.3

for c in {1..5}
do
    sbatch -N 1 -n 20 -p general -t 6-00:00:00 --mem=20g -o sims R CMD BATCH --no-restore "--args $c" runGLMM_binary11.R binary11_${c}.rout
    sbatch -N 1 -n 20 -p general -t 6-00:00:00 --mem=20g -o sims R CMD BATCH --no-restore "--args $c" runGLMM_binary12.R binary12_${c}.rout
    sbatch -N 1 -n 20 -p general -t 6-00:00:00 --mem=20g -o sims R CMD BATCH --no-restore "--args $c" runGLMM_binary21.R binary21_${c}.rout
    sbatch -N 1 -n 20 -p general -t 6-00:00:00 --mem=20g -o sims R CMD BATCH --no-restore "--args $c" runGLMM_binary22.R binary22_${c}.rout
done





