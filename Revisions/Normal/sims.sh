#!/bin/bash

# module unload r 
module load r/4.1.3

sbatch -N 1 -n 20 -p general -t 2-00:00:00 --mem=10g -o sims R CMD BATCH --no-restore runGLMM_normal11.R normal11.rout
sbatch -N 1 -n 20 -p general -t 2-00:00:00 --mem=10g -o sims R CMD BATCH --no-restore runGLMM_normal21.R normal21.rout
sbatch -N 1 -n 20 -p general -t 2-00:00:00 --mem=10g -o sims R CMD BATCH --no-restore runGLMM_normal12.R normal12.rout
sbatch -N 1 -n 20 -p general -t 2-00:00:00 --mem=10g -o sims R CMD BATCH --no-restore runGLMM_normal22.R normal22.rout
