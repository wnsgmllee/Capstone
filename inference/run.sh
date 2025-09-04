#!/usr/bin/bash

#SBATCH -J Diff-Foley
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=29G
#SBATCH -p batch_ugrad
#SBATCH -t 0-6
#SBATCH -o /data/jhlee39/workspace/repos/Diff-Foley/inference/logs/slurm-%A.out


python diff_foley_inference.py 
