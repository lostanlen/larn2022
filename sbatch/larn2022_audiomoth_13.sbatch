#!/bin/bash

#BATCH --job-name=larn2022_audiomoth_13
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=96:00:00
#SBATCH --mem=64GB
#SBATCH --begin=now+780
#SBATCH --output=../slurm/larn2022_audiomoth_13_%j.out

module purge

# The first argument is the name of the WAV file input.
cd /home/vl1019/BirdNET
