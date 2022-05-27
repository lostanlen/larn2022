#!/bin/bash

#BATCH --job-name=larn2022_sm4_D1_20220429_005357
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --mem=16GB
#SBATCH --begin=now+200
#SBATCH --output=../slurm/larn2022_sm4_D1_20220429_005357_%j.out

module purge

# The first argument is the name of the WAV file input.
cd /home/vl1019/BirdNET

python analyze.py --i /scratch/vl1019/LARN_SONS_2022/SM4_ROSELIERES_DONGES/D1_DONGES/2022-04-29/D1_20220429_005357.wav --o /scratch/vl1019/LARN_SONS_2022/BirdNET/SM4_ROSELIERES_DONGES --lat 47.34 --lon -2.20 --week 16 --min_conf 0.1