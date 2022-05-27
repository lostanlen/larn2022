#!/bin/bash

#BATCH --job-name=larn2022_sm4_RA1_20220429_021731
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --mem=16GB
#SBATCH --begin=now+615
#SBATCH --output=../slurm/larn2022_sm4_RA1_20220429_021731_%j.out

module purge

# The first argument is the name of the WAV file input.
cd /home/vl1019/BirdNET

python analyze.py --i /scratch/vl1019/LARN_SONS_2022/SM4_ROSELIERES_DONGES/RA1_DONGES/2022-04-29/RA1_20220429_021731.wav --o /scratch/vl1019/LARN_SONS_2022/BirdNET/SM4_ROSELIERES_DONGES --lat 47.34 --lon -2.20 --week 16 --min_conf 0.1