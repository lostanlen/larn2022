#!/bin/bash

#BATCH --job-name=larn2022_sm4_20220506_020001
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --mem=16GB
#SBATCH --begin=now+15
#SBATCH --output=../slurm/larn2022_sm4_20220506_020001_%j.out

module purge

# The first argument is the name of the WAV file input.
cd /home/vl1019/BirdNET

python analyze.py --i /scratch/vl1019/LARN_SONS_2022/sensor18/2022-05-06/20220506_020001.WAV --o /scratch/vl1019/LARN_SONS_2022/BirdNET/sensor18 --lat 47.34 --lon -2.20 --week 16 --min_conf 0.1