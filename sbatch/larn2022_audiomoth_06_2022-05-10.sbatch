#!/bin/bash

#BATCH --job-name=larn2022_audiomoth_06_2022-05-10
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --mem=64GB
#SBATCH --begin=now+360
#SBATCH --output=../slurm/larn2022_audiomoth_06_2022-05-10_%j.out

module purge

# The first argument is the name of the WAV file input.
cd /home/vl1019/BirdNET

python analyze.py --i /scratch/vl1019/LARN_SONS_2022/AUDIOMOTH_ESPACES_BOISES/6/2022-05-10 --o /scratch/vl1019/LARN_SONS_2022/BirdNET/AUDIOMOTH_ESPACES_BOISES-v2/6 --lat 47.34 --lon -2.20 --week 17 --min_conf 0.1