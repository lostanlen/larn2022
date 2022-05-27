#!/bin/bash

#BATCH --job-name=larn2022_audiomoth_31
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=96:00:00
#SBATCH --mem=64GB
#SBATCH --begin=now+1860
#SBATCH --output=../slurm/larn2022_audiomoth_31_%j.out

module purge

# The first argument is the name of the WAV file input.
cd /home/vl1019/BirdNET

python analyze.py --i /scratch/vl1019/LARN_SONS_2022/AUDIOMOTH_ESPACES_BOISES/31/2022-05-05/20220505_000100.WAV --o /scratch/vl1019/LARN_SONS_2022/BirdNET/AUDIOMOTH_ESPACES_BOISES/31 --lat 47.34 --lon -2.20 --week 16 --min_conf 0.1
python analyze.py --i /scratch/vl1019/LARN_SONS_2022/AUDIOMOTH_ESPACES_BOISES/31/2022-05-05/20220505_020002.WAV --o /scratch/vl1019/LARN_SONS_2022/BirdNET/AUDIOMOTH_ESPACES_BOISES/31 --lat 47.34 --lon -2.20 --week 16 --min_conf 0.1
python analyze.py --i /scratch/vl1019/LARN_SONS_2022/AUDIOMOTH_ESPACES_BOISES/31/2022-05-06/20220506_000100.WAV --o /scratch/vl1019/LARN_SONS_2022/BirdNET/AUDIOMOTH_ESPACES_BOISES/31 --lat 47.34 --lon -2.20 --week 16 --min_conf 0.1
python analyze.py --i /scratch/vl1019/LARN_SONS_2022/AUDIOMOTH_ESPACES_BOISES/31/2022-05-06/20220506_020002.WAV --o /scratch/vl1019/LARN_SONS_2022/BirdNET/AUDIOMOTH_ESPACES_BOISES/31 --lat 47.34 --lon -2.20 --week 16 --min_conf 0.1
python analyze.py --i /scratch/vl1019/LARN_SONS_2022/AUDIOMOTH_ESPACES_BOISES/31/2022-05-07/20220507_000100.WAV --o /scratch/vl1019/LARN_SONS_2022/BirdNET/AUDIOMOTH_ESPACES_BOISES/31 --lat 47.34 --lon -2.20 --week 16 --min_conf 0.1
python analyze.py --i /scratch/vl1019/LARN_SONS_2022/AUDIOMOTH_ESPACES_BOISES/31/2022-05-07/20220507_020002.WAV --o /scratch/vl1019/LARN_SONS_2022/BirdNET/AUDIOMOTH_ESPACES_BOISES/31 --lat 47.34 --lon -2.20 --week 16 --min_conf 0.1
python analyze.py --i /scratch/vl1019/LARN_SONS_2022/AUDIOMOTH_ESPACES_BOISES/31/2022-05-08/20220508_000100.WAV --o /scratch/vl1019/LARN_SONS_2022/BirdNET/AUDIOMOTH_ESPACES_BOISES/31 --lat 47.34 --lon -2.20 --week 17 --min_conf 0.1
python analyze.py --i /scratch/vl1019/LARN_SONS_2022/AUDIOMOTH_ESPACES_BOISES/31/2022-05-08/20220508_020002.WAV --o /scratch/vl1019/LARN_SONS_2022/BirdNET/AUDIOMOTH_ESPACES_BOISES/31 --lat 47.34 --lon -2.20 --week 17 --min_conf 0.1
python analyze.py --i /scratch/vl1019/LARN_SONS_2022/AUDIOMOTH_ESPACES_BOISES/31/2022-05-09/20220509_000100.WAV --o /scratch/vl1019/LARN_SONS_2022/BirdNET/AUDIOMOTH_ESPACES_BOISES/31 --lat 47.34 --lon -2.20 --week 17 --min_conf 0.1
python analyze.py --i /scratch/vl1019/LARN_SONS_2022/AUDIOMOTH_ESPACES_BOISES/31/2022-05-09/20220509_020002.WAV --o /scratch/vl1019/LARN_SONS_2022/BirdNET/AUDIOMOTH_ESPACES_BOISES/31 --lat 47.34 --lon -2.20 --week 17 --min_conf 0.1
python analyze.py --i /scratch/vl1019/LARN_SONS_2022/AUDIOMOTH_ESPACES_BOISES/31/2022-05-10/20220510_000100.WAV --o /scratch/vl1019/LARN_SONS_2022/BirdNET/AUDIOMOTH_ESPACES_BOISES/31 --lat 47.34 --lon -2.20 --week 17 --min_conf 0.1
python analyze.py --i /scratch/vl1019/LARN_SONS_2022/AUDIOMOTH_ESPACES_BOISES/31/2022-05-10/20220510_020002.WAV --o /scratch/vl1019/LARN_SONS_2022/BirdNET/AUDIOMOTH_ESPACES_BOISES/31 --lat 47.34 --lon -2.20 --week 17 --min_conf 0.1
python analyze.py --i /scratch/vl1019/LARN_SONS_2022/AUDIOMOTH_ESPACES_BOISES/31/2022-05-11/20220511_000100.WAV --o /scratch/vl1019/LARN_SONS_2022/BirdNET/AUDIOMOTH_ESPACES_BOISES/31 --lat 47.34 --lon -2.20 --week 17 --min_conf 0.1
python analyze.py --i /scratch/vl1019/LARN_SONS_2022/AUDIOMOTH_ESPACES_BOISES/31/2022-05-11/20220511_020002.WAV --o /scratch/vl1019/LARN_SONS_2022/BirdNET/AUDIOMOTH_ESPACES_BOISES/31 --lat 47.34 --lon -2.20 --week 17 --min_conf 0.1
python analyze.py --i /scratch/vl1019/LARN_SONS_2022/AUDIOMOTH_ESPACES_BOISES/31/2022-05-12/20220512_000100.WAV --o /scratch/vl1019/LARN_SONS_2022/BirdNET/AUDIOMOTH_ESPACES_BOISES/31 --lat 47.34 --lon -2.20 --week 17 --min_conf 0.1
python analyze.py --i /scratch/vl1019/LARN_SONS_2022/AUDIOMOTH_ESPACES_BOISES/31/2022-05-12/20220512_020002.WAV --o /scratch/vl1019/LARN_SONS_2022/BirdNET/AUDIOMOTH_ESPACES_BOISES/31 --lat 47.34 --lon -2.20 --week 17 --min_conf 0.1
python analyze.py --i /scratch/vl1019/LARN_SONS_2022/AUDIOMOTH_ESPACES_BOISES/31/2022-05-13/20220513_000100.WAV --o /scratch/vl1019/LARN_SONS_2022/BirdNET/AUDIOMOTH_ESPACES_BOISES/31 --lat 47.34 --lon -2.20 --week 17 --min_conf 0.1
python analyze.py --i /scratch/vl1019/LARN_SONS_2022/AUDIOMOTH_ESPACES_BOISES/31/2022-05-13/20220513_020002.WAV --o /scratch/vl1019/LARN_SONS_2022/BirdNET/AUDIOMOTH_ESPACES_BOISES/31 --lat 47.34 --lon -2.20 --week 17 --min_conf 0.1