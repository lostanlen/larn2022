import datetime
import glob
import h5py
import librosa
import logging
import numpy as np
import os
import pandas as pd
import resampy
import scipy
import sys
import time


# Define constants.
script_name = "analyze.py"
script_dir = "/home/vl1019/BirdNET"
dataset_name = "AUDIOMOTH_ESPACES_BOISES"
run_str = "BirdNET"

# Print header.
start_time = int(time.time())
print(str(datetime.datetime.now()) + " Start.")
print("Scheduling BirdNET on " + dataset_name + ".")
print("")
print("h5py version: {:s}".format(h5py.__version__))
print("librosa version: {:s}".format(librosa.__version__))
print("numpy version: {:s}".format(np.__version__))
print("pandas version: {:s}".format(pd.__version__))
print("resampy version: {:s}".format(resampy.__version__))
print("scipy version: {:s}".format(scipy.__version__))
print("")

# Create folder.
sbatch_dir = "sbatch"
os.makedirs(sbatch_dir, exist_ok=True)
slurm_dir = "slurm"
os.makedirs(slurm_dir, exist_ok=True)

# Define input directory.
input_dir = os.path.join("/scratch/vl1019/LARN_SONS_2022", dataset_name)


# Define output directory.
out_dir = os.path.join(
    "/scratch/vl1019/LARN_SONS_2022", run_str, dataset_name + "-v2")
os.makedirs(out_dir, exist_ok=True)


# Loop over directories.
n_sensors = 36
job_names = []
for sensor_id in range(1, 1+n_sensors):

    sensor_name = str(sensor_id).zfill(2)
    sensor_dir = os.path.join(input_dir, str(sensor_id))
    out_sensor_dir = os.path.join(out_dir, str(sensor_id))
    os.makedirs(out_sensor_dir, exist_ok=True)

    dates = os.listdir(sensor_dir)

    for date in dates:
        n_minutes = 0
        n_hours = 24
        walltime_str = ":".join([
            str(n_hours).zfill(2), str(n_minutes).zfill(2), "00"])

        # Make job name
        job_name = "_".join([
            "larn2022", "audiomoth", sensor_name, date])
        sbatch_name = job_name + ".sbatch"
        sbatch_path = os.path.join(sbatch_dir, sbatch_name)

        # Create SBATCH file.
        with open(sbatch_path, "w") as f:
            f.write("#!/bin/bash\n")
            f.write("\n")
            f.write("#BATCH --job-name=" + job_name + "\n")
            f.write("#SBATCH --nodes=1\n")
            f.write("#SBATCH --tasks-per-node=1\n")
            f.write("#SBATCH --cpus-per-task=1\n")
            f.write("#SBATCH --time=" + walltime_str + "\n")
            f.write("#SBATCH --mem=64GB\n")
            f.write("#SBATCH --begin=now+{}\n".format(str(sensor_id*60)))
            f.write("#SBATCH --output=../slurm/" + job_name + "_%j.out\n")
            f.write("\n")
            f.write("module purge\n")
            f.write("\n")
            f.write("# The first argument is the name of the WAV file input.\n")
            f.write("cd " + script_dir + "\n")

            # Find date.
            year = int(date.split("-")[0])
            month = int(date.split("-")[1])
            day = int(date.split("-")[2])
            week_id = (month-1) * 4 + (day-1) // 7

            # Create file path.
            script_path_with_args = " ".join([
                script_name,
                "--i", os.path.join(sensor_dir, date),
                "--o", out_sensor_dir,
                "--lat", "47.34",
                "--lon", "-2.20",
                "--week", str(week_id),
                "--min_conf", "0.1"])

            f.write("\n")
            f.write("python " + script_path_with_args)

        job_names.append(job_name)


# Open shell file.
shell_name = "larn2022_audiomoth_v2.sh"
shell_path = os.path.join(sbatch_dir, shell_name)
with open(shell_path, "w") as f:
    # Print header
    f.write("# This shell script schedules all Slurm jobs " +\
        "for running {} on {} (v2).\n".format(run_str, dataset_name))
    f.write("\n")

    # Loop over WAV files.
    for job_name in job_names:
        # Define job name.
        sbatch_str = "sbatch " + job_name + ".sbatch"

        # Write SBATCH command to shell file.
        f.write(sbatch_str + "\n")


# Grant permission to execute the shell file.
# https://stackoverflow.com/a/30463972
mode = os.stat(shell_path).st_mode
mode |= (mode & 0o444) >> 2
os.chmod(shell_path, mode)


# Print elapsed time.
print(str(datetime.datetime.now()) + " Finish.")
elapsed_time = time.time() - int(start_time)
elapsed_hours = int(elapsed_time / (60 * 60))
elapsed_minutes = int((elapsed_time % (60 * 60)) / 60)
elapsed_seconds = elapsed_time % 60.
elapsed_str = "{:>02}:{:>02}:{:>05.2f}".format(elapsed_hours,
                                               elapsed_minutes,
                                               elapsed_seconds)
print("Total elapsed time: " + elapsed_str + ".")
