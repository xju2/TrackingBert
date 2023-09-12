#!/bin/bash
#SBATCH -J TrackingBert
#SBATCH -q regular
#SBATCH -C gpu
#SBATCH --gpus 1
#SBATCH -G 1
#SBATCH -N 1
#SBATCH -c 1
#SBATCH -t 12:00:00
#SBATCH -A m3443


source activate gnn # Change the environment name
cd /pscratch/sd/a/andrish/hept # Change to the TrackingBert directory
python train.py