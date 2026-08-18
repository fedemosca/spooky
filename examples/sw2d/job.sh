#!/bin/bash

#SBATCH -J sw2d
#SBATCH -N 1
#SBATCH -o sw2d.out
#SBATCH -n 2
#SBATCH --gpus 1
#SBATCH -t 01:00:00

ml python/3.11.3
export NUMPY_BACKEND='jax'
source /share/data2/fmosca/environments/spooky/bin/activate
python3 time_marching.py
