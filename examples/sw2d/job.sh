#!/bin/bash

# Submit with ./submit.sh <case_dir>, which passes CASE in and redirects the log into
# that directory. The -o below is only the fallback for a bare `sbatch job.sh`, which
# fails on the CASE check anyway.

#SBATCH -J sw2d
#SBATCH -N 1
#SBATCH -o sw2d.out
#SBATCH -n 2
#SBATCH --gpus 1
#SBATCH -t 01:00:00

ml cuda/11.8
ml python/3.11.3
echo $CUDA_VISIBLE_DEVICES
export NUMPY_BACKEND='jax'
source /share/data2/fmosca/environments/spooky/bin/activate
python3 -c "import jax; print(jax.devices())"

CASE="${CASE:?set by submit.sh -- run ./submit.sh <case_dir> instead of sbatch job.sh}"
python3 time_marching.py "$CASE"
python3 video.py "$CASE/hhms.npy"

