#!/bin/bash
# Submits one case, with the SLURM log written into the case directory alongside its
# outputs. This has to happen here rather than in job.sh: #SBATCH directives are read by
# sbatch at submit time and never expand shell variables, so the output path can only be
# set from the command line.
#
# Usage: ./submit.sh ./outs/1

set -euo pipefail

case_dir="${1:?usage: $0 <case_dir>}"
case_dir="${case_dir%/}"

if [[ ! -d "$case_dir" ]]; then
    echo "case directory not found: $case_dir" >&2
    exit 1
fi
if [[ ! -f "$case_dir/params.py" ]]; then
    echo "no params.py in $case_dir" >&2
    exit 1
fi

# %j keeps one log per job: re-running a case leaves the earlier log in place instead of
# overwriting it. SLURM opens this file itself, so the directory must already exist.
sbatch --export=ALL,CASE="$case_dir" -o "$case_dir/sw2d-%j.out" job.sh
