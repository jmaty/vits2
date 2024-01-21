#!/bin/bash
set -e

if [[ "$#" -lt 2 ]]; then
     echo "Usage: run_train.sh run_name config_path [runs]"
     exit 1
fi

RUN_NAME=$1
CONFIG=$2

# Default values
RUNS=1

if [[ "$#" -gt 2 ]]; then
     # Number of runs
     RUNS=$3
fi

WALLTIME="-l walltime=24:00:00"
SELECT="-l select=1:ncpus=2:ngpus=1:mem=64gb:gpu_mem=40gb:scratch_local=20gb"
DEPS=""
PREV_JOBID=""

SINGULARITY="$MYHOME/singularity/pbs_job_23.12-r2.sh"

# Run the script sequentially with the given number of repetions `RUNS`
for i in $(seq 1 $RUNS); do
     # Run PBS script
     JOBID=$(qsub -N "$RUN_NAME" \
          -q gpu \
          -j oe \
          $WALLTIME \
          $SELECT \
          $DEPS \
          -- $SINGULARITY \
          python3 train.py -c $CONFIG -m "$RUN_NAME")
     echo "$RUN_NAME: RUN: $i, $JOBID: $PREV_JOBID -> $JOBID"
     # Update dependencies to enable sequential run
     DEPS="-W depend=afterany:$JOBID"
     PREV_JOBID=$JOBID
done
