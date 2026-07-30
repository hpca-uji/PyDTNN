#!/bin/bash

export OMP_NUM_THREADS=16
export PYTHONOPTIMIZE=2
export PYTHONUNBUFFERED="True"

pwd
mpirun -np 4 ./scripts/models/run_resnet_folder_unbalanced_real.sh