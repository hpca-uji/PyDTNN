#!/bin/bash

self=$(realpath "$0")
dir="${self%/*}"

export OMP_NUM_THREADS=1
export PYTHONOPTIMIZE=2
export PYTHONUNBUFFERED="True"
python3 "${dir:?}/export_dataset.py" \
  --model=simplecnn \
  --dataset=mnist \
  --dataset-path=datasets/mnist \
  --export-weights=1,1,1,1 \
  --parallel=sequential \
  --shared-storage=True
