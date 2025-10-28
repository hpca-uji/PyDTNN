#!/bin/bash

self=$(realpath "$0")
dir="${self%/*}"

export OMP_NUM_THREADS=1
export PYTHONOPTIMIZE=2
export PYTHONUNBUFFERED="True"
python3 "${dir:?}/export_dataset.py" \
  --model=simplecnn \
  --dataset=mnist \
  --dataset_path=datasets/mnist \
  --export_weights=1,1.125,1.25,1.5 \
  --parallel=sequential \
  --shared_storage=True
