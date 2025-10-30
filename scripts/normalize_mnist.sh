#!/bin/bash

self=$(realpath "$0")
dir="${self%/*}"

export OMP_NUM_THREADS=1
export PYTHONOPTIMIZE=2
export PYTHONUNBUFFERED="True"
python3 "${dir:?}/normalize_dataset.py" \
  --model=simplecnn \
  --dataset=chestxray \
  --dataset_path=datasets/chestxray \
  --parallel=sequential \
  --shared_storage=True
