#!/bin/bash

self=$(realpath "$0")
dir="${self%/*}"

export OMP_NUM_THREADS=1
export PYTHONOPTIMIZE=2
export PYTHONUNBUFFERED="True"
python3 "${dir:?}/export_dataset.py" \
  --model=simplecnn \
  --dataset=mnist \
  --dataset_train_path=datasets/mnist \
  --dataset_test_path=datasets/mnist \
  --dataset_export_split_weights=1,1.125,1.25,1.5 \
  --dataset_raw_path='datasets/mnist/dataset.${split}.npz' \
  --parallel=sequential \
  --shared_storage=True
