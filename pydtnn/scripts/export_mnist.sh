#!/bin/bash

export OMP_NUM_THREADS=1
export PYTHONOPTIMIZE=2
export PYTHONUNBUFFERED="True"
python3 export_dataset.py \
  --model=simplecnn \
  --dataset=mnist \
  --dataset_train_path=datasets/mnist \
  --dataset_test_path=datasets/mnist \
  --dataset_raw_path="datasets/mnist/dataset.npz" \
  --parallel=sequential \
  --shared_storage=True
