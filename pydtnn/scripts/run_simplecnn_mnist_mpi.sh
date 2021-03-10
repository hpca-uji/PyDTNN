#!/bin/bash

export OMP_NUM_THREADS=1
mpirun -np 4 \
  python3 -u test/benchmarks_CNN.py \
  --model=simplecnn \
  --dataset=mnist \
  --dataset_train_path=datasets/mnist \
  --dataset_test_path=datasets/mnist \
  --test_as_validation=False \
  --flip_images=True \
  --batch_size=64 \
  --validation_split=0.2 \
  --num_epochs=50 \
  --evaluate=True \
  --optimizer=adam \
  --learning_rate=0.01 \
  --loss_func=categorical_cross_entropy \
  --lr_schedulers=warm_up,reduce_lr_every_nepochs \
  --reduce_lr_every_nepochs_factor=0.5 \
  --reduce_lr_every_nepochs_nepochs=30 \
  --reduce_lr_every_nepochs_min_lr=0.001 \
  --early_stopping_metric=val_categorical_cross_entropy \
  --early_stopping_patience=20 \
  --parallel=data \
  --non_blocking_mpi=False \
  --tracing=False \
  --profile=False \
  --enable_gpu=False \
  --dtype=float32
