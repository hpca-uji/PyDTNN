#!/bin/bash

export OMP_NUM_THREADS=1
export PYTHONOPTIMIZE=2
export PYTHONUNBUFFERED="True"
mpirun -np 4 \
  pydtnn-benchmark \
  --model=simplecnn \
  --dataset=mnist \
  --dataset-path=datasets/mnist \
  --model-sync-freq=64 \
  --test-as-validation=False \
  --augment-flip=0.5 \
  --batch-size=64 \
  --validation-split=0.2 \
  --num-epochs=50 \
  --evaluate=True \
  --optimizer=adam \
  --learning-rate=0.01 \
  --loss-func=categorical_cross_entropy \
  --schedulers=warm_up,reduce_lr_every_nepochs \
  --reduce-lr-every-nepochs-factor=0.5 \
  --reduce-lr-every-nepochs-nepochs=30 \
  --reduce-lr-every-nepochs-min-lr=0.001 \
  --early-stopping-metric=val_categorical_cross_entropy \
  --early-stopping-patience=20 \
  --parallel=data \
  --use-blocking-mpi=True \
  --tracing=False \
  --profile=False \
  --backend=cpu \
  --enable-cudnn=False \
  --dtype=float32
