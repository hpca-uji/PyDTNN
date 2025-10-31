#!/bin/bash

export OMP_NUM_THREADS=4
export PYTHONOPTIMIZE=2
export PYTHONUNBUFFERED="True"
pydtnn-benchmark \
  --model=simplecnn \
  --dataset=archive \
  --dataset-path=datasets/mnist \
  --test-as-validation=False \
  --augment-flip=True \
  --batch-size=64 \
  --validation-split=0.2 \
  --num-epochs=50 \
  --evaluate=True \
  --optimizer=sgd \
  --learning-rate=0.01 \
  --loss-func=categorical_cross_entropy \
  --schedulers=warm_up,reduce_lr_every_nepochs \
  --reduce-lr-every-nepochs-factor=0.5 \
  --reduce-lr-every-nepochs-nepochs=30 \
  --reduce-lr-every-nepochs-min-lr=0.001 \
  --early-stopping-metric=val_categorical_cross_entropy \
  --early-stopping-patience=20 \
  --parallel=sequential \
  --tracing=False \
  --profile=False \
  --enable-gpu=False \
  --dtype=float32
