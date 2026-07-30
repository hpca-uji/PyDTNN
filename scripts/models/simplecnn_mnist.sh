#!/bin/bash

export OMP_NUM_THREADS=16
export PYTHONOPTIMIZE=2
export PYTHONUNBUFFERED="True"

pydtnn-benchmark \
  --model=simplecnn \
  --dataset=mnist \
  --dataset-path=datasets/mnist \
  --no-test-as-validation \
  --no-input-crop \
  --input-crop-perc=0.8 \
  --no-input-scale \
  --input-scale-size=16 \
  --augment-shuffle \
  --batch-size=64 \
  --num-epochs=50 \
  --steps-per-epoch=0 \
  --validation-split=0.2 \
  --no-evaluate \
  --optimizer=sgd \
  --learning-rate=0.01 \
  --loss-func=negative_log_likelihood \
  --schedulers=warm_up,reduce_lr_every_nepochs \
  --reduce-lr-every-nepochs-factor=0.5 \
  --reduce-lr-every-nepochs-nepochs=30 \
  --reduce-lr-every-nepochs-min-lr=0.001 \
  --early-stopping-metric=val_negative_log_likelihood \
  --early-stopping-patience=20 \
  --no-parallel-data \
  --no-tracing \
  --no-profile \
  --backend=cpu \
  --dtype=float32
