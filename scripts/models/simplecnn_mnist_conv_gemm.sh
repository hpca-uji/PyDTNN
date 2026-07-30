#!/bin/bash

export OMP_NUM_THREADS=4
export PYTHONOPTIMIZE=2
export PYTHONUNBUFFERED="True"

pydtnn-benchmark \
  --model=simplecnn \
  --dataset=mnist \
  --dataset-path=datasets/mnist \
  --no-test-as-validation \
  --augment-horizontal-flip=0.5 \
  --batch-size=64 \
  --validation-split=0.2 \
  --num-epochs=50 \
  --evaluate \
  --optimizer=adam \
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
  --backend="cpu;conv_2d:gemm" \
  --dtype=float32
