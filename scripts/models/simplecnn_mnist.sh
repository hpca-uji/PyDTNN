#!/bin/bash

export OMP_NUM_THREADS=16
export PYTHONOPTIMIZE=2
export PYTHONUNBUFFERED="True"

pydtnn-benchmark \
  --model=simplecnn \
  --dataset=mnist \
  --dataset-path=datasets/mnist \
  --test-as-validation=False \
  --transform-crop=False \
  --transform-crop-perc=0.8 \
  --transform-resize=False \
  --transform-resize-size=16 \
  --augment-shuffle=True \
  --quantize=True \
  --batch-size=64 \
  --num-epochs=50 \
  --steps-per-epoch=0 \
  --validation-split=0.2 \
  --evaluate=False \
  --optimizer=sgd \
  --learning-rate=0.01 \
  --loss-func=categorical_cross_entropy \
  --schedulers=warm_up,reduce_lr_every_nepochs \
  --reduce-lr-every-nepochs-factor=0.5 \
  --reduce-lr-every-nepochs-nepochs=30 \
  --reduce-lr-every-nepochs-min-lr=0.001 \
  --early-stopping-metric=val_categorical_cross_entropy \
  --early-stopping-patience=20 \
  --parallel-data=False \
  --tracing=False \
  --profile=False \
  --backend=cpu,gemm \
  --enable-cudnn=False \
  --dtype=float32
