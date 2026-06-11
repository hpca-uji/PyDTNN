#!/bin/bash

export OMP_NUM_THREADS=16
export PYTHONOPTIMIZE=2
export PYTHONUNBUFFERED="True"

pydtnn-benchmark \
  --model=resnet10 \
  --dataset=cifar10 \
  --dataset-path=datasets/cifar10 \
  --test-as-validation=False \
  --augment-crop=False \
  --augment-crop-perc=0.8 \
  --augment-scale=False \
  --augment-scale-size=16 \
  --augment-shuffle=True \
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
  --backend=cpu \
  --enable-cudnn=False \
  --dtype=float32
