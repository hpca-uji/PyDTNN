#!/bin/bash

export OMP_NUM_THREADS=4
export PYTHONOPTIMIZE=2
export PYTHONUNBUFFERED="True"
pydtnn-benchmark \
  --model=simplemlp \
  --dataset=mnist \
  --dataset-path=datasets/mnist \
  --test-as-validation=True \
  --batch-size=256 \
  --validation-split=0.2 \
  --steps-per-epoch=0 \
  --num-epochs=300 \
  --evaluate=False \
  --optimizer=sgd \
  --learning-rate=0.1 \
  --optimizer-momentum=0.0 \
  --loss-func=categorical_cross_entropy \
  --metrics=categorical_accuracy \
  --schedulers="" \
  --warm-up-epochs=5 \
  --early-stopping-metric=val_categorical_cross_entropy \
  --early-stopping-patience=10 \
  --reduce-lr-on-plateau-metric=val_categorical_cross_entropy \
  --reduce-lr-on-plateau-factor=0.1 \
  --reduce-lr-on-plateau-patience=5 \
  --reduce-lr-on-plateau-min-lr=0 \
  --parallel=sequential \
  --use-blocking-mpi=True \
  --tracing=False \
  --profile=False \
  --enable-cudnn=False \
  --dtype=float32
