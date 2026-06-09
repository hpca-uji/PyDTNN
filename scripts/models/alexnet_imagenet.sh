#!/bin/bash

export PYTHONOPTIMIZE=2
export PYTHONUNBUFFERED="True"

pydtnn-benchmark \
  --model=alexnet_imagenet \
  --dataset=synthetic \
  --synthetic-train-samples=1281167 \
  --synthetic-test-samples=50000 \
  --synthetic-input-shape=3,227,227 \
  --synthetic-output-shape=1000 \
  --test-as-validation=False \
  --batch-size=64 \
  --validation-split=0.2 \
  --steps-per-epoch=0 \
  --num-epochs=30 \
  --evaluate=True \
  --optimizer=sgd \
  --learning-rate=0.01 \
  --optimizer-momentum=0.9 \
  --loss-func=categorical_cross_entropy \
  --metrics=categorical_accuracy \
  --schedulers=early_stopping,reduce_lr_on_plateau \
  --warm-up-epochs=5 \
  --early-stopping-metric=val_categorical_cross_entropy \
  --early-stopping-patience=10 \
  --reduce-lr-on-plateau-metric=val_categorical_cross_entropy \
  --reduce-lr-on-plateau-factor=0.1 \
  --reduce-lr-on-plateau-patience=5 \
  --reduce-lr-on-plateau-min-lr=0 \
  --parallel-data=False \
  --use-blocking-mpi=True \
  --tracing=False \
  --profile=False \
  --backend=cpu \
  --enable-cudnn=False \
  --dtype=float32
