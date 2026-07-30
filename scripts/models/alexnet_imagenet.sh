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
  --no-test-as-validation \
  --batch-size=64 \
  --validation-split=0.2 \
  --steps-per-epoch=0 \
  --num-epochs=30 \
  --evaluate \
  --optimizer=sgd \
  --learning-rate=0.01 \
  --optimizer-momentum=0.9 \
  --loss-func=negative_log_likelihood \
  --metrics=categorical_accuracy \
  --schedulers=early_stopping,reduce_lr_on_plateau \
  --warm-up-epochs=5 \
  --early-stopping-metric=val_negative_log_likelihood \
  --early-stopping-patience=10 \
  --reduce-lr-on-plateau-metric=val_negative_log_likelihood \
  --reduce-lr-on-plateau-factor=0.1 \
  --reduce-lr-on-plateau-patience=5 \
  --reduce-lr-on-plateau-min-lr=0 \
  --no-parallel-data \
  --use-blocking-mpi \
  --no-tracing \
  --no-profile \
  --backend=cpu \
  --no-use-cudnn \
  --dtype=float32
