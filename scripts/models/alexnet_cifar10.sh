#!/bin/bash

export PYTHONOPTIMIZE=2
export PYTHONUNBUFFERED="True"

pydtnn-benchmark \
  --model=alexnet_cifar10 \
  --dataset=cifar10 \
  --dataset-path=datasets/cifar10 \
  --augment-normalize \
  --augment-normalize-offset=-0.472 \
  --augment-normalize-scale=1 \
  --test-as-validation \
  --batch-size=64 \
  --validation-split=0.2 \
  --steps-per-epoch=0 \
  --num-epochs=30 \
  --evaluate \
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
  --no-parallel-data \
  --use-blocking-mpi \
  --no-tracing \
  --no-profile \
  --backend=cpu \
  --no-use-cudnn \
  --dtype=float32
