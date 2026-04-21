#!/bin/bash

export OMP_NUM_THREADS=16
export PYTHONOPTIMIZE=2
export PYTHONUNBUFFERED="True"
mpirun -np 2 \
  pydtnn-benchmark \
  --model=googlenet_cifar10 \
  --dataset=cifar10 \
  --dataset-path=datasets/cifar10 \
  --normalize=True \
  --normalize-offset=-0.472 \
  --normalize-scale=1 \
  --augment-flip=0.5 \
  --augment-crop=0.5 \
  --augment-crop-size=16 \
  --test-as-validation=True \
  --batch-size=128 \
  --validation-split=0.2 \
  --steps-per-epoch=0 \
  --num-epochs=400 \
  --evaluate=False \
  --optimizer=sgd \
  --optimizer-nesterov=False \
  --learning-rate=0.01 \
  --optimizer-momentum=0.9 \
  --loss-func=categorical_cross_entropy \
  --metrics=categorical_accuracy \
  --schedulers=warm_up,reduce_lr_on_plateau,early_stopping \
  --warm-up-epochs=5 \
  --early-stopping-metric=val_categorical_cross_entropy \
  --early-stopping-patience=40 \
  --reduce-lr-on-plateau-metric=val_categorical_cross_entropy \
  --reduce-lr-on-plateau-factor=0.5 \
  --reduce-lr-on-plateau-patience=15 \
  --reduce-lr-on-plateau-min-lr=0.0001 \
  --stop-at-loss-metric=val_categorical_accuracy \
  --stop-at-loss-threshold=70.0 \
  --parallel=sequential \
  --use-blocking-mpi=False \
  --tracing=False \
  --profile=False \
  --backend=gpu \
  --enable-cudnn=True \
  --enable-gpudirect=False \
  --history-file="results/result_googlenet.history" \
  --dtype=float32
