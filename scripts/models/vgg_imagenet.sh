#!/bin/bash

export OMP_NUM_THREADS=1
export PYTHONOPTIMIZE=2
export PYTHONUNBUFFERED="True"

pydtnn-benchmark \
  --model=vgg11_imagenet \
  --dataset=imagenet \
  --dataset-path=datasets/imagenet \
  --augment-crop=True \
  --augment-crop-perc=0.875 \
  --augment-scale=True \
  --augment-scale-size=227 \
  --augment-normalize=True \
  --augment-normalize-offset=-0.449 \
  --augment-normalize-scale=3.537 \
  --use-synthetic-data=False \
  --tensor-format=nchw \
  --batch-size=64 \
  --validation-split=0.2 \
  --num-epochs=10 \
  --evaluate=True \
  --optimizer=adam \
  --learning-rate=0.0001 \
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
