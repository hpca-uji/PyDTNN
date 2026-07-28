#!/bin/bash

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
#export GOMP_CPU_AFFINITY="2 4 6 8 10 12 14 16"
export PYTHONOPTIMIZE=2
#export OMP_DISPLAY_ENV=True
#export OMP_DISPLAY_AFFINITY=True
export PYTHONUNBUFFERED="True"

pydtnn-benchmark \
  --model=vgg3dobn \
  --dataset=cifar10 \
  --dataset-path=datasets/cifar10 \
  --input-normalize \
  --input-normalize-offset=-0.472 \
  --input-normalize-scale=1 \
  --test-as-validation \
  --batch-size=64 \
  --validation-split=0.2 \
  --steps-per-epoch=0 \
  --num-epochs=10 \
  --no-evaluate \
  --optimizer=sgd \
  --no-optimizer-nesterov \
  --learning-rate=0.01 \
  --optimizer-decay=1e-4 \
  --optimizer-momentum=0.9 \
  --loss-func=categorical_cross_entropy \
  --metrics=categorical_accuracy \
  --schedulers=warm_up,reduce_lr_on_plateau \
  --warm-up-epochs=5 \
  --early-stopping-metric=val_categorical_cross_entropy \
  --early-stopping-patience=20 \
  --reduce-lr-on-plateau-metric=val_categorical_cross_entropy \
  --reduce-lr-on-plateau-factor=0.1 \
  --reduce-lr-on-plateau-patience=15 \
  --reduce-lr-on-plateau-min-lr=0.0001 \
  --reduce-lr-every-nepochs-factor=0.5 \
  --reduce-lr-every-nepochs-nepochs=50 \
  --reduce-lr-every-nepochs-min-lr=0.001 \
  --stop-at-loss-metric=val_categorical_accuracy \
  --stop-at-loss-threshold=70.0 \
  --no-parallel-data \
  --use-blocking-mpi \
  --no-tracing \
  --no-profile \
  --backend=cpu \
  --no-use-cudnn \
  --no-use-cudnn-auto-conv-algo \
  --no-use-gpudirect \
  --history-file="results/result_vgg3dobn.history" \
  --dtype=float32
