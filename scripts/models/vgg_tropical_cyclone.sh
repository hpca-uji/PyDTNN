#!/bin/bash

export OMP_NUM_THREADS=12
export MKL_NUM_THREADS=12
#export GOMP_CPU_AFFINITY="2 4 6 8 10 12 14 16 18 20"
#export GOMP_CPU_AFFINITY="0 1 2 3 4 5 6 7 8 9 10 11"
export OMP_PROC_BIND=True
export OMP_PLACES="{0:11}"
export PYTHONOPTIMIZE=2
#export OMP_DISPLAY_ENV=True
#export OMP_DISPLAY_AFFINITY=True
export PYTHONUNBUFFERED="True"
export OMP_MAX_ACTIVE_LEVELS=1

pydtnn-benchmark \
  --model=vgg_tropical_cyclone \
  --dataset=cyclones \
  --dataset-train-path=datasets/cifar10 \
  --test-as-validation \
  --no-flip-images \
  --no-crop-images \
  --batch-size=64 \
  --validation-split=0.2 \
  --steps-per-epoch=10 \
  --num-epochs=1 \
  --no-evaluate \
  --optimizer=sgd \
  --no-nesterov \
  --learning-rate=0.01 \
  --optimizerdecay=1e-4 \
  --optimizer-momentum=0.9 \
  --loss-func=negative_likelihood \
  --metrics=categorical_accuracy \
  --schedulers=warm_up,reduce_lr_on_plateau \
  --warm-up-epochs=5 \
  --early-stopping-metric=val_negative_likelihood \
  --early-stopping-patience=20 \
  --reduce-lr-on-plateau-metric=val_negative_likelihood \
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
  --backend="cpu;conv_2d:winograd" \
  --no-enable-best-of \
  --no-use-cudnn-auto-conv-algo \
  --no-use-gpudirect \
  --history-file="results/result_vgg3dobn.history" \
  --dtype=float32
