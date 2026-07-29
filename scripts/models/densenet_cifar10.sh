#!/bin/bash

export OMP_NUM_THREADS=16
export PYTHONOPTIMIZE=2
export PYTHONUNBUFFERED="True"

MPI_ARGS=()
export MPICH_UNBUFFERED_STDIO="true"
if mpirun --version | grep -q 'Open MPI) [5-9].'; then
  MPI_ARGS+=("--output=:raw")
fi

mpirun -np 1 "${MPI_ARGS[@]}" \
  pydtnn-benchmark \
  --model=densenet121_cifar10 \
  --dataset=cifar10 \
  --dataset-path=datasets/cifar10 \
  --input-normalize \
  --augment-horizontal-flip=0.5 \
  --augment-mask=0.5 \
  --augment-mask-size=16 \
  --test-as-validation \
  --batch-size=64 \
  --validation-split=0.2 \
  --steps-per-epoch=0 \
  --num-epochs=400 \
  --no-evaluate \
  --optimizer=sgd \
  --no-optimizer-nesterov \
  --learning-rate=0.01 \
  --optimizer-momentum=0.9 \
  --optimizer-decay=1e-4 \
  --loss-func=negative_log_likelihood \
  --metrics=categorical_accuracy \
  --schedulers=reduce_lr_on_plateau \
  --warm-up-epochs=5 \
  --early-stopping-metric=val_categorical_cross_entropy \
  --early-stopping-patience=40 \
  --reduce-lr-on-plateau-metric=val_categorical_cross_entropy \
  --reduce-lr-on-plateau-factor=0.1 \
  --reduce-lr-on-plateau-patience=15 \
  --reduce-lr-on-plateau-min-lr=0.0001 \
  --reduce-lr-every-nepochs-factor=0.1 \
  --reduce-lr-every-nepochs-nepochs=90 \
  --reduce-lr-every-nepochs-min-lr=0.001 \
  --stop-at-loss-metric=val_categorical_accuracy \
  --stop-at-loss-threshold=70.0 \
  --parallel-data \
  --use-blocking-mpi \
  --no-tracing \
  --no-profile \
  --backend=cpu \
  --no-use-cudnn \
  --no-use-gpudirect \
  --history-file="results/result_googlenet.history" \
  --dtype=float32
