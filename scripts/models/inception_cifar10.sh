#!/bin/bash

export OMP_NUM_THREADS=16
export PYTHONOPTIMIZE=2
export PYTHONUNBUFFERED="True"

MPI_ARGS=()
export MPICH_UNBUFFERED_STDIO="true"
if mpirun --version | grep -q 'Open MPI) [5-9].'; then
  MPI_ARGS+=("--output=:raw")
fi

mpirun -np 2 "${MPI_ARGS[@]}" \
  pydtnn-benchmark \
  --model=googlenet_cifar10 \
  --dataset=cifar10 \
  --dataset-path=datasets/cifar10 \
  --augment-normalize=True \
  --augment-normalize-offset=-0.472 \
  --augment-normalize-scale=1 \
  --augment-horizontal-flip=0.5 \
  --augment-mask=0.5 \
  --augment-mask-size=16 \
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
  --parallel-data=False \
  --use-blocking-mpi=False \
  --tracing=False \
  --profile=False \
  --backend=gpu \
  --enable-cudnn=True \
  --enable-gpudirect=False \
  --history-file="results/result_googlenet.history" \
  --dtype=float32
