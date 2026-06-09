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
  --augment-normalize=True \
  --augment-normalize-offset=-0.472 \
  --augment-normalize-scale=1 \
  --augment-horizontal-flip=0.5 \
  --augment-mask=0.5 \
  --augment-mask-size=16 \
  --test-as-validation=True \
  --batch-size=64 \
  --validation-split=0.2 \
  --steps-per-epoch=0 \
  --num-epochs=400 \
  --evaluate=False \
  --optimizer=sgd \
  --optimizer-nesterov=False \
  --learning-rate=0.01 \
  --optimizer-momentum=0.9 \
  --optimizer-decay=1e-4 \
  --loss-func=categorical_cross_entropy \
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
  --parallel-data=True \
  --use-blocking-mpi=True \
  --tracing=False \
  --profile=False \
  --backend=cpu \
  --enable-cudnn=False \
  --enable-gpudirect=False \
  --history-file="results/result_googlenet.history" \
  --dtype=float32
