#!/bin/bash

export OMP_NUM_THREADS=16
export PYTHONOPTIMIZE=2
export PYTHONUNBUFFERED="True"

MPI_ARGS=()
export MPICH_UNBUFFERED_STDIO="true"
if mpirun --version | grep -q 'Open MPI) [5-9].'; then
  MPI_ARGS+=("--output=:raw")
fi

mpirun -np 4 "${MPI_ARGS[@]}" \
  pydtnn-benchmark \
  --model=resnet \
  --dataset=chestxray \
  --dataset-path=datasets/chest_xray \
  --no-test-as-validation \
  --batch-size=10 \
  --validation-split=0.2 \
  --steps-per-epoch=0 \
  --num-epochs=15 \
  --no-evaluate \
  --optimizer=sgd \
  --optimizer-nesterov \
  --learning-rate=0.1 \
  --optimizer-momentum=0.9 \
  --loss-func=negative_log_likelihood \
  --metrics=categorical_accuracy \
  --schedulers=warm_up,reduce_lr_on_plateau,early_stopping \
  --warm-up-epochs=5 \
  --early-stopping-metric=val_categorical_cross_entropy \
  --early-stopping-patience=40 \
  --reduce-lr-on-plateau-metric=val_categorical_cross_entropy \
  --reduce-lr-on-plateau-factor=0.5 \
  --reduce-lr-on-plateau-patience=15 \
  --reduce-lr-on-plateau-min-lr=0.00001 \
  --reduce-lr-every-nepochs-nepochs=30 \
  --reduce-lr-every-nepochs-min-lr=0.00001 \
  --reduce-lr-every-nepochs-factor=0.1 \
  --stop-at-loss-metric=val_categorical_accuracy \
  --stop-at-loss-threshold=70.0 \
  --parallel-data \
  --no-use-blocking-mpi \
  --no-tracing \
  --no-profile \
  --backend=cpu \
  --no-use-cudnn \
  --no-use-gpudirect \
  --dtype=float32 \
  --input-scale \
  --input-scale-size=300
