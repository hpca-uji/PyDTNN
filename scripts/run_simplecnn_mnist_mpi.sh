#!/bin/bash

export OMP_NUM_THREADS=1
export PYTHONOPTIMIZE=2
export PYTHONUNBUFFERED="True"

MPI_ARGS=()
export MPICH_UNBUFFERED_STDIO="true"
if $(mpirun --version | grep -q 'Open MPI) [5-9].'); then
  MPI_ARGS+=("--output=:raw")
fi

mpirun -np 1 "${MPI_ARGS[@]}" \
  pydtnn-benchmark \
  --model=simplecnn \
  --dataset=mnist \
  --dataset-path=datasets/mnist \
  --test-as-validation=False \
  --augment-flip=True \
  --batch-size=64 \
  --validation-split=0.2 \
  --encryption= \
  --model-sync-freq=0 \
  --num-epochs=50 \
  --final-model-sync=False \
  --evaluate=True \
  --steps-per-epoch=0 \
  --optimizer=sgd \
  --learning-rate=0.01 \
  --loss-func=categorical_cross_entropy \
  --schedulers=warm_up,reduce_lr_every_nepochs \
  --reduce-lr-every-nepochs-factor=0.5 \
  --reduce-lr-every-nepochs-nepochs=30 \
  --reduce-lr-every-nepochs-min-lr=0.001 \
  --early-stopping-metric=val_categorical_cross_entropy \
  --early-stopping-patience=20 \
  --parallel=data \
  --use-blocking-mpi=True \
  --use-mpi-buffers=False \
  --tracing=False \
  --profile=True \
  --backend=cpu \
  --enable-cudnn=False \
  --dtype=float32
