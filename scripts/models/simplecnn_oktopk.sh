#!/bin/bash

export OMP_NUM_THREADS=1
export PYTHONOPTIMIZE=2
export PYTHONUNBUFFERED="True"

MPI_ARGS=()
export MPICH_UNBUFFERED_STDIO="true"
if mpirun --version | grep -q 'Open MPI) [5-9].'; then
  MPI_ARGS+=("--output=:raw")
fi

mpirun -np 4 "${MPI_ARGS[@]}" \
  pydtnn-benchmark \
  --model=simplecnn \
  --dataset=mnist \
  --dataset-path=datasets/mnist \
  --test-as-validation=False \
  --augment-crop=False \
  --augment-crop-perc=0.8 \
  --augment-scale=False \
  --augment-scale-size=16 \
  --augment-shuffle=True \
  --batch-size=64 \
  --num-epochs=50 \
  --steps-per-epoch=0 \
  --validation-split=0.2 \
  --evaluate=False \
  --optimizer=oktopk \
  --model-sync-freq=-1 \
  --learning-rate=0.01 \
  --loss-func=categorical_cross_entropy \
  --schedulers=warm_up,reduce_lr_every_nepochs \
  --reduce-lr-every-nepochs-factor=0.5 \
  --reduce-lr-every-nepochs-nepochs=30 \
  --reduce-lr-every-nepochs-min-lr=0.001 \
  --early-stopping-metric=val_categorical_cross_entropy \
  --early-stopping-patience=20 \
  --parallel-data=True \
  --tracing=False \
  --profile=False \
  --backend=cpu \
  --backend=cpu \
  --enable-cudnn=False \
  --dtype=float32
