#!/bin/bash

export OMP_NUM_THREADS=1
export PYTHONOPTIMIZE=2
export PYTHONUNBUFFERED="True"

MPI_ARGS=()
export MPICH_UNBUFFERED_STDIO="true"
if mpirun --version | grep -q 'Open MPI) [5-9].'; then
  MPI_ARGS+=("--output=:raw")
fi

mpirun -np 1 --oversubscribe "${MPI_ARGS[@]}" \
  pydtnn-benchmark \
  --model=simplecnn \
  --dataset=mnist \
  --dataset-path=datasets/mnist \
  --test-as-validation=False \
  --augment-shuffle=False \
  --batch-size=64 \
  --num-epochs=10 \
  --steps-per-epoch=0 \
  --validation-split=0.2 \
  --evaluate=True \
  --model-sync-freq=0 \
  --initial-model-sync=False \
  --final-model-sync=False \
  --optimizer=sgd \
  --learning-rate=0.001 \
  --optimizer-momentum=0.9 \
  --optimizer-decay=0.0005 \
  --optimizer-density=0.05 \
  --optimizer-tau=32 \
  --optimizer-tau-prime=64 \
  --oktopk-min-k=0 \
  --loss-func=categorical_cross_entropy \
  --schedulers=warm_up,reduce_lr_every_nepochs,early_stopping \
  --reduce-lr-every-nepochs-factor=0.001 \
  --reduce-lr-every-nepochs-nepochs=100 \
  --reduce-lr-every-nepochs-min-lr=0.001 \
  --early-stopping-metric=val_categorical_cross_entropy \
  --early-stopping-patience=100 \
  --parallel-data=True \
  --tracing=False \
  --profile=False \
  --backend=cpu \
  --enable-cudnn=False \
  --dtype=float32
