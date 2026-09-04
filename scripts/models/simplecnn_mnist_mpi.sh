#!/bin/bash

export OMP_NUM_THREADS=1
export PYTHONOPTIMIZE=2
export PYTHONUNBUFFERED="True"

MPI_ARGS=()
export MPICH_UNBUFFERED_STDIO="true"
if mpirun --version | grep -q 'Open MPI) [5-9].'; then
  MPI_ARGS+=("--output=:raw")
fi

mpirun -np 8 "${MPI_ARGS[@]}" \
  pydtnn-benchmark \
  --model=simplecnn \
  --dataset=mnist \
  --dataset-path=datasets/mnist \
  --no-test-as-validation \
  --batch-size=64 \
  --validation-split=0.2 \
  --encryption= \
  --model-sync-freq=0 \
  --no-final-model-sync \
  --steps-per-epoch=0 \
  --num-epochs=10 \
  --evaluate \
  --optimizer=sgd \
  --learning-rate=0.01 \
  --loss-func=negative_likelihood \
  --schedulers=warm_up,reduce_lr_every_nepochs \
  --reduce-lr-every-nepochs-factor=0.5 \
  --reduce-lr-every-nepochs-nepochs=30 \
  --reduce-lr-every-nepochs-min-lr=0.001 \
  --early-stopping-metric=val_negative_likelihood \
  --early-stopping-patience=20 \
  --parallel-data \
  --use-blocking-mpi \
  --no-use-mpi-buffers \
  --no-tracing \
  --no-profile \
  --backend=cpu \
  --dtype=float32
