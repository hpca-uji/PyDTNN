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
  --no-test-as-validation \
  --no-augment-shuffle \
  --validation-split=0.2 \
  --num-epochs=10 \
  --steps-per-epoch=0 \
  --batch-size=64 \
  --no-evaluate \
  --optimizer=oktopk \
  --model-sync-freq=-1 \
  --no-initial-model-sync \
  --no-final-model-sync \
  --learning-rate=0.0001 \
  --optimizer-momentum=0.9 \
  --optimizer-decay=0.0005 \
  --optimizer-density=0.05 \
  --optimizer-tau=64 \
  --optimizer-tau-prime=16 \
  --oktopk-min-k=0 \
  --loss-func=negative_log_likelihood \
  --schedulers= \
  --parallel-data \
  --no-use-mpi-buffers \
  --use-blocking-mpi \
  --backend=cpu \
  --dtype=float32
