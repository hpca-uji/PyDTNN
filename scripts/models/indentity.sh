#!/bin/bash

export OMP_NUM_THREADS=16
export PYTHONOPTIMIZE=2
export PYTHONUNBUFFERED="True"

pydtnn-benchmark \
  --model=identity \
  --dataset=synthetic \
  --synthetic-train-samples=640 \
  --synthetic-test-samples=640 \
  --synthetic-input-shape=1,4,4 \
  --synthetic-output-shape=16 \
  --no-test-as-validation \
  --augment-shuffle \
  --batch-size=64 \
  --num-epochs=10 \
  --steps-per-epoch=0 \
  --validation-split=0.2 \
  --evaluate \
  --optimizer=sgd \
  --learning-rate=0.01 \
  --loss-func=negative_likelihood \
  --schedulers= \
  --no-parallel-data \
  --no-tracing \
  --no-profile \
  --backend=cpu \
  --dtype=float32
