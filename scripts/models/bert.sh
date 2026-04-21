#!/bin/bash

export PYTHONOPTIMIZE=2
export PYTHONUNBUFFERED="True"
pydtnn-benchmark \
  --model=bert \
  --dataset=mask_lang \
  --dataset-lang=en \
  --dataset-path=datasets/iwslt/wiki_split.txt \
  --test-as-validation=True \
  --batch-size=8 \
  --validation-split=0.2 \
  --num-epochs=10 \
  --evaluate=False \
  --optimizer=sgd \
  --learning-rate=0.0001 \
  --optimizer-momentum=0.9 \
  --loss-func=kl_divergence \
  --parallel-data=False \
  --use-blocking-mpi=True \
  --tracing=False \
  --profile=False \
  --backend=cpu \
  --enable-cudnn=False \
  --dtype=float32
