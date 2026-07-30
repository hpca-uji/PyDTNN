#!/bin/bash

export PYTHONOPTIMIZE=0
export PYTHONUNBUFFERED="True"

pydtnn-benchmark \
  --model=iwslt \
  --dataset=iwslt \
  --dataset-lang=en \
  --dataset-lang2=de \
  --dataset-path=datasets/iwslt/iwslt.txt \
  --test-as-validation \
  --batch-size=32 \
  --validation-split=0.2 \
  --steps-per-epoch=0 \
  --num-epochs=3 \
  --no-evaluate \
  --optimizer=adam \
  --learning-rate=0.01 \
  --optimizer-momentum=0.9 \
  --loss-func=kl_divergence \
  --metrics=kl_divergence_metric \
  --schedulers=early_stopping,reduce_lr_on_plateau \
  --warm-up-epochs=5 \
  --early-stopping-metric=val_negative_log_likelihood \
  --early-stopping-patience=10 \
  --reduce-lr-on-plateau-metric=val_negative_log_likelihood \
  --reduce-lr-on-plateau-factor=0.1 \
  --reduce-lr-on-plateau-patience=5 \
  --reduce-lr-on-plateau-min-lr=0 \
  --no-parallel-data \
  --use-blocking-mpi \
  --no-tracing \
  --no-profile \
  --backend=cpu \
  --dtype=float32

