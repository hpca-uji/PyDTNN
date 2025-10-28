#!/bin/bash

export OMP_NUM_THREADS=16
export PYTHONOPTIMIZE=2
export PYTHONUNBUFFERED="True"

pydtnn_benchmark \
  --model=simplecnn \
  --dataset=mnist \
  --dataset_path=datasets/mnist \
  --test_as_validation=False \
  --transform_crop=False \
  --transform_crop_perc=0.8 \
  --transform_resize=False \
  --transform_resize_size=16 \
  --augment_crop=False \
  --augment_flip=False \
  --augment_shuffle=True \
  --batch_size=64 \
  --num_epochs=50 \
  --steps_per_epoch=0 \
  --validation_split=0.2 \
  --evaluate=False \
  --optimizer=sgd \
  --learning_rate=0.01 \
  --loss_func=categorical_cross_entropy \
  --schedulers=warm_up,reduce_lr_every_nepochs \
  --reduce_lr_every_nepochs_factor=0.5 \
  --reduce_lr_every_nepochs_nepochs=30 \
  --reduce_lr_every_nepochs_min_lr=0.001 \
  --early_stopping_metric=val_categorical_cross_entropy \
  --early_stopping_patience=20 \
  --parallel=sequential \
  --tracing=False \
  --profile=False \
  --enable_gpu=False \
  --dtype=float32
