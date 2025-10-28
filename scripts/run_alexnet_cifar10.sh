#!/bin/bash

export PYTHONOPTIMIZE=2
export PYTHONUNBUFFERED="True"
pydtnn_benchmark \
  --model=alexnet_cifar10 \
  --dataset=cifar10 \
  --dataset_path=datasets/cifar10 \
  --normalize=True \
  --normalize_offset=-0.472 \
  --normalize_scale=1 \
  --test_as_validation=True \
  --batch_size=64 \
  --validation_split=0.2 \
  --steps_per_epoch=0 \
  --num_epochs=30 \
  --evaluate=True \
  --optimizer=sgd \
  --learning_rate=0.01 \
  --momentum=0.9 \
  --loss_func=categorical_cross_entropy \
  --metrics=categorical_accuracy \
  --schedulers=early_stopping,reduce_lr_on_plateau \
  --warm_up_epochs=5 \
  --early_stopping_metric=val_categorical_cross_entropy \
  --early_stopping_patience=10 \
  --reduce_lr_on_plateau_metric=val_categorical_cross_entropy \
  --reduce_lr_on_plateau_factor=0.1 \
  --reduce_lr_on_plateau_patience=5 \
  --reduce_lr_on_plateau_min_lr=0 \
  --parallel=sequential \
  --use_blocking_mpi=True \
  --tracing=False \
  --profile=False \
  --enable_gpu=False \
  --dtype=float32
