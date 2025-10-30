#!/bin/bash

export OMP_NUM_THREADS=1
export PYTHONOPTIMIZE=2
export PYTHONUNBUFFERED="True"
#mpirun --use-hwthread-cpus -np 16 \
#  --steps_per_epoch=10 \
pydtnn_benchmark \
  --model=vgg11_imagenet \
  --dataset=imagenet \
  --dataset_path=datasets/imagenet \
  --transform_crop=True \
  --transform_crop_perc=0.875 \
  --transform_resize=True \
  --transform_reize_size=227 \
  --normalize=True \
  --normalize_offset=-0.449 \
  --normalize_scale=3.537 \
  --use_synthetic_data=False \
  --tensor_format=NCHW \
  --batch_size=64 \
  --validation_split=0.2 \
  --num_epochs=10 \
  --evaluate=True \
  --optimizer=adam \
  --learning_rate=0.0001 \
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
