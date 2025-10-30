#!/bin/bash

export OMP_NUM_THREADS=16
export PYTHONOPTIMIZE=2
export PYTHONUNBUFFERED="True"
mpirun -np 2 \
  pydtnn_benchmark \
  --model=googlenet_imagenet \
  --dataset=synthetic \
  --synthetic_train_samples=1281167 \
  --synthetic_test_samples=50000 \
  --synthetic_input_shape=3,227,227 \
  --synthetic_output_shape=1000 \
  --test_as_validation=False \
  --augment_flip=True \
  --augment_crop=True \
  --augment_crop_size=16 \
  --test_as_validation=True \
  --batch_size=128 \
  --validation_split=0.2 \
  --steps_per_epoch=0 \
  --num_epochs=400 \
  --evaluate=False \
  --optimizer=sgd \
  --nesterov=False \
  --learning_rate=0.01 \
  --momentum=0.9 \
  --loss_func=categorical_cross_entropy \
  --metrics=categorical_accuracy \
  --schedulers=warm_up,reduce_lr_on_plateau,early_stopping \
  --warm_up_epochs=5 \
  --early_stopping_metric=val_categorical_cross_entropy \
  --early_stopping_patience=40 \
  --reduce_lr_on_plateau_metric=val_categorical_cross_entropy \
  --reduce_lr_on_plateau_factor=0.5 \
  --reduce_lr_on_plateau_patience=15 \
  --reduce_lr_on_plateau_min_lr=0.0001 \
  --stop_at_loss_metric=val_categorical_accuracy \
  --stop_at_loss_threshold=70.0 \
  --parallel=sequential \
  --use_blocking_mpi=False \
  --tracing=False \
  --profile=False \
  --enable_gpu=True \
  --enable_gpudirect=False \
  --history_file="results/result_googlenet.history" \
  --dtype=float32
