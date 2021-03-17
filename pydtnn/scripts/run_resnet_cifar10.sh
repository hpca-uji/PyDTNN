#!/bin/bash

export OMP_NUM_THREADS=16
export PYTHONOPTIMIZE=2
export PYTHONUNBUFFERED="True"
mpirun -np 2 \
  pydtnn_benchmark \
  --model=resnet18_cifar10 \
  --dataset=cifar10 \
  --dataset_train_path=/scratch/cifar-10/cifar-10-batches-bin \
  --dataset_test_path=/scratch/cifar-10/cifar-10-batches-bin \
  --flip_images=True \
  --crop_images=True \
  --crop_images_size=16 \
  --test_as_validation=True \
  --batch_size=64 \
  --validation_split=0.2 \
  --steps_per_epoch=0 \
  --num_epochs=400 \
  --evaluate=False \
  --optimizer=sgd \
  --nesterov=False \
  --learning_rate=0.1 \
  --momentum=0.9 \
  --loss_func=categorical_cross_entropy \
  --metrics=categorical_accuracy \
  --lr_schedulers=warm_up,reduce_lr_on_plateau,early_stopping \
  --warm_up_epochs=5 \
  --early_stopping_metric=val_categorical_cross_entropy \
  --early_stopping_patience=40 \
  --reduce_lr_on_plateau_metric=val_categorical_cross_entropy \
  --reduce_lr_on_plateau_factor=0.5 \
  --reduce_lr_on_plateau_patience=15 \
  --reduce_lr_on_plateau_min_lr=0.00001 \
  --reduce_lr_every_nepochs_nepochs=30 \
  --reduce_lr_every_nepochs_min_lr=0.00001 \
  --reduce_lr_every_nepochs_factor=0.1 \
  --stop_at_loss_metric=val_categorical_accuracy \
  --stop_at_loss_threshold=70.0 \
  --parallel=sequential \
  --non_blocking_mpi=False \
  --tracing=False \
  --profile=False \
  --enable_gpu=True \
  --enable_gpudirect=False \
  --history_file="results/result_resnet.history" \
  --dtype=float32
