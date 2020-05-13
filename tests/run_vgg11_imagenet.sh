#!/bin/bash

python3 -u benchmarks_CNN.py \
         --model=vgg11 \
         --dataset=imagenet \
         --dataset_train_path=/scratch/imagenet/np/train \
         --dataset_test_path=/scratch/imagenet/np/validation \
         --batch_size=64 \
         --validation_split=0.2 \
         --steps_per_epoch=0 \
         --num_epochs=30 \
         --evaluate=True \
         --optimizer=Adam \
         --learning_rate=0.0001 \
         --momentum=0.9 \
         --loss_func=categorical_accuracy,categorical_cross_entropy \
         --lr_schedulers=early_stopping,reduce_lr_on_plateau \
         --warm_up_batches=500 \
         --early_stopping_metric=val_categorical_cross_entropy \
         --early_stopping_patience=10 \
         --reduce_lr_on_plateau_metric=val_categorical_cross_entropy \
         --reduce_lr_on_plateau_factor=0.1 \
         --reduce_lr_on_plateau_patience=5 \
         --reduce_lr_on_plateau_min_lr=0 \
         --parallel=sequential \
         --non_blocking_mpi=False \
         --tracing=False \
         --profile=False \
         --enable_gpu=False \
         --dtype=float32
