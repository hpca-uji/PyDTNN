#!/bin/bash

export OMP_NUM_THREADS=4
python3 -u benchmarks_CNN.py \
         --model=vgg3dobn \
         --dataset=cifar10 \
         --dataset_train_path=/Users/mdolz/Downloads/cifar-10-batches-bin \
         --dataset_test_path=/Users/mdolz/Downloads/cifar-10-batches-bin \
         --test_as_validation=True \
         --batch_size=64 \
         --validation_split=0.2 \
         --steps_per_epoch=0 \
         --num_epochs=200 \
         --evaluate=False \
         --optimizer=sgd \
         --learning_rate=0.001 \
         --momentum=0.9 \
         --loss_func=categorical_cross_entropy \
         --metrics=categorical_accuracy \
         --lr_schedulers=warm_up,stop_at_loss \
         --warm_up_epochs=5 \
         --early_stopping_metric=val_categorical_cross_entropy \
         --early_stopping_patience=20 \
         --reduce_lr_on_plateau_metric=val_categorical_cross_entropy \
         --reduce_lr_on_plateau_factor=0.1 \
         --reduce_lr_on_plateau_patience=5 \
         --reduce_lr_on_plateau_min_lr=0 \
         --stop_at_loss_metric=val_categorical_accuracy \
         --stop_at_loss_threshold=70.0 \
         --parallel=data \
         --non_blocking_mpi=False \
         --tracing=False \
         --profile=False \
         --enable_gpu=False \
         --history_file="results/result_vgg3dobn.history" \
         --dtype=float32
