#!/bin/bash

export OMP_NUM_THREADS=4
python3 -u benchmarks_CNN.py \
         --model=simplemlp \
         --dataset=mnist \
         --dataset_train_path=../datasets/mnist \
         --dataset_test_path=../datasets/mnist \
         --test_as_validation=True \
         --batch_size=256 \
         --validation_split=0.2 \
         --steps_per_epoch=0 \
         --num_epochs=300 \
         --evaluate=False \
         --optimizer=sgd \
         --learning_rate=0.1 \
         --momentum=0.0 \
         --loss=categorical_cross_entropy \
         --metrics=categorical_accuracy \
         --lr_schedulers="" \
         --warm_up_epochs=5 \
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
         --enable_gpu=True \
         --dtype=float32

