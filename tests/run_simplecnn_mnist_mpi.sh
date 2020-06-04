#!/bin/bash

export OMP_NUM_THREADS=1
mpirun -np 4 \
   python3 -u benchmarks_CNN.py \
         --model=simplecnn \
         --dataset=mnist \
         --dataset_train_path=../datasets/mnist \
         --dataset_test_path=../datasets/mnist \
         --test_as_validation=True \
         --batch_size=64 \
         --validation_split=0.2 \
         --steps_per_epoch=0 \
         --num_epochs=50 \
         --evaluate=True \
         --optimizer=rmsprop \
         --learning_rate=0.01 \
         --momentum=0.9 \
         --loss_func=categorical_accuracy,categorical_cross_entropy \
         --lr_schedulers=early_stopping,reduce_lr_on_plateau \
         --warm_up_epochs=5 \
         --early_stopping_metric=val_categorical_cross_entropy \
         --early_stopping_patience=10 \
         --reduce_lr_on_plateau_metric=val_categorical_cross_entropy \
         --reduce_lr_on_plateau_factor=0.1 \
         --reduce_lr_on_plateau_patience=5 \
         --reduce_lr_on_plateau_min_lr=0 \
         --parallel=data \
         --non_blocking_mpi=False \
         --tracing=False \
         --profile=False \
         --enable_gpu=False \
         --dtype=float32

