#!/bin/bash

export OMP_NUM_THREADS=4
python3 -u tests/benchmarks_CNN.py \
         --model=alexnet_cifar10 \
         --dataset=cifar10 \
         --dataset_train_path=/home/barrachi/Descargas/data/cifar-10-batches-bin/ \
         --dataset_test_path=/home/barrachi/Descargas/data/cifar-10-batches-bin/ \
         --test_as_validation=False \
         --batch_size=64 \
         --validation_split=0.2 \
         --steps_per_epoch=0 \
         --num_epochs=30 \
         --evaluate=True \
         --optimizer=sgd \
         --learning_rate=0.01 \
         --momentum=0.9 \
         --loss=categorical_cross_entropy \
         --metrics=categorical_accuracy \
         --lr_schedulers=early_stopping,reduce_lr_on_plateau \
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
         --enable_gpu=False \
         --dtype=float32 \
         --enable_conv_gemm=False
