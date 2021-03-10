#!/bin/bash

export OMP_NUM_THREADS=16
#mpirun -np 2 \
python3.8 -u benchmarks_CNN.py \
         --model=vgg3dobn \
         --dataset=cifar10 \
         --dataset_train_path=/Users/mdolz/Downloads/cifar-10-batches-bin \
         --dataset_test_path=/Users/mdolz/Downloads/cifar-10-batches-bin \
         --test_as_validation=True \
         --flip_images=True \
         --crop_images=True \
         --batch_size=64 \
         --validation_split=0.2 \
         --steps_per_epoch=0 \
         --num_epochs=400 \
         --evaluate=False \
         --optimizer=sgd \
         --nesterov=False \
         --learning_rate=0.01 \
         --decay=1e-4 \
         --momentum=0.9 \
         --loss_func=categorical_cross_entropy \
         --metrics=categorical_accuracy \
         --lr_schedulers=warm_up,reduce_lr_on_plateau \
         --warm_up_epochs=5 \
         --early_stopping_metric=val_categorical_cross_entropy \
         --early_stopping_patience=20 \
         --reduce_lr_on_plateau_metric=val_categorical_cross_entropy \
         --reduce_lr_on_plateau_factor=0.1 \
         --reduce_lr_on_plateau_patience=15 \
         --reduce_lr_on_plateau_min_lr=0.0001 \
         --reduce_lr_every_nepochs_factor=0.5 \
         --reduce_lr_every_nepochs_nepochs=50 \
         --reduce_lr_every_nepochs_min_lr=0.001 \
         --stop_at_loss_metric=val_categorical_accuracy \
         --stop_at_loss_threshold=70.0 \
         --parallel=sequential \
         --non_blocking_mpi=False \
         --tracing=False \
         --profile=False \
         --enable_gpu=True \
         --enable_gpudirect=False \
         --history_file="results/result_vgg3dobn.history" \
         --dtype=float32
