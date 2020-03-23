#!/bin/bash

python3 -u benchmarks_CNN.py \
         --model=vgg11 \
         --dataset=imagenet \
         --dataset_train_path=/scratch/imagenet/np/train \
         --dataset_test_path=/scratch/imagenet/np/validation \
         --batch_size=128 \
         --validation_split=0.0 \
         --steps_per_epoch=2 \
         --num_epochs=10 \
         --optimizer=SGDMomentum \
         --learning_rate=0.01 \
         --decay_rate=0.99 \
         --epsilon=1e-08 \
         --momentum=0.9 \
         --loss_func=categorical_accuracy,categorical_cross_entropy \
         --parallel=sequential \
         --blocking_mpi \
         --dtype=float32
