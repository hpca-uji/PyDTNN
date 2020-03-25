#!/bin/bash

python3 -u benchmarks_CNN.py \
         --model=simplecnn \
         --dataset=mnist \
         --dataset_train_path=../datasets/mnist \
         --dataset_test_path=../datasets/mnist \
         --batch_size=64 \
         --validation_split=0.2 \
         --num_epochs=1 \
         --optimizer=Adam \
         --learning_rate=0.001 \
         --loss_func=categorical_accuracy,categorical_cross_entropy \
         --parallel=sequential \
         --dtype=float32 --evaluate

         #--test_as_validation
         #--profile
         #--enable_gpu
         #--evaluate
