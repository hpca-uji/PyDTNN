#!/bin/bash

python3 -u benchmarks_CNN.py \
         --model=simplecnn \
         --dataset=mnist \
         --dataset_train_path=../datasets/mnist \
         --dataset_test_path=../datasets/mnist \
         --batch_size=64 \
         --validation_split=0.2 \
         --num_epochs=10 \
         --optimizer=Adam \
         --learning_rate=0.001 \
         --loss_func=accuracy_class,cross_entropy_class \
         --parallel=sequential \
         --blocking_mpi \
         --dtype=float32 --test_as_validation

         # 
         # --decay_rate=0.99 \
         # --epsilon=1e-08 \
         # --momentum=0.9 \ 
         # --steps_per_epoch=0 \
         # --evaluate=True \
         # --tracing=False \
         # --profile=False \