#!/bin/bash

export OMP_NUM_THREADS=1
mpirun -np 4 \
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
         --loss_func=categorical_accuracy,categorical_cross_entropy \
         --parallel=data \
         --dtype=float32 --evaluate --non_blocking_mpi
         #--test_as_validation
         #--profile
         #--enable_gpu
