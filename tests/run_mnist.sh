#!/bin/bash

python3 -u benchmarks_CNN.py \
       --batch_size 64 --num_epochs 10 \
       --model simplecnn --dataset mnist
