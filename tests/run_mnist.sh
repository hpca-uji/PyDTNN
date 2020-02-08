#!/bin/bash

python -u benchmarks_CNN.py \
       --batch_size 64 --num_epochs 10 \
       --model simplemlp --dataset mnist
