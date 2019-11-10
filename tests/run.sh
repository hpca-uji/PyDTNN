#!/bin/bash

#set -x # Debugging flag
export PYTHONPATH=/home/dolzm/install/extrae-3.6.0/libexec:$PYTHONPATH
export EXTRAE_CONFIG_FILE=./extrae.xml
export MKL_NUM_THREADS=16

EXTRAELIB=/home/dolzm/install/extrae-3.6.0/lib/libmpitrace.so

NODETYPE=hasw
NUMPR=8
LASTH=`echo $NUMPR - 1 | bc`
HOSTS=$(for i in `seq 0 $LASTH`; do printf "%s%02d," ${NODETYPE} ${i}; done)

mpirun -genv LD_PRELOAD $EXTRAELIB \
       -hosts $HOSTS -ppn 1 -np $NUMPR python -u benchmarks_CNN.py \
       --batch_size 64 --num_steps 5 --parallel data \
       --model alexnet --dataset imagenet
       #--model simplecnn --dataset mnist
