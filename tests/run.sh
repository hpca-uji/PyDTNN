#!/bin/bash

#set -x # Debugging flag
export PYTHONPATH=/home/dolzm/install/extrae-3.6.0/libexec:$PYTHONPATH
export EXTRAE_CONFIG_FILE=./extrae.xml
export MKL_NUM_THREADS=12
export EXTRAE_ON=1

EXTRAELIB=/home/dolzm/install/extrae-3.6.0/lib/libptmpitrace.so

NODETYPE=hasw
NUMPR=12
LASTH=`echo $NUMPR - 1 | bc`
HOSTS=$(for i in `seq 0 $LASTH`; do printf "%s%02d," ${NODETYPE} ${i}; done)

mpirun -genv LD_PRELOAD $EXTRAELIB \
       -hosts $HOSTS -ppn 1 -np $NUMPR python -u benchmarks_CNN.py \
       --batch_size 64 --steps_per_epoch 5 --num_epochs 1 \
       --model vgg11 --dataset imagenet --parallel data
       #--batch_size 64 --num_epochs 200 \
      # --model simplecnn --dataset mnist --parallel data
      # --batch_size 64 --num_epochs 1 \
