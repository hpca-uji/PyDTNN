#!/bin/bash

#set -x # Debugging flag
export PYTHONPATH=/home/dolzm/install/extrae-3.6.0/libexec:$PYTHONPATH
export EXTRAE_CONFIG_FILE=./extrae.xml
export MKL_NUM_THREADS=12
export EXTRAE_ON=1

EXTRAELIB=/home/dolzm/install/extrae-3.6.0/lib/libompitrace.so

NODETYPE=hexa
NUMPR=12
LASTH=`echo $NUMPR - 1 | bc`
HOSTS=$(for i in `seq 0 $LASTH`; do printf "%s%02d," ${NODETYPE} ${i}; done)
HOSTS=hexa00,hexa02,hexa03,hexa04,hexa05,hexa06,hexa07,hexa08,hexa09,hexa10,hexa11,hexa12

mpirun -genv LD_PRELOAD $EXTRAELIB -iface ib0 \
       -hosts $HOSTS -ppn 1 -np $NUMPR python -u benchmarks_CNN.py \
       --batch_size 64 --steps_per_epoch 5 --num_epochs 1 \
       --model alexnet --dataset imagenet --parallel data
       #--batch_size 64 --num_epochs 200 \
      # --model simplecnn --dataset mnist --parallel data
      # --batch_size 64 --num_epochs 1 \
