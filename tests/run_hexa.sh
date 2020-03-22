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
#HOSTS=hexa00,hexa02,hexa03,hexa04,hexa05,hexa06,hexa07,hexa08,hexa09,hexa10,hexa11,hexa12

mpirun -genv LD_PRELOAD $EXTRAELIB -iface ib0 \
       -hosts $HOSTS -ppn 1 -np $NUMPR \
       python3 -u benchmarks_CNN.py \
          --model=alexnet \
          --dataset=imagenet \
          --dataset_train_path=/scratch/imagenet/np/train \
          --dataset_test_path=/scratch/imagenet/np/validation \
          --batch_size=64 \
          --validation_split=0.2 \
          --num_epochs=10 \
          --optimizer=SGDMomentum \
          --learning_rate=0.01 \
          --loss_func=accuracy,categorical_cross_entropy \
          --parallel=data \
          --blocking_mpi \
          --dtype=float32
