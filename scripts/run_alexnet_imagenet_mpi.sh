#!/bin/bash

# set -x # Debugging flag
# export PYTHONPATH=/home/dolzm/install/extrae-3.6.0/libexec:$PYTHONPATH
# export EXTRAE_CONFIG_FILE=./extrae.xml
# export MKL_NUM_THREADS=12
# export EXTRAE_ON=1
#
# EXTRAELIB=/home/dolzm/install/extrae-3.6.0/lib/libompitrace.so

NUMNODES=15
NUMPROCS=15
PROCS_PER_NODE=$(($NUMPROCS / $NUMNODES))
export OMP_NUM_THREADS=12

NODETYPE=hexa
LASTH=$(echo $NUMNODES - 1 | bc)
HOSTS=$(for i in $(seq 0 $LASTH); do printf "%s%02d," ${NODETYPE} ${i}; done)

export PYTHONOPTIMIZE=2
export PYTHONUNBUFFERED="True"

# -genv LD_PRELOAD $EXTRAELIB
mpirun -iface ib0 -hosts $HOSTS -ppn $PROCS_PER_NODE -np $NUMPROCS \
  pydtnn_benchmark \
  --model=alexnet \
  --dataset=imagenet \
  --dataset_metadata_path=datasets/imagenet/ILSVRC2012_devkit_t12.tar.gz \
  --dataset_train_path=datasets/imagenet/ILSVRC2012_img_train.tar \
  --dataset_test_path=datasets/imagenet/ILSVRC2012_img_val.tar \
  --normalize=True \
  --normalize_offset=-0.449 \
  --normalize_scale=3.537 \
  --test_as_validation=False \
  --batch_size=64 \
  --validation_split=0.2 \
  --steps_per_epoch=0 \
  --num_epochs=300 \
  --evaluate=False \
  --optimizer=adam \
  --learning_rate=0.5 \
  --momentum=0.9 \
  --loss_func=categorical_cross_entropy \
  --metrics=categorical_accuracy \
  --schedulers=early_stopping,reduce_lr_on_plateau \
  --early_stopping_metric=val_categorical_cross_entropy \
  --early_stopping_patience=8 \
  --reduce_lr_on_plateau_metric=val_categorical_cross_entropy \
  --reduce_lr_on_plateau_factor=0.1 \
  --reduce_lr_on_plateau_patience=4 \
  --reduce_lr_on_plateau_min_lr=0 \
  --parallel=data \
  --use_blocking_mpi=True \
  --tracing=False \
  --profile=False \
  --enable_gpu=False \
  --dtype=float32
