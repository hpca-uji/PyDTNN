#!/bin/bash

# set -x # Debugging flag
export PYTHONPATH=/mnt/beegfs/users/dolzm/install/extrae-3.6.0/libexec:$PYTHONPATH
export EXTRAE_CONFIG_FILE=./extrae.xml
export EXTRAE_ON=1
#
EXTRAELIB=/mnt/beegfs/users/dolzm/install/extrae-3.6.0/lib/libmpitrace.so
#EXTRAELIB=/mnt/beegfs/users/dolzm/install/extrae-3.6.0/lib/libptmpitrace.so

NUMNODES=6
NUMPROCS=6
PROCS_PER_NODE=$(($NUMPROCS / $NUMNODES))
export OMP_NUM_THREADS=12

NODETYPE=altec
LASTH=$(echo $NUMNODES - 1 | bc)
HOSTS=$(for i in $(seq 0 $LASTH); do printf "%s%02d," ${NODETYPE} ${i}; done)
HOSTS=altec2,altec3,altec4,altec5,altec7,altec8

export PYTHONOPTIMIZE=2
export PYTHONUNBUFFERED="True"

MPI_ARGS=()
export MPICH_UNBUFFERED_STDIO="true"
if $(mpirun --version | grep -q 'Open MPI) [5-9].'); then
  MPI_ARGS+=("--output=:raw")
fi

# -genv LD_PRELOAD $EXTRAELIB
mpirun -genv LD_PRELOAD $EXTRAELIB -iface ib0 -hosts $HOSTS -ppn $PROCS_PER_NODE -np $NUMPROCS "${MPI_ARGS[@]}" \
  pydtnn_benchmark \
  --model=alexnet_cifar10 \
  --dataset=cifar10 \
  --dataset_train_path=datasets/cifar10 \
  --dataset_test_path=datasets/cifar10 \
  --test_as_validation=False \
  --batch_size=64 \
  --validation_split=0.2 \
  --steps_per_epoch=0 \
  --num_epochs=10 \
  --evaluate=True \
  --optimizer=sgd \
  --learning_rate=0.001 \
  --momentum=0.9 \
  --loss_func=categorical_cross_entropy \
  --metrics=categorical_accuracy \
  --lr_schedulers=early_stopping,reduce_lr_on_plateau \
  --warm_up_epochs=5 \
  --early_stopping_metric=val_categorical_cross_entropy \
  --early_stopping_patience=10 \
  --reduce_lr_on_plateau_metric=val_categorical_cross_entropy \
  --reduce_lr_on_plateau_factor=0.1 \
  --reduce_lr_on_plateau_patience=5 \
  --reduce_lr_on_plateau_min_lr=0 \
  --parallel=data \
  --use_blocking_mpi=True \
  --tracing=True \
  --profile=False \
  --enable_gpu=False \
  --dtype=float32
