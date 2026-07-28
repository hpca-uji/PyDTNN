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
if mpirun --version | grep -q 'Open MPI) [5-9].'; then
  MPI_ARGS+=("--output=:raw")
fi

# -genv LD_PRELOAD $EXTRAELIB
mpirun -genv LD_PRELOAD $EXTRAELIB -iface ib0 -hosts $HOSTS -ppn $PROCS_PER_NODE -np $NUMPROCS "${MPI_ARGS[@]}" \
  pydtnn-benchmark \
  --model=alexnet_cifar10 \
  --dataset=cifar10 \
  --dataset-path=datasets/cifar10 \
  --input-normalize \
  --no-test-as-validation \
  --batch-size=64 \
  --validation-split=0.2 \
  --steps-per-epoch=0 \
  --num-epochs=10 \
  --evaluate \
  --optimizer=sgd \
  --learning-rate=0.001 \
  --optimizer-momentum=0.9 \
  --loss-func=categorical_cross_entropy \
  --metrics=categorical_accuracy \
  --schedulers=early_stopping,reduce_lr_on_plateau \
  --warm-up-epochs=5 \
  --early-stopping-metric=val_categorical_cross_entropy \
  --early-stopping-patience=10 \
  --reduce-lr-on-plateau-metric=val_categorical_cross_entropy \
  --reduce-lr-on-plateau-factor=0.1 \
  --reduce-lr-on-plateau-patience=5 \
  --reduce-lr-on-plateau-min-lr=0 \
  --parallel-data \
  --use-blocking-mpi \
  --tracing \
  --no-profile \
  --backend=cpu \
  --no-use-cudnn \
  --dtype=float32
