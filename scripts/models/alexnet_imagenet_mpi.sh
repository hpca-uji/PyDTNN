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

MPI_ARGS=()
export MPICH_UNBUFFERED_STDIO="true"
if mpirun --version | grep -q 'Open MPI) [5-9].'; then
  MPI_ARGS+=("--output=:raw")
fi

# -genv LD_PRELOAD $EXTRAELIB
mpirun -iface ib0 -hosts $HOSTS -ppn $PROCS_PER_NODE -np $NUMPROCS "${MPI_ARGS[@]}" \
  pydtnn-benchmark \
  --model=alexnet \
  --dataset=imagenet \
  --dataset-path=datasets/imagenet \
  --augment-crop \
  --augment-crop-perc=0.875 \
  --augment-scale \
  --augment-scale-size=227 \
  --augment-normalize \
  --augment-normalize-offset=-0.449 \
  --augment-normalize-scale=3.537 \
  --no-test-as-validation \
  --batch-size=64 \
  --validation-split=0.2 \
  --steps-per-epoch=0 \
  --num-epochs=300 \
  --no-evaluate \
  --optimizer=adam \
  --learning-rate=0.5 \
  --optimizer-momentum=0.9 \
  --loss-func=categorical_cross_entropy \
  --metrics=categorical_accuracy \
  --schedulers=early_stopping,reduce_lr_on_plateau \
  --early-stopping-metric=val_categorical_cross_entropy \
  --early-stopping-patience=8 \
  --reduce-lr-on-plateau-metric=val_categorical_cross_entropy \
  --reduce-lr-on-plateau-factor=0.1 \
  --reduce-lr-on-plateau-patience=4 \
  --reduce-lr-on-plateau-min-lr=0 \
  --parallel-data \
  --use-blocking-mpi \
  --no-tracing \
  --no-profile \
  --backend=cpu \
  --no-use-cudnn \
  --dtype=float32
