#!/bin/bash

export OMP_NUM_THREADS=16
export PYTHONOPTIMIZE=2
export PYTHONUNBUFFERED="True"

MPI_ARGS=()
export MPICH_UNBUFFERED_STDIO="true"
if $(mpirun --version | grep -q 'Open MPI) [5-9].'); then
  MPI_ARGS+=("--output=:raw")
fi

mpirun -np 2 "${MPI_ARGS[@]}" \
  pydtnn-benchmark \
  --model=resnet50 \
  --dataset=folder \
  --dataset-path=/home/pluijter/proyecto/Datasets/dataset_prueba \
  --test-as-validation=False \
  --batch-size=25 \
  --validation-split=0.3 \
  --steps-per-epoch=0 \
  --num-epochs=250 \
  --evaluate=False \
  --optimizer=nadam \
  --optimizer-nesterov=True \
  --learning-rate=2.01e-2 \
  --optimizer-momentum=0.9 \
  --optimizer-beta1=0.9 \
  --optimizer-beta2=0.999 \
  --loss-func=categorical_cross_entropy \
  --optimizer-epsilon=1e-8 \
  --metrics=categorical_accuracy,categorical_hinge,categorical_mse,categorical_mae,regression_mse,regression_mae \
  --schedulers=warm_up,reduce_lr_on_plateau \
  --warm-up-epochs=25 \
  --reduce-lr-on-plateau-metric=val_categorical_cross_entropy \
  --reduce-lr-on-plateau-factor=0.1 \
  --reduce-lr-on-plateau-patience=15 \
  --reduce-lr-on-plateau-min-lr=1e-5 \
  --reduce-lr-every-nepochs-nepochs=25 \
  --reduce-lr-every-nepochs-min-lr=1e-5 \
  --reduce-lr-every-nepochs-factor=0.1 \
  --stop-at-loss-metric=val_categorical_accuracy \
  --stop-at-loss-threshold=70.0 \
  --parallel-data=True \
  --use-blocking-mpi=False \
  --tracing=False \
  --profile=False \
  --backend=cpu \
  --enable-cudnn=False \
  --enable-gpudirect=False \
  --dtype=float32 \
  --history \
  --gradient-scaling=True \
  --augment-scale=True \
  --augment-scale-size=200 \
  --augment-horizontal-flip=0.2 \
  --augment-vertical-flip=0.6 \
  --augment-rotate=0.6 #\
  #--augment-brightness=0.75 #\
  #--augment-brightness-factor=1 #\
  #--augment-contrast=0.75 #\
  #--augment-contrast-factor=0.6 #\
  #--augment-saturation=0.75 #\
  #--augment-saturation-factor=0.1 #\
  #--model-sync-freq=4
