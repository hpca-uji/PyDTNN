#!/bin/bash

export OMP_NUM_THREADS=16
export PYTHONOPTIMIZE=2
export PYTHONUNBUFFERED="True"

MPI_ARGS=()
export MPICH_UNBUFFERED_STDIO="true"
if mpirun --version | grep -q 'Open MPI) [5-9].'; then
  MPI_ARGS+=("--output=:raw")
fi

mpirun -np 4 "${MPI_ARGS[@]}" \
 pydtnn-benchmark \
  --model=resnet50_from_pytorch \
  --dataset=folder \
  --dataset-path=/home/pluijter/proyecto/Datasets/dataset_prueba2 \
  --no-test-as-validation \
  --batch-size=30 \
  --validation-split=0.3 \
  --steps-per-epoch=0 \
  --num-epochs=30 \
  --no-evaluate \
  --optimizer=adam \
  --learning-rate=3.01e-4 \
  --optimizer-momentum=0.9 \
  --optimizer-epsilon=1e-8 \
  --loss-func=negative_likelihood \
  --metrics=categorical_accuracy,categorical_hinge,categorical_mse,categorical_mae,regression_mse,regression_mae \
  --schedulers=reduce_lr_on_plateau \
  --warm-up-epochs=5 \
  --learning-rate-scaling \
  --reduce-lr-on-plateau-metric=val_categorical_accuracy \
  --reduce-lr-on-plateau-factor=0.1 \
  --reduce-lr-on-plateau-patience=5 \
  --reduce-lr-on-plateau-min-lr=1e-3 \
  --stop-at-loss-threshold=70.0 \
  --parallel-data \
  --use-blocking-mpi \
  --no-tracing \
  --no-profile \
  --backend=gpu \
  --no-use-gpudirect \
  --dtype=float64 \
  --use-class-weights \
  --history \
  --model-sync-freq=4 \
  --use-mpi-cuda
  # --augment-brightness=0.75 \
  # --augment-brightness-factor=1 \
  # --augment-contrast=0.75 \
  # --augment-contrast-factor=0.6 \
  # --augment-saturation=0.75 \
  # --augment-horizontal-flip=0.2 \
  # --augment-vertical-flip=0.6 \
  # --augment-rotate=0.6 \
  # --augment-saturation-factor=0.1 \

#--batch-size=48 \
#--input-scale \
#--input-scale-size=524 \
#--augment-horizontal-flip=0.2 \
#--augment-vertical-flip=0.6 \
#--augment-rotate=0.6 \
