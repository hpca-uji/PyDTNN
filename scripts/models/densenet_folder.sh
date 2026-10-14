#!/bin/bash

export OMP_NUM_THREADS=16
export PYTHONOPTIMIZE=2
export PYTHONUNBUFFERED="True"

MPI_ARGS=()
export MPICH_UNBUFFERED_STDIO="true"
if $(mpirun --version | grep -q 'Open MPI) [5-9].'); then
  MPI_ARGS+=("--output=:raw")
fi

mpirun "${MPI_ARGS[@]}" pydtnn-benchmark \
  --model=densenet169 \
  --dataset=folder \
  --dataset-path=/home/pluijter/proyecto/Datasets/dataset_prueba/ \
  --test-as-validation=False \
  --batch-size=4 \
  --validation-split=0.2 \
  --steps-per-epoch=0 \
  --num-epochs=5 \
  --evaluate=False \
  --optimizer=sgd \
  --optimizer-nesterov=True \
  --learning-rate=0.1 \
  --optimizer-momentum=0.9 \
  --loss-func=categorical_cross_entropy \
  --metrics=categorical_accuracy,categorical_hinge,categorical_mse,categorical_mae,regression_mse,regression_mae,binary_confusion_matrix,precision,recall,f1_score \
  --schedulers=warm_up,reduce_lr_on_plateau,early_stopping \
  --warm-up-epochs=5 \
  --early-stopping-metric=val_categorical_cross_entropy \
  --early-stopping-patience=40 \
  --reduce-lr-on-plateau-metric=val_categorical_cross_entropy \
  --reduce-lr-on-plateau-factor=0.5 \
  --reduce-lr-on-plateau-patience=15 \
  --reduce-lr-on-plateau-min-lr=0.00001 \
  --reduce-lr-every-nepochs-nepochs=30 \
  --reduce-lr-every-nepochs-min-lr=0.00001 \
  --reduce-lr-every-nepochs-factor=0.1 \
  --stop-at-loss-metric=val_categorical_accuracy \
  --stop-at-loss-threshold=70.0 \
  --parallel=data \
  --use-blocking-mpi=False \
  --tracing=False \
  --profile=False \
  --enable-gpu=True \
  --enable-gpudirect=False \
  --dtype=float32 \
  --transform-resize=True \
  --transform-resize-size=250 \
  --history-file="results_densenet_retinografias" \
  --tensor-format="NHWC"
