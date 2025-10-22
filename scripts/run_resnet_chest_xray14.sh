#!/bin/bash

export OMP_NUM_THREADS=16
export PYTHONOPTIMIZE=2
export PYTHONUNBUFFERED="True"

MPI_ARGS=()
export MPICH_UNBUFFERED_STDIO="true"
if $(mpirun --version | grep -q 'Open MPI) [5-9].'); then
  MPI_ARGS+=("--output=:raw")
fi

mpirun -np 4 "${MPI_ARGS[@]}" \
  pydtnn_benchmark \
  --model=resnet \
  --dataset=chestxray14 \
  --dataset_metadata_path=datasets/chest_xray14 \
  --dataset_path=datasets/chest_xray14 \
  --test_as_validation=False \
  --batch_size=10 \
  --validation_split=0.2 \
  --steps_per_epoch=0 \
  --num_epochs=15 \
  --evaluate=False \
  --optimizer=sgd \
  --nesterov=True \
  --learning_rate=0.1 \
  --momentum=0.9 \
  --loss_func=categorical_cross_entropy \
  --metrics=categorical_accuracy \
  --lr_schedulers=warm_up,reduce_lr_on_plateau,early_stopping \
  --warm_up_epochs=5 \
  --early_stopping_metric=val_categorical_cross_entropy \
  --early_stopping_patience=40 \
  --reduce_lr_on_plateau_metric=val_categorical_cross_entropy \
  --reduce_lr_on_plateau_factor=0.5 \
  --reduce_lr_on_plateau_patience=15 \
  --reduce_lr_on_plateau_min_lr=0.00001 \
  --reduce_lr_every_nepochs_nepochs=30 \
  --reduce_lr_every_nepochs_min_lr=0.00001 \
  --reduce_lr_every_nepochs_factor=0.1 \
  --stop_at_loss_metric=val_categorical_accuracy \
  --stop_at_loss_threshold=70.0 \
  --parallel=data \
  --use_blocking_mpi=False \
  --tracing=False \
  --profile=False \
  --enable_gpu=False \
  --enable_gpudirect=False \
  --dtype=float32 \
  --resize=True \
  --resize_dimension=300
