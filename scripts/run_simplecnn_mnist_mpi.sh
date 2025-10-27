#!/bin/bash

export OMP_NUM_THREADS=1
export PYTHONOPTIMIZE=2
export PYTHONUNBUFFERED="True"

MPI_ARGS=()
export MPICH_UNBUFFERED_STDIO="true"
if $(mpirun --version | grep -q 'Open MPI) [5-9].'); then
  MPI_ARGS+=("--output=:raw")
fi

mpirun -np 1 "${MPI_ARGS[@]}" \
  pydtnn_benchmark \
  --model=simplecnn \
  --dataset=mnist \
  --dataset_path=datasets/mnist \
  --dataset_train_path=datasets/mnist \
  --dataset_test_path=datasets/mnist \
  --test_as_validation=False \
  --flip_images=True \
  --batch_size=64 \
  --validation_split=0.2 \
  --encryption= \
  --model_sync_freq=0 \
  --num_epochs=50 \
  --final_model_sync=False \
  --evaluate=True \
  --steps_per_epoch=0 \
  --optimizer=sgd \
  --learning_rate=0.01 \
  --loss_func=categorical_cross_entropy \
  --schedulers=warm_up,reduce_lr_every_nepochs \
  --reduce_lr_every_nepochs_factor=0.5 \
  --reduce_lr_every_nepochs_nepochs=30 \
  --reduce_lr_every_nepochs_min_lr=0.001 \
  --early_stopping_metric=val_categorical_cross_entropy \
  --early_stopping_patience=20 \
  --parallel=data \
  --use_blocking_mpi=True \
  --tracing=False \
  --profile=False \
  --enable_gpu=False \
  --dtype=float32
