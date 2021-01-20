#!/bin/bash

#-------------------------
# Configurable parameters
#-------------------------
MODEL=${MODEL:-alexnet_cifar10}
DATASET=${DATASET:-cifar10}
if [ "${DATASET}" == "cifar10" ]; then
  DATASET_TRAIN_PATH=${DATASET_TRAIN_PATH:-${HOME}/opt/hpca_pydtnn/data/cifar-10-batches-bin}
  USE_SYNTHETIC_DATA=${USE_SYNTHETIC_DATA:-False}
elif [ "${DATASET}" == "imagenet" ]; then
  DATASET_TRAIN_PATH=${DATASET_TRAIN_PATH:-${HOME}/opt/hpca_pydtnn/data/imagenet}
  USE_SYNTHETIC_DATA=${USE_SYNTHETIC_DATA:-True}
else
  echo "Dataset '${DATASET}' not supported"
  exit 1
fi
DATASET_TEST_PATH=${DATASET_TEST_PATH:-${DATASET_TRAIN_PATH}}
EVALUATE=${EVALUATE:-True}
NUM_EPOCHS=${NUM_EPOCHS:-30}
PROFILE=${PROFILE:-False}
TRACING=${TRACING:-False}
ENABLE_CONV_GEMM=${ENABLE_CONV_GEMM:-True}

#--------------------------
# Only training parameters
#--------------------------
if [ -n "${ONLY_TRAINING}" ]; then
  EVALUATE=False
  TEST_AS_VALIDATION=False
  VALIDATION_SPLIT=0.2
  STEPS_PER_EPOCH=12
fi

#------------------
# OpeMP parameters
#------------------
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export OMP_DISPLAY_ENV=${OMP_DISPLAY_ENV:-True}

case $(hostname) in
jetson6)
  export GOMP_CPU_AFFINITY="${GOMP_CPU_AFFINITY:-2 4 6 1 3 5 7 0}"
  ;;
nowherman)
  export GOMP_CPU_AFFINITY="${GOMP_CPU_AFFINITY:-3 5 7 9 11 13 15 17 19 21 23 25 27 29 31 33 35 37 39 1 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 0}"
  ;;
lorca)
  export GOMP_CPU_AFFINITY="${GOMP_CPU_AFFINITY:-4 5 6 7 2 3 1 0}"
  ;;
*)
  export OMP_PLACES="cores"
  export OMP_PROC_BIND="close"
  ;;
esac

#---------------------------
# Script related parameters
#---------------------------
SCRIPT_PATH="$(
  cd "$(dirname "$0")" >/dev/null 2>&1 || exit 1
  pwd -P
)"

#----------------------------
# File name for output files
#----------------------------
FILE_NAME="$(uname -n)_${MODEL}"
if [ "${ENABLE_CONV_GEMM}" == "True" ]; then
  FILE_NAME="${FILE_NAME}_conv_gemm"
else
  FILE_NAME="${FILE_NAME}_i2c_mm"
fi
FILE_NAME="${FILE_NAME}_$(printf '%02d' "${OMP_NUM_THREADS:-1}")t"
FILE_NAME="${FILE_NAME}_$(printf '%02d' "${NUM_EPOCHS:-1}")e"
FILE_NAME="${FILE_NAME}-$(date +"%Y%m%d-%H_%M")"
HISTORY_FILENAME="${FILE_NAME}.history"
OUTPUT_FILENAME="${FILE_NAME}.out"
SIMPLE_TRACER_OUTPUT="${SIMPLE_TRACER_OUTPUT:-${FILE_NAME}.simple_tracer.csv}"

#------------------------
# Model dependent options
#------------------------
if [ "${MODEL}" == "alexnet_cifar10" ]; then
  TEST_AS_VALIDATION="${TEST_AS_VALIDATION:-True}"
elif [ "${MODEL}" == "alexnet_imagenet" ]; then
  TEST_AS_VALIDATION="${TEST_AS_VALIDATION:-False}"
else
  echo "Model '${MODEL}' not supported"
  exit 1
fi

#---------------
# Launch PyDTNN
#---------------

python3 -Ou "${SCRIPT_PATH}"/benchmarks_CNN.py \
  --model="${MODEL}" \
  --dataset="${DATASET}" \
  --dataset_train_path="${DATASET_TRAIN_PATH}" \
  --dataset_test_path="${DATASET_TEST_PATH}" \
  --use_synthetic_data="${USE_SYNTHETIC_DATA}" \
  --test_as_validation="${TEST_AS_VALIDATION}" \
  --batch_size=64 \
  --validation_split="${VALIDATION_SPLIT:-0.2}" \
  --steps_per_epoch="${STEPS_PER_EPOCH:-0}" \
  --num_epochs="${NUM_EPOCHS}" \
  --evaluate="${EVALUATE}" \
  --optimizer=sgd \
  --learning_rate=0.01 \
  --momentum=0.9 \
  --loss=categorical_cross_entropy \
  --metrics=categorical_accuracy \
  --lr_schedulers=early_stopping,reduce_lr_on_plateau \
  --warm_up_epochs=5 \
  --early_stopping_metric=val_categorical_cross_entropy \
  --early_stopping_patience=10 \
  --reduce_lr_on_plateau_metric=val_categorical_cross_entropy \
  --reduce_lr_on_plateau_factor=0.1 \
  --reduce_lr_on_plateau_patience=5 \
  --reduce_lr_on_plateau_min_lr=0 \
  --parallel=sequential \
  --non_blocking_mpi=False \
  --profile="${PROFILE}" \
  --tracing="${TRACING}" \
  --simple_tracer_output="${SIMPLE_TRACER_OUTPUT}" \
  --enable_gpu=False \
  --dtype=float32 \
  --enable_conv_gemm="${ENABLE_CONV_GEMM}" \
  --history="${HISTORY_FILENAME}" |
  tee "${OUTPUT_FILENAME}"
