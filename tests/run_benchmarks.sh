#!/bin/bash

#-------------------------
# Command line option
#-------------------------
[ -n "$1" ] && export MODEL="$1"
[ -n "$2" ] && export DATASET="$2"

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
STEPS_PER_EPOCH=${STEPS_PER_EPOCH:-0}
BATCH_SIZE=${BATCH_SIZE:-64}
PROFILE=${PROFILE:-False}
TRACING=${TRACING:-False}
ENABLE_CONV_GEMM=${ENABLE_CONV_GEMM:-True}
CONV_GEMM_FALLBACK_TO_IM2COL=${CONV_GEMM_FALLBACK_TO_IM2COL:-True}
NODES=${NODES:-1}

#--------------------------
# Only training parameters
#--------------------------
if [ -n "${ONLY_TRAINING}" ]; then
  EVALUATE=False
  TEST_AS_VALIDATION=False
  VALIDATION_SPLIT=0.2
fi

#-------------------
# OpenMP parameters
#-------------------
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
volta)
  export GOMP_CPU_AFFINITY="${GOMP_CPU_AFFINITY:-3 5 7 9 11 13 15 17 19 21 23 1 2 4 6 8 10 12 14 16 18 20 22}"
  ;;
*)
  if hostname | grep -q altec; then
    export GOMP_CPU_AFFINITY="${GOMP_CPU_AFFINITY:-14 15 16 17 18 19 20 21 22 23 24 25 26 27 2 3 4 5 6 7 8 9 10 11 12 13 1 0}"
  else
    export OMP_PLACES="cores"
    export OMP_PROC_BIND="close"
  fi
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
  if [ "${CONV_GEMM_FALLBACK_TO_IM2COL}" == "True" ]; then
    FILE_NAME="${FILE_NAME}_fb"
  fi
else
  FILE_NAME="${FILE_NAME}_i2c_mm"
fi
FILE_NAME="${FILE_NAME}_$(printf '%03d' "${NUM_EPOCHS:-1}")e"
FILE_NAME="${FILE_NAME}_$(printf '%03d' "${STEPS_PER_EPOCH:-1}")s"
FILE_NAME="${FILE_NAME}_$(printf '%02d' "${NODES:-1}")n"
FILE_NAME="${FILE_NAME}_$(printf '%02d' "${OMP_NUM_THREADS:-1}")t"
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
elif [ "${MODEL}" == "vgg16_imagenet" ]; then
  TEST_AS_VALIDATION="${TEST_AS_VALIDATION:-False}"
  LEARNING_RATE="${LEARNING_RATE:-0.0001}"
  OPTIMIZER="${OPTIMIZER:-adam}"
elif [ "${MODEL}" == "resnet101_imagenet" ]; then
  TEST_AS_VALIDATION="${TEST_AS_VALIDATION:-False}"
  LEARNING_RATE="${LEARNING_RATE:-0.1}"
  LR_SCHEDULERS="${LR_SCHEDULERS:-warm_up,reduce_lr_on_plateau,early_stopping}"
  EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-40}"
  REDUCE_LR_ON_PLATEAU_FACTOR="${REDUCE_LR_ON_PLATEAU_FACTOR:-0.5}"
  REDUCE_LR_ON_PLATEAU_PATIENCE="${REDUCE_LR_ON_PLATEAU_PATIENCE:-15}"
  REDUCE_LR_ON_PLATEAU_MIN_LR="${REDUCE_LR_ON_PLATEAU_MIN_LR:-0.00001}"
  EXTRA_FLAGS=$(
    sed -z 's/ //g;s/\n/ /g' <<-__EOF
    --nesterov=False
    --reduce_lr_every_nepochs_factor=0.1
    --reduce_lr_every_nepochs_nepochs=30
    --reduce_lr_every_nepochs_min_lr=0.00001
    --stop_at_loss_metric=val_categorical_accuracy
    --stop_at_loss_threshold=70.0
__EOF
  )
elif [ "${MODEL}" == "densenet121_imagenet" ]; then
  TEST_AS_VALIDATION="${TEST_AS_VALIDATION:-False}"
  LR_SCHEDULERS="${LR_SCHEDULERS:-reduce_lr_on_plateau}"
  EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-40}"
  REDUCE_LR_ON_PLATEAU_PATIENCE="${REDUCE_LR_ON_PLATEAU_PATIENCE:-15}"
  REDUCE_LR_ON_PLATEAU_MIN_LR="${REDUCE_LR_ON_PLATEAU_MIN_LR:-0.00001}"
  EXTRA_FLAGS=$(
    sed -z 's/ //g;s/\n/ /g' <<-__EOF
    --nesterov=False
    --decay=1e-4
    --reduce_lr_every_nepochs_factor=0.1
    --reduce_lr_every_nepochs_nepochs=90
    --reduce_lr_every_nepochs_min_lr=0.001
    --stop_at_loss_metric=val_categorical_accuracy
    --stop_at_loss_threshold=70.0
__EOF
  )
else
  echo "Model '${MODEL}' not yet supported"
  exit 1
fi

#---------------
# Launch PyDTNN
#---------------

function run_benchmark() {
  python3 -Ou "${SCRIPT_PATH}"/benchmarks_CNN.py \
    --model="${MODEL}" \
    --dataset="${DATASET}" \
    --dataset_train_path="${DATASET_TRAIN_PATH}" \
    --dataset_test_path="${DATASET_TEST_PATH}" \
    --use_synthetic_data="${USE_SYNTHETIC_DATA}" \
    --test_as_validation="${TEST_AS_VALIDATION}" \
    --batch_size="${BATCH_SIZE}" \
    --validation_split="${VALIDATION_SPLIT:-0.2}" \
    --steps_per_epoch="${STEPS_PER_EPOCH:-0}" \
    --num_epochs="${NUM_EPOCHS}" \
    --evaluate="${EVALUATE}" \
    --optimizer="${OPTIMIZER:-sgd}" \
    --learning_rate="${LEARNING_RATE:-0.01}" \
    --momentum=0.9 \
    --loss=categorical_cross_entropy \
    --metrics=categorical_accuracy \
    --lr_schedulers=LR_SCHEDULERS="${LR_SCHEDULERS:-early_stopping,reduce_lr_on_plateau}" \
    --warm_up_epochs=5 \
    --early_stopping_metric=val_categorical_cross_entropy \
    --early_stopping_patience="${EARLY_STOPPING_PATIENCE:-10}" \
    --reduce_lr_on_plateau_metric=val_categorical_cross_entropy \
    --reduce_lr_on_plateau_factor="${REDUCE_LR_ON_PLATEAU_FACTOR:-0.1}" \
    --reduce_lr_on_plateau_patience="${REDUCE_LR_ON_PLATEAU_PATIENCE:-5}" \
    --reduce_lr_on_plateau_min_lr="${REDUCE_LR_ON_PLATEAU_MIN_LR:-0}" \
    --parallel=sequential \
    --non_blocking_mpi=False \
    --profile="${PROFILE}" \
    --tracing="${TRACING}" \
    --simple_tracer_output="${SIMPLE_TRACER_OUTPUT}" \
    --enable_gpu=False \
    --dtype=float32 \
    --enable_conv_gemm="${ENABLE_CONV_GEMM}" \
    --conv_gemm_fallback_to_im2col="${CONV_GEMM_FALLBACK_TO_IM2COL}" \
    --history="${HISTORY_FILENAME}" \
    ${EXTRA_FLAGS:-} | # shellcheck disable=SC2086
    tee "${OUTPUT_FILENAME}"
}

if [ "${NODES}" == 1 ]; then
  run_benchmark
else
  echo "To be implemented"
fi
