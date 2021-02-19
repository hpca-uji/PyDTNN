#!/bin/bash

#-----------------------
# Command line options
#-----------------------
[ -n "$1" ] && MODEL="$1"
[ -n "$2" ] && DATASET="$2"

#-------------------------
# Configurable parameters
#-------------------------
MODEL=${MODEL:-alexnet_cifar10}
DATASET=${DATASET:-cifar10}
case "${DATASET}" in
cifar10)
  DATASET_TRAIN_PATH=${DATASET_TRAIN_PATH:-${HOME}/opt/hpca_pydtnn/data/cifar-10-batches-bin}
  ;;
imagenet)
  DATASET_TRAIN_PATH=${DATASET_TRAIN_PATH:-${HOME}/opt/hpca_pydtnn/data/imagenet}
  ;;
*)
  echo "Dataset '${DATASET}' not yet supported"
  exit 1
  ;;
esac
DATASET_TEST_PATH=${DATASET_TEST_PATH:-${DATASET_TRAIN_PATH}}
ENABLE_CONV_GEMM=${ENABLE_CONV_GEMM:-True}
CONV_GEMM_FALLBACK_TO_IM2COL=${CONV_GEMM_FALLBACK_TO_IM2COL:-False}
CONV_GEMM_CACHE=${CONV_GEMM_CACHE:-True}
CONV_GEMM_DECONV=${CONV_GEMM_DECONV:-False}
CONV_GEMM_TRANS=${CONV_GEMM_TRANS:-False}
NODES=${NODES:-1}

#--------------------------
# Only training parameters
#--------------------------
if [ -n "${ONLY_TRAINING}" ]; then
  # shellcheck disable=SC2034
  EVALUATE=False
  # shellcheck disable=SC2034
  TEST_AS_VALIDATION=False
  # shellcheck disable=SC2034
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
FILE_NAME="${MODEL}"
if [ "${ENABLE_CONV_GEMM}" == "True" ]; then
  FILE_NAME="${FILE_NAME}_cg"
  if [ "${CONV_GEMM_FALLBACK_TO_IM2COL}" == "True" ]; then
    FILE_NAME="${FILE_NAME}-fb"
  fi
  if [ "${CONV_GEMM_TRANS}" == "True" ]; then
    FILE_NAME="${FILE_NAME}-cgt"
  fi
  if [ "${CONV_GEMM_DECONV}" == "True" ]; then
    FILE_NAME="${FILE_NAME}-dg"
  fi
  if [ "${CONV_GEMM_CACHE}" == "True" ]; then
    FILE_NAME="${FILE_NAME}-pm"
  fi
else
  FILE_NAME="${FILE_NAME}_i2c-mm"
fi
FILE_NAME="${FILE_NAME}_$(printf '%03d' "${NUM_EPOCHS:-1}")e"
FILE_NAME="${FILE_NAME}_$(printf '%03d' "${STEPS_PER_EPOCH:-1}")s"
FILE_NAME="${FILE_NAME}_$(printf '%02d' "${NODES:-1}")n"
FILE_NAME="${FILE_NAME}_$(printf '%02d' "${OMP_NUM_THREADS:-1}")t"
FILE_NAME_NO_MACHINE_NO_DATE="${FILE_NAME}"
MACHINE="$(uname -n)"
if [[ "${MACHINE}" == "altec"* ]]; then
  MACHINE="altec"
fi
FILE_NAME="${MACHINE}_${FILE_NAME}-$(date +"%Y%m%d-%H_%M")"
HISTORY_FILENAME="${FILE_NAME}.history"
OUTPUT_FILENAME="${FILE_NAME}.out"
SIMPLE_TRACER_OUTPUT="${SIMPLE_TRACER_OUTPUT:-${FILE_NAME}.simple_tracer.csv}"

#--------------------------------------------------------------------------------
# Do not launch the experiment if the same experiment has already been completed
#--------------------------------------------------------------------------------
SEARCH_TEXT="Testing maximum memory"
if grep -q "${SEARCH_TEXT}" ./*"${FILE_NAME_NO_MACHINE_NO_DATE}"*.out 2>/dev/null; then
  echo "The next result files (with '${SEARCH_TEXT}') have been found:"
  grep --files-with-matches "${SEARCH_TEXT}" ./*"${FILE_NAME_NO_MACHINE_NO_DATE}"*.out
  echo "Refusing to relaunch the same experiment."
  echo
  exit
else
  echo "No other result files with the pattern '${FILE_NAME_NO_MACHINE_NO_DATE}' have been found."
  echo "Proceeding..."
fi

#----------------------------
# Set model flags
#----------------------------
function set_model_flags() {
  # Model dependent parameters (extracted from run_benchmarks_data.csv)
  MODEL_FLAGS=""
  # 1) Get column for model
  # model;alexnet_cifar10;alexnet_imagenet;vgg16_cifar10;vgg16_imagenet;resnet34_cifar10;resnet34_imagenet
  models_line=$(grep ";" "${SCRIPT_PATH}"/run_benchmarks_data.csv | grep model)
  for i in 2 3 4 5 6 7; do
    if [ "$(echo "${models_line}" | cut -d ";" -f ${i})" = "${MODEL}" ]; then
      model_column=${i}
      break
    fi
  done
  [ -n "${model_column}" ] || {
    echo "Error: Model '${MODEL}' not found in run_benchmarks_data.csv"
    exit 1
  }
  # 2) Extract parameters for model
  while read -r line; do
    parameter=$(echo "$line" | cut -d ";" -f 1)
    PARAMETER=${parameter^^} # uppercase of parameter
    value=$(echo "$line" | cut -d ";" -f ${model_column})
    if [ -z "${value}" ]; then
      # If parameter has no value, only add it if it has been previously defined
      if [ -n "${!PARAMETER}" ]; then
        MODEL_FLAGS="${MODEL_FLAGS} --${parameter}=${!PARAMETER} "
      fi
    else
      # If parameter has a value, use it as default in case it has not been previously defined
      MODEL_FLAGS="${MODEL_FLAGS} --${parameter}=${!PARAMETER:-${value}} "
    fi
  done < <(grep ";" "${SCRIPT_PATH}"/run_benchmarks_data.csv | grep -v model)
}

function run_benchmark() {
  # 1) set model flags and
  set_model_flags

  # 2) Select sequential or parallel mode
  if [ "${NODES}" == 1 ]; then
    PARALLEL=sequential
    CMD=""
  else
    PARALLEL=data
    # Example of MPI CMD: mpirun -np $procs -iface ib0 -ppn 1 -host $hosts --bind-to none
    # shellcheck disable=SC2086  # MPI_EXTRA_FLAGS must be without ""
    CMD="mpirun -np "${NODES}" -ppn "${MPI_PPN:-1}" -iface "${MPI_IFACE:-ib0}" ${MPI_EXTRA_FLAGS}"
  fi

  # 3) Launch benchmarks_CNN
  # shellcheck disable=SC2086  # To allow MODEL_FLAGS without ""
  ${CMD} python3 -Ou "${SCRIPT_PATH}"/benchmarks_CNN.py \
    --model="${MODEL}" \
    --dataset_train_path="${DATASET_TRAIN_PATH}" \
    --dataset_test_path="${DATASET_TEST_PATH}" \
    --parallel="${PARALLEL}" \
    --simple_tracer_output="${SIMPLE_TRACER_OUTPUT}" \
    --enable_conv_gemm="${ENABLE_CONV_GEMM}" \
    --conv_gemm_fallback_to_im2col="${CONV_GEMM_FALLBACK_TO_IM2COL}" \
    --conv_gemm_cache="${CONV_GEMM_CACHE}" \
    --conv_gemm_deconv="${CONV_GEMM_DECONV}" \
    --conv_gemm_trans="${CONV_GEMM_TRANS}" \
    --history="${HISTORY_FILENAME}" \
    ${MODEL_FLAGS} |
    tee "${OUTPUT_FILENAME}"
}

run_benchmark
