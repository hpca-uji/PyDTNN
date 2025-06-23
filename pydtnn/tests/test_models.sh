#!/usr/bin/env bash
# Quick all model test

export PYTHONOPTIMIZE=0
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED="True"
DATASET="${DATASET:-cifar10}"

# Constants
self=$(realpath "$0")
root="${self%/pydtnn/*}"

# Functions
function models() {
  find "${root:?}/pydtnn/models" -type f -name '*.py' ! -name '*_*' | sed 's|.*/||; s|.py||'
}

function run_model() {
  pydtnn_benchmark \
    --model="${1:?}" \
    --dataset="${DATASET:?}" \
    --dataset_train_path="datasets/${DATASET:?}" \
    --dataset_test_path="datasets/${DATASET:?}" \
    --batch_size=1 \
    --steps_per_epoch=1 \
    --num_epochs=1 \
    --evaluate=True \
    --optimizer=sgd \
    --learning_rate=0.01 \
    --loss_func=categorical_cross_entropy \
    --lr_schedulers= \
    --parallel=sequential \
    --tracing=False \
    --profile=False \
    --enable_gpu=False \
    --dtype=float32
}

function print_state() {
  printf "%s %s${3:-\n}" "${1:?}" "${2:?}"
}

# Main fragment
if [ $# -eq 0 ]; then
  set -- $(models)
fi

for model; do
  print_state '⟳' "${model:?}" "\r"
  if stdout=$(run_model "${model:?}" 2>&1); then
    print_state '✓' "${model:?}"
  else
    print_state '✕' "${model:?}"
    echo "${stdout:?}"
  fi
done
