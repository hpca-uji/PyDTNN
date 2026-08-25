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
  pydtnn-benchmark \
    --model="${1:?}" \
    --dataset="${DATASET:?}" \
    --dataset-train-path="datasets/${DATASET:?}" \
    --dataset-test-path="datasets/${DATASET:?}" \
    --batch-size=1 \
    --steps-per-epoch=1 \
    --num-epochs=1 \
    --evaluate \
    --optimizer=sgd \
    --learning-rate=0.01 \
    --loss-func=negative_likelihood \
    --schedulers= \
    --no-parallel-data \
    --no-tracing \
    --no-profile \
    --backend=cpu \
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
