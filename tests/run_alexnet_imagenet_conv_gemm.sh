#!/bin/bash

export MODEL="alexnet_imagenet"
export DATASET="imagenet"

SCRIPT_PATH="$(
  cd "$(dirname "$0")" >/dev/null 2>&1 || exit 1
  pwd -P
)"

"${SCRIPT_PATH}"/run_alexnet_cifar10_conv_gemm.sh
