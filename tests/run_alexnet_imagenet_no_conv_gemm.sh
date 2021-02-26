#!/bin/bash

export ENABLE_CONV_GEMM=False

SCRIPT_PATH="$(
  cd "$(dirname "$0")" >/dev/null 2>&1 || exit 1
  pwd -P
)"
"${SCRIPT_PATH}"/run_alexnet_imagenet_conv_gemm.sh
