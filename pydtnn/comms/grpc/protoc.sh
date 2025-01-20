#!/usr/bin/env bash
self=$(realpath "$0")
root="${self%/pydtnn/*}"
impl="${self%/*}"

proto="${impl:?}/mpi.proto"
python -m grpc_tools.protoc \
  --proto_path="${root:?}" \
  --python_out="${root:?}" \
  --pyi_out="${root:?}" \
  --grpc_python_out="${root:?}" \
  "${proto:?}"