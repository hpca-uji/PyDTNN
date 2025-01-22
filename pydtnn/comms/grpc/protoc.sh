#!/usr/bin/env bash
# Compile protobuf specification

# Constants
self=$(realpath "$0")
root="${self%/pydtnn/*}"
impl="${self%/*}"

# Compile
proto="${impl:?}/grpc.proto"
python -m grpc_tools.protoc \
  --proto_path="${root:?}" \
  --python_out="${root:?}" \
  --pyi_out="${root:?}" \
  --grpc_python_out="${root:?}" \
  "${proto:?}"