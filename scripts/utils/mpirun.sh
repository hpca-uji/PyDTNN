#!/usr/bin/env bash
# mpirun unbuffered wrapper
export PYTHONUNBUFFERED="True"

ARGS=()
export MPICH_UNBUFFERED_STDIO="true"
if mpirun --version | grep -q 'Open MPI) [5-9].'; then
  ARGS+=("--output=:raw")
fi

exec mpirun "${ARGS[@]}"