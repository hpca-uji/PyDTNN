#!/usr/bin/env bash
# Communication IOPS tests
self=$(realpath "$0")
root="${self%/*/*/*}/pydtnn"
mode="${1:?mode}"

function cmd() {
	local mode="${1:?}" size="${2:?}" reps="${3:?}"
	"${root:?}/scripts/run_comms_pair.sh" "${root:?}/tests/test_comms_iops.py" "${mode:?}" --size "${size:?}" --reps "${reps:?}"
}

cmd "${mode:?}"          1_000    500_000  #   1KB
cmd "${mode:?}"         10_000    500_000  #  10KB
cmd "${mode:?}"        100_000     50_000  # 100KB
cmd "${mode:?}"      1_000_000      5_000  #   1MB
cmd "${mode:?}"     10_000_000        500  #  10MB
cmd "${mode:?}"    100_000_000        500  # 100MB
cmd "${mode:?}"  1_000_000_000         10  #   1GB
cmd "${mode:?}" 10_000_000_000          1  #  10GB

# FIXME: sequen 1B x 1.5M (block) fixed
# FIXME: random 1MB x 1K (block) fixed
# FIXME: random 10GB x 1 (block) fixed

# TCP deadlock, client+server send both block as reciver does not clear recive buffer

setblocking false on selector notify recv

fixme closing sequence