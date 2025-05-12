#!/usr/bin/env bash
# Communication pair
python "${1:?}" 'server' "${@:2}" &>'/dev/null' &
trap "kill $! 2>/dev/null" INT TERM EXIT
python "${1:?}" 'client' "${@:2}"
wait