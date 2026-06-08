#!/usr/bin/env sh
# Show code quality
SRC="${1:-.}"

flake8 -qq --statistics "${SRC:?}" | sort -n