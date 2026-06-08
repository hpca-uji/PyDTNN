#!/usr/bin/env sh
# Show code quality
SRC="${1:-.}"

radon mi --min B --show --sort "${SRC:?}"
flake8 -qq --statistics "${SRC:?}" | sort -n