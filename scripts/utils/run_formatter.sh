#!/usr/bin/env sh
# Fixup sources for PEP8
SRC="${1:-.}"
MAX_LINE_LENGTH="${PEP8_MAX_LINE_LENGTH:-200}"

find "${SRC:?}" -name '*.py' -exec absolufy-imports '{}' '+'
autopep8 -iaaar --max-line-length "${MAX_LINE_LENGTH:?}" "${SRC:?}"
isort "${SRC:?}"