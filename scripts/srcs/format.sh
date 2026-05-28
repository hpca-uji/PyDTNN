#!/usr/bin/env bash
# Fixup sources for PEP8
set -xe; set -o pipefail

SRC="${1:-.}"
MAX_LINE_LENGTH="${MAX_LINE_LENGTH:-100}"
PYS=('(' -name '*.py' -o -name '*.py' -o -name '*.pyi' -o -name '*.pyx' ')')

ruff check --fix "${SRC:?}"
ruff format --line-length "${MAX_LINE_LENGTH:?}" "${SRC:?}"
find "${SRC:?}" "${PYS[@]}" -exec absolufy-imports '{}' '+'
EXPR='#{2,}' && find "${SRC:?}" "${PYS[@]}" -exec grep -qEe "${EXPR:?}" '{}' ';' -print -exec sed -Ei "s/${EXPR:?}/#/g" '{}' ';'
EXPR='\s+#+$' && find "${SRC:?}" "${PYS[@]}" -exec grep -qEe "${EXPR:?}" '{}' ';' -print -exec sed -Ei "s/${EXPR:?}//g" '{}' ';'
EXPR='^\s*#\s*[_=/-]+\s*(end|END)\b' && find "${SRC:?}" "${PYS[@]}" -exec grep -qEe "${EXPR:?}" '{}' ';' -print -exec sed -Ei "/${EXPR:?}/d" '{}' ';'
EXPR='^\s*#\s*[_=/-]{2,}[^A-Z]*$' && find "${SRC:?}" "${PYS[@]}" -exec grep -qEe "${EXPR:?}" '{}' ';' -print -exec sed -Ei "/${EXPR:?}/d" '{}' ';'
autopep8 -iaaar --max-line-length "${MAX_LINE_LENGTH:?}" "${SRC:?}"
isort -e --line-length "${MAX_LINE_LENGTH:?}" "${SRC:?}"