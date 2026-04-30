#!/usr/bin/env bash
# Fixup sources for PEP8
SRC="${1:-.}"
MAX_LINE_LENGTH="${PEP8_MAX_LINE_LENGTH:-200}"
PYS=('(' -name '*.py' -o -name '*.py' -o -name '*.pyi' -o -name '*.pyx' ')')

find "${SRC:?}" "${PYS[@]}" -exec absolufy-imports '{}' '+'
EXPR='#{2,}' && find "${SRC:?}" "${PYS[@]}" -exec grep -qEe "${EXPR:?}" '{}' ';' -exec sed -Ei "s/${EXPR:?}/#/g" '{}' ';'
EXPR='\s+#+$' && find "${SRC:?}" "${PYS[@]}" -exec grep -qEe "${EXPR:?}" '{}' ';' -exec sed -Ei "s/${EXPR:?}//g" '{}' ';'
EXPR='^\s*#\s*[_=/-]+\s*(end|END)\b' && find "${SRC:?}" "${PYS[@]}" -exec grep -qEe "${EXPR:?}" '{}' ';' -exec sed -Ei "/${EXPR:?}/d" '{}' ';'
EXPR='^\s*#\s*[_=/-]{2,}[^A-Z]*$' && find "${SRC:?}" "${PYS[@]}" -exec grep -qEe "${EXPR:?}" '{}' ';' -exec sed -Ei "/${EXPR:?}/d" '{}' ';'
autopep8 -iaaar --max-line-length "${MAX_LINE_LENGTH:?}" "${SRC:?}"
isort -q "${SRC:?}"