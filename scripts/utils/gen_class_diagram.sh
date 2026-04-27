#!/usr/bin/env bash
# Generate a SVG from class hierarchy
SRC="${1:-.}"
EXC="${2:-?}"

grep -wR '^class' "${SRC:?}" |  # Find classes
grep -vP "${EXC:?}" |  # Exclude libs
sed -E 's|.*class ||; s|\[[^]]+\]||g; s|\w+=||g; s|:||; s| ||g; s|\(|:|; s|\)||' |  # Parsable format
sed -E 's|(\w+):(\w+),(.*)|\1:\2\n\1:\3|g' |  # Expand multiple inheritance (1)
sed -E 's|(\w+):(\w+),(.*)|\1:\2\n\1:\3|g' |  # Expand multiple inheritance (2)
sed -E 's|(\w+):(\w+),(.*)|\1:\2\n\1:\3|g' |  # Expand multiple inheritance (3)
sed -E 's|(\w+):(\w+),(.*)|\1:\2\n\1:\3|g' |  # Expand multiple inheritance (4)
sed -E 's|(\w+):(\w+),(.*)|\1:\2\n\1:\3|g' |  # Expand multiple inheritance (5)
sed -E 's|(\w+\.)+||g' |  # Remove package prefixes
sort -u |  # Unique classes
sed -E 's|^|    |; s|:| -> |' |  # DOT format
sed '1idigraph {' |  # DOT header
sed '2irankdir=RL' |  # DOT configuration
sed '$a}' |  # DOT footer
dot -Tsvg /dev/stdin  # Generate graph