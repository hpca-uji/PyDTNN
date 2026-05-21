#!/usr/bin/env bash
# Find package versions

src="${1:-pyproject.toml}"

git log --follow --reverse --format="%H" -- "${src:?}" | while read -r commit; do
    version=$(
        git show "${commit:?}:${src:?}" 2>/dev/null |
        grep -E '^_*version_*\s*=' |
        head -n1 |
        sed -E 's/.*=\s*"([^"]+)".*/\1/'
    )

    if [[ -n "$version" && "$version" != "$last_version" ]]; then
        printf '%s %s\n' "${commit:?}" "${version:?}"
        last_version="${version:?}"
    fi
done