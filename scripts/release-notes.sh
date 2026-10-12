#!/usr/bin/env bash
# release-notes.sh TAG [CHANGELOG]: print the changelog's section for TAG,
# from its "## TAG" heading to the next "## ", with repository-relative links
# made absolute for the release page. Fails when the section is missing or
# empty, so a tag cannot be released without its changelog entry.
set -euo pipefail
tag=$1
file=${2:-CHANGELOG.md}
notes=$(awk -v tag="$tag" '/^## / { if (found) exit; if ($2 == tag) { found = 1; next } } found' "$file")
if [ -z "${notes//[[:space:]]/}" ]; then
  echo "release-notes: $file has no section for $tag" >&2
  exit 1
fi
base="https://github.com/TensorRails/warmctl/blob/$tag/"
printf '%s\n' "$notes" | sed -E "s#\]\(([^):\#][^):]*)\)#](${base}\1)#g"
