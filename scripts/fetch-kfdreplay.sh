#!/bin/sh
# Fetch the pinned libkfdreplay release (AMD GPU state for DMTCP snapshots,
# https://github.com/TensorRails/libkfdreplay) with the checksums pinned in
# this source tree, into NEW_DESTINATION/<tag>/ with the library's mtime kept:
# a DMTCP restart maps it again from that very file.
# Usage: scripts/fetch-kfdreplay.sh NEW_DESTINATION   (e.g. /opt/warmctl/kfdreplay)
set -eu
repo=$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd)
root=${1:?usage: fetch-kfdreplay.sh NEW_DESTINATION}
tag=$(cat "$repo/internal/kfdreplay/VERSION")
dest=$root/$tag
[ ! -e "$dest" ] || { echo "refusing to overwrite $dest" >&2; exit 1; }
asset=libkfdreplay-$tag-linux-amd64.tar.gz
sums=$repo/internal/kfdreplay/SHA256SUMS
expected=$(awk -v name="$asset" '$2 == name { print $1 }' "$sums")
lib=$(awk '$2 == "libkfdreplay.so" { print $1 }' "$sums")
[ ${#expected} -eq 64 ] && [ ${#lib} -eq 64 ] || { echo "missing pinned checksums for $tag" >&2; exit 1; }
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
trap 'exit 1' HUP INT TERM
curl --proto '=https' --tlsv1.2 -fsSL --retry 3 \
  "https://github.com/TensorRails/libkfdreplay/releases/download/$tag/$asset" -o "$tmp/$asset"
check() {
  if command -v sha256sum >/dev/null 2>&1; then
    printf '%s  %s\n' "$1" "$2" | sha256sum -c -
  else
    printf '%s  %s\n' "$1" "$2" | shasum -a 256 -c -
  fi
}
check "$expected" "$tmp/$asset"
mkdir "$tmp/out"
tar -xzf "$tmp/$asset" --strip-components=1 -C "$tmp/out"
check "$lib" "$tmp/out/libkfdreplay.so"
grep -Fqx "release=$tag" "$tmp/out/BUILDINFO"
grep -Fqx "arch=amd64" "$tmp/out/BUILDINFO"
chmod 0755 "$tmp/out/libkfdreplay.so"
mkdir -p "$root"
mv "$tmp/out" "$dest"
