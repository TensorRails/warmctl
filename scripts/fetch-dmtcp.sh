#!/bin/sh
# Fetch a released native runtime with the checksum pinned in this source tree.
# Usage: scripts/fetch-dmtcp.sh amd64|arm64 NEW_DESTINATION
set -eu
repo=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
arch=${1:?usage: fetch-dmtcp.sh amd64|arm64 NEW_DESTINATION}
dest=${2:?usage: fetch-dmtcp.sh amd64|arm64 NEW_DESTINATION}
case "$arch" in amd64|arm64) ;; *) echo 'unsupported DMTCP architecture' >&2; exit 1 ;; esac
[ ! -e "$dest" ] || { echo "refusing to overwrite $dest" >&2; exit 1; }
version=$(cat "$repo/internal/dmtcp/VERSION")
asset=dmtcp-$version-linux-$arch-debian12.tar.gz
expected=$(awk -v name="$asset" '$2 == name { print $1 }' "$repo/internal/dmtcp/SHA256SUMS")
[ ${#expected} -eq 64 ] || { echo "missing pinned checksum for $asset" >&2; exit 1; }
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
trap 'exit 1' HUP INT TERM
curl --proto '=https' --tlsv1.2 -fsSL --retry 3 \
  "https://github.com/TensorRails/dmtcp/releases/download/$version/$asset" -o "$tmp/$asset"
if command -v sha256sum >/dev/null 2>&1; then
  printf '%s  %s\n' "$expected" "$tmp/$asset" | sha256sum -c -
else
  printf '%s  %s\n' "$expected" "$tmp/$asset" | shasum -a 256 -c -
fi
mkdir "$tmp/runtime"
tar -xzf "$tmp/$asset" --strip-components=1 -C "$tmp/runtime"
test -x "$tmp/runtime/bin/dmtcp_launch"
test -f "$tmp/runtime/lib/libdmtcp_fdkeep.so"
grep -Fqx "release=$version" "$tmp/runtime/BUILDINFO"
grep -Fqx "arch=$arch" "$tmp/runtime/BUILDINFO"
mkdir -p "$(dirname "$dest")"
mv "$tmp/runtime" "$dest"
