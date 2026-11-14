#!/bin/sh
# KFD topologies for the AMD end-to-end test, from the MI350X pod's
# (internal/host/testdata/kfd-mi350x-pod): a/ and b/ each see one MI350X,
# another GPU of the same host (gpu_id, render node, PCI address); ab/ sees both.
set -eu
src=$(dirname "$0")/../../../internal/host/testdata/kfd-mi350x-pod
out=${1:?usage: fixtures.sh OUT}
rm -rf "$out"
gpu() { # dir node gpu_id minor location unique
  d=$1/sys/topology/nodes/$2
  mkdir -p "$d/mem_banks/0"
  sed -e "s/^location_id .*/location_id $5/" -e "s/^drm_render_minor .*/drm_render_minor $4/" \
      -e "s/^unique_id .*/unique_id $6/" "$src/sys/topology/nodes/3/properties" > "$d/properties"
  echo "$3" > "$d/gpu_id"
  cp "$src/sys/topology/nodes/3/mem_banks/0/properties" "$d/mem_banks/0/properties"
}
base() {
  mkdir -p "$1/dev" "$1/sys/amdgpu" "$1/sys/topology/nodes/0"
  : > "$1/dev/kfd"
  cp "$src/sys/amdgpu/version" "$1/sys/amdgpu/version"
  cp "$src/sys/topology/nodes/0/properties" "$src/sys/topology/nodes/0/gpu_id" "$1/sys/topology/nodes/0/"
}
for f in a b ab; do base "$out/$f"; done
gpu "$out/a" 3 41001 136 2304 1229782938247303441
gpu "$out/b" 4 41002 144 49408 2459565876494606882
gpu "$out/ab" 3 41001 136 2304 1229782938247303441
gpu "$out/ab" 4 41002 144 49408 2459565876494606882
