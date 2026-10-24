# Snapshots in registries and caches

How `push`, `pull`, `up`, `catalog` and `mirror` move snapshots around.

## Names and platforms

A short name stands for a repository in the configured registry:
`warmctl up glm53-flash` pulls `oci://registry.warmctl.io/warmctl/glm53-flash`.
The registry comes from `/etc/warmctl/config.yaml`, then
`~/.config/warmctl/config.yaml`, then the environment (`WARMCTL_REGISTRY`,
`WARMCTL_MIRRORS`, `WARMCTL_CACHE`):

```yaml
registry: harbor.acme.internal/warmctl
```

A snapshot restores only on the hardware it was built on: the same GPU model
and count, the same driver branch, a CPU with the same features. So a model
has one snapshot per platform, all in one repository. `warmctl push SNAP NAME`
pushes a snapshot under its platform tag (`8xh100-80gb-sxm-580.95.05`) and
adds it to the repository's `latest`, an OCI image index. `up NAME` and
`pull NAME` work like a multi-arch image pull: they take the entry this host
can restore, or say for each platform why it does not fit. A tag
(`NAME:TAG`, `oci://…/REPO:TAG`) skips the index.

## Catalog

`warmctl catalog` lists the models a registry publishes, one row per platform
this host can restore. `--all` adds the other platforms with the reason they
do not fit, `--json` prints JSON.

```
MODEL               PLATFORM          SIZE     STATUS  LICENSE
deepseek-v41-flash  8xh100-80gb-sxm   678 GiB  ready   …
qwen36-35b-a3b      1xh100-80gb-sxm   43 GiB   pulled  Apache-2.0
```

The list is the artifact `<namespace>/catalog:latest`, published from
[catalog/catalog.yaml](../catalog/catalog.yaml) by `make catalog-push`. A
model marked `gated` there is pulled by `up` only after `--accept-license`,
once per host.

## Caches and mirrors

GPU clouds put a fast shared file system next to their nodes (a managed NFS
or virtiofs share, Weka, VAST). warmctl uses one as a cache of OCI image layouts, one directory per model:

```yaml
# /etc/warmctl/config.yaml
registry: registry.warmctl.io/warmctl
mirrors: [harbor.acme.internal/warmctl]
cache: /mnt/shared/warmctl
```

`up NAME` and `pull NAME` try the cache, then each mirror, then the registry.
A snapshot downloaded from a registry is copied into the cache, so the next
node reads it at file system speed. The restore itself always runs from local
disk: a snapshot read from a virtiofs share restored in 63 s instead of 14 s.

`up` does not wait for the whole download. It pulls the CRIU images and the
runtime trees first and the weights last, and restores the processes and the
GPU state while the weights still arrive; the engine wakes and reloads them
as soon as they are all there, so a fresh host spends the restore steps
inside the download. Every file appears in the snapshot directory once its
SHA-256 is checked, before the store syncs it to disk: on a disk slower than
the network, the engine reloads from memory while the disk catches up.
`manifest.json` comes last, once every blob is synced, so a pull that fails
leaves no snapshot behind: the restore stops, or, after the engine started
serving, the snapshot is not kept; the partial directory is removed, and the
next `up` resumes from the pieces already checked.

`warmctl mirror NAME DESTINATION` copies a model into a directory or another
registry, and `warmctl mirror --catalog DESTINATION` copies the whole catalog.
It goes blob by blob without touching the local disk and skips what the
destination already has. By default it copies only the platforms this host
can restore; `--all-platforms` copies all of them.

## Layout in a registry

`push` works with any OCI registry. A snapshot is one OCI artifact: its
`manifest.json` is the config, and its files are layers in order. A file
larger than `--piece-size` (64 MiB) goes as consecutive pieces, and a run of
files each under a quarter piece goes as one bundle, their bytes back to back.
A 667 GiB snapshot of 6974 files becomes about 10,700 layers and a 2.5 MB
manifest. Every layer then fits one request under the Cloudflare Workers
limit and one CDN cache object.

`push` and `pull` move 64 layers at once over as many HTTP/1.1 connections
(`--concurrency`). `pull` checks each piece against its digest and each file
against the SHA-256 in `manifest.json`. An interrupted pull resumes: pieces it
already checked stay next to the store and are hashed again instead of
downloaded.

A registry that rejects a whole blob as too large (HTTP 413) gets 64 MiB OCI
chunks from then on; `--chunk-size` asks for chunks up front. A chunk that
fails resumes from the offset the registry reports.
