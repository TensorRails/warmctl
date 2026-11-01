# Releasing

## Cut a release

1. In CHANGELOG.md, rename `## Unreleased` to `## vX.Y.Z — YYYY-MM-DD`. That
   section becomes the release notes (`scripts/release-notes.sh`); a tag
   without one fails to release.
2. Commit, then tag and push:

   ```bash
   git tag -a vX.Y.Z -m vX.Y.Z
   git push origin main vX.Y.Z
   ```

3. `.github/workflows/release.yml` builds cuInterpose, fetches the pinned
   DMTCP packages, builds and signs archives, tests the Linux archives and
   installer on native amd64 and arm64 runners, then publishes the release:

   | Asset | What |
   |---|---|
   | `warmctl-vX.Y.Z-linux-amd64.tar.gz` | `warmctl`, `cuinterpose/`, versioned `dmtcp/`, LICENSE, NOTICE, README |
   | `warmctl-vX.Y.Z-linux-arm64.tar.gz` | `warmctl` and versioned `dmtcp/`, without cuInterpose; CPU-tested, not GPU-qualified |
   | `warmctl-vX.Y.Z-darwin-{amd64,arm64}.tar.gz` | `warmctl` for the registry side: `catalog`, `images`, `mirror`, `push`, `pull`, `profile` |
   | `warmctl-vX.Y.Z-checksums.txt` | SHA-256 of every archive |
   | `warmctl-vX.Y.Z-checksums.txt.sigstore.json` | cosign signature of the checksums, keyless from the workflow |
   | `*.sbom.json` | an SPDX SBOM per archive |

   It also records build provenance for every archive. A tag with a suffix
   (`-beta.N`, `-rc.N`) is published as a pre-release.

4. The Kubernetes runtime image (amd64, about 26 GB unpacked) is built on a
   host with Docker and a fast uplink. `docker push` sends a large layer
   whole, which registry.warmctl.io refuses, so the image goes through a
   local registry and regctl copies it in 64 MiB chunks, then tags `latest`:

   ```bash
   docker run -d --name release-registry -p 127.0.0.1:5000:5000 registry:2
   make pod-image VERSION=vX.Y.Z POD_IMAGE=localhost:5000/warmctl/runtime:vX.Y.Z
   docker push localhost:5000/warmctl/runtime:vX.Y.Z
   regctl registry set localhost:5000 --tls disabled
   regctl registry set registry.warmctl.io --blob-chunk 67108864 --blob-max 67108864
   regctl image copy localhost:5000/warmctl/runtime:vX.Y.Z registry.warmctl.io/warmctl/runtime:vX.Y.Z
   regctl image copy registry.warmctl.io/warmctl/runtime:vX.Y.Z registry.warmctl.io/warmctl/runtime:latest
   ```

   Alternatively, dispatch `.github/workflows/runtime-image.yml` with the
   published release tag. It uses the same Dockerfile, verifies the released
   cuInterpose archive and provenance, runs CPU runtime checks, then copies
   the image to registry.warmctl.io in bounded chunks. Configure the repository
   secret `WARMCTL_REGISTRY_AUTH` as a Docker auth JSON containing only that
   registry. An optional custom base must be pinned by digest. The workflow
   publishes the requested tag and records the digest; promote `latest` only
   after the GPU bake and cold-start checks succeed.

5. If `catalog/catalog.yaml` changed: `make catalog-push`.

## DMTCP dependency

DMTCP has independent `v5.0.0-warmctl.N` tags in TensorRails/dmtcp, with
native Linux amd64 and arm64 packages containing both the runtime and fdkeep.
The plugin source is maintained in that fork's `plugin/fdkeep`, not duplicated
here. DMTCP and its shared libraries are native Linux code, not portable Go
executables. Packages are built on Debian 12 (glibc >= 2.36 and compatible
libstdc++); macOS archives contain only the warmctl client.

For an update, first publish and verify the DMTCP release. Update
`internal/dmtcp/VERSION` and copy its two native archive checksums into
`internal/dmtcp/SHA256SUMS`. `scripts/fetch-dmtcp.sh` verifies those pinned
checksums before extracting. Both GoReleaser and the runtime Dockerfile
consume these assets; neither compiles the fork. `make dmtcp` fetches them
for local release builds.

Linux CLI archives keep the runtime at
`dmtcp/linux-ARCH/VERSION/` beside `warmctl`. The CLI resolves its executable
symlink to find that directory, including custom installation prefixes. The
installer preserves its bytes and mtimes, refuses to overwrite a different
build at the same version, and retains older versions for existing snapshots.
Source-built CLIs without a bundle fall back to `/opt/warmctl/dmtcp`, as used
by the runtime image. Explicit `--dmtcp-dir` and `--dmtcp-plugin` still override
the defaults.

Bake fresh snapshots after a runtime upgrade. Donor and receiver need the
same mapped libraries at the same paths; keep old runtime images and versioned
directories for old snapshots. An updated CLI archive does not publish a new
vLLM runtime image: build and publish that separately using step 4. ARM64
packages pass CPU checkpoint/restore tests; the parallel reader and H200 GPU
qualification are x86-64 only.

## Verify the CLI release

```bash
V=vX.Y.Z
sha256sum --ignore-missing -c warmctl-$V-checksums.txt
cosign verify-blob warmctl-$V-checksums.txt \
  --bundle warmctl-$V-checksums.txt.sigstore.json \
  --certificate-oidc-issuer https://token.actions.githubusercontent.com \
  --certificate-identity "https://github.com/TensorRails/warmctl/.github/workflows/release.yml@refs/tags/$V"
gh attestation verify warmctl-$V-linux-amd64.tar.gz --repo TensorRails/warmctl
```

## Try it without publishing

`make release` builds the same archives into `dist/` as a snapshot (needs
goreleaser v2 and Docker for cuInterpose). A pull request that changes the
release files runs the workflow as a snapshot and keeps the archives as a
workflow artifact for a week.
