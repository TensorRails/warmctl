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

3. `.github/workflows/release.yml` builds cuInterpose and publishes the GitHub
   release:

   | Asset | What |
   |---|---|
   | `warmctl-vX.Y.Z-linux-amd64.tar.gz` | `warmctl`, `cuinterpose/`, LICENSE, NOTICE, README |
   | `warmctl-vX.Y.Z-linux-arm64.tar.gz` | `warmctl` without cuInterpose; not proven on a GPU yet |
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

5. If `catalog/catalog.yaml` changed: `make catalog-push`.

## Verify a release

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
