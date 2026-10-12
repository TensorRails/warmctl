# Security

Report a vulnerability privately, through **Report a vulnerability** on this
repository's Security tab or to hello@warmctl.io, not in a public issue. You
get an answer within a few days. Fixes go into the latest release.

warmctl runs as root and starts privileged containers, and a snapshot is a
process image: restoring one runs its code. `pull` checks every file against
the SHA-256 in the snapshot's manifest, which comes from the registry, so
restore snapshots only from registries you trust.

Release archives can be checked before use: see "Verify a release" in
[RELEASING.md](RELEASING.md).
