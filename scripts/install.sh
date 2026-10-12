#!/bin/sh
# Install warmctl from its GitHub release:
#
#   curl -fsSL https://warmctl.io/install.sh | sh
#
# It puts the release in /opt/warmctl (warmctl and, on linux-amd64,
# cuinterpose/ next to it) and links /usr/local/bin/warmctl to it. Run it
# again to upgrade. Settings, from the environment:
#
#   WARMCTL_VERSION   release tag, e.g. v2.0.0-beta.5 (default: the newest,
#                     pre-releases included)
#   WARMCTL_PREFIX    install directory (default /opt/warmctl)
#   WARMCTL_BIN_DIR   where the warmctl link goes (default /usr/local/bin;
#                     empty to skip)
#   WARMCTL_DOWNLOAD  base URL of the release assets, for a mirror (default
#                     https://github.com/TensorRails/warmctl/releases/download)
#
# The archive is checked against the release's SHA-256 checksums; with cosign
# on PATH, the checksums' signature is checked too.
#
# Everything is inside main, so a download cut short runs nothing.

set -eu

REPO=TensorRails/warmctl

say() { printf 'warmctl-install: %s\n' "$*" >&2; }
die() { say "error: $*"; exit 1; }
has() { command -v "$1" >/dev/null 2>&1; }

# fetch URL FILE downloads URL to FILE; with FILE "-", to stdout.
fetch() {
	if has curl; then
		curl --proto '=https' --tlsv1.2 -fsSL --retry 3 -o "$2" "$1"
	elif has wget; then
		wget -q -O "$2" "$1"
	else
		die "needs curl or wget"
	fi
}

# exists URL reports whether URL answers 200.
exists() {
	if has curl; then
		curl --proto '=https' --tlsv1.2 -fsSLI --retry 3 -o /dev/null "$1" 2>/dev/null
	else
		wget -q --spider "$1" 2>/dev/null
	fi
}

sha256() {
	if has sha256sum; then
		sha256sum "$1" | cut -d' ' -f1
	elif has shasum; then
		shasum -a 256 "$1" | cut -d' ' -f1
	elif has openssl; then
		openssl dgst -sha256 -r "$1" | cut -d' ' -f1
	else
		die "needs sha256sum, shasum or openssl to check the download"
	fi
}

platform() {
	case $(uname -s) in
		Linux) os=linux ;;
		Darwin) os=darwin ;;
		*) die "no warmctl build for $(uname -s); releases are for Linux and macOS" ;;
	esac
	case $(uname -m) in
		x86_64 | amd64) arch=amd64 ;;
		aarch64 | arm64) arch=arm64 ;;
		*) die "no warmctl build for $(uname -m); releases are for amd64 and arm64" ;;
	esac
}

# latest prints the newest release tag. GitHub's releases/latest skips
# pre-releases, so read the list instead, newest first: from the API, or
# from the releases feed when the API rate-limits this address.
latest() {
	tag=$(fetch "https://api.github.com/repos/$REPO/releases?per_page=1" - 2>/dev/null |
		sed -n 's/.*"tag_name"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' | head -n 1) || true
	if [ -z "$tag" ]; then
		tag=$(fetch "https://github.com/$REPO/releases.atom" - 2>/dev/null |
			sed -n 's|.*/releases/tag/\([^"<]*\)".*|\1|p' | head -n 1) || true
	fi
	printf '%s\n' "$tag"
}

# verify checks the archive against the release checksums. Releases before
# the checksums file existed (v2.0.0-beta.5 and older) are only fetched over
# HTTPS; say so.
verify() {
	sums=$tmp/$name-checksums.txt
	if ! exists "$base/$name-checksums.txt"; then
		say "warning: $version publishes no checksums; trusting HTTPS alone"
		return
	fi
	fetch "$base/$name-checksums.txt" "$sums"
	want=$(awk -v f="$archive" '$2 == f || $2 == "*" f { print $1 }' "$sums")
	[ -n "$want" ] || die "$archive is not in $name-checksums.txt"
	got=$(sha256 "$tmp/$archive")
	[ "$got" = "$want" ] || die "$archive checksum mismatch: got $got, want $want"
	say "checksum ok"
	if has cosign; then
		fetch "$base/$name-checksums.txt.sigstore.json" "$sums.sigstore.json"
		cosign verify-blob "$sums" --bundle "$sums.sigstore.json" \
			--certificate-oidc-issuer https://token.actions.githubusercontent.com \
			--certificate-identity "https://github.com/$REPO/.github/workflows/release.yml@refs/tags/$version" \
			>/dev/null 2>&1 || die "cosign could not verify the signature of $name-checksums.txt"
		say "signature ok"
	fi
}

# as_root runs a command, through sudo when the target is not writable.
as_root() {
	if [ -n "$sudo" ]; then
		$sudo "$@"
	else
		"$@"
	fi
}

# writable reports whether DIR, or the nearest existing parent, is writable.
writable() {
	w=$1
	while [ ! -e "$w" ]; do w=$(dirname "$w"); done
	[ -w "$w" ]
}

main() {
	platform
	prefix=${WARMCTL_PREFIX:-/opt/warmctl}
	bindir=${WARMCTL_BIN_DIR-/usr/local/bin}
	download=${WARMCTL_DOWNLOAD:-https://github.com/$REPO/releases/download}
	version=${WARMCTL_VERSION:-}
	if [ -z "$version" ]; then
		version=$(latest) || true
		[ -n "$version" ] || die "could not find the newest release on GitHub; set WARMCTL_VERSION"
	fi
	case $version in v*) ;; *) version=v$version ;; esac

	name=warmctl-$version
	archive=$name-$os-$arch.tar.gz
	base=$download/$version

	sudo=
	if [ "$(id -u)" != 0 ]; then
		for d in "$prefix" ${bindir:+"$bindir"}; do
			writable "$d" && continue
			has sudo || die "$d is not writable; run as root or set WARMCTL_PREFIX and WARMCTL_BIN_DIR"
			sudo=sudo
		done
	fi

	tmp=$(mktemp -d 2>/dev/null || mktemp -d -t warmctl)
	trap 'rm -rf "$tmp"' EXIT
	trap 'exit 1' HUP INT TERM

	say "downloading $archive"
	fetch "$base/$archive" "$tmp/$archive" ||
		die "no $archive in release $version: see https://github.com/$REPO/releases"
	verify
	mkdir "$tmp/x"
	tar -xzf "$tmp/$archive" -C "$tmp/x"

	# The binary is warmctl since the archives were built by goreleaser,
	# warmctl-linux-amd64 before.
	bin=
	for b in warmctl "warmctl-$os-$arch"; do
		if [ -f "$tmp/x/$b" ]; then bin=$tmp/x/$b; break; fi
	done
	[ -n "$bin" ] || die "$archive holds no warmctl binary"
	chmod 0755 "$bin"
	# Some archives carry cuinterpose/ as 0700; it is read by everyone.
	chmod -R a+rX "$tmp/x"

	# Every file goes in under a temporary name and is renamed over the old
	# one, so a running warmctl keeps its binary and a failed install leaves
	# the old release whole.
	as_root mkdir -p "$prefix"
	for f in LICENSE NOTICE README.md; do
		[ -f "$tmp/x/$f" ] || continue
		as_root cp "$tmp/x/$f" "$prefix/.$f.new"
		as_root mv -f "$prefix/.$f.new" "$prefix/$f"
	done
	if [ -d "$tmp/x/cuinterpose" ]; then
		as_root rm -rf "$prefix/.cuinterpose.new" "$prefix/.cuinterpose.old"
		as_root cp -R "$tmp/x/cuinterpose" "$prefix/.cuinterpose.new"
		if [ -e "$prefix/cuinterpose" ]; then
			as_root mv "$prefix/cuinterpose" "$prefix/.cuinterpose.old"
		fi
		as_root mv "$prefix/.cuinterpose.new" "$prefix/cuinterpose"
		as_root rm -rf "$prefix/.cuinterpose.old"
	fi
	as_root cp "$bin" "$prefix/.warmctl.new"
	as_root mv -f "$prefix/.warmctl.new" "$prefix/warmctl"
	# Older installs linked warmctl to warmctl-linux-amd64; the link is now
	# the binary itself.
	as_root rm -f "$prefix/warmctl-linux-amd64"

	if [ -n "$bindir" ]; then
		as_root mkdir -p "$bindir"
		as_root ln -sf "$prefix/warmctl" "$bindir/warmctl"
	fi

	installed=$("$prefix/warmctl" version 2>/dev/null) || installed="warmctl $version"
	say "installed $installed in $prefix"
	if [ -n "$bindir" ]; then
		case :$PATH: in
			*:"$bindir":*) ;;
			*) say "add $bindir to PATH, or run $prefix/warmctl" ;;
		esac
	else
		say "add $prefix to PATH"
	fi
	if [ "$os" = linux ]; then
		say "next: sudo warmctl doctor, then see https://warmctl.io/docs/install/#host"
	else
		say "on macOS warmctl works with the registry (catalog, push, pull, mirror); restores need a Linux GPU host"
	fi
}

main "$@"
