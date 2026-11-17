// Package kfdreplay pins the libkfdreplay release warmctl runs AMD engines
// with (https://github.com/TensorRails/libkfdreplay): the library an engine
// preloads so that its own processes can checkpoint and restore their GPU
// state (kfdr_checkpoint, kfdr_restore), and DMTCP can then snapshot them as
// CPU processes. The runtime image installs the release's archive, checked
// against SHA256SUMS, with scripts/fetch-kfdreplay.sh.
package kfdreplay

import (
	"crypto/sha256"
	_ "embed"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/TensorRails/warmctl/v2/internal/manifest"
)

//go:embed VERSION
var tag string

//go:embed SHA256SUMS
var sums string

// Env overrides where the library is, for research builds: a snapshot
// records the path and digest it ran with, and a restore needs the same.
const Env = "WARMCTL_KFDREPLAY"

// Root holds one directory per release, as the runtime image installs them:
// an image keeps older releases for the snapshots taken with them.
const Root = "/opt/warmctl/kfdreplay"

// Tag is the pinned GitHub release tag.
func Tag() string { return strings.TrimSpace(tag) }

// Version is what kfdr_version() answers for the pinned release.
func Version() string { return strings.TrimPrefix(Tag(), "v") }

// SHA256 is the pinned digest of a release asset.
func SHA256(asset string) string {
	for _, line := range strings.Split(sums, "\n") {
		if sum, name, ok := strings.Cut(strings.TrimSpace(line), "  "); ok && name == asset {
			return sum
		}
	}
	return ""
}

// Path is where the pinned library is: $WARMCTL_KFDREPLAY, else
// Root/<tag>/libkfdreplay.so.
func Path() string {
	if p := os.Getenv(Env); p != "" {
		return p
	}
	return filepath.Join(Root, Tag(), "libkfdreplay.so")
}

// Locate checks the library at Path and describes it for the manifest. The
// pinned release must match its digest; a library named by $WARMCTL_KFDREPLAY
// is taken as it is (research builds), with its own digest recorded.
func Locate() (manifest.MappedFile, error) {
	p := Path()
	f, err := Describe(p)
	if err != nil {
		return f, fmt.Errorf("libkfdreplay: %w (the ROCm runtime image installs %s; or set %s)", err, p, Env)
	}
	if os.Getenv(Env) != "" {
		f.Version = ""
		return f, nil
	}
	if want := SHA256("libkfdreplay.so"); f.SHA256 != want {
		return f, fmt.Errorf("libkfdreplay: %s has sha256 %s, release %s is %s", p, f.SHA256, Tag(), want)
	}
	f.Version = Version()
	return f, nil
}

// Describe is the path, size, modification time and digest of a file.
func Describe(path string) (manifest.MappedFile, error) {
	out := manifest.MappedFile{Path: path}
	f, err := os.Open(path)
	if err != nil {
		return out, err
	}
	defer f.Close()
	info, err := f.Stat()
	if err != nil {
		return out, err
	}
	if !info.Mode().IsRegular() {
		return out, fmt.Errorf("%s is not a regular file", path)
	}
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return out, err
	}
	out.Size, out.MtimeNS, out.SHA256 = info.Size(), info.ModTime().UnixNano(), hex.EncodeToString(h.Sum(nil))
	return out, nil
}

// Same checks that the file a snapshot recorded is at its path unchanged:
// DMTCP maps an unmodified file area again from the file, after comparing
// its size and modification time, so a restore needs the very same file.
func Same(want manifest.MappedFile) error {
	got, err := Describe(want.Path)
	if err != nil {
		return err
	}
	var diff []string
	if got.Size != want.Size {
		diff = append(diff, fmt.Sprintf("size %d, the snapshot's %d", got.Size, want.Size))
	}
	if got.MtimeNS != want.MtimeNS {
		diff = append(diff, fmt.Sprintf("mtime %d ns, the snapshot's %d", got.MtimeNS, want.MtimeNS))
	}
	if want.SHA256 != "" && got.SHA256 != want.SHA256 {
		diff = append(diff, fmt.Sprintf("sha256 %s, the snapshot's %s", got.SHA256, want.SHA256))
	}
	if len(diff) > 0 {
		return errors.New(want.Path + ": " + strings.Join(diff, "; "))
	}
	return nil
}
