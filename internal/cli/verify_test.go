package cli

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/TensorRails/warmctl/v2/internal/manifest"
	"github.com/TensorRails/warmctl/v2/internal/zframes"
)

// verify checks a `build --compress` directory against the files' own
// sizes and digests: stored sizes, and with --deep the decoded bytes.
func TestVerifyEncoded(t *testing.T) {
	dir := t.TempDir()
	plain, image := []byte("launch"), bytes.Repeat([]byte("process image "), 1<<20)
	if err := os.MkdirAll(filepath.Join(dir, "criu"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "launch.json"), plain, 0o644); err != nil {
		t.Fatal(err)
	}
	src, dst := filepath.Join(t.TempDir(), "image"), filepath.Join(dir, "criu/image")
	if err := os.WriteFile(src, image, 0o644); err != nil {
		t.Fatal(err)
	}
	ends, split, err := zframes.Encode(context.Background(), src, dst, 2, 0)
	if err != nil {
		t.Fatal(err)
	}
	digest := func(b []byte) string { s := sha256.Sum256(b); return hex.EncodeToString(s[:]) }
	m := manifest.Manifest{
		Schema: manifest.Schema, ID: "v1",
		Launch:  manifest.Launch{Ranks: 1, Argv: []string{"vllm", "serve", "/models/m"}},
		Process: manifest.Process{RootPID: 40000},
		Weights: manifest.Weights{Ranks: []manifest.Rank{{}}},
		Files: []manifest.File{
			{Path: "launch.json", Size: int64(len(plain)), SHA256: digest(plain)},
			{Path: "criu/image", Size: int64(len(image)), SHA256: digest(image),
				Encoding: manifest.EncodingZframes, Frames: ends, Split: split},
		},
	}
	if err := m.Write(filepath.Join(dir, "manifest.json")); err != nil {
		t.Fatal(err)
	}
	verify := func(args ...string) (int, string) {
		var stdout, stderr bytes.Buffer
		code := Run(context.Background(), append([]string{"verify"}, args...), &stdout, &stderr, "test")
		return code, stdout.String() + stderr.String()
	}
	for _, args := range [][]string{{dir}, {"--deep", dir}} {
		if code, out := verify(args...); code != 0 || !strings.HasPrefix(out, "ok ") {
			t.Fatalf("verify %v: exit %d: %s", args, code, out)
		}
	}
	// A damaged frame keeps the stored size: only --deep sees it.
	stored, err := os.ReadFile(dst)
	if err != nil {
		t.Fatal(err)
	}
	stored[len(stored)/2] ^= 0xff
	if err := os.WriteFile(dst, stored, 0o644); err != nil {
		t.Fatal(err)
	}
	if code, out := verify(dir); code != 0 {
		t.Fatalf("verify: exit %d: %s", code, out)
	}
	if code, out := verify("--deep", dir); code == 0 || !strings.Contains(out, "criu/image") {
		t.Fatalf("verify --deep of a damaged frame: exit %d: %s", code, out)
	}
}
