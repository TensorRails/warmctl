package restore

import (
	"bytes"
	"context"
	"errors"
	"io"
	"os"
	"path/filepath"
	"testing"

	"github.com/TensorRails/warmctl/v2/internal/manifest"
)

func TestStage(t *testing.T) {
	ctx := context.Background()
	src, dst := t.TempDir(), filepath.Join(t.TempDir(), "staged")
	image := bytes.Repeat([]byte("dmtcp-image-"), 300_000)
	files := map[string][]byte{"criu/ckpt_a.dmtcp": image, "weights/rank0/part-0": []byte("weights"), "fs/tmp.tar": nil}
	var m manifest.Manifest
	for p, body := range files {
		if err := os.MkdirAll(filepath.Dir(filepath.Join(src, p)), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(src, p), body, 0o444); err != nil {
			t.Fatal(err)
		}
		role := manifest.RoleRuntimeFS
		if filepath.Dir(p) == "criu" {
			role = manifest.RoleCRIU
		}
		m.Files = append(m.Files, manifest.File{Path: p, Size: int64(len(body)), Role: role})
	}
	if err := os.WriteFile(filepath.Join(src, "manifest.json"), []byte(`{"id":"q06"}`), 0o644); err != nil {
		t.Fatal(err)
	}
	n, err := Stage(ctx, src, dst, m)
	if err != nil || n != int64(len(image)+len("weights")) {
		t.Fatalf("copied %d: %v", n, err)
	}
	for p, body := range withFile(files, "manifest.json", []byte(`{"id":"q06"}`)) {
		if got, err := os.ReadFile(filepath.Join(dst, p)); err != nil || !bytes.Equal(got, body) {
			t.Fatalf("%s differs: %v", p, err)
		}
	}
	if n, err := Stage(ctx, src, dst, m); err != nil || n != 0 {
		t.Fatalf("a complete copy was copied again: %d %v", n, err)
	}
	// A source shorter than the manifest says is an error, not a short copy.
	if err := os.WriteFile(filepath.Join(src, "manifest.json"), []byte(`{"id":"q06","v":2}`), 0o644); err != nil {
		t.Fatal(err)
	}
	for i := range m.Files {
		if m.Files[i].Path == "criu/ckpt_a.dmtcp" {
			m.Files[i].Size = int64(len(image)) + 10
		}
	}
	if _, err := Stage(ctx, src, dst, m); !errors.Is(err, io.ErrUnexpectedEOF) {
		t.Fatalf("short source: %v", err)
	}
	if _, err := os.Stat(filepath.Join(dst, "manifest.json")); err == nil {
		t.Fatal("a failed copy looks complete")
	}
}

// withFile is files and one more.
func withFile(m map[string][]byte, k string, v []byte) map[string][]byte {
	out := map[string][]byte{k: v}
	for p, b := range m {
		out[p] = b
	}
	return out
}
