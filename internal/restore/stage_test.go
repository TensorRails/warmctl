package restore

import (
	"bytes"
	"context"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/TensorRails/warmctl/v2/internal/manifest"
	"github.com/TensorRails/warmctl/v2/internal/zframes"
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
	var mu sync.Mutex
	var done []string
	n, err := Stage(ctx, src, dst, m, func(p string) { mu.Lock(); done = append(done, p); mu.Unlock() })
	if err != nil || n != int64(len(image)+len("weights")) {
		t.Fatalf("copied %d: %v", n, err)
	}
	if len(done) != len(files) {
		t.Fatalf("done heard of %v", done)
	}
	for p, body := range withFile(files, "manifest.json", []byte(`{"id":"q06"}`)) {
		if got, err := os.ReadFile(filepath.Join(dst, p)); err != nil || !bytes.Equal(got, body) {
			t.Fatalf("%s differs: %v", p, err)
		}
	}
	if n, err := Stage(ctx, src, dst, m, nil); err != nil || n != 0 {
		t.Fatalf("a complete copy was copied again: %d %v", n, err)
	}
	// A source that stays shorter than the manifest says is an error, not a
	// short copy.
	defer func(wait time.Duration) { arrivalWait = wait }(arrivalWait)
	arrivalWait = 300 * time.Millisecond
	if err := os.WriteFile(filepath.Join(src, "manifest.json"), []byte(`{"id":"q06","v":2}`), 0o644); err != nil {
		t.Fatal(err)
	}
	for i := range m.Files {
		if m.Files[i].Path == "criu/ckpt_a.dmtcp" {
			m.Files[i].Size = int64(len(image)) + 10
		}
	}
	if _, err := Stage(ctx, src, dst, m, nil); err == nil || !strings.Contains(err.Error(), "not there") {
		t.Fatalf("short source: %v", err)
	}
	if _, err := os.Stat(filepath.Join(dst, "manifest.json")); err == nil {
		t.Fatal("a failed copy looks complete")
	}
}

// A file the manifest lists but the volume shows late, as a Global Volume
// shows another pod's writes, is waited for.
func TestStageWaitsForLateFiles(t *testing.T) {
	src, dst := t.TempDir(), filepath.Join(t.TempDir(), "staged")
	body := []byte("weights")
	m := manifest.Manifest{Files: []manifest.File{{Path: "weights/part-0", Size: int64(len(body))}}}
	if err := os.WriteFile(filepath.Join(src, "manifest.json"), []byte(`{"id":"late"}`), 0o644); err != nil {
		t.Fatal(err)
	}
	go func() {
		time.Sleep(700 * time.Millisecond)
		os.MkdirAll(filepath.Join(src, "weights"), 0o755)
		os.WriteFile(filepath.Join(src, "weights/part-0"), body, 0o444)
	}()
	if n, err := Stage(context.Background(), src, dst, m, nil); err != nil || n != int64(len(body)) {
		t.Fatalf("copied %d: %v", n, err)
	}
	if got, err := os.ReadFile(filepath.Join(dst, "weights/part-0")); err != nil || !bytes.Equal(got, body) {
		t.Fatalf("staged %q: %v", got, err)
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

// A file stored as zstd frames is staged as the file itself, the frames
// decoded on the workers, and the stage counts the bytes it read.
func TestStageDecodesFrames(t *testing.T) {
	src, dst := t.TempDir(), filepath.Join(t.TempDir(), "staged")
	image := bytes.Repeat([]byte("dmtcp-image-"), 6_000_000) // 72 MB: three frames
	raw := filepath.Join(t.TempDir(), "raw")
	if err := os.WriteFile(raw, image, 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(src, "criu"), 0o755); err != nil {
		t.Fatal(err)
	}
	ends, err := zframes.Encode(context.Background(), raw, filepath.Join(src, "criu/ckpt_a.dmtcp"), 2)
	if err != nil || len(ends) != 3 {
		t.Fatalf("ends %v: %v", ends, err)
	}
	m := manifest.Manifest{Files: []manifest.File{{Path: "criu/ckpt_a.dmtcp", Size: int64(len(image)),
		Role: manifest.RoleCRIU, Encoding: manifest.EncodingZframes, Frames: ends}}}
	if !m.Encoded() || m.StoredBytes() != ends[2] {
		t.Fatalf("encoded %t, stored %d", m.Encoded(), m.StoredBytes())
	}
	if err := os.WriteFile(filepath.Join(src, "manifest.json"), []byte(`{"id":"z"}`), 0o644); err != nil {
		t.Fatal(err)
	}
	var mu sync.Mutex
	var done []string
	n, err := Stage(context.Background(), src, dst, m, func(p string) { mu.Lock(); done = append(done, p); mu.Unlock() })
	if err != nil || n != ends[2] || len(done) != 1 {
		t.Fatalf("read %d of %d, done %v: %v", n, ends[2], done, err)
	}
	if got, err := os.ReadFile(filepath.Join(dst, "criu/ckpt_a.dmtcp")); err != nil || !bytes.Equal(got, image) {
		t.Fatalf("staged image differs: %v", err)
	}
}
