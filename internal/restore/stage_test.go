package restore

import (
	"bytes"
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"testing"
	"time"

	"github.com/TensorRails/warmctl/v2/internal/manifest"
	"github.com/TensorRails/warmctl/v2/internal/vllm"
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
	if left, _ := filepath.Glob(filepath.Join(dst, "*", "*"+staging)); len(left) != 0 {
		t.Fatalf("still staging: %v", left)
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

// A stage whose reads hang fails once no range finished for stageStall,
// without waiting for them.
func TestStageStall(t *testing.T) {
	defer func(d time.Duration) { stageStall = d }(stageStall)
	stageStall = 200 * time.Millisecond
	src, dst := t.TempDir(), t.TempDir()
	// Opening a FIFO with no writer blocks, as a hung network read does.
	fifo := filepath.Join(src, "part-0")
	if err := syscall.Mkfifo(fifo, 0o644); err != nil {
		t.Fatal(err)
	}
	defer func() {
		// Let the hung worker go: a writer opens, the read fails on the pipe.
		if w, err := os.OpenFile(fifo, os.O_WRONLY, 0); err == nil {
			w.Close()
		}
	}()
	if err := os.WriteFile(filepath.Join(dst, "part-0"+staging), make([]byte, 16), 0o644); err != nil {
		t.Fatal(err)
	}
	left := map[string]*atomic.Int64{"part-0": new(atomic.Int64)}
	left["part-0"].Store(16)
	start := time.Now()
	_, err := copyJobs(context.Background(), src, dst, []prefetchJob{{path: "part-0", size: 16}}, left,
		func(string) { t.Error("a hung file arrived") })
	if err == nil || !strings.Contains(err.Error(), "its reads hang") {
		t.Fatalf("stall: %v", err)
	}
	if took := time.Since(start); took > 5*time.Second {
		t.Fatalf("the stall took %s to notice", took)
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
	ends, split, err := zframes.Encode(context.Background(), raw, filepath.Join(src, "criu/ckpt_a.dmtcp"), 2, 2)
	if err != nil || len(ends) != 3 || split != 2 {
		t.Fatalf("ends %v split %d: %v", ends, split, err)
	}
	m := manifest.Manifest{Files: []manifest.File{{Path: "criu/ckpt_a.dmtcp", Size: int64(len(image)),
		Role: manifest.RoleCRIU, Encoding: manifest.EncodingZframes, Frames: ends, Split: split}}}
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

// While the weights arrive the reload runs beside the wait: a failed arrival
// ends it, a finished one lets it end on its own.
func TestReloadWhileArriving(t *testing.T) {
	ctx := context.Background()
	block := func(ctx context.Context) ([]vllm.Reloaded, error) { <-ctx.Done(); return nil, ctx.Err() }
	r := &restore{weights: make(chan error, 1)}
	r.weights <- errors.New("blob 3 failed")
	ctx2, cancel := context.WithCancel(ctx)
	defer cancel()
	if _, err := r.reload(ctx2, block); err == nil || !strings.Contains(err.Error(), "blob 3 failed") {
		t.Fatalf("failed arrival: %v", err)
	}
	r = &restore{weights: make(chan error, 1)}
	r.weights <- nil
	rows, err := r.reload(ctx, func(context.Context) ([]vllm.Reloaded, error) {
		time.Sleep(20 * time.Millisecond)
		return []vllm.Reloaded{{}}, nil
	})
	if err != nil || len(rows) != 1 {
		t.Fatalf("finished arrival: %v %v", rows, err)
	}
}

// A stage without room restores a plain snapshot from where it is and
// refuses an encoded one before copying anything.
func TestStageRoom(t *testing.T) {
	if room, err := stageRoom("/var/lib/warmctl/staged", 30<<30, 23<<30, true); !room || err != nil {
		t.Fatalf("room: %t %v", room, err)
	}
	if room, err := stageRoom("/var/lib/warmctl/staged", 10<<30, 23<<30, false); room || err != nil {
		t.Fatalf("plain without room: %t %v", room, err)
	}
	if room, err := stageRoom("/var/lib/warmctl/staged", 10<<30, 23<<30, true); room || err == nil ||
		!strings.Contains(err.Error(), "--state-dir") {
		t.Fatalf("encoded without room: %t %v", room, err)
	}
}
