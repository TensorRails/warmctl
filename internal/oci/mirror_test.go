package oci

import (
	"context"
	"errors"
	"io"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"

	ocispec "github.com/opencontainers/image-spec/specs-go/v1"
	"oras.land/oras-go/v2/content/memory"

	"github.com/TensorRails/warmctl/v2/internal/manifest"
	"github.com/TensorRails/warmctl/v2/internal/store"
)

// flaky counts fetches and fails those fail picks.
type flaky struct {
	*memory.Store
	mu      sync.Mutex
	fetched int
	fail    func(ocispec.Descriptor) bool
}

func (f *flaky) Fetch(ctx context.Context, d ocispec.Descriptor) (io.ReadCloser, error) {
	f.mu.Lock()
	f.fetched++
	f.mu.Unlock()
	if f.fail != nil && f.fail(d) {
		return nil, errors.New("connection reset")
	}
	return f.Store.Fetch(ctx, d)
}

// A pull that failed on one piece resumes: the next one fetches only that
// piece and the store ends up with the whole file.
func TestPullResumes(t *testing.T) {
	defer func(b time.Duration) { chunkBackoff = b }(chunkBackoff)
	chunkBackoff = time.Millisecond
	ctx := context.Background()
	src := filepath.Join(t.TempDir(), "a")
	snapshotDir(t, src, "weights-bytes-of-thirty-chars")
	reg := &flaky{Store: memory.New()}
	o := Options{PieceSize: 4, Concurrency: 2}
	if _, err := Push(ctx, reg, src, "v1", o); err != nil {
		t.Fatal(err)
	}
	reg.fail = func(d ocispec.Descriptor) bool {
		return d.MediaType == PieceType && d.Annotations[AnnotationPieceOffset] == "16"
	}
	s := store.Store{Root: filepath.Join(t.TempDir(), "state")}
	if _, err := Pull(ctx, reg, "v1", filepath.Join(t.TempDir(), "out1"), s, o); err == nil {
		t.Fatal("the failing piece went unnoticed")
	}
	reg.fail, reg.fetched = nil, 0
	var logs []string
	o.Log = func(f string, _ ...any) { logs = append(logs, f) }
	out := filepath.Join(t.TempDir(), "out2")
	m, err := Pull(ctx, reg, "v1", out, s, o)
	if err != nil {
		t.Fatal(err)
	}
	// manifest, config and the one missing piece
	if reg.fetched != 3 {
		t.Fatalf("%d fetches on resume", reg.fetched)
	}
	if left, _ := filepath.Glob(filepath.Join(s.Blobs(), ".assemble-*")); len(left) != 0 {
		t.Fatalf("assembly files left: %v", left)
	}
	for _, f := range m.Files {
		if !s.Has(f.SHA256) {
			t.Fatalf("%s missing", f.Path)
		}
	}
	if _, err := os.Stat(filepath.Join(out, DigestsFile)); err != nil {
		t.Fatalf("the pull saved no digests: %v", err)
	}
}

// Mirror copies a repository's index and snapshots into an OCI layout, a
// pull from the layout gets the same files, and a second mirror sends
// nothing.
func TestMirrorToLayout(t *testing.T) {
	ctx := context.Background()
	reg := memory.New()
	h100, mh := platformSnapshot(t, "h100-80gb-sxm", "580.95.05")
	b200, mb := platformSnapshot(t, "b200-180gb", "580.95.05")
	pushIndexed(t, reg, h100, mh)
	pushIndexed(t, reg, b200, mb)
	dir := filepath.Join(t.TempDir(), "cache", "model")
	dst, tag, err := Open(ctx, DirScheme+dir+":latest", Options{})
	if err != nil || tag != "latest" {
		t.Fatalf("%q %v", tag, err)
	}
	onlyH100 := func(m manifest.Manifest) error {
		if m.Hardware.Row.ID != "h100-80gb-sxm" {
			return errors.New("not here")
		}
		return nil
	}
	var logs []string
	o := Options{Log: func(f string, _ ...any) { logs = append(logs, f) }}
	if n, err := Mirror(ctx, reg, IndexTag, dst, onlyH100, o); err != nil || n != 1 {
		t.Fatalf("mirrored %d, %v", n, err)
	}
	entries, err := Platforms(ctx, dst)
	if err != nil || len(entries) != 1 || entries[0].Annotations[AnnotationPlatform] != "1xh100-80gb-sxm" {
		t.Fatalf("layout index %+v %v", entries, err)
	}
	s := store.Store{Root: filepath.Join(t.TempDir(), "state")}
	tag, err = Choose(ctx, dst, IndexTag, nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := Pull(ctx, dst, tag, filepath.Join(t.TempDir(), "out"), s, Options{}); err != nil {
		t.Fatalf("pull from the layout: %v", err)
	}
	if n, err := Mirror(ctx, reg, IndexTag, dst, onlyH100, Options{}); err != nil || n != 1 {
		t.Fatalf("second mirror: %d %v", n, err)
	}
	if n, err := Mirror(ctx, reg, IndexTag, dst, nil, Options{}); err != nil || n != 2 {
		t.Fatalf("all platforms: %d %v", n, err)
	}
	if entries, _ := Platforms(ctx, dst); len(entries) != 2 {
		t.Fatalf("%d entries after mirroring all", len(entries))
	}
}

// cut serves the first n bytes of a body and then fails it.
type cut struct {
	io.ReadCloser
	n int
}

func (c *cut) Read(p []byte) (int, error) {
	if c.n <= 0 {
		return 0, io.ErrUnexpectedEOF
	}
	if len(p) > c.n {
		p = p[:c.n]
	}
	n, err := c.ReadCloser.Read(p)
	c.n -= n
	return n, err
}

// stalls serves a body that sends nothing and blocks until closed.
type stalls struct{ closed chan struct{} }

func (s *stalls) Read([]byte) (int, error) {
	<-s.closed
	return 0, errors.New("use of closed network connection")
}
func (s *stalls) Close() error {
	select {
	case <-s.closed:
	default:
		close(s.closed)
	}
	return nil
}

// cutting fails the first fetch of every blob: half the time a body cut
// short, half the time one that stalls.
type cutting struct {
	*memory.Store
	mu   sync.Mutex
	seen map[string]int
}

func (c *cutting) Fetch(ctx context.Context, d ocispec.Descriptor) (io.ReadCloser, error) {
	rc, err := c.Store.Fetch(ctx, d)
	if err != nil || d.MediaType == ocispec.MediaTypeImageManifest || d.MediaType == ConfigType {
		return rc, err
	}
	c.mu.Lock()
	c.seen[d.Digest.String()]++
	first, odd := c.seen[d.Digest.String()] == 1, len(c.seen)%2 == 1
	c.mu.Unlock()
	if !first {
		return rc, nil
	}
	if odd {
		rc.Close()
		return &stalls{closed: make(chan struct{})}, nil
	}
	return &cut{ReadCloser: rc, n: int(d.Size / 2)}, nil
}

// Every kind of blob, piece, bundle and whole file, survives a body cut
// short or stalled on its first fetch.
func TestPullRetriesBlobs(t *testing.T) {
	defer func(d, b time.Duration) { stallAfter, chunkBackoff = d, b }(stallAfter, chunkBackoff)
	stallAfter, chunkBackoff = 50*time.Millisecond, time.Millisecond
	ctx := context.Background()
	src := filepath.Join(t.TempDir(), "a")
	snapshotFiles(t, src, map[string]testFile{
		"criu/core-32768.img":             {"core", manifest.RoleCRIU},
		"criu/fdinfo-2.img":               {"fd", manifest.RoleCRIU},
		"criu/pages-1.img":                {"0123456789abcdefghij-pages", manifest.RoleCRIU},
		"fs/workspace.tar":                {"tar-of-the-workspace", manifest.RoleRuntimeFS},
		"weights/rank-000/part-00000.bin": {"weights-bytes-of-thirty-chars", manifest.RoleWeights},
	})
	reg := &cutting{Store: memory.New(), seen: map[string]int{}}
	o := Options{PieceSize: 20, Concurrency: 3}
	if _, err := Push(ctx, reg, src, "v1", o); err != nil {
		t.Fatal(err)
	}
	s := store.Store{Root: filepath.Join(t.TempDir(), "state")}
	m, err := Pull(ctx, reg, "v1", filepath.Join(t.TempDir(), "out"), s, o)
	if err != nil {
		t.Fatal(err)
	}
	for _, f := range m.Files {
		if !s.Has(f.SHA256) {
			t.Fatalf("%s missing", f.Path)
		}
	}
	kinds := map[string]bool{}
	for d := range reg.seen {
		kinds[d] = true
	}
	if len(kinds) < 4 {
		t.Fatalf("only %d blobs went through the failures", len(kinds))
	}
}
