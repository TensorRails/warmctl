package oci

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	ocispec "github.com/opencontainers/image-spec/specs-go/v1"
	"oras.land/oras-go/v2/content/memory"

	"github.com/TensorRails/warmctl/v2/internal/manifest"
	"github.com/TensorRails/warmctl/v2/internal/store"
)

// gated holds back the weights: a fetch of a weights layer waits until open
// is closed, or fails while fail is set.
type gated struct {
	*memory.Store
	open chan struct{}
	fail bool
}

func weightsLayer(d ocispec.Descriptor) bool {
	p := d.Annotations[AnnotationPieceOf]
	if p == "" {
		p = d.Annotations[ocispec.AnnotationTitle]
	}
	return strings.HasPrefix(p, "weights/")
}

func (g *gated) Fetch(ctx context.Context, d ocispec.Descriptor) (io.ReadCloser, error) {
	if weightsLayer(d) {
		if g.fail {
			return nil, errors.New("connection reset")
		}
		select {
		case <-g.open:
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}
	return g.Store.Fetch(ctx, d)
}

// twoRanks writes a snapshot with bundled, pieced and duplicated files and
// the weights of two ranks, and pushes it to a gated registry as v1.
func twoRanks(t *testing.T, o Options) *gated {
	t.Helper()
	src := filepath.Join(t.TempDir(), "a")
	snapshotFiles(t, src, map[string]testFile{
		"criu/core-32768.img":             {"core", manifest.RoleCRIU},
		"criu/fdinfo-2.img":               {"fd", manifest.RoleCRIU},
		"criu/fdinfo-3.img":               {"fd", manifest.RoleCRIU},
		"criu/pages-1.img":                {"0123456789abcdefghij-pages", manifest.RoleCRIU},
		"fs/workspace.tar":                {"tar", manifest.RoleRuntimeFS},
		"weights/rank-000/part-00000.bin": {"rank0-part0-weights-bytes", manifest.RoleWeights},
		"weights/rank-001/part-00000.bin": {"rank1-part0-weights-bytes", manifest.RoleWeights},
	})
	reg := &gated{Store: memory.New(), open: make(chan struct{})}
	if _, err := Push(context.Background(), reg.Store, src, "v1", o); err != nil {
		t.Fatal(err)
	}
	return reg
}

func paths(files []manifest.File) []string {
	var out []string
	for _, f := range files {
		out = append(out, f.Path)
	}
	return out
}

func checkFiles(t *testing.T, dir string, files []manifest.File) {
	t.Helper()
	for _, f := range files {
		raw, err := os.ReadFile(filepath.Join(dir, f.Path))
		sum := sha256.Sum256(raw)
		if err != nil || hex.EncodeToString(sum[:]) != f.SHA256 {
			t.Fatalf("%s differs: %v", f.Path, err)
		}
	}
}

// A pull with Arrivals writes the snapshot directory in place: the files a
// restore needs first are there, checked, while the weights still arrive,
// and manifest.json comes last.
func TestPullArrives(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	o := Options{PieceSize: 8, Concurrency: 3}
	reg := twoRanks(t, o)
	a := NewArrivals()
	o.Arrivals = a
	out := filepath.Join(t.TempDir(), "pulled", "a")
	s := store.Store{Root: filepath.Join(t.TempDir(), "state")}
	done := make(chan error, 1)
	go func() {
		_, err := Pull(ctx, reg, "v1", out, s, o)
		done <- err
	}()
	select {
	case <-a.Begun():
	case err := <-done:
		t.Fatalf("the pull ended before it began: %v", err)
	}
	m := a.Manifest()
	if a.Dir() != out || m.ID != "oci-test" {
		t.Fatalf("began in %s with %+v", a.Dir(), m)
	}
	if err := a.Wait(ctx, paths(m.FirstLane())); err != nil {
		t.Fatal(err)
	}
	checkFiles(t, out, m.FirstLane())
	for _, name := range []string{"manifest.json", "weights/rank-000/part-00000.bin"} {
		if _, err := os.Stat(filepath.Join(out, name)); err == nil {
			t.Fatalf("%s is there before the weights arrived", name)
		}
	}
	if _, err := os.Stat(out + ".pulling"); err != nil {
		t.Fatalf("no mark while the pull runs: %v", err)
	}
	close(reg.open)
	if err := a.Done(ctx); err != nil {
		t.Fatal(err)
	}
	if err := <-done; err != nil {
		t.Fatal(err)
	}
	checkFiles(t, out, m.Files)
	if _, err := os.Stat(filepath.Join(out, "manifest.json")); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(out + ".pulling"); !os.IsNotExist(err) {
		t.Fatalf("the mark outlived the pull: %v", err)
	}
	if err := a.Wait(ctx, []string{"weights/rank-001/part-00000.bin"}); err != nil {
		t.Fatal(err)
	}
}

// A pull that fails after a restore started on it tells the restore,
// removes the partial directory and leaves the pieces it checked for the
// next pull.
func TestPullArrivalsFail(t *testing.T) {
	defer func(b time.Duration) { chunkBackoff = b }(chunkBackoff)
	chunkBackoff = time.Millisecond
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	o := Options{PieceSize: 8, Concurrency: 3}
	reg := twoRanks(t, o)
	reg.fail = true
	a := NewArrivals()
	o.Arrivals = a
	out := filepath.Join(t.TempDir(), "pulled", "a")
	s := store.Store{Root: filepath.Join(t.TempDir(), "state")}
	if _, err := Pull(ctx, reg, "v1", out, s, o); err == nil {
		t.Fatal("the failing weights went unnoticed")
	}
	if err := a.Done(ctx); err == nil || !strings.Contains(err.Error(), "connection reset") {
		t.Fatalf("the restore was told %v", err)
	}
	if err := a.Wait(ctx, []string{"weights/rank-000/part-00000.bin"}); err == nil {
		t.Fatal("a file of the failed pull is waited for")
	}
	for _, name := range []string{out, out + ".pulling"} {
		if _, err := os.Stat(name); !os.IsNotExist(err) {
			t.Fatalf("%s outlived the failed pull: %v", name, err)
		}
	}
	if left, _ := filepath.Glob(filepath.Join(s.Blobs(), ".assemble-*.pieces")); len(left) == 0 {
		t.Fatal("no assembly left to resume")
	}
	reg.fail = false
	close(reg.open)
	a = NewArrivals()
	o.Arrivals = a
	m, err := Pull(ctx, reg, "v1", out, s, o)
	if err != nil {
		t.Fatal(err)
	}
	checkFiles(t, out, m.Files)
}

// A file whose bytes another one of the snapshot shares, in the same bundle
// or not, arrives once that blob is stored, and a file the store has
// arrives at once.
func TestPullArrivesShared(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	o := Options{PieceSize: 8, Concurrency: 2}
	reg := twoRanks(t, o)
	close(reg.open)
	s := store.Store{Root: filepath.Join(t.TempDir(), "state")}
	first := filepath.Join(t.TempDir(), "first")
	if _, err := Pull(ctx, reg, "v1", first, s, o); err != nil {
		t.Fatal(err)
	}
	a := NewArrivals()
	o.Arrivals = a
	out := filepath.Join(t.TempDir(), "second")
	m, err := Pull(ctx, reg, "v1", out, s, o)
	if err != nil {
		t.Fatal(err)
	}
	if err := a.Wait(ctx, paths(m.Files)); err != nil {
		t.Fatal(err)
	}
	checkFiles(t, out, m.Files)
}

func TestArrivalOrder(t *testing.T) {
	m := manifest.Manifest{}
	var files []fileLayers
	for _, f := range []manifest.File{
		{Path: "criu/core-1.img", Role: manifest.RoleCRIU},
		{Path: "fs/workspace.tar", Role: manifest.RoleRuntimeFS},
		{Path: "weights/rank-000/part-00000.bin", Role: manifest.RoleWeights},
		{Path: "weights/rank-000/part-00001.bin", Role: manifest.RoleWeights},
		{Path: "weights/rank-000/table-emb.bin", Role: manifest.RoleHostTable},
		{Path: "weights/rank-001/part-00000.bin", Role: manifest.RoleWeights},
		{Path: "z/late-small-file", Role: manifest.RoleCRIU},
	} {
		m.Files = append(m.Files, f)
		files = append(files, fileLayers{files: []manifest.File{f}})
	}
	var got []string
	for _, fl := range arrivalOrder(m, files) {
		got = append(got, fl.files[0].Path)
	}
	want := "criu/core-1.img fs/workspace.tar z/late-small-file weights/rank-000/part-00000.bin " +
		"weights/rank-001/part-00000.bin weights/rank-000/part-00001.bin weights/rank-000/table-emb.bin"
	if strings.Join(got, " ") != want {
		t.Fatalf("order %v", got)
	}
}

func TestClaimOutput(t *testing.T) {
	dir := t.TempDir()
	out := filepath.Join(dir, "snap")
	// What a pull that died left: its mark and a partial directory.
	os.MkdirAll(filepath.Join(out, "criu"), 0o755)
	os.WriteFile(filepath.Join(out, "criu", "core-1.img"), []byte("partial"), 0o644)
	os.WriteFile(out+".pulling", nil, 0o644)
	release, err := claimOutput(out)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(out); !os.IsNotExist(err) {
		t.Fatalf("the partial directory stayed: %v", err)
	}
	if _, err := claimOutput(out); err == nil || !strings.Contains(err.Error(), "another pull") {
		t.Fatalf("a second pull into the same directory: %v", err)
	}
	release()
	// A directory without the mark is not this pull's to remove.
	os.MkdirAll(out, 0o755)
	os.WriteFile(filepath.Join(out, "notes"), []byte("mine"), 0o644)
	release, err = claimOutput(out)
	if err != nil {
		t.Fatal(err)
	}
	release()
	if _, err := os.Stat(filepath.Join(out, "notes")); err != nil {
		t.Fatalf("a directory the pull did not write was removed: %v", err)
	}
	os.WriteFile(filepath.Join(out, "manifest.json"), []byte("{}"), 0o644)
	if _, err := claimOutput(out); err == nil || !strings.Contains(err.Error(), "holds a snapshot") {
		t.Fatalf("a pull over a snapshot: %v", err)
	}
}
