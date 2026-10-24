package oci

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	ocispec "github.com/opencontainers/image-spec/specs-go/v1"
	"oras.land/oras-go/v2/content"
	"oras.land/oras-go/v2/content/memory"

	"github.com/TensorRails/warmctl/v2/internal/manifest"
	"github.com/TensorRails/warmctl/v2/internal/store"
)

// A tiny piece size splits every file of the test snapshot.
func TestPushPullPieces(t *testing.T) {
	ctx := context.Background()
	root := t.TempDir()
	reg := memory.New()
	var logs []string
	o := Options{PieceSize: 4, Concurrency: 3, Log: func(f string, _ ...any) { logs = append(logs, f) }}
	src := filepath.Join(root, "a")
	snapshotDir(t, src, "weights-bytes-of-thirty-chars")
	desc, err := Push(ctx, reg, src, "v1", o)
	if err != nil {
		t.Fatal(err)
	}
	am := fetchManifest(t, reg, desc)
	pieces := 0
	for _, l := range am.Layers {
		if l.MediaType == PieceType {
			pieces++
		}
	}
	if pieces != 8 { // the weights part is 29 bytes: 8 pieces; "core" and "tar" fit one piece
		t.Fatalf("%d pieces in %d layers", pieces, len(am.Layers))
	}
	out := filepath.Join(root, "pulled")
	s := store.Store{Root: filepath.Join(root, "state")}
	m, err := Pull(ctx, reg, "v1", out, s, o)
	if err != nil {
		t.Fatal(err)
	}
	for _, f := range m.Files {
		raw, err := os.ReadFile(filepath.Join(out, f.Path))
		sum := sha256.Sum256(raw)
		if err != nil || hex.EncodeToString(sum[:]) != f.SHA256 {
			t.Fatalf("%s differs after pull: %v", f.Path, err)
		}
	}
	// A second push of the same files sends nothing.
	logs = nil
	if _, err := Push(ctx, reg, src, "v2", o); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(strings.Join(logs, "\n"), "pushed") {
		t.Fatalf("logs %v", logs)
	}
}

// A repeat push takes piece and bundle digests from the digests file and
// reads none of their bytes: scribbled over after the first push, the files
// still make the same artifact. Once manifest.json records other bytes, or
// the digests file is damaged, they are hashed again.
func TestRepushReadsNoPieces(t *testing.T) {
	ctx := context.Background()
	root := t.TempDir()
	reg := memory.New()
	var logs []string
	o := Options{PieceSize: 20, Concurrency: 3, Log: func(f string, a ...any) { logs = append(logs, fmt.Sprintf(f, a...)) }}
	src := filepath.Join(root, "a")
	snapshotFiles(t, src, map[string]testFile{
		"criu/core-32768.img":             {"core", manifest.RoleCRIU},
		"criu/fdinfo-2.img":               {"fd", manifest.RoleCRIU},
		"weights/rank-000/part-00000.bin": {"weights-bytes-of-thirty-chars", manifest.RoleWeights},
	})
	first, err := Push(ctx, reg, src, "v1", o)
	if err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"criu/core-32768.img", "criu/fdinfo-2.img", "weights/rank-000/part-00000.bin"} {
		p := filepath.Join(src, name)
		info, err := os.Stat(p)
		if err != nil || os.WriteFile(p, bytes.Repeat([]byte("x"), int(info.Size())), 0o644) != nil {
			t.Fatalf("scribble %s: %v", name, err)
		}
	}
	logs = nil
	again, err := Push(ctx, reg, src, "v2", o)
	if err != nil || again.Digest != first.Digest || !strings.Contains(strings.Join(logs, "\n"), " 0 uploaded") {
		t.Fatalf("repeat push %s, first %s: %v %v", again.Digest, first.Digest, logs, err)
	}

	snapshotFiles(t, src, map[string]testFile{
		"criu/core-32768.img":             {"CORE", manifest.RoleCRIU},
		"criu/fdinfo-2.img":               {"fd", manifest.RoleCRIU},
		"weights/rank-000/part-00000.bin": {"weights-bytes-of-thirty-CHARS", manifest.RoleWeights},
	})
	changed, err := Push(ctx, reg, src, "v3", o)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(src, DigestsFile), []byte("{"), 0o644); err != nil {
		t.Fatal(err)
	}
	if d, err := Push(ctx, reg, src, "v4", o); err != nil || d.Digest != changed.Digest {
		t.Fatalf("push over a damaged digests file: %s, want %s: %v", d.Digest, changed.Digest, err)
	}
	out := filepath.Join(root, "pulled")
	m, err := Pull(ctx, reg, "v3", out, store.Store{Root: filepath.Join(root, "state")}, o)
	if err != nil {
		t.Fatal(err)
	}
	for _, f := range m.Files {
		raw, err := os.ReadFile(filepath.Join(out, f.Path))
		sum := sha256.Sum256(raw)
		if err != nil || hex.EncodeToString(sum[:]) != f.SHA256 {
			t.Fatalf("%s differs after pull: %v", f.Path, err)
		}
	}
}

func TestPieceCorruptionCaught(t *testing.T) {
	ctx := context.Background()
	root := t.TempDir()
	src := filepath.Join(root, "a")
	snapshotDir(t, src, "weights-bytes-of-thirty-chars")
	m, err := manifest.Read(filepath.Join(src, "manifest.json"))
	if err != nil {
		t.Fatal(err)
	}
	sources := newSources(src, m.Files)
	layers, err := describe(ctx, sources, Options{PieceSize: 4})
	if err != nil {
		t.Fatal(err)
	}
	reg := memory.New()
	for _, d := range layers {
		body := readLayer(t, sources, d)
		if d.MediaType == PieceType && d.Annotations[AnnotationPieceOffset] == "8" {
			body = bytes.Repeat([]byte("x"), len(body)) // the registry serves other bytes
		}
		if err := reg.Push(ctx, d, bytes.NewReader(body)); err != nil && !strings.Contains(err.Error(), "digest") {
			t.Fatal(err)
		}
	}
	for _, fl := range mustPlan(t, m, layers) {
		if fl.layers[0].MediaType != PieceType || fl.files[0].Role != manifest.RoleWeights {
			continue
		}
		a, err := newAssembly(ctx, store.Store{Root: filepath.Join(root, "state")}, fl.files[0], fl.layers)
		if err != nil {
			t.Fatal(err)
		}
		defer a.abort()
		var failed bool
		for _, d := range fl.layers {
			if err := a.fetchPiece(ctx, reg, d); err != nil {
				failed = true
			}
		}
		if !failed {
			t.Fatal("a corrupted piece was accepted")
		}
	}
}

func TestPlanRejectsBadCoverage(t *testing.T) {
	f := manifest.File{Path: "criu/pages-1.img", Size: 8, SHA256: strings.Repeat("a", 64), Role: manifest.RoleCRIU}
	m := manifest.Manifest{Files: []manifest.File{f}}
	piece := func(off, size int64) ocispec.Descriptor {
		return ocispec.Descriptor{MediaType: PieceType, Size: size, Annotations: map[string]string{
			AnnotationPieceOf: f.Path, AnnotationPieceOffset: strconv.FormatInt(off, 10)}}
	}
	if _, err := plan(m, []ocispec.Descriptor{piece(0, 4), piece(4, 4)}); err != nil {
		t.Fatalf("exact cover refused: %v", err)
	}
	for name, layers := range map[string][]ocispec.Descriptor{
		"gap":     {piece(0, 4), piece(5, 3)},
		"short":   {piece(0, 4)},
		"long":    {piece(0, 4), piece(4, 8)},
		"extra":   {piece(0, 4), piece(4, 4), piece(8, 1)},
		"missing": {},
	} {
		if _, err := plan(m, layers); err == nil {
			t.Errorf("%s accepted", name)
		}
	}
}

// Small files travel back to back in bundle layers; a file twice in one
// bundle is stored once, and every file comes out byte for byte.
func TestPushPullBundles(t *testing.T) {
	ctx := context.Background()
	root := t.TempDir()
	reg := memory.New()
	src := filepath.Join(root, "a")
	snapshotFiles(t, src, map[string]testFile{
		"criu/core-32768.img":             {"core", manifest.RoleCRIU},
		"criu/fdinfo-2.img":               {"fd", manifest.RoleCRIU},
		"criu/fdinfo-3.img":               {"fd", manifest.RoleCRIU},
		"criu/pages-1.img":                {"0123456789abcdefghij-pages", manifest.RoleCRIU},
		"fs/workspace.tar":                {"tar", manifest.RoleRuntimeFS},
		"weights/rank-000/part-00000.bin": {"weights-bytes-of-thirty-chars", manifest.RoleWeights},
	})
	o := Options{PieceSize: 20, Concurrency: 3}
	desc, err := Push(ctx, reg, src, "v1", o)
	if err != nil {
		t.Fatal(err)
	}
	var kinds []string
	for _, l := range fetchManifest(t, reg, desc).Layers {
		kinds = append(kinds, l.MediaType[strings.LastIndex(l.MediaType, ".")+1:]+"/"+l.Annotations[AnnotationBundleFiles])
	}
	// core, both fdinfo: one bundle; pages and weights: two pieces each,
	// tar alone between them: its own layer.
	if got := strings.Join(kinds, " "); got != "bundle/3 piece/ piece/ file/ piece/ piece/" {
		t.Fatalf("layers %s", got)
	}
	out := filepath.Join(root, "pulled")
	m, err := Pull(ctx, reg, "v1", out, store.Store{Root: filepath.Join(root, "state")}, o)
	if err != nil {
		t.Fatal(err)
	}
	for _, f := range m.Files {
		raw, err := os.ReadFile(filepath.Join(out, f.Path))
		sum := sha256.Sum256(raw)
		if err != nil || hex.EncodeToString(sum[:]) != f.SHA256 {
			t.Fatalf("%s differs after pull: %v", f.Path, err)
		}
	}
}

func TestBundleCorruptionCaught(t *testing.T) {
	ctx := context.Background()
	root := t.TempDir()
	src := filepath.Join(root, "a")
	snapshotFiles(t, src, map[string]testFile{
		"criu/core-32768.img": {"core", manifest.RoleCRIU},
		"criu/fdinfo-2.img":   {"fd2", manifest.RoleCRIU},
	})
	m, err := manifest.Read(filepath.Join(src, "manifest.json"))
	if err != nil {
		t.Fatal(err)
	}
	sources := newSources(src, m.Files)
	layers, err := describe(ctx, sources, Options{PieceSize: 64})
	if err != nil || len(layers) != 1 || layers[0].MediaType != BundleType {
		t.Fatalf("layers %v: %v", layers, err)
	}
	d := layers[0]
	body := readLayer(t, sources, d)
	body[len(body)-1] ^= 1
	d.Digest = content.NewDescriptorFromBytes("", body).Digest // a registry that serves its own bytes
	reg := memory.New()
	if err := reg.Push(ctx, d, bytes.NewReader(body)); err != nil {
		t.Fatal(err)
	}
	s := store.Store{Root: filepath.Join(root, "state")}
	want := map[string]bool{m.Files[0].SHA256: true, m.Files[1].SHA256: true}
	write := func(sha string, r io.Reader, size int64) error { return s.Ingest(ctx, r, sha, size) }
	if err := fetchBundle(ctx, reg, d, m.Files, want, s.Has, write); err == nil {
		t.Fatal("a corrupted bundle was accepted")
	}
	if !s.Has(m.Files[0].SHA256) || s.Has(m.Files[1].SHA256) {
		t.Fatal("the intact file should be stored, the corrupted one not")
	}
}

func TestPlanRejectsBadBundle(t *testing.T) {
	a := manifest.File{Path: "criu/a.img", Size: 2, SHA256: strings.Repeat("a", 64), Role: manifest.RoleCRIU}
	b := manifest.File{Path: "criu/b.img", Size: 3, SHA256: strings.Repeat("b", 64), Role: manifest.RoleCRIU}
	m := manifest.Manifest{Files: []manifest.File{a, b}}
	bundle := func(of, n string, size int64) ocispec.Descriptor {
		return ocispec.Descriptor{MediaType: BundleType, Size: size, Annotations: map[string]string{
			AnnotationBundleOf: of, AnnotationBundleFiles: n}}
	}
	if _, err := plan(m, []ocispec.Descriptor{bundle(a.Path, "2", 5)}); err != nil {
		t.Fatalf("exact bundle refused: %v", err)
	}
	for name, layers := range map[string][]ocispec.Descriptor{
		"size":     {bundle(a.Path, "2", 6)},
		"too many": {bundle(a.Path, "3", 5)},
		"zero":     {bundle(a.Path, "0", 0), bundle(a.Path, "2", 5)},
		"start":    {bundle(b.Path, "2", 5)},
		"short":    {bundle(a.Path, "1", 2)},
	} {
		if _, err := plan(m, layers); err == nil {
			t.Errorf("%s accepted", name)
		}
	}
}

func mustPlan(t *testing.T, m manifest.Manifest, layers []ocispec.Descriptor) []fileLayers {
	t.Helper()
	files, err := plan(m, layers)
	if err != nil {
		t.Fatal(err)
	}
	return files
}

func readLayer(t *testing.T, src sources, d ocispec.Descriptor) []byte {
	t.Helper()
	r, err := src.open(d)()
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	var b bytes.Buffer
	if _, err := b.ReadFrom(r); err != nil {
		t.Fatal(err)
	}
	return b.Bytes()
}

func fetchManifest(t *testing.T, reg *memory.Store, desc ocispec.Descriptor) ocispec.Manifest {
	t.Helper()
	raw, err := content.FetchAll(context.Background(), reg, desc)
	if err != nil {
		t.Fatal(err)
	}
	var am ocispec.Manifest
	if err := json.Unmarshal(raw, &am); err != nil {
		t.Fatal(err)
	}
	return am
}

// Two pulls on one host never assemble the same file at once.
func TestAssemblyLocked(t *testing.T) {
	ctx := context.Background()
	s := store.Store{Root: t.TempDir()}
	f := manifest.File{Path: "criu/pages-1.img", Size: 8, SHA256: strings.Repeat("c", 64)}
	a, err := newAssembly(ctx, s, f, nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := newAssembly(ctx, s, f, nil); err == nil || !strings.Contains(err.Error(), "another pull") {
		t.Fatalf("a second assembly of the same file: %v", err)
	}
	a.abort()
	b, err := newAssembly(ctx, s, f, nil)
	if err != nil {
		t.Fatalf("the lock outlived the first assembly: %v", err)
	}
	b.abort()
}
