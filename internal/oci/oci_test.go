package oci

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"oras.land/oras-go/v2/content/memory"
	"oras.land/oras-go/v2/registry/remote/retry"

	"github.com/TensorRails/warmctl/v2/internal/manifest"
	"github.com/TensorRails/warmctl/v2/internal/store"
)

// snapshotDir writes a small but valid snapshot.
func snapshotDir(t *testing.T, dir string, part string) {
	t.Helper()
	snapshotFiles(t, dir, map[string]testFile{
		"criu/core-32768.img":             {"core", manifest.RoleCRIU},
		"fs/workspace.tar":                {"tar", manifest.RoleRuntimeFS},
		"weights/rank-000/part-00000.bin": {part, manifest.RoleWeights},
	})
}

type testFile struct{ body, role string }

// snapshotFiles writes files and a manifest.json listing them; the one
// weights file is rank 0's only part.
func snapshotFiles(t *testing.T, dir string, files map[string]testFile) {
	t.Helper()
	m := manifest.Manifest{Schema: manifest.Schema, ID: "oci-test", CreatedAt: time.Unix(1760000000, 0).UTC(),
		Launch:  manifest.Launch{Argv: []string{"vllm", "serve", "/model"}, Ranks: 1},
		Process: manifest.Process{RootPID: 32768},
		Weights: manifest.Weights{PartBytes: 1 << 30, Ranks: []manifest.Rank{{Rank: 0}}}}
	for name, f := range files {
		p := filepath.Join(dir, name)
		os.MkdirAll(filepath.Dir(p), 0o755)
		if err := os.WriteFile(p, []byte(f.body), 0o644); err != nil {
			t.Fatal(err)
		}
		sum := sha256.Sum256([]byte(f.body))
		m.Files = append(m.Files, manifest.File{Path: name, Size: int64(len(f.body)), SHA256: hex.EncodeToString(sum[:]), Role: f.role})
	}
	sortFiles(m.Files)
	for _, f := range m.Files {
		if f.Role == manifest.RoleWeights {
			m.Weights.Ranks[0].Parts = []manifest.Part{{Index: 0, Path: f.Path, Size: f.Size, SHA256: f.SHA256}}
			m.Weights.Ranks[0].Allocations = []manifest.Allocation{{Index: 0, Ptr: "0x1", Bytes: f.Size, Tag: "weights"}}
		}
	}
	if err := m.Write(filepath.Join(dir, "manifest.json")); err != nil {
		t.Fatal(err)
	}
}

func sortFiles(files []manifest.File) {
	for i := range files {
		for j := i + 1; j < len(files); j++ {
			if files[j].Path < files[i].Path {
				files[i], files[j] = files[j], files[i]
			}
		}
	}
}

func TestPushPull(t *testing.T) {
	ctx := context.Background()
	root := t.TempDir()
	reg := memory.New()
	var logs []string
	o := Options{Log: func(f string, _ ...any) { logs = append(logs, strings.TrimSpace(f)) }}
	snapshotDir(t, filepath.Join(root, "a"), "weights-bytes")
	desc, err := Push(ctx, reg, filepath.Join(root, "a"), "v1", o)
	if err != nil {
		t.Fatal(err)
	}
	// A second snapshot sharing the weights part uploads only what differs.
	snapshotDir(t, filepath.Join(root, "b"), "weights-bytes")
	os.WriteFile(filepath.Join(root, "b", "criu", "core-32768.img"), []byte("core"), 0o644)
	if _, err := Push(ctx, reg, filepath.Join(root, "b"), "v2", o); err != nil {
		t.Fatal(err)
	}
	s := store.Store{Root: filepath.Join(root, "state")}
	out := filepath.Join(root, "pulled", "a")
	m, err := Pull(ctx, reg, "v1", out, s, o)
	if err != nil {
		t.Fatal(err)
	}
	if m.ID != "oci-test" || desc.Digest == "" {
		t.Fatalf("pulled %+v", m)
	}
	for _, f := range m.Files {
		raw, err := os.ReadFile(filepath.Join(out, f.Path))
		sum := sha256.Sum256(raw)
		if err != nil || hex.EncodeToString(sum[:]) != f.SHA256 {
			t.Fatalf("%s differs after pull: %v", f.Path, err)
		}
	}
	if _, err := os.Stat(filepath.Join(out, "manifest.json")); err != nil {
		t.Fatal(err)
	}
	// Pulling again into another directory downloads nothing.
	logs = nil
	if _, err := Pull(ctx, reg, "v1", filepath.Join(root, "pulled", "again"), s, o); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(strings.Join(logs, "\n"), "pulled") {
		t.Fatalf("logs %v", logs)
	}
	if _, err := Pull(ctx, reg, "v1", out, s, o); err == nil {
		t.Fatal("pulled over an existing directory")
	}
	if _, err := Pull(ctx, reg, "missing", filepath.Join(root, "x"), s, o); err == nil {
		t.Fatal("pulled a missing tag")
	}
}

func TestRef(t *testing.T) {
	repo, tag, err := Repository("oci://localhost:5000/snaps/qwen3-8b-tp1:h100", Options{PlainHTTP: true})
	if err != nil || tag != "h100" || repo.Reference.Repository != "snaps/qwen3-8b-tp1" || !repo.PlainHTTP {
		t.Fatalf("%+v %s %v", repo, tag, err)
	}
	if !IsRef("oci://x/y:z") || IsRef("/snapshots/x") {
		t.Fatal("IsRef")
	}
}

// The bulk transport speaks HTTP/1.1 to a server that also offers HTTP/2,
// one connection per parallel request.
func TestTransportHTTP1(t *testing.T) {
	srv := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(r.Proto))
	}))
	srv.EnableHTTP2 = true
	srv.StartTLS()
	defer srv.Close()
	tr := transport()
	inner := srv.Client().Transport.(*http.Transport)
	base := tr.(*retry.Transport).Base.(*http.Transport)
	base.TLSClientConfig.RootCAs = inner.TLSClientConfig.RootCAs
	resp, err := (&http.Client{Transport: tr}).Get(srv.URL)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	if string(body) != "HTTP/1.1" {
		t.Fatalf("served over %s", body)
	}
}
