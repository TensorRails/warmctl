package kfdreplay

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestPin(t *testing.T) {
	if Tag() != "v0.1.1" || Version() != "0.1.1" {
		t.Fatalf("tag %q version %q", Tag(), Version())
	}
	if len(SHA256("libkfdreplay.so")) != 64 || len(SHA256("libkfdreplay-"+Tag()+"-linux-amd64.tar.gz")) != 64 {
		t.Fatal("the release's assets are pinned")
	}
	if SHA256("other") != "" {
		t.Fatal("unknown asset")
	}
	t.Setenv(Env, "")
	if Path() != "/opt/warmctl/kfdreplay/v0.1.1/libkfdreplay.so" {
		t.Fatal(Path())
	}
}

func TestLocate(t *testing.T) {
	dir := t.TempDir()
	lib := filepath.Join(dir, "libkfdreplay.so")
	if err := os.WriteFile(lib, []byte("not the release"), 0o755); err != nil {
		t.Fatal(err)
	}
	// A research build named by the variable: taken with its own digest.
	t.Setenv(Env, lib)
	f, err := Locate()
	if err != nil || f.Path != lib || f.Version != "" || len(f.SHA256) != 64 || f.Size != 15 {
		t.Fatalf("%+v %v", f, err)
	}
	if err := Same(f); err != nil {
		t.Fatal(err)
	}
	// The same bytes with another mtime are another file for DMTCP.
	if err := os.Chtimes(lib, time.Now(), time.Now().Add(time.Second)); err != nil {
		t.Fatal(err)
	}
	if err := Same(f); err == nil || !strings.Contains(err.Error(), "mtime") {
		t.Fatalf("mtime: %v", err)
	}
	if err := os.WriteFile(lib, []byte("another build...."), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := Same(f); err == nil || !strings.Contains(err.Error(), "size") {
		t.Fatalf("size: %v", err)
	}
	// Missing: the error says where the image puts it.
	t.Setenv(Env, filepath.Join(dir, "missing.so"))
	if _, err := Locate(); err == nil || !strings.Contains(err.Error(), "runtime image") {
		t.Fatalf("missing: %v", err)
	}
}
