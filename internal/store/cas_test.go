package store

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"testing"
)

const abc = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"

func TestAddLinkGC(t *testing.T) {
	ctx := context.Background()
	root := t.TempDir()
	s := Store{Root: root}
	src := filepath.Join(root, "in", "abc.txt")
	if err := os.MkdirAll(filepath.Dir(src), 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(src, []byte("abc"), 0o644); err != nil {
		t.Fatal(err)
	}
	digest, size, err := s.Add(ctx, src)
	if err != nil || digest != abc || size != 3 {
		t.Fatalf("add: %s %d %v", digest, size, err)
	}
	if !s.Has(abc) {
		t.Fatal("blob missing")
	}
	blob, _ := s.BlobPath(abc)
	info, _ := os.Stat(blob)
	if info.Mode().Perm() != 0o444 {
		t.Fatalf("blob mode %v", info.Mode())
	}
	if _, _, err := s.Add(ctx, src); err != nil {
		t.Fatal(err)
	}
	dst := filepath.Join(root, "out", "copy.txt")
	if err := s.Link(ctx, abc, dst); err != nil {
		t.Fatal(err)
	}
	a, _ := os.Stat(blob)
	b, _ := os.Stat(dst)
	if !os.SameFile(a, b) {
		t.Fatal("link is not a hard link")
	}
	if err := s.Link(ctx, abc, dst); err == nil {
		t.Fatal("linking over an existing file succeeded")
	}
	list, _ := s.List()
	if len(list) != 1 || list[0] != abc {
		t.Fatalf("list %v", list)
	}
	freed, err := s.GC(ctx, map[string]bool{abc: true})
	if err != nil || freed != 0 || !s.Has(abc) {
		t.Fatalf("gc with keep: %d %v", freed, err)
	}
	freed, err = s.GC(ctx, nil)
	if err != nil || freed != 3 || s.Has(abc) {
		t.Fatalf("gc: %d %v", freed, err)
	}
	if raw, err := os.ReadFile(dst); err != nil || string(raw) != "abc" {
		t.Fatal("linked copy lost its data")
	}
}

func TestCopyFile(t *testing.T) {
	root := t.TempDir()
	src := filepath.Join(root, "a")
	if err := os.WriteFile(src, []byte("abc"), 0o644); err != nil {
		t.Fatal(err)
	}
	digest, n, err := CopyFile(context.Background(), src, filepath.Join(root, "b"))
	if err != nil || digest != abc || n != 3 {
		t.Fatalf("%s %d %v", digest, n, err)
	}
	if _, _, err := CopyFile(context.Background(), src, filepath.Join(root, "b")); err == nil {
		t.Fatal("overwrote an existing file")
	}
	if _, err := (Store{Root: root}).BlobPath("zz"); err == nil {
		t.Fatal("accepted a bad digest")
	}
}

// Concurrent writers of the same content end with one blob that every
// linked path shares.
func TestConcurrentSameDigest(t *testing.T) {
	ctx := context.Background()
	root := t.TempDir()
	s := Store{Root: filepath.Join(root, "state")}
	const n = 16
	errs := make(chan error, n)
	for i := 0; i < n; i++ {
		go func(i int) {
			src := filepath.Join(root, "in", fmt.Sprint(i))
			os.MkdirAll(filepath.Dir(src), 0o755)
			os.WriteFile(src, []byte("abc"), 0o644)
			if err := s.AddHashed(ctx, src, abc); err != nil {
				errs <- err
				return
			}
			errs <- s.Link(ctx, abc, filepath.Join(root, "out", fmt.Sprint(i)))
		}(i)
	}
	for i := 0; i < n; i++ {
		if err := <-errs; err != nil {
			t.Fatal(err)
		}
	}
	blob, _ := s.BlobPath(abc)
	bi, _ := os.Stat(blob)
	for i := 0; i < n; i++ {
		oi, _ := os.Stat(filepath.Join(root, "out", fmt.Sprint(i)))
		if !os.SameFile(bi, oi) {
			t.Fatalf("path %d does not share the blob", i)
		}
	}
	if entries, _ := os.ReadDir(s.Blobs()); len(entries) != 1 {
		t.Fatalf("temporary files left: %d entries", len(entries))
	}
}
