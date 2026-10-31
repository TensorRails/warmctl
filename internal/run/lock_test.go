package run

import (
	"os"
	"path/filepath"
	"strings"
	"syscall"
	"testing"
)

func TestLock(t *testing.T) {
	dir := t.TempDir()
	first, err := lockDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	// flock conflicts between open files, in one process too.
	if _, err := lockDir(dir); err == nil || !strings.Contains(err.Error(), "another warmctl run is in place") {
		t.Fatalf("second lock: %v", err)
	}
	first.Release()
	first.Release()
	// A read-only directory locks too: a flock writes nothing.
	if err := os.Chmod(dir, 0o555); err != nil {
		t.Fatal(err)
	}
	second, err := lockDir(dir)
	if err != nil {
		t.Fatalf("after release: %v", err)
	}
	second.Release()
	var none *Lock
	none.Release()
}

// Overlayfs refuses to rename an image layer's directory (EXDEV): Place
// copies it aside, links included, and removes it.
func TestPlaceAcrossLayers(t *testing.T) {
	defer func() { rename = os.Rename }()
	rename = func(string, string) error { return &os.LinkError{Op: "rename", Err: syscall.EXDEV} }
	dir, image := t.TempDir(), filepath.Join(t.TempDir(), "adapter")
	if err := os.MkdirAll(filepath.Join(image, "pkg"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(image, "pkg", "a.py"), []byte("x = 1\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink("pkg/a.py", filepath.Join(image, "link")); err != nil {
		t.Fatal(err)
	}
	if err := Place(dir, image); err != nil {
		t.Fatal(err)
	}
	if target, err := os.Readlink(image); err != nil || target != dir {
		t.Fatalf("link %q: %v", target, err)
	}
	if b, err := os.ReadFile(filepath.Join(image+SetAside, "link")); err != nil || string(b) != "x = 1\n" {
		t.Fatalf("set aside: %q %v", b, err)
	}
}
