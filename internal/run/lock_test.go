package run

import (
	"os"
	"strings"
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
