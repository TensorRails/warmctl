package run

import (
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

func TestLock(t *testing.T) {
	path := filepath.Join(t.TempDir(), "run", "warmctl.lock")
	first, err := lockFile(path)
	if err != nil {
		t.Fatal(err)
	}
	// flock conflicts between open files, in one process too.
	_, err = lockFile(path)
	if err == nil || !strings.Contains(err.Error(), "process "+strconv.Itoa(os.Getpid())+" runs in place") {
		t.Fatalf("second lock: %v", err)
	}
	first.Release()
	first.Release()
	second, err := lockFile(path)
	if err != nil {
		t.Fatalf("after release: %v", err)
	}
	second.Release()
	var none *Lock
	none.Release()
}
