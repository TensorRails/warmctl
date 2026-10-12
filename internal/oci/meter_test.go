package oci

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"oras.land/oras-go/v2/content/memory"

	"github.com/TensorRails/warmctl/v2/internal/manifest"
	"github.com/TensorRails/warmctl/v2/internal/store"
)

// pushed returns a registry holding a small snapshot tagged v1.
func pushed(t *testing.T) *memory.Store {
	t.Helper()
	src := filepath.Join(t.TempDir(), "a")
	snapshotDir(t, src, "weights-bytes-of-thirty-chars")
	reg := memory.New()
	if _, err := Push(context.Background(), reg, src, "v1", Options{}); err != nil {
		t.Fatal(err)
	}
	return reg
}

// A host the snapshot cannot restore on, or a disk too small, fails the
// pull before any blob is downloaded.
func TestPullChecksBeforeDownload(t *testing.T) {
	reg := pushed(t)
	s := store.Store{Root: filepath.Join(t.TempDir(), "state")}
	refuse := errors.New("not this host")
	var seen manifest.Manifest
	o := Options{Check: func(m manifest.Manifest) error { seen = m; return refuse }}
	if _, err := Pull(context.Background(), reg, "v1", filepath.Join(t.TempDir(), "out"), s, o); !errors.Is(err, refuse) {
		t.Fatalf("pull went on: %v", err)
	}
	if seen.ID != "oci-test" {
		t.Fatalf("check saw %+v", seen)
	}
	defer func(f func(string) (int64, error)) { freeBytes = f }(freeBytes)
	freeBytes = func(string) (int64, error) { return 10, nil }
	_, err := Pull(context.Background(), reg, "v1", filepath.Join(t.TempDir(), "out"), s, Options{})
	if err == nil || !strings.Contains(err.Error(), "GiB more in") {
		t.Fatalf("a full disk was not caught: %v", err)
	}
	if blobs, _ := s.List(); len(blobs) != 0 {
		t.Fatalf("%d blobs downloaded before the checks", len(blobs))
	}
	if _, err := os.Stat(filepath.Join(s.Root, "blobs")); err != nil {
		t.Fatalf("store not prepared: %v", err)
	}
}

func TestMeter(t *testing.T) {
	defer func(d time.Duration) { meterEvery = d }(meterEvery)
	meterEvery = time.Millisecond
	var mu sync.Mutex
	var lines []string
	o := Options{Log: func(format string, a ...any) {
		mu.Lock()
		lines = append(lines, fmt.Sprintf(format, a...))
		mu.Unlock()
	}}
	got := &tally{}
	got.add(1 << 30)
	stop := meter(o, "pulled", 4<<30, got)
	deadline := time.Now().Add(5 * time.Second)
	for {
		mu.Lock()
		n := len(lines)
		mu.Unlock()
		if n > 0 || time.Now().After(deadline) {
			break
		}
		time.Sleep(time.Millisecond)
	}
	stop()
	mu.Lock()
	defer mu.Unlock()
	if len(lines) == 0 || !strings.HasPrefix(lines[0], "pulled 1.0 of 4.0 GiB, ") || !strings.Contains(lines[0], "ETA ") {
		t.Fatalf("lines %q", lines)
	}
}
