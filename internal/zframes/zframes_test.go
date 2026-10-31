package zframes

import (
	"bytes"
	"context"
	"math/rand"
	"os"
	"path/filepath"
	"testing"
)

// A file of two and a half frames, half of it compressible, comes back
// frame by frame in any order.
func TestEncodeDecode(t *testing.T) {
	dir := t.TempDir()
	body := make([]byte, 2*Size+Size/2)
	rand.New(rand.NewSource(1)).Read(body[:Size])
	src, dst := filepath.Join(dir, "image"), filepath.Join(dir, "image.zst")
	if err := os.WriteFile(src, body, 0o644); err != nil {
		t.Fatal(err)
	}
	ends, err := Encode(context.Background(), src, dst, 2)
	if err != nil || len(ends) != 3 {
		t.Fatalf("ends %v: %v", ends, err)
	}
	stored, err := os.ReadFile(dst)
	if err != nil || int64(len(stored)) != ends[2] || ends[2] >= int64(len(body)) {
		t.Fatalf("stored %d bytes, ends %v: %v", len(stored), ends, err)
	}
	d, err := NewDecoder()
	if err != nil {
		t.Fatal(err)
	}
	defer d.Close()
	for _, i := range []int{2, 0, 1} {
		at, n, offset, size := Frame(ends, int64(len(body)), i)
		out, err := d.Decode(stored[at:at+n], nil, size)
		if err != nil || !bytes.Equal(out, body[offset:offset+size]) {
			t.Fatalf("frame %d: %v", i, err)
		}
	}
	if _, err := d.Decode(stored[:ends[0]], nil, Size+1); err == nil {
		t.Fatal("a frame of the wrong size decoded")
	}
}

// A file of exactly one frame, and an empty one.
func TestEncodeEdges(t *testing.T) {
	dir := t.TempDir()
	for _, n := range []int{Size, 0} {
		src, dst := filepath.Join(dir, "f"), filepath.Join(dir, "f.zst")
		if err := os.WriteFile(src, make([]byte, n), 0o644); err != nil {
			t.Fatal(err)
		}
		ends, err := Encode(context.Background(), src, dst, 4)
		if err != nil || len(ends) != n/Size {
			t.Fatalf("%d bytes: ends %v: %v", n, ends, err)
		}
	}
}
