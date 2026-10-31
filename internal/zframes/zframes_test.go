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
	ends, split, err := Encode(context.Background(), src, dst, 2, 1)
	if err != nil || len(ends) != 3 || split != 1 {
		t.Fatalf("ends %v split %d: %v", ends, split, err)
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
		out, err := d.Decode(stored[at:at+n], nil, size, 1)
		if err != nil || !bytes.Equal(out, body[offset:offset+size]) {
			t.Fatalf("frame %d: %v", i, err)
		}
	}
	if _, err := d.Decode(stored[:ends[0]], nil, Size+1, 1); err == nil {
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
		ends, _, err := Encode(context.Background(), src, dst, 4, 0)
		if err != nil || len(ends) != n/Size {
			t.Fatalf("%d bytes: ends %v: %v", n, ends, err)
		}
	}
}

// 16-bit values whose high bytes repeat and low bytes are noise, as BF16
// weights: split 2 is chosen, stores less than split 1, and decodes back,
// an odd tail included.
func TestSplit(t *testing.T) {
	dir := t.TempDir()
	body := make([]byte, Size+Size/3+1)
	r := rand.New(rand.NewSource(2))
	for i := range body {
		if i%2 == 1 {
			body[i] = byte(0x3c + r.Intn(3))
		} else {
			body[i] = byte(r.Intn(256))
		}
	}
	src := filepath.Join(dir, "weights")
	if err := os.WriteFile(src, body, 0o644); err != nil {
		t.Fatal(err)
	}
	sizes := map[int]int64{}
	for _, want := range []int{0, 1} {
		dst := filepath.Join(dir, "weights.zst")
		ends, split, err := Encode(context.Background(), src, dst, 2, want)
		if err != nil {
			t.Fatal(err)
		}
		if want == 0 && split != 2 {
			t.Fatalf("chose split %d", split)
		}
		sizes[split] = ends[len(ends)-1]
		stored, _ := os.ReadFile(dst)
		d, _ := NewDecoder()
		var got []byte
		for i := range ends {
			at, n, _, size := Frame(ends, int64(len(body)), i)
			out, err := d.Decode(stored[at:at+n], nil, size, split)
			if err != nil {
				t.Fatal(err)
			}
			got = append(got, out...)
		}
		d.Close()
		if !bytes.Equal(got, body) {
			t.Fatalf("split %d does not decode back", split)
		}
	}
	if sizes[2] >= sizes[1] {
		t.Fatalf("split 2 stores %d, split 1 %d", sizes[2], sizes[1])
	}
}
