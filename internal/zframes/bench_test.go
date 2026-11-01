package zframes

import (
	"context"
	"math/rand"
	"os"
	"path/filepath"
	"testing"
)

// BF16-like data, 8 frames: the bake's case.
func BenchmarkEncode(b *testing.B) {
	dir := b.TempDir()
	body := make([]byte, 8*Size)
	r := rand.New(rand.NewSource(1))
	for i := 0; i < len(body); i += 2 {
		body[i], body[i+1] = byte(r.Intn(256)), byte(0x3c+r.Intn(3))
	}
	src := filepath.Join(dir, "w")
	if err := os.WriteFile(src, body, 0o644); err != nil {
		b.Fatal(err)
	}
	b.SetBytes(int64(len(body)))
	for range b.N {
		if _, _, err := Encode(context.Background(), src, src+".zst", 4, 0); err != nil {
			b.Fatal(err)
		}
	}
}
