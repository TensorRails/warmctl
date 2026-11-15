package accel

import (
	"context"
	"testing"

	"github.com/TensorRails/warmctl/v2/internal/manifest"
	"github.com/TensorRails/warmctl/v2/internal/rocm"
)

// NVIDIA moves nothing through the engine: its GPUState never calls it.
func TestNVIDIAStateIsNone(t *testing.T) {
	for _, v := range []string{"", NVIDIA} {
		s, err := For(v, "")
		if err != nil {
			t.Fatal(err)
		}
		if _, ok := s.(None); !ok {
			t.Fatalf("%q: %T", v, s)
		}
		// A nil engine: None must not touch it.
		if r, err := s.Checkpoint(context.Background(), nil); err != nil || len(r.Ranks) != 0 {
			t.Fatal(r, err)
		}
		if r, err := s.Restore(context.Background(), nil); err != nil || len(r.Ranks) != 0 {
			t.Fatal(r, err)
		}
	}
	if s, err := For(AMD, "0.1.1"); err != nil || s.(rocm.State).Version != "0.1.1" {
		t.Fatal(s, err)
	}
	if _, err := For("intel", ""); err == nil {
		t.Fatal("unknown vendor")
	}
}

func TestOf(t *testing.T) {
	if Of(manifest.Hardware{}) != NVIDIA || Of(manifest.Hardware{Vendor: AMD}) != AMD {
		t.Fatal("Of")
	}
}
