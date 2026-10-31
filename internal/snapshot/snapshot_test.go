package snapshot

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/TensorRails/warmctl/v2/internal/manifest"
	"github.com/TensorRails/warmctl/v2/internal/store"
	"github.com/TensorRails/warmctl/v2/internal/vllm"
)

// P201: publishing links, it does not copy; P148: two ranks may produce
// byte-identical parts, and one blob serves both paths.
func TestPublishLinksAndSharesDigests(t *testing.T) {
	ctx := context.Background()
	root := t.TempDir()
	write := func(name, body string) string {
		p := filepath.Join(root, "io", name)
		os.MkdirAll(filepath.Dir(p), 0o755)
		os.WriteFile(p, []byte(body), 0o644)
		return p
	}
	a := write("weights/rank-000/part-00000.bin", "same bytes")
	b := write("weights/rank-001/part-00000.bin", "same bytes")
	c := write("criu/core-32768.img", "core")
	sum, _, _ := store.HashFile(ctx, a)
	sources := []source{
		{path: a, rel: "weights/rank-000/part-00000.bin", role: manifest.RoleWeights, digest: sum},
		{path: b, rel: "weights/rank-001/part-00000.bin", role: manifest.RoleWeights, digest: sum},
		{path: c, rel: "criu/core-32768.img", role: manifest.RoleCRIU},
	}
	s := store.Store{Root: filepath.Join(root, "state")}
	out := filepath.Join(root, "snap")
	files, err := Publish(ctx, s, sources, out, 3)
	if err != nil {
		t.Fatal(err)
	}
	if len(files) != 3 || files[0].Path != "criu/core-32768.img" || files[1].SHA256 != files[2].SHA256 {
		t.Fatalf("files %+v", files)
	}
	blob, _ := s.BlobPath(sum)
	bi, _ := os.Stat(blob)
	for _, rel := range []string{"weights/rank-000/part-00000.bin", "weights/rank-001/part-00000.bin"} {
		oi, err := os.Stat(filepath.Join(out, rel))
		if err != nil || !os.SameFile(bi, oi) {
			t.Fatalf("%s is not a link to the blob: %v", rel, err)
		}
	}
	src, _ := os.Stat(c)
	pub, _ := os.Stat(filepath.Join(out, "criu/core-32768.img"))
	if !os.SameFile(src, pub) {
		t.Fatal("the CRIU image was copied, not linked")
	}
}

// After sleep the manifest keeps only the exported allocations the
// allocator still lists; indices are renumbered, parts and offsets kept.
func TestSurviving(t *testing.T) {
	exported := []manifest.Allocation{
		{Index: 0, Ptr: "0x1000", Bytes: 10, Tag: "weights", Part: 0, Offset: 0},
		{Index: 1, Ptr: "0x2000", Bytes: 20, Tag: "weights", Part: 0, Offset: 10},
		{Index: 2, Ptr: "0x3000", Bytes: 30, Tag: "weights", Part: 1, Offset: 5},
	}
	table := []vllm.Allocation{{Ptr: "0x1000", Bytes: 10, Tag: "weights", Asleep: true}, {Ptr: "0x3000", Bytes: 30, Tag: "weights", Asleep: true},
		{Ptr: "0x2000", Bytes: 21, Tag: "weights"}}
	kept, dropped := Surviving(exported, table)
	if dropped != 1 || len(kept) != 2 || kept[1].Ptr != "0x3000" || kept[1].Index != 1 || kept[1].Part != 1 || kept[1].Offset != 5 {
		t.Fatalf("kept %+v dropped %d", kept, dropped)
	}
}

func TestServedName(t *testing.T) {
	if n, err := ServedName(map[string]any{"served-model-name": []any{"glm53-flash", "x"}}); err != nil || n != "glm53-flash" {
		t.Fatal(n, err)
	}
	if _, err := ServedName(map[string]any{}); err == nil {
		t.Fatal("missing name accepted")
	}
}
