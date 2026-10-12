package restore

import (
	"context"
	"os"
	"path/filepath"
	"reflect"
	"testing"
	"time"

	"github.com/TensorRails/warmctl/v2/internal/manifest"
)

// The CRIU images come first, then the weights one part per rank in turn;
// big files split into chunks; the budget cuts the tail.
func TestPrefetchPlan(t *testing.T) {
	criu := []manifest.File{{Path: "criu/pages-1.img", Size: 10}, {Path: "criu/core.img", Size: 3}}
	weights := []manifest.File{
		{Path: "weights/rank-000/part-00000.bin", Size: 4}, {Path: "weights/rank-000/part-00001.bin", Size: 4},
		{Path: "weights/rank-001/part-00000.bin", Size: 4}, {Path: "weights/rank-001/part-00001.bin", Size: 4},
	}
	var got []string
	for _, j := range prefetchPlan(criu, weights, 4, 0) {
		got = append(got, filepath.Base(filepath.Dir(j.path))+"/"+filepath.Base(j.path))
	}
	want := []string{"criu/pages-1.img", "criu/pages-1.img", "criu/pages-1.img", "criu/core.img",
		"rank-000/part-00000.bin", "rank-001/part-00000.bin", "rank-000/part-00001.bin", "rank-001/part-00001.bin"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("order %v, want %v", got, want)
	}
	if jobs := prefetchPlan(criu, weights, 4, 13); len(jobs) != 4 {
		t.Errorf("budget 13 keeps the CRIU images only, got %d jobs", len(jobs))
	}
	if last := prefetchPlan(criu, weights, 4, 0)[2]; last.offset != 8 || last.size != 2 {
		t.Errorf("last chunk of a 10-byte file at 8+2, got %d+%d", last.offset, last.size)
	}
}

// Prefetch reads every planned byte of a snapshot directory and returns.
func TestPrefetchReadsSnapshot(t *testing.T) {
	dir := t.TempDir()
	m := manifest.Manifest{Files: []manifest.File{
		{Path: "criu/pages-1.img", Size: 3 << 20, Role: manifest.RoleCRIU},
		{Path: "weights/rank-000/part-00000.bin", Size: 5 << 20, Role: manifest.RoleWeights},
	}}
	for _, f := range m.Files {
		p := filepath.Join(dir, f.Path)
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, make([]byte, f.Size), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	for _, order := range []string{"", "together"} {
		t.Setenv("WARMCTL_PREFETCH_WORKERS", "3")
		t.Setenv("WARMCTL_PREFETCH_ORDER", order)
		st := &prefetchStats{start: time.Now()}
		Prefetch(context.Background(), dir, m, st)
		if got := st.bytes.Load(); got != 8<<20 {
			t.Errorf("order %q: read %d bytes, want %d", order, got, 8<<20)
		}
		if st.criuDone.Load() == 0 || st.done.Load() == 0 {
			t.Errorf("order %q: stats not finished: %s", order, st)
		}
	}
}
