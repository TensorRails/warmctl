package pyext

import (
	"os"
	"path/filepath"
	"testing"
)

func TestMaterialize(t *testing.T) {
	root := t.TempDir()
	dir, sum, err := Materialize(root)
	if err != nil || len(sum) != 64 {
		t.Fatal(dir, sum, err)
	}
	for _, name := range []string{"warmctl_vllm/__init__.py", "warmctl_vllm/worker.py", "warmctl_vllm/pool.py", ".complete"} {
		if _, err := os.Stat(filepath.Join(dir, name)); err != nil {
			t.Error(err)
		}
	}
	again, sum2, err := Materialize(root)
	if err != nil || again != dir || sum2 != sum {
		t.Fatal("not reused")
	}
}
