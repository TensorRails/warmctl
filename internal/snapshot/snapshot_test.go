package snapshot

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/TensorRails/warmctl/v2/internal/agent"
	"github.com/TensorRails/warmctl/v2/internal/interpose"
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

// A donor container runs CRIU, a bake in place DMTCP; CRIU in place is
// refused until it is supported.
func TestBakeBackend(t *testing.T) {
	for _, c := range []struct {
		want    string
		inPlace bool
		got     string
	}{
		{"", false, agent.BackendCRIU},
		{"auto", false, agent.BackendCRIU},
		{"criu", false, agent.BackendCRIU},
		{"dmtcp", false, ""},
		{"auto", true, agent.BackendDMTCP},
		{"dmtcp", true, agent.BackendDMTCP},
		{"criu", true, ""},
		{"podman", true, ""},
	} {
		got, err := bakeBackend(c.want, c.inPlace)
		if got != c.got || (err != nil) != (c.got == "") {
			t.Errorf("bakeBackend(%q, %t) = %q %v", c.want, c.inPlace, got, err)
		}
	}
	if _, err := bakeBackend("criu", true); err == nil || !strings.Contains(err.Error(), "not supported yet") {
		t.Errorf("criu in place: %v", err)
	}
}

// A DMTCP engine maps the cuInterpose build where it is, by its real path,
// and the snapshot keeps a copy as its record.
func TestInterposerDMTCP(t *testing.T) {
	build := t.TempDir()
	for _, name := range interpose.Files {
		if err := os.WriteFile(filepath.Join(build, name), []byte(name), 0o755); err != nil {
			t.Fatal(err)
		}
	}
	link := filepath.Join(t.TempDir(), "cuinterpose")
	if err := os.Symlink(build, link); err != nil {
		t.Fatal(err)
	}
	b := &bake{o: Options{Cuinterpose: link}, dir: t.TempDir(), backend: agent.BackendDMTCP}
	b.launch.Profile.IPC, b.launch.Profile.Parallel.TP = interpose.Auto, 2
	if err := b.interposer(); err != nil {
		t.Fatal(err)
	}
	resolved, _ := filepath.EvalSymlinks(build)
	if b.cfg.IPC != interpose.Auto || b.cfg.IPCBuild != resolved ||
		b.cfg.IPCCoordinator != filepath.Join(resolved, interpose.Coordinator) {
		t.Fatalf("config %+v", b.cfg)
	}
	if err := interpose.Check(b.ipcDir); err != nil {
		t.Fatalf("the snapshot's copy: %v", err)
	}
	// CRIU mounts the copy at the guest path instead.
	b = &bake{o: Options{Cuinterpose: link}, dir: t.TempDir(), backend: agent.BackendCRIU}
	b.launch.Profile.IPC, b.launch.Profile.Parallel.TP = interpose.Interpose, 2
	if err := b.interposer(); err != nil {
		t.Fatal(err)
	}
	if b.cfg.IPCBuild != "" || b.cfg.IPCCoordinator != interpose.GuestDir+"/"+interpose.Coordinator {
		t.Fatalf("criu config %+v", b.cfg)
	}
}
