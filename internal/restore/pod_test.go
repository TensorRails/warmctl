package restore

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/TensorRails/warmctl/v2/internal/agent"
	"github.com/TensorRails/warmctl/v2/internal/docker"
	"github.com/TensorRails/warmctl/v2/internal/host"
	"github.com/TensorRails/warmctl/v2/internal/interpose"
	"github.com/TensorRails/warmctl/v2/internal/manifest"
	"github.com/TensorRails/warmctl/v2/internal/profile"
	"github.com/TensorRails/warmctl/v2/internal/pyext"
	"github.com/TensorRails/warmctl/v2/internal/runtimefs"
)

func TestCheckInit(t *testing.T) {
	for _, ok := range []string{"tini\n", "tini-static", "docker-init", "pause", "catatonit"} {
		if err := checkInit(ok, 7); err != nil {
			t.Errorf("%q refused: %v", ok, err)
		}
	}
	for comm, self := range map[string]int{"tini": 1, "systemd": 7, "init": 7, "bash": 7} {
		if err := checkInit(comm, self); err == nil {
			t.Errorf("PID 1 %q with warmctl as %d accepted", comm, self)
		}
	}
	if err := checkInit("tini", 40000); err == nil {
		t.Error("a PID in the snapshot's range accepted")
	}
}

func TestCheckStatus(t *testing.T) {
	for _, name := range docker.CheckpointCaps {
		if _, ok := capBits[name]; !ok {
			t.Fatalf("no capability number for %s", name)
		}
	}
	// A privileged container: every capability up to 40, no seccomp.
	privileged := "Name:\twarmctl\nCapEff:\t000001ffffffffff\nSeccomp:\t0\n"
	if err := checkStatus(privileged); err != nil {
		t.Fatal(err)
	}
	// Docker's default set lacks SYS_ADMIN, SYS_PTRACE and the rest.
	if err := checkStatus("CapEff:\t00000000a80425fb\nSeccomp:\t0\n"); err == nil ||
		!strings.Contains(err.Error(), "SYS_ADMIN") || !strings.Contains(err.Error(), "privileged") {
		t.Errorf("default capabilities: %v", err)
	}
	if err := checkStatus("CapEff:\t000001ffffffffff\nSeccomp:\t2\n"); err == nil || !strings.Contains(err.Error(), "Unconfined") {
		t.Errorf("a seccomp filter accepted: %v", err)
	}
}

func TestCheckTree(t *testing.T) {
	ws := agent.Tree{Name: "workspace", Path: "/workspace"}
	roomy := fsInfo{Free: 4 << 30}
	if err := checkTree(ws, roomy, 1<<30); err != nil {
		t.Fatal(err)
	}
	if err := checkTree(ws, fsInfo{Free: 4 << 30, ReadOnly: true}, 0); err == nil || !strings.Contains(err.Error(), "read-only") {
		t.Errorf("read-only: %v", err)
	}
	if err := checkTree(ws, roomy, 8<<30); err == nil || !strings.Contains(err.Error(), "needs 8.00 GiB") {
		t.Errorf("full: %v", err)
	}
}

// A receiver container has tmpfs at /tmp and /dev/shm, the profile's shm
// size; warmctl mounts the same in place and checks only the rest.
func TestTmpfsTrees(t *testing.T) {
	r := &restore{m: podManifest(false, "")}
	r.m.Profile.Resources.SHM = "16GiB"
	if size, ours := r.tmpfs("/dev/shm"); !ours || size != 16<<30 {
		t.Errorf("/dev/shm: %d %t", size, ours)
	}
	if size, ours := r.tmpfs("/tmp"); !ours || size != 1<<30 {
		t.Errorf("/tmp: %d %t", size, ours)
	}
	if _, ours := r.tmpfs("/workspace"); ours {
		t.Error("/workspace mounted as tmpfs")
	}
}

func TestCheckImage(t *testing.T) {
	const pinned = "vllm/vllm-openai:cu134-nightly@sha256:ad9d"
	for base, snap := range map[string]string{
		pinned:                         pinned,
		"vllm/vllm-openai@sha256:ad9d": pinned,
		"sha256:0123":                  "sha256:0123",
	} {
		if err := checkImage(base, snap); err != nil {
			t.Errorf("%s for %s: %v", base, snap, err)
		}
	}
	if err := checkImage("vllm/vllm-openai:latest", pinned); err == nil || !strings.Contains(err.Error(), "BASE="+pinned) {
		t.Errorf("another base accepted: %v", err)
	}
	if err := checkImage("vllm/vllm-openai@sha256:beef", pinned); err == nil {
		t.Error("another digest accepted")
	}
}

func TestCheckListen(t *testing.T) {
	for _, ok := range []string{":8080", "0.0.0.0:9000", "127.0.0.1:18000"} {
		if err := checkListen(ok); err != nil {
			t.Errorf("%s: %v", ok, err)
		}
	}
	for _, bad := range []string{":8000", "0.0.0.0:8000", "127.0.0.1:8000", "8080"} {
		if err := checkListen(bad); err == nil {
			t.Errorf("%s accepted", bad)
		}
	}
}

func TestMemoryLimit(t *testing.T) {
	if n, ok := memoryLimit("68719476736\n"); !ok || n != 64<<30 {
		t.Errorf("limit %d %t", n, ok)
	}
	if _, ok := memoryLimit("max\n"); ok {
		t.Error("max read as a limit")
	}
}

func TestVisibleGPUs(t *testing.T) {
	inventory := []host.GPU{{Index: 0, UUID: "GPU-a"}, {Index: 1, UUID: "GPU-b"}, {Index: 2, UUID: "GPU-c"}}
	got, err := VisibleGPUs([]string{"GPU-c", " GPU-a", ""}, inventory)
	if err != nil || got != "2,0" {
		t.Fatalf("UUIDs: %q %v", got, err)
	}
	if got, err := VisibleGPUs([]string{"1", "2"}, inventory); err != nil || got != "1,2" {
		t.Fatalf("indices: %q %v", got, err)
	}
	if _, err := VisibleGPUs([]string{"MIG-x"}, inventory); err == nil {
		t.Fatal("an unknown device accepted")
	}
}

func TestPodEnv(t *testing.T) {
	got := podEnv([]string{"PATH=/bin", "HOME=/root", "HF_HUB_OFFLINE=0"},
		map[string]string{"HOME": "/workspace/home", "HF_HUB_OFFLINE": "1", "A": ""})
	want := "PATH=/bin A= HF_HUB_OFFLINE=1 HOME=/workspace/home"
	if strings.Join(got, " ") != want {
		t.Fatalf("env %q", got)
	}
}

// A restore in place empties a directory only when it is empty or an
// earlier restore marked it: a volume of the operator's stays as it is.
func TestClaimDir(t *testing.T) {
	marked := func(dir string) bool {
		_, err := os.Stat(filepath.Join(dir, runtimefs.Mark))
		return err == nil
	}
	empty := t.TempDir()
	if err := claimDir(empty); err != nil || !marked(empty) {
		t.Fatalf("empty: %v", err)
	}
	// Marked by the claim above, then filled by the restore.
	os.MkdirAll(filepath.Join(empty, "logs", "deep"), 0o755)
	os.WriteFile(filepath.Join(empty, "logs", "engine.log"), []byte("x"), 0o644)
	if err := claimDir(empty); err != nil {
		t.Fatal(err)
	}
	if entries, _ := os.ReadDir(empty); len(entries) != 1 || !marked(empty) {
		t.Fatalf("an earlier restore's directory left %v", entries)
	}
	if err := claimDir(empty); err != nil || !marked(empty) {
		t.Fatalf("mark only: %v", err)
	}
	foreign := t.TempDir()
	os.WriteFile(filepath.Join(foreign, "dataset.parquet"), []byte("mine"), 0o644)
	if err := claimDir(foreign); err == nil || !strings.Contains(err.Error(), "mount an emptyDir at "+foreign) {
		t.Fatalf("foreign data: %v", err)
	}
	if raw, err := os.ReadFile(filepath.Join(foreign, "dataset.parquet")); err != nil || string(raw) != "mine" ||
		marked(foreign) {
		t.Fatal("foreign data touched")
	}
	missing := filepath.Join(t.TempDir(), "run")
	if err := claimDir(missing); err != nil || !marked(missing) {
		t.Fatalf("missing: %v", err)
	}
}

func TestCheckUTS(t *testing.T) {
	if err := checkUTS("uts:[4026532611]"); err != nil {
		t.Fatal(err)
	}
	if err := checkUTS("uts:[4026531838]"); err == nil || !strings.Contains(err.Error(), "hostNetwork") {
		t.Fatalf("the node's UTS namespace: %v", err)
	}
}

func podManifest(needsModel bool, ipc string) manifest.Manifest {
	return manifest.Manifest{
		Profile: profile.Profile{Model: profile.Model{ContainerPath: "/model"}},
		Launch:  manifest.Launch{ModelDir: "/donor/models/x"},
		Process: manifest.Process{NeedsModel: needsModel, IPC: ipc},
		Files:   []manifest.File{{Path: "fs/tmp.tar", Size: 1234}, {Path: "fs/tmp.json", Size: 10}},
	}
}

func TestEngineMounts(t *testing.T) {
	r := &restore{snapshot: "/snap", m: podManifest(true, interpose.Interpose)}
	got := r.engineMounts("/models/x")
	want := []docker.Mount{
		{Source: "/snap/adapter", Target: pyext.GuestPath, ReadOnly: true},
		{Source: "/snap/ipc", Target: interpose.GuestDir, ReadOnly: true},
		{Source: "/models/x", Target: "/model", ReadOnly: true},
	}
	if len(got) != len(want) {
		t.Fatalf("mounts %+v", got)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("mount %d: %+v, want %+v", i, got[i], want[i])
		}
	}
	// A pod that mounts the model where the profile wants it needs no bind.
	r.m = podManifest(true, "")
	if got := r.engineMounts("/model"); len(got) != 1 {
		t.Errorf("mounts %+v", got)
	}
	if n := treeBytes(r.m, "tmp"); n != 1234 {
		t.Errorf("tmp tree %d bytes", n)
	}
}

func TestPodModel(t *testing.T) {
	if got, err := podModel("/given", podManifest(true, "")); err != nil || got != "/given" {
		t.Errorf("given: %q %v", got, err)
	}
	if got, err := podModel("", podManifest(false, "")); err != nil || got != "" {
		t.Errorf("not needed: %q %v", got, err)
	}
	m := podManifest(true, "")
	m.Profile.Model.ContainerPath = t.TempDir()
	if _, err := podModel("", m); err == nil || !strings.Contains(err.Error(), "--model-dir") {
		t.Errorf("an empty model mount accepted: %v", err)
	}
	os.WriteFile(filepath.Join(m.Profile.Model.ContainerPath, "config.json"), []byte("{}"), 0o644)
	if got, err := podModel("", m); err != nil || got != "" {
		t.Errorf("mounted model: %q %v", got, err)
	}
}
