package rocm

import (
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/TensorRails/warmctl/v2"
	"github.com/TensorRails/warmctl/v2/internal/host"
	"github.com/TensorRails/warmctl/v2/internal/manifest"
	"github.com/TensorRails/warmctl/v2/internal/profile"
)

var mi350x = host.GPU{Index: 0, UUID: "GPU-1111111111111111", Name: "AMD Instinct MI350X", PCIBusID: "0000:09:00.0",
	MemoryMiB: 294896, Driver: "7.1.3.31500000", Vendor: host.VendorAMD, Arch: "gfx950", KFDGPUID: 41001,
	RenderNode: "/dev/dri/renderD136"}

func TestSupported(t *testing.T) {
	if note, err := Supported([]host.GPU{mi350x}); err != nil || note != "" {
		t.Fatalf("the verified platform: %q %v", note, err)
	}
	other := mi350x
	other.Driver = "7.2.0.1"
	if note, err := Supported([]host.GPU{other}); err != nil || !strings.Contains(note, "7.1.3") {
		t.Fatalf("another driver warns: %q %v", note, err)
	}
	mi300 := mi350x
	mi300.Name, mi300.Arch = "AMD Instinct MI300X", "gfx942"
	if _, err := Supported([]host.GPU{mi300}); err == nil || !strings.Contains(err.Error(), "gfx950") {
		t.Fatalf("an unverified ISA is refused: %v", err)
	}
}

func amdLaunch(t *testing.T, id string, set ...string) (profile.Launch, error) {
	t.Helper()
	l := profile.Loader{Assets: warmctl.Assets, Vendor: "amd"}
	defaults, err := l.Defaults()
	if err != nil {
		t.Fatal(err)
	}
	p, err := l.Load(id)
	if err != nil {
		t.Fatal(err)
	}
	for _, s := range set {
		if s == "tp=2" {
			p.Parallel.TP = 2
			continue
		}
		if err := p.Set(s); err != nil {
			t.Fatal(err)
		}
	}
	return profile.Resolve(defaults, p)
}

// The ROCm layer: what libkfdreplay needs, and no flag the ROCm vLLM lacks
// (vLLM 0.29 has no --enable-nccl-comm-suspend and refuses to start with it).
func TestROCmLayer(t *testing.T) {
	l, err := amdLaunch(t, "qwen3-8b-mi350x-tp1")
	if err != nil {
		t.Fatal(err)
	}
	for _, a := range l.Argv {
		if strings.Contains(a, "nccl-comm-suspend") {
			t.Errorf("argv has %s", a)
		}
	}
	for _, want := range []string{"--enable-sleep-mode", "--worker-extension-cls=warmctl_vllm.worker.Extension"} {
		found := false
		for _, a := range l.Argv {
			found = found || a == want
		}
		if !found {
			t.Errorf("argv lacks %s", want)
		}
	}
	// The image's own PYTHONPATH is added at bake time (snapshot.imagePythonPath).
	if l.Env["PYTHONPATH"] != "/opt/warmctl/adapter" {
		t.Errorf("PYTHONPATH %q", l.Env["PYTHONPATH"])
	}
	if l.Env["HSA_ENABLE_SDMA"] != "0" || l.Env["HOME"] != "/workspace/home" || l.Env["MIOPEN_USER_DB_PATH"] == "" {
		t.Errorf("env %v", l.Env)
	}
	if !strings.HasPrefix(l.Profile.Image, "rocm/vllm:") || l.Profile.IPC != "native" {
		t.Errorf("image %s ipc %s", l.Profile.Image, l.Profile.IPC)
	}
}

func TestCheckLaunch(t *testing.T) {
	for _, id := range []string{"qwen3-0.6b-mi350x-tp1", "qwen3-8b-mi350x-tp1"} {
		l, err := amdLaunch(t, id)
		if err != nil {
			t.Fatal(err)
		}
		if err := CheckLaunch(l); err != nil {
			t.Errorf("%s: %v", id, err)
		}
	}
	// A shipped NVIDIA profile, laid over the ROCm defaults, also runs.
	if l, err := amdLaunch(t, "qwen3-8b-tp1"); err != nil || CheckLaunch(l) != nil {
		t.Errorf("an NVIDIA profile on an AMD host: %v %v", err, CheckLaunch(l))
	}
	for set, want := range map[string]string{
		"tp=2":                  "one GPU per engine",
		"env.HSA_ENABLE_SDMA=1": "HSA_ENABLE_SDMA=0",
		"env.PYTORCH_HIP_ALLOC_CONF=expandable_segments:True": "PYTORCH_HIP_ALLOC_CONF",
	} {
		l, err := amdLaunch(t, "qwen3-0.6b-mi350x-tp1", set)
		if err == nil {
			err = CheckLaunch(l)
		}
		if err == nil || !strings.Contains(err.Error(), want) {
			t.Errorf("%s: %v, want %q", set, err, want)
		}
	}
}

func amdHardware() manifest.Hardware {
	return manifest.Hardware{Vendor: manifest.VendorAMD, Driver: mi350x.Driver, GPUs: []manifest.GPU{{
		Rank: 0, UUID: mi350x.UUID, Name: mi350x.Name, MemoryMiB: mi350x.MemoryMiB, Arch: "gfx950",
		KFDGPUID: 41001, Device: "/dev/dri/renderD136"}}}
}

func TestCompatible(t *testing.T) {
	h := amdHardware()
	if err := Compatible(h, []host.GPU{mi350x}); err != nil {
		t.Fatal(err)
	}
	other := mi350x
	other.KFDGPUID, other.RenderNode, other.UUID = 41002, "/dev/dri/renderD144", "GPU-other"
	if err := Compatible(h, []host.GPU{other}); err != nil {
		t.Fatalf("another GPU of the model restores (libkfdreplay translates gpu_id): %v", err)
	}
	for name, change := range map[string]func(*host.GPU){
		"driver": func(g *host.GPU) { g.Driver = "7.1.4" },
		"model":  func(g *host.GPU) { g.Name = "AMD Instinct MI355X" },
		"memory": func(g *host.GPU) { g.MemoryMiB-- },
		"isa":    func(g *host.GPU) { g.Arch = "gfx942" },
		"nvidia": func(g *host.GPU) { g.Vendor = "" },
	} {
		g := mi350x
		change(&g)
		if err := Compatible(h, []host.GPU{g}); err == nil {
			t.Errorf("%s: accepted", name)
		}
	}
	if err := Compatible(manifest.Hardware{GPUs: h.GPUs}, []host.GPU{mi350x}); err == nil {
		t.Error("an NVIDIA snapshot on AMD GPUs")
	}
}

func TestRenderNodes(t *testing.T) {
	h := amdHardware()
	other := mi350x
	other.KFDGPUID = 41002
	if err := RenderNodes(h, []host.GPU{mi350x}, 8); err != nil {
		t.Errorf("the donor's GPU, however many are visible: %v", err)
	}
	if err := RenderNodes(h, []host.GPU{other}, 1); err != nil {
		t.Errorf("another GPU, alone in the container: %v", err)
	}
	if err := RenderNodes(h, []host.GPU{other}, 2); err == nil {
		t.Error("another GPU among several: libkfdreplay would not know which")
	}
}

// fakeProc writes /proc/<pid>/{maps,fd/*} for each process.
func fakeProc(t *testing.T, procs map[int]struct{ maps, fds []string }) {
	t.Helper()
	root := t.TempDir()
	for pid, p := range procs {
		dir := filepath.Join(root, itoa(pid))
		if err := os.MkdirAll(filepath.Join(dir, "fd"), 0o755); err != nil {
			t.Fatal(err)
		}
		var lines []string
		for _, m := range p.maps {
			lines = append(lines, "7f0000000000-7f0000001000 r-xp 00000000 08:01 123 "+m)
		}
		lines = append(lines, "7ffd00000000-7ffd00021000 rw-p 00000000 00:00 0 [stack]")
		if err := os.WriteFile(filepath.Join(dir, "maps"), []byte(strings.Join(lines, "\n")+"\n"), 0o644); err != nil {
			t.Fatal(err)
		}
		for i, target := range p.fds {
			if err := os.Symlink(target, filepath.Join(dir, "fd", itoa(i))); err != nil {
				t.Fatal(err)
			}
		}
	}
	old := Proc
	Proc = root
	t.Cleanup(func() { Proc = old })
}

func itoa(i int) string { return strconv.Itoa(i) }

const lib = "/opt/warmctl/kfdreplay/v0.1.1/libkfdreplay.so"

func TestVerifyReleased(t *testing.T) {
	hip := "/opt/python/lib/_rocm_sdk_core/lib/libamdhip64.so.7"
	fakeProc(t, map[int]struct{ maps, fds []string }{
		10: {maps: []string{"/usr/bin/python3.14"}, fds: []string{"socket:[1]"}},
		11: {maps: []string{hip, lib, "/usr/lib/libc.so.6"}, fds: []string{"/workspace/logs/engine.log"}},
	})
	owners, err := VerifyReleased([]int{10, 11, 99}, lib)
	if err != nil || len(owners) != 1 || owners[0] != 11 {
		t.Fatalf("owners %v, %v", owners, err)
	}
	fakeProc(t, map[int]struct{ maps, fds []string }{
		11: {maps: []string{hip, lib}, fds: []string{"/dev/kfd"}},
	})
	if _, err := VerifyReleased([]int{11}, lib); err == nil || !strings.Contains(err.Error(), "/dev/kfd") {
		t.Fatalf("a process still holding /dev/kfd: %v", err)
	}
	fakeProc(t, map[int]struct{ maps, fds []string }{
		12: {maps: []string{hip, "/dev/dri/renderD136"}},
	})
	if _, err := VerifyReleased([]int{12}, lib); err == nil {
		t.Fatal("a process mapping the render node")
	}
	fakeProc(t, map[int]struct{ maps, fds []string }{13: {maps: []string{hip}}})
	if _, err := VerifyReleased([]int{13}, lib); err == nil || !strings.Contains(err.Error(), "without") {
		t.Fatalf("ROCm without libkfdreplay: %v", err)
	}
	if err := VerifyPreloaded([]int{13}, lib); err == nil {
		t.Fatal("restored ROCm process without libkfdreplay")
	}
	fakeProc(t, map[int]struct{ maps, fds []string }{14: {maps: []string{hip, lib}}, 15: {}})
	if err := VerifyPreloaded([]int{14, 15}, lib); err != nil {
		t.Fatal(err)
	}
}

// A file of the image anywhere (the ROCm image's Python is under
// /root/.local) is older than the container's PID 1.
func TestContainerStart(t *testing.T) {
	root := t.TempDir()
	if err := os.MkdirAll(filepath.Join(root, "1"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "stat"), []byte("cpu  1 2 3\nbtime 1763164800\nprocesses 9\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	// starttime (field 22) of 1500 ticks: 15 s after boot; the command has a space and a parenthesis.
	line := "1 (tini (x) y) S 0 1 1 0 -1 4194560 1 0 0 0 0 0 0 0 20 0 1 0 1500 1 1 18446744073709551615\n"
	if err := os.WriteFile(filepath.Join(root, "1", "stat"), []byte(line), 0o644); err != nil {
		t.Fatal(err)
	}
	old := Proc
	Proc = root
	defer func() { Proc = old }()
	got, ok := containerStart()
	if !ok || !got.Equal(time.Unix(1763164815, 0)) {
		t.Fatalf("%v %v", got, ok)
	}
	f := filepath.Join(t.TempDir(), "python3.14")
	if err := os.WriteFile(f, []byte("x"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.Chtimes(f, time.Unix(1763164000, 0), time.Unix(1763164000, 0)); err != nil {
		t.Fatal(err)
	}
	if !imageFile(f, time.Now()) {
		t.Error("a file older than the container is the image's")
	}
	if err := os.Chtimes(f, time.Unix(1763164900, 0), time.Unix(1763164900, 0)); err != nil {
		t.Fatal(err)
	}
	if imageFile(f, time.Now()) {
		t.Error("a file written since the container started, outside the image's directories")
	}
}

func TestCover(t *testing.T) {
	root := t.TempDir()
	model := filepath.Join(root, "models", "q")
	cache := filepath.Join(root, "root-cache")
	for _, d := range []string{model, cache} {
		if err := os.MkdirAll(d, 0o755); err != nil {
			t.Fatal(err)
		}
	}
	weights := filepath.Join(model, "model.safetensors")
	stray := filepath.Join(cache, "kernel.so")
	for _, f := range []string{weights, stray} {
		if err := os.WriteFile(f, []byte("x"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	fakeProc(t, map[int]struct{ maps, fds []string }{
		20: {maps: []string{weights, lib, "/workspace/triton/k.so", "/dev/kfd", "/memfd:x (deleted)"},
			fds: []string{"/carried/tmp/vllm_dist_1/x", "socket:[3]", "/dev/shm/sem.mp-1", "pipe:[2]"}},
	})
	// Trees other than the real /tmp: t.TempDir is under /tmp on Linux.
	c, err := Cover([]int{20}, []string{"/workspace", "/carried/tmp", "/dev/shm"}, model, lib, time.Now())
	if err != nil {
		t.Fatal(err)
	}
	if len(c.Uncovered) != 0 || len(c.Model) != 1 || c.Model[0].Path != weights || c.Model[0].Size != 1 {
		t.Fatalf("%+v", c)
	}
	if err := SameFiles(c.Model); err != nil {
		t.Fatal(err)
	}
	if err := os.Chtimes(weights, time.Now(), time.Now().Add(time.Hour)); err != nil {
		t.Fatal(err)
	}
	if err := SameFiles(c.Model); err == nil {
		t.Fatal("a model file touched since the snapshot")
	}
	fakeProc(t, map[int]struct{ maps, fds []string }{21: {maps: []string{stray, weights}}})
	c, err = Cover([]int{21}, []string{"/workspace"}, model, lib, time.Now())
	if err != nil || len(c.Uncovered) != 1 || c.Uncovered[0] != stray {
		t.Fatalf("a cache outside the trees: %+v %v", c, err)
	}
}
