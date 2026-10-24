package restore

import (
	"context"
	"fmt"
	"io"
	"os"
	"strings"
	"testing"

	"github.com/TensorRails/warmctl/v2/internal/host"
	"github.com/TensorRails/warmctl/v2/internal/manifest"
	"github.com/TensorRails/warmctl/v2/internal/profile"
	"github.com/TensorRails/warmctl/v2/internal/run"
)

func donor() manifest.Hardware {
	return manifest.Hardware{
		Driver: "580.173.02", Topology: "nvlink", CPUFlags: []string{"avx2", "avx512f", "sse4_2"},
		GPUs: []manifest.GPU{
			{Rank: 0, UUID: "GPU-d0", Name: "NVIDIA H100 80GB HBM3", MemoryMiB: 81559, PCIBusID: "0000:18:00.0"},
			{Rank: 1, UUID: "GPU-d1", Name: "NVIDIA H100 80GB HBM3", MemoryMiB: 81559, PCIBusID: "0000:2a:00.0"},
		},
	}
}

func receiver() []host.GPU {
	return []host.GPU{
		{Index: 0, UUID: "GPU-r0", Name: "NVIDIA H100 80GB HBM3", MemoryMiB: 81559, Driver: "580.173.02", PCIBusID: "0000:18:00.0"},
		{Index: 1, UUID: "GPU-r1", Name: "NVIDIA H100 80GB HBM3", MemoryMiB: 81559, Driver: "580.173.02", PCIBusID: "0000:2a:00.0"},
	}
}

func TestCompatible(t *testing.T) {
	flags := []string{"avx2", "avx512f", "sse4_2", "sha_ni"}
	if err := Compatible(donor(), receiver(), "nvlink", flags); err != nil {
		t.Fatal(err)
	}
	cases := map[string]func(g []host.GPU) ([]host.GPU, string, []string){
		"count": func(g []host.GPU) ([]host.GPU, string, []string) { return g[:1], "nvlink", flags },
		"model": func(g []host.GPU) ([]host.GPU, string, []string) {
			g[1].Name = "NVIDIA L40S"
			return g, "nvlink", flags
		},
		"memory": func(g []host.GPU) ([]host.GPU, string, []string) { g[0].MemoryMiB = 95000; return g, "nvlink", flags },
		"driver": func(g []host.GPU) ([]host.GPU, string, []string) {
			g[0].Driver = "580.95.05"
			return g, "nvlink", flags
		},
		"topology": func(g []host.GPU) ([]host.GPU, string, []string) { return g, "pcie", flags },
		"cpu":      func(g []host.GPU) ([]host.GPU, string, []string) { return g, "nvlink", []string{"avx2", "sse4_2"} },
	}
	for name, mutate := range cases {
		g, topo, f := mutate(receiver())
		if err := Compatible(donor(), g, topo, f); err == nil {
			t.Errorf("%s mismatch accepted", name)
		}
	}
	if err := Compatible(donor(), receiver(), "nvlink", []string{"avx2"}); err == nil || !strings.Contains(err.Error(), "avx512f") {
		t.Errorf("missing flags not named: %v", err)
	}
}

// A snapshot built with the driver's version check off restores under
// another release of the same branch, never under another branch.
func TestCompatibleDriverBranch(t *testing.T) {
	flags := []string{"avx2", "avx512f", "sse4_2"}
	h := donor()
	h.DriverPortable = true
	for driver, ok := range map[string]bool{"580.178.04": true, "580.95.05": true, "595.91.07": false, "570.195.03": false} {
		g := receiver()
		for i := range g {
			g[i].Driver = driver
		}
		if err := Compatible(h, g, "nvlink", flags); (err == nil) != ok {
			t.Errorf("driver %s: %v, want ok=%v", driver, err, ok)
		}
	}
}

// P149: ranks follow PCI order, never the order GPUs were listed in, and
// the device map pairs donor rank i with receiver rank i.
func TestDeviceMapFollowsRanks(t *testing.T) {
	all := receiver()
	permuted, err := host.Select(all, "1,0")
	if err != nil {
		t.Fatal(err)
	}
	r := &restore{o: Options{CUDAExecution: "auto"}, gpus: permuted, m: manifest.Manifest{Hardware: donor(),
		Process: manifest.Process{CollectivesDetached: true, Owners: []manifest.Owner{
			{PID: 32768, Rank: -1, Role: "engine"}, {PID: 32790, Rank: 0, Role: "worker"}, {PID: 32791, Rank: 1, Role: "worker"},
		}}}}
	args, err := r.cudaArgs()
	if err != nil {
		t.Fatal(err)
	}
	if args.DeviceMap != "GPU-d0=GPU-r0,GPU-d1=GPU-r1" {
		t.Fatalf("device map %s", args.DeviceMap)
	}
	if len(args.Owners) != 3 || len(args.Parallel) != 2 || args.Parallel[0] != 32790 {
		t.Fatalf("owners %v parallel %v", args.Owners, args.Parallel)
	}
	r.o.CUDAExecution = "serial-v1"
	if args, _ := r.cudaArgs(); len(args.Parallel) != 0 {
		t.Fatal("serial-v1 ran workers in parallel")
	}
	r.o.CUDAExecution = "auto"
	r.m.Process.CollectivesDetached = false
	if args, _ := r.cudaArgs(); len(args.Parallel) != 0 {
		t.Fatal("linked collectives ran workers in parallel")
	}
}

func TestModelDir(t *testing.T) {
	m := manifest.Manifest{Profile: profile.Profile{Model: profile.Model{ContainerPath: "/model"}}}
	if dir, err := ModelDir("", m); err != nil || dir != "" {
		t.Fatalf("no reference needs no model: %q %v", dir, err)
	}
	m.Process.NeedsModel = true
	m.Launch.ModelDir = t.TempDir()
	if dir, err := ModelDir("", m); err != nil || dir != m.Launch.ModelDir {
		t.Fatalf("same-host fallback: %q %v", dir, err)
	}
	if dir, err := ModelDir("/given", m); err != nil || dir != "/given" {
		t.Fatalf("operator's dir: %q %v", dir, err)
	}
	os.RemoveAll(m.Launch.ModelDir)
	if _, err := ModelDir("", m); err == nil || !strings.Contains(err.Error(), "--model-dir") {
		t.Fatalf("missing model accepted: %v", err)
	}
}

// fakeNVML is a host of NVLinked H100s on driver 580.173.02, with busy[i]
// compute processes on GPU i.
type fakeNVML struct {
	uuids []string
	busy  map[int]int
}

func (f fakeNVML) Driver() (string, error) { return "580.173.02", nil }
func (f fakeNVML) Count() (int, error)     { return len(f.uuids), nil }
func (f fakeNVML) GPU(i int) (host.GPU, error) {
	return host.GPU{UUID: f.uuids[i], Name: "NVIDIA H100 80GB HBM3", MemoryMiB: 81559,
		PCIBusID: fmt.Sprintf("00000000:%02X:00.0", 0x18+16*i)}, nil
}
func (f fakeNVML) Processes(i int) (int, error)  { return f.busy[i], nil }
func (f fakeNVML) NVLink(int, int) (bool, error) { return true, nil }
func (f fakeNVML) Path(int, int) (string, error) { return "SYS", nil }
func (f fakeNVML) Close() error                  { return nil }

func useNVML(t *testing.T, f fakeNVML) {
	t.Helper()
	old := host.Open
	host.Open = func() (host.Session, error) { return f, nil }
	t.Cleanup(func() { host.Open = old })
}

// CheckHost picks the first free GPUs of the snapshot's model and names
// what is missing when the host cannot restore it.
func TestCheckHost(t *testing.T) {
	useNVML(t, fakeNVML{uuids: []string{"GPU-r0", "GPU-r1", "GPU-r2"}, busy: map[int]int{0: 1}})
	h := donor()
	h.Topology, h.CPUFlags = "nvlink", nil
	gpus, err := CheckHost(context.Background(), h, "", t.TempDir())
	if err != nil || len(gpus) != 2 || gpus[0].Index != 1 || gpus[1].Index != 2 {
		t.Fatalf("picked %+v, %v", gpus, err)
	}
	if _, err := CheckHost(context.Background(), h, "0,1", t.TempDir()); err == nil ||
		!strings.Contains(err.Error(), "GPU 0 runs 1 other processes") {
		t.Fatalf("busy GPU listed explicitly: %v", err)
	}
	h.Driver = "580.95.05"
	if _, err := CheckHost(context.Background(), h, "", t.TempDir()); err == nil ||
		!strings.Contains(err.Error(), "cuda-checkpoint needs the same driver") {
		t.Fatalf("driver mismatch: %v", err)
	}
}

type fakeArrival struct {
	m      manifest.Manifest
	waited []string
	err    error
}

func (a *fakeArrival) Manifest() manifest.Manifest { return a.m }
func (a *fakeArrival) ManifestJSON() []byte        { return []byte(`{"id":"s"}`) }
func (a *fakeArrival) Wait(_ context.Context, paths []string) error {
	a.waited = append(a.waited, paths...)
	return a.err
}

// During a pull the process restore waits for all but the weights lane,
// and the engine wakes once the weights lane has arrived.
func TestArrival(t *testing.T) {
	a := &fakeArrival{m: manifest.Manifest{ID: "s", Files: []manifest.File{
		{Path: "criu/core-1.img", Role: manifest.RoleCRIU},
		{Path: "fs/workspace.tar", Role: manifest.RoleRuntimeFS},
		{Path: "weights/rank-000/part-00000.bin", Role: manifest.RoleWeights},
		{Path: "weights/rank-000/table-emb.bin", Role: manifest.RoleHostTable},
	}}}
	r := &restore{o: Options{Arrival: a, Snapshot: "/snap"}, p: run.NewProgress(io.Discard, "plain")}
	ctx := context.Background()
	if err := r.read(ctx); err != nil || r.m.ID != "s" || r.snapshot != "/snap" {
		t.Fatalf("read %v: %+v", err, r.m)
	}
	if err := r.arrive(ctx); err != nil {
		t.Fatal(err)
	}
	if got := strings.Join(a.waited, " "); got != "criu/core-1.img fs/workspace.tar" {
		t.Fatalf("waited for %s", got)
	}
	a.waited = nil
	if err := r.arrived(ctx); err != nil {
		t.Fatal(err)
	}
	if got := strings.Join(a.waited, " "); got != "weights/rank-000/part-00000.bin weights/rank-000/table-emb.bin" {
		t.Fatalf("waited for %s", got)
	}
	if r.manifestPath() != "/work/manifest.json" {
		t.Fatalf("the reload reads %s", r.manifestPath())
	}
	a.err = fmt.Errorf("pull failed")
	if err := r.arrived(ctx); err == nil {
		t.Fatal("woke after a failed pull")
	}
	// Without a pull in progress neither step waits.
	r.o.Arrival = nil
	if r.arrive(ctx) != nil || r.arrived(ctx) != nil || r.manifestPath() != "/capture/manifest.json" {
		t.Fatal("a pulled snapshot waited")
	}
}
