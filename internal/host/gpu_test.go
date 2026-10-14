package host

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"
)

// fakeNVML is a host whose GPUs, compute processes and links a test sets.
type fakeNVML struct {
	driver string
	gpus   []GPU // UUID, name, bus ID as NVML prints it, memory
	procs  map[int]int
	nvlink [][2]int // pairs joined by NVLink
	opened int
	closed int
}

func (f *fakeNVML) Driver() (string, error) { return f.driver, nil }
func (f *fakeNVML) Count() (int, error)     { return len(f.gpus), nil }
func (f *fakeNVML) GPU(i int) (GPU, error)  { return f.gpus[i], nil }
func (f *fakeNVML) Processes(i int) (int, error) {
	return f.procs[i], nil
}
func (f *fakeNVML) NVLink(i, j int) (bool, error) {
	for _, p := range f.nvlink {
		if p == [2]int{i, j} || p == [2]int{j, i} {
			return true, nil
		}
	}
	return false, nil
}
func (f *fakeNVML) Path(int, int) (string, error) { return "PIX", nil }
func (f *fakeNVML) Close() error                  { f.closed++; return nil }

// use makes f the host's NVML for the test.
func (f *fakeNVML) use(t *testing.T) {
	old := Open
	Open = func() (Session, error) { f.opened++; return f, nil }
	t.Cleanup(func() { Open = old })
}

func h100s() *fakeNVML {
	return &fakeNVML{driver: "580.95.05", gpus: []GPU{
		{UUID: "GPU-aaaa", Name: "NVIDIA H100 80GB HBM3", PCIBusID: "00000000:1B:00.0", MemoryMiB: 81559},
		{UUID: "GPU-bbbb", Name: "NVIDIA H100 80GB HBM3", PCIBusID: "00000000:18:00.0", MemoryMiB: 81559},
		{UUID: "GPU-cccc", Name: "NVIDIA H100 80GB HBM3", PCIBusID: "00000000:2A:00.0", MemoryMiB: 81559},
	}, nvlink: [][2]int{{0, 1}, {1, 2}}}
}

func inventoryOf(t *testing.T, f *fakeNVML) []GPU {
	t.Helper()
	f.use(t)
	gpus, err := Inventory(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	return gpus
}

func TestInventory(t *testing.T) {
	f := h100s()
	gpus := inventoryOf(t, f)
	if len(gpus) != 3 || gpus[0].PCIBusID != "0000:1b:00.0" || gpus[2].Index != 2 || gpus[2].MemoryMiB != 81559 ||
		gpus[1].Driver != "580.95.05" {
		t.Fatalf("%+v", gpus)
	}
	if f.opened != 1 || f.closed != 1 {
		t.Fatalf("session opened %d, closed %d times", f.opened, f.closed)
	}
	sel, err := Select(gpus, "0-1")
	if err != nil || len(sel) != 2 || sel[0].Index != 1 || sel[1].Index != 0 {
		t.Fatalf("PCI order: %+v %v", sel, err)
	}
	if _, err := Select(gpus, "0,0"); err == nil {
		t.Fatal("duplicate accepted")
	}
	if _, err := Select(gpus, "7"); err == nil {
		t.Fatal("missing GPU accepted")
	}
	if all, _ := Select(gpus, ""); len(all) != 3 {
		t.Fatal("empty spec is not all GPUs")
	}
}

func TestInventoryRefusesOddHosts(t *testing.T) {
	(&fakeNVML{driver: "580.95.05"}).use(t)
	if _, err := Inventory(context.Background()); err == nil || !strings.Contains(err.Error(), "no GPUs") {
		t.Fatalf("no GPUs: %v", err)
	}
	f := h100s()
	f.gpus[1].UUID = "MIG-1"
	f.use(t)
	if _, err := Inventory(context.Background()); err == nil || !strings.Contains(err.Error(), "MIG-1") {
		t.Fatalf("odd UUID: %v", err)
	}
	Open = func() (Session, error) { return nil, errors.New("nvml: no driver") }
	if _, err := Inventory(context.Background()); err == nil || err.Error() != "nvml: no driver" {
		t.Fatalf("no library: %v", err)
	}
}

// A query that NVML never answers gives up with the caller's context.
func TestQueryGivesUp(t *testing.T) {
	stuck := make(chan struct{})
	defer close(stuck)
	old := Open
	Open = func() (Session, error) { <-stuck; return nil, errors.New("late") }
	t.Cleanup(func() { Open = old })
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer cancel()
	if _, err := Inventory(ctx); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("stuck NVML: %v", err)
	}
}

func TestProcesses(t *testing.T) {
	f := h100s()
	f.procs = map[int]int{1: 2}
	gpus := inventoryOf(t, f)
	procs, err := Processes(context.Background())
	if err != nil || len(procs) != 1 || procs["GPU-bbbb"] != 2 {
		t.Fatalf("%v %v", procs, err)
	}
	if n, err := Busy(context.Background(), []GPU{gpus[0], gpus[2]}); err != nil || n != 0 {
		t.Fatalf("idle GPUs busy: %d %v", n, err)
	}
	if n, err := Busy(context.Background(), gpus); err != nil || n != 2 {
		t.Fatalf("busy GPUs: %d %v", n, err)
	}
}

func TestTopology(t *testing.T) {
	f := h100s()
	gpus := inventoryOf(t, f)
	for _, c := range []struct {
		gpus []GPU
		want string
	}{{gpus[:2], "nvlink"}, {gpus[1:], "nvlink"}, {gpus, "pcie"}, {gpus[:1], "single"}} {
		class, sum, err := Topology(context.Background(), c.gpus)
		if err != nil || class != c.want || len(sum) != 64 {
			t.Fatalf("%d GPUs from %d: %s %q %v, want %s", len(c.gpus), c.gpus[0].Index, class, sum, err, c.want)
		}
	}
	m, err := links(f)
	if err != nil {
		t.Fatal(err)
	}
	if want := "\tGPU0\tGPU1\tGPU2\nGPU0\tX\tNV\tPIX\nGPU1\tNV\tX\tNV\nGPU2\tPIX\tNV\tX\n"; m.String() != want {
		t.Fatalf("matrix:\n%s", m)
	}
	if c := TopologyClass(m, []GPU{gpus[0], {Index: 5}}); c != "pcie" {
		t.Fatalf("GPU outside the matrix: %s", c)
	}
}

func TestLease(t *testing.T) {
	dir := t.TempDir()
	gpus := inventoryOf(t, h100s())
	a, err := Acquire(dir, gpus[:2])
	if err != nil {
		t.Fatal(err)
	}
	if _, err := Acquire(dir, gpus[1:]); err == nil {
		t.Fatal("overlapping lease granted")
	}
	a.Release()
	b, err := Acquire(dir, gpus[1:])
	if err != nil {
		t.Fatal(err)
	}
	b.Release()
}

// Pick skips GPUs that run a process, that another run leases or that are
// another model, and keeps PCI order.
func TestPick(t *testing.T) {
	f := h100s()
	f.gpus = append(f.gpus, GPU{UUID: "GPU-dddd", Name: "NVIDIA L40S", PCIBusID: "00000000:3A:00.0", MemoryMiB: 46068})
	gpus := inventoryOf(t, f)
	locks := t.TempDir()
	lease, err := Acquire(locks, gpus[2:3]) // GPU 2
	if err != nil {
		t.Fatal(err)
	}
	defer lease.Release()
	busy := map[string]int{"GPU-bbbb": 1} // GPU 1
	got, err := Pick(gpus, 1, "NVIDIA H100 80GB HBM3", busy, locks)
	if err != nil || len(got) != 1 || got[0].Index != 0 {
		t.Fatalf("picked %+v, %v", got, err)
	}
	if _, err := Pick(gpus, 2, "NVIDIA H100 80GB HBM3", busy, locks); err == nil ||
		err.Error() != "needs 2 free NVIDIA H100 80GB HBM3, this host has 3× NVIDIA H100 80GB HBM3, 1× NVIDIA L40S with 1 of them free" {
		t.Fatalf("too few free GPUs: %v", err)
	}
	if !Leased(locks, gpus[2]) || Leased(locks, gpus[0]) {
		t.Fatal("lease state misread")
	}
}
