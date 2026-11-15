package host

import (
	"context"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// useKFDFixture points the KFD paths at testdata/<name>, where the other
// pods' GPU nodes refuse reads as on RunPod, and NVML at a missing library.
func useKFDFixture(t *testing.T, name string, refused ...string) {
	t.Helper()
	root, err := filepath.Abs(filepath.Join("testdata", name))
	if err != nil {
		t.Fatal(err)
	}
	old := []string{KFDDevice, KFDTopology, AMDGPUModule, DRIDir, NVIDIADevice}
	oldRead, oldOpen := readFile, Open
	KFDDevice, KFDTopology = filepath.Join(root, "dev", "kfd"), filepath.Join(root, "sys", "topology")
	AMDGPUModule, DRIDir = filepath.Join(root, "sys", "amdgpu"), "/dev/dri"
	NVIDIADevice = filepath.Join(root, "dev", "nvidiactl")
	readFile = func(name string) ([]byte, error) {
		for _, r := range refused {
			if strings.Contains(name, "/nodes/"+r+"/") {
				return nil, &fs.PathError{Op: "read", Path: name, Err: fs.ErrPermission}
			}
		}
		return os.ReadFile(name)
	}
	Open = openSession
	t.Cleanup(func() {
		KFDDevice, KFDTopology, AMDGPUModule, DRIDir, NVIDIADevice = old[0], old[1], old[2], old[3], old[4]
		readFile, Open = oldRead, oldOpen
	})
}

var podRefused = []string{"2", "4", "5", "6", "7", "8", "9"}

func TestKFDInventoryOfAPod(t *testing.T) {
	useKFDFixture(t, "kfd-mi350x-pod", podRefused...)
	if !KFDPresent() {
		t.Fatal("the fixture is an AMD host")
	}
	gpus, err := Inventory(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(gpus) != 1 {
		t.Fatalf("the pod sees one GPU, got %+v", gpus)
	}
	g := gpus[0]
	want := GPU{Index: 0, UUID: "GPU-1111111111111111", Name: "AMD Instinct MI350X", PCIBusID: "0000:09:00.0",
		MemoryMiB: 294896, Driver: "7.1.3.31500000", Vendor: VendorAMD, Arch: "gfx950", KFDGPUID: 41001,
		RenderNode: "/dev/dri/renderD136"}
	if g != want {
		t.Errorf("got  %+v\nwant %+v", g, want)
	}
	if VendorOf(gpus) != VendorAMD || VendorOf(nil) != VendorNVIDIA {
		t.Error("VendorOf")
	}
	topo, sum, err := Topology(context.Background(), gpus)
	if err != nil || topo != "single" || sum == "" {
		t.Errorf("topology %q %q %v", topo, sum, err)
	}
	if _, err := Processes(context.Background()); err != nil {
		t.Error(err)
	}
}

// Every node readable: the GPUs come in node order.
func TestKFDInventoryOfAHost(t *testing.T) {
	useKFDFixture(t, "kfd-mi350x-pod")
	gpus, err := Inventory(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(gpus) != 8 {
		t.Fatalf("%d GPUs", len(gpus))
	}
	for i, g := range gpus {
		if g.Index != i || g.Vendor != VendorAMD || g.Arch != "gfx950" {
			t.Errorf("GPU %d: %+v", i, g)
		}
	}
	// Nodes without a device_id the table knows still get a usable name.
	if !strings.HasPrefix(gpus[0].Name, "AMD GPU gfx950") || gpus[1].Name != "AMD Instinct MI350X" {
		t.Errorf("names %q %q", gpus[0].Name, gpus[1].Name)
	}
}

// A host with an NVIDIA device keeps NVML's answer even with /dev/kfd (an
// AMD APU next to NVIDIA GPUs): warmctl on NVIDIA hosts behaves as before.
func TestNVIDIADeviceWinsOverKFD(t *testing.T) {
	useKFDFixture(t, "kfd-mi350x-pod", podRefused...)
	dir := t.TempDir()
	NVIDIADevice = filepath.Join(dir, "nvidiactl")
	if err := os.WriteFile(NVIDIADevice, nil, 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := Inventory(context.Background()); err == nil || !strings.Contains(err.Error(), "nvml") {
		t.Fatalf("an NVIDIA host must get NVML's error, got %v", err)
	}
}

// Neither vendor: NVML's error, as before.
func TestNoGPUHostKeepsNVMLError(t *testing.T) {
	useKFDFixture(t, "kfd-mi350x-pod", podRefused...)
	KFDDevice = filepath.Join(t.TempDir(), "kfd")
	_, err := Inventory(context.Background())
	if err == nil || !strings.Contains(err.Error(), "nvml") {
		t.Fatalf("got %v", err)
	}
	if KFDPresent() {
		t.Fatal("no /dev/kfd")
	}
}

func TestKFDNoReadableGPU(t *testing.T) {
	useKFDFixture(t, "kfd-mi350x-pod", "2", "3", "4", "5", "6", "7", "8", "9")
	if KFDPresent() {
		t.Fatal("no node may be read")
	}
	if _, err := openKFD(); err == nil {
		t.Fatal("openKFD with no readable GPU")
	}
}

func TestGFXAndPCI(t *testing.T) {
	for v, want := range map[uint64]string{90500: "gfx950", 90402: "gfx942", 90010: "gfx90a", 110000: "gfx1100"} {
		if got := GFX(v); got != want {
			t.Errorf("GFX(%d) = %s, want %s", v, got, want)
		}
	}
	if got := pciAddress(map[string]uint64{"location_id": 0xc1 << 8, "domain": 0}); got != "0000:c1:00.0" {
		t.Error(got)
	}
	if got := pciAddress(map[string]uint64{"location_id": 0x0901 | (3 << 3), "domain": 1}); got != "0001:09:03.1" {
		t.Error(got)
	}
}

func TestKFDHolders(t *testing.T) {
	if _, err := os.Stat("/proc/self/fd"); err != nil {
		t.Skip("no /proc")
	}
	if n := kfdHolders(os.Getpid()); n != 0 {
		t.Errorf("%d processes hold %s on a test host", n, KFDDevice)
	}
}
