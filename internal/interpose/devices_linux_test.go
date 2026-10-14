//go:build linux

package interpose

import (
	"context"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// TestProbe reads a stand-in libcuda.so.1 (testdata/fakecuda.c) through the
// bindings the agent uses on a GPU host.
func TestProbe(t *testing.T) {
	cc, err := exec.LookPath("cc")
	if err != nil {
		t.Skip("needs a C compiler")
	}
	lib := filepath.Join(t.TempDir(), "libcuda.so.1")
	out, err := exec.Command(cc, "-O2", "-shared", "-fPIC", "-nostdlib", "-ffreestanding", "-fno-stack-protector",
		"-Wl,--no-undefined", "-Wl,-soname,libcuda.so.1", "-o", lib, "testdata/fakecuda.c").CombinedOutput()
	if err != nil {
		t.Fatalf("%v: %s", err, out)
	}
	cudaLibrary = lib
	devices, err := Devices(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	want := []Device{
		{Index: 0, Name: "NVIDIA H100 80GB HBM3", ComputeCapability: "9.0", VMM: true, PosixFD: true, Multicast: true},
		{Index: 1, Name: "Fake vGPU", ComputeCapability: "8.9"},
	}
	if len(devices) != 2 || devices[0] != want[0] || devices[1] != want[1] {
		t.Fatalf("got %+v, want %+v", devices, want)
	}
	if err := Supported(devices); err == nil || !strings.Contains(err.Error(), "GPU 1 (Fake vGPU)") {
		t.Fatal(err)
	}
	if _, err := describe(5); err == nil || !strings.Contains(err.Error(), "invalid device ordinal (101)") {
		t.Fatalf("a missing ordinal: %v", err)
	}
}
