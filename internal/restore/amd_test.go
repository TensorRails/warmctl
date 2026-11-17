package restore

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/TensorRails/warmctl/v2/internal/accel"
	"github.com/TensorRails/warmctl/v2/internal/kfdreplay"
	"github.com/TensorRails/warmctl/v2/internal/manifest"
)

func amdManifest(t *testing.T) manifest.Manifest {
	t.Helper()
	lib := filepath.Join(t.TempDir(), "libkfdreplay.so")
	if err := os.WriteFile(lib, []byte("lib"), 0o755); err != nil {
		t.Fatal(err)
	}
	f, err := kfdreplay.Describe(lib)
	if err != nil {
		t.Fatal(err)
	}
	var m manifest.Manifest
	m.Schema, m.Hardware.Vendor = manifest.SchemaROCm, manifest.VendorAMD
	m.Process.Backend, m.Process.GPUState, m.Process.GPULib = manifest.BackendDMTCP, manifest.GPUStateKFDReplay, &f
	return m
}

// Every snapshot taken so far is NVIDIA's and restores on an NVIDIA host
// as before; across vendors a restore is refused before anything is pulled.
func TestCanRestoreVendor(t *testing.T) {
	var nvidia manifest.Manifest
	nvidia.Process.Backend = manifest.BackendDMTCP
	if err := canRestoreVendor(nvidia, true, accel.NVIDIA); err != nil {
		t.Fatal(err)
	}
	if err := canRestoreVendor(nvidia, false, accel.NVIDIA); err != nil {
		t.Fatal(err)
	}
	if err := canRestoreVendor(nvidia, true, accel.AMD); err == nil || !strings.Contains(err.Error(), "nvidia GPUs") {
		t.Fatalf("NVIDIA snapshot on AMD: %v", err)
	}
	amd := amdManifest(t)
	if err := canRestoreVendor(amd, true, accel.NVIDIA); err == nil {
		t.Fatal("AMD snapshot on NVIDIA")
	}
	if err := canRestoreVendor(amd, true, accel.AMD); err != nil {
		t.Fatal(err)
	}
	if err := canRestoreVendor(amd, false, accel.AMD); err == nil || !strings.Contains(err.Error(), "in place") {
		t.Fatalf("through Docker: %v", err)
	}
	moved := amd
	lib := *amd.Process.GPULib
	lib.Path += ".missing"
	moved.Process.GPULib = &lib
	if err := canRestoreVendor(moved, true, accel.AMD); err == nil || !strings.Contains(err.Error(), "libkfdreplay") {
		t.Fatalf("without the snapshot's libkfdreplay: %v", err)
	}
	criu := amd
	criu.Process.Backend = ""
	if err := canRestoreVendor(criu, true, accel.AMD); err == nil {
		t.Fatal("an AMD snapshot of another backend")
	}
}
