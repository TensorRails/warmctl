package manifest_test

import (
	"path/filepath"
	"strings"
	"testing"

	"github.com/TensorRails/warmctl/v2/internal/manifest"
)

// The schema and the GPU vendor go together: NVIDIA's snapshots keep the
// schema every release reads, AMD's carry one that releases before AMD
// support refuse.
func TestSchemaGoesWithVendor(t *testing.T) {
	raw := readFixture(t, "dmtcp-h200-tp1-beta17.json")
	m := mustRead(t, raw)
	if m.Hardware.VendorName() != manifest.VendorNVIDIA {
		t.Fatalf("a snapshot without the field is NVIDIA's, got %q", m.Hardware.VendorName())
	}
	for _, c := range []struct {
		schema, vendor string
		ok             bool
	}{
		{manifest.Schema, "", true},
		{manifest.SchemaROCm, manifest.VendorAMD, true},
		{manifest.Schema, manifest.VendorAMD, false},
		{manifest.SchemaROCm, "", false},
		{manifest.Schema, manifest.VendorNVIDIA, false},
		{"warmctl.snapshot/v3", "", false},
	} {
		m.Schema, m.Hardware.Vendor = c.schema, c.vendor
		if err := m.Validate(); (err == nil) != c.ok {
			t.Errorf("schema %s vendor %q: %v", c.schema, c.vendor, err)
		}
	}
}

// An AMD manifest writes and reads back with every AMD field.
func TestAMDRoundtrip(t *testing.T) {
	m := mustRead(t, readFixture(t, "dmtcp-h200-tp1-beta17.json"))
	m.Schema, m.Hardware.Vendor = manifest.SchemaROCm, manifest.VendorAMD
	m.Hardware.GPUCheckpoint = "libkfdreplay 0.1.1 sha256:7ce6"
	m.Hardware.ROCm = &manifest.ROCm{HIP: "7.16.26385", Runtime: "rocm-sdk-core 10.1.0"}
	m.Hardware.GPUs[0].Arch, m.Hardware.GPUs[0].KFDGPUID = "gfx950", 41001
	m.Process.GPUState = manifest.GPUStateKFDReplay
	m.Process.GPULib = &manifest.MappedFile{Path: "/opt/warmctl/kfdreplay/v0.1.1/libkfdreplay.so", Size: 157456,
		MtimeNS: 1763164800123456789, SHA256: "7ce6", Version: "0.1.1"}
	m.Process.GPUOwners = []manifest.Owner{{PID: 40001, Rank: 0, Role: "worker"}}
	m.Process.GPUMaxWptr = 41467
	m.Process.Mapped = []manifest.MappedFile{{Path: "/nv/models/q/model.safetensors", Size: 1, MtimeNS: 2}}
	m.Process.Owners = nil
	name := filepath.Join(t.TempDir(), "manifest.json")
	if err := m.Write(name); err != nil {
		t.Fatal(err)
	}
	back, err := manifest.Read(name)
	if err != nil {
		t.Fatal(err)
	}
	if back.Hardware.VendorName() != manifest.VendorAMD || *back.Process.GPULib != *m.Process.GPULib ||
		back.Hardware.GPUs[0].KFDGPUID != 41001 || back.Process.GPUMaxWptr != 41467 ||
		back.Hardware.ROCm.HIP != "7.16.26385" || len(back.Process.Mapped) != 1 {
		t.Fatalf("lost AMD fields: %+v %+v", back.Hardware, back.Process)
	}
}

func mustRead(t *testing.T, raw []byte) manifest.Manifest {
	t.Helper()
	name := filepath.Join(t.TempDir(), "manifest.json")
	if err := writeFile(name, raw); err != nil {
		t.Fatal(err)
	}
	m, err := manifest.Read(name)
	if err != nil && !strings.Contains(err.Error(), "schema") {
		t.Fatal(err)
	}
	return m
}
