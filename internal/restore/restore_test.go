package restore

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/TensorRails/warmctl/v2/internal/host"
	"github.com/TensorRails/warmctl/v2/internal/manifest"
	"github.com/TensorRails/warmctl/v2/internal/profile"
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

// fakeSMI makes nvidia-smi a script printing the given inventory, compute
// processes and an all-NVLink topology of n GPUs.
func fakeSMI(t *testing.T, inventory, apps string, n int) {
	t.Helper()
	topo := "\t"
	for i := 0; i < n; i++ {
		topo += fmt.Sprintf("GPU%d\t", i)
	}
	topo += "CPU Affinity\n"
	for i := 0; i < n; i++ {
		topo += fmt.Sprintf("GPU%d", i)
		for j := 0; j < n; j++ {
			if i == j {
				topo += "\t X "
			} else {
				topo += "\tNV18"
			}
		}
		topo += "\t0-47\n"
	}
	script := fmt.Sprintf("#!/bin/sh\ncase \"$*\" in\n*--query-gpu=*) printf '%%s' '%s' ;;\n"+
		"*--query-compute-apps=*) printf '%%s' '%s' ;;\n*topo*) printf '%%s' '%s' ;;\nesac\n", inventory, apps, topo)
	name := filepath.Join(t.TempDir(), "nvidia-smi")
	if err := os.WriteFile(name, []byte(script), 0o755); err != nil {
		t.Fatal(err)
	}
	old := host.SMI
	host.SMI = name
	t.Cleanup(func() { host.SMI = old })
}

// CheckHost picks the first free GPUs of the snapshot's model and names
// what is missing when the host cannot restore it.
func TestCheckHost(t *testing.T) {
	inv := "0, GPU-r0, NVIDIA H100 80GB HBM3, 00000000:18:00.0, 81559, 580.173.02\n" +
		"1, GPU-r1, NVIDIA H100 80GB HBM3, 00000000:2a:00.0, 81559, 580.173.02\n" +
		"2, GPU-r2, NVIDIA H100 80GB HBM3, 00000000:3a:00.0, 81559, 580.173.02\n"
	fakeSMI(t, inv, "GPU-r0, 4242\n", 3)
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
