package host

import "testing"

const smi = `0, GPU-aaaa, NVIDIA H100 80GB HBM3, 00000000:1B:00.0, 81559, 580.95.05
1, GPU-bbbb, NVIDIA H100 80GB HBM3, 00000000:18:00.0, 81559, 580.95.05
2, GPU-cccc, NVIDIA H100 80GB HBM3, 00000000:2A:00.0, 81559, 580.95.05
`

const topo = `	GPU0	GPU1	GPU2	CPU Affinity	NUMA Affinity
GPU0	 X 	NV18	PIX	0-47	0
GPU1	NV18	 X 	NV18	0-47	0
GPU2	PIX	NV18	 X 	0-47	0
`

func TestInventory(t *testing.T) {
	gpus, err := ParseInventory(smi)
	if err != nil || len(gpus) != 3 || gpus[0].PCIBusID != "0000:1b:00.0" || gpus[2].MemoryMiB != 81559 || gpus[1].Driver != "580.95.05" {
		t.Fatalf("%+v %v", gpus, err)
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
	if c := TopologyClass(topo, gpus[:2]); c != "nvlink" {
		t.Fatalf("GPU0+GPU1: %s", c)
	}
	if c := TopologyClass(topo, gpus); c != "pcie" {
		t.Fatalf("all three: %s", c)
	}
	if c := TopologyClass(topo, gpus[:1]); c != "single" {
		t.Fatal(c)
	}
}

func TestLease(t *testing.T) {
	dir := t.TempDir()
	gpus, _ := ParseInventory(smi)
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

// P129: nvidia-smi underlines the header with SGR escapes; parsing it raw
// found no header and would call any multi-GPU host NVLink.
func TestTopologyWithEscapes(t *testing.T) {
	gpus, _ := ParseInventory(smi)
	underlined := "\t\x1b[4mGPU0\tGPU1\tGPU2\tCPU Affinity\tNUMA Affinity\x1b[0m\n" +
		"\x1b[4mGPU0\x1b[0m\t X \tNV18\tPIX\t0-47\t0\n" +
		"\x1b[4mGPU1\x1b[0m\tNV18\t X \tNV18\t0-47\t0\n" +
		"\x1b[4mGPU2\x1b[0m\tPIX\tNV18\t X \t0-47\t0\n"
	if c := TopologyClass(underlined, gpus); c != "pcie" {
		t.Fatalf("escaped matrix, all three GPUs: %s", c)
	}
	if c := TopologyClass(underlined, gpus[:2]); c != "nvlink" {
		t.Fatalf("escaped matrix, GPU0+GPU1: %s", c)
	}
}

// Pick skips GPUs that run a process, that another run leases or that are
// another model, and keeps PCI order.
func TestPick(t *testing.T) {
	gpus, _ := ParseInventory(smi + "3, GPU-dddd, NVIDIA L40S, 00000000:3A:00.0, 46068, 580.95.05\n")
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
