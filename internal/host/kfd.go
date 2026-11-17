package host

import (
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path"
	"sort"
	"strconv"
	"strings"
)

// GPU vendors, as GPU.Vendor names them; NVIDIA's GPUs leave it empty.
const (
	VendorNVIDIA = "nvidia"
	VendorAMD    = "amd"
)

// VendorOf is the vendor of gpus: AMD when they came from the KFD topology,
// NVIDIA otherwise.
func VendorOf(gpus []GPU) string {
	if len(gpus) > 0 && gpus[0].Vendor == VendorAMD {
		return VendorAMD
	}
	return VendorNVIDIA
}

// KFD is where an AMD host's kernel driver describes its GPUs, and the device
// a ROCm process opens. Tests point them at a fixture.
var (
	KFDDevice    = "/dev/kfd"
	KFDTopology  = "/sys/class/kfd/kfd/topology"
	AMDGPUModule = "/sys/module/amdgpu"
	DRIDir       = "/dev/dri"
	// readFile reads one sysfs file; a container may list another pod's GPU
	// node and refuse to read it (EPERM), which leaves that GPU out.
	readFile = os.ReadFile
)

// TestKFDRootEnv points the KFD paths at a fixture laid out as
// testdata/kfd-mi350x-pod (dev/kfd, sys/topology, sys/amdgpu): the CPU
// end-to-end test of the AMD path (tests/integration/amd) runs warmctl in a
// container that has no AMD GPU. Never set in production.
const TestKFDRootEnv = "WARMCTL_TEST_KFD_ROOT"

func init() {
	if root := os.Getenv(TestKFDRootEnv); root != "" {
		KFDDevice = path.Join(root, "dev", "kfd")
		KFDTopology = path.Join(root, "sys", "topology")
		AMDGPUModule = path.Join(root, "sys", "amdgpu")
		DRIDir = path.Join(root, "dev", "dri")
	}
}

// amdNames are the Instinct models by PCI device ID, the name warmctl and
// hardware.yaml use: KFD's topology has no marketing name, and amd-smi is
// not in every image.
var amdNames = map[uint64]string{
	0x74a0: "AMD Instinct MI300A",
	0x74a1: "AMD Instinct MI300X",
	0x74a2: "AMD Instinct MI308X",
	0x74a5: "AMD Instinct MI325X",
	0x75a0: "AMD Instinct MI350X",
	0x75a3: "AMD Instinct MI355X",
}

// kfdNode is one GPU node of the KFD topology this process may read.
type kfdNode struct {
	node     int
	gpuID    uint64
	props    map[string]uint64
	memBytes uint64
}

// KFDPresent reports whether this is an AMD compute host warmctl can read:
// /dev/kfd exists and the topology lists at least one GPU it may read.
func KFDPresent() bool {
	if _, err := os.Stat(KFDDevice); err != nil {
		return false
	}
	nodes, err := kfdNodes()
	return err == nil && len(nodes) > 0
}

// kfdNodes reads the GPU nodes of the topology (CPU nodes have no SIMDs),
// in node order, skipping the ones this process may not read.
func kfdNodes() ([]kfdNode, error) {
	entries, err := os.ReadDir(path.Join(KFDTopology, "nodes"))
	if err != nil {
		return nil, fmt.Errorf("kfd: %w", err)
	}
	var out []kfdNode
	for _, e := range entries {
		n, err := strconv.Atoi(e.Name())
		if err != nil {
			continue
		}
		dir := path.Join(KFDTopology, "nodes", e.Name())
		raw, err := readFile(path.Join(dir, "properties"))
		if errors.Is(err, fs.ErrPermission) {
			continue
		}
		if err != nil {
			return nil, fmt.Errorf("kfd: node %d: %w", n, err)
		}
		props := parseProperties(string(raw))
		minor, hasRender := props["drm_render_minor"]
		if props["simd_count"] == 0 || props["gfx_target_version"] == 0 || !hasRender {
			continue
		}
		// KFD sysfs can describe every card on the host while the container
		// has one render node. Match libkfdreplay's device discovery.
		if _, err := os.Stat(path.Join(DRIDir, fmt.Sprintf("renderD%d", minor))); err != nil {
			if errors.Is(err, fs.ErrNotExist) || errors.Is(err, fs.ErrPermission) {
				continue
			}
			return nil, fmt.Errorf("kfd: node %d render device: %w", n, err)
		}
		idRaw, err := readFile(path.Join(dir, "gpu_id"))
		if errors.Is(err, fs.ErrPermission) {
			continue
		}
		if err != nil {
			return nil, fmt.Errorf("kfd: node %d gpu_id: %w", n, err)
		}
		id, err := strconv.ParseUint(strings.TrimSpace(string(idRaw)), 10, 64)
		if err != nil || id == 0 {
			continue
		}
		out = append(out, kfdNode{node: n, gpuID: id, props: props, memBytes: kfdMemory(dir, props)})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].node < out[j].node })
	return out, nil
}

// kfdMemory is the node's VRAM: the sum of its memory banks. local_mem_size
// reads 0 on MI350X.
func kfdMemory(dir string, props map[string]uint64) uint64 {
	banks, err := os.ReadDir(path.Join(dir, "mem_banks"))
	if err != nil {
		return props["local_mem_size"]
	}
	var total uint64
	for _, b := range banks {
		raw, err := readFile(path.Join(dir, "mem_banks", b.Name(), "properties"))
		if err != nil {
			continue
		}
		p := parseProperties(string(raw))
		// Heap type 1 and 2 are public and private frame-buffer memory.
		if t := p["heap_type"]; t == 1 || t == 2 {
			total += p["size_in_bytes"]
		}
	}
	if total == 0 {
		return props["local_mem_size"]
	}
	return total
}

func parseProperties(raw string) map[string]uint64 {
	out := map[string]uint64{}
	for _, line := range strings.Split(raw, "\n") {
		key, value, ok := strings.Cut(strings.TrimSpace(line), " ")
		if !ok {
			continue
		}
		if v, err := strconv.ParseUint(strings.TrimSpace(value), 10, 64); err == nil {
			out[key] = v
		}
	}
	return out
}

// GFX is the node's ISA name: gfx_target_version 90500 is gfx950.
func GFX(version uint64) string {
	major, minor, step := version/10000, version/100%100, version%100
	return fmt.Sprintf("gfx%d%x%x", major, minor, step)
}

// pciAddress is the node's PCI address as sysfs writes it: location_id holds
// bus<<8 | device<<3 | function.
func pciAddress(props map[string]uint64) string {
	loc := props["location_id"]
	return fmt.Sprintf("%04x:%02x:%02x.%x", props["domain"], loc>>8, (loc>>3)&0x1f, loc&0x7)
}

// amdDriver is the amdgpu module's version (the DKMS driver writes one), or
// the kernel's release for the in-tree driver.
func amdDriver() string {
	if raw, err := readFile(path.Join(AMDGPUModule, "version")); err == nil {
		if v := strings.TrimSpace(string(raw)); v != "" {
			return v
		}
	}
	return "inbox-" + Kernel()
}

// kfdSession answers the Session queries from the KFD topology. Process
// counts come from /proc: a container sees only its own processes, and a
// GPU is busy for warmctl when another of them holds /dev/kfd open.
type kfdSession struct{ nodes []kfdNode }

func openKFD() (Session, error) {
	nodes, err := kfdNodes()
	if err != nil {
		return nil, err
	}
	if len(nodes) == 0 {
		return nil, errors.New("kfd: no GPU this process may read")
	}
	return &kfdSession{nodes: nodes}, nil
}

func (s *kfdSession) Driver() (string, error) { return amdDriver(), nil }
func (s *kfdSession) Count() (int, error)     { return len(s.nodes), nil }

func (s *kfdSession) GPU(i int) (GPU, error) {
	if i < 0 || i >= len(s.nodes) {
		return GPU{}, fmt.Errorf("kfd: no GPU %d", i)
	}
	n := s.nodes[i]
	device := n.props["device_id"]
	name, ok := amdNames[device]
	if !ok {
		name = fmt.Sprintf("AMD GPU %s (device 0x%04x)", GFX(n.props["gfx_target_version"]), device)
	}
	g := GPU{
		Vendor: VendorAMD, Name: name, PCIBusID: pciAddress(n.props),
		MemoryMiB: int64(n.memBytes >> 20), Arch: GFX(n.props["gfx_target_version"]), KFDGPUID: n.gpuID,
		UUID: fmt.Sprintf("GPU-%016x", n.props["unique_id"]),
	}
	g.RenderNode = fmt.Sprintf("/dev/dri/renderD%d", n.props["drm_render_minor"])
	if n.props["unique_id"] == 0 {
		// Without a unique ID the GPU ID stands in; it is unique on a host.
		g.UUID = fmt.Sprintf("GPU-kfd-%d", n.gpuID)
	}
	return g, nil
}

// Processes counts the other processes visible here that hold /dev/kfd: a
// ROCm process opens it once for all GPUs, so each one counts against
// every GPU.
func (s *kfdSession) Processes(int) (int, error) { return kfdHolders(os.Getpid()), nil }

func (s *kfdSession) NVLink(int, int) (bool, error) { return false, nil }

// Path between two AMD GPUs is not classified yet: warmctl restores one GPU
// per engine on AMD.
func (s *kfdSession) Path(int, int) (string, error) { return "SYS", nil }
func (s *kfdSession) Close() error                  { return nil }

// kfdHolders counts the processes other than self that hold KFDDevice open.
func kfdHolders(self int) int {
	procs, err := os.ReadDir("/proc")
	if err != nil {
		return 0
	}
	n := 0
	for _, p := range procs {
		pid, err := strconv.Atoi(p.Name())
		if err != nil || pid == self {
			continue
		}
		fds, err := os.ReadDir(fmt.Sprintf("/proc/%d/fd", pid))
		if err != nil {
			continue
		}
		for _, fd := range fds {
			if target, err := os.Readlink(fmt.Sprintf("/proc/%d/fd/%s", pid, fd.Name())); err == nil &&
				target == KFDDevice {
				n++
				break
			}
		}
	}
	return n
}
