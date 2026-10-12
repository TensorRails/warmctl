// Package host reads the facts a snapshot depends on from the machine: the
// GPUs (nvidia-smi), their topology, the driver and kernel, and holds an
// exclusive lease on the GPUs a run uses.
package host

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/TensorRails/warmctl/v2/internal/process"
)

// GPU is one device as nvidia-smi reports it.
type GPU struct {
	Index     int    `json:"index"`
	UUID      string `json:"uuid"`
	Name      string `json:"name"`
	PCIBusID  string `json:"pci_bus_id"`
	MemoryMiB int64  `json:"memory_mib"`
	Driver    string `json:"driver"`
}

// SMI is the nvidia-smi executable.
var SMI = "nvidia-smi"

// Inventory lists every GPU of the host.
func Inventory(ctx context.Context) ([]GPU, error) {
	out, err := process.Capture(ctx, 30*time.Second, SMI,
		"--query-gpu=index,uuid,name,pci.bus_id,memory.total,driver_version", "--format=csv,noheader,nounits")
	if err != nil {
		return nil, fmt.Errorf("nvidia-smi: %w %s", err, out.Stderr)
	}
	return ParseInventory(out.Stdout)
}

// ParseInventory reads nvidia-smi's CSV.
func ParseInventory(text string) ([]GPU, error) {
	var gpus []GPU
	for _, line := range strings.Split(strings.TrimSpace(text), "\n") {
		if strings.TrimSpace(line) == "" {
			continue
		}
		f := strings.Split(line, ",")
		if len(f) != 6 {
			return nil, fmt.Errorf("nvidia-smi: unexpected line %q", line)
		}
		for i := range f {
			f[i] = strings.TrimSpace(f[i])
		}
		index, err1 := strconv.Atoi(f[0])
		memory, err2 := strconv.ParseInt(f[4], 10, 64)
		if err1 != nil || err2 != nil || !strings.HasPrefix(f[1], "GPU-") {
			return nil, fmt.Errorf("nvidia-smi: unexpected line %q", line)
		}
		gpus = append(gpus, GPU{Index: index, UUID: f[1], Name: f[2], PCIBusID: normalizeBus(f[3]),
			MemoryMiB: memory, Driver: f[5]})
	}
	if len(gpus) == 0 {
		return nil, errors.New("nvidia-smi: no GPUs")
	}
	return gpus, nil
}

// normalizeBus turns nvidia-smi's 00000000:18:00.0 into sysfs's 0000:18:00.0.
func normalizeBus(bus string) string {
	bus = strings.ToLower(bus)
	if domain, rest, ok := strings.Cut(bus, ":"); ok && len(domain) == 8 {
		return domain[4:] + ":" + rest
	}
	return bus
}

// Select picks GPUs by index list ("0-3,6") or all when spec is empty, in
// PCI bus order: the order CUDA gives ranks with CUDA_DEVICE_ORDER=PCI_BUS_ID.
func Select(all []GPU, spec string) ([]GPU, error) {
	byIndex := map[int]GPU{}
	for _, g := range all {
		byIndex[g.Index] = g
	}
	var out []GPU
	if strings.TrimSpace(spec) == "" {
		out = append(out, all...)
	} else {
		indices, err := ParseIndices(spec)
		if err != nil {
			return nil, err
		}
		for _, i := range indices {
			g, ok := byIndex[i]
			if !ok {
				return nil, fmt.Errorf("no GPU %d on this host", i)
			}
			out = append(out, g)
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i].PCIBusID < out[j].PCIBusID })
	return out, nil
}

// ParseIndices reads "0-3,6".
func ParseIndices(spec string) ([]int, error) {
	seen := map[int]bool{}
	var out []int
	for _, item := range strings.Split(spec, ",") {
		low, high, ranged := strings.Cut(strings.TrimSpace(item), "-")
		a, err := strconv.Atoi(low)
		b := a
		if ranged {
			b, err = strconv.Atoi(high)
		}
		if err != nil || a < 0 || b < a {
			return nil, fmt.Errorf("invalid GPU list %q", spec)
		}
		for i := a; i <= b; i++ {
			if seen[i] {
				return nil, fmt.Errorf("GPU %d listed twice", i)
			}
			seen[i] = true
			out = append(out, i)
		}
	}
	return out, nil
}

// Busy counts compute processes on the GPUs.
func Busy(ctx context.Context, gpus []GPU) (int, error) {
	procs, err := Processes(ctx)
	n := 0
	for _, g := range gpus {
		n += procs[g.UUID]
	}
	return n, err
}

// Processes counts the compute processes on each GPU, by UUID.
func Processes(ctx context.Context) (map[string]int, error) {
	out, err := process.Capture(ctx, 30*time.Second, SMI, "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader")
	if err != nil {
		return nil, fmt.Errorf("nvidia-smi: %w", err)
	}
	procs := map[string]int{}
	for _, line := range strings.Split(out.Stdout, "\n") {
		if uuid, _, _ := strings.Cut(line, ","); strings.TrimSpace(uuid) != "" {
			procs[strings.TrimSpace(uuid)]++
		}
	}
	return procs, nil
}

// Pick chooses n GPUs named name, in PCI order, that run no compute process
// (procs, from Processes) and that no warmctl run leases in locks.
func Pick(all []GPU, n int, name string, procs map[string]int, locks string) ([]GPU, error) {
	sorted := append([]GPU(nil), all...)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i].PCIBusID < sorted[j].PCIBusID })
	var out []GPU
	for _, g := range sorted {
		if g.Name == name && procs[g.UUID] == 0 && !Leased(locks, g) {
			if out = append(out, g); len(out) == n {
				return out, nil
			}
		}
	}
	return nil, fmt.Errorf("needs %d free %s, this host has %s with %d of them free", n, name, Describe(all), len(out))
}

// Describe summarizes GPUs by model: "8× NVIDIA H100 80GB HBM3".
func Describe(gpus []GPU) string {
	var names []string
	count := map[string]int{}
	for _, g := range gpus {
		if count[g.Name]++; count[g.Name] == 1 {
			names = append(names, g.Name)
		}
	}
	parts := make([]string, len(names))
	for i, name := range names {
		parts[i] = fmt.Sprintf("%d× %s", count[name], name)
	}
	if len(parts) == 0 {
		return "no GPUs"
	}
	return strings.Join(parts, ", ")
}

var (
	nvlinkCell = regexp.MustCompile(`^NV\d+$`)
	// nvidia-smi underlines the topology header with SGR escapes (P129).
	ansi = regexp.MustCompile(`\x1b\[[0-9;]*[A-Za-z]`)
)

// Topology is the class of the links between the GPUs (nvlink or pcie) and
// a digest of nvidia-smi's matrix for the record.
func Topology(ctx context.Context, gpus []GPU) (string, string, error) {
	out, err := process.Capture(ctx, 30*time.Second, SMI, "topo", "-m")
	if err != nil {
		return "", "", fmt.Errorf("nvidia-smi topo: %w", err)
	}
	sum := sha256.Sum256([]byte(out.Stdout))
	return TopologyClass(out.Stdout, gpus), hex.EncodeToString(sum[:]), nil
}

// TopologyClass is nvlink when every pair of the selected GPUs is joined by
// NVLink, pcie otherwise; a single GPU is "single".
func TopologyClass(matrix string, gpus []GPU) string {
	if len(gpus) < 2 {
		return "single"
	}
	want := map[string]bool{}
	for _, g := range gpus {
		want[fmt.Sprintf("GPU%d", g.Index)] = true
	}
	var header []string
	for _, line := range strings.Split(ansi.ReplaceAllString(matrix, ""), "\n") {
		f := strings.Fields(line)
		if len(f) > 0 && f[0] == "GPU0" && header == nil {
			header = f
			continue
		}
		if header == nil || len(f) == 0 || !want[f[0]] {
			continue
		}
		for i, cell := range f[1:] {
			if i < len(header) && want[header[i]] && header[i] != f[0] && !nvlinkCell.MatchString(cell) {
				return "pcie"
			}
		}
	}
	return "nvlink"
}

// Kernel is uname -r.
func Kernel() string {
	raw, _ := os.ReadFile("/proc/sys/kernel/osrelease")
	return strings.TrimSpace(string(raw))
}

// IOUringDisabled is kernel.io_uring_disabled; CRIU cannot dump io_uring,
// so a snapshot needs 2.
func IOUringDisabled() (int, error) {
	raw, err := os.ReadFile("/proc/sys/kernel/io_uring_disabled")
	if errors.Is(err, os.ErrNotExist) {
		return 2, nil
	}
	if err != nil {
		return 0, err
	}
	return strconv.Atoi(strings.TrimSpace(string(raw)))
}

// Lease holds an exclusive flock per GPU for the life of a run.
type Lease struct{ files []*os.File }

// Acquire locks <dir>/gpu-<uuid>.lock for every GPU without waiting.
func Acquire(dir string, gpus []GPU) (*Lease, error) {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, err
	}
	l := &Lease{}
	for _, g := range gpus {
		f, err := os.OpenFile(filepath.Join(dir, "gpu-"+g.UUID+".lock"), os.O_CREATE|os.O_RDWR, 0o644)
		if err != nil {
			l.Release()
			return nil, err
		}
		if err := syscall.Flock(int(f.Fd()), syscall.LOCK_EX|syscall.LOCK_NB); err != nil {
			f.Close()
			l.Release()
			return nil, fmt.Errorf("GPU %d (%s) is leased by another warmctl run", g.Index, g.UUID)
		}
		l.files = append(l.files, f)
	}
	return l, nil
}

// Leased reports whether a warmctl run holds the GPU's lease in dir.
func Leased(dir string, g GPU) bool {
	if dir == "" {
		return false
	}
	f, err := os.Open(filepath.Join(dir, "gpu-"+g.UUID+".lock"))
	if err != nil {
		return false
	}
	defer f.Close()
	if syscall.Flock(int(f.Fd()), syscall.LOCK_EX|syscall.LOCK_NB) != nil {
		return true
	}
	_ = syscall.Flock(int(f.Fd()), syscall.LOCK_UN)
	return false
}

// Release drops the locks.
func (l *Lease) Release() {
	for _, f := range l.files {
		_ = syscall.Flock(int(f.Fd()), syscall.LOCK_UN)
		f.Close()
	}
	l.files = nil
}

// CPUFlags are the first CPU's feature flags, sorted: a restored process
// may use any instruction the donor's CPU had.
func CPUFlags() []string {
	raw, err := os.ReadFile("/proc/cpuinfo")
	if err != nil {
		return nil
	}
	for _, line := range strings.Split(string(raw), "\n") {
		if _, flags, ok := strings.Cut(line, ":"); ok && strings.HasPrefix(line, "flags") {
			out := strings.Fields(flags)
			sort.Strings(out)
			return out
		}
	}
	return nil
}

// MissingFlags are the donor's CPU flags this host lacks.
func MissingFlags(donor, here []string) []string {
	have := map[string]bool{}
	for _, f := range here {
		have[f] = true
	}
	var missing []string
	for _, f := range donor {
		if !have[f] {
			missing = append(missing, f)
		}
	}
	return missing
}
