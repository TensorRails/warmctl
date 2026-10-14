package host

import (
	"errors"
	"fmt"
	"sync"

	"github.com/ebitengine/purego"
)

// nvmlLibrary ships with the NVIDIA driver; the NVIDIA container runtime
// mounts it into a container with the utility capability.
const nvmlLibrary = "libnvidia-ml.so.1"

// From nvml.h.
const (
	nvmlSuccess          = 0
	nvmlNotSupported     = 3
	nvmlInsufficientSize = 7
	p2pCapsNVLink        = 2
	p2pStatusOK          = 0
	uuidSize             = 96 // NVML_DEVICE_UUID_V2_BUFFER_SIZE
	nameSize             = 96 // NVML_DEVICE_NAME_V2_BUFFER_SIZE
	driverSize           = 80 // NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE
)

// pciInfo is nvmlPciInfo_t: the legacy bus ID, domain, bus, device, device
// and subsystem IDs, then the bus ID warmctl reads.
type pciInfo struct {
	_     [16]byte
	_     [5]uint32
	busID [32]byte
}

// memoryInfo is nvmlMemory_t: total, free, used.
type memoryInfo struct {
	total uint64
	_     [2]uint64
}

// processInfo is nvmlProcessInfo_t (v2): pid, used memory, GPU and compute
// instance. warmctl passes none, only the type.
type processInfo struct {
	_ uint32
	_ uint64
	_ [2]uint32
}

// pathNames are nvmlGpuTopologyLevel_t values as nvidia-smi topo -m prints them.
var pathNames = map[uint32]string{0: "BOARD", 10: "PIX", 20: "PXB", 30: "PHB", 40: "NODE", 50: "SYS"}

// nvml holds the library's functions, bound once per process.
var nvml struct {
	once          sync.Once
	err           error
	init          func() uint32
	shutdown      func() uint32
	errorString   func(uint32) string
	driverVersion func(*byte, uint32) uint32
	count         func(*uint32) uint32
	handle        func(uint32, *uintptr) uint32
	uuid          func(uintptr, *byte, uint32) uint32
	name          func(uintptr, *byte, uint32) uint32
	pci           func(uintptr, *pciInfo) uint32
	memory        func(uintptr, *memoryInfo) uint32
	processes     func(uintptr, *uint32, *processInfo) uint32
	p2p           func(uintptr, uintptr, uint32, *uint32) uint32
	ancestor      func(uintptr, uintptr, *uint32) uint32
}

func loadNVML() error {
	nvml.once.Do(func() {
		lib, err := purego.Dlopen(nvmlLibrary, purego.RTLD_NOW|purego.RTLD_GLOBAL)
		if err != nil {
			nvml.err = fmt.Errorf("nvml: %w (no NVIDIA driver, or a container without the utility capability)", err)
			return
		}
		for _, f := range []struct {
			fn   any
			name string
		}{
			{&nvml.init, "nvmlInit_v2"}, {&nvml.shutdown, "nvmlShutdown"}, {&nvml.errorString, "nvmlErrorString"},
			{&nvml.driverVersion, "nvmlSystemGetDriverVersion"}, {&nvml.count, "nvmlDeviceGetCount_v2"},
			{&nvml.handle, "nvmlDeviceGetHandleByIndex_v2"}, {&nvml.uuid, "nvmlDeviceGetUUID"},
			{&nvml.name, "nvmlDeviceGetName"}, {&nvml.pci, "nvmlDeviceGetPciInfo_v3"},
			{&nvml.memory, "nvmlDeviceGetMemoryInfo"}, {&nvml.processes, "nvmlDeviceGetComputeRunningProcesses_v3"},
			{&nvml.p2p, "nvmlDeviceGetP2PStatus"}, {&nvml.ancestor, "nvmlDeviceGetTopologyCommonAncestor"},
		} {
			sym, err := purego.Dlsym(lib, f.name)
			if err != nil {
				nvml.err = fmt.Errorf("nvml: %s: %w", f.name, err)
				return
			}
			purego.RegisterFunc(f.fn, sym)
		}
	})
	return nvml.err
}

func nvmlError(what string, ret uint32) error {
	if ret == nvmlSuccess {
		return nil
	}
	return fmt.Errorf("nvml: %s: %s (%d)", what, nvml.errorString(ret), ret)
}

func openNVML() (Session, error) {
	if err := loadNVML(); err != nil {
		return nil, err
	}
	if err := nvmlError("init", nvml.init()); err != nil {
		return nil, err
	}
	return &nvmlSession{handles: map[int]uintptr{}}, nil
}

type nvmlSession struct{ handles map[int]uintptr }

func (s *nvmlSession) device(i int) (uintptr, error) {
	if h, ok := s.handles[i]; ok {
		return h, nil
	}
	var h uintptr
	if err := nvmlError(fmt.Sprintf("GPU %d", i), nvml.handle(uint32(i), &h)); err != nil {
		return 0, err
	}
	s.handles[i] = h
	return h, nil
}

func (s *nvmlSession) Driver() (string, error) {
	buf := make([]byte, driverSize)
	err := nvmlError("driver version", nvml.driverVersion(&buf[0], driverSize))
	return cString(buf), err
}

func (s *nvmlSession) Count() (int, error) {
	var n uint32
	err := nvmlError("device count", nvml.count(&n))
	return int(n), err
}

func (s *nvmlSession) GPU(i int) (GPU, error) {
	d, err := s.device(i)
	if err != nil {
		return GPU{}, err
	}
	uuid, name := make([]byte, uuidSize), make([]byte, nameSize)
	var pci pciInfo
	var mem memoryInfo
	err = errors.Join(
		nvmlError(fmt.Sprintf("GPU %d UUID", i), nvml.uuid(d, &uuid[0], uuidSize)),
		nvmlError(fmt.Sprintf("GPU %d name", i), nvml.name(d, &name[0], nameSize)),
		nvmlError(fmt.Sprintf("GPU %d PCI info", i), nvml.pci(d, &pci)),
		nvmlError(fmt.Sprintf("GPU %d memory", i), nvml.memory(d, &mem)))
	return GPU{UUID: cString(uuid), Name: cString(name), PCIBusID: cString(pci.busID[:]),
		MemoryMiB: int64(mem.total >> 20)}, err
}

// Processes asks for the list with no room for it: NVML answers with the
// count, which is all warmctl needs.
func (s *nvmlSession) Processes(i int) (int, error) {
	d, err := s.device(i)
	if err != nil {
		return 0, err
	}
	var n uint32
	if ret := nvml.processes(d, &n, nil); ret != nvmlInsufficientSize {
		return int(n), nvmlError(fmt.Sprintf("GPU %d processes", i), ret)
	}
	return int(n), nil
}

func (s *nvmlSession) pair(i, j int) (uintptr, uintptr, error) {
	a, err := s.device(i)
	if err != nil {
		return 0, 0, err
	}
	b, err := s.device(j)
	return a, b, err
}

func (s *nvmlSession) NVLink(i, j int) (bool, error) {
	a, b, err := s.pair(i, j)
	if err != nil {
		return false, err
	}
	var status uint32
	ret := nvml.p2p(a, b, p2pCapsNVLink, &status)
	if ret == nvmlNotSupported {
		return false, nil
	}
	return status == p2pStatusOK, nvmlError(fmt.Sprintf("GPU %d-%d NVLink", i, j), ret)
}

func (s *nvmlSession) Path(i, j int) (string, error) {
	a, b, err := s.pair(i, j)
	if err != nil {
		return "", err
	}
	var level uint32
	ret := nvml.ancestor(a, b, &level)
	if ret == nvmlNotSupported {
		return "?", nil
	}
	if name, ok := pathNames[level]; ok || ret != nvmlSuccess {
		return name, nvmlError(fmt.Sprintf("GPU %d-%d path", i, j), ret)
	}
	return fmt.Sprintf("L%d", level), nil
}

func (s *nvmlSession) Close() error {
	return nvmlError("shutdown", nvml.shutdown())
}

func cString(b []byte) string {
	for i, c := range b {
		if c == 0 {
			return string(b[:i])
		}
	}
	return string(b)
}
