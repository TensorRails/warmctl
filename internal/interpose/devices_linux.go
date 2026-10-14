//go:build linux

package interpose

import (
	"bytes"
	"fmt"
	"strings"
	"sync"
	"unsafe"

	"github.com/ebitengine/purego"
)

// cudaLibrary is the driver API, which ships with the NVIDIA driver; the
// NVIDIA container runtime mounts it into a container with the compute
// capability. A test points it at a stand-in.
var cudaLibrary = "libcuda.so.1"

// From cuda.h: CUdevice_attribute values and the result codes read here.
const (
	attrComputeMajor = 75
	attrComputeMinor = 76
	attrVMM          = 102
	attrPosixFD      = 103
	attrMulticast    = 132
	cudaSuccess      = 0
	nameSize         = 256
)

// cu holds the driver's functions, bound once per process. The probe calls
// cuInit only, so it creates no context and takes no device memory.
var cu struct {
	once        sync.Once
	err         error
	init        func(uint32) uint32
	count       func(*int32) uint32
	device      func(*int32, int32) uint32
	name        func(*byte, int32, int32) uint32
	attribute   func(*int32, int32, int32) uint32
	errorString func(uint32, **byte) uint32
}

func loadCUDA() error {
	cu.once.Do(func() {
		lib, err := purego.Dlopen(cudaLibrary, purego.RTLD_NOW)
		if err != nil {
			cu.err = fmt.Errorf("device probe: %w (no NVIDIA driver, or a container without the compute capability)", err)
			return
		}
		for _, f := range []struct {
			fn   any
			name string
		}{
			{&cu.init, "cuInit"}, {&cu.count, "cuDeviceGetCount"}, {&cu.device, "cuDeviceGet"},
			{&cu.name, "cuDeviceGetName"}, {&cu.attribute, "cuDeviceGetAttribute"},
			{&cu.errorString, "cuGetErrorString"},
		} {
			sym, err := purego.Dlsym(lib, f.name)
			if err != nil {
				cu.err = fmt.Errorf("device probe: %s: %w", f.name, err)
				return
			}
			purego.RegisterFunc(f.fn, sym)
		}
	})
	return cu.err
}

func cuError(what string, ret uint32) error {
	if ret == cudaSuccess {
		return nil
	}
	var p *byte
	if cu.errorString(ret, &p) != cudaSuccess || p == nil {
		return fmt.Errorf("device probe: %s: CUDA error %d", what, ret)
	}
	return fmt.Errorf("device probe: %s: %s (%d)", what, goString(p), ret)
}

// goString copies the NUL-terminated string at p.
func goString(p *byte) string {
	n := 0
	for *(*byte)(unsafe.Add(unsafe.Pointer(p), n)) != 0 {
		n++
	}
	return strings.Clone(unsafe.String(p, n))
}

func probeDevices() ([]Device, error) {
	if err := loadCUDA(); err != nil {
		return nil, err
	}
	if err := cuError("cuInit", cu.init(0)); err != nil {
		return nil, err
	}
	var n int32
	if err := cuError("device count", cu.count(&n)); err != nil {
		return nil, err
	}
	devices := make([]Device, 0, n)
	for i := range int(n) {
		d, err := describe(i)
		if err != nil {
			return nil, err
		}
		devices = append(devices, d)
	}
	return devices, nil
}

// describe reads device ordinal i: its name and the attributes Device holds.
func describe(i int) (Device, error) {
	var dev int32
	if err := cuError(fmt.Sprintf("GPU %d", i), cu.device(&dev, int32(i))); err != nil {
		return Device{}, err
	}
	name := make([]byte, nameSize)
	if err := cuError(fmt.Sprintf("GPU %d name", i), cu.name(&name[0], nameSize, dev)); err != nil {
		return Device{}, err
	}
	var attrs [5]int32
	for k, a := range [...]int32{attrComputeMajor, attrComputeMinor, attrVMM, attrPosixFD, attrMulticast} {
		if err := cuError(fmt.Sprintf("GPU %d attribute %d", i, a), cu.attribute(&attrs[k], a, dev)); err != nil {
			return Device{}, err
		}
	}
	if end := bytes.IndexByte(name, 0); end >= 0 {
		name = name[:end]
	}
	return Device{Index: i, Name: string(name), ComputeCapability: fmt.Sprintf("%d.%d", attrs[0], attrs[1]),
		VMM: attrs[2] != 0, PosixFD: attrs[3] != 0, Multicast: attrs[4] != 0}, nil
}
