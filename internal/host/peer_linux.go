//go:build linux

package host

import (
	"encoding/json"
	"fmt"
	"io"
	"runtime"
	"time"
	"unsafe"

	"github.com/ebitengine/purego"
)

// peerBytes is how much one direction of the probe copies.
const peerBytes = 64 << 20

// From cuda.h.
const (
	cudaSuccess                  = 0
	cudaPeerAccessAlreadyEnabled = 704
)

var cu struct {
	init        func(uint32) uint32
	count       func(*int32) uint32
	device      func(*int32, int32) uint32
	uuid        func(*[16]byte, int32) uint32
	retain      func(*uintptr, int32) uint32
	setCurrent  func(uintptr) uint32
	alloc       func(*uint64, uint64) uint32
	memset      func(uint64, uint8, uint64) uint32
	canAccess   func(*int32, int32, int32) uint32
	enablePeer  func(uintptr, uint32) uint32
	copyPeer    func(uint64, uintptr, uint64, uintptr, uint64) uint32
	synchronize func() uint32
	copyToHost  func(unsafe.Pointer, uint64, uint64) uint32
	errorString func(uint32, **byte) uint32
}

func loadCUDA() error {
	lib, err := purego.Dlopen("libcuda.so.1", purego.RTLD_NOW)
	if err != nil {
		return fmt.Errorf("cuda driver API: %w", err)
	}
	for _, f := range []struct {
		fn   any
		name string
	}{
		{&cu.init, "cuInit"}, {&cu.count, "cuDeviceGetCount"}, {&cu.device, "cuDeviceGet"},
		{&cu.uuid, "cuDeviceGetUuid_v2"}, {&cu.retain, "cuDevicePrimaryCtxRetain"},
		{&cu.setCurrent, "cuCtxSetCurrent"}, {&cu.alloc, "cuMemAlloc_v2"}, {&cu.memset, "cuMemsetD8_v2"},
		{&cu.canAccess, "cuDeviceCanAccessPeer"}, {&cu.enablePeer, "cuCtxEnablePeerAccess"},
		{&cu.copyPeer, "cuMemcpyPeer"}, {&cu.synchronize, "cuCtxSynchronize"},
		{&cu.copyToHost, "cuMemcpyDtoH_v2"}, {&cu.errorString, "cuGetErrorString"},
	} {
		sym, err := purego.Dlsym(lib, f.name)
		if err != nil {
			return fmt.Errorf("cuda driver API: %s: %w", f.name, err)
		}
		purego.RegisterFunc(f.fn, sym)
	}
	return nil
}

func cuCheck(what string, ret uint32) error {
	if ret == cudaSuccess {
		return nil
	}
	var msg *byte
	cu.errorString(ret, &msg)
	text := "unknown"
	if msg != nil {
		text = goString(msg)
	}
	return fmt.Errorf("%s: CUDA error %d (%s)", what, ret, text)
}

func goString(p *byte) string {
	var b []byte
	for ; *p != 0; p = (*byte)(unsafe.Add(unsafe.Pointer(p), 1)) {
		b = append(b, *p)
	}
	return string(b)
}

// RunPeerProbe is the child side of ProbePeers: uuids are the GPUs' UUIDs
// in rank order, as NVML reports them. It writes a line "copy I J" before
// each copy and a PeerResult as JSON at the end to w.
func RunPeerProbe(uuids []string, w io.Writer) error {
	if len(uuids) < 2 {
		return fmt.Errorf("%s needs two or more GPU UUIDs", PeerProbeCommand)
	}
	// A context is current per OS thread.
	runtime.LockOSThread()
	if err := loadCUDA(); err != nil {
		return err
	}
	if err := cuCheck("cuInit", cu.init(0)); err != nil {
		return err
	}
	devs, err := devicesByUUID(uuids)
	if err != nil {
		return err
	}
	res := PeerResult{}
	if res.Claimed, err = peersClaimed(devs); err != nil {
		return err
	}
	if res.Claimed {
		// No engine takes a peer path the driver does not offer.
		if res.Copies, err = copyAll(devs, w); err != nil {
			return err
		}
	}
	return json.NewEncoder(w).Encode(res)
}

// peersClaimed is whether the driver reports peer access between every
// ordered pair of devs.
func peersClaimed(devs []int32) (bool, error) {
	for i := range devs {
		for j := range devs {
			var can int32
			if i == j {
				continue
			}
			if err := cuCheck("cuDeviceCanAccessPeer", cu.canAccess(&can, devs[j], devs[i])); err != nil {
				return false, err
			}
			if can == 0 {
				return false, nil
			}
		}
	}
	return true, nil
}

// copyAll gives every device a context and a buffer, then copies between
// every ordered pair, announcing each copy on w.
func copyAll(devs []int32, w io.Writer) ([]PeerCopy, error) {
	ctxs, bufs := make([]uintptr, len(devs)), make([]uint64, len(devs))
	for i := range devs {
		if err := cuCheck("cuDevicePrimaryCtxRetain", cu.retain(&ctxs[i], devs[i])); err != nil {
			return nil, err
		}
		if err := cuCheck("cuCtxSetCurrent", cu.setCurrent(ctxs[i])); err != nil {
			return nil, err
		}
		if err := cuCheck("cuMemAlloc", cu.alloc(&bufs[i], peerBytes)); err != nil {
			return nil, err
		}
	}
	var copies []PeerCopy
	for i := range devs {
		for j := range devs {
			if i == j {
				continue
			}
			fmt.Fprintf(w, "copy %d %d\n", i, j)
			c, err := peerCopy(ctxs, bufs, i, j)
			if err != nil {
				return nil, err
			}
			copies = append(copies, c)
		}
	}
	return copies, nil
}

// devicesByUUID finds the CUDA devices of uuids. NVML's PCI bus IDs need not
// be CUDA's (a RunPod pod's NVML shows 66:00.0 where CUDA shows A6:00.0);
// UUIDs are the same.
func devicesByUUID(uuids []string) ([]int32, error) {
	var n int32
	if err := cuCheck("cuDeviceGetCount", cu.count(&n)); err != nil {
		return nil, err
	}
	byUUID := map[string]int32{}
	for i := int32(0); i < n; i++ {
		var dev int32
		var id [16]byte
		if err := cuCheck("cuDeviceGet", cu.device(&dev, i)); err != nil {
			return nil, err
		}
		if err := cuCheck("cuDeviceGetUuid", cu.uuid(&id, dev)); err != nil {
			return nil, err
		}
		byUUID[formatUUID(id)] = dev
	}
	devs := make([]int32, len(uuids))
	for i, u := range uuids {
		dev, ok := byUUID[u]
		if !ok {
			return nil, fmt.Errorf("no CUDA device has UUID %s", u)
		}
		devs[i] = dev
	}
	return devs, nil
}

// peerCopy copies GPU i's buffer into GPU j's with j's context mapping i's
// memory, as an engine's collectives do, and checks the bytes.
func peerCopy(ctxs []uintptr, bufs []uint64, i, j int) (PeerCopy, error) {
	if err := cuCheck("cuCtxSetCurrent", cu.setCurrent(ctxs[i])); err != nil {
		return PeerCopy{}, err
	}
	mark := uint8(0x40 + i)
	if err := cuCheck("cuMemsetD8", cu.memset(bufs[i], mark, peerBytes)); err != nil {
		return PeerCopy{}, err
	}
	if err := cuCheck("cuCtxSynchronize", cu.synchronize()); err != nil {
		return PeerCopy{}, err
	}
	if err := cuCheck("cuCtxSetCurrent", cu.setCurrent(ctxs[j])); err != nil {
		return PeerCopy{}, err
	}
	if ret := cu.enablePeer(ctxs[i], 0); ret != cudaPeerAccessAlreadyEnabled {
		if err := cuCheck("cuCtxEnablePeerAccess", ret); err != nil {
			return PeerCopy{}, err
		}
	}
	start := time.Now()
	if err := cuCheck("cuMemcpyPeer", cu.copyPeer(bufs[j], ctxs[j], bufs[i], ctxs[i], peerBytes)); err != nil {
		return PeerCopy{}, err
	}
	if err := cuCheck("cuCtxSynchronize", cu.synchronize()); err != nil {
		return PeerCopy{}, err
	}
	seconds := time.Since(start).Seconds()
	var head [4096]byte
	if err := cuCheck("cuMemcpyDtoH", cu.copyToHost(unsafe.Pointer(&head[0]), bufs[j], uint64(len(head)))); err != nil {
		return PeerCopy{}, err
	}
	for _, b := range head {
		if b != mark {
			return PeerCopy{}, fmt.Errorf("a peer copy from GPU %d to GPU %d delivered other bytes", i, j)
		}
	}
	return PeerCopy{From: i, To: j, GBps: peerBytes / seconds / 1e9}, nil
}
