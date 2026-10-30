//go:build linux

package cuda

import (
	"context"
	"fmt"
	"runtime"
	"strings"
	"sync"
	"unsafe"

	"github.com/ebitengine/purego"
)

// cudaLibrary is the driver API. The agent resolves it with the same
// environment cuda-checkpoint would inherit (LD_LIBRARY_PATH, the driver's
// version-check override), so both load the same libcuda. A test points it
// at a stand-in.
var cudaLibrary = "libcuda.so.1"

// minDriverVersion is the first CUDA driver API whose restore arguments
// carry the GPU pairs at the start of the struct (CUDA 13.0); 13.4 adds a
// custom-storage pointer after the count, NULL when zeroed.
const minDriverVersion = 13000

// From cuda.h: CUprocessState, and the 64-byte argument structs of the
// checkpoint calls. Every reserved field stays zero.
var processStates = map[int32]string{0: "running", 1: "locked", 2: "checkpointed", 3: "failed"}

type lockArgs struct {
	TimeoutMS uint32
	_         uint32
	_         [7]uint64
}

type plainArgs struct{ _ [8]uint64 }

type gpuPair struct{ Old, New [16]byte }

type restoreArgs struct {
	Pairs *gpuPair
	Count uint32
	_     [44]byte
	_     uint64
}

var driver struct {
	once        sync.Once
	err         error
	version     int32
	getVersion  func(*int32) uint32
	errorString func(uint32, **byte) uint32
	getState    func(int32, *int32) uint32
	lock        func(int32, *lockArgs) uint32
	checkpoint  func(int32, *plainArgs) uint32
	restore     func(int32, *restoreArgs) uint32
	unlock      func(int32, *plainArgs) uint32
}

func loadDriver() error {
	driver.once.Do(func() {
		lib, err := purego.Dlopen(cudaLibrary, purego.RTLD_NOW)
		if err != nil {
			driver.err = fmt.Errorf("cuda driver API: %w", err)
			return
		}
		for _, f := range []struct {
			fn   any
			name string
		}{
			{&driver.getVersion, "cuDriverGetVersion"}, {&driver.errorString, "cuGetErrorString"},
			{&driver.getState, "cuCheckpointProcessGetState"}, {&driver.lock, "cuCheckpointProcessLock"},
			{&driver.checkpoint, "cuCheckpointProcessCheckpoint"}, {&driver.restore, "cuCheckpointProcessRestore"},
			{&driver.unlock, "cuCheckpointProcessUnlock"},
		} {
			sym, err := purego.Dlsym(lib, f.name)
			if err != nil {
				driver.err = fmt.Errorf("cuda driver API: %s: %w", f.name, err)
				return
			}
			purego.RegisterFunc(f.fn, sym)
		}
		// cuDriverGetVersion needs no cuInit; neither do the checkpoint
		// calls, so the agent never creates a context.
		if ret := driver.getVersion(&driver.version); ret != 0 {
			driver.err = fmt.Errorf("cuda driver API: cuDriverGetVersion: CUDA error %d", ret)
			return
		}
		if driver.version < minDriverVersion {
			driver.err = fmt.Errorf("cuda driver API %d.%d: restore arguments without GPU pairs (needs 13.0)",
				driver.version/1000, driver.version%1000/10)
		}
	})
	return driver.err
}

// Driver calls the CUDA checkpoint API of libcuda in this process.
type Driver struct{}

// NewDriver loads libcuda and checks that its restore arguments have a
// layout known here.
func NewDriver() (*Driver, error) {
	if err := loadDriver(); err != nil {
		return nil, err
	}
	return &Driver{}, nil
}

// Name is the backend as the logs show it.
func (*Driver) Name() string {
	return fmt.Sprintf("%s (CUDA %d.%d)", BackendDriver, driver.version/1000, driver.version%1000/10)
}

// call runs a blocking driver call under CallTimeout and ctx. A call that
// overruns keeps its thread until the driver returns; the caller fails the
// restore, which removes the processes the call waits on.
func call(ctx context.Context, what string, pid int, fn func() uint32) error {
	ctx, cancel := context.WithTimeout(ctx, CallTimeout)
	defer cancel()
	done := make(chan uint32, 1)
	go func() {
		runtime.LockOSThread()
		defer runtime.UnlockOSThread()
		done <- fn()
	}()
	select {
	case ret := <-done:
		if ret == 0 {
			return nil
		}
		return fmt.Errorf("%s on %d: %s", what, pid, cuErrorString(ret))
	case <-ctx.Done():
		return fmt.Errorf("%s on %d: %w", what, pid, ctx.Err())
	}
}

func cuErrorString(ret uint32) string {
	var p *byte
	if driver.errorString(ret, &p) != 0 || p == nil {
		return fmt.Sprintf("CUDA error %d", ret)
	}
	n := 0
	for *(*byte)(unsafe.Add(unsafe.Pointer(p), n)) != 0 {
		n++
	}
	return fmt.Sprintf("%s (%d)", strings.Clone(unsafe.String(p, n)), ret)
}

// State is the process's CUDA state: running, locked, checkpointed or failed.
func (*Driver) State(ctx context.Context, pid int) (string, error) {
	var state int32
	if err := call(ctx, "cuCheckpointProcessGetState", pid, func() uint32 {
		return driver.getState(int32(pid), &state)
	}); err != nil {
		return "", err
	}
	if name, ok := processStates[state]; ok {
		return name, nil
	}
	return fmt.Sprintf("state %d", state), nil
}

// Lock blocks the process's CUDA calls, waiting up to timeoutMS for work in flight.
func (*Driver) Lock(ctx context.Context, pid, timeoutMS int) error {
	args := &lockArgs{TimeoutMS: uint32(timeoutMS)}
	return call(ctx, "cuCheckpointProcessLock", pid, func() uint32 { return driver.lock(int32(pid), args) })
}

// Checkpoint moves the locked process's device memory to host memory.
func (*Driver) Checkpoint(ctx context.Context, pid int) error {
	return call(ctx, "cuCheckpointProcessCheckpoint", pid, func() uint32 {
		return driver.checkpoint(int32(pid), &plainArgs{})
	})
}

// Restore puts the checkpointed process's memory back, on the pairs' target GPUs.
func (*Driver) Restore(ctx context.Context, pid int, pairs []Pair) error {
	native := make([]gpuPair, len(pairs))
	for i, p := range pairs {
		var err error
		if native[i].Old, err = uuidBytes(p.Source); err != nil {
			return err
		}
		if native[i].New, err = uuidBytes(p.Target); err != nil {
			return err
		}
	}
	if len(native) == 0 {
		return fmt.Errorf("cuda: restore of %d without a device map", pid)
	}
	// The driver reads the pairs through a pointer inside the arguments:
	// pin them so nothing moves them while the call runs, which may outlive
	// this function when it times out.
	args := &restoreArgs{Pairs: &native[0], Count: uint32(len(native))}
	var pin runtime.Pinner
	pin.Pin(args)
	pin.Pin(&native[0])
	return call(ctx, "cuCheckpointProcessRestore", pid, func() uint32 {
		defer pin.Unpin()
		return driver.restore(int32(pid), args)
	})
}

// Unlock lets the restored process's CUDA calls run again.
func (*Driver) Unlock(ctx context.Context, pid int) error {
	return call(ctx, "cuCheckpointProcessUnlock", pid, func() uint32 {
		return driver.unlock(int32(pid), &plainArgs{})
	})
}
