//go:build linux

package cuda

import (
	"context"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"unsafe"

	"github.com/ebitengine/purego"
)

// The argument structs must match cuda.h byte for byte: 64 bytes each, the
// pairs pointer at 0 and their count at 8.
func TestLayout(t *testing.T) {
	if unsafe.Sizeof(lockArgs{}) != 64 || unsafe.Sizeof(plainArgs{}) != 64 || unsafe.Sizeof(restoreArgs{}) != 64 {
		t.Fatalf("sizes %d %d %d", unsafe.Sizeof(lockArgs{}), unsafe.Sizeof(plainArgs{}), unsafe.Sizeof(restoreArgs{}))
	}
	if unsafe.Offsetof(restoreArgs{}.Count) != 8 || unsafe.Sizeof(gpuPair{}) != 32 {
		t.Fatalf("count at %d, pair %d bytes", unsafe.Offsetof(restoreArgs{}.Count), unsafe.Sizeof(gpuPair{}))
	}
}

// fakeLibcuda builds testdata/fakecheckpoint.c for a driver version.
func fakeLibcuda(t *testing.T, version string) string {
	t.Helper()
	cc, err := exec.LookPath("cc")
	if err != nil {
		t.Skip("needs a C compiler")
	}
	lib := filepath.Join(t.TempDir(), "libcuda.so.1")
	out, err := exec.Command(cc, "-O2", "-shared", "-fPIC", "-nostdlib", "-ffreestanding", "-fno-stack-protector",
		"-Wl,--no-undefined", "-Wl,-soname,libcuda.so.1", "-DVERSION="+version, "-o", lib,
		"testdata/fakecheckpoint.c").CombinedOutput()
	if err != nil {
		t.Fatalf("%v: %s", err, out)
	}
	return lib
}

func resetDriver(lib string) {
	cudaLibrary = lib
	driver.once = sync.Once{}
	driver.err, driver.version = nil, 0
}

func TestDriverBackend(t *testing.T) {
	lib := fakeLibcuda(t, "13040")
	resetDriver(lib)
	b, why, err := Select(BackendAuto, "/nonexistent/cuda-checkpoint")
	if err != nil || why != "" {
		t.Fatal(err, why)
	}
	if b.Name() != "driver (CUDA 13.4)" {
		t.Fatal(b.Name())
	}
	ctx := context.Background()
	s := NewSession(b, []int{100, 101, 102})
	if err := s.Checkpoint(ctx); err != nil {
		t.Fatal(err)
	}
	pairs := []Pair{
		{"GPU-56ce119a-6848-3ae0-cf4f-8c1564626fc6", "GPU-60189cb1-3991-359c-3824-bff18a3b0fa0"},
		{"GPU-b08d56ca-5cb9-43ff-7615-be0cd3a5d198", "GPU-5a6ee95f-489c-bf29-1330-959d1fb0e8cd"},
	}
	m, err := DeviceMap(pairs)
	if err != nil {
		t.Fatal(err)
	}
	s.Parallel = []int{101, 102}
	s.Concurrent = true
	if err := s.Restore(ctx, m); err != nil {
		t.Fatal(err)
	}
	for _, pid := range []int{100, 101, 102} {
		if st, err := b.State(ctx, pid); err != nil || st != "running" {
			t.Fatalf("%d: %q %v", pid, st, err)
		}
	}
	h, err := purego.Dlopen(lib, purego.RTLD_NOW)
	if err != nil {
		t.Fatal(err)
	}
	var lastTimeout func() uint32
	var lastPairs func(*gpuPair) uint32
	purego.RegisterLibFunc(&lastTimeout, h, "fakeLastTimeout")
	purego.RegisterLibFunc(&lastPairs, h, "fakeLastPairs")
	var native [8]gpuPair
	if n := lastPairs(&native[0]); n != 2 || lastTimeout() != LockTimeoutMS {
		t.Fatalf("%d pairs, timeout %d", n, lastTimeout())
	}
	want, _ := uuidBytes(pairs[1].Target)
	if native[1].New != want || native[0].Old[0] != 0x56 || native[0].Old[15] != 0xc6 {
		t.Fatalf("pairs %x", native[:2])
	}
	// Out of order: the driver's error comes back with its name and code.
	if err := b.Unlock(ctx, 100); err == nil ||
		!strings.Contains(err.Error(), "cuCheckpointProcessUnlock on 100: operation not permitted") {
		t.Fatal(err)
	}
	if err := b.Lock(ctx, 7, 1000); err == nil || !strings.Contains(err.Error(), "named symbol not found (500)") {
		t.Fatal(err)
	}
}

// A driver older than CUDA 13 has no GPU pairs in its restore arguments:
// auto falls back to cuda-checkpoint and says why; driver refuses.
func TestDriverTooOld(t *testing.T) {
	resetDriver(fakeLibcuda(t, "12090"))
	b, why, err := Select(BackendAuto, "/usr/bin/cuda-checkpoint")
	if err != nil || b.Name() != BackendCLI || !strings.Contains(why, "12.9") {
		t.Fatal(b, why, err)
	}
	resetDriver(fakeLibcuda(t, "12090"))
	if _, _, err := Select(BackendDriver, ""); err == nil {
		t.Fatal("driver accepted a 12.9 libcuda")
	}
	resetDriver(filepath.Join(t.TempDir(), "missing.so"))
	if b, why, _ := Select(BackendAuto, "x"); b.Name() != BackendCLI || why == "" {
		t.Fatal(b, why)
	}
}
