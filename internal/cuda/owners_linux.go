//go:build linux

// Package cuda finds the processes of a tree that hold CUDA state and moves
// that state with NVIDIA's cuda-checkpoint: lock, checkpoint to host memory
// before the CRIU dump; restore onto the receiver's GPUs and unlock after.
package cuda

import (
	"fmt"
	"os"
	"sort"
	"strings"
)

// Owners are the processes among pids that hold CUDA state: they map
// libcuda and keep a /dev/nvidia* descriptor open. A vLLM engine at TP8 has
// ten: eight workers, the engine core and the API server.
func Owners(pids []int) ([]int, error) {
	var out []int
	for _, pid := range pids {
		maps, err := os.ReadFile(fmt.Sprintf("/proc/%d/maps", pid))
		if os.IsNotExist(err) {
			continue
		}
		if err != nil {
			return nil, err
		}
		if !strings.Contains(string(maps), "libcuda.so") {
			continue
		}
		if holdsDevice(pid) {
			out = append(out, pid)
		}
	}
	sort.Ints(out)
	return out, nil
}

func holdsDevice(pid int) bool {
	dir := fmt.Sprintf("/proc/%d/fd", pid)
	fds, err := os.ReadDir(dir)
	if err != nil {
		return false
	}
	for _, fd := range fds {
		target, err := os.Readlink(dir + "/" + fd.Name())
		if err == nil && strings.HasPrefix(target, "/dev/nvidia") {
			return true
		}
	}
	return false
}
