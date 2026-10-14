// Package interpose runs an engine under NVIDIA's cuInterpose, which lets
// cross-process CUDA memory (legacy IPC, VMM exports, multicast) survive a
// CUDA checkpoint. See docs/interpose.md.
package interpose

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

// Modes of a profile's ipc.
const (
	Native    = "native"
	Interpose = "interpose"
	Auto      = "auto"
)

// Files of a cuInterpose build.
const (
	Frontend    = "libcuinterpose.so"
	Core        = "libcuinterpose_core.so"
	Launcher    = "cuinterpose-launch"
	Coordinator = "cuinterpose-coordinator"
	// State is the coordinator's record of a prepared checkpoint.
	State = "cuinterpose.state"
)

// Files are what a bake takes from a cuInterpose build.
var Files = []string{Frontend, Core, Launcher, Coordinator}

// Container paths: the build (and, in a receiver, the snapshot's state) is
// mounted at GuestDir in donor and receiver alike, because the restored
// processes have the libraries mapped from there. Every interposed process
// listens on a control socket in SocketDir, a tree the snapshot carries.
const (
	GuestDir  = "/opt/warmctl/ipc"
	SocketDir = "/tmp"
)

// DefaultDir is cuinterpose/ next to this executable, where a release puts
// the build.
func DefaultDir() string {
	self, err := os.Executable()
	if err != nil {
		return "cuinterpose"
	}
	return filepath.Join(filepath.Dir(self), "cuinterpose")
}

// Check reports the first file a build directory lacks.
func Check(dir string) error {
	for _, name := range Files {
		if info, err := os.Stat(filepath.Join(dir, name)); err != nil || !info.Mode().IsRegular() {
			return fmt.Errorf("cuInterpose build %s lacks %s: run make cuinterpose or pass --cuinterpose", dir, name)
		}
	}
	return nil
}

// Wrap runs argv under the launcher, which checks the libraries' digests
// and preloads the frontend.
func Wrap(argv []string) []string {
	return append([]string{GuestDir + "/" + Launcher, "--library", GuestDir + "/" + Frontend, "--"}, argv...)
}

// Socket is where the interposed process pid listens for the coordinator.
func Socket(pid int) string { return SocketDir + "/cuinterpose-" + strconv.Itoa(pid) + ".sock" }

// Participants are the processes among pids that listen for the
// coordinator. A process opens its socket on its first intercepted CUDA
// call; one that never made such a call (an API server that only loaded
// the driver) holds no shared memory and takes the plain CUDA checkpoint.
func Participants(pids []int, exists func(path string) bool) []int {
	var out []int
	for _, pid := range pids {
		if exists(Socket(pid)) {
			out = append(out, pid)
		}
	}
	return out
}

// Command is the coordinator's command line for action (prepare or
// restore) over the interposed processes pids, keeping its state in dir.
func Command(coordinator, action, dir string, pids []int) []string {
	argv := []string{coordinator, "--" + action, "--checkpoint-dir", dir, "--socket-dir", SocketDir}
	for _, pid := range pids {
		argv = append(argv, "--process", strconv.Itoa(pid))
	}
	return argv
}

// Supported is nil when every GPU can hold an interposed engine: the
// interposer backs every allocation with VMM memory it can export as a
// POSIX file descriptor.
func Supported(devices []Device) error {
	if len(devices) == 0 {
		return errors.New("no visible GPU")
	}
	var lacking []string
	for _, d := range devices {
		if !d.VMM || !d.PosixFD {
			lacking = append(lacking, fmt.Sprintf("GPU %d (%s)", d.Index, d.Name))
		}
	}
	if len(lacking) > 0 {
		return fmt.Errorf("%s lack VMM with POSIX handles", strings.Join(lacking, ", "))
	}
	return nil
}
