package run

import (
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
)

// ContainerLock is held by the in-place run that owns this container's
// trees: /tmp, /dev/shm and /workspace are its engine's, so a container
// runs one in-place run at a time.
const ContainerLock = "/run/warmctl.lock"

// Lock is an exclusive flock held for the life of a run.
type Lock struct{ f *os.File }

// LockContainer takes ContainerLock without waiting, or names the process
// that holds it.
func LockContainer() (*Lock, error) { return lockFile(ContainerLock) }

func lockFile(path string) (*Lock, error) {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return nil, err
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_RDWR, 0o644)
	if err != nil {
		return nil, err
	}
	if err := syscall.Flock(int(f.Fd()), syscall.LOCK_EX|syscall.LOCK_NB); err != nil {
		holder, _ := os.ReadFile(path)
		f.Close()
		return nil, fmt.Errorf("warmctl process %s runs in place in this container (%s): its engine has the "+
			"container's /tmp, /dev/shm and /workspace, so stop it first", strings.TrimSpace(string(holder)), path)
	}
	if err := f.Truncate(0); err == nil {
		_, _ = f.WriteAt([]byte(strconv.Itoa(os.Getpid())+"\n"), 0)
	}
	return &Lock{f}, nil
}

// Release drops the lock; nil has none to drop.
func (l *Lock) Release() {
	if l == nil || l.f == nil {
		return
	}
	_ = syscall.Flock(int(l.f.Fd()), syscall.LOCK_UN)
	l.f.Close()
	l.f = nil
}
