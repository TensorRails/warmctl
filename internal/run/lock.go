package run

import (
	"errors"
	"fmt"
	"os"
	"syscall"
)

// ContainerLock is the directory an in-place run locks: the container's
// root, which every container has, read-only or not, and no mount of a
// tree hides. The run owns the container's /tmp, /dev/shm and /workspace,
// so a container runs one in-place run at a time.
const ContainerLock = "/"

// Lock is an exclusive flock held for the life of a run.
type Lock struct{ f *os.File }

// LockContainer takes ContainerLock without waiting.
func LockContainer() (*Lock, error) { return lockDir(ContainerLock) }

func lockDir(path string) (*Lock, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	if err := syscall.Flock(int(f.Fd()), syscall.LOCK_EX|syscall.LOCK_NB); err != nil {
		f.Close()
		if errors.Is(err, syscall.EWOULDBLOCK) {
			return nil, errors.New("another warmctl run is in place in this container (up, or build --in-place): " +
				"its engine has the container's /tmp, /dev/shm and /workspace, so stop it first")
		}
		return nil, fmt.Errorf("lock %s: %w", path, err)
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
