//go:build linux

package restore

import (
	"fmt"

	"golang.org/x/sys/unix"
)

func statTree(path string) (fsInfo, error) {
	var s unix.Statfs_t
	if err := unix.Statfs(path, &s); err != nil {
		return fsInfo{}, err
	}
	return fsInfo{Free: int64(s.Bavail) * s.Bsize, ReadOnly: s.Flags&unix.ST_RDONLY != 0}, nil
}

func writable(path string) error { return unix.Access(path, unix.W_OK) }

// bindMount mounts src at dst in this container's mount namespace, which
// the agent, CRIU and the restored tree share.
func bindMount(src, dst string, readOnly bool) error {
	if err := unix.Mount(src, dst, "", unix.MS_BIND, ""); err != nil {
		return err
	}
	if !readOnly {
		return nil
	}
	if err := unix.Mount("", dst, "", unix.MS_BIND|unix.MS_REMOUNT|unix.MS_RDONLY, ""); err != nil {
		_ = unix.Unmount(dst, unix.MNT_DETACH)
		return err
	}
	return nil
}

func unmount(dst string) error { return unix.Unmount(dst, unix.MNT_DETACH) }

// mountTmpfs mounts a fresh tmpfs of size bytes at dst, as Docker mounts a
// container's /tmp (exec, for JIT compilers) and /dev/shm.
func mountTmpfs(dst string, size int64, exec bool) error {
	flags := uintptr(unix.MS_NOSUID | unix.MS_NODEV)
	if !exec {
		flags |= unix.MS_NOEXEC
	}
	return unix.Mount("tmpfs", dst, "tmpfs", flags, fmt.Sprintf("mode=1777,size=%d", size))
}
