//go:build linux

package oci

import (
	"errors"
	"os"

	"golang.org/x/sys/unix"
)

// preallocate reserves a file's blocks before pieces land in any order, so
// the file is not fragmented; filesystems without fallocate get a sparse
// file.
func preallocate(f *os.File, size int64) error {
	err := unix.Fallocate(int(f.Fd()), 0, 0, size)
	if errors.Is(err, unix.EOPNOTSUPP) || errors.Is(err, unix.ENOSYS) {
		return f.Truncate(size)
	}
	return err
}
