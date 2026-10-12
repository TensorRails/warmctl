//go:build unix

package oci

import "golang.org/x/sys/unix"

// freeBytes is the space an unprivileged writer has left on dir's file system.
var freeBytes = func(dir string) (int64, error) {
	var st unix.Statfs_t
	if err := unix.Statfs(dir, &st); err != nil {
		return 0, err
	}
	return int64(st.Bavail) * int64(st.Bsize), nil //nolint:unconvert // field types differ by OS
}
