//go:build unix

package oci

import (
	"os"

	"golang.org/x/sys/unix"
)

// lockFile takes f for this process, failing when another holds it; the
// lock goes with the file's descriptor.
func lockFile(f *os.File) error { return unix.Flock(int(f.Fd()), unix.LOCK_EX|unix.LOCK_NB) }
