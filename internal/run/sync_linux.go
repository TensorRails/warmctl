//go:build linux

package run

import (
	"os"

	"golang.org/x/sys/unix"
)

// SyncFS flushes the filesystem holding dir. A bake ends with it so the
// gigabytes it just wrote are on disk before the next container starts:
// otherwise `docker create` waits for that writeback (v1 P138: 40 s).
func SyncFS(dir string) error {
	f, err := os.Open(dir)
	if err != nil {
		return err
	}
	defer f.Close()
	return unix.Syncfs(int(f.Fd()))
}
