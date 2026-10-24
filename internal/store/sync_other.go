//go:build unix && !linux

package store

import "golang.org/x/sys/unix"

// syncFS writes every filesystem to disk where syncfs(2) is missing.
func syncFS(string) error {
	unix.Sync()
	return nil
}
