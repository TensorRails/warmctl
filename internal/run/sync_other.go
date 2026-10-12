//go:build !linux

package run

// SyncFS does nothing outside Linux.
func SyncFS(string) error { return nil }
