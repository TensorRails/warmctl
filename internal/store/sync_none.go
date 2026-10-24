//go:build !unix

package store

// syncFS has no way to flush a filesystem here.
func syncFS(string) error { return nil }
