//go:build !linux

package oci

import "os"

func preallocate(f *os.File, size int64) error { return f.Truncate(size) }
