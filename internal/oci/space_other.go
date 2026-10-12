//go:build !unix

package oci

import "errors"

var freeBytes = func(string) (int64, error) { return 0, errors.New("free space unknown on this OS") }
