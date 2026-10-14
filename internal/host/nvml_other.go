//go:build !linux

package host

import "errors"

func openNVML() (Session, error) {
	return nil, errors.New("nvml: GPUs are read on linux only")
}
