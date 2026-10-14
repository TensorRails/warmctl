//go:build !linux

package interpose

import "errors"

func probeDevices() ([]Device, error) {
	return nil, errors.New("device probe: GPUs are read on linux only")
}
