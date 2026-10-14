package interpose

import (
	"context"
	"fmt"
	"time"
)

// Device is one visible GPU as the CUDA driver describes it.
type Device struct {
	Index             int
	Name              string
	ComputeCapability string
	// VMM, PosixFD and Multicast are the driver attributes the interposer
	// and fast collectives need: virtual memory management, its POSIX file
	// descriptor handles, and multicast objects.
	VMM       bool
	PosixFD   bool
	Multicast bool
}

// Probe reads the visible GPUs from the CUDA driver; tests replace it.
var Probe = probeDevices

// ProbeTimeout bounds one probe: cuInit can block for minutes on a GPU that
// fell off the bus, and the caller should hear about it.
const ProbeTimeout = 30 * time.Second

// Devices runs the probe under ProbeTimeout. On timeout it returns and
// leaves the blocked call behind.
func Devices(ctx context.Context) ([]Device, error) {
	ctx, cancel := context.WithTimeout(ctx, ProbeTimeout)
	defer cancel()
	type result struct {
		devices []Device
		err     error
	}
	done := make(chan result, 1)
	probe := Probe // a call left behind must not read Probe while a test swaps it
	go func() {
		devices, err := probe()
		done <- result{devices, err}
	}()
	select {
	case r := <-done:
		return r.devices, r.err
	case <-ctx.Done():
		return nil, fmt.Errorf("device probe: %w", ctx.Err())
	}
}
