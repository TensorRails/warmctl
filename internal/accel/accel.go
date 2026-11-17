// Package accel is the one place warmctl's host steps ask which GPU vendor
// they run for. NVIDIA moves GPU state with cuda-checkpoint in the agent,
// over the processes' PIDs, as it always has; AMD moves it inside the
// engine's own processes (internal/rocm). The vendor is never asked of the
// user: a bake takes it from the host, a restore from the snapshot.
package accel

import (
	"context"
	"fmt"

	"github.com/TensorRails/warmctl/v2/internal/host"
	"github.com/TensorRails/warmctl/v2/internal/manifest"
	"github.com/TensorRails/warmctl/v2/internal/rocm"
	"github.com/TensorRails/warmctl/v2/internal/vllm"
)

// Vendors.
const (
	NVIDIA = manifest.VendorNVIDIA
	AMD    = manifest.VendorAMD
)

// Host is this host's GPU vendor: NVIDIA wherever an NVIDIA device exists
// or no AMD one does (so every host warmctl ran on before stays NVIDIA's),
// AMD where /dev/kfd lists GPUs this process may read.
func Host() string {
	if host.NVIDIADevicePresent() || !host.KFDPresent() {
		return NVIDIA
	}
	return AMD
}

// Of is the vendor of a snapshot's GPUs.
func Of(h manifest.Hardware) string { return h.VendorName() }

// GPUState moves GPU state that lives in the engine's processes, through
// the engine. NVIDIA's moves in the agent's cuda step instead: None.
type GPUState interface {
	// Checkpoint runs after sleep and checkpoint_prepare, before the
	// agent's steps.
	Checkpoint(ctx context.Context, e *vllm.Engine) (rocm.Result, error)
	// Restore runs after the agent's steps, before the engine wakes.
	Restore(ctx context.Context, e *vllm.Engine) (rocm.Result, error)
}

// None is NVIDIA's GPUState: nothing to do through the engine.
type None struct{}

// Checkpoint does nothing: the agent's cuda step checkpoints NVIDIA GPUs.
func (None) Checkpoint(context.Context, *vllm.Engine) (rocm.Result, error) { return rocm.Result{}, nil }

// Restore does nothing: the agent's cuda step restores NVIDIA GPUs.
func (None) Restore(context.Context, *vllm.Engine) (rocm.Result, error) { return rocm.Result{}, nil }

// For is the GPUState of a vendor; version is the libkfdreplay release an
// AMD engine must run ("" for any).
func For(vendor, version string) (GPUState, error) {
	switch vendor {
	case "", NVIDIA:
		return None{}, nil
	case AMD:
		return rocm.State{Version: version}, nil
	}
	return nil, fmt.Errorf("unknown GPU vendor %q", vendor)
}

// KFDReplayVersion is the release an AMD bake runs: the pinned one, or any
// for a library named by $WARMCTL_KFDREPLAY.
func KFDReplayVersion(lib manifest.MappedFile) string {
	if lib.Version != "" {
		return lib.Version
	}
	return ""
}
