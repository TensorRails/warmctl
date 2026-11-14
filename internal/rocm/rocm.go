// Package rocm is warmctl on AMD Instinct GPUs: what an AMD host must offer,
// how a profile runs there, when a snapshot restores on it, and the GPU state
// an AMD engine moves itself.
//
// There is no cuda-checkpoint for AMD. The engine's processes preload
// libkfdreplay (internal/kfdreplay), which records every GPU object the ROCm
// runtime creates; asked through the engine (warmctl_gpu_checkpoint), each
// worker dumps its GPU memory, parks the runtime's threads and releases the
// devices, so that DMTCP snapshots a CPU-only process tree. After the
// restart each worker replays its objects at the same addresses
// (warmctl_gpu_restore) before the engine wakes. Every other step of a
// snapshot, its files and its distribution are the NVIDIA path's.
package rocm

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/TensorRails/warmctl/v2/internal/host"
	"github.com/TensorRails/warmctl/v2/internal/manifest"
	"github.com/TensorRails/warmctl/v2/internal/profile"
	"github.com/TensorRails/warmctl/v2/internal/vllm"
)

// Support is one platform an AMD snapshot is known to restore on: the GPU
// ISA, the amdgpu driver releases and the libkfdreplay release it was
// verified with (docs/amd.md).
type Support struct {
	Arch      string
	Driver    string // a prefix of /sys/module/amdgpu/version
	KFDReplay string
}

// Matrix is what has been verified; a GPU of another ISA is refused, since
// libkfdreplay mirrors what the runtime does on the GPUs it was tested on.
var Matrix = []Support{
	{Arch: "gfx950", Driver: "7.1.3", KFDReplay: "0.1.0"},
}

// Supported refuses GPUs no row of Matrix covers and warns (the returned
// note) when the driver is not one a row names.
func Supported(gpus []host.GPU) (note string, err error) {
	if len(gpus) == 0 {
		return "", errors.New("amd: no GPU")
	}
	var archs []string
	for _, s := range Matrix {
		archs = append(archs, s.Arch)
		if s.Arch != gpus[0].Arch {
			continue
		}
		if strings.HasPrefix(gpus[0].Driver, s.Driver) {
			return "", nil
		}
		return fmt.Sprintf("amdgpu %s: AMD snapshots were verified on %s.x; libkfdreplay refuses what it does not know",
			gpus[0].Driver, s.Driver), nil
	}
	return "", fmt.Errorf("%s (%s) is not supported yet: AMD snapshots run on %s", gpus[0].Name, gpus[0].Arch,
		strings.Join(archs, ", "))
}

// RequiredEnv is what every AMD engine runs with (profiles/_defaults-rocm.yaml);
// a profile that changes one of them cannot be checkpointed.
var RequiredEnv = map[string]string{
	"HSA_ENABLE_SDMA":                  "0",
	"HSA_ENABLE_SCRATCH_ASYNC_RECLAIM": "0",
	"HSA_NO_SCRATCH_RECLAIM":           "1",
	"PYTORCH_HIP_ALLOC_CONF":           "expandable_segments:False",
	"USE_LIBUV":                        "0",
}

// CheckLaunch refuses what an AMD engine cannot checkpoint: more than one
// GPU (no cross-process GPU memory restores yet), cuInterpose, and an
// environment without RequiredEnv.
func CheckLaunch(l profile.Launch) error {
	p := l.Profile
	if r := p.Parallel.Ranks(); r != 1 {
		return fmt.Errorf("profile %s: %d GPUs per engine; AMD snapshots run one GPU per engine (tp: 1)", p.ID, r)
	}
	if mode := p.IPCMode(); mode == "interpose" {
		return fmt.Errorf("profile %s: ipc %s is CUDA's; AMD engines run with ipc: native", p.ID, mode)
	}
	var bad []string
	for k, v := range RequiredEnv {
		if l.Env[k] != v {
			bad = append(bad, fmt.Sprintf("%s=%s (has %q)", k, v, l.Env[k]))
		}
	}
	if len(bad) > 0 {
		return fmt.Errorf("profile %s: an AMD engine needs %s", p.ID, strings.Join(bad, ", "))
	}
	return nil
}

// Hardware is what an AMD snapshot records of the donor's GPUs beyond the
// common fields: ISA, the kernel's gpu_id and the render node per GPU, and
// the runtime the workers reported.
func Hardware(h *manifest.Hardware, gpus []host.GPU, w vllm.Worker, lib manifest.MappedFile) {
	h.Vendor = manifest.VendorAMD
	h.ROCm = &manifest.ROCm{HIP: w.HIP, Runtime: w.ROCm}
	h.GPUCheckpoint = "libkfdreplay " + w.KFDReplay + " sha256:" + lib.SHA256
	for i := range h.GPUs {
		if i < len(gpus) {
			h.GPUs[i].Arch, h.GPUs[i].KFDGPUID, h.GPUs[i].Device = gpus[i].Arch, gpus[i].KFDGPUID, gpus[i].RenderNode
		}
	}
}

// Compatible checks this host's GPUs against an AMD snapshot's: the same
// model, memory and ISA, and the very same driver: libkfdreplay replays the
// donor's driver calls, and a restore across driver releases is unverified.
// Another GPU of the same model is fine: libkfdreplay translates its gpu_id.
func Compatible(h manifest.Hardware, gpus []host.GPU) error {
	if h.VendorName() != manifest.VendorAMD {
		return fmt.Errorf("the snapshot was taken on %s GPUs", h.VendorName())
	}
	if len(gpus) != len(h.GPUs) {
		return fmt.Errorf("the snapshot has %d ranks, %d GPUs selected", len(h.GPUs), len(gpus))
	}
	for i, g := range gpus {
		want := h.GPUs[i]
		if g.Vendor != host.VendorAMD {
			return fmt.Errorf("GPU %d is not an AMD GPU", g.Index)
		}
		if g.Name != want.Name || g.MemoryMiB != want.MemoryMiB || g.Arch != want.Arch {
			return fmt.Errorf("GPU %d is %s (%s) with %d MiB, the snapshot needs %s (%s) with %d MiB",
				g.Index, g.Name, g.Arch, g.MemoryMiB, want.Name, want.Arch, want.MemoryMiB)
		}
		if g.Driver != h.Driver {
			return fmt.Errorf("amdgpu %s, the snapshot was taken on %s; AMD snapshots restore under the same driver",
				g.Driver, h.Driver)
		}
	}
	return nil
}

// RenderNodes checks that libkfdreplay can find the GPU a restore runs on:
// it reopens the donor's render node, and on another GPU translates the
// gpu_id only when the container sees exactly one GPU.
func RenderNodes(h manifest.Hardware, gpus []host.GPU, visible int) error {
	for i, g := range gpus {
		if i >= len(h.GPUs) {
			break
		}
		if g.KFDGPUID != h.GPUs[i].KFDGPUID && visible != 1 {
			return fmt.Errorf("GPU %d is another GPU than the donor's (gpu_id %d, the snapshot's %d) and this "+
				"container sees %d GPUs; libkfdreplay moves a process to another GPU only when one is visible: "+
				"restore in a container with that GPU alone", g.Index, g.KFDGPUID, h.GPUs[i].KFDGPUID, visible)
		}
	}
	return nil
}

// State moves an AMD engine's GPU state through the engine: every worker
// calls libkfdreplay itself. Version is the libkfdreplay release the
// workers must run, "" for a research build.
type State struct {
	Version string
	Logf    func(string, ...any)
}

// Result is what one checkpoint or restore of every rank took.
type Result struct {
	Ranks   []vllm.GPUState
	Seconds float64
}

// MaxWptr is the most packets any rank's queue has taken.
func (r Result) MaxWptr() uint64 {
	var m uint64
	for _, g := range r.Ranks {
		m = max(m, g.MaxWptr)
	}
	return m
}

// Checkpoint runs after sleep and checkpoint_prepare, before the agent's
// steps: afterwards no worker holds a GPU device and DMTCP may dump.
func (s State) Checkpoint(ctx context.Context, e *vllm.Engine) (Result, error) {
	start := time.Now()
	rows, err := e.GPUCheckpoint(ctx, s.Version)
	if err != nil {
		return Result{}, fmt.Errorf("GPU checkpoint (libkfdreplay): %w", err)
	}
	return s.check(rows, "checkpointed_cold", start)
}

// Restore runs after the process restore, before the engine wakes.
func (s State) Restore(ctx context.Context, e *vllm.Engine) (Result, error) {
	start := time.Now()
	rows, err := e.GPURestore(ctx, s.Version)
	if err != nil {
		return Result{}, fmt.Errorf("GPU restore (libkfdreplay): %w", err)
	}
	return s.check(rows, "restored", start)
}

func (s State) check(rows []vllm.GPUState, want string, start time.Time) (Result, error) {
	for _, r := range rows {
		if r.State != want {
			return Result{}, fmt.Errorf("rank %d: libkfdreplay state %q, want %q", r.Rank, r.State, want)
		}
		if s.Version != "" && r.Version != s.Version {
			return Result{}, fmt.Errorf("rank %d runs libkfdreplay %s, want %s", r.Rank, r.Version, s.Version)
		}
	}
	return Result{Ranks: rows, Seconds: time.Since(start).Seconds()}, nil
}

// MaxWptrWarn is how many queue packets make a restore's replay take
// seconds (~1 µs each): the snapshot should be of a freshly warmed engine.
const MaxWptrWarn = 10_000_000
