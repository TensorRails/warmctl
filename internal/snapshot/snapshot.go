// Package snapshot bakes a snapshot: it starts the engine in a donor
// container, warms it up, records a canary, exports the weights pool, puts
// the engine to sleep, checkpoints CUDA, dumps the process tree with CRIU,
// archives the trees it wrote and publishes everything with one manifest.
package snapshot

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/TensorRails/warmctl/v2"
	"github.com/TensorRails/warmctl/v2/internal/agent"
	"github.com/TensorRails/warmctl/v2/internal/dmtcp"
	"github.com/TensorRails/warmctl/v2/internal/docker"
	"github.com/TensorRails/warmctl/v2/internal/host"
	"github.com/TensorRails/warmctl/v2/internal/manifest"
	"github.com/TensorRails/warmctl/v2/internal/profile"
	"github.com/TensorRails/warmctl/v2/internal/run"
	"github.com/TensorRails/warmctl/v2/internal/vllm"
)

// Options of one bake.
type Options struct {
	Profile        string
	ProfileDirs    []string
	ModelDir       string
	GPUs           string
	Output         string
	ID             string
	StateDir       string
	DockerSocket   string
	CRIU           string
	CUDACheckpoint string
	// CUDABackend is the agent's cuda.Select mode: auto, driver or cli.
	CUDABackend string
	// Cuinterpose is a cuInterpose build directory, for profiles whose ipc
	// is not native; default: cuinterpose/ next to this executable.
	Cuinterpose  string
	Set          []string
	NoVerify     bool
	PartBytes    int64
	Threads      int
	ReadyTimeout time.Duration
	// CPUTest runs without GPU devices or the io_uring check: the
	// end-to-end test on a CPU host with a stand-in engine.
	CPUTest  bool
	Version  string
	Progress *run.Progress
	// InPlace bakes in this container, without Docker: the engine runs as a
	// child of the agent, itself a child of this process, and sees real
	// paths (docs/dmtcp.md).
	InPlace bool
	// Backend snapshots the process tree in place: auto, criu or dmtcp. A
	// donor container always runs CRIU.
	Backend string
	// Compress stores the process images, runtime trees and weights in Output
	// as zstd frames, for a snapshot restored from network storage: about
	// 2.8x fewer bytes of images and 1.45x of BF16 weights to read, decoded
	// in parallel as up stages its copy.
	Compress bool
	DMTCP    dmtcp.Tools
}

type bake struct {
	o      Options
	p      *run.Progress
	loader profile.Loader
	table  profile.HardwareTable
	launch profile.Launch
	// ipcDir holds the cuInterpose build the donor mounts, when ipc is not
	// native; ipc is what the donor decided.
	ipcDir string
	ipc    agent.IPCResult
	row    profile.Hardware
	gpus   []host.GPU
	lease  *host.Lease
	// owns is an in-place bake's hold on the container (run.LockContainer).
	owns     *run.Lock
	topology string
	topoSHA  string
	dk       *docker.Client
	dir      string
	cfg      agent.Config
	adapter  string
	model    string
	served   string
	image    docker.Image
	c        *run.Container
	workers  []vllm.Worker
	canary   vllm.Canary
	exports  []vllm.Export
	// driverPaths are the host driver libraries the dump maps, copied into
	// the snapshot; nil when they could not be copied.
	driverPaths []string
	prepared    []vllm.Prepared
	cudaRes     agent.CUDAResult
	frozen      agent.FreezeResult
	dumped      agent.DumpResult
	manifest    manifest.Manifest
	// backend snapshots the process tree: agent.BackendCRIU or BackendDMTCP.
	backend string
}

// Run bakes and returns the published manifest.
func Run(ctx context.Context, o Options) (manifest.Manifest, error) {
	if err := run.CheckID(o.ID); err != nil {
		return manifest.Manifest{}, err
	}
	b := &bake{o: o, p: o.Progress}
	defer b.cleanup()
	steps := []struct {
		name string
		fn   func(context.Context) error
	}{
		{"resolve profile", b.resolve}, {"preflight", b.preflight}, {"prepare", b.prepare},
		{"start donor", b.start}, {"warm up", b.warm}, {"export weights", b.export},
		{"sleep", b.sleep}, {"checkpoint", b.checkpoint}, {"publish", b.publish},
	}
	for _, s := range steps {
		if err := s.fn(ctx); err != nil {
			b.diagnose()
			return manifest.Manifest{}, fmt.Errorf("%s: %w", s.name, err)
		}
	}
	return b.manifest, nil
}

func (b *bake) resolve(context.Context) error {
	if err := b.chooseBackend(); err != nil {
		return err
	}
	b.loader = profile.Loader{Assets: warmctl.Assets, Dirs: b.o.ProfileDirs}
	defaults, err := b.loader.Defaults()
	if err != nil {
		return err
	}
	p, err := b.loader.Load(b.o.Profile)
	if err != nil {
		return err
	}
	for _, s := range b.o.Set {
		if err := p.Set(s); err != nil {
			return err
		}
	}
	if b.o.NoVerify {
		p.Canary.Mode = "warn"
	}
	if b.launch, err = profile.Resolve(defaults, p); err != nil {
		return err
	}
	b.table, err = b.loader.Hardware()
	b.p.Printf("profile %s: %d ranks, %d graph sizes, %d warmup cases", b.launch.Profile.ID,
		b.launch.Profile.Parallel.Ranks(), len(b.launch.CaptureSizes), len(b.launch.WarmupCases()))
	return err
}

func (b *bake) preflight(ctx context.Context) error {
	if disabled, err := host.IOUringDisabled(); !b.o.CPUTest && (err != nil || disabled != 2) {
		if b.backend != agent.BackendDMTCP {
			return errors.New("kernel.io_uring_disabled must be 2 (sysctl -w kernel.io_uring_disabled=2): " +
				"CRIU cannot dump io_uring")
		}
		// A container that may not run CRIU may not set the sysctl either.
		b.p.Printf("warning: kernel.io_uring_disabled is %d, not 2 (%v): the engine runs with UV_USE_IO_URING=0, "+
			"and a process that sets up io_uring anyway will not restart", disabled, err)
	}
	all, err := host.Inventory(ctx)
	if err != nil {
		return err
	}
	if b.gpus, err = host.Select(all, b.o.GPUs); err != nil {
		return err
	}
	if b.row, err = Hardware(b.table, b.gpus, b.launch.Profile.Parallel.Ranks()); err != nil {
		return err
	}
	if !b.launch.Profile.Supports(b.row) {
		b.p.Printf("warning: profile %s does not list %s; running anyway", b.launch.Profile.ID, b.row.ID)
	}
	busy, err := host.Busy(ctx, b.gpus)
	if err != nil {
		return err
	}
	if busy > 0 {
		return fmt.Errorf("the GPUs run %d other processes", busy)
	}
	if b.lease, err = host.Acquire(filepath.Join(b.o.StateDir, "locks"), b.gpus); err != nil {
		return err
	}
	if b.topology, b.topoSHA, err = host.Topology(ctx, b.gpus); err != nil {
		return err
	}
	if b.o.InPlace {
		return b.inPlaceImage()
	}
	if b.dk, err = docker.New(ctx, b.o.DockerSocket); err != nil {
		return err
	}
	return b.dk.CheckInit(ctx)
}

// inPlaceImage names the image an in-place engine runs in: the runtime
// image's base, which a receiver pod compares with its own. Without the
// runtime image nothing names it, and the manifest keeps the profile's.
func (b *bake) inPlaceImage() error {
	rt, ok, err := run.ReadRuntime(run.RuntimeFile)
	if err != nil {
		return err
	}
	if !ok {
		b.p.Printf("warning: no %s: nothing names this container's image; the manifest records the profile's, %s",
			run.RuntimeFile, b.launch.Profile.Image)
		return nil
	}
	b.launch.Profile.Image = rt.Base
	return nil
}

// chooseBackend decides how the tree is dumped (bakeBackend).
func (b *bake) chooseBackend() error {
	backend, err := bakeBackend(b.o.Backend, b.o.InPlace)
	if err != nil {
		return err
	}
	b.backend = backend
	if backend == agent.BackendDMTCP {
		b.p.Printf("process snapshot backend: dmtcp (in place)")
		return b.o.DMTCP.Check()
	}
	return nil
}

// bakeBackend is the backend of a bake asked for want (--backend): CRIU in
// the donor container warmctl makes, DMTCP in place. An in-place CRIU bake
// is not supported yet: the bake links the model in (run.Place), so CRIU
// would record files under the link's target, while an in-place CRIU
// restore mounts the model at its mount path.
func bakeBackend(want string, inPlace bool) (string, error) {
	switch want {
	case "", host.BackendAuto:
		if inPlace {
			return agent.BackendDMTCP, nil
		}
		return agent.BackendCRIU, nil
	case host.BackendCRIU:
		if inPlace {
			return "", errors.New("backend criu: an in-place bake runs DMTCP, CRIU in place is not supported yet; " +
				"bake through Docker (without --in-place)")
		}
		return agent.BackendCRIU, nil
	case host.BackendDMTCP:
		if !inPlace {
			return "", errors.New("backend dmtcp: only an in-place bake (--in-place) runs DMTCP")
		}
		return agent.BackendDMTCP, nil
	}
	return "", fmt.Errorf("unknown backend %q: auto, criu or dmtcp", want)
}

// Hardware finds the hardware row of the selected GPUs: all of one model,
// as many as the profile has ranks.
func Hardware(table profile.HardwareTable, gpus []host.GPU, ranks int) (profile.Hardware, error) {
	if len(gpus) != ranks {
		return profile.Hardware{}, fmt.Errorf("the profile needs %d GPUs, %d selected", ranks, len(gpus))
	}
	row, ok := table.ByName(gpus[0].Name)
	if !ok {
		return row, fmt.Errorf("GPU %q has no row in hardware/hardware.yaml", gpus[0].Name)
	}
	for _, g := range gpus {
		if g.Name != gpus[0].Name {
			return row, fmt.Errorf("mixed GPU models: %s and %s", gpus[0].Name, g.Name)
		}
	}
	return row, nil
}

// diagnose prints the tails of the engine and agent logs after a failure.
func (b *bake) diagnose() {
	if b.c == nil {
		return
	}
	if tail := b.c.EngineLog(30); tail != "" {
		b.p.Printf("engine log tail:\n%s", tail)
	}
	if tail := b.c.Tail(30); tail != "" {
		b.p.Printf("agent log tail:\n%s", tail)
	}
}

// cleanup removes the donor and the bake directory, keeping the logs under
// <state>/logs/<id>, and releases the GPUs.
func (b *bake) cleanup() {
	logs := filepath.Join(b.o.StateDir, "logs", b.o.ID)
	if b.c != nil {
		_ = os.MkdirAll(logs, 0o755)
		b.c.SaveLogs(filepath.Join(logs, "agent.log"))
		b.c.Remove()
	}
	if b.dir != "" {
		keepLogs(b.dir, logs)
		_ = os.RemoveAll(b.dir)
	}
	if b.lease != nil {
		b.lease.Release()
	}
	b.owns.Release()
}

func keepLogs(dir, logs string) {
	_ = os.MkdirAll(logs, 0o755)
	for _, name := range []string{"workspace/logs/engine.log", "io/out/criu-work/dump.log", "work/agent.json"} {
		_ = run.CopyFile(filepath.Join(dir, name), filepath.Join(logs, filepath.Base(name)), 0o644)
	}
}
