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
}

type bake struct {
	o      Options
	p      *run.Progress
	loader profile.Loader
	table  profile.HardwareTable
	launch profile.Launch
	// ipcDir holds the cuInterpose build the donor mounts, when ipc is not
	// native; ipc is what the donor decided.
	ipcDir   string
	ipc      agent.IPCResult
	row      profile.Hardware
	gpus     []host.GPU
	lease    *host.Lease
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
		return errors.New("kernel.io_uring_disabled must be 2 (sysctl -w kernel.io_uring_disabled=2): " +
			"CRIU cannot dump io_uring")
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
	if busy, err := host.Busy(ctx, b.gpus); err != nil || busy > 0 {
		return fmt.Errorf("the GPUs run %d other processes (%v)", busy, err)
	}
	if b.lease, err = host.Acquire(filepath.Join(b.o.StateDir, "locks"), b.gpus); err != nil {
		return err
	}
	if b.topology, b.topoSHA, err = host.Topology(ctx, b.gpus); err != nil {
		return err
	}
	if b.dk, err = docker.New(ctx, b.o.DockerSocket); err != nil {
		return err
	}
	return b.dk.CheckInit(ctx)
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
}

func keepLogs(dir, logs string) {
	_ = os.MkdirAll(logs, 0o755)
	for _, name := range []string{"workspace/logs/engine.log", "io/out/criu-work/dump.log", "work/agent.json"} {
		_ = run.CopyFile(filepath.Join(dir, name), filepath.Join(logs, filepath.Base(name)), 0o644)
	}
}
