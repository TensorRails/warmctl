// Package restore brings a snapshot back to serving: a receiver container
// installs the archived trees, CRIU restores the process tree stopped, the
// agent thaws it and puts its CUDA state on this host's GPUs, the engine
// reloads its weights pool and checks it, and the host gateway opens.
package restore

import (
	"context"
	"errors"
	"fmt"
	"net"
	"os"
	"path/filepath"
	"time"

	"github.com/TensorRails/warmctl/v2/internal/docker"
	"github.com/TensorRails/warmctl/v2/internal/gateway"
	"github.com/TensorRails/warmctl/v2/internal/host"
	"github.com/TensorRails/warmctl/v2/internal/manifest"
	"github.com/TensorRails/warmctl/v2/internal/run"
)

// Options of one restore.
type Options struct {
	Snapshot       string
	ID             string
	GPUs           string
	Listen         string
	StateDir       string
	DockerSocket   string
	CUDAExecution  string
	Canary         string
	ModelDir       string
	CRIU           string
	CUDACheckpoint string
	NoVerify       bool
	VerifyWeights  bool
	Threads        int
	CPUTest        bool
	// Serve keeps the engine serving until ctx ends; otherwise Run returns
	// once serving started and leaves the receiver running.
	Serve bool
	// InPlace restores in this container, which Kubernetes made as the
	// receiver (docs/kubernetes.md): the agent runs as this process's
	// child instead of in a Docker container.
	InPlace bool
	// RunDir, in place, holds the agent's config, sockets and CRIU work.
	RunDir string
	// Gate, in place, is the gateway, opened before the pull.
	Gate *Gate
	// Arrival, when set, is the pull still filling Snapshot: the restore
	// starts on the files it reads first and waits for the weights before
	// the engine wakes, not for the pull to end.
	Arrival  Arrival
	Progress *run.Progress
	// TLSCert and TLSKey are --tls-cert and --tls-key; TLS is the pair
	// loaded from them, nil for plain HTTP.
	TLSCert string
	TLSKey  string
	TLS     *gateway.TLS
}

// Arrival is a snapshot still being pulled: its manifest is known, and its
// files appear in its directory as they are checked, manifest.json last.
type Arrival interface {
	Manifest() manifest.Manifest
	// ManifestJSON is the bytes manifest.json will hold.
	ManifestJSON() []byte
	// Wait returns once every file of paths is in place, or with the error
	// that ended the pull.
	Wait(ctx context.Context, paths []string) error
}

type restore struct {
	o        Options
	p        *run.Progress
	m        manifest.Manifest
	snapshot string
	gpus     []host.GPU
	lease    *host.Lease
	dk       *docker.Client
	dir      string
	image    docker.Image
	c        *run.Container
	listener net.Listener
	gw       *gateway.Gateway
	prefetch context.CancelFunc
	pf       *prefetchStats
	// pulling is the engine image's pull that runs while the snapshot
	// still arrives.
	pulling chan imagePull
	// mounts are a pod restore's bind mounts.
	mounts []string
}

// Instance is <state>/restores/<id>/instance.json: what `warmctl stop`
// needs to find a running restore.
type Instance struct {
	ID        string    `json:"id"`
	Container string    `json:"container"`
	PID       int       `json:"pid"`
	Snapshot  string    `json:"snapshot"`
	Listen    string    `json:"listen"`
	Started   time.Time `json:"started"`
}

// Run restores and, with Serve, serves until ctx ends.
func Run(ctx context.Context, o Options) error {
	if err := run.CheckID(o.ID); err != nil {
		return err
	}
	r := &restore{o: o, p: o.Progress}
	defer r.cleanup()
	steps := []struct {
		name string
		fn   func(context.Context) error
	}{
		{"read snapshot", r.read}, {"preflight", r.preflight}, {"wait for launch files", r.launching},
		{"prepare", r.prepare}, {"start receiver", r.start}, {"wait for files", r.arrive},
		{"restore processes", r.processes}, {"wait for weights", r.arrived},
		{"wake engine", r.wake}, {"verify", r.verify}, {"admit", r.admit},
	}
	for _, s := range steps {
		if err := s.fn(ctx); err != nil {
			r.diagnose()
			return fmt.Errorf("%s: %w", s.name, err)
		}
	}
	if !o.Serve {
		r.c = nil
		return nil
	}
	return r.serve(ctx)
}

func (r *restore) read(context.Context) error {
	r.snapshot = r.o.Snapshot
	if r.o.Arrival != nil {
		r.m = r.o.Arrival.Manifest()
		r.p.Printf("snapshot %s: profile %s, %d ranks, %.2f GiB, restoring while it is pulled", r.m.ID,
			r.m.Profile.ID, r.m.Launch.Ranks, float64(r.m.Bytes())/(1<<30))
		return nil
	}
	var err error
	r.m, err = manifest.Read(filepath.Join(r.snapshot, "manifest.json"))
	if err != nil {
		return err
	}
	r.p.Printf("snapshot %s: profile %s, %d ranks, %.2f GiB", r.m.ID, r.m.Profile.ID, r.m.Launch.Ranks,
		float64(r.m.Bytes())/(1<<30))
	return nil
}

func (r *restore) preflight(ctx context.Context) error {
	if r.o.InPlace {
		if err := r.checkPod(); err != nil {
			return err
		}
	}
	var err error
	locks := filepath.Join(r.o.StateDir, "locks")
	if r.gpus, err = CheckHost(ctx, r.m.Hardware, r.o.GPUs, locks); err != nil {
		return err
	}
	if r.lease, err = host.Acquire(locks, r.gpus); err != nil {
		return err
	}
	if r.o.InPlace {
		return nil
	}
	if r.dk, err = docker.New(ctx, r.o.DockerSocket); err != nil {
		return err
	}
	return r.dk.CheckInit(ctx)
}

// CheckHost chooses the GPUs a snapshot taken on h restores onto, the ones
// spec lists or else the first free ones of its model, and checks that it
// can restore there. It takes no lease, so it can run before a pull.
func CheckHost(ctx context.Context, h manifest.Hardware, spec, locks string) ([]host.GPU, error) {
	return checkHost(ctx, h, spec, locks, true)
}

// Fits checks that this host has GPUs a snapshot taken on h restores onto,
// whether or not they are busy now: what a pull for later needs.
func Fits(ctx context.Context, h manifest.Hardware) error {
	_, err := checkHost(ctx, h, "", "", false)
	return err
}

func checkHost(ctx context.Context, h manifest.Hardware, spec, locks string, free bool) ([]host.GPU, error) {
	if len(h.GPUs) == 0 {
		return nil, errors.New("the snapshot records no GPUs")
	}
	all, err := host.Inventory(ctx)
	if err != nil {
		return nil, err
	}
	procs := map[string]int{}
	if free {
		if procs, err = host.Processes(ctx); err != nil {
			return nil, err
		}
	}
	var gpus []host.GPU
	if spec == "" {
		gpus, err = host.Pick(all, len(h.GPUs), h.GPUs[0].Name, procs, locks)
	} else {
		gpus, err = host.Select(all, spec)
	}
	if err != nil {
		return nil, err
	}
	topology, _, err := host.Topology(ctx, gpus)
	if err != nil {
		return nil, err
	}
	if err := Compatible(h, gpus, topology, host.CPUFlags()); err != nil {
		return nil, err
	}
	for _, g := range gpus {
		if procs[g.UUID] > 0 {
			return nil, fmt.Errorf("GPU %d runs %d other processes", g.Index, procs[g.UUID])
		}
	}
	return gpus, nil
}

// Compatible checks this host against the snapshot's: the same GPU model,
// memory and driver, as many GPUs, the same topology class, and every CPU
// feature the donor had.
func Compatible(h manifest.Hardware, gpus []host.GPU, topology string, flags []string) error {
	if len(gpus) != len(h.GPUs) {
		return fmt.Errorf("the snapshot has %d ranks, %d GPUs selected", len(h.GPUs), len(gpus))
	}
	for i, g := range gpus {
		want := h.GPUs[i]
		if g.Name != want.Name || g.MemoryMiB != want.MemoryMiB {
			return fmt.Errorf("GPU %d is %s with %d MiB, the snapshot needs %s with %d MiB",
				g.Index, g.Name, g.MemoryMiB, want.Name, want.MemoryMiB)
		}
		if g.Driver != h.Driver {
			if !h.DriverPortable {
				return fmt.Errorf("driver %s, the snapshot was taken on %s; cuda-checkpoint needs the same driver",
					g.Driver, h.Driver)
			}
			if manifest.DriverBranch(g.Driver) != manifest.DriverBranch(h.Driver) {
				return fmt.Errorf("driver %s, the snapshot was taken on %s; it restores under driver %s.x only",
					g.Driver, h.Driver, manifest.DriverBranch(h.Driver))
			}
		}
	}
	if h.Topology != "" && topology != h.Topology {
		return fmt.Errorf("GPU topology %s, the snapshot was taken on %s", topology, h.Topology)
	}
	if missing := host.MissingFlags(h.CPUFlags, flags); len(missing) > 0 {
		return fmt.Errorf("this CPU lacks %v that the donor's had", missing)
	}
	return nil
}

func (r *restore) diagnose() {
	if r.c == nil {
		return
	}
	if tail := r.c.EngineLog(30); tail != "" {
		r.p.Printf("engine log tail:\n%s", tail)
	}
	if tail := r.c.Tail(30); tail != "" {
		r.p.Printf("agent log tail:\n%s", tail)
	}
}

// cleanup stops and removes the receiver, keeps the logs under
// <state>/logs/<id>, removes the instance and releases the GPUs. A pod's
// logs are its container's: the agent's output went there.
func (r *restore) cleanup() {
	if r.prefetch != nil {
		r.prefetch()
	}
	if r.listener != nil {
		r.listener.Close()
	}
	logs := filepath.Join(r.o.StateDir, "logs", r.o.ID)
	if r.c != nil {
		_ = r.c.Stop(20 * time.Second)
		if !r.o.InPlace {
			_ = os.MkdirAll(logs, 0o755)
			r.c.SaveLogs(filepath.Join(logs, "agent.log"))
		}
		r.c.Remove()
	}
	if r.o.InPlace {
		r.unmountPod()
		if r.dir != "" {
			_ = claimDir(r.dir)
		}
	} else if r.dir != "" {
		_ = os.MkdirAll(logs, 0o755)
		_ = run.CopyFile(filepath.Join(r.dir, "workspace/logs/engine.log"), filepath.Join(logs, "engine.log"), 0o644)
		_ = run.CopyFile(filepath.Join(r.dir, "io/work/restore.log"), filepath.Join(logs, "restore.log"), 0o644)
		_ = os.RemoveAll(r.dir)
	}
	if r.lease != nil {
		r.lease.Release()
	}
}

// ErrStopped is returned when the engine exits while serving.
var ErrStopped = errors.New("engine exited")
