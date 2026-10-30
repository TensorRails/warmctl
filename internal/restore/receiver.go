package restore

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/TensorRails/warmctl/v2/internal/agent"
	"github.com/TensorRails/warmctl/v2/internal/cuda"
	"github.com/TensorRails/warmctl/v2/internal/docker"
	"github.com/TensorRails/warmctl/v2/internal/interpose"
	"github.com/TensorRails/warmctl/v2/internal/manifest"
	"github.com/TensorRails/warmctl/v2/internal/run"
	"github.com/TensorRails/warmctl/v2/internal/snapshot"
)

// paths are where the agent and the engine see the snapshot, the agent's
// sockets and its work directory: mounts in a receiver container, the real
// directories in a pod.
type paths struct{ capture, state, work string }

func (r *restore) guest() paths {
	if r.o.InPlace {
		return paths{r.snapshot, filepath.Join(r.dir, "io"), filepath.Join(r.dir, "work")}
	}
	return paths{"/capture", "/state", "/work"}
}

func (r *restore) claim() error {
	if r.o.InPlace {
		return r.claimPod()
	}
	r.dir = filepath.Join(r.o.StateDir, "restores", r.o.ID)
	if err := run.Claim(r.dir); err != nil {
		r.dir = ""
		return err
	}
	for _, d := range []string{"work", "io", "workspace"} {
		if err := os.MkdirAll(filepath.Join(r.dir, d), 0o755); err != nil {
			return err
		}
	}
	return nil
}

// tools is the agent's base config: in a container it runs this binary and
// the host's tools from /work; in a pod they are where they are.
func (r *restore) tools() (agent.Config, error) {
	if !r.o.InPlace {
		return snapshot.Tools(filepath.Join(r.dir, "work"), r.o.CRIU, r.o.CUDACheckpoint)
	}
	cfg := snapshot.AgentConfig(r.guest().state)
	// The agent runs in a UTS namespace of its own and names it as a
	// receiver container is named; the pod's own host name stays.
	cfg.Hostname = docker.Hostname
	if r.o.CRIU != "" {
		cfg.CRIU = r.o.CRIU
	}
	if r.o.CUDACheckpoint != "" {
		cfg.CUDACheckpoint = r.o.CUDACheckpoint
	}
	return cfg, nil
}

func (r *restore) prepare(ctx context.Context) error {
	if err := r.claim(); err != nil {
		return err
	}
	cfg, err := r.tools()
	if err != nil {
		return err
	}
	at := r.guest()
	cfg.Mode = agent.ModeRestore
	cfg.Images, cfg.FS, cfg.Work = at.capture+"/criu", at.capture+"/fs", at.state+"/work"
	cfg.Root, cfg.MaxID = r.m.Process.RootPID, r.m.Process.MaxID
	if r.interposed() {
		// Blobs in the store are read-only: the coordinator runs from a copy.
		coordinator := filepath.Join(r.dir, "work", interpose.Coordinator)
		if err := run.CopyFile(filepath.Join(r.snapshot, "ipc", interpose.Coordinator), coordinator, 0o755); err != nil {
			return err
		}
		cfg.IPC, cfg.IPCCoordinator = interpose.Interpose, at.work+"/"+interpose.Coordinator
	}
	if err := cfg.Write(filepath.Join(r.dir, "work", "agent.json")); err != nil {
		return err
	}
	if r.o.Arrival != nil {
		if err := os.WriteFile(filepath.Join(r.dir, "work", "manifest.json"), r.o.Arrival.ManifestJSON(), 0o644); err != nil {
			return err
		}
	}
	inst := Instance{ID: r.o.ID, Container: run.ContainerName(r.o.ID), PID: os.Getpid(), Snapshot: r.snapshot,
		Listen: r.o.Listen, Started: time.Now().UTC()}
	raw, _ := json.MarshalIndent(inst, "", "  ")
	if err := os.WriteFile(filepath.Join(r.dir, "instance.json"), raw, 0o644); err != nil {
		return err
	}
	if r.o.InPlace {
		return nil
	}
	r.image, err = r.ensureImage(ctx)
	return err
}

func (r *restore) spec() (docker.Spec, error) {
	res := r.m.Profile.Resources
	var uuids []string
	for _, g := range r.gpus {
		if !r.o.CPUTest {
			uuids = append(uuids, g.UUID)
		}
	}
	model, err := ModelDir(r.o.ModelDir, r.m)
	if err != nil {
		return docker.Spec{}, err
	}
	drivers, err := r.driverMounts()
	if err != nil {
		return docker.Spec{}, err
	}
	mounts := append(append([]docker.Mount{
		{Source: filepath.Join(r.dir, "work"), Target: "/work", ReadOnly: true},
		{Source: r.snapshot, Target: "/capture", ReadOnly: true},
		{Source: filepath.Join(r.dir, "io"), Target: "/state"},
		{Source: filepath.Join(r.dir, "workspace"), Target: "/workspace"},
	}, r.engineMounts(model)...), drivers...)
	memory, _ := parseSize(res.Memory)
	shm, _ := parseSize(res.SHM)
	return docker.Spec{
		Name: run.ContainerName(r.o.ID), Image: r.image.ID,
		Entrypoint: []string{"/work/warmctl", "agent", "--config", "/work/agent.json"},
		Env:        r.m.Launch.Env,
		Labels:     map[string]string{"io.warmctl.id": r.o.ID, "io.warmctl.kind": "restore"},
		Mounts:     mounts, GPUs: uuids, Memory: memory, SHM: shm, CPUs: max(res.CPUs, 1), PIDs: int64(max(res.PIDs, 1024)),
		Checkpoint: true,
	}, nil
}

// driverMounts put the donor's driver libraries back at the paths its
// container had them, when this host runs another release of the driver
// branch: the restored processes map them by path, and CRIU checks their
// mode: the store keeps files read-only, the container toolkit mounted them
// 0644.
func (r *restore) driverMounts() ([]docker.Mount, error) {
	if len(r.gpus) == 0 || r.gpus[0].Driver == r.m.Hardware.Driver {
		return nil, nil
	}
	var mounts []docker.Mount
	for _, p := range r.m.Hardware.DriverFiles {
		copied := filepath.Join(r.dir, "driver", filepath.Base(p))
		if err := os.MkdirAll(filepath.Dir(copied), 0o755); err != nil {
			return nil, err
		}
		_ = os.Remove(copied)
		if err := run.CopyFile(filepath.Join(r.snapshot, "driver", filepath.Base(p)), copied, 0o644); err != nil {
			return nil, err
		}
		if err := os.Chmod(copied, 0o644); err != nil {
			return nil, err
		}
		mounts = append(mounts, docker.Mount{Source: copied, Target: p, ReadOnly: true})
	}
	return mounts, nil
}

func (r *restore) start(ctx context.Context) error {
	if err := r.launch(ctx); err != nil {
		return err
	}
	if r.m.Profile.Parallel.DP > 1 {
		r.c.ReplyThrough(r.guest().state)
	}
	return nil
}

func (r *restore) launch(ctx context.Context) error {
	if r.o.InPlace {
		return r.startPod(ctx)
	}
	spec, err := r.spec()
	if err != nil {
		return err
	}
	if r.c, err = run.Launch(ctx, r.dk, spec, filepath.Join(r.dir, "io"), r.m.Launch.Ranks); err != nil {
		return err
	}
	r.p.Printf("receiver %s started", r.c.Name)
	return nil
}

// processes installs the trees, restores the tree with CRIU, thaws it and
// puts the CUDA state of its owners on this host's GPUs.
func (r *restore) processes(ctx context.Context) error {
	var installed, restored struct {
		Seconds float64 `json:"seconds"`
	}
	// The CRIU images and then the weights are read ahead in parallel while
	// the restore steps run; CRIU and the weights reload then read memory.
	// A pull in progress leaves what it writes in memory itself.
	if r.o.Arrival == nil {
		pctx, cancel := context.WithCancel(context.Background())
		r.prefetch = cancel
		r.pf = &prefetchStats{start: time.Now()}
		go Prefetch(pctx, r.snapshot, r.m, r.pf)
	}
	if err := r.c.Step(ctx, "install-fs", nil, &installed); err != nil {
		return err
	}
	if err := r.c.Step(ctx, "criu", nil, &restored); err != nil {
		return err
	}
	r.p.Printf("trees installed in %.1fs, CRIU restore in %.1fs", installed.Seconds, restored.Seconds)
	if err := r.c.Step(ctx, "thaw", nil, nil); err != nil {
		return err
	}
	args, err := r.cudaArgs()
	if err != nil {
		return err
	}
	var res agent.CUDAResult
	if err := r.c.Step(ctx, "cuda", args, &res); err != nil {
		return err
	}
	r.p.Printf("CUDA restored: %d owners (%d in parallel) in %.1fs%s",
		len(args.Owners), len(args.Parallel), res.Seconds, phases(res.Phases))
	if !r.interposed() || len(r.m.Process.IPCProcesses) == 0 {
		return nil
	}
	var ipc agent.IPCResult
	if err := r.c.Step(ctx, "ipc", agent.IPCRestoreArgs{Processes: r.m.Process.IPCProcesses}, &ipc); err != nil {
		return err
	}
	r.p.Printf("cross-process memory restored in %.1fs", ipc.Seconds)
	return nil
}

func (r *restore) interposed() bool { return r.m.Process.IPC == interpose.Interpose }

func (r *restore) cudaArgs() (agent.CUDARestoreArgs, error) {
	pairs := make([]cuda.Pair, len(r.gpus))
	for i, g := range r.gpus {
		pairs[i] = cuda.Pair{Source: r.m.Hardware.GPUs[i].UUID, Target: g.UUID}
	}
	deviceMap, err := cuda.DeviceMap(pairs)
	if err != nil {
		return agent.CUDARestoreArgs{}, err
	}
	args := agent.CUDARestoreArgs{DeviceMap: deviceMap, Concurrent: Concurrent(r.o.CUDAExecution, r.m.Process)}
	parallel := Parallel(r.o.CUDAExecution, r.m.Process)
	for _, o := range r.m.Process.Owners {
		args.Owners = append(args.Owners, o.PID)
		if parallel && o.Rank >= 0 {
			args.Parallel = append(args.Parallel, o.PID)
		}
	}
	return args, nil
}

// ModelDir is the model directory to mount in the receiver: the operator's,
// else, when the dumped processes keep model files open, the donor's if
// this host has it.
func ModelDir(given string, m manifest.Manifest) (string, error) {
	if given != "" || !m.Process.NeedsModel {
		return given, nil
	}
	if info, err := os.Stat(m.Launch.ModelDir); err == nil && info.IsDir() {
		return m.Launch.ModelDir, nil
	}
	return "", fmt.Errorf("the snapshot's processes keep files of %s open and %q is not on this host; pass --model-dir",
		m.Profile.Model.MountPath(), m.Launch.ModelDir)
}

// Parallel decides whether the workers' CUDA restores run concurrently:
// only when no collective keeps peer memory mapped between them, unless
// the operator forces serial.
func Parallel(mode string, p manifest.Process) bool {
	switch mode {
	case "serial", "serial-v1":
		return false
	case "parallel", "parallel-all", "tp8-independent-workers-v1":
		return true
	}
	return p.CollectivesDetached
}

// Concurrent decides whether the owners without a rank (API servers,
// engine cores) also restore concurrently, and every owner unlocks at once:
// in auto whenever the workers do, since no process then maps another's
// memory. parallel keeps them serial, the order first qualified on GPUs.
func Concurrent(mode string, p manifest.Process) bool {
	switch mode {
	case "parallel-all":
		return true
	case "auto", "":
		return p.CollectivesDetached
	}
	return false
}

// phases formats the CUDA restore's phase seconds, in the order they run.
func phases(p map[string]float64) string {
	if len(p) == 0 {
		return ""
	}
	var parts []string
	for _, k := range []string{"restore_first", "restore_parallel", "unlock", "check"} {
		if v, ok := p[k]; ok {
			parts = append(parts, fmt.Sprintf("%s %.1fs", strings.ReplaceAll(k, "_", " "), v))
		}
	}
	return " (" + strings.Join(parts, ", ") + ")"
}
