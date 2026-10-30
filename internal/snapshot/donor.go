package snapshot

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/TensorRails/warmctl/v2/internal/agent"
	"github.com/TensorRails/warmctl/v2/internal/docker"
	"github.com/TensorRails/warmctl/v2/internal/interpose"
	"github.com/TensorRails/warmctl/v2/internal/pyext"
	"github.com/TensorRails/warmctl/v2/internal/run"
)

// EngineAddr is where the engine listens inside its container.
const EngineAddr = "127.0.0.1:8000"

// Container paths.
const (
	guestWork      = "/work"
	guestState     = "/state"
	guestWorkspace = "/workspace"
	guestCapture   = "/capture"
)

// EngineLog is the engine's log file, in a tree the snapshot carries: a
// restored engine keeps writing it.
const EngineLog = guestWorkspace + "/logs/engine.log"

func (b *bake) prepare(ctx context.Context) error {
	b.dir = filepath.Join(b.o.StateDir, "bakes", b.o.ID)
	if _, err := os.Stat(b.o.Output); err == nil {
		return fmt.Errorf("output %s already exists", b.o.Output)
	}
	if err := run.Claim(b.dir); err != nil {
		b.dir = ""
		return err
	}
	for _, d := range []string{"work", "io/out/weights", "workspace"} {
		if err := os.MkdirAll(filepath.Join(b.dir, d), 0o755); err != nil {
			return err
		}
	}
	cfg, err := Tools(filepath.Join(b.dir, "work"), b.o.CRIU, b.o.CUDACheckpoint)
	if err != nil {
		return err
	}
	cfg.CUDABackend = b.o.CUDABackend
	if b.model, err = run.ModelSHA256(b.o.ModelDir); err != nil {
		return err
	}
	var sum string
	if b.adapter, sum, err = pyext.Materialize(filepath.Join(b.o.StateDir, "adapter")); err != nil {
		return err
	}
	b.launch.Env["WARMCTL_ADAPTER_SHA256"] = sum
	b.launch.Env["WARMCTL_PROFILE"] = b.launch.Profile.ID
	if _, set := b.launch.Env[DriverCheckEnv]; !set {
		b.launch.Env[DriverCheckEnv] = "1"
	}
	if _, set := b.launch.Env["LD_LIBRARY_PATH"]; !set && len(b.gpus) > 0 && hostUMD(b.gpus[0].Driver, len(b.gpus)) {
		b.launch.Env["LD_LIBRARY_PATH"] = hostUMDLibraryPath
	}
	b.cfg = cfg
	b.cfg.Mode, b.cfg.Out = agent.ModeSnapshot, guestState+"/out"
	b.cfg.Argv = append(append([]string(nil), b.launch.Argv...), "--host", "127.0.0.1", "--port", "8000")
	b.cfg.Log = EngineLog
	b.cfg.Watch = []string{b.launch.Profile.Model.MountPath()}
	if err := b.interposer(); err != nil {
		return err
	}
	if err := b.cfg.Write(filepath.Join(b.dir, "work", "agent.json")); err != nil {
		return err
	}
	b.p.Printf("pulling %s", b.launch.Profile.Image)
	b.image, err = b.dk.EnsureImage(ctx, b.launch.Profile.Image, nil)
	return err
}

// interposer puts the cuInterpose build in the bake for a profile whose ipc
// is not native; the agent then decides whether the engine runs under it.
func (b *bake) interposer() error {
	mode := b.launch.Profile.IPCMode()
	// A CPU test has no GPU memory to share.
	if mode == interpose.Native || b.o.CPUTest {
		return nil
	}
	src := b.o.Cuinterpose
	if src == "" {
		src = interpose.DefaultDir()
	}
	if err := interpose.Check(src); err != nil {
		return err
	}
	b.ipcDir = filepath.Join(b.dir, "ipc")
	if err := os.MkdirAll(b.ipcDir, 0o755); err != nil {
		return err
	}
	for _, name := range interpose.Files {
		// The libraries get the read-only mode the snapshot store gives every
		// file: CRIU maps a library back only onto a file of the same mode.
		mode := os.FileMode(0o755)
		if strings.HasSuffix(name, ".so") {
			mode = 0o444
		}
		if err := run.CopyFile(filepath.Join(src, name), filepath.Join(b.ipcDir, name), mode); err != nil {
			return err
		}
	}
	b.cfg.IPC, b.cfg.IPCCoordinator = mode, interpose.GuestDir+"/"+interpose.Coordinator
	return nil
}

// AgentConfig is what every agent needs, its sockets in state: the
// image's CRIU and cuda-checkpoint, the engine's address and the trees.
func AgentConfig(state string) agent.Config {
	return agent.Config{
		Control: state + "/control.sock", Inference: state + "/inference.sock", EngineAddr: EngineAddr,
		CRIU: "criu", CUDACheckpoint: "cuda-checkpoint", Trees: agent.DefaultTrees,
	}
}

// Tools fills a container agent's config and puts this binary and the
// optional host CRIU and cuda-checkpoint into work.
func Tools(work, criuPath, helperPath string) (agent.Config, error) {
	cfg := AgentConfig(guestState)
	if err := run.CopyExecutable(work, "warmctl"); err != nil {
		return cfg, err
	}
	for _, tool := range []struct {
		host string
		dst  *string
		name string
	}{{criuPath, &cfg.CRIU, "criu"}, {helperPath, &cfg.CUDACheckpoint, "cuda-checkpoint"}} {
		if tool.host == "" {
			continue
		}
		if err := run.CopyFile(tool.host, filepath.Join(work, tool.name), 0o755); err != nil {
			return cfg, err
		}
		*tool.dst = guestWork + "/" + tool.name
	}
	return cfg, nil
}

func (b *bake) spec() docker.Spec {
	res := b.launch.Profile.Resources
	var uuids []string
	for _, g := range b.gpus {
		if !b.o.CPUTest {
			uuids = append(uuids, g.UUID)
		}
	}
	mounts := []docker.Mount{
		{Source: filepath.Join(b.dir, "work"), Target: guestWork, ReadOnly: true},
		{Source: filepath.Join(b.dir, "io"), Target: guestState},
		{Source: filepath.Join(b.dir, "workspace"), Target: guestWorkspace},
		{Source: b.o.ModelDir, Target: b.launch.Profile.Model.MountPath(), ReadOnly: true},
		{Source: b.adapter, Target: pyext.GuestPath, ReadOnly: true},
	}
	if b.ipcDir != "" {
		mounts = append(mounts, docker.Mount{Source: b.ipcDir, Target: interpose.GuestDir, ReadOnly: true})
	}
	return docker.Spec{
		Name: run.ContainerName(b.o.ID), Image: b.image.ID,
		Entrypoint: []string{guestWork + "/warmctl", "agent", "--config", guestWork + "/agent.json"},
		Env:        b.launch.Env,
		Labels:     map[string]string{"io.warmctl.id": b.o.ID, "io.warmctl.kind": "snapshot"},
		Mounts:     mounts,
		GPUs:       uuids, Memory: b.launch.Memory, SHM: b.launch.SHM,
		CPUs: max(res.CPUs, 1), PIDs: int64(max(res.PIDs, 1024)), Checkpoint: true,
	}
}

func (b *bake) start(ctx context.Context) error {
	var err error
	ranks := b.launch.Profile.Parallel.Ranks()
	if b.c, err = run.Launch(ctx, b.dk, b.spec(), filepath.Join(b.dir, "io"), ranks); err != nil {
		return err
	}
	if b.launch.Profile.Parallel.DP > 1 {
		b.c.ReplyThrough(guestState)
	}
	b.p.Printf("donor %s started; loading the model", b.c.Name)
	timeout := b.o.ReadyTimeout
	if err := b.c.Engine.WaitReady(ctx, timeout, func() error { return b.c.EngineAlive(ctx) }); err != nil {
		return err
	}
	b.p.Printf("engine ready")
	if b.workers, err = b.c.Engine.Describe(ctx); err != nil {
		return err
	}
	for _, w := range b.workers {
		if w.AdapterSHA256 != b.launch.Env["WARMCTL_ADAPTER_SHA256"] {
			return fmt.Errorf("rank %d runs adapter %q, mounted %q",
				w.Rank, w.AdapterSHA256, b.launch.Env["WARMCTL_ADAPTER_SHA256"])
		}
	}
	w := b.workers[0]
	b.p.Printf("%d workers: %s, vLLM %s, torch %s", len(b.workers), w.Arch, w.VLLM, w.Torch)
	return nil
}

// ServedName is the first served model name of a launch.
func ServedName(options map[string]any) (string, error) {
	switch names := options["served-model-name"].(type) {
	case []any:
		if len(names) > 0 {
			return fmt.Sprint(names[0]), nil
		}
	case string:
		return names, nil
	}
	return "", errors.New("launch has no served-model-name")
}

func (b *bake) warm(ctx context.Context) error {
	model, err := ServedName(b.launch.Options)
	if err != nil {
		return err
	}
	b.served = model
	cases := b.launch.WarmupCases()
	dp := max(b.launch.Profile.Parallel.DP, 1)
	step := max(len(cases)/10, 1)
	err = b.c.Engine.Warmup(ctx, model, dp, cases, func(i int, c profileCase, seconds float64) {
		if (i+1)%step == 0 || i+1 == len(cases) {
			b.p.Printf("warmup %d/%d (%d tokens x %d: %.2fs)", i+1, len(cases), c.PromptTokens, c.Batch, seconds)
		}
	})
	if err != nil {
		return err
	}
	if err := b.c.Engine.ResetPrefixCache(ctx); err != nil {
		return err
	}
	prefix := max(b.launch.Profile.Canary.PrefixTokens, 32)
	if b.canary, err = b.c.Engine.RunCanary(ctx, model, prefix, 32); err != nil {
		return err
	}
	b.p.Printf("canary recorded: %d-token prompt, miss %.3fs, hit %.3fs (%d cached)", prefix,
		b.canary.MissSeconds, b.canary.HitSeconds, b.canary.HitCached)
	return nil
}

func threads(n int) int {
	if n > 0 {
		return n
	}
	if v, err := strconv.Atoi(os.Getenv("WARMCTL_POOL_THREADS")); err == nil && v > 0 {
		return v
	}
	return 4
}
