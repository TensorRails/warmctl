package snapshot

import (
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"syscall"
	"time"

	"github.com/TensorRails/warmctl/v2/internal/accel"
	"github.com/TensorRails/warmctl/v2/internal/kfdreplay"
	"github.com/TensorRails/warmctl/v2/internal/rocm"
	"github.com/TensorRails/warmctl/v2/internal/run"
	"github.com/TensorRails/warmctl/v2/internal/vllm"
)

// cachePass runs the engine once outside DMTCP, with the bake's very argv,
// environment and trees, through the warmup grid and the canary, and stops
// it: the compile caches (torch.compile, Triton's kernels and their
// launchers, hip_utils) are then warm in /workspace for the engine DMTCP
// runs. A process that has initialized ROCm and forks a compiler under
// DMTCP (v5.0.0-warmctl.2) fails about half the time (gcc exits 99 on a
// DMTCP assertion), so the engine under DMTCP must find nothing to compile.
func (b *bake) cachePass(ctx context.Context) error {
	start := time.Now()
	b.p.Printf("compile cache pass: the engine runs once outside DMTCP")
	log, err := os.Create(filepath.Join(b.dir, "cache-pass.log"))
	if err != nil {
		return err
	}
	defer log.Close()
	argv := b.cfg.Argv
	cmd := exec.Command(argv[0], argv[1:]...)
	cmd.Env = run.MergeEnv(os.Environ(), b.launch.Env)
	cmd.Dir, cmd.Stdout, cmd.Stderr = b.cfg.Dir, log, log
	cmd.SysProcAttr = &syscall.SysProcAttr{Setsid: true}
	if err := cmd.Start(); err != nil {
		return err
	}
	exited := make(chan error, 1)
	go func() { exited <- cmd.Wait() }()
	stop := func() {
		_ = syscall.Kill(-cmd.Process.Pid, syscall.SIGTERM)
		select {
		case <-exited:
		case <-time.After(30 * time.Second):
			_ = syscall.Kill(-cmd.Process.Pid, syscall.SIGKILL)
			<-exited
		}
	}
	engine := vllm.NewTCP(EngineAddr, b.launch.Profile.Parallel.Ranks())
	alive := func() error {
		select {
		case err := <-exited:
			exited <- err
			return fmt.Errorf("the cache pass's engine exited (%v); see %s", err, log.Name())
		default:
			return nil
		}
	}
	if err := engine.WaitReady(ctx, b.o.ReadyTimeout, alive); err != nil {
		stop()
		return err
	}
	model, err := ServedName(b.launch.Options)
	if err == nil {
		err = engine.Warmup(ctx, model, max(b.launch.Profile.Parallel.DP, 1), b.launch.WarmupCases(), nil)
	}
	if err == nil {
		_, err = engine.RunCanary(ctx, model, max(b.launch.Profile.Canary.PrefixTokens, 32), 32)
	}
	stop()
	if err != nil {
		return fmt.Errorf("compile cache pass: %w", err)
	}
	b.p.Printf("compile cache pass done in %.0fs", time.Since(start).Seconds())
	return nil
}

// chooseVendor takes the GPU vendor from the host. On an AMD host a bake
// runs in place under DMTCP: in the ROCm runtime image it does so without
// being asked (--in-place is implied); elsewhere it says how.
func (b *bake) chooseVendor() error {
	b.vendor = accel.Host()
	if b.vendor != accel.AMD {
		return nil
	}
	b.p.Printf("AMD Instinct host: GPU state moves through libkfdreplay, the process tree through DMTCP (preview)")
	if b.o.InPlace {
		return nil
	}
	if _, ok, err := run.ReadRuntime(run.RuntimeFile); err != nil || !ok {
		return errors.New("an AMD bake runs in place under DMTCP, in the ROCm runtime image " +
			"(or pass --in-place where DMTCP and libkfdreplay are installed)")
	}
	b.o.InPlace = true
	b.p.Printf("in the runtime image: baking in place")
	return nil
}

// imagePythonPath keeps the image's own PYTHONPATH after the adapter's: a
// ROCm image may find its modules through it (rocm/vllm finds amdsmi there,
// and vLLM finds the ROCm platform through amdsmi), and the profile's
// PYTHONPATH would hide them.
func imagePythonPath(env map[string]string, image string) {
	if image == "" {
		return
	}
	have := map[string]bool{}
	var out []string
	for _, p := range strings.Split(env["PYTHONPATH"]+":"+image, ":") {
		if p != "" && !have[p] {
			have[p] = true
			out = append(out, p)
		}
	}
	env["PYTHONPATH"] = strings.Join(out, ":")
}

// amdPreflight checks the GPUs against what AMD snapshots are verified on,
// and finds the pinned libkfdreplay.
func (b *bake) amdPreflight() error {
	note, err := rocm.Supported(b.gpus)
	if err != nil {
		return err
	}
	if note != "" {
		b.p.Printf("warning: %s", note)
	}
	if b.gpuLib, err = kfdreplay.Locate(); err != nil {
		return err
	}
	imagePythonPath(b.launch.Env, os.Getenv("PYTHONPATH"))
	b.p.Printf("libkfdreplay %s at %s (sha256 %.12s)", orAny(b.gpuLib.Version), b.gpuLib.Path, b.gpuLib.SHA256)
	return nil
}

func orAny(v string) string {
	if v == "" {
		return "(research build)"
	}
	return v
}

// checkWorkers checks that every AMD worker runs the libkfdreplay the bake
// preloads: a worker without it could not be checkpointed.
func (b *bake) checkWorkers(workers []vllm.Worker) error {
	for _, w := range workers {
		switch {
		case !strings.EqualFold(w.PCIBusID, b.gpus[0].PCIBusID):
			return fmt.Errorf("rank %d reports PCI device %q, the selected GPU is %s", w.Rank, w.PCIBusID, b.gpus[0].PCIBusID)
		case w.KFDReplay == "":
			return fmt.Errorf("rank %d runs without libkfdreplay (LD_PRELOAD of %s did not reach it)", w.Rank, b.gpuLib.Path)
		case b.gpuLib.Version != "" && w.KFDReplay != b.gpuLib.Version:
			return fmt.Errorf("rank %d runs libkfdreplay %s, the bake preloads %s", w.Rank, w.KFDReplay, b.gpuLib.Version)
		}
	}
	return nil
}
