//go:build linux

package agent

import (
	"context"
	"errors"
	"fmt"
	"os"
	"strings"
	"syscall"
	"time"

	"github.com/TensorRails/warmctl/v2/internal/criu"
	"github.com/TensorRails/warmctl/v2/internal/rocm"
)

// CoverageEnv set to "warn" reports files an AMD tree maps outside the
// snapshot instead of failing the freeze (research builds only).
const CoverageEnv = "WARMCTL_AMD_COVERAGE"

// amdReleased is the cuda step of an AMD snapshot: the workers checkpointed
// their GPU state themselves (the host asked them through the engine); the
// other processes that still hold a GPU device and run warmctl's vLLM
// plugin checkpoint theirs on a signal (rocm.CheckpointSignal); then no
// process of the tree may hold one.
func (a *agent) amdReleased(ctx context.Context, pids []int, start time.Time) (any, error) {
	holders, err := rocm.Holders(pids)
	if err != nil {
		return nil, err
	}
	if err := a.signalGPU(ctx, holders, rocm.CheckpointSignal, "checkpointed", false); err != nil {
		return nil, fmt.Errorf("GPU checkpoint of the engine's other processes: %w", err)
	}
	owners, err := rocm.VerifyReleased(pids, a.cfg.GPULib)
	if err != nil {
		return nil, err
	}
	if owners, err = a.tree.snapshotPIDs(ctx, owners); err != nil {
		return nil, err
	}
	return CUDAResult{Owners: owners, Seconds: time.Since(start).Seconds(), Backend: "libkfdreplay"}, nil
}

// amdPreloaded is the cuda step of an AMD restore: the workers replay their
// GPU state themselves once the host asks; every restored process that runs
// ROCm must have libkfdreplay to do so.
func (a *agent) amdPreloaded(ctx context.Context, args CUDARestoreArgs, start time.Time) (any, error) {
	tree, err := criu.VisibleTree(a.enginePID())
	if err != nil {
		return nil, err
	}
	if err := rocm.VerifyPreloaded(tree.PIDs(), a.cfg.GPULib); err != nil {
		return nil, err
	}
	// The processes a signal checkpointed restore the same way; workers
	// answer "skipped" and restore when the host asks them.
	hosts, err := a.tree.hostPIDs(ctx, args.Owners)
	if err != nil {
		return nil, err
	}
	if err := a.signalGPU(ctx, hosts, rocm.RestoreSignal, "restored", true); err != nil {
		return nil, fmt.Errorf("GPU restore of the engine's other processes: %w", err)
	}
	return CUDAResult{Owners: args.Owners, Seconds: time.Since(start).Seconds(), Backend: "libkfdreplay"}, nil
}

// signalGPU sends sig to those of pids that catch it and waits for each to
// report want.
func (a *agent) signalGPU(ctx context.Context, pids []int, sig syscall.Signal, want string, skip bool) error {
	var hosts []int
	for _, pid := range pids {
		if rocm.Catches(pid, sig) {
			hosts = append(hosts, pid)
		}
	}
	if len(hosts) == 0 {
		return nil
	}
	snaps, err := a.tree.snapshotPIDs(ctx, hosts)
	if err != nil {
		return err
	}
	procs := make([]rocm.Process, len(hosts))
	for i := range hosts {
		procs[i] = rocm.Process{Host: hosts[i], Snapshot: snaps[i]}
	}
	a.logf("GPU %s by signal: %d processes", want, len(procs))
	return rocm.Signal(ctx, procs, sig, rocm.StatusDir, want, skip)
}

// freeze readies the tree for the dump; on AMD it also checks that the
// snapshot carries every file the tree maps (rocm.Cover).
func (a *agent) freeze(ctx context.Context) (FreezeResult, error) {
	res, err := a.tree.freeze(ctx, a.enginePID())
	if err != nil || a.cfg.GPU != GPUAMD {
		return res, err
	}
	visible, err := criu.VisibleTree(a.enginePID())
	if err != nil {
		return res, err
	}
	carried := []string{a.cfg.Out, a.cfg.DMTCP.TmpDir, a.cfg.Dir, a.cfg.DMTCP.Dir}
	for _, t := range a.cfg.Trees {
		carried = append(carried, t.Path)
	}
	model := ""
	if len(a.cfg.Watch) > 0 {
		model = a.cfg.Watch[len(a.cfg.Watch)-1]
	}
	c, err := rocm.Cover(visible.PIDs(), nonEmpty(carried), model, a.cfg.GPULib, a.started)
	if err != nil {
		return res, err
	}
	if len(c.Uncovered) > 0 {
		msg := fmt.Sprintf("the engine maps or holds %d files the snapshot does not carry, which a restart "+
			"needs unchanged at the same path: %s", len(c.Uncovered), strings.Join(first(c.Uncovered, 20), ", "))
		if os.Getenv(CoverageEnv) != "warn" {
			return res, errors.New(msg + " (set HOME, XDG_CACHE_HOME or the cache's own variable under /workspace)")
		}
		a.logf("warning: %s", msg)
	}
	res.Coverage = &c
	return res, nil
}

func nonEmpty(paths []string) []string {
	var out []string
	for _, p := range paths {
		if p != "" {
			out = append(out, p)
		}
	}
	return out
}

func first(s []string, n int) []string {
	if len(s) > n {
		return append(s[:n:n], fmt.Sprintf("… %d more", len(s)-n))
	}
	return s
}
