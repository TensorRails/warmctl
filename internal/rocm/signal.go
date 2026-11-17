package rocm

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"syscall"
	"time"
)

// Processes of the engine other than vLLM's workers can initialize ROCm too:
// vLLM 0.33's API server and engine core import kernels that probe the GPU
// (aiter, Triton's AMD target, flash-linear-attention). They hold no GPU
// memory of their own, but their runtime holds the devices. warmctl's vLLM
// plugin (python/warmctl_vllm/plugin.py) lets such a process checkpoint and
// restore its own GPU state through libkfdreplay on a signal, and reports the
// outcome in a file named by its PID (the one the process sees: under DMTCP,
// the snapshot PID) in StatusDir.
const (
	// CheckpointSignal and RestoreSignal are SIGRTMIN+6 and SIGRTMIN+7 of
	// glibc (DMTCP keeps SIGUSR2 and its own real-time signals below).
	CheckpointSignal = syscall.Signal(40)
	RestoreSignal    = syscall.Signal(41)
	// StatusEnv names the status directory in the engine's environment; the
	// plugin does nothing without it, so engines other than AMD ones never
	// install the handlers.
	StatusEnv = "WARMCTL_KFDR_STATUS"
	// StatusDir is where the plugin writes "checkpointed", "restored",
	// "skipped" (a process the signal did not checkpoint, as a worker), or
	// "error: <reason>".
	StatusDir       = "/workspace/.warmctl-kfdr"
	legacyStatusDir = "/run/warmctl/kfdr"
)

// PluginStatusDir selects the location saved in the engine's launch
// environment. Older snapshots can contain the original receiver path.
func PluginStatusDir() (string, error) {
	switch dir := os.Getenv(StatusEnv); dir {
	case StatusDir:
		return StatusDir, nil
	case "", legacyStatusDir:
		return legacyStatusDir, nil
	default:
		return "", fmt.Errorf("unsupported %s directory %q", StatusEnv, dir)
	}
}

// SignalTimeout bounds one process's checkpoint or restore.
var SignalTimeout = 2 * time.Minute

// Process is one process of the tree: its PID on the host and the one it sees.
type Process struct {
	Host, Snapshot int
}

// Holders are the processes among pids that hold or map an AMD GPU device.
func Holders(pids []int) ([]int, error) {
	var out []int
	for _, pid := range pids {
		f, err := readProc(pid)
		if os.IsNotExist(err) {
			continue
		}
		if err != nil {
			return nil, err
		}
		for _, p := range append(append([]string(nil), f.maps...), f.fds...) {
			if device(p) {
				out = append(out, pid)
				break
			}
		}
	}
	sort.Ints(out)
	return out, nil
}

// Catches reports whether pid has a handler for sig: only those are sent it,
// as the default action of a real-time signal ends the process.
func Catches(pid int, sig syscall.Signal) bool {
	raw, err := os.ReadFile(filepath.Join(Proc, strconv.Itoa(pid), "status"))
	if err != nil {
		return false
	}
	for _, line := range strings.Split(string(raw), "\n") {
		if mask, ok := strings.CutPrefix(line, "SigCgt:"); ok {
			bits, err := strconv.ParseUint(strings.TrimSpace(mask), 16, 64)
			return err == nil && bits&(1<<(uint(sig)-1)) != 0
		}
	}
	return false
}

// Signal sends sig to each process and waits for its status: one of want,
// or "skipped" when skip is true. A process reporting an error, or none in
// time, fails the step with its reason.
func Signal(ctx context.Context, procs []Process, sig syscall.Signal, dir, want string, skip bool) error {
	if len(procs) == 0 {
		return nil
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	for _, p := range procs {
		name := filepath.Join(dir, strconv.Itoa(p.Snapshot))
		if err := clearPreviousStatus(name); err != nil {
			return err
		}
		if err := syscall.Kill(p.Host, sig); err != nil {
			return fmt.Errorf("signal process %d: %w", p.Host, err)
		}
	}
	for _, p := range procs {
		if err := wait(ctx, filepath.Join(dir, strconv.Itoa(p.Snapshot)), p.Host, want, skip); err != nil {
			return err
		}
	}
	return nil
}

// A legacy snapshot can leave a previous reply at the same virtual PID. A
// different file at that path is not part of the signal protocol.
func clearPreviousStatus(name string) error {
	info, err := os.Lstat(name)
	if os.IsNotExist(err) {
		return nil
	}
	if err != nil {
		return err
	}
	if info.Mode().IsRegular() {
		raw, err := os.ReadFile(name)
		if err != nil {
			return err
		}
		got := strings.TrimSpace(string(raw))
		if got == "restored" || got == "skipped" || got == "checkpointed" || strings.HasPrefix(got, "error: ") {
			return os.Remove(name)
		}
	}
	return fmt.Errorf("status path %s already contains an unrelated file", name)
}

func wait(ctx context.Context, name string, pid int, want string, skip bool) error {
	deadline := time.Now().Add(SignalTimeout)
	for {
		raw, err := os.ReadFile(name)
		if err == nil {
			if err := os.Remove(name); err != nil {
				return err
			}
			got := strings.TrimSpace(string(raw))
			switch {
			case got == want, skip && got == "skipped":
				return nil
			case strings.HasPrefix(got, "error: "):
				return fmt.Errorf("process %d: libkfdreplay: %s", pid, strings.TrimPrefix(got, "error: "))
			default:
				return fmt.Errorf("process %d reported %q, want %q", pid, got, want)
			}
		}
		if _, err := os.Stat(filepath.Join(Proc, strconv.Itoa(pid))); err != nil {
			return fmt.Errorf("process %d exited before it reported its GPU %s", pid, want)
		}
		if time.Now().After(deadline) {
			return fmt.Errorf("process %d did not report its GPU %s in %s (see the engine log)", pid, want, SignalTimeout)
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(50 * time.Millisecond):
		}
	}
}
