//go:build linux

package agent

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"syscall"
	"time"

	"golang.org/x/sys/unix"

	"github.com/TensorRails/warmctl/v2/internal/criu"
	"github.com/TensorRails/warmctl/v2/internal/process"
)

// tree snapshots and restarts the engine's process tree: CRIU where the
// container may run it, DMTCP where it may not (docs/dmtcp.md). The steps
// around it, CUDA, cross-process memory and the trees on disk, are the same
// for both.
type tree interface {
	// launch readies the backend and returns the engine's command line and
	// environment under it.
	launch(ctx context.Context, argv, env []string) ([]string, []string, error)
	// started checks the engine's PID once it runs.
	started(pid int) error
	// freeze readies the running tree rooted at root for the dump and reports
	// it in the snapshot's PIDs.
	freeze(ctx context.Context, root int) (FreezeResult, error)
	// dump writes the tree's images and ends it; exited closes when the
	// engine is gone.
	dump(ctx context.Context, root int, exited <-chan struct{}) (DumpResult, error)
	// restore brings the tree back from the images and returns its root on
	// this host and a wait that returns once the root exits.
	restore(ctx context.Context) (RestoreResult, int, func() error, error)
	// thaw lets the restored tree run.
	thaw(ctx context.Context) error
	// snapshotPIDs turns PIDs of the running tree into the ones the snapshot
	// records; hostPIDs turns recorded ones into the restored tree's.
	snapshotPIDs(ctx context.Context, pids []int) ([]int, error)
	hostPIDs(ctx context.Context, pids []int) ([]int, error)
	// stop asks the tree rooted at root to exit.
	stop(root int)
}

func newTree(cfg Config, log func(string, ...any)) tree {
	if cfg.Backend == BackendDMTCP {
		return &dmtcpTree{cfg: cfg, logf: log}
	}
	return &criuTree{cfg: cfg}
}

// criuTree runs CRIU: the engine starts above a moved PID floor, the tree is
// stopped through pidfds and dumped, and restored stopped with its PIDs.
type criuTree struct {
	cfg    Config
	frozen *criu.Frozen
}

func (t *criuTree) launch(_ context.Context, argv, env []string) ([]string, []string, error) {
	return argv, env, criu.ReservePIDFloor()
}

func (t *criuTree) started(pid int) error {
	if pid < criu.PIDFloor {
		return fmt.Errorf("engine PID %d is below the floor %d", pid, criu.PIDFloor)
	}
	return nil
}

func (t *criuTree) freeze(ctx context.Context, root int) (FreezeResult, error) {
	f, err := criu.Freeze(ctx, root)
	if err != nil {
		return FreezeResult{}, err
	}
	t.frozen = f
	return FreezeResult{Tree: f.Tree, References: references(f.Tree, t.cfg.Watch)}, nil
}

// dump checkpoints the frozen tree. CRIU kills it afterwards.
func (t *criuTree) dump(ctx context.Context, root int, exited <-chan struct{}) (DumpResult, error) {
	if t.frozen == nil {
		return DumpResult{}, errors.New("the tree is not frozen")
	}
	images, work := filepath.Join(t.cfg.Out, "criu"), filepath.Join(t.cfg.Out, "criu-work")
	if err := os.MkdirAll(images, 0o755); err != nil {
		return DumpResult{}, err
	}
	if err := criu.PrepareWork(work); err != nil {
		return DumpResult{}, err
	}
	version := criuVersion(ctx, t.cfg.CRIU)
	start := time.Now()
	if err := criu.Dump(ctx, t.cfg.CRIU, version, root, images, work); err != nil {
		return DumpResult{}, err
	}
	seconds := time.Since(start).Seconds()
	select {
	case <-exited:
	case <-time.After(30 * time.Second):
		return DumpResult{}, errors.New("the engine outlived its dump")
	}
	_ = t.frozen.Close()
	t.frozen = nil
	return DumpResult{Seconds: seconds, Version: version}, nil
}

func criuVersion(ctx context.Context, program string) string {
	out, err := process.Capture(ctx, 10*time.Second, program, "--version")
	if err != nil {
		return ""
	}
	line, _, _ := strings.Cut(out.Stdout, "\n")
	return strings.TrimSpace(strings.TrimPrefix(line, "Version:"))
}

// restore brings the tree back stopped; its root's pidfd turns readable when
// it exits, and PID 1 (docker-init or tini), its new parent, reaps it.
func (t *criuTree) restore(ctx context.Context) (RestoreResult, int, func() error, error) {
	start := time.Now()
	f, err := criu.RestoreStopped(ctx, t.cfg.CRIU, t.cfg.Images, t.cfg.Work, t.cfg.Root, t.cfg.MaxID)
	if err != nil {
		return RestoreResult{}, 0, nil, err
	}
	t.frozen = f
	fd := f.RootFD()
	wait := func() error {
		fds := []unix.PollFd{{Fd: int32(fd), Events: unix.POLLIN}}
		for {
			_, err := unix.Poll(fds, -1)
			if err == nil || !errors.Is(err, unix.EINTR) {
				return errors.New("engine exited")
			}
		}
	}
	return RestoreResult{Tree: f.Tree, Seconds: time.Since(start).Seconds()}, f.Tree.Root, wait, nil
}

func (t *criuTree) thaw(ctx context.Context) error { return t.frozen.Thaw(ctx) }

// CRIU restores every PID as it was.
func (t *criuTree) snapshotPIDs(_ context.Context, pids []int) ([]int, error) { return pids, nil }
func (t *criuTree) hostPIDs(_ context.Context, pids []int) ([]int, error)     { return pids, nil }

func (t *criuTree) stop(root int) {
	if t.frozen != nil {
		_ = t.frozen.Thaw(context.Background())
		_ = t.frozen.Signal(unix.SIGTERM)
		return
	}
	_ = syscall.Kill(root, syscall.SIGTERM)
}
