//go:build linux

package agent

import (
	"context"
	"errors"
	"fmt"
	"os"
	"syscall"
	"time"

	"github.com/TensorRails/warmctl/v2/internal/criu"
	"github.com/TensorRails/warmctl/v2/internal/dmtcp"
)

// How long DMTCP may take to write or read back a tree's images.
const (
	dmtcpCheckpointTimeout = 15 * time.Minute
	dmtcpRestartTimeout    = 15 * time.Minute
)

// dmtcpTree runs DMTCP, in user space: the engine starts under dmtcp_launch
// and its coordinator, DMTCP stops the tree itself inside the dump, and the
// restarted tree keeps DMTCP's virtual PIDs inside while it has new real
// ones outside. The snapshot records the virtual PIDs: the ones the
// processes report (vLLM's workers) and a restart keeps.
type dmtcpTree struct {
	cfg  Config
	logf func(string, ...any)
	// restarted maps the restarted tree's virtual PIDs to real ones.
	restarted dmtcp.Map
}

// images is where the process images go: the dump's directory, so that the
// images of both backends publish, distribute and prefetch alike.
func (t *dmtcpTree) images() string {
	if t.cfg.Mode == ModeRestore {
		return t.cfg.Images
	}
	return t.cfg.Out + "/criu"
}

// launch starts the coordinator. DMTCP's virtual PIDs start at 40000, above
// criu.PIDFloor, so there is no allocator to move.
func (t *dmtcpTree) launch(ctx context.Context, argv, env []string) ([]string, []string, error) {
	if err := t.cfg.DMTCP.Check(); err != nil {
		return nil, nil, err
	}
	if err := os.MkdirAll(t.images(), 0o755); err != nil {
		return nil, nil, err
	}
	if err := t.cfg.DMTCP.Coordinator(ctx); err != nil {
		return nil, nil, fmt.Errorf("dmtcp coordinator: %w", err)
	}
	if len(env) == 0 {
		env = os.Environ()
	}
	// Later entries win in exec.Cmd.Env.
	env = append(append([]string(nil), env...), dmtcp.Env...)
	return t.cfg.DMTCP.Wrap(argv, t.images()), env, nil
}

func (t *dmtcpTree) started(int) error { return nil }

func (t *dmtcpTree) peers(ctx context.Context) (dmtcp.Map, error) {
	peers, err := t.cfg.DMTCP.List(ctx)
	if err != nil {
		return dmtcp.Map{}, err
	}
	return dmtcp.NewMap(peers), nil
}

// freeze only reports the tree: DMTCP stops it inside the dump.
func (t *dmtcpTree) freeze(ctx context.Context, root int) (FreezeResult, error) {
	visible, err := criu.VisibleTree(root)
	if err != nil {
		return FreezeResult{}, err
	}
	m, err := t.peers(ctx)
	if err != nil {
		return FreezeResult{}, err
	}
	virtual, err := virtualTree(visible, m)
	if err != nil {
		return FreezeResult{}, err
	}
	return FreezeResult{Tree: virtual, References: references(visible, t.cfg.Watch)}, nil
}

// virtualTree is tree in DMTCP's PIDs. Threads are dropped: their virtual
// IDs are not listed, and nothing restores them by number.
func virtualTree(tree criu.Tree, m dmtcp.Map) (criu.Tree, error) {
	root, ok := m.ToVirtual[tree.Root]
	if !ok {
		return criu.Tree{}, fmt.Errorf("dmtcp: the tree's root %d is not in the computation", tree.Root)
	}
	out := criu.Tree{Root: root}
	for _, p := range tree.Processes {
		pid, ok := m.ToVirtual[p.PID]
		if !ok {
			return criu.Tree{}, fmt.Errorf("dmtcp: process %d of the tree is not in the computation", p.PID)
		}
		// The root's parent, the agent, is outside the computation: 0.
		out.Processes = append(out.Processes, criu.Process{PID: pid, Parent: m.ToVirtual[p.Parent], Start: p.Start})
	}
	return out, nil
}

// dump writes every process's image and ends the computation.
func (t *dmtcpTree) dump(ctx context.Context, _ int, exited <-chan struct{}) (DumpResult, error) {
	start := time.Now()
	if err := t.cfg.DMTCP.Checkpoint(ctx, dmtcpCheckpointTimeout); err != nil {
		return DumpResult{}, err
	}
	seconds := time.Since(start).Seconds()
	if err := t.cfg.DMTCP.Kill(ctx); err != nil {
		return DumpResult{}, err
	}
	select {
	case <-exited:
	case <-time.After(30 * time.Second):
		return DumpResult{}, errors.New("the engine outlived its dump")
	}
	return DumpResult{Seconds: seconds, Version: t.cfg.DMTCP.Version(ctx)}, nil
}

// restore restarts every image under a new coordinator. dmtcp_restart's
// process becomes the tree's root, a child of the agent.
func (t *dmtcpTree) restore(ctx context.Context) (RestoreResult, int, func() error, error) {
	start := time.Now()
	images, err := dmtcp.Images(t.images())
	if err != nil {
		return RestoreResult{}, 0, nil, err
	}
	if err := t.cfg.DMTCP.Coordinator(ctx); err != nil {
		return RestoreResult{}, 0, nil, fmt.Errorf("dmtcp coordinator: %w", err)
	}
	cmd, err := t.cfg.DMTCP.Restart(images, os.Stderr)
	if err != nil {
		return RestoreResult{}, 0, nil, err
	}
	fail := func(err error) (RestoreResult, int, func() error, error) {
		_ = cmd.Process.Kill()
		_ = cmd.Wait()
		return RestoreResult{}, 0, nil, err
	}
	peers, err := t.cfg.DMTCP.WaitPeers(ctx, len(images), dmtcpRestartTimeout)
	if err != nil {
		return fail(err)
	}
	t.restarted = dmtcp.NewMap(peers)
	visible, err := criu.VisibleTree(cmd.Process.Pid)
	if err != nil {
		return fail(err)
	}
	virtual, err := virtualTree(visible, t.restarted)
	if err != nil {
		return fail(err)
	}
	t.logf("dmtcp restarted %d processes in %.2fs", len(peers), time.Since(start).Seconds())
	return RestoreResult{Tree: virtual, Seconds: time.Since(start).Seconds()}, cmd.Process.Pid, cmd.Wait, nil
}

// thaw has nothing to do: the restarted tree runs, and its CUDA calls wait
// for the cuda step.
func (t *dmtcpTree) thaw(context.Context) error { return nil }

func (t *dmtcpTree) snapshotPIDs(ctx context.Context, pids []int) ([]int, error) {
	if len(pids) == 0 {
		return pids, nil
	}
	m, err := t.peers(ctx)
	if err != nil {
		return nil, err
	}
	return m.Virtual(pids)
}

func (t *dmtcpTree) hostPIDs(_ context.Context, pids []int) ([]int, error) {
	return t.restarted.Real(pids)
}

func (t *dmtcpTree) stop(root int) { _ = syscall.Kill(root, syscall.SIGTERM) }
