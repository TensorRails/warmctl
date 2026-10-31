//go:build linux

package agent

import (
	"context"
	"errors"
	"fmt"
	"os"
	"time"

	"github.com/TensorRails/warmctl/v2/internal/criu"
	"github.com/TensorRails/warmctl/v2/internal/dmtcp"
)

// How long DMTCP may take to write or read back a tree's images.
const (
	dmtcpCheckpointTimeout = 15 * time.Minute
	dmtcpRestartTimeout    = 15 * time.Minute
)

func (a *agent) dmtcp() bool { return a.cfg.Backend == BackendDMTCP }

// underDMTCP starts the coordinator and puts the engine under dmtcp_launch,
// its images written where the dump step's are.
func (a *agent) underDMTCP(ctx context.Context, argv, env []string) ([]string, []string, error) {
	if err := a.cfg.DMTCP.Check(); err != nil {
		return nil, nil, err
	}
	images := a.images()
	if err := os.MkdirAll(images, 0o755); err != nil {
		return nil, nil, err
	}
	if err := a.cfg.DMTCP.Coordinator(ctx); err != nil {
		return nil, nil, fmt.Errorf("dmtcp coordinator: %w", err)
	}
	if len(env) == 0 {
		env = os.Environ()
	}
	// Later entries win in exec.Cmd.Env.
	env = append(append([]string(nil), env...), dmtcp.Env...)
	return a.cfg.DMTCP.Wrap(argv, images), env, nil
}

// images is where the snapshot's process images go: the dump's directory,
// so the CRIU and DMTCP images publish and prefetch alike.
func (a *agent) images() string {
	if a.cfg.Mode == ModeRestore {
		return a.cfg.Images
	}
	return a.cfg.Out + "/criu"
}

func (a *agent) peers(ctx context.Context) (dmtcp.Map, error) {
	peers, err := a.cfg.DMTCP.List(ctx)
	if err != nil {
		return dmtcp.Map{}, err
	}
	return dmtcp.NewMap(peers), nil
}

// virtualTree is tree with DMTCP's PIDs, the ones the processes see and a
// restart keeps. Threads are dropped: their virtual IDs are not listed.
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
		// The root's parent is the agent, outside the computation.
		parent := m.ToVirtual[p.Parent]
		out.Processes = append(out.Processes, criu.Process{PID: pid, Parent: parent, Start: p.Start})
	}
	return out, nil
}

// dmtcpFreeze reports the running tree: DMTCP stops it itself, inside the
// dump.
func (a *agent) dmtcpFreeze(ctx context.Context) (any, error) {
	tree, err := criu.VisibleTree(a.enginePID())
	if err != nil {
		return nil, err
	}
	refs := references(tree, a.cfg.Watch)
	m, err := a.peers(ctx)
	if err != nil {
		return nil, err
	}
	if tree, err = virtualTree(tree, m); err != nil {
		return nil, err
	}
	return FreezeResult{Tree: tree, References: refs}, nil
}

// dmtcpDump writes every process's image and ends the computation.
func (a *agent) dmtcpDump(ctx context.Context) (any, error) {
	start := time.Now()
	if err := a.cfg.DMTCP.Checkpoint(ctx, dmtcpCheckpointTimeout); err != nil {
		return nil, err
	}
	seconds := time.Since(start).Seconds()
	if err := a.cfg.DMTCP.Kill(ctx); err != nil {
		return nil, err
	}
	select {
	case <-a.exited:
	case <-time.After(30 * time.Second):
		return nil, errors.New("the engine outlived its dump")
	}
	return DumpResult{Seconds: seconds, Version: a.cfg.DMTCP.Version(ctx)}, nil
}

// dmtcpRestore restarts the tree from its images. dmtcp_restart's process
// becomes the tree's root; the agent ends when it exits.
func (a *agent) dmtcpRestore(ctx context.Context) (any, error) {
	start := time.Now()
	images, err := dmtcp.Images(a.images())
	if err != nil {
		return nil, err
	}
	if err := a.cfg.DMTCP.Coordinator(ctx); err != nil {
		return nil, fmt.Errorf("dmtcp coordinator: %w", err)
	}
	cmd, err := a.cfg.DMTCP.Restart(images, a.log)
	if err != nil {
		return nil, err
	}
	a.mu.Lock()
	a.engine = cmd.Process.Pid
	a.mu.Unlock()
	go func() {
		a.setExit(cmd.Wait())
		a.logf("restored engine exited")
		a.finish()
	}()
	peers, err := a.cfg.DMTCP.WaitPeers(ctx, len(images), dmtcpRestartTimeout)
	if err != nil {
		return nil, err
	}
	a.dmtcpMap = dmtcp.NewMap(peers)
	tree, err := criu.VisibleTree(cmd.Process.Pid)
	if err != nil {
		return nil, err
	}
	if tree, err = virtualTree(tree, a.dmtcpMap); err != nil {
		return nil, err
	}
	return RestoreResult{Tree: tree, Seconds: time.Since(start).Seconds()}, nil
}
