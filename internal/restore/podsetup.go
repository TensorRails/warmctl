package restore

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/TensorRails/warmctl/v2/internal/agent"
	"github.com/TensorRails/warmctl/v2/internal/dmtcp"
	"github.com/TensorRails/warmctl/v2/internal/docker"
	"github.com/TensorRails/warmctl/v2/internal/run"
	"github.com/TensorRails/warmctl/v2/internal/snapshot"
)

// podSetup makes this container what a receiver container is for the
// snapshot's backend, Kubernetes or RunPod having made it rather than
// Docker. CRIU's restore needs a privileged pod, and gets the mounts and
// namespaces Docker would have given (mountedPod); DMTCP's runs without
// privileges, with the files linked where its processes reopen them by
// path (linkedPod).
type podSetup interface {
	// check is the pod half of the preflight.
	check() error
	// claimTree makes one of the trees the engine owns this restore's.
	claimTree(t agent.Tree) error
	// agent sets the agent config's part of the setup.
	agent(cfg *agent.Config)
	// start puts the engine's files in place and starts the agent as this
	// process's child.
	start(ctx context.Context) error
	// undo reverses what start and claimTree left, once the agent is gone.
	undo()
}

func (r *restore) podSetup() podSetup {
	if r.dmtcp() {
		return linkedPod{r}
	}
	return mountedPod{r}
}

// mountedPod is CRIU's: a privileged pod, where the restore mounts what
// Docker would have mounted, at its tmpfs sizes, and the agent names a UTS
// namespace of its own.
type mountedPod struct{ r *restore }

func (p mountedPod) check() error { return p.r.checkPod() }

func (p mountedPod) claimTree(t agent.Tree) error {
	size, ours := p.r.tmpfs(t.Path)
	if !ours {
		return claimDir(t.Path)
	}
	if err := mountTmpfs(t.Path, size, t.Path == "/tmp"); err != nil {
		return fmt.Errorf("tmpfs at %s: %w", t.Path, err)
	}
	p.r.mounts = append(p.r.mounts, t.Path)
	return nil
}

// agent runs in a UTS namespace of its own and names it as a receiver
// container is named; the pod's own host name stays.
func (p mountedPod) agent(cfg *agent.Config) {
	cfg.Hostname = docker.Hostname
	if p.r.o.CRIU != "" {
		cfg.CRIU = p.r.o.CRIU
	}
}

func (p mountedPod) start(ctx context.Context) error {
	model, err := podModel(p.r.o.ModelDir, p.r.m)
	if err != nil {
		return err
	}
	mounts, err := p.r.podMounts(model)
	if err != nil {
		return err
	}
	for _, m := range mounts {
		if err := mountPoint(m.Source, m.Target); err != nil {
			return err
		}
		if err := bindMount(m.Source, m.Target, m.ReadOnly); err != nil {
			return fmt.Errorf("mount %s at %s: %w", m.Source, m.Target, err)
		}
		p.r.mounts = append(p.r.mounts, m.Target)
	}
	return p.r.spawnAgent(ctx, true)
}

func (p mountedPod) undo() { p.r.unmountPod() }

// linkedPod is DMTCP's: a container without privileges, where nothing is
// mounted and no namespace made. The trees are claimed as they are, and the
// files the processes reopen by path linked where the donor had them: the
// model, the adapter, the GPUs' device nodes.
type linkedPod struct{ r *restore }

func (p linkedPod) check() error { return p.r.checkUnprivileged() }

// claimTree takes /tmp and /dev/shm as the container has them: a tmpfs of
// the receiver's size needs a mount.
func (p linkedPod) claimTree(t agent.Tree) error { return claimDir(t.Path) }

// agent keeps the pod's host name: the restarted engine runs under it.
func (p linkedPod) agent(cfg *agent.Config) {
	cfg.Backend, cfg.DMTCP = agent.BackendDMTCP, p.tools()
}

// tools are DMTCP's programs as the agent runs them, its coordinator's and
// workers' files in the run directory, out of the trees.
func (p linkedPod) tools() dmtcp.Tools {
	t := p.r.o.DMTCP
	t.TmpDir = filepath.Join(p.r.dir, "dmtcp-tmp")
	return t
}

func (p linkedPod) start(ctx context.Context) error {
	model, err := dmtcpModel(p.r.o.ModelDir, p.r.m)
	if err != nil {
		return err
	}
	if err := p.r.linkDevices(); err != nil {
		return err
	}
	mounts, err := p.r.podMounts(model)
	if err != nil {
		return err
	}
	for _, m := range mounts {
		if err := run.Place(m.Source, m.Target); err != nil {
			return fmt.Errorf("place %s at %s: %w", m.Source, m.Target, err)
		}
	}
	return p.r.spawnAgent(ctx, false)
}

// undo ends the DMTCP coordinator the agent left running, and the engine
// connected to it: an agent killed before its release.
func (p linkedPod) undo() {
	if p.r.dir == "" {
		return
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	if err := p.tools().QuitLeft(ctx); err != nil {
		p.r.p.Printf("warning: ending the DMTCP coordinator the agent left: %v", err)
	}
}

// podMounts are the engine's mounts and the driver libraries it maps
// under this pod's driver.
func (r *restore) podMounts(model string) ([]docker.Mount, error) {
	drivers, err := r.driverMounts()
	if err != nil {
		return nil, err
	}
	if len(drivers) == 0 && !r.o.CPUTest {
		r.warnDriver()
	}
	return append(r.engineMounts(model), drivers...), nil
}

// spawnAgent starts the agent as this process's child, in a UTS namespace
// of its own with newUTS.
func (r *restore) spawnAgent(ctx context.Context, newUTS bool) error {
	self, err := os.Executable()
	if err != nil {
		return err
	}
	r.c, err = run.Spawn(ctx, run.Child{
		Name: r.o.ID, Argv: []string{self, "agent", "--config", filepath.Join(r.dir, "work", "agent.json")},
		Env: podEnv(os.Environ(), r.m.Launch.Env), IO: filepath.Join(r.dir, "io"), Log: snapshot.EngineLog,
		AgentLog: filepath.Join(r.dir, "agent.log"), Output: os.Stderr, Ranks: r.m.Launch.Ranks, NewUTS: newUTS,
	})
	if err != nil {
		return err
	}
	r.p.Printf("receiver agent started in this pod")
	return nil
}
