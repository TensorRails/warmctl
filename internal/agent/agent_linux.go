//go:build linux

package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"syscall"
	"time"

	"golang.org/x/sys/unix"

	"github.com/TensorRails/warmctl/v2/internal/criu"
	"github.com/TensorRails/warmctl/v2/internal/cuda"
	"github.com/TensorRails/warmctl/v2/internal/interpose"
	"github.com/TensorRails/warmctl/v2/internal/process"
	"github.com/TensorRails/warmctl/v2/internal/runtimefs"
)

// Grace is how long a stopping engine gets before the container goes.
const Grace = 20 * time.Second

type agent struct {
	cfg Config
	log io.Writer

	// Steps run one at a time: connections are served in turn.
	pos       int
	step      string
	failed    error
	finishing bool
	frozen    *criu.Frozen
	// interposed is whether the engine runs under cuInterpose, and why.
	interposed bool
	ipcReason  string

	mu      sync.Mutex
	engine  int
	exited  chan struct{}
	exitErr error

	done chan struct{}
	once sync.Once
}

// Run serves the control socket until a snapshot session finishes or a
// restored engine exits. It is the container's main process, under
// docker-init, or in a pod a child of an in-place `warmctl up` under tini.
func Run(ctx context.Context, configPath string, log io.Writer) error {
	cfg, err := ReadConfig(configPath)
	if err != nil {
		return err
	}
	cfg.CRIU, cfg.CUDACheckpoint = tool(cfg.CRIU), tool(cfg.CUDACheckpoint)
	a := &agent{cfg: cfg, log: log, pos: -1, done: make(chan struct{}), exited: make(chan struct{})}
	if cfg.Hostname != "" {
		if err := ownHostname(cfg.Hostname); err != nil {
			return err
		}
		a.logf("host name %s in the agent's own UTS namespace", cfg.Hostname)
	}
	l, err := listenUnix(cfg.Control)
	if err != nil {
		return err
	}
	defer l.Close()
	if cfg.Inference != "" {
		go func() {
			if err := Proxy(ctx, cfg.Inference, cfg.EngineAddr); err != nil {
				a.logf("inference proxy: %v", err)
			}
		}()
	}
	if cfg.Mode == ModeSnapshot {
		if err := a.startEngine(ctx); err != nil {
			return err
		}
	}
	a.logf("%s agent ready on %s", cfg.Mode, cfg.Control)
	go a.accept(ctx, l)
	select {
	case <-a.done:
		return a.failed
	case <-ctx.Done():
		a.shutdown()
		return nil
	}
}

// ownHostname sets the host name of this process's UTS namespace, which
// CRIU and the restored tree share, and refuses when the namespace is its
// parent's: that would rename the container, or the node.
func ownHostname(name string) error {
	self, err := os.Readlink("/proc/self/ns/uts")
	if err != nil {
		return err
	}
	parent, err := os.Readlink(fmt.Sprintf("/proc/%d/ns/uts", os.Getppid()))
	if err != nil {
		return err
	}
	if self == parent {
		return fmt.Errorf("agent: host name %s asked in the UTS namespace of its parent (%s); start it in one of its own",
			name, self)
	}
	return unix.Sethostname([]byte(name))
}

// tool resolves a bare tool name through PATH and then the sbin
// directories images keep CRIU and cuda-checkpoint in (vLLM's image:
// /usr/local/sbin), which a container's PATH may lack.
func tool(name string) string {
	if name == "" || strings.Contains(name, "/") {
		return name
	}
	if p, err := exec.LookPath(name); err == nil {
		return p
	}
	for _, dir := range []string{"/usr/local/sbin", "/usr/sbin", "/sbin"} {
		if info, err := os.Stat(filepath.Join(dir, name)); err == nil && info.Mode()&0o111 != 0 {
			return filepath.Join(dir, name)
		}
	}
	return name
}

func (a *agent) logf(format string, args ...any) {
	fmt.Fprintf(a.log, "%s agent: %s\n", time.Now().UTC().Format("15:04:05.000"), fmt.Sprintf(format, args...))
}

func (a *agent) finish() { a.once.Do(func() { close(a.done) }) }

func (a *agent) accept(ctx context.Context, l net.Listener) {
	for {
		conn, err := l.Accept()
		if err != nil {
			return
		}
		if err := Serve(ctx, conn, a.handle); err != nil {
			a.logf("control connection: %v", err)
		}
		if a.finishing {
			a.finish()
			return
		}
	}
}

func (a *agent) handle(ctx context.Context, step string, args json.RawMessage) (any, error) {
	if step == "status" {
		return a.status(), nil
	}
	if a.failed != nil {
		return nil, fmt.Errorf("%w: %v", ErrPoisoned, a.failed)
	}
	if err := a.advance(step); err != nil {
		return nil, err
	}
	start := time.Now()
	a.logf("step %s", step)
	result, err := a.run(ctx, step, args)
	if err != nil {
		a.failed = fmt.Errorf("%s: %w", step, err)
		a.logf("step %s failed: %v", step, err)
		return nil, err
	}
	a.logf("step %s done in %.2fs", step, time.Since(start).Seconds())
	return result, nil
}

func (a *agent) advance(step string) error {
	for i, s := range Steps[a.cfg.Mode] {
		if s != step {
			continue
		}
		if i <= a.pos {
			return fmt.Errorf("step %q after %q is out of order", step, a.step)
		}
		a.pos, a.step = i, step
		return nil
	}
	return fmt.Errorf("no step %q in %s mode", step, a.cfg.Mode)
}

func (a *agent) run(ctx context.Context, step string, args json.RawMessage) (any, error) {
	switch a.cfg.Mode + "/" + step {
	case "snapshot/ipc":
		return a.ipcPrepare(ctx)
	case "restore/ipc":
		return a.ipcRestore(ctx, args)
	case "snapshot/cuda":
		return a.cudaCheckpoint(ctx)
	case "restore/cuda":
		return a.cudaRestore(ctx, args)
	case "snapshot/freeze":
		f, err := criu.Freeze(ctx, a.enginePID())
		if err != nil {
			return nil, err
		}
		a.frozen = f
		return FreezeResult{Tree: f.Tree, References: references(f.Tree, a.cfg.Watch)}, nil
	case "snapshot/dump":
		return a.dump(ctx)
	case "snapshot/fs":
		return a.archive(ctx)
	case "snapshot/finish":
		a.finishing = true
		return Final{Result: a.status()}, nil
	case "restore/install-fs":
		return a.install(ctx)
	case "restore/criu":
		return a.restore(ctx)
	case "restore/thaw":
		return nil, a.frozen.Thaw(ctx)
	}
	return nil, fmt.Errorf("no step %q in %s mode", step, a.cfg.Mode)
}

func (a *agent) enginePID() int {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.engine
}

func (a *agent) setExit(err error) {
	a.mu.Lock()
	a.exitErr = err
	a.mu.Unlock()
	close(a.exited)
}

func (a *agent) status() Status {
	a.mu.Lock()
	defer a.mu.Unlock()
	s := Status{Mode: a.cfg.Mode, Step: a.step, EnginePID: a.engine}
	select {
	case <-a.exited:
		s.Exit = fmt.Sprint(a.exitErr)
	default:
		s.Running = a.engine > 0
	}
	if a.failed != nil {
		s.Failed = a.failed.Error()
	}
	return s
}

// startEngine moves the PID allocator to the floor and starts the engine
// in its own session, under cuInterpose when ipc asks for it, logging into
// a carried tree so the restored process finds its log file where it left
// it.
func (a *agent) startEngine(ctx context.Context) error {
	var err error
	if a.interposed, a.ipcReason, err = a.chooseIPC(ctx); err != nil {
		return err
	}
	argv := a.cfg.Argv
	if a.interposed {
		argv = interpose.Wrap(argv)
	}
	if err := criu.ReservePIDFloor(); err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(a.cfg.Log), 0o755); err != nil {
		return err
	}
	log, err := os.OpenFile(a.cfg.Log, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return err
	}
	defer log.Close()
	cmd := exec.Command(argv[0], argv[1:]...)
	// No env in the config: the engine inherits the container's, which is
	// the image's environment plus the profile's.
	if len(a.cfg.Env) > 0 {
		cmd.Env = a.cfg.Env
	}
	cmd.Dir, cmd.Stdout, cmd.Stderr = a.cfg.Dir, log, log
	cmd.SysProcAttr = &syscall.SysProcAttr{Setsid: true}
	if err := cmd.Start(); err != nil {
		return err
	}
	a.mu.Lock()
	a.engine = cmd.Process.Pid
	a.mu.Unlock()
	go func() { a.setExit(cmd.Wait()) }()
	if cmd.Process.Pid < criu.PIDFloor {
		_ = cmd.Process.Kill()
		return fmt.Errorf("engine PID %d is below the floor %d", cmd.Process.Pid, criu.PIDFloor)
	}
	a.logf("engine started as PID %d (ipc: interposed %t, %s): %s", cmd.Process.Pid, a.interposed, a.ipcReason,
		strings.Join(argv, " "))
	return nil
}

// cudaCheckpoint locks every process of the running tree that holds CUDA
// state and moves its device memory into host memory, where the dump
// takes it.
func (a *agent) cudaCheckpoint(ctx context.Context) (any, error) {
	start := time.Now()
	tree, err := criu.VisibleTree(a.enginePID())
	if err != nil {
		return nil, err
	}
	owners, err := cuda.Owners(tree.PIDs())
	if err != nil {
		return nil, err
	}
	if len(owners) > 0 {
		if err := cuda.NewSession(a.cfg.CUDACheckpoint, owners).Checkpoint(ctx); err != nil {
			return nil, err
		}
	}
	return CUDAResult{Owners: owners, Seconds: time.Since(start).Seconds()}, nil
}

// cudaRestore puts the owners' device memory back on this container's
// GPUs, the parallel ones concurrently, and unlocks them.
func (a *agent) cudaRestore(ctx context.Context, raw json.RawMessage) (any, error) {
	var args CUDARestoreArgs
	if len(raw) > 0 {
		if err := json.Unmarshal(raw, &args); err != nil {
			return nil, err
		}
	}
	start := time.Now()
	if len(args.Owners) > 0 {
		s := cuda.NewSession(a.cfg.CUDACheckpoint, args.Owners)
		s.Parallel = args.Parallel
		if err := s.Restore(ctx, args.DeviceMap); err != nil {
			return nil, err
		}
	}
	return CUDAResult{Owners: args.Owners, Seconds: time.Since(start).Seconds()}, nil
}

// references counts the tree's open files and file mappings under each
// watched path.
func references(tree criu.Tree, watch []string) map[string]int {
	out := map[string]int{}
	for _, w := range watch {
		out[w] = 0
	}
	for _, pid := range tree.PIDs() {
		for _, target := range openPaths(pid) {
			for _, w := range watch {
				if target == w || strings.HasPrefix(target, w+"/") {
					out[w]++
				}
			}
		}
	}
	return out
}

// openPaths are the files a process maps or holds open.
func openPaths(pid int) []string {
	var paths []string
	if maps, err := os.ReadFile(fmt.Sprintf("/proc/%d/maps", pid)); err == nil {
		for _, line := range strings.Split(string(maps), "\n") {
			if i := strings.Index(line, " /"); i >= 0 {
				paths = append(paths, strings.TrimSpace(line[i:]))
			}
		}
	}
	dir := fmt.Sprintf("/proc/%d/fd", pid)
	fds, _ := os.ReadDir(dir)
	for _, fd := range fds {
		if target, err := os.Readlink(dir + "/" + fd.Name()); err == nil {
			paths = append(paths, target)
		}
	}
	return paths
}

// dump checkpoints the frozen tree. CRIU kills it afterwards.
func (a *agent) dump(ctx context.Context) (any, error) {
	if a.frozen == nil {
		return nil, errors.New("the tree is not frozen")
	}
	images, work := filepath.Join(a.cfg.Out, "criu"), filepath.Join(a.cfg.Out, "criu-work")
	if err := os.MkdirAll(images, 0o755); err != nil {
		return nil, err
	}
	if err := criu.PrepareWork(work); err != nil {
		return nil, err
	}
	start := time.Now()
	if err := criu.Dump(ctx, a.cfg.CRIU, a.enginePID(), images, work); err != nil {
		return nil, err
	}
	seconds := time.Since(start).Seconds()
	select {
	case <-a.exited:
	case <-time.After(30 * time.Second):
		return nil, errors.New("the engine outlived its dump")
	}
	_ = a.frozen.Close()
	a.frozen = nil
	return DumpResult{Seconds: seconds, Version: criuVersion(ctx, a.cfg.CRIU)}, nil
}

func criuVersion(ctx context.Context, program string) string {
	out, err := process.Capture(ctx, 10*time.Second, program, "--version")
	if err != nil {
		return ""
	}
	line, _, _ := strings.Cut(out.Stdout, "\n")
	return strings.TrimSpace(strings.TrimPrefix(line, "Version:"))
}

// archive takes the trees after the dump: the tree is gone, nothing writes,
// and the link_remap files CRIU left next to deleted-but-open files are
// ordinary members.
func (a *agent) archive(ctx context.Context) (any, error) {
	start := time.Now()
	dir := filepath.Join(a.cfg.Out, "fs")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, err
	}
	var res FSResult
	for _, t := range a.cfg.Trees {
		tar := filepath.Join(dir, t.Name+".tar")
		m, err := runtimefs.Snapshot(ctx, t.Path, tar, t.Path)
		if err != nil {
			return nil, fmt.Errorf("%s: %w", t.Path, err)
		}
		if err := m.Write(filepath.Join(dir, t.Name+".json")); err != nil {
			return nil, err
		}
		info, err := os.Stat(tar)
		if err != nil {
			return nil, err
		}
		res.Trees = append(res.Trees, FSTree{Name: t.Name, Path: t.Path, Entries: len(m.Entries), Bytes: info.Size()})
	}
	res.Seconds = time.Since(start).Seconds()
	return res, nil
}

func (a *agent) install(ctx context.Context) (any, error) {
	start := time.Now()
	var res FSResult
	for _, t := range a.cfg.Trees {
		m, err := runtimefs.Read(filepath.Join(a.cfg.FS, t.Name+".json"))
		if err != nil {
			return nil, err
		}
		if err := runtimefs.Install(ctx, filepath.Join(a.cfg.FS, t.Name+".tar"), t.Path, m); err != nil {
			return nil, fmt.Errorf("%s: %w", t.Path, err)
		}
		res.Trees = append(res.Trees, FSTree{Name: t.Name, Path: t.Path, Entries: len(m.Entries)})
	}
	res.Seconds = time.Since(start).Seconds()
	return res, nil
}

func (a *agent) restore(ctx context.Context) (any, error) {
	start := time.Now()
	f, err := criu.RestoreStopped(ctx, a.cfg.CRIU, a.cfg.Images, a.cfg.Work, a.cfg.Root, a.cfg.MaxID)
	if err != nil {
		return nil, err
	}
	a.frozen = f
	a.mu.Lock()
	a.engine = f.Tree.Root
	a.mu.Unlock()
	go a.watch(f.RootFD())
	return RestoreResult{Tree: f.Tree, Seconds: time.Since(start).Seconds()}, nil
}

// watch ends the agent when the restored root exits: its pidfd turns
// readable. PID 1 (docker-init or tini), its new parent, reaps it.
func (a *agent) watch(fd int) {
	fds := []unix.PollFd{{Fd: int32(fd), Events: unix.POLLIN}}
	for {
		_, err := unix.Poll(fds, -1)
		if err == nil || !errors.Is(err, unix.EINTR) {
			break
		}
	}
	a.setExit(errors.New("engine exited"))
	a.logf("restored engine exited")
	a.finish()
}

// shutdown asks the engine to stop and gives it Grace; the container's
// teardown kills whatever is left.
func (a *agent) shutdown() {
	pid := a.enginePID()
	select {
	case <-a.exited:
		return
	default:
	}
	if pid == 0 {
		return
	}
	a.logf("stopping engine %d", pid)
	if a.frozen != nil {
		_ = a.frozen.Thaw(context.Background())
		_ = a.frozen.Signal(unix.SIGTERM)
	} else {
		_ = syscall.Kill(pid, syscall.SIGTERM)
	}
	select {
	case <-a.exited:
	case <-time.After(Grace):
		a.logf("engine %d ignored SIGTERM", pid)
	}
}
