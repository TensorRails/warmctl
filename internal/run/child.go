package run

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"syscall"
	"time"
)

// Child is an agent started as a child of this process: how a restore runs
// inside a Kubernetes pod, where no container runtime is in reach. The
// pod's init (tini) is PID 1 and reaps what CRIU restores.
type Child struct {
	Name string
	Argv []string
	Env  []string
	// IO holds the agent's sockets and Log is the engine's log file.
	IO  string
	Log string
	// AgentLog keeps the agent's output, which is copied to Output too.
	AgentLog string
	Output   io.Writer
	Ranks    int
	// NewUTS starts the agent in a UTS namespace of its own, where it may
	// set the host name without touching the container's.
	NewUTS bool
}

// Spawn starts the agent and connects to it. The agent gets its own process
// group: only this process signals it, as only Docker signals a container.
func Spawn(ctx context.Context, s Child) (*Container, error) {
	if len(s.Argv) == 0 {
		return nil, errors.New("run: no agent command")
	}
	logFile, err := os.OpenFile(s.AgentLog, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return nil, err
	}
	// A pipe of our own rather than exec's copying goroutine: Wait must not
	// hang on a descendant (CRIU, the restored tree) that kept it open.
	r, w, err := os.Pipe()
	if err != nil {
		logFile.Close()
		return nil, err
	}
	cmd := exec.Command(s.Argv[0], s.Argv[1:]...)
	cmd.Env, cmd.Stdout, cmd.Stderr = s.Env, w, w
	if cmd.SysProcAttr, err = procAttr(s.NewUTS); err == nil {
		err = cmd.Start()
	}
	w.Close()
	if err != nil {
		r.Close()
		logFile.Close()
		return nil, err
	}
	out := s.Output
	if out == nil {
		out = io.Discard
	}
	go func() {
		_, _ = io.Copy(io.MultiWriter(logFile, out), r)
		r.Close()
		logFile.Close()
	}()
	b := &childBox{cmd: cmd, log: s.AgentLog, done: make(chan struct{})}
	go func() {
		b.err = cmd.Wait()
		close(b.done)
	}()
	c := &Container{Name: s.Name, IO: s.IO, Log: s.Log, box: b}
	return c.connect(ctx, s.Ranks)
}

type childBox struct {
	cmd  *exec.Cmd
	log  string
	done chan struct{}
	err  error
}

func (b *childBox) exited(context.Context) error {
	select {
	case <-b.done:
		return fmt.Errorf("agent exited %d (%v)", b.cmd.ProcessState.ExitCode(), b.err)
	default:
		return nil
	}
}

func (b *childBox) logs(_ context.Context, tail int, w io.Writer) error {
	if tail <= 0 {
		f, err := os.Open(b.log)
		if err != nil {
			return err
		}
		defer f.Close()
		_, err = io.Copy(w, f)
		return err
	}
	_, err := io.WriteString(w, tailFile(b.log, tail))
	return err
}

func (b *childBox) wait(ctx context.Context) (int, error) {
	select {
	case <-ctx.Done():
		return 0, ctx.Err()
	case <-b.done:
		return b.cmd.ProcessState.ExitCode(), nil
	}
}

// stop sends SIGTERM and, after grace, SIGKILL.
func (b *childBox) stop(grace time.Duration) error {
	_ = b.cmd.Process.Signal(syscall.SIGTERM)
	select {
	case <-b.done:
		return nil
	case <-time.After(grace):
	}
	_ = b.cmd.Process.Kill()
	select {
	case <-b.done:
		return nil
	case <-time.After(10 * time.Second):
		return fmt.Errorf("agent %d did not exit after SIGKILL", b.cmd.Process.Pid)
	}
}

func (b *childBox) remove() {
	select {
	case <-b.done:
	default:
		_ = b.stop(0)
	}
}
