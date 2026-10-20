package run

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/TensorRails/warmctl/v2/internal/agent"
	"github.com/TensorRails/warmctl/v2/internal/docker"
	"github.com/TensorRails/warmctl/v2/internal/vllm"
)

// Container is where the agent runs and the engine it drives: a Docker
// container under docker-init, or, inside a Kubernetes pod, a child of this
// process under the pod's init (Spawn).
type Container struct {
	Name string
	// IO is the directory holding the agent's sockets, as this process
	// sees it (mounted at /state in a container).
	IO string
	// Log is the engine's own log file, as this process sees it.
	Log    string
	Agent  *agent.Client
	Engine *vllm.Engine
	box    box
}

// box runs the agent: a container, or a child process.
type box interface {
	// exited is nil while the agent runs, else why it is gone.
	exited(ctx context.Context) error
	logs(ctx context.Context, tail int, w io.Writer) error
	wait(ctx context.Context) (int, error)
	stop(grace time.Duration) error
	remove()
}

// Launch creates and starts the container and connects to its agent. The
// engine logs into the workspace next to io.
func Launch(ctx context.Context, dk *docker.Client, spec docker.Spec, io string, ranks int) (*Container, error) {
	if err := ClearName(ctx, dk, spec.Name); err != nil {
		return nil, err
	}
	id, err := dk.Create(ctx, spec)
	if err != nil {
		return nil, err
	}
	c := &Container{Name: spec.Name, IO: io, Log: filepath.Join(filepath.Dir(io), "workspace", "logs", "engine.log"),
		box: &dockerBox{dk: dk, id: id, name: spec.Name}}
	if err := dk.Start(ctx, id); err != nil {
		c.Remove()
		return nil, err
	}
	return c.connect(ctx, ranks)
}

// connect waits for the agent's control socket; on failure the agent is
// removed.
func (c *Container) connect(ctx context.Context, ranks int) (*Container, error) {
	conn, err := c.dial(ctx, filepath.Join(c.IO, "control.sock"))
	if err != nil {
		c.Remove()
		return nil, err
	}
	c.Agent = agent.NewClient(conn)
	c.Engine = vllm.New(filepath.Join(c.IO, "inference.sock"), ranks)
	return c, nil
}

// ReplyThrough has the engine's RPC results come back as files under the
// IO directory, which the engine sees at guest: a data-parallel engine
// answers /collective_rpc for its first replica only.
func (c *Container) ReplyThrough(guest string) {
	c.Engine.ReplyThrough(filepath.Join(c.IO, "rpc"), guest+"/rpc")
}

func (c *Container) dial(ctx context.Context, sock string) (net.Conn, error) {
	deadline := time.Now().Add(2 * time.Minute)
	for {
		conn, err := net.Dial("unix", sock)
		if err == nil {
			return conn, nil
		}
		if err := c.Running(ctx); err != nil {
			return nil, err
		}
		if time.Now().After(deadline) {
			return nil, fmt.Errorf("agent socket %s: %w", sock, err)
		}
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-time.After(100 * time.Millisecond):
		}
	}
}

// Running fails when the agent has stopped, with its last log lines.
func (c *Container) Running(ctx context.Context) error {
	if err := c.box.exited(ctx); err != nil {
		return fmt.Errorf("%w: %s", err, c.Tail(20))
	}
	return nil
}

// Step runs one agent step.
func (c *Container) Step(ctx context.Context, step string, args, out any) error {
	return c.Agent.Call(ctx, step, args, out)
}

// EngineAlive fails when the engine process is gone.
func (c *Container) EngineAlive(ctx context.Context) error {
	var st agent.Status
	if err := c.Agent.Call(ctx, "status", nil, &st); err != nil {
		return err
	}
	if !st.Running {
		return fmt.Errorf("engine exited (%s); log tail:\n%s", st.Exit, c.EngineLog(40))
	}
	return nil
}

// Tail is the agent's last log lines.
func (c *Container) Tail(n int) string {
	var b strings.Builder
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	_ = c.box.logs(ctx, n, &b)
	return b.String()
}

// EngineLog is the last lines of the engine's own log in the workspace.
func (c *Container) EngineLog(n int) string {
	return tailFile(c.Log, n)
}

func tailFile(name string, n int) string {
	f, err := os.Open(name)
	if err != nil {
		return ""
	}
	defer f.Close()
	info, err := f.Stat()
	if err != nil {
		return ""
	}
	const window = 64 << 10
	start := max(info.Size()-window, 0)
	buf := make([]byte, info.Size()-start)
	if _, err := f.ReadAt(buf, start); err != nil && !errors.Is(err, io.EOF) {
		return ""
	}
	lines := strings.Split(strings.TrimRight(string(buf), "\n"), "\n")
	if len(lines) > n {
		lines = lines[len(lines)-n:]
	}
	return strings.Join(lines, "\n")
}

// SaveLogs writes the agent's whole log to name.
func (c *Container) SaveLogs(name string) {
	f, err := os.Create(name)
	if err != nil {
		return
	}
	defer f.Close()
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	_ = c.box.logs(ctx, 0, f)
}

// Wait waits for the agent to stop and returns its exit code.
func (c *Container) Wait(ctx context.Context) (int, error) {
	return c.box.wait(ctx)
}

// Stop asks the agent to stop and gives it grace.
func (c *Container) Stop(grace time.Duration) error {
	return c.box.stop(grace)
}

// Remove kills what is left and closes the agent connection.
func (c *Container) Remove() {
	if c.Agent != nil {
		_ = c.Agent.Close()
	}
	c.box.remove()
}

type dockerBox struct {
	dk   *docker.Client
	id   string
	name string
}

func (d *dockerBox) exited(ctx context.Context) error {
	st, err := d.dk.Inspect(ctx, d.id)
	if err != nil {
		return err
	}
	if st.State.Running {
		return nil
	}
	return fmt.Errorf("container %s exited %d (oom=%t)", d.name, st.State.ExitCode, st.State.OOMKilled)
}

func (d *dockerBox) logs(ctx context.Context, tail int, w io.Writer) error {
	return d.dk.Logs(ctx, d.id, tail, w)
}

func (d *dockerBox) wait(ctx context.Context) (int, error) { return d.dk.Wait(ctx, d.id) }

func (d *dockerBox) stop(grace time.Duration) error {
	ctx, cancel := context.WithTimeout(context.Background(), grace+30*time.Second)
	defer cancel()
	return d.dk.Stop(ctx, d.id, grace)
}

func (d *dockerBox) remove() {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	_ = d.dk.Remove(ctx, d.id)
}
