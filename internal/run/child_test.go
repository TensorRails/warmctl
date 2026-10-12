package run

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"sync"
	"syscall"
	"testing"
	"time"

	"github.com/TensorRails/warmctl/v2/internal/agent"
)

// TestMain doubles as a fake agent: Spawn's tests start this binary with
// WARMCTL_FAKE_AGENT set to the directory of its control socket.
func TestMain(m *testing.M) {
	if dir := os.Getenv("WARMCTL_FAKE_AGENT"); dir != "" {
		os.Exit(fakeAgent(dir))
	}
	os.Exit(m.Run())
}

func fakeAgent(dir string) int {
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGTERM)
	defer stop()
	l, err := net.Listen("unix", filepath.Join(dir, "control.sock"))
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		return 1
	}
	fmt.Fprintln(os.Stderr, "fake agent ready")
	go func() {
		for {
			conn, err := l.Accept()
			if err != nil {
				return
			}
			go agent.Serve(ctx, conn, func(_ context.Context, step string, _ json.RawMessage) (any, error) {
				if step == "exit" {
					os.Exit(3)
				}
				return map[string]string{"step": step, "env": os.Getenv("FAKE_ENV")}, nil
			})
		}
	}()
	<-ctx.Done()
	fmt.Fprintln(os.Stderr, "fake agent stopping")
	return 0
}

type syncBuffer struct {
	mu sync.Mutex
	b  bytes.Buffer
}

func (s *syncBuffer) Write(p []byte) (int, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.b.Write(p)
}

func (s *syncBuffer) String() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.b.String()
}

func spawnFake(t *testing.T) (*Container, *syncBuffer, string) {
	t.Helper()
	dir, err := os.MkdirTemp("", "wc-child")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { os.RemoveAll(dir) })
	self, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	var out syncBuffer
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	c, err := Spawn(ctx, Child{Name: "fake", Argv: []string{self, "-test.run=^$"},
		Env: append(os.Environ(), "WARMCTL_FAKE_AGENT="+dir, "FAKE_ENV=pod"), IO: dir,
		Log: filepath.Join(dir, "engine.log"), AgentLog: filepath.Join(dir, "agent.log"), Output: &out, Ranks: 1})
	if err != nil {
		t.Fatal(err)
	}
	return c, &out, dir
}

func TestSpawnStepsAndStop(t *testing.T) {
	c, out, dir := spawnFake(t)
	defer c.Remove()
	ctx := context.Background()
	if err := c.Running(ctx); err != nil {
		t.Fatal(err)
	}
	var got map[string]string
	if err := c.Step(ctx, "install-fs", nil, &got); err != nil || got["step"] != "install-fs" || got["env"] != "pod" {
		t.Fatalf("step: %v %v", got, err)
	}
	if err := c.Stop(10 * time.Second); err != nil {
		t.Fatal(err)
	}
	if code, err := c.Wait(ctx); err != nil || code != 0 {
		t.Fatalf("exit %d %v", code, err)
	}
	if err := c.Running(ctx); err == nil || !strings.Contains(err.Error(), "fake agent stopping") {
		t.Fatalf("a stopped agent runs: %v", err)
	}
	// The agent's output goes to the pod's output and stays in its log.
	deadline := time.Now().Add(5 * time.Second)
	for !strings.Contains(out.String(), "fake agent stopping") && time.Now().Before(deadline) {
		time.Sleep(10 * time.Millisecond)
	}
	if !strings.Contains(out.String(), "fake agent ready") || !strings.Contains(out.String(), "fake agent stopping") {
		t.Errorf("output %q", out.String())
	}
	if tail := c.Tail(1); tail != "fake agent stopping" {
		t.Errorf("tail %q", tail)
	}
	saved := filepath.Join(dir, "saved.log")
	c.SaveLogs(saved)
	if raw, _ := os.ReadFile(saved); !strings.HasPrefix(string(raw), "fake agent ready\n") {
		t.Errorf("saved %q", raw)
	}
}

func TestSpawnAgentExit(t *testing.T) {
	c, _, _ := spawnFake(t)
	defer c.Remove()
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if err := c.Step(ctx, "exit", nil, nil); err == nil {
		t.Fatal("a step the agent died in succeeded")
	}
	if code, err := c.Wait(ctx); err != nil || code != 3 {
		t.Fatalf("exit %d %v", code, err)
	}
}

func TestSpawnNoAgent(t *testing.T) {
	dir := t.TempDir()
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	_, err := Spawn(ctx, Child{Argv: []string{"/bin/sh", "-c", "echo no socket here; exit 5"}, IO: dir,
		AgentLog: filepath.Join(dir, "agent.log")})
	if err == nil || !strings.Contains(err.Error(), "agent exited 5") || !strings.Contains(err.Error(), "no socket here") {
		t.Fatalf("an agent that never listened: %v", err)
	}
}

func TestFollow(t *testing.T) {
	name := filepath.Join(t.TempDir(), "engine.log")
	os.WriteFile(name, []byte("before the restore\n"), 0o644)
	var out syncBuffer
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() { Follow(ctx, name, &out); close(done) }()
	time.Sleep(100 * time.Millisecond)
	f, _ := os.OpenFile(name, os.O_APPEND|os.O_WRONLY, 0)
	f.WriteString("INFO serving\nhalf a li")
	deadline := time.Now().Add(5 * time.Second)
	for out.String() != "INFO serving\n" && time.Now().Before(deadline) {
		time.Sleep(20 * time.Millisecond)
	}
	f.WriteString("ne\n")
	f.Close()
	for out.String() != "INFO serving\nhalf a line\n" && time.Now().Before(deadline) {
		time.Sleep(20 * time.Millisecond)
	}
	cancel()
	<-done
	if out.String() != "INFO serving\nhalf a line\n" {
		t.Fatalf("followed %q", out.String())
	}
}
