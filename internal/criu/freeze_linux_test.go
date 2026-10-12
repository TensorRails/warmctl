//go:build linux

package criu

import (
	"context"
	"os/exec"
	"syscall"
	"testing"
	"time"
)

// Freezing needs no namespace: a shell with two sleeping children is a tree.
func TestFreezeThaw(t *testing.T) {
	cmd := exec.Command("sh", "-c", "sleep 30 & sleep 30 & wait")
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	defer func() { _ = syscall.Kill(-cmd.Process.Pid, syscall.SIGKILL); _ = cmd.Wait() }()
	time.Sleep(200 * time.Millisecond)
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	f, err := Freeze(ctx, cmd.Process.Pid)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	if len(f.Tree.Processes) != 3 || f.Tree.Root != cmd.Process.Pid || f.Tree.MaxID() < cmd.Process.Pid {
		t.Fatalf("tree %+v", f.Tree)
	}
	if err := f.Stopped(); err != nil {
		t.Fatal(err)
	}
	if err := f.Thaw(ctx); err != nil {
		t.Fatal(err)
	}
	time.Sleep(50 * time.Millisecond)
	if err := f.Stopped(); err == nil {
		t.Fatal("tree still stopped after thaw")
	}
}
