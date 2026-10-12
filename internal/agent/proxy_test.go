package agent

import (
	"bufio"
	"context"
	"net"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestProxy(t *testing.T) {
	echo, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer echo.Close()
	go func() {
		for {
			c, err := echo.Accept()
			if err != nil {
				return
			}
			go func() {
				defer c.Close()
				line, _ := bufio.NewReader(c).ReadString('\n')
				c.Write([]byte("echo " + line))
			}()
		}
	}()
	dir, err := os.MkdirTemp("/tmp", "px")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(dir)
	sock := filepath.Join(dir, "inference.sock")
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() { done <- Proxy(ctx, sock, echo.Addr().String()) }()
	var c net.Conn
	for i := 0; i < 100; i++ {
		if c, err = net.Dial("unix", sock); err == nil {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
	if err != nil {
		t.Fatal(err)
	}
	c.Write([]byte("hi\n"))
	got, _ := bufio.NewReader(c).ReadString('\n')
	c.Close()
	if got != "echo hi\n" {
		t.Fatalf("proxy answered %q", got)
	}
	cancel()
	if err := <-done; err != nil {
		t.Fatal(err)
	}
}

func TestConfigValidate(t *testing.T) {
	snap := Config{Mode: ModeSnapshot, Control: "/state/control.sock", CRIU: "criu", Trees: DefaultTrees, Argv: []string{"vllm"}, Log: "/workspace/logs/engine.log", Out: "/state/out"}
	if err := snap.Validate(); err != nil {
		t.Fatal(err)
	}
	restore := Config{Mode: ModeRestore, Control: "/state/control.sock", CRIU: "criu", Trees: DefaultTrees, Images: "/capture/criu", FS: "/capture/fs", Work: "/state/work", Root: 32768, MaxID: 32800}
	if err := restore.Validate(); err != nil {
		t.Fatal(err)
	}
	bad := restore
	bad.Root = 100
	if bad.Validate() == nil {
		t.Fatal("restore root below the PID floor accepted")
	}
	bad = snap
	bad.Trees = []Tree{{Name: "x", Path: "relative"}}
	if bad.Validate() == nil {
		t.Fatal("relative tree accepted")
	}
	name := filepath.Join(t.TempDir(), "c.json")
	if err := snap.Write(name); err != nil {
		t.Fatal(err)
	}
	if back, err := ReadConfig(name); err != nil || back.Argv[0] != "vllm" {
		t.Fatalf("%+v %v", back, err)
	}
}
