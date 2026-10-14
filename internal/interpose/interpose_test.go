package interpose

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestCommands(t *testing.T) {
	got := strings.Join(Wrap([]string{"vllm", "serve", "/model"}), " ")
	if got != "/opt/warmctl/ipc/cuinterpose-launch --library /opt/warmctl/ipc/libcuinterpose.so -- vllm serve /model" {
		t.Fatal(got)
	}
	got = strings.Join(Command("/work/cuinterpose-coordinator", "restore", GuestDir, []int{40001, 40007}), " ")
	if got != "/work/cuinterpose-coordinator --restore --checkpoint-dir /opt/warmctl/ipc --socket-dir /tmp "+
		"--process 40001 --process 40007" {
		t.Fatal(got)
	}
}

func TestParticipants(t *testing.T) {
	listening := map[string]bool{"/tmp/cuinterpose-40002.sock": true, "/tmp/cuinterpose-40003.sock": true}
	got := Participants([]int{40001, 40002, 40003}, func(p string) bool { return listening[p] })
	if len(got) != 2 || got[0] != 40002 || got[1] != 40003 {
		t.Fatal(got)
	}
	if Participants([]int{40001}, func(string) bool { return false }) != nil {
		t.Fatal("a process without a socket participates")
	}
}

func TestSupported(t *testing.T) {
	devices := []Device{
		{Index: 0, Name: "NVIDIA H100 80GB HBM3", ComputeCapability: "9.0", VMM: true, PosixFD: true, Multicast: true},
		{Index: 1, Name: "NVIDIA L40S", ComputeCapability: "8.9", VMM: true, PosixFD: true},
	}
	if err := Supported(devices); err != nil {
		t.Fatalf("PCIe GPUs with VMM refused: %v", err)
	}
	devices[1].PosixFD = false
	if err := Supported(devices); err == nil || !strings.Contains(err.Error(), "GPU 1 (NVIDIA L40S)") {
		t.Fatal(err)
	}
	if err := Supported(nil); err == nil {
		t.Fatal("no GPUs accepted")
	}
}

func TestDevices(t *testing.T) {
	old := Probe
	t.Cleanup(func() { Probe = old })
	Probe = func() ([]Device, error) { return []Device{{Index: 0, Name: "NVIDIA H100 80GB HBM3", VMM: true}}, nil }
	devices, err := Devices(context.Background())
	if err != nil || len(devices) != 1 || !devices[0].VMM {
		t.Fatal(devices, err)
	}
	Probe = func() ([]Device, error) { return nil, errors.New("CUDA error 100") }
	if _, err := Devices(context.Background()); err == nil || !strings.Contains(err.Error(), "CUDA error 100") {
		t.Fatal(err)
	}
	// A probe that never returns, like cuInit on a GPU that fell off the bus.
	block := make(chan struct{})
	t.Cleanup(func() { close(block) })
	Probe = func() ([]Device, error) { <-block; return nil, nil }
	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()
	if _, err := Devices(ctx); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("a blocked probe: %v", err)
	}
}

func TestCheck(t *testing.T) {
	dir := t.TempDir()
	for _, name := range Files[:3] {
		if err := os.WriteFile(filepath.Join(dir, name), nil, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	if err := Check(dir); err == nil || !strings.Contains(err.Error(), Coordinator) {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, Coordinator), nil, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := Check(dir); err != nil {
		t.Fatal(err)
	}
}
