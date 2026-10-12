package interpose

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
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
	devices, err := ParseDevices([]byte(`{"devices": [` +
		`{"index": 0, "name": "NVIDIA H100 80GB HBM3", "compute_capability": "9.0", "vmm": true, "posix_fd": true, "multicast": true},` +
		`{"index": 1, "name": "NVIDIA L40S", "compute_capability": "8.9", "vmm": true, "posix_fd": true, "multicast": false}]}`))
	if err != nil || len(devices) != 2 || devices[1].Multicast {
		t.Fatal(devices, err)
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
	if _, err := ParseDevices([]byte("Traceback")); err == nil {
		t.Fatal("garbage parsed")
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
