package rocm

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestHoldersAndCatches(t *testing.T) {
	hip := "/opt/venv/lib/python3.12/site-packages/_rocm_sdk_core/lib/libamdhip64.so.7"
	fakeProc(t, map[int]struct{ maps, fds []string }{
		20: {maps: []string{hip, lib}, fds: []string{"/dev/kfd", "/dev/dri/renderD136"}}, // the API server
		21: {maps: []string{hip, lib}},                                                     // a checkpointed worker
		22: {maps: []string{"/usr/bin/python3.12"}, fds: []string{"pipe:[7]"}},
	})
	got, err := Holders([]int{20, 21, 22, 99})
	if err != nil || len(got) != 1 || got[0] != 20 {
		t.Fatalf("holders %v, %v", got, err)
	}
	// SigCgt with SIGRTMIN+6 (40) caught; the default action of a real-time
	// signal would end a process that has no handler.
	status := "Name:\tpython3\nSigBlk:\t0000000000000000\nSigCgt:\t0000008000004a02\n"
	if err := os.WriteFile(filepath.Join(Proc, "20", "status"), []byte(status), 0o644); err != nil {
		t.Fatal(err)
	}
	if !Catches(20, CheckpointSignal) || Catches(20, RestoreSignal) || Catches(21, CheckpointSignal) {
		t.Fatal("SigCgt read wrong")
	}
}

func TestWaitForTheStatus(t *testing.T) {
	fakeProc(t, map[int]struct{ maps, fds []string }{30: {}})
	dir := t.TempDir()
	name := filepath.Join(dir, "40000")
	old := SignalTimeout
	SignalTimeout = 300 * time.Millisecond
	t.Cleanup(func() { SignalTimeout = old })
	ctx := context.Background()

	write := func(s string) {
		if err := os.WriteFile(name, []byte(s+"\n"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	write("checkpointed")
	if err := wait(ctx, name, 30, "checkpointed", false); err != nil {
		t.Fatal(err)
	}
	write("skipped")
	if err := wait(ctx, name, 30, "restored", true); err != nil {
		t.Fatalf("a worker skips the signal's restore: %v", err)
	}
	if err := wait(ctx, name, 30, "checkpointed", false); err == nil {
		t.Fatal("skipped is no checkpoint")
	}
	write("error: REFUSE: 1 unknown KFD ioctl(s)")
	if err := wait(ctx, name, 30, "checkpointed", false); err == nil || !strings.Contains(err.Error(), "unknown KFD ioctl") {
		t.Fatalf("the refusal is carried: %v", err)
	}
	if err := os.Remove(name); err != nil {
		t.Fatal(err)
	}
	if err := wait(ctx, name, 30, "checkpointed", false); err == nil || !strings.Contains(err.Error(), "did not report") {
		t.Fatalf("no answer in time: %v", err)
	}
	if err := wait(ctx, name, 31, "checkpointed", false); err == nil || !strings.Contains(err.Error(), "exited") {
		t.Fatalf("a process that is gone: %v", err)
	}
}
