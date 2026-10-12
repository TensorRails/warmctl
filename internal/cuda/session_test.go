package cuda

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"
	"testing"
)

// fakeHelper keeps a state per PID like cuda-checkpoint does.
type fakeHelper struct {
	mu    sync.Mutex
	state map[int]string
	calls []string
	fail  string
}

func (f *fakeHelper) run(_ context.Context, args ...string) (string, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	var pid int
	fmt.Sscan(args[1], &pid)
	line := strings.Join(args, " ")
	f.calls = append(f.calls, line)
	if f.fail != "" && strings.Contains(line, f.fail) {
		return "", errors.New("injected")
	}
	if args[2] == "--get-state" {
		return f.state[pid], nil
	}
	next := map[string]string{"lock": "locked", "checkpoint": "checkpointed", "restore": "locked", "unlock": "running"}
	f.state[pid] = next[args[3]]
	return "", nil
}

func TestCheckpointRestore(t *testing.T) {
	f := &fakeHelper{state: map[int]string{10: "running", 11: "running", 12: "running"}}
	s := NewSession("/work/cuda-checkpoint", []int{10, 11, 12})
	s.run = f.run
	if err := s.Checkpoint(context.Background()); err != nil {
		t.Fatal(err)
	}
	if f.state[11] != "checkpointed" {
		t.Fatalf("%v", f.state)
	}
	m, err := DeviceMap([]Pair{{"GPU-a", "GPU-x"}, {"GPU-b", "GPU-y"}})
	if err != nil || m != "GPU-a=GPU-x,GPU-b=GPU-y" {
		t.Fatal(m, err)
	}
	s.Parallel = []int{11, 12}
	f.calls = nil
	if err := s.Restore(context.Background(), m); err != nil {
		t.Fatal(err)
	}
	restores := 0
	for i, c := range f.calls {
		if strings.Contains(c, "--action restore") {
			restores++
			if restores == 1 && !strings.HasPrefix(c, "--pid 10 ") {
				t.Errorf("serial owner not restored first: %v", f.calls)
			}
			if !strings.Contains(c, "--device-map "+m) {
				t.Errorf("call %d lacks the device map: %s", i, c)
			}
		}
	}
	if restores != 3 || f.state[12] != "running" {
		t.Fatalf("%d restores, state %v", restores, f.state)
	}
	if !strings.Contains(strings.Join(f.calls, "\n"), "--pid 10 --action unlock") {
		t.Fatal("no unlock")
	}
}

func TestFailures(t *testing.T) {
	f := &fakeHelper{state: map[int]string{10: "running", 11: "locked"}}
	s := NewSession("x", []int{10, 11})
	s.run = f.run
	if err := s.Checkpoint(context.Background()); err == nil || !strings.Contains(err.Error(), `11 is "locked"`) {
		t.Fatalf("wrong state accepted: %v", err)
	}
	f = &fakeHelper{state: map[int]string{10: "checkpointed", 11: "checkpointed"}, fail: "--pid 11 --action restore"}
	s.run, s.Parallel = f.run, []int{10, 11}
	if err := s.Restore(context.Background(), "GPU-a=GPU-b"); err == nil {
		t.Fatal("failed restore accepted")
	}
	if strings.Contains(strings.Join(f.calls, "\n"), "unlock") {
		t.Fatal("unlocked after a failed restore")
	}
	if _, err := DeviceMap([]Pair{{"GPU-a", "GPU-x"}, {"GPU-a", "GPU-y"}}); err == nil {
		t.Fatal("duplicate source accepted")
	}
}
