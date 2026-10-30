package cuda

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"
	"testing"
	"time"
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
	c := NewCLI("/work/cuda-checkpoint")
	c.run = f.run
	s := NewSession(c, []int{10, 11, 12})
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
	c := NewCLI("x")
	c.run = f.run
	s := NewSession(c, []int{10, 11})
	if err := s.Checkpoint(context.Background()); err == nil || !strings.Contains(err.Error(), `11 is "locked"`) {
		t.Fatalf("wrong state accepted: %v", err)
	}
	f = &fakeHelper{state: map[int]string{10: "checkpointed", 11: "checkpointed"}, fail: "--pid 11 --action restore"}
	c.run, s.Parallel = f.run, []int{10, 11}
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

// overlapHelper records the most calls of each action in flight at once.
type overlapHelper struct {
	fakeHelper
	inflight, peak map[string]int
}

func (o *overlapHelper) run(ctx context.Context, args ...string) (string, error) {
	action := args[2]
	if len(args) > 3 {
		action = args[3]
	}
	if action == "restore" && args[1] < "13" {
		action = "restore-first"
	}
	o.mu.Lock()
	o.inflight[action]++
	o.peak[action] = max(o.peak[action], o.inflight[action])
	o.mu.Unlock()
	time.Sleep(20 * time.Millisecond)
	out, err := o.fakeHelper.run(ctx, args...)
	o.mu.Lock()
	o.inflight[action]--
	o.mu.Unlock()
	return out, err
}

func TestRestoreConcurrency(t *testing.T) {
	for _, concurrent := range []bool{false, true} {
		o := &overlapHelper{fakeHelper: fakeHelper{state: map[int]string{}}, inflight: map[string]int{}, peak: map[string]int{}}
		for pid := 10; pid < 16; pid++ {
			o.state[pid] = "checkpointed"
		}
		c := NewCLI("x")
		c.run = o.run
		s := NewSession(c, []int{10, 11, 12, 13, 14, 15})
		s.Parallel, s.Concurrent = []int{13, 14, 15}, concurrent
		if err := s.Restore(context.Background(), "GPU-a=GPU-b"); err != nil {
			t.Fatal(err)
		}
		for pid, st := range o.state {
			if st != "running" {
				t.Fatalf("concurrent %t: %d is %q", concurrent, pid, st)
			}
		}
		if o.peak["--get-state"] != 6 {
			t.Errorf("state checks ran %d at a time, want 6", o.peak["--get-state"])
		}
		// The parallel owners overlap in both modes; the owners without a
		// rank and the unlock only in the concurrent one. The serial ranks
		// still finish before the parallel ones start.
		first, all := map[bool]int{false: 1, true: 3}[concurrent], map[bool]int{false: 1, true: 6}[concurrent]
		if o.peak["restore-first"] != first || o.peak["restore"] != 3 || o.peak["unlock"] != all {
			t.Errorf("concurrent %t: peaks %v", concurrent, o.peak)
		}
		for _, k := range []string{"check", "restore_first", "restore_parallel", "unlock"} {
			if s.Phases[k] <= 0 {
				t.Errorf("concurrent %t: no %s phase in %v", concurrent, k, s.Phases)
			}
		}
	}
}

func TestParseDeviceMap(t *testing.T) {
	pairs, err := ParseDeviceMap("GPU-a=GPU-x,GPU-b=GPU-y")
	if err != nil || len(pairs) != 2 || pairs[1] != (Pair{"GPU-b", "GPU-y"}) {
		t.Fatal(pairs, err)
	}
	for _, bad := range []string{"", "GPU-a", "GPU-a=GPU-x,GPU-a=GPU-y", "a=b"} {
		if _, err := ParseDeviceMap(bad); err == nil {
			t.Errorf("%q accepted", bad)
		}
	}
	if b, err := uuidBytes("GPU-56ce119a-6848-3ae0-cf4f-8c1564626fc6"); err != nil || b[0] != 0x56 || b[15] != 0xc6 {
		t.Fatal(b, err)
	}
	for _, bad := range []string{"56ce119a-6848-3ae0-cf4f-8c1564626fc6", "GPU-56ce", "GPU-zzce119a-6848-3ae0-cf4f-8c1564626fc6"} {
		if _, err := uuidBytes(bad); err == nil {
			t.Errorf("%q accepted", bad)
		}
	}
}

func TestSelect(t *testing.T) {
	if b, _, err := Select(BackendCLI, "/x/cuda-checkpoint"); err != nil || b.Name() != BackendCLI {
		t.Fatal(b, err)
	}
	if _, _, err := Select("fast", ""); err == nil {
		t.Fatal("unknown backend accepted")
	}
}
