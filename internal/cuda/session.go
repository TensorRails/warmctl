package cuda

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"strings"
	"sync"
	"time"
)

// CallTimeout bounds one checkpoint call; LockTimeoutMS is how long a lock
// may wait for in-flight CUDA work.
const (
	CallTimeout   = 90 * time.Second
	LockTimeoutMS = 60000
)

// Pair maps a donor GPU to a receiver GPU.
type Pair struct {
	Source string `json:"source"`
	Target string `json:"target"`
}

// DeviceMap is cuda-checkpoint's --device-map: every GPU the processes saw,
// donor UUID to receiver UUID, in rank order.
func DeviceMap(pairs []Pair) (string, error) {
	seen := map[string]bool{}
	parts := make([]string, 0, len(pairs))
	for _, p := range pairs {
		uuids := strings.HasPrefix(p.Source, "GPU-") && strings.HasPrefix(p.Target, "GPU-")
		if !uuids || seen["s"+p.Source] || seen["t"+p.Target] {
			return "", fmt.Errorf("cuda: device map needs distinct GPU UUIDs, got %+v", p)
		}
		seen["s"+p.Source], seen["t"+p.Target] = true, true
		parts = append(parts, p.Source+"="+p.Target)
	}
	if len(parts) == 0 {
		return "", errors.New("cuda: empty device map")
	}
	return strings.Join(parts, ","), nil
}

// Session moves the CUDA state of a set of processes through a Backend.
// Each phase is a barrier: every owner finishes it before any starts the
// next. A failure is final; the caller removes the container.
type Session struct {
	Backend Backend
	Owners  []int
	// Parallel owners are restored concurrently, after the others: the
	// workers, when no collective keeps peer memory mapped between them.
	Parallel []int
	// Concurrent also restores the other owners concurrently, as their own
	// barrier before the parallel ones, and unlocks every owner at once.
	Concurrent bool
	// Phases holds the seconds each phase of the last Restore took.
	Phases map[string]float64
}

// NewSession moves the owners' state through b.
func NewSession(b Backend, owners []int) *Session {
	return &Session{Backend: b, Owners: append([]int(nil), owners...)}
}

// Checkpoint locks every owner and moves its device memory to the host.
func (s *Session) Checkpoint(ctx context.Context) error {
	if err := s.expect(ctx, "running"); err != nil {
		return err
	}
	lock := func(ctx context.Context, pid int) error { return s.Backend.Lock(ctx, pid, LockTimeoutMS) }
	if err := s.each(ctx, s.Owners, false, lock); err != nil {
		return err
	}
	if err := s.expect(ctx, "locked"); err != nil {
		return err
	}
	if err := s.each(ctx, s.Owners, false, s.Backend.Checkpoint); err != nil {
		return err
	}
	return s.expect(ctx, "checkpointed")
}

// Restore puts every owner's device memory back on the mapped GPUs (a
// DeviceMap) and unlocks it.
func (s *Session) Restore(ctx context.Context, deviceMap string) error {
	pairs, err := ParseDeviceMap(deviceMap)
	if err != nil {
		return err
	}
	s.Phases = map[string]float64{}
	if err := s.timed("check", func() error { return s.expect(ctx, "checkpointed") }); err != nil {
		return err
	}
	var first, rest []int
	for _, pid := range s.Owners {
		if slices.Contains(s.Parallel, pid) {
			rest = append(rest, pid)
		} else {
			first = append(first, pid)
		}
	}
	restore := func(ctx context.Context, pid int) error { return s.Backend.Restore(ctx, pid, pairs) }
	if err := s.timed("restore_first", func() error { return s.each(ctx, first, s.Concurrent, restore) }); err != nil {
		return err
	}
	if err := s.timed("restore_parallel", func() error { return s.each(ctx, rest, true, restore) }); err != nil {
		return err
	}
	if err := s.timed("check", func() error { return s.expect(ctx, "locked") }); err != nil {
		return err
	}
	unlock := func() error { return s.each(ctx, s.Owners, s.Concurrent, s.Backend.Unlock) }
	if err := s.timed("unlock", unlock); err != nil {
		return err
	}
	return s.timed("check", func() error { return s.expect(ctx, "running") })
}

// timed adds the seconds fn takes to the named phase.
func (s *Session) timed(phase string, fn func() error) error {
	start := time.Now()
	err := fn()
	if s.Phases != nil {
		s.Phases[phase] += time.Since(start).Seconds()
	}
	return err
}

// each runs op for every pid, one after another or all at once; concurrent
// calls are all joined, and one failure fails the phase.
func (s *Session) each(ctx context.Context, pids []int, concurrent bool, op func(context.Context, int) error) error {
	if !concurrent {
		for _, pid := range pids {
			if err := op(ctx, pid); err != nil {
				return err
			}
		}
		return nil
	}
	errs := make([]error, len(pids))
	var wg sync.WaitGroup
	for i, pid := range pids {
		wg.Add(1)
		go func(i, pid int) {
			defer wg.Done()
			errs[i] = op(ctx, pid)
		}(i, pid)
	}
	wg.Wait()
	return errors.Join(errs...)
}

// expect checks every owner's state; the queries only read, so they run
// concurrently.
func (s *Session) expect(ctx context.Context, state string) error {
	return s.each(ctx, s.Owners, true, func(ctx context.Context, pid int) error {
		got, err := s.Backend.State(ctx, pid)
		if err == nil && got != state {
			err = fmt.Errorf("cuda: process %d is %q, expected %q", pid, got, state)
		}
		return err
	})
}
