package cuda

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/TensorRails/warmctl/v2/internal/process"
)

// CallTimeout bounds one cuda-checkpoint call; LockTimeoutMS is how long a
// lock may wait for in-flight CUDA work.
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

// Session moves the CUDA state of a set of processes with cuda-checkpoint.
// Each phase is a barrier: every owner finishes it before any starts the
// next. A failure is final; the caller removes the container.
type Session struct {
	Helper string
	Owners []int
	// Parallel owners are restored concurrently, after the others: the
	// workers, when no collective keeps peer memory mapped between them.
	Parallel []int
	run      func(ctx context.Context, args ...string) (string, error)
}

// NewSession uses the helper binary at path.
func NewSession(helper string, owners []int) *Session {
	s := &Session{Helper: helper, Owners: append([]int(nil), owners...)}
	s.run = func(ctx context.Context, args ...string) (string, error) {
		out, err := process.Capture(ctx, CallTimeout, s.Helper, args...)
		if err != nil {
			return "", fmt.Errorf("%w: %s", err, out.Stderr)
		}
		return strings.TrimSpace(out.Stdout), nil
	}
	return s
}

// Checkpoint locks every owner and moves its device memory to the host.
func (s *Session) Checkpoint(ctx context.Context) error {
	if err := s.expect(ctx, "running"); err != nil {
		return err
	}
	if err := s.serial(ctx, s.Owners, "--action", "lock", "--timeout", strconv.Itoa(LockTimeoutMS)); err != nil {
		return err
	}
	if err := s.expect(ctx, "locked"); err != nil {
		return err
	}
	if err := s.serial(ctx, s.Owners, "--action", "checkpoint"); err != nil {
		return err
	}
	return s.expect(ctx, "checkpointed")
}

// Restore puts every owner's device memory back on the mapped GPUs and
// unlocks it.
func (s *Session) Restore(ctx context.Context, deviceMap string) error {
	if err := s.expect(ctx, "checkpointed"); err != nil {
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
	args := []string{"--action", "restore", "--device-map", deviceMap}
	if err := s.serial(ctx, first, args...); err != nil {
		return err
	}
	if err := s.concurrent(ctx, rest, args...); err != nil {
		return err
	}
	if err := s.expect(ctx, "locked"); err != nil {
		return err
	}
	if err := s.serial(ctx, s.Owners, "--action", "unlock"); err != nil {
		return err
	}
	return s.expect(ctx, "running")
}

func (s *Session) call(ctx context.Context, pid int, args ...string) (string, error) {
	out, err := s.run(ctx, append([]string{"--pid", strconv.Itoa(pid)}, args...)...)
	if err != nil {
		return "", fmt.Errorf("cuda-checkpoint %s on %d: %w", strings.Join(args, " "), pid, err)
	}
	return out, nil
}

func (s *Session) serial(ctx context.Context, pids []int, args ...string) error {
	for _, pid := range pids {
		if _, err := s.call(ctx, pid, args...); err != nil {
			return err
		}
	}
	return nil
}

// concurrent joins every launched call; one failure fails the phase.
func (s *Session) concurrent(ctx context.Context, pids []int, args ...string) error {
	errs := make([]error, len(pids))
	var wg sync.WaitGroup
	for i, pid := range pids {
		wg.Add(1)
		go func(i, pid int) {
			defer wg.Done()
			_, errs[i] = s.call(ctx, pid, args...)
		}(i, pid)
	}
	wg.Wait()
	return errors.Join(errs...)
}

func (s *Session) expect(ctx context.Context, state string) error {
	for _, pid := range s.Owners {
		got, err := s.call(ctx, pid, "--get-state")
		if err != nil {
			return err
		}
		if got != state {
			return fmt.Errorf("cuda: process %d is %q, expected %q", pid, got, state)
		}
	}
	return nil
}
