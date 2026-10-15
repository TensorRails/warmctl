//go:build linux

package criu

import (
	"context"
	"errors"
	"fmt"
	"time"

	"golang.org/x/sys/unix"
)

// Frozen is a stopped tree held through pidfds, so a signal reaches the
// process it was meant for even if its PID has been reused.
type Frozen struct {
	Tree   Tree
	fds    map[int]int
	closed bool
}

// Freeze stops root and every descendant with SIGSTOP in rounds until no
// new process appears, then waits for every thread to be stopped. A process
// that forks between rounds is caught by the next one; a stopped process
// cannot fork at all.
func Freeze(ctx context.Context, root int) (*Frozen, error) {
	f := &Frozen{fds: map[int]int{}}
	fail := func(err error) (*Frozen, error) {
		_ = f.Close()
		return nil, err
	}
	for round := 0; round < 64; round++ {
		if err := ctx.Err(); err != nil {
			return fail(err)
		}
		tree, err := visibleTree(root)
		if err != nil {
			return fail(err)
		}
		added, err := f.stop(tree)
		if err != nil {
			return fail(err)
		}
		if added {
			continue
		}
		if err := waitStopped(ctx, tree); err != nil {
			return fail(err)
		}
		f.Tree = tree
		return f, nil
	}
	return fail(errors.New("process tree kept changing while freezing"))
}

// stop sends SIGSTOP to the processes of tree not stopped yet and reports
// whether there were any.
func (f *Frozen) stop(tree Tree) (bool, error) {
	added := false
	for _, p := range tree.Processes {
		if _, known := f.fds[p.PID]; known {
			continue
		}
		fd, err := unix.PidfdOpen(p.PID, 0)
		if errors.Is(err, unix.ESRCH) {
			continue
		}
		if err != nil {
			return false, fmt.Errorf("pidfd_open %d: %w", p.PID, err)
		}
		f.fds[p.PID] = fd
		if err := unix.PidfdSendSignal(fd, unix.SIGSTOP, nil, 0); err != nil && !errors.Is(err, unix.ESRCH) {
			return false, fmt.Errorf("stop %d: %w", p.PID, err)
		}
		added = true
	}
	return added, nil
}

func waitStopped(ctx context.Context, tree Tree) error {
	deadline := time.Now().Add(30 * time.Second)
	for {
		pending, err := runnable(tree)
		if err != nil {
			return err
		}
		if pending == 0 {
			return nil
		}
		if time.Now().After(deadline) {
			return fmt.Errorf("%d tasks did not stop", pending)
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(5 * time.Millisecond):
		}
	}
}

func runnable(tree Tree) (int, error) {
	n := 0
	for _, p := range tree.Processes {
		for _, tid := range p.Threads {
			state, _, err := taskState(p.PID, tid)
			if err != nil {
				continue
			}
			if state != "T" && state != "t" && state != "Z" && state != "X" {
				n++
			}
		}
	}
	return n, nil
}

// Thaw sends SIGCONT to every process. A process whose PID is gone is skipped.
func (f *Frozen) Thaw(ctx context.Context) error {
	for _, p := range f.Tree.Processes {
		if err := ctx.Err(); err != nil {
			return err
		}
		fd, ok := f.fds[p.PID]
		if !ok {
			continue
		}
		if err := unix.PidfdSendSignal(fd, unix.SIGCONT, nil, 0); err != nil && !errors.Is(err, unix.ESRCH) {
			return fmt.Errorf("continue %d: %w", p.PID, err)
		}
	}
	return nil
}

// RootFD is the pidfd of the tree's root; it becomes readable when the
// root exits.
func (f *Frozen) RootFD() int { return f.fds[f.Tree.Root] }

// Signal sends sig to the root.
func (f *Frozen) Signal(sig unix.Signal) error {
	return unix.PidfdSendSignal(f.fds[f.Tree.Root], sig, nil, 0)
}

// Close releases the pidfds. It never resumes or kills anything.
func (f *Frozen) Close() error {
	if f == nil || f.closed {
		return nil
	}
	f.closed = true
	var errs []error
	for _, fd := range f.fds {
		errs = append(errs, unix.Close(fd))
	}
	return errors.Join(errs...)
}
