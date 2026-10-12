//go:build linux

package criu

import (
	"context"
	"fmt"

	"golang.org/x/sys/unix"
)

// RestoreStopped runs as PID 1 of a fresh namespace: it moves the PID
// allocator above the dumped tree, restores the images with every task left
// stopped, checks the root PID is the expected one and holds the tree
// through pidfds until Thaw.
func RestoreStopped(ctx context.Context, program, images, work string, expectedRoot, maxID int) (*Frozen, error) {
	if err := PrepareRestorePIDs(maxID); err != nil {
		return nil, err
	}
	if err := PrepareWork(work); err != nil {
		return nil, err
	}
	pid, err := Restore(ctx, program, images, work)
	if err != nil {
		return nil, err
	}
	if pid != expectedRoot {
		return nil, fmt.Errorf("criu restored root %d, snapshot has %d", pid, expectedRoot)
	}
	tree, err := visibleTree(pid)
	if err != nil {
		return nil, err
	}
	f := &Frozen{Tree: tree, fds: map[int]int{}}
	for _, p := range tree.Processes {
		fd, err := unix.PidfdOpen(p.PID, 0)
		if err != nil {
			_ = f.Close()
			return nil, fmt.Errorf("pidfd_open %d: %w", p.PID, err)
		}
		f.fds[p.PID] = fd
		for _, tid := range p.Threads {
			state, _, err := taskState(p.PID, tid)
			if err != nil || (state != "T" && state != "t" && state != "Z") {
				_ = f.Close()
				return nil, fmt.Errorf("restored task %d/%d is %q, not stopped (%v)", p.PID, tid, state, err)
			}
		}
	}
	return f, nil
}
