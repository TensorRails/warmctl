//go:build linux

package criu

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

// ErrNotIsolated is returned when a namespace-wide operation would touch the
// host's PID namespace.
var ErrNotIsolated = errors.New("criu: refusing to move the PID allocator outside a container")

// isolated reports a private PID namespace: a container's init (docker-init
// or this binary) is PID 1, not the host's systemd, and our own PID is small.
func isolated() error {
	comm, err := os.ReadFile("/proc/1/comm")
	if err != nil {
		return err
	}
	switch strings.TrimSpace(string(comm)) {
	case "systemd", "init":
		return ErrNotIsolated
	}
	if os.Getpid() >= PIDFloor {
		return ErrNotIsolated
	}
	return nil
}

const lastPID = "/proc/sys/kernel/ns_last_pid"

// ReservePIDFloor moves the namespace's PID allocator just below PIDFloor,
// so everything started after it gets a PID a receiver can honour.
func ReservePIDFloor() error {
	if err := isolated(); err != nil {
		return err
	}
	n, err := readInt(lastPID)
	if err != nil {
		return err
	}
	if n >= PIDFloor {
		return fmt.Errorf("PID allocator already past the floor: %d", n)
	}
	return os.WriteFile(lastPID, []byte(strconv.Itoa(PIDFloor-1)), 0o600)
}

// PrepareRestorePIDs checks that nothing in the receiver namespace occupies
// a PID the dump needs and moves the allocator above the dumped tree, so
// CRIU's helpers never take a PID the tree wants back.
func PrepareRestorePIDs(maxID int) error {
	if err := isolated(); err != nil {
		return err
	}
	occupied, err := occupiedIDs()
	if err != nil {
		return err
	}
	for _, id := range occupied {
		if id >= PIDFloor {
			return fmt.Errorf("receiver is not fresh: task %d sits in the snapshot's PID range", id)
		}
	}
	limit, err := readInt("/proc/sys/kernel/pid_max")
	if err != nil {
		return err
	}
	const gap = 1024
	if maxID < PIDFloor || maxID+gap >= limit {
		return fmt.Errorf("snapshot PIDs up to %d do not fit pid_max %d", maxID, limit)
	}
	return os.WriteFile(lastPID, []byte(strconv.Itoa(maxID+gap)), 0o600)
}

func occupiedIDs() ([]int, error) {
	entries, err := os.ReadDir("/proc")
	if err != nil {
		return nil, err
	}
	var ids []int
	for _, e := range entries {
		if _, err := strconv.Atoi(e.Name()); err != nil {
			continue
		}
		tasks, err := os.ReadDir(filepath.Join("/proc", e.Name(), "task"))
		if err != nil {
			continue
		}
		for _, t := range tasks {
			if tid, err := strconv.Atoi(t.Name()); err == nil {
				ids = append(ids, tid)
			}
		}
	}
	return ids, nil
}

func readInt(name string) (int, error) {
	raw, err := os.ReadFile(name)
	if err != nil {
		return 0, err
	}
	return strconv.Atoi(strings.TrimSpace(string(raw)))
}
