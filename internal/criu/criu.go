// Package criu drives CRIU around a process tree: freeze it with pidfds,
// dump it with link remaps and an empty plugin directory (GPU state is
// cuda-checkpoint's business, never CRIU's), restore it left stopped in a
// fresh PID namespace, thaw it. The namespace's PID allocator is moved so a
// restored PID never collides with the receiver's own processes.
package criu

import (
	"fmt"
	"path/filepath"
	"sort"
)

// PIDFloor is the first PID of a snapshotted tree. The donor namespace is
// moved there before the engine starts; a receiver keeps its own processes
// (PID 1, CRIU, helpers) below it.
const PIDFloor = 32768

// GhostLimit is how large a deleted-but-open file CRIU may carry inside its
// images instead of as a link remap next to the original.
const GhostLimit = 64 << 20

// Process is one process of a tree with its threads and start ticks.
type Process struct {
	PID     int    `json:"pid"`
	Parent  int    `json:"parent"`
	Threads []int  `json:"threads"`
	Start   string `json:"start"`
}

// Tree is a process and its descendants, root first, then by PID.
type Tree struct {
	Root      int       `json:"root"`
	Processes []Process `json:"processes"`
}

// MaxID is the highest PID or TID in the tree.
func (t Tree) MaxID() int {
	m := 0
	for _, p := range t.Processes {
		m = max(m, p.PID)
		for _, tid := range p.Threads {
			m = max(m, tid)
		}
	}
	return m
}

// PIDs lists the processes in tree order.
func (t Tree) PIDs() []int {
	out := make([]int, 0, len(t.Processes))
	for _, p := range t.Processes {
		out = append(out, p.PID)
	}
	return out
}

func (t Tree) sorted() Tree {
	sort.SliceStable(t.Processes, func(i, j int) bool {
		if t.Processes[i].PID == t.Root || t.Processes[j].PID == t.Root {
			return t.Processes[i].PID == t.Root && t.Processes[j].PID != t.Root
		}
		return t.Processes[i].PID < t.Processes[j].PID
	})
	return t
}

// DumpArgs is the qualified dump invocation: shell job, file locks,
// external unix sockets, established TCP, link remaps for deleted files,
// and a plugin directory that is empty on purpose.
func DumpArgs(tree int, images, work string) []string {
	return []string{
		"dump", "-v4", "--tree", fmt.Sprint(tree),
		"--images-dir", images, "--work-dir", work, "--log-file", filepath.Join(work, "dump.log"),
		"--shell-job", "--file-locks", "--ext-unix-sk", "--tcp-established",
		"--link-remap", "--ghost-limit", fmt.Sprint(GhostLimit),
		"-L", filepath.Join(work, "plugins"),
	}
}

// RestoreArgs restores detached and leaves every task stopped, so the
// receiver can re-attach the GPUs before anything runs.
func RestoreArgs(images, work string) []string {
	return []string{
		"restore", "-v4", "--images-dir", images, "--restore-detached", "--leave-stopped",
		"--work-dir", work, "--pidfile", filepath.Join(work, "root.pid"), "--log-file", filepath.Join(work, "restore.log"),
		"--shell-job", "--file-locks", "--ext-unix-sk", "--tcp-established",
		"--link-remap", "-L", filepath.Join(work, "plugins"),
	}
}
