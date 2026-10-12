//go:build linux

package criu

import (
	"errors"
	"fmt"
	"os"
	"sort"
	"strconv"
	"strings"
)

func statFields(name string) ([]string, error) {
	raw, err := os.ReadFile(name)
	if err != nil {
		return nil, err
	}
	end := strings.LastIndex(string(raw), ")")
	if end < 0 {
		return nil, errors.New("malformed " + name)
	}
	fields := strings.Fields(string(raw[end+1:]))
	if len(fields) < 20 {
		return nil, errors.New("short " + name)
	}
	return fields, nil
}

// taskState is the state letter and start ticks of one task.
func taskState(pid, tid int) (state, start string, err error) {
	fields, err := statFields(fmt.Sprintf("/proc/%d/task/%d/stat", pid, tid))
	if err != nil {
		return "", "", err
	}
	return fields[0], fields[19], nil
}

func readProcess(pid int) (Process, error) {
	fields, err := statFields(fmt.Sprintf("/proc/%d/stat", pid))
	if err != nil {
		return Process{}, err
	}
	parent, err := strconv.Atoi(fields[1])
	if err != nil {
		return Process{}, err
	}
	tasks, err := os.ReadDir(fmt.Sprintf("/proc/%d/task", pid))
	if err != nil {
		return Process{}, err
	}
	p := Process{PID: pid, Parent: parent, Start: fields[19]}
	for _, task := range tasks {
		tid, err := strconv.Atoi(task.Name())
		if err != nil {
			continue
		}
		p.Threads = append(p.Threads, tid)
	}
	sort.Ints(p.Threads)
	return p, nil
}

// visibleTree is root and its descendants as /proc shows them now.
func visibleTree(root int) (Tree, error) {
	entries, err := os.ReadDir("/proc")
	if err != nil {
		return Tree{}, err
	}
	all := map[int]Process{}
	children := map[int][]int{}
	for _, e := range entries {
		pid, err := strconv.Atoi(e.Name())
		if err != nil {
			continue
		}
		p, err := readProcess(pid)
		if errors.Is(err, os.ErrNotExist) {
			continue
		}
		if err != nil {
			return Tree{}, err
		}
		all[pid] = p
		children[p.Parent] = append(children[p.Parent], pid)
	}
	if _, ok := all[root]; !ok {
		return Tree{}, fmt.Errorf("process %d is gone", root)
	}
	t := Tree{Root: root}
	queue := []int{root}
	seen := map[int]bool{}
	for len(queue) > 0 {
		pid := queue[0]
		queue = queue[1:]
		if seen[pid] {
			continue
		}
		seen[pid] = true
		t.Processes = append(t.Processes, all[pid])
		queue = append(queue, children[pid]...)
	}
	return t.sorted(), nil
}

// VisibleTree is root and its descendants as /proc shows them now.
func VisibleTree(root int) (Tree, error) { return visibleTree(root) }
