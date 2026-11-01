// Package dmtcp checkpoints and restarts an engine's process tree with DMTCP,
// entirely in user space: the backend for containers that cannot run CRIU
// (no CAP_SYS_ADMIN or CAP_CHECKPOINT_RESTORE, a read-only /proc/sys,
// unshare refused by seccomp), such as RunPod pods.
//
// DMTCP must start the engine (dmtcp_launch preloads its wrappers), so the
// whole life of a snapshotted engine runs under one coordinator. The fdkeep
// plugin repairs what DMTCP loses of the CUDA driver's helper fds, thread
// names, listening sockets and epoll watches (docs/dmtcp.md).
//
// Inside the tree DMTCP keeps virtual PIDs (40000 and up); outside, after a
// restart, the processes have new real PIDs. The coordinator's list maps one
// to the other.
package dmtcp

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/TensorRails/warmctl/v2/internal/process"
)

// Where the runtime image installs DMTCP and fdkeep. Donor and receiver must
// run this very build, so a restore never takes another DMTCP from PATH.
const (
	DefaultDir    = "/opt/warmctl/dmtcp/bin"
	DefaultPlugin = "/opt/warmctl/dmtcp/lib/libdmtcp_fdkeep.so"
)

// Env is what an engine needs to run under DMTCP:
//   - fork with live OpenBLAS/OpenMP thread pools hangs the child (torch's
//     import runs ldconfig in a subprocess), so the pools get one thread;
//   - vLLM's spawn hands the engine core a pipe DMTCP turned into a
//     socketpair and does not drain in the child, so the checkpoint never
//     ends: the engine core is forked instead;
//   - libuv must not batch epoll_ctl through io_uring, which DMTCP cannot see;
//   - UCX (NIXL's transport, in the vLLM image) must not hook mmap and shmat:
//     its hooks call the originals past DMTCP's wrappers.
var Env = []string{
	"OMP_NUM_THREADS=1", "OPENBLAS_NUM_THREADS=1", "MKL_NUM_THREADS=1",
	"VLLM_WORKER_MULTIPROC_METHOD=fork", "UV_USE_IO_URING=0", "UCX_MEM_MMAP_HOOK_MODE=none",
}

// Tools are DMTCP's programs and the fdkeep plugin.
type Tools struct {
	// Dir holds dmtcp_launch, dmtcp_command, dmtcp_restart and
	// dmtcp_coordinator; empty means PATH.
	Dir string `json:"dir,omitempty"`
	// Plugin is libdmtcp_fdkeep.so.
	Plugin string `json:"plugin"`
	// Port is the coordinator's on loopback. 0 lets Coordinator take a free
	// one, which a coordinator another run left behind cannot hold.
	Port int `json:"port,omitempty"`
	// TmpDir is DMTCP_TMPDIR, where the coordinator and the workers keep
	// their own files: outside the trees a snapshot carries.
	TmpDir string `json:"tmp_dir,omitempty"`
}

// Setenv makes TmpDir this process's DMTCP_TMPDIR, inherited by the
// coordinator, the engine and dmtcp_restart.
func (t Tools) Setenv() error {
	if t.TmpDir == "" {
		return nil
	}
	if err := os.MkdirAll(t.TmpDir, 0o755); err != nil {
		return err
	}
	return os.Setenv("DMTCP_TMPDIR", t.TmpDir)
}

func (t Tools) path(name string) string {
	if t.Dir == "" {
		return name
	}
	return filepath.Join(t.Dir, name)
}

func (t Tools) port() string { return strconv.Itoa(t.Port) }

// Check finds every program and the plugin.
func (t Tools) Check() error {
	for _, name := range []string{"dmtcp_launch", "dmtcp_command", "dmtcp_restart", "dmtcp_coordinator"} {
		if _, err := exec.LookPath(t.path(name)); err != nil {
			return fmt.Errorf("dmtcp: %s: %w", name, err)
		}
	}
	if t.Plugin == "" {
		return errors.New("dmtcp: no fdkeep plugin given")
	}
	if _, err := os.Stat(t.Plugin); err != nil {
		return fmt.Errorf("dmtcp: fdkeep plugin: %w", err)
	}
	return nil
}

// Version is dmtcp_launch's first line: "dmtcp_launch (DMTCP) 5.0".
func (t Tools) Version(ctx context.Context) string {
	out, err := process.Capture(ctx, 10*time.Second, t.path("dmtcp_launch"), "--version")
	if err != nil {
		return ""
	}
	line, _, _ := strings.Cut(out.Stdout, "\n")
	return strings.TrimSpace(line)
}

// Coordinator starts a coordinator in the background and returns its port:
// Port, or the free one the system gave it when Port is 0, which it writes
// to a file before it forks into the background. It exits with the last
// process of its computation; one that no process joined waits for Quit.
func (t Tools) Coordinator(ctx context.Context) (int, error) {
	f, err := os.CreateTemp(t.TmpDir, "coordinator-*.port")
	if err != nil {
		return 0, err
	}
	_ = f.Close()
	defer os.Remove(f.Name())
	if _, err := process.Capture(ctx, 30*time.Second, t.path("dmtcp_coordinator"),
		"--daemon", "--exit-on-last", "--port", t.port(), "--port-file", f.Name()); err != nil {
		return 0, err
	}
	data, err := os.ReadFile(f.Name())
	if err != nil {
		return 0, err
	}
	port, err := strconv.Atoi(strings.TrimSpace(string(data)))
	if err != nil || port <= 0 {
		return 0, fmt.Errorf("dmtcp: the coordinator wrote no port: %q", data)
	}
	return port, nil
}

// Wrap is argv under dmtcp_launch: raw images into dir, the plugin loaded.
func (t Tools) Wrap(argv []string, dir string) []string {
	out := []string{t.path("dmtcp_launch"), "--coord-port", t.port(), "--no-gzip", "--ckptdir", dir,
		"--with-plugin", t.Plugin}
	return append(out, argv...)
}

// Checkpoint writes the images of every process into dir and returns once
// all are written; the processes keep running. No image is written until
// every process has drained its sockets, which takes seconds; a drain that
// does not end fails the checkpoint once drain has passed with no image in
// dir, instead of at timeout.
func (t Tools) Checkpoint(ctx context.Context, dir string, drain, timeout time.Duration) error {
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	done := make(chan error, 1)
	go func() {
		_, err := process.Capture(ctx, timeout, t.path("dmtcp_command"), "--coord-port", t.port(), "--bcheckpoint")
		done <- err
	}()
	return awaitCheckpoint(dir, drain, done)
}

// awaitCheckpoint returns the checkpoint's result, or an error when drain
// passes before the first image appears in dir (DMTCP writes ckpt_*.temp,
// then renames it).
func awaitCheckpoint(dir string, drain time.Duration, done <-chan error) error {
	select {
	case err := <-done:
		return err
	case <-time.After(drain):
	}
	if written, _ := filepath.Glob(filepath.Join(dir, "ckpt_*")); len(written) == 0 {
		return fmt.Errorf("dmtcp: no image written %s into the checkpoint: a process is still draining "+
			"a socket whose peer does not answer (docs/dmtcp.md)", drain)
	}
	return <-done
}

// Kill ends the computation.
func (t Tools) Kill(ctx context.Context) error {
	_, err := process.Capture(ctx, 30*time.Second, t.path("dmtcp_command"), "--coord-port", t.port(), "--kill")
	return err
}

// Quit kills the processes connected to the coordinator and ends it.
func (t Tools) Quit(ctx context.Context) error {
	_, err := process.Capture(ctx, 10*time.Second, t.path("dmtcp_command"), "--coord-port", t.port(), "--quit")
	return err
}

// RestartArgs are dmtcp_restart's arguments. --no-strict-checking drops its
// one check, that each image's owner is this user: warmctl checks every file
// of a snapshot against its SHA-256 instead, and a network file system
// reports another owner (nobody under geesefs or NFS root_squash).
func (t Tools) RestartArgs(images []string) []string {
	return append([]string{"--coord-port", t.port(), "--no-strict-checking"}, images...)
}

// Restart starts dmtcp_restart on images. Its process becomes the first
// restored process, the root of the tree; the caller waits on it.
func (t Tools) Restart(images []string, log io.Writer) (*exec.Cmd, error) {
	args := t.RestartArgs(images)
	cmd := exec.Command(t.path("dmtcp_restart"), args...)
	cmd.Stdout, cmd.Stderr = log, log
	if err := cmd.Start(); err != nil {
		return nil, err
	}
	return cmd, nil
}

// Peer is a process of the computation.
type Peer struct {
	Name    string `json:"name"`
	Virtual int    `json:"virtual"`
	Real    int    `json:"real"`
	State   string `json:"state"`
}

// List asks the coordinator for the computation's processes.
func (t Tools) List(ctx context.Context) ([]Peer, error) {
	out, err := process.Capture(ctx, 10*time.Second, t.path("dmtcp_command"), "--coord-port", t.port(), "--list")
	if err != nil {
		return nil, err
	}
	return ParseList(out.Stdout + "\n" + out.Stderr)
}

// "1, python3.12[40000:47148]@b5266da1239b, 314ceaedf4104db7-40000-3941ee323f43e, WorkerState::RUNNING, "
var peerLine = regexp.MustCompile(`^\s*\d+,\s*(.*)\[(\d+):(\d+)\]@[^,]*,\s*[^,]*,\s*(?:WorkerState::)?([A-Z_]+)`)

// ParseList reads dmtcp_command --list: one peer per client line.
func ParseList(text string) ([]Peer, error) {
	var peers []Peer
	for _, line := range strings.Split(text, "\n") {
		m := peerLine.FindStringSubmatch(line)
		if m == nil {
			continue
		}
		v, _ := strconv.Atoi(m[2])
		r, _ := strconv.Atoi(m[3])
		peers = append(peers, Peer{Name: m[1], Virtual: v, Real: r, State: m[4]})
	}
	if len(peers) == 0 {
		return nil, fmt.Errorf("dmtcp: no peers in the coordinator's list: %q", strings.TrimSpace(text))
	}
	sort.Slice(peers, func(i, j int) bool { return peers[i].Virtual < peers[j].Virtual })
	return peers, nil
}

// Images are the checkpoint images in dir, one per process.
func Images(dir string) ([]string, error) {
	images, err := filepath.Glob(filepath.Join(dir, "ckpt_*.dmtcp"))
	if err != nil {
		return nil, err
	}
	if len(images) == 0 {
		return nil, fmt.Errorf("dmtcp: no checkpoint images in %s", dir)
	}
	sort.Strings(images)
	return images, nil
}

// WaitPeers polls the coordinator until n peers run. A restart whose process
// asserted never gets there, the others wait at a barrier for it, so it
// fails as soon as a peer leaves or failed, checked on every poll, says so.
func (t Tools) WaitPeers(ctx context.Context, n int, timeout time.Duration, failed func() error) ([]Peer, error) {
	return waitPeers(ctx, t.List, n, timeout, failed)
}

func waitPeers(ctx context.Context, list func(context.Context) ([]Peer, error), n int, timeout time.Duration,
	failed func() error,
) ([]Peer, error) {
	deadline := time.Now().Add(timeout)
	w := peerWait{n: n}
	for {
		if failed != nil {
			if err := failed(); err != nil {
				return nil, err
			}
		}
		peers, err := list(ctx)
		if err == nil {
			if err := w.left(peers); err != nil {
				return nil, err
			}
			if err = w.running(peers); err == nil {
				return peers, nil
			}
		}
		if time.Now().After(deadline) {
			return nil, fmt.Errorf("dmtcp: restart not done after %s: %w", timeout, err)
		}
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-time.After(50 * time.Millisecond):
		}
	}
}

// peerWait follows a restart's peer list.
type peerWait struct{ n, most int }

// left fails once the list is shorter than it was: a restarted process died.
func (w *peerWait) left(peers []Peer) error {
	if len(peers) < w.most {
		return fmt.Errorf("dmtcp: %d of %d restarted processes left; the log above says why", w.most-len(peers), w.most)
	}
	w.most = len(peers)
	return nil
}

// running is nil once all n peers run.
func (w *peerWait) running(peers []Peer) error {
	running := 0
	for _, p := range peers {
		if p.State == "RUNNING" {
			running++
		}
	}
	if running == w.n && len(peers) == w.n {
		return nil
	}
	return fmt.Errorf("%d of %d peers running", running, w.n)
}

// Map is a peer list as virtual-to-real and real-to-virtual PIDs.
type Map struct{ ToReal, ToVirtual map[int]int }

// NewMap indexes peers.
func NewMap(peers []Peer) Map {
	m := Map{ToReal: map[int]int{}, ToVirtual: map[int]int{}}
	for _, p := range peers {
		m.ToReal[p.Virtual], m.ToVirtual[p.Real] = p.Real, p.Virtual
	}
	return m
}

// Real translates virtual PIDs; it fails on one the computation lacks.
func (m Map) Real(virtual []int) ([]int, error) { return translate(virtual, m.ToReal, "virtual") }

// Virtual translates real PIDs; it fails on one the computation lacks.
func (m Map) Virtual(pids []int) ([]int, error) { return translate(pids, m.ToVirtual, "real") }

func translate(in []int, table map[int]int, kind string) ([]int, error) {
	out := make([]int, 0, len(in))
	for _, pid := range in {
		to, ok := table[pid]
		if !ok {
			return nil, fmt.Errorf("dmtcp: %s PID %d is not in the computation", kind, pid)
		}
		out = append(out, to)
	}
	return out, nil
}
