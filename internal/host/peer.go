package host

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os/exec"
	"strings"
	"sync"
	"time"
)

// PeerProbeCommand is the warmctl subcommand that runs the probe in a child
// process, so that a copy which never finishes cannot hang warmctl itself.
const PeerProbeCommand = "peer-probe"

// PeerTimeout bounds a probe: CUDA's start on every GPU, then 64 MiB per
// direction, which a working peer path moves in milliseconds. On a broken
// one, where the PCIe fabric between the GPUs (ACS, an IOMMU) does not carry
// what the driver reports, the copy never finishes.
var PeerTimeout = 10 * time.Second

// PeerCopy is one direction of the probe.
type PeerCopy struct {
	From int     `json:"from"`
	To   int     `json:"to"`
	GBps float64 `json:"gbps"`
}

// PeerResult is what the probe's child reports.
type PeerResult struct {
	// Claimed is whether the driver reports peer access for every pair.
	Claimed bool       `json:"claimed"`
	Copies  []PeerCopy `json:"copies"`
}

// PeerReport says whether the GPUs' peer path works.
type PeerReport struct {
	PeerResult
	// Broken is set when the driver claims peer access and a copy did not
	// finish within PeerTimeout; Why says which GPUs.
	Broken bool
	Why    string
}

// String is the report as a log line.
func (r PeerReport) String() string {
	if r.Broken {
		return r.Why
	}
	if !r.Claimed {
		return "the driver reports no peer access"
	}
	var parts []string
	for _, c := range r.Copies {
		parts = append(parts, fmt.Sprintf("%d->%d %.1f GB/s", c.From, c.To, c.GBps))
	}
	return strings.Join(parts, ", ")
}

// ProbePeers copies 64 MiB between every ordered pair of gpus, peer to peer,
// in a child process (self PeerProbeCommand). Engines trust the driver's
// report of peer access: vLLM's custom all-reduce and NCCL's P2P transport
// map each other's memory and spin on it, so a path that does not carry the
// bytes hangs the engine or faults the GPUs. A child that runs past
// PeerTimeout is killed; the path is broken if it was inside a copy.
func ProbePeers(ctx context.Context, self string, gpus []GPU) (PeerReport, error) {
	args := []string{PeerProbeCommand}
	for _, g := range gpus {
		args = append(args, g.UUID)
	}
	ctx, cancel := context.WithTimeout(ctx, PeerTimeout)
	defer cancel()
	var stdout, stderr lockedBuffer
	cmd := exec.Command(self, args...)
	cmd.Stdout, cmd.Stderr = &stdout, &stderr
	if err := cmd.Start(); err != nil {
		return PeerReport{}, fmt.Errorf("peer probe: %w", err)
	}
	exited := make(chan error, 1)
	go func() { exited <- cmd.Wait() }()
	var err error
	select {
	case err = <-exited:
	case <-ctx.Done():
		// A child inside a broken copy leaves the driver only when the copy
		// gives up (a minute on RunPod's L40S): killed, it is reaped in the
		// background while the bake goes on.
		_ = cmd.Process.Kill()
	}
	if errors.Is(ctx.Err(), context.DeadlineExceeded) {
		// Broken only if the time ran out inside a copy: a child stuck
		// before one says nothing about the peer path.
		from, to, copying := lastCopy(stdout.String())
		if !copying {
			return PeerReport{}, fmt.Errorf("peer probe: no copy started in %s: %s", PeerTimeout,
				strings.TrimSpace(stderr.String()))
		}
		return PeerReport{PeerResult: PeerResult{Claimed: true}, Broken: true,
			Why: fmt.Sprintf("a 64 MiB peer copy from GPU %d to GPU %d did not finish in %s, though the driver "+
				"reports peer access", gpus[from].Index, gpus[to].Index, PeerTimeout)}, nil
	}
	if err != nil {
		return PeerReport{}, fmt.Errorf("peer probe: %w: %s", err, strings.TrimSpace(stderr.String()))
	}
	var res PeerResult
	out := stdout.String()
	if err := json.Unmarshal([]byte(out[strings.LastIndex(out, "\n{")+1:]), &res); err != nil {
		return PeerReport{}, fmt.Errorf("peer probe: %w", err)
	}
	return PeerReport{PeerResult: res}, nil
}

// lastCopy reads the child's "copy I J" lines: the pair it was copying when
// it stopped, if it stopped inside a copy.
func lastCopy(out string) (int, int, bool) {
	lines := strings.Split(strings.TrimSpace(out), "\n")
	var from, to int
	if n, _ := fmt.Sscanf(lines[len(lines)-1], "copy %d %d", &from, &to); n != 2 {
		return 0, 0, false
	}
	return from, to, true
}

// formatUUID writes a CUDA device UUID as NVML does: GPU-8-4-4-4-12 hex.
func formatUUID(id [16]byte) string {
	h := fmt.Sprintf("%x", id[:])
	return "GPU-" + h[:8] + "-" + h[8:12] + "-" + h[12:16] + "-" + h[16:20] + "-" + h[20:]
}

// lockedBuffer is a child's output, read while the child may still write.
type lockedBuffer struct {
	mu  sync.Mutex
	buf bytes.Buffer
}

func (b *lockedBuffer) Write(p []byte) (int, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.buf.Write(p)
}

func (b *lockedBuffer) String() string {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.buf.String()
}
