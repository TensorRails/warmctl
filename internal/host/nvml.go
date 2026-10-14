package host

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"strings"
	"time"
)

// Session is an open NVML session. Devices are numbered in NVML's order,
// which is nvidia-smi's index.
type Session interface {
	Driver() (string, error)
	Count() (int, error)
	// GPU is device i's UUID, name, PCI bus ID as NVML prints it and total
	// memory; Index and Driver are left to the caller.
	GPU(i int) (GPU, error)
	// Processes counts the compute processes on device i.
	Processes(i int) (int, error)
	// NVLink reports whether devices i and j reach each other over NVLink.
	NVLink(i, j int) (bool, error)
	// Path is the closest PCIe ancestor of devices i and j in nvidia-smi
	// topo -m's terms: PIX, PXB, PHB, NODE or SYS.
	Path(i, j int) (string, error)
	Close() error
}

// Open starts an NVML session; tests replace it with a fake.
var Open = openNVML

// Timeout bounds one query: a GPU that fell off the bus can block NVML for
// minutes, and the caller should hear about it.
const Timeout = 30 * time.Second

// query runs fn in a session of its own, opened and shut down around it, so
// warmctl holds no NVIDIA device between queries. On timeout it returns and
// leaves the blocked call behind.
func query[T any](ctx context.Context, fn func(Session) (T, error)) (T, error) {
	ctx, cancel := context.WithTimeout(ctx, Timeout)
	defer cancel()
	type result struct {
		value T
		err   error
	}
	done := make(chan result, 1)
	go func() {
		s, err := Open()
		if err != nil {
			done <- result{err: err}
			return
		}
		value, err := fn(s)
		done <- result{value, errors.Join(err, s.Close())}
	}()
	select {
	case r := <-done:
		return r.value, r.err
	case <-ctx.Done():
		var zero T
		return zero, fmt.Errorf("nvml: %w", ctx.Err())
	}
}

func inventory(s Session) ([]GPU, error) {
	driver, err := s.Driver()
	if err != nil {
		return nil, err
	}
	n, err := s.Count()
	if err != nil {
		return nil, err
	}
	if n == 0 {
		return nil, errors.New("nvml: no GPUs")
	}
	gpus := make([]GPU, 0, n)
	for i := range n {
		g, err := s.GPU(i)
		if err != nil {
			return nil, err
		}
		if !strings.HasPrefix(g.UUID, "GPU-") {
			return nil, fmt.Errorf("nvml: GPU %d has UUID %q", i, g.UUID)
		}
		g.Index, g.PCIBusID, g.Driver = i, normalizeBus(g.PCIBusID), driver
		gpus = append(gpus, g)
	}
	return gpus, nil
}

func processes(s Session) (map[string]int, error) {
	n, err := s.Count()
	if err != nil {
		return nil, err
	}
	procs := map[string]int{}
	for i := range n {
		g, err := s.GPU(i)
		if err != nil {
			return nil, err
		}
		p, err := s.Processes(i)
		if err != nil {
			return nil, err
		}
		if p > 0 {
			procs[g.UUID] = p
		}
	}
	return procs, nil
}

// LinkMatrix is how every pair of the host's GPUs is linked: X on the
// diagonal, NV for NVLink, else the PCIe path.
type LinkMatrix [][]string

func links(s Session) (LinkMatrix, error) {
	n, err := s.Count()
	if err != nil {
		return nil, err
	}
	m := make(LinkMatrix, n)
	for i := range n {
		m[i] = make([]string, n)
		for j := range n {
			if i == j {
				m[i][j] = "X"
				continue
			}
			nv, err := s.NVLink(i, j)
			if err != nil {
				return nil, err
			}
			if nv {
				m[i][j] = "NV"
			} else if m[i][j], err = s.Path(i, j); err != nil {
				return nil, err
			}
		}
	}
	return m, nil
}

// String renders the matrix as nvidia-smi topo -m lays it out.
func (m LinkMatrix) String() string {
	var b strings.Builder
	for i := range m {
		fmt.Fprintf(&b, "\tGPU%d", i)
	}
	b.WriteString("\n")
	for i, row := range m {
		fmt.Fprintf(&b, "GPU%d\t%s\n", i, strings.Join(row, "\t"))
	}
	return b.String()
}

// TopologyClass is nvlink when every pair of the selected GPUs is joined by
// NVLink, pcie otherwise; a single GPU is "single".
func TopologyClass(m LinkMatrix, gpus []GPU) string {
	if len(gpus) < 2 {
		return "single"
	}
	for _, a := range gpus {
		for _, b := range gpus {
			if a.Index == b.Index {
				continue
			}
			if a.Index >= len(m) || b.Index >= len(m[a.Index]) || m[a.Index][b.Index] != "NV" {
				return "pcie"
			}
		}
	}
	return "nvlink"
}

func digest(m LinkMatrix) string {
	sum := sha256.Sum256([]byte(m.String()))
	return hex.EncodeToString(sum[:])
}
