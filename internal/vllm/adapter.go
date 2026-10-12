package vllm

import (
	"context"
	"fmt"
	"sort"
	"time"

	"github.com/TensorRails/warmctl/v2/internal/manifest"
)

// RPC timeouts: moving a rank's pool is disk- and SHA-bound.
const (
	ShortRPC = 2 * time.Minute
	PoolRPC  = 60 * time.Minute
)

// Worker is warmctl_describe of one rank.
type Worker struct {
	Rank          int    `json:"rank"`
	PID           int    `json:"pid"`
	Device        int    `json:"device"`
	Name          string `json:"name"`
	Arch          string `json:"arch"`
	PCIBusID      string `json:"pci_bus_id"`
	UUID          string `json:"uuid"`
	VLLM          string `json:"vllm"`
	Torch         string `json:"torch"`
	AdapterSHA256 string `json:"adapter_sha256"`
}

// Export is warmctl_export_pool of one rank.
type Export struct {
	Rank        int                   `json:"rank"`
	PID         int                   `json:"pid"`
	Allocations []manifest.Allocation `json:"allocations"`
	Parts       []manifest.Part       `json:"parts"`
	Seconds     float64               `json:"seconds"`
}

// Allocation is one row of warmctl_allocations.
type Allocation struct {
	Ptr    string `json:"ptr"`
	Bytes  int64  `json:"bytes"`
	Tag    string `json:"tag"`
	Asleep bool   `json:"asleep"`
}

// Table is warmctl_allocations of one rank.
type Table struct {
	Rank        int          `json:"rank"`
	Allocations []Allocation `json:"allocations"`
}

// Prepared is warmctl_sleep_prepare of one rank.
type Prepared struct {
	Rank        int      `json:"rank"`
	Collectives []string `json:"collectives"`
	Detached    bool     `json:"detached"`
	NCCLMode    string   `json:"nccl_mode"`
}

// Reloaded is warmctl_reload_pool of one rank.
type Reloaded struct {
	Rank        int     `json:"rank"`
	Bytes       int64   `json:"bytes"`
	Allocations int     `json:"allocations"`
	Seconds     float64 `json:"seconds"`
}

func byRank[T any](rows []T, rank func(T) int, ranks int) ([]T, error) {
	sort.Slice(rows, func(i, j int) bool { return rank(rows[i]) < rank(rows[j]) })
	for i, r := range rows {
		if rank(r) != i {
			return nil, fmt.Errorf("worker ranks %d..%d are not 0..%d", rank(rows[0]), rank(rows[len(rows)-1]), ranks-1)
		}
	}
	return rows, nil
}

// Describe asks every worker who it is.
func (e *Engine) Describe(ctx context.Context) ([]Worker, error) {
	var rows []Worker
	if err := e.RPC(ctx, "warmctl_describe", nil, ShortRPC, &rows); err != nil {
		return nil, err
	}
	return byRank(rows, func(w Worker) int { return w.Rank }, e.Ranks)
}

// ExportPool fingerprints and writes every rank's weights pool under
// directory (a path inside the container).
func (e *Engine) ExportPool(ctx context.Context, directory string, partBytes int64, threads int) ([]Export, error) {
	var rows []Export
	kwargs := map[string]any{"directory": directory, "part_bytes": partBytes, "threads": threads}
	if err := e.RPC(ctx, "warmctl_export_pool", kwargs, PoolRPC, &rows); err != nil {
		return nil, err
	}
	return byRank(rows, func(x Export) int { return x.Rank }, e.Ranks)
}

// Allocations reads every rank's allocator table.
func (e *Engine) Allocations(ctx context.Context) ([]Table, error) {
	var rows []Table
	if err := e.RPC(ctx, "warmctl_allocations", nil, ShortRPC, &rows); err != nil {
		return nil, err
	}
	return byRank(rows, func(t Table) int { return t.Rank }, e.Ranks)
}

// PrepareCheckpoint runs vLLM's checkpoint_prepare, then warmctl's, on a
// sleeping engine.
func (e *Engine) PrepareCheckpoint(ctx context.Context) ([]Prepared, error) {
	if err := e.RPC(ctx, "checkpoint_prepare", nil, ShortRPC, nil); err != nil {
		return nil, err
	}
	var rows []Prepared
	if err := e.RPC(ctx, "warmctl_sleep_prepare", nil, ShortRPC, &rows); err != nil {
		return nil, err
	}
	return byRank(rows, func(p Prepared) int { return p.Rank }, e.Ranks)
}

// FinishRestore runs vLLM's checkpoint_restore, then warmctl's, on a
// restored, still sleeping engine.
func (e *Engine) FinishRestore(ctx context.Context) error {
	if err := e.RPC(ctx, "checkpoint_restore", nil, ShortRPC, nil); err != nil {
		return err
	}
	return e.RPC(ctx, "warmctl_restore_prepare", nil, ShortRPC, nil)
}

// ReloadPool copies every rank's pool back from the snapshot mounted at
// directory, whose manifest is at manifestPath, and checks fingerprints.
func (e *Engine) ReloadPool(ctx context.Context, manifestPath, directory string, threads int,
	verify bool) ([]Reloaded, error) {
	var rows []Reloaded
	kwargs := map[string]any{"manifest": manifestPath, "directory": directory, "threads": threads, "verify": verify}
	if err := e.RPC(ctx, "warmctl_reload_pool", kwargs, PoolRPC, &rows); err != nil {
		return nil, err
	}
	return byRank(rows, func(r Reloaded) int { return r.Rank }, e.Ranks)
}
