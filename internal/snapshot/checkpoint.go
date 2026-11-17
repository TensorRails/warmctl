package snapshot

import (
	"context"
	"fmt"

	"github.com/TensorRails/warmctl/v2/internal/accel"
	"github.com/TensorRails/warmctl/v2/internal/manifest"
	"github.com/TensorRails/warmctl/v2/internal/profile"
	"github.com/TensorRails/warmctl/v2/internal/rocm"
	"github.com/TensorRails/warmctl/v2/internal/vllm"
)

type profileCase = profile.WarmupCase

func (b *bake) export(ctx context.Context) error {
	part := b.o.PartBytes
	if part == 0 {
		part = 1 << 30
	}
	var err error
	if b.exports, err = b.c.Engine.ExportPool(ctx, b.state()+"/out/weights", part, threads(b.o.Threads)); err != nil {
		return err
	}
	var bytes int64
	slowest := 0.0
	for _, x := range b.exports {
		for _, p := range x.Parts {
			bytes += p.Size
		}
		slowest = max(slowest, x.Seconds)
	}
	b.p.Printf("weights exported: %.2f GiB in %d ranks, slowest rank %.1fs",
		float64(bytes)/(1<<30), len(b.exports), slowest)
	return nil
}

// sleep discards the pool and keeps, per rank, only the exported
// allocations still in the allocator's table: a segment the allocator
// released while going to sleep held nothing, and a restore maps only what
// the table lists.
func (b *bake) sleep(ctx context.Context) error {
	if err := b.c.Engine.Sleep(ctx, 2); err != nil {
		return err
	}
	tables, err := b.c.Engine.Allocations(ctx)
	if err != nil {
		return err
	}
	dropped := 0
	for i := range b.exports {
		var n int
		b.exports[i].Allocations, n = Surviving(b.exports[i].Allocations, tables[i].Allocations)
		dropped += n
	}
	prepare := b.c.Engine.PrepareCheckpoint
	if b.vendor == accel.AMD {
		// ROCm's vLLM images may predate checkpoint_prepare.
		prepare = b.c.Engine.PrepareCheckpointIfAny
	}
	if b.prepared, err = prepare(ctx); err != nil {
		return err
	}
	b.p.Printf("engine asleep; %d released segments dropped; collectives detached: %t", dropped, b.detached())
	return nil
}

// Surviving keeps the exported allocations the asleep table still has,
// renumbered, and reports how many went.
func Surviving(exported []manifest.Allocation, table []vllm.Allocation) ([]manifest.Allocation, int) {
	live := map[string]bool{}
	for _, a := range table {
		live[fmt.Sprintf("%s/%d/%s", a.Ptr, a.Bytes, a.Tag)] = true
	}
	var out []manifest.Allocation
	for _, a := range exported {
		if live[fmt.Sprintf("%s/%d/%s", a.Ptr, a.Bytes, a.Tag)] {
			a.Index = len(out)
			out = append(out, a)
		}
	}
	return out, len(exported) - len(out)
}

func (b *bake) detached() bool {
	for _, p := range b.prepared {
		if !p.Detached {
			return false
		}
	}
	return len(b.prepared) > 0
}

// engineGPUState checkpoints the GPU state the engine's own processes move
// (AMD); nothing on NVIDIA, whose state the agent's cuda step moves.
func (b *bake) engineGPUState(ctx context.Context) error {
	state, err := accel.For(b.vendor, b.gpuLib.Version)
	if err != nil {
		return err
	}
	if b.gpuRes, err = state.Checkpoint(ctx, b.c.Engine); err != nil {
		return err
	}
	if b.vendor != accel.AMD {
		return nil
	}
	b.p.Printf("GPU state checkpointed by %d workers (libkfdreplay) in %.1fs", len(b.gpuRes.Ranks), b.gpuRes.Seconds)
	if n := b.gpuRes.MaxWptr(); n > rocm.MaxWptrWarn {
		b.p.Printf("warning: a GPU queue has taken %d packets; a restore replays them (~1 µs each): "+
			"snapshot a freshly warmed engine", n)
	}
	return nil
}

// checkpoint drives the agent: cross-process memory released, CUDA to host
// memory, freeze, dump, archive.
func (b *bake) checkpoint(ctx context.Context) error {
	if err := b.engineGPUState(ctx); err != nil {
		return err
	}
	if err := b.c.Step(ctx, "ipc", nil, &b.ipc); err != nil {
		return err
	}
	if b.ipc.Interposed {
		b.p.Printf("cross-process memory prepared in %.1fs: %d of the CUDA processes interposed (%s)",
			b.ipc.Seconds, len(b.ipc.Participants), b.ipc.Reason)
	}
	if err := b.c.Step(ctx, "cuda", nil, &b.cudaRes); err != nil {
		return err
	}
	if b.vendor == accel.AMD {
		b.p.Printf("GPU devices released by every process: %d run ROCm with libkfdreplay", len(b.cudaRes.Owners))
	} else {
		b.p.Printf("CUDA checkpointed: %d owners in %.1fs, %s", len(b.cudaRes.Owners), b.cudaRes.Seconds, b.cudaRes.Backend)
	}
	if err := b.c.Step(ctx, "freeze", nil, &b.frozen); err != nil {
		return err
	}
	if b.frozen.Coverage != nil {
		b.coverage = b.frozen.Coverage
		b.p.Printf("files outside the snapshot: %d to check at restore, %d uncovered",
			len(b.coverage.Mapped), len(b.coverage.Uncovered))
	}
	if err := b.c.Step(ctx, "dump", nil, &b.dumped); err != nil {
		return err
	}
	b.p.Printf("%s dump of %d processes in %.1fs (%s)", b.backend,
		len(b.frozen.Tree.Processes), b.dumped.Seconds, b.dumped.Version)
	var fs struct {
		Trees []struct {
			Name  string `json:"name"`
			Bytes int64  `json:"bytes"`
		} `json:"trees"`
		Seconds float64 `json:"seconds"`
	}
	if err := b.c.Step(ctx, "fs", nil, &fs); err != nil {
		return err
	}
	b.p.Printf("trees archived in %.1fs: %+v", fs.Seconds, fs.Trees)
	if err := b.c.Step(ctx, "finish", nil, nil); err != nil {
		return err
	}
	code, err := b.c.Wait(ctx)
	if err != nil {
		return err
	}
	if code != 0 {
		return fmt.Errorf("donor agent exited %d", code)
	}
	return nil
}
