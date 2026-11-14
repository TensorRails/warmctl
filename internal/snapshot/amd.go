package snapshot

import (
	"errors"
	"fmt"

	"github.com/TensorRails/warmctl/v2/internal/accel"
	"github.com/TensorRails/warmctl/v2/internal/kfdreplay"
	"github.com/TensorRails/warmctl/v2/internal/rocm"
	"github.com/TensorRails/warmctl/v2/internal/run"
	"github.com/TensorRails/warmctl/v2/internal/vllm"
)

// chooseVendor takes the GPU vendor from the host. On an AMD host a bake
// runs in place under DMTCP: in the ROCm runtime image it does so without
// being asked (--in-place is implied); elsewhere it says how.
func (b *bake) chooseVendor() error {
	b.vendor = accel.Host()
	if b.vendor != accel.AMD {
		return nil
	}
	b.p.Printf("AMD Instinct host: GPU state moves through libkfdreplay, the process tree through DMTCP (preview)")
	if b.o.InPlace {
		return nil
	}
	if _, ok, err := run.ReadRuntime(run.RuntimeFile); err != nil || !ok {
		return errors.New("an AMD bake runs in place under DMTCP, in the ROCm runtime image " +
			"(or pass --in-place where DMTCP and libkfdreplay are installed)")
	}
	b.o.InPlace = true
	b.p.Printf("in the runtime image: baking in place")
	return nil
}

// amdPreflight checks the GPUs against what AMD snapshots are verified on,
// and finds the pinned libkfdreplay.
func (b *bake) amdPreflight() error {
	note, err := rocm.Supported(b.gpus)
	if err != nil {
		return err
	}
	if note != "" {
		b.p.Printf("warning: %s", note)
	}
	if b.gpuLib, err = kfdreplay.Locate(); err != nil {
		return err
	}
	b.p.Printf("libkfdreplay %s at %s (sha256 %.12s)", orAny(b.gpuLib.Version), b.gpuLib.Path, b.gpuLib.SHA256)
	return nil
}

func orAny(v string) string {
	if v == "" {
		return "(research build)"
	}
	return v
}

// checkWorkers checks that every AMD worker runs the libkfdreplay the bake
// preloads: a worker without it could not be checkpointed.
func (b *bake) checkWorkers(workers []vllm.Worker) error {
	for _, w := range workers {
		switch {
		case w.KFDReplay == "":
			return fmt.Errorf("rank %d runs without libkfdreplay (LD_PRELOAD of %s did not reach it)", w.Rank, b.gpuLib.Path)
		case b.gpuLib.Version != "" && w.KFDReplay != b.gpuLib.Version:
			return fmt.Errorf("rank %d runs libkfdreplay %s, the bake preloads %s", w.Rank, w.KFDReplay, b.gpuLib.Version)
		}
	}
	return nil
}
