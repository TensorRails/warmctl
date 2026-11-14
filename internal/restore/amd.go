package restore

import (
	"errors"
	"fmt"

	"github.com/TensorRails/warmctl/v2/internal/accel"
	"github.com/TensorRails/warmctl/v2/internal/host"
	"github.com/TensorRails/warmctl/v2/internal/kfdreplay"
	"github.com/TensorRails/warmctl/v2/internal/manifest"
	"github.com/TensorRails/warmctl/v2/internal/rocm"
)

// canRestoreVendor checks that a snapshot's GPUs are of this host's vendor,
// before anything is pulled. Every snapshot before AMD support is NVIDIA's,
// and on an NVIDIA host (or one without GPUs) this passes as before. An
// AMD snapshot restores in place under DMTCP, with its very libkfdreplay.
func canRestoreVendor(m manifest.Manifest, inPlace bool, here string) error {
	want := accel.Of(m.Hardware)
	if want != here {
		return fmt.Errorf("the snapshot was taken on %s GPUs, this host's are %s", want, here)
	}
	if want != accel.AMD {
		return nil
	}
	if m.Process.BackendName() != manifest.BackendDMTCP || m.Process.GPUState != manifest.GPUStateKFDReplay ||
		m.Process.GPULib == nil {
		return errors.New("an AMD snapshot needs the dmtcp backend and libkfdreplay; this one names neither")
	}
	if !inPlace {
		return errors.New("an AMD snapshot restores in place (--mode in-place), in the ROCm runtime image")
	}
	if err := kfdreplay.Same(*m.Process.GPULib); err != nil {
		return fmt.Errorf("the snapshot's processes mapped libkfdreplay %s, which this container lacks: %w "+
			"(restore in the runtime image the snapshot was taken in)", m.Process.GPULib.Version, err)
	}
	return nil
}

// compatible is Compatible for the snapshot's vendor: AMD checks its GPUs
// with rocm.Compatible, then the topology and CPU as NVIDIA's do; visible
// is how many GPUs this container sees.
func compatible(h manifest.Hardware, gpus []host.GPU, topology string, visible int) error {
	if accel.Of(h) != accel.AMD {
		return Compatible(h, gpus, topology, host.CPUFlags())
	}
	if err := rocm.Compatible(h, gpus); err != nil {
		return err
	}
	if err := rocm.RenderNodes(h, gpus, visible); err != nil {
		return err
	}
	if h.Topology != "" && topology != h.Topology {
		return fmt.Errorf("GPU topology %s, the snapshot was taken on %s", topology, h.Topology)
	}
	if missing := host.MissingFlags(h.CPUFlags, host.CPUFlags()); len(missing) > 0 {
		return fmt.Errorf("this CPU lacks %v that the donor's had", missing)
	}
	return nil
}

// amdFiles checks, before the restart, the files outside the snapshot the
// donor's processes mapped (the model's): a restart maps them again.
func (r *restore) amdFiles() error {
	if !r.amd() {
		return nil
	}
	return rocm.SameFiles(r.m.Process.Mapped)
}
