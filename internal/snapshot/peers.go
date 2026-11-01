package snapshot

import (
	"context"
	"os"

	"github.com/TensorRails/warmctl/v2/internal/host"
	"github.com/TensorRails/warmctl/v2/internal/profile"
)

// PeerOff keeps an engine's collectives off the GPUs' peer path: vLLM's
// custom all-reduce and its FlashInfer and symmetric-memory all-reduces map
// the other ranks' memory through CUDA IPC, and NCCL_P2P_DISABLE reaches
// only NCCL, which then goes through shared host memory.
var PeerOff = []string{
	"vllm.disable-custom-all-reduce=true",
	"env.NCCL_P2P_DISABLE=1",
	"env.VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC=0",
	"env.VLLM_ALLREDUCE_USE_SYMM_MEM=0",
}

// checkPeers copies between the GPUs of a PCIe host before the engine
// starts. Some clouds' hosts report peer access their PCIe fabric does not
// carry (ACS or an IOMMU between the GPUs: RunPod's L40S and A40 hosts):
// vLLM trusts the report, and its custom all-reduce hangs the engine at
// warmup, crawls at a fraction of a token per second, or faults the GPUs
// until the host resets them. Where a copy does not finish, the bake turns
// every peer-path collective off (PeerOff) and says so.
func (b *bake) checkPeers(ctx context.Context) error {
	if b.o.CPUTest || len(b.gpus) < 2 || b.topology != "pcie" {
		return nil
	}
	self, err := os.Executable()
	if err != nil {
		return err
	}
	r, err := host.ProbePeers(ctx, self, b.gpus)
	if err != nil {
		b.p.Printf("warning: %v; the profile's collectives stay as they are", err)
		return nil
	}
	b.p.Printf("peer copies between the GPUs: %s", r)
	if !r.Broken {
		return nil
	}
	b.p.Printf("warning: the GPUs' peer path does not work: custom all-reduce and NCCL P2P are off for this " +
		"engine (collectives go through host memory)")
	b.launch, err = withPeerOff(b.defaults, b.prof)
	return err
}

// withPeerOff resolves p again with PeerOff applied.
func withPeerOff(defaults, p profile.Profile) (profile.Launch, error) {
	for _, s := range PeerOff {
		if err := p.Set(s); err != nil {
			return profile.Launch{}, err
		}
	}
	return profile.Resolve(defaults, p)
}
