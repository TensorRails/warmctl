package snapshot

import (
	"slices"
	"testing"

	"github.com/TensorRails/warmctl/v2"
	"github.com/TensorRails/warmctl/v2/internal/profile"
)

// A profile that runs custom all-reduce on PCIe (2×L40S) loses it, and NCCL
// P2P with it, when the bake finds the peer path broken.
func TestWithPeerOff(t *testing.T) {
	l := profile.Loader{Assets: warmctl.Assets}
	defaults, err := l.Defaults()
	if err != nil {
		t.Fatal(err)
	}
	p, err := l.Load("qwen3-32b-l40s-tp2")
	if err != nil {
		t.Fatal(err)
	}
	before, err := profile.Resolve(defaults, p)
	if err != nil {
		t.Fatal(err)
	}
	if slices.Contains(before.Argv, "--disable-custom-all-reduce") {
		t.Fatal("the profile already turns custom all-reduce off; pick one that runs it")
	}
	after, err := withPeerOff(defaults, p)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Contains(after.Argv, "--disable-custom-all-reduce") {
		t.Fatalf("custom all-reduce still on: %v", after.Argv)
	}
	for key, want := range map[string]string{"NCCL_P2P_DISABLE": "1", "VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC": "0",
		"VLLM_ALLREDUCE_USE_SYMM_MEM": "0"} {
		if after.Env[key] != want {
			t.Errorf("%s = %q, want %q", key, after.Env[key], want)
		}
	}
}
