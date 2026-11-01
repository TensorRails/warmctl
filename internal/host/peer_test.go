package host

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// fakeProbe is a stand-in for warmctl's peer-probe child.
func fakeProbe(t *testing.T, body string) string {
	t.Helper()
	name := filepath.Join(t.TempDir(), "warmctl")
	if err := os.WriteFile(name, []byte("#!/bin/sh\n"+body+"\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	return name
}

func TestProbePeers(t *testing.T) {
	gpus := []GPU{{Index: 2, UUID: "GPU-a"}, {Index: 5, UUID: "GPU-b"}}
	ctx := context.Background()

	// The child gets the subcommand and the UUIDs in rank order.
	ok := fakeProbe(t, `[ "$*" = "peer-probe GPU-a GPU-b" ] || exit 3
echo "copy 0 1"; echo "copy 1 0"
echo '{"claimed":true,"copies":[{"from":0,"to":1,"gbps":24.5},{"from":1,"to":0,"gbps":25}]}'`)
	r, err := ProbePeers(ctx, ok, gpus)
	if err != nil || r.Broken || !r.Claimed || len(r.Copies) != 2 {
		t.Fatalf("working path: %+v %v", r, err)
	}
	if !strings.Contains(r.String(), "0->1 24.5 GB/s") {
		t.Fatalf("report %q", r)
	}

	// A probe that cannot run is an error, not a broken path.
	if r, err := ProbePeers(ctx, fakeProbe(t, "echo no libcuda >&2; exit 1"), gpus); err == nil || r.Broken {
		t.Fatalf("failed probe: %+v %v", r, err)
	}

	// A copy that never finishes: the child is killed and the path broken,
	// named by the host's GPU indices.
	// Long enough for the shell to print under a loaded go test -race ./...
	defer func(d time.Duration) { PeerTimeout = d }(PeerTimeout)
	PeerTimeout = 2 * time.Second
	r, err = ProbePeers(ctx, fakeProbe(t, `echo "copy 0 1"; echo "copy 1 0"; sleep 10`), gpus)
	if err != nil || !r.Broken || !strings.Contains(r.Why, "from GPU 5 to GPU 2 did not finish") {
		t.Fatalf("hung copy: %+v %v", r, err)
	}
	// A child stuck before any copy (CUDA's start) says nothing about the
	// peer path: an error, and the profile stays.
	if r, err := ProbePeers(ctx, fakeProbe(t, "sleep 10"), gpus); err == nil || r.Broken {
		t.Fatalf("stuck before a copy: %+v %v", r, err)
	}
}

func TestFormatUUID(t *testing.T) {
	id := [16]byte{0x8f, 0x1e, 0x2a, 0x33, 0x01, 0x02, 0x03, 0x04, 0xaa, 0xbb, 0, 1, 2, 3, 4, 5}
	if got := formatUUID(id); got != "GPU-8f1e2a33-0102-0304-aabb-000102030405" {
		t.Fatal(got)
	}
}
