package restore

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/TensorRails/warmctl/v2/internal/agent"
	"github.com/TensorRails/warmctl/v2/internal/dmtcp"
	"github.com/TensorRails/warmctl/v2/internal/docker"
	"github.com/TensorRails/warmctl/v2/internal/runtimefs"
)

// The snapshot's backend picks the pod setup: a CRIU agent names a UTS
// namespace of its own and runs the given CRIU; a DMTCP agent keeps the
// pod's name and keeps DMTCP's files in the run directory, where undo
// finds a coordinator it left.
func TestPodSetup(t *testing.T) {
	r := &restore{m: podManifest(false, ""), dir: "/run/warmctl"}
	r.o.CRIU, r.o.DMTCP = "/usr/sbin/criu", dmtcp.Tools{Dir: "/opt/warmctl/dmtcp/bin"}
	var cfg agent.Config
	r.podSetup().agent(&cfg)
	if _, ok := r.podSetup().(mountedPod); !ok || cfg.Hostname != docker.Hostname || cfg.CRIU != "/usr/sbin/criu" ||
		cfg.Backend != "" {
		t.Fatalf("criu: %+v", cfg)
	}
	r.m.Process.Backend = agent.BackendDMTCP
	cfg = agent.Config{}
	r.podSetup().agent(&cfg)
	if _, ok := r.podSetup().(linkedPod); !ok || cfg.Hostname != "" || cfg.Backend != agent.BackendDMTCP ||
		cfg.DMTCP.Dir != "/opt/warmctl/dmtcp/bin" || cfg.DMTCP.TmpDir != "/run/warmctl/dmtcp-tmp" {
		t.Fatalf("dmtcp: %+v", cfg)
	}
	// Without privileges /tmp is claimed as it is, never mounted.
	tree := filepath.Join(t.TempDir(), "tmp")
	if err := (linkedPod{r}).claimTree(agent.Tree{Name: "tmp", Path: tree}); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(tree, runtimefs.Mark)); err != nil || len(r.mounts) != 0 {
		t.Fatalf("claimed: %v, mounts %v", err, r.mounts)
	}
}
