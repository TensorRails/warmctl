package host

import (
	"strings"
	"testing"
)

func TestCRIUReason(t *testing.T) {
	// A RunPod pod: Docker's default capabilities and seccomp profile, a
	// read-only /proc/sys.
	runpod := ParseStatus("Name:\tbash\nCapEff:\t00000000a80425fb\nSeccomp:\t2\n")
	if runpod.Seccomp != 2 || runpod.CapEff != 0xa80425fb {
		t.Fatalf("%+v", runpod)
	}
	if why := runpod.CRIUReason(); !strings.Contains(why, "CAP_CHECKPOINT_RESTORE") || !strings.Contains(why, "seccomp mode 2") {
		t.Fatalf("reason %q", why)
	}
	// A privileged container.
	privileged := BackendProbe{CapEff: 0x000001ffffffffff, LastPIDWritable: true}
	if why := privileged.CRIUReason(); why != "" {
		t.Fatalf("privileged: %q", why)
	}
}
