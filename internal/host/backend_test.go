package host

import (
	"strings"
	"testing"
)

func TestChooseBackend(t *testing.T) {
	// A RunPod pod: Docker's default capabilities and seccomp profile, a
	// read-only /proc/sys.
	runpod := ParseStatus("Name:\tbash\nCapEff:\t00000000a80425fb\nSeccomp:\t2\n")
	if runpod.Seccomp != 2 || runpod.CapEff != 0xa80425fb {
		t.Fatalf("%+v", runpod)
	}
	// A privileged container.
	privileged := BackendProbe{CapEff: 0x000001ffffffffff, LastPIDWritable: true}

	for _, c := range []struct {
		want  string
		probe BackendProbe
		got   string
		fails bool
	}{
		{"auto", runpod, BackendDMTCP, false},
		{"", privileged, BackendCRIU, false},
		{"criu", runpod, "", true},
		{"criu", privileged, BackendCRIU, false},
		{"dmtcp", privileged, BackendDMTCP, false},
		{"podman", privileged, "", true},
	} {
		got, why, err := ChooseBackend(c.want, c.probe)
		if (err != nil) != c.fails || got != c.got {
			t.Errorf("ChooseBackend(%q, %+v) = %q %q %v", c.want, c.probe, got, why, err)
		}
	}
	if why := runpod.CRIUReason(); !strings.Contains(why, "CAP_CHECKPOINT_RESTORE") || !strings.Contains(why, "seccomp mode 2") {
		t.Fatalf("reason %q", why)
	}
}
