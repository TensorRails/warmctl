package restore

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func probe(marker bool, dockerUp bool, kube ...string) Probe {
	return Probe{
		Runtime: func() (Runtime, bool, error) {
			return Runtime{Warmctl: "v2.0.0-beta.4", Base: "vllm/vllm-openai@sha256:ad9d"}, marker, nil
		},
		Docker: func(context.Context) error {
			if dockerUp {
				return nil
			}
			return errors.New("dial unix /var/run/docker.sock: no such file")
		},
		Kubernetes: func() []string { return kube },
	}
}

func TestChooseMode(t *testing.T) {
	ctx := context.Background()
	for _, c := range []struct {
		name         string
		mode         string
		probe        Probe
		inPlace      bool
		reason, fail string
	}{
		// 1. An explicit mode wins over the marker and over Docker.
		{"host forced in the image", ModeHost, probe(true, false), false, "", ""},
		{"in place forced on a Docker host", ModeInPlace, probe(false, true), true, "--mode in-place", ""},
		// 2. The runtime image's marker: in place, Docker or not.
		{"image", ModeAuto, probe(true, false), true, "warmctl runtime image v2.0.0-beta.4", ""},
		{"image with a Docker socket", "", probe(true, true), true, "restoring in place", ""},
		{"image in a pod", ModeAuto, probe(true, false, "KUBERNETES_SERVICE_HOST"), true,
			"Kubernetes pod (KUBERNETES_SERVICE_HOST), warmctl runtime image", ""},
		// 3. No marker, Docker answers: a host restore, as before.
		{"host", ModeAuto, probe(false, true), false, "", ""},
		{"host that looks like a pod", ModeAuto, probe(false, true, "kubepods cgroup"), false, "", ""},
		// 4. Neither: no guess from missing Docker or from a pod's signs.
		{"bare host", ModeAuto, probe(false, false), false, "", "install Docker, or run the warmctl image in Kubernetes"},
		{"pod of another image", ModeAuto, probe(false, false, "service account"), false, "",
			"Kubernetes pod (service account) running another image"},
		{"unknown mode", "pod", probe(true, true), false, "", "want auto, host or in-place"},
	} {
		inPlace, reason, err := Choose(ctx, c.mode, c.probe)
		switch {
		case c.fail != "":
			if err == nil || !strings.Contains(err.Error(), c.fail) {
				t.Errorf("%s: %v, want an error with %q", c.name, err, c.fail)
			}
		case err != nil:
			t.Errorf("%s: %v", c.name, err)
		case inPlace != c.inPlace || !strings.Contains(reason, c.reason) || (c.reason == "" && reason != ""):
			t.Errorf("%s: in place %t, %q", c.name, inPlace, reason)
		}
	}
	broken := probe(false, true)
	broken.Runtime = func() (Runtime, bool, error) { return Runtime{}, false, errors.New("unreadable marker") }
	if _, _, err := Choose(ctx, ModeAuto, broken); err == nil {
		t.Error("an unreadable marker fell through to Docker")
	}
}

func TestRuntimeMarker(t *testing.T) {
	dir := t.TempDir()
	name := filepath.Join(dir, "RUNTIME")
	if _, ok, err := ReadRuntime(name); ok || err != nil {
		t.Fatalf("no marker: %t %v", ok, err)
	}
	os.WriteFile(name, []byte("warmctl=v2.0.0-beta.4\nbase=vllm/vllm-openai@sha256:ad9d\n"), 0o644)
	rt, ok, err := ReadRuntime(name)
	if !ok || err != nil || rt.Warmctl != "v2.0.0-beta.4" || rt.Base != "vllm/vllm-openai@sha256:ad9d" {
		t.Fatalf("marker: %+v %t %v", rt, ok, err)
	}
	os.WriteFile(name, []byte("base=x\n"), 0o644)
	if _, ok, err := ReadRuntime(name); ok || err == nil {
		t.Fatalf("a marker without the version: %t %v", ok, err)
	}
}
