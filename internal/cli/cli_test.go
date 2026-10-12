package cli

import (
	"bytes"
	"context"
	"flag"
	"io"
	"strings"
	"testing"

	"github.com/TensorRails/warmctl/v2/internal/config"
)

// The old command names still reach the commands that replaced them.
func TestAliases(t *testing.T) {
	for _, c := range []struct {
		names []string
		want  string
	}{
		{[]string{"build", "snapshot", "bake"}, "build needs --profile"},
		{[]string{"up", "restore"}, "up needs a snapshot"},
		{[]string{"down", "stop"}, "down needs an instance id"},
	} {
		for _, name := range c.names {
			var stdout, stderr bytes.Buffer
			if code := Run(context.Background(), []string{name}, &stdout, &stderr, "test"); code != 1 ||
				!strings.Contains(stderr.String(), c.want) {
				t.Errorf("%s: exit %d, %q", name, code, stderr.String())
			}
		}
	}
}

func TestRepoName(t *testing.T) {
	for ref, want := range map[string]string{
		"oci://registry.warmctl.io/warmctl/glm53-flash:8xh100": "glm53-flash",
		"oci://localhost:5000/e2e/fake-tp2:v1":                 "fake-tp2",
		"oci://localhost:5000/fake-tp2":                        "fake-tp2",
		"oci://reg.io/a/b@sha256:0123":                         "b",
	} {
		if got := repoName(ref); got != want {
			t.Errorf("repoName(%q) = %q, want %q", ref, got, want)
		}
	}
}

func TestReference(t *testing.T) {
	cfg := config.Config{Registry: "reg.io/warmctl"}
	dir := t.TempDir()
	for arg, want := range map[string]string{
		"glm53-flash":            "oci://reg.io/warmctl/glm53-flash",
		"glm53-flash:8xh100":     "oci://reg.io/warmctl/glm53-flash:8xh100",
		"team/model":             "oci://reg.io/warmctl/team/model",
		"oci://other.io/x/y:v1":  "oci://other.io/x/y:v1",
		dir:                      "",
		"./snaps/glm":            "",
		"/var/lib/warmctl/x/glm": "",
		"ghcr.io/you/model:h100": "oci://ghcr.io/you/model:h100",
		"localhost:5000/e2e/x":   "oci://localhost:5000/e2e/x",
		"localhost/x":            "oci://localhost/x",
	} {
		if got := reference(arg, cfg); got != want {
			t.Errorf("reference(%q) = %q, want %q", arg, got, want)
		}
	}
}

func TestSources(t *testing.T) {
	cfg := config.Config{Registry: "reg.io/warmctl", Cache: "/mnt/shared/warmctl", Mirrors: []string{"oci://harbor.acme/warmctl/"}}
	got := sources("oci://reg.io/warmctl/glm53-flash:8xh100", cfg)
	want := []string{"dir:/mnt/shared/warmctl/glm53-flash:8xh100", "oci://harbor.acme/warmctl/glm53-flash:8xh100",
		"oci://reg.io/warmctl/glm53-flash:8xh100"}
	if strings.Join(got, " ") != strings.Join(want, " ") {
		t.Fatalf("sources %v", got)
	}
	if got := sources("oci://other.io/x/y", cfg); len(got) != 1 {
		t.Fatalf("a ref outside the registry has other sources: %v", got)
	}
	for arg, want := range map[string]string{"dir:/mnt/c/": "dir:/mnt/c", "/mnt/c": "dir:/mnt/c",
		"harbor.acme/warmctl": "oci://harbor.acme/warmctl", "oci://h.io/ns": "oci://h.io/ns"} {
		if got := namespace(arg); got != want {
			t.Errorf("namespace(%q) = %q, want %q", arg, got, want)
		}
	}
}

// In place the gateway opens before anything is pulled, and never on the
// restored engine's port.
func TestUpInPlaceListen(t *testing.T) {
	var stdout, stderr bytes.Buffer
	args := []string{"up", "--mode", "in-place", "--listen", "0.0.0.0:8000", "--run-dir", t.TempDir(), t.TempDir()}
	if code := Run(context.Background(), args, &stdout, &stderr, "test"); code != 1 ||
		!strings.Contains(stderr.String(), "port 8000 is the restored engine's") {
		t.Fatalf("exit %d, %q", code, stderr.String())
	}
}

// Flags count wherever they stand around the positional argument.
func TestParse(t *testing.T) {
	for _, args := range [][]string{
		{"name", "--plain-http", "--id", "x"},
		{"--plain-http", "name", "--id", "x"},
		{"--plain-http", "--id", "x", "name"},
	} {
		fs := flag.NewFlagSet("up", flag.ContinueOnError)
		plain := fs.Bool("plain-http", false, "")
		id := fs.String("id", "", "")
		got, err := parse(fs, args)
		if err != nil || got != "name" || !*plain || *id != "x" {
			t.Errorf("%v: %q %t %q %v", args, got, *plain, *id, err)
		}
	}
	fs := flag.NewFlagSet("up", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	if _, err := parse(fs, []string{"a", "b"}); err == nil {
		t.Error("two positional arguments accepted")
	}
}
