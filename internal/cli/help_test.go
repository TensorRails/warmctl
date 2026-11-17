package cli

import (
	"bytes"
	"context"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

var updateHelp = flag.Bool("update", false, "rewrite testdata/help from the current code")

// helpCommands are every command and alias warmctl has today. Each one's
// usage, flags, defaults and help texts are pinned in testdata/help: adding a
// GPU vendor must not change a single word a user sees (a new command or flag
// is a deliberate -update).
var helpCommands = []string{
	"", "build", "snapshot", "bake", "up", "restore", "ps", "down", "stop", "images", "ls", "catalog",
	"push", "pull", "mirror", "verify", "compress", "gc", "doctor", "hardware", "profile",
}

func TestHelpGolden(t *testing.T) {
	// Defaults that come from the environment are pinned to fixed values.
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("XDG_CONFIG_HOME", "")
	t.Setenv("XDG_STATE_HOME", "")
	t.Setenv("XDG_CACHE_HOME", "")
	t.Setenv("WARMCTL_STATE", "")
	t.Setenv("WARMCTL_REGISTRY", "")
	for _, name := range helpCommands {
		args := []string{}
		if name != "" {
			args = append(args, name, "-h")
		}
		var stdout, stderr bytes.Buffer
		code := Run(context.Background(), args, &stdout, &stderr, "test")
		got := fmt.Sprintf("exit %d\n--- stdout\n%s--- stderr\n%s", code, stdout.String(), stderr.String())
		got = strings.ReplaceAll(got, home, "$HOME")
		file := name
		if file == "" {
			file = "_usage"
		}
		path := filepath.Join("testdata", "help", file+".txt")
		if *updateHelp {
			if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(path, []byte(got), 0o644); err != nil {
				t.Fatal(err)
			}
			continue
		}
		want, err := os.ReadFile(path)
		if err != nil {
			t.Fatalf("%s: %v (new command? run go test ./internal/cli -run TestHelpGolden -update)", file, err)
		}
		if got != string(want) {
			t.Errorf("warmctl %s -h changed:\n--- got\n%s\n--- want\n%s", name, got, want)
		}
	}
}
