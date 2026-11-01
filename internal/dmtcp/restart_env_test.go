package dmtcp

import (
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

func TestRestartEnv(t *testing.T) {
	for _, tc := range []struct{ cpus, readers int }{
		{0, 1}, {1, 1}, {2, 2}, {4, 4}, {8, 8}, {9, 8}, {15, 8}, {16, 16}, {24, 16},
	} {
		t.Run(strconv.Itoa(tc.cpus), func(t *testing.T) {
			got := strings.Join(restartEnv([]string{"UNCHANGED=yes"}, tc.cpus), "\n")
			want := "UNCHANGED=yes\nDMTCP_RESTART_HUGEPAGES=1\nDMTCP_RESTART_READERS=" +
				strconv.Itoa(tc.readers) + "\nDMTCP_RESTART_MMAP=0"
			if got != want {
				t.Fatalf("got %q, want %q", got, want)
			}
		})
	}
	explicit := []string{"DMTCP_RESTART_HUGEPAGES=0", "DMTCP_RESTART_READERS=1", "DMTCP_RESTART_MMAP="}
	if got := strings.Join(restartEnv(explicit, 32), "\n"); got != strings.Join(explicit, "\n") {
		t.Fatalf("overrode explicit settings: %q", got)
	}
	prefixed := restartEnv([]string{"DMTCP_RESTART_READERS_EXTRA=1"}, 16)
	if !strings.Contains(strings.Join(prefixed, "\n"), "DMTCP_RESTART_READERS=16") {
		t.Fatal("mistook a variable prefix for an override")
	}
}

func TestRestartPassesReaderEnvironment(t *testing.T) {
	dir := t.TempDir()
	output := filepath.Join(dir, "environment")
	script := "#!/bin/sh\nprintf '%s\\n' \"$DMTCP_RESTART_HUGEPAGES\" \"$DMTCP_RESTART_READERS\" \"$DMTCP_RESTART_MMAP\" > \"$RESTART_ENV_OUTPUT\"\n"
	if err := os.WriteFile(filepath.Join(dir, "dmtcp_restart"), []byte(script), 0o755); err != nil {
		t.Fatal(err)
	}
	t.Setenv("RESTART_ENV_OUTPUT", output)
	t.Setenv("DMTCP_RESTART_HUGEPAGES", "0")
	t.Setenv("DMTCP_RESTART_READERS", "4")
	t.Setenv("DMTCP_RESTART_MMAP", "0")
	cmd, err := (Tools{Dir: dir}).Restart([]string{"unused.dmtcp"}, io.Discard)
	if err != nil {
		t.Fatal(err)
	}
	if err := cmd.Wait(); err != nil {
		t.Fatal(err)
	}
	got, err := os.ReadFile(output)
	if err != nil || string(got) != "0\n4\n0\n" {
		t.Fatalf("restart environment = %q, %v", got, err)
	}
}

func TestDefaultToolsBundle(t *testing.T) {
	prefix := t.TempDir()
	self := filepath.Join(prefix, "warmctl")
	if err := os.WriteFile(self, nil, 0o755); err != nil {
		t.Fatal(err)
	}
	link := filepath.Join(t.TempDir(), "warmctl")
	if err := os.Symlink(self, link); err != nil {
		t.Fatal(err)
	}
	prefix, err := filepath.EvalSymlinks(prefix)
	if err != nil {
		t.Fatal(err)
	}
	root := filepath.Join(prefix, "dmtcp", "linux-amd64", "v5.0.0-warmctl.1")
	if err := os.MkdirAll(root, 0o755); err != nil {
		t.Fatal(err)
	}
	for _, path := range []string{self, link} {
		tools := defaultTools(path, "linux-amd64", "v5.0.0-warmctl.1")
		if tools.Dir != filepath.Join(root, "bin") || tools.Plugin != filepath.Join(root, "lib", "libdmtcp_fdkeep.so") {
			t.Fatalf("wrong bundle for %s: %+v", path, tools)
		}
	}
	if tools := defaultTools(self, "linux-amd64", "missing"); tools.Dir != DefaultDir || tools.Plugin != DefaultPlugin {
		t.Fatalf("source build lost runtime image defaults: %+v", tools)
	}
}
