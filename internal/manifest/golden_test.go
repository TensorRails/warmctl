package manifest_test

import (
	"bufio"
	"bytes"
	"compress/gzip"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"

	"github.com/TensorRails/warmctl/v2/internal/manifest"
)

var update = flag.Bool("update", false, "rewrite testdata/golden.sha256 from the current code")

// goldenFixtures are manifest.json files of snapshots earlier releases
// published, GPU UUIDs anonymised (the B300 one also keeps only 40 of its
// CRIU images). Whatever the code becomes, it must read them and write them
// back byte for byte as it does today: snapshots already in registries and
// on disks keep restoring, and a bake of today's kind keeps its manifest.
var goldenFixtures = []struct {
	file    string
	backend string
	ipc     string
}{
	{"dmtcp-h200-tp1-beta17.json", manifest.BackendDMTCP, ""},
	{"criu-interpose-b300-tp8-beta9.json.gz", manifest.BackendCRIU, "interpose"},
}

func TestGoldenManifests(t *testing.T) {
	want := readGolden(t)
	got := map[string]string{}
	for _, f := range goldenFixtures {
		raw := readFixture(t, f.file)
		dir := t.TempDir()
		in := filepath.Join(dir, "in.json")
		if err := os.WriteFile(in, raw, 0o644); err != nil {
			t.Fatal(err)
		}
		m, err := manifest.Read(in)
		if err != nil {
			t.Fatalf("%s: a published snapshot no longer reads: %v", f.file, err)
		}
		if m.Process.BackendName() != f.backend || m.Process.IPC != f.ipc {
			t.Errorf("%s: backend %q ipc %q, want %q %q", f.file, m.Process.BackendName(), m.Process.IPC, f.backend, f.ipc)
		}
		out := filepath.Join(dir, "out.json")
		if err := m.Write(out); err != nil {
			t.Fatalf("%s: %v", f.file, err)
		}
		written, err := os.ReadFile(out)
		if err != nil {
			t.Fatal(err)
		}
		// A field added without omitempty would show up in every manifest
		// written from now on: name it rather than only report a digest.
		if extra := newKeys(t, raw, written); len(extra) > 0 {
			t.Errorf("%s: writing adds keys the published snapshot does not have: %v", f.file, extra)
		}
		sum := sha256.Sum256(written)
		got[f.file] = hex.EncodeToString(sum[:])
		if !*update && got[f.file] != want[f.file] {
			t.Errorf("%s: the manifest written back changed (sha256 %s, golden %s); diff %s against the "+
				"output of the base branch", f.file, got[f.file], want[f.file], keep(t, f.file, written))
		}
	}
	if *update {
		writeGolden(t, got)
	}
}

func readFixture(t *testing.T, name string) []byte {
	t.Helper()
	f, err := os.Open(filepath.Join("testdata", name))
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	var r io.Reader = f
	if strings.HasSuffix(name, ".gz") {
		z, err := gzip.NewReader(f)
		if err != nil {
			t.Fatal(err)
		}
		defer z.Close()
		r = z
	}
	raw, err := io.ReadAll(r)
	if err != nil {
		t.Fatal(err)
	}
	return raw
}

// newKeys lists the JSON key paths of b that a does not have, array
// indices folded into "[]".
func newKeys(t *testing.T, a, b []byte) []string {
	t.Helper()
	paths := func(raw []byte) map[string]bool {
		var v any
		if err := json.Unmarshal(raw, &v); err != nil {
			t.Fatal(err)
		}
		out := map[string]bool{}
		var walk func(string, any)
		walk = func(p string, v any) {
			switch x := v.(type) {
			case map[string]any:
				for k, child := range x {
					out[p+"."+k] = true
					walk(p+"."+k, child)
				}
			case []any:
				for _, child := range x {
					walk(p+"[]", child)
				}
			}
		}
		walk("", v)
		return out
	}
	have := paths(a)
	var extra []string
	for p := range paths(b) {
		if !have[p] {
			extra = append(extra, p)
		}
	}
	sort.Strings(extra)
	return extra
}

// keep saves what the code wrote, for a diff, and returns where.
func keep(t *testing.T, name string, raw []byte) string {
	t.Helper()
	p := filepath.Join(os.TempDir(), "warmctl-golden-"+strings.TrimSuffix(name, ".gz"))
	if err := os.WriteFile(p, raw, 0o644); err != nil {
		return "(not saved: " + err.Error() + ")"
	}
	return p
}

func readGolden(t *testing.T) map[string]string {
	t.Helper()
	out := map[string]string{}
	f, err := os.Open(filepath.Join("testdata", "golden.sha256"))
	if os.IsNotExist(err) && *update {
		return out
	}
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	s := bufio.NewScanner(f)
	for s.Scan() {
		sum, name, ok := strings.Cut(s.Text(), "  ")
		if ok {
			out[name] = sum
		}
	}
	return out
}

func writeGolden(t *testing.T, sums map[string]string) {
	t.Helper()
	var b bytes.Buffer
	names := make([]string, 0, len(sums))
	for n := range sums {
		names = append(names, n)
	}
	sort.Strings(names)
	for _, n := range names {
		fmt.Fprintf(&b, "%s  %s\n", sums[n], n)
	}
	if err := os.WriteFile(filepath.Join("testdata", "golden.sha256"), b.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
}

func writeFile(name string, raw []byte) error { return os.WriteFile(name, raw, 0o644) }
