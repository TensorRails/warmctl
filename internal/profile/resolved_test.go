package profile_test

import (
	"bufio"
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"

	"github.com/TensorRails/warmctl/v2/internal/profile"
)

var update = flag.Bool("update", false, "rewrite testdata/resolved.sha256 from the current code")

// Every shipped profile resolves to the launch it resolves to today, and the
// hardware table reads as it does today: a change to the loader, the
// defaults or the encoder that is meant for another GPU vendor must not
// move a single byte of an existing launch. testdata/resolved.sha256 holds
// the SHA-256 of each launch's JSON; go test -run TestResolvedProfilesGolden
// -update rewrites it, which a change of an existing profile does on purpose.
func TestResolvedProfilesGolden(t *testing.T) {
	l, defaults := loader(t)
	ids, err := l.IDs()
	if err != nil {
		t.Fatal(err)
	}
	table, err := l.Hardware()
	if err != nil {
		t.Fatal(err)
	}
	amd := l
	amd.Vendor = "amd"
	amdDefaults, err := amd.Defaults()
	if err != nil {
		t.Fatal(err)
	}
	got := map[string]string{}
	encoded := map[string][]byte{}
	for _, id := range ids {
		p, err := l.Load(id)
		if err != nil {
			t.Fatal(err)
		}
		base := defaults
		// A profile for AMD rows runs on an AMD host: the ROCm layer applies.
		if table.VendorOf(p) == "amd" {
			base = amdDefaults
		}
		launch, err := profile.Resolve(base, p)
		if err != nil {
			t.Fatal(err)
		}
		raw, err := json.MarshalIndent(launch, "", "  ")
		if err != nil {
			t.Fatal(err)
		}
		encoded[id] = raw
		got[id] = digest(raw)
	}
	// The NVIDIA rows, which manifests embed: rows of another vendor may be
	// added, these may not change.
	nvidia := profile.HardwareTable{Schema: table.Schema}
	for _, r := range table.Rows {
		if r.Vendor == "" {
			nvidia.Rows = append(nvidia.Rows, r)
		}
	}
	raw, err := json.MarshalIndent(nvidia, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	encoded["hardware-table"] = raw
	got["hardware-table"] = digest(raw)

	golden := filepath.Join("testdata", "resolved.sha256")
	if *update {
		writeSums(t, golden, got)
		return
	}
	want := readSums(t, golden)
	for id, sum := range got {
		w, ok := want[id]
		switch {
		case !ok:
			t.Errorf("%s: new profile; run go test ./internal/profile -run TestResolvedProfilesGolden -update", id)
		case w != sum:
			p := filepath.Join(os.TempDir(), "warmctl-resolved-"+id+".json")
			_ = os.WriteFile(p, encoded[id], 0o644)
			t.Errorf("%s: resolves differently than before (written to %s for a diff against the base branch)", id, p)
		}
	}
	for id := range want {
		if _, ok := got[id]; !ok {
			t.Errorf("%s: profile removed; run with -update if that is intended", id)
		}
	}
}

func digest(raw []byte) string {
	sum := sha256.Sum256(raw)
	return hex.EncodeToString(sum[:])
}

func readSums(t *testing.T, name string) map[string]string {
	t.Helper()
	f, err := os.Open(name)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	out := map[string]string{}
	s := bufio.NewScanner(f)
	for s.Scan() {
		if sum, id, ok := strings.Cut(s.Text(), "  "); ok {
			out[id] = sum
		}
	}
	return out
}

func writeSums(t *testing.T, name string, sums map[string]string) {
	t.Helper()
	ids := make([]string, 0, len(sums))
	for id := range sums {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	var b bytes.Buffer
	for _, id := range ids {
		fmt.Fprintf(&b, "%s  %s\n", sums[id], id)
	}
	if err := os.MkdirAll(filepath.Dir(name), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(name, b.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
}
