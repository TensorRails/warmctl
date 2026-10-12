package manifest

import (
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"
)

func sample() Manifest {
	d := strings.Repeat("ab", 32)
	return Manifest{
		Schema: Schema, ID: "qwen3-8b-tp1-20261010", CreatedAt: time.Date(2026, 10, 10, 12, 0, 0, 0, time.UTC), Tool: "test",
		Launch:  Launch{Image: "vllm@sha256:" + d, Argv: []string{"vllm", "serve", "/model"}, Env: map[string]string{"A": "1"}, Ranks: 1},
		Process: Process{RootPID: 32768, Owners: []Owner{{PID: 32770, StartTicks: 7, Rank: 0, Role: "worker"}}},
		Files: []File{
			{Path: "criu.tar", Size: 10, SHA256: d, Role: RoleCRIU},
			{Path: "weights/rank-000/part-00000.bin", Size: 100, SHA256: d, Role: RoleWeights},
		},
		Weights: Weights{PartBytes: 1 << 30, Ranks: []Rank{{
			Rank:        0,
			Parts:       []Part{{Index: 0, Path: "weights/rank-000/part-00000.bin", Size: 100, SHA256: d}},
			Allocations: []Allocation{{Index: 0, Ptr: "0x7f0000000000", Bytes: 60, Tag: "weights"}, {Index: 1, Ptr: "0x7f0000010000", Bytes: 40, Tag: "weights", Offset: 60}},
			Fingerprint: Fingerprint{Sum: [3]int64{1, 2, 3}, Allocations: 2, Bytes: 100},
		}}},
	}
}

func TestRoundtrip(t *testing.T) {
	name := filepath.Join(t.TempDir(), "manifest.json")
	want := sample()
	if err := want.Write(name); err != nil {
		t.Fatal(err)
	}
	got, err := Read(name)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("roundtrip changed the manifest:\n%+v\n%+v", got, want)
	}
	if len(got.FirstLane()) != 1 || len(got.WeightLane()) != 1 || got.Bytes() != 110 || got.WeightBytes() != 100 {
		t.Fatal("lanes or sizes are wrong")
	}
}

func TestValidateRefuses(t *testing.T) {
	cases := map[string]func(*Manifest){
		"schema":       func(m *Manifest) { m.Schema = "warmctl.snapshot/v1" },
		"ranks":        func(m *Manifest) { m.Launch.Ranks = 2 },
		"digest":       func(m *Manifest) { m.Files[0].SHA256 = "nope" },
		"absolute":     func(m *Manifest) { m.Files[0].Path = "/criu.tar" },
		"duplicate":    func(m *Manifest) { m.Files = append(m.Files, m.Files[0]) },
		"unlistedPart": func(m *Manifest) { m.Weights.Ranks[0].Parts[0].Path = "elsewhere" },
		"overflow":     func(m *Manifest) { m.Weights.Ranks[0].Allocations[1].Bytes = 41 },
		"badPart":      func(m *Manifest) { m.Weights.Ranks[0].Allocations[0].Part = 1 },
		"rankOrder":    func(m *Manifest) { m.Weights.Ranks[0].Rank = 1 },
	}
	for name, mutate := range cases {
		m := sample()
		mutate(&m)
		if err := m.Validate(); err == nil {
			t.Errorf("%s: accepted", name)
		}
	}
}
