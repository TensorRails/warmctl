package profile_test

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/TensorRails/warmctl/v2"
	"github.com/TensorRails/warmctl/v2/internal/profile"
)

func loader(t *testing.T) (profile.Loader, profile.Profile) {
	t.Helper()
	l := profile.Loader{Assets: warmctl.Assets}
	defaults, err := l.Defaults()
	if err != nil {
		t.Fatal(err)
	}
	return l, defaults
}

func jobValue(body, key string) string {
	for _, line := range strings.Split(body, "\n") {
		if value, ok := strings.CutPrefix(line, key); ok {
			return strings.Trim(strings.TrimSpace(value), "'")
		}
	}
	return ""
}

// The stock A/B baselines of the GPU harness run the profile's flags minus
// warmctl's lifecycle flags. Those job files pin the encoder.
func TestStockArgsMatchHarnessJobs(t *testing.T) {
	l, defaults := loader(t)
	jobs, err := filepath.Glob("../../tests/acceptance/jobs/*.job")
	if err != nil || len(jobs) != 5 {
		t.Fatalf("job files: %v %v", jobs, err)
	}
	for _, job := range jobs {
		raw, err := os.ReadFile(job)
		if err != nil {
			t.Fatal(err)
		}
		body := string(raw)
		id := jobValue(body, "# profile: ")
		p, err := l.Load(id)
		if err != nil {
			t.Fatalf("%s: %v", job, err)
		}
		launch, err := profile.Resolve(defaults, p)
		if err != nil {
			t.Fatalf("%s: %v", job, err)
		}
		if got, want := strings.Join(launch.StockArgs(), " "), jobValue(body, "STOCK_ARGS="); got != want {
			t.Errorf("%s: STOCK_ARGS differ from %s:\n got %s\nwant %s", filepath.Base(job), id, got, want)
		}
		if got, want := launch.Profile.Image, jobValue(body, "STOCK_IMAGE="); got != want {
			t.Errorf("%s: image %s, job %s", id, got, want)
		}
		if len(launch.Argv) != len(launch.StockArgs())+3+len(profile.Lifecycle) {
			t.Errorf("%s: lifecycle flags missing from %v", id, launch.Argv)
		}
	}
}

func TestEveryShippedProfileResolves(t *testing.T) {
	l, defaults := loader(t)
	table, err := l.Hardware()
	if err != nil {
		t.Fatal(err)
	}
	ids, err := l.IDs()
	if err != nil || len(ids) < 20 {
		t.Fatalf("%v %v", ids, err)
	}
	for _, id := range ids {
		p, err := l.Load(id)
		if err != nil {
			t.Fatal(err)
		}
		launch, err := profile.Resolve(defaults, p)
		if err != nil {
			t.Fatal(err)
		}
		if len(launch.Profile.Hardware) == 0 {
			t.Errorf("%s lists no hardware", id)
		}
		for _, hw := range launch.Profile.Hardware {
			if _, ok := table.ByID(hw); !ok {
				t.Errorf("%s: unknown hardware %s", id, hw)
			}
		}
		if len(launch.WarmupCases()) == 0 || len(launch.CaptureSizes) == 0 || launch.Memory == 0 {
			t.Errorf("%s: empty warmup, graphs or memory", id)
		}
		for _, flag := range profile.Lifecycle {
			if !contains(launch.Argv, flag) {
				t.Errorf("%s: %s missing", id, flag)
			}
		}
		if launch.Env["NCCL_SOCKET_IFNAME"] != "=lo" || launch.Env["PYTHONPATH"] != "/opt/warmctl/adapter" {
			t.Errorf("%s: base environment missing", id)
		}
	}
}

func contains(list []string, s string) bool {
	for _, item := range list {
		if item == s {
			return true
		}
	}
	return false
}

func TestArbitraryProfileFromYAML(t *testing.T) {
	l, defaults := loader(t)
	name := filepath.Join(t.TempDir(), "llama31-8b-tp1.yaml")
	body := `schema: warmctl.profile/v1
model: {id: meta-llama/Llama-3.1-8B-Instruct, revision: 0e9e39f249a16976918f6564b8830bc894c89659}
parallel: {tp: 1}
hardware: [h100-80gb-sxm]
vllm: {max-model-len: 8192, kv-cache-memory-bytes: 21474836480, enable-prefix-caching: false}
`
	if err := os.WriteFile(name, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	p, err := l.Load(name)
	if err != nil {
		t.Fatal(err)
	}
	launch, err := profile.Resolve(defaults, p)
	if err != nil {
		t.Fatal(err)
	}
	args := strings.Join(launch.Argv, " ")
	for _, want := range []string{"vllm serve /model ", "--max-model-len 8192", "--kv-cache-memory-bytes 21474836480", "--served-model-name llama31-8b-tp1", "--no-enable-prefix-caching", "--enable-sleep-mode", "--max-num-seqs 8", `"cudagraph_capture_sizes":[1,2,4,8]`} {
		if !strings.Contains(args, want) {
			t.Errorf("argv lacks %q: %s", want, args)
		}
	}
	if launch.Profile.Image != defaults.Image || launch.Profile.Canary.Mode != "warn" || launch.Memory != 64<<30 {
		t.Errorf("defaults not applied: %+v", launch.Profile)
	}
	if n := len(launch.WarmupCases()); n != 12 {
		t.Errorf("generic warmup grid has %d cases", n)
	}
}

func TestNullRemovesDefault(t *testing.T) {
	_, defaults := loader(t)
	p := profile.Profile{Schema: profile.Schema, ID: "x", Model: profile.Model{ID: "m"}, Parallel: profile.Parallel{TP: 1},
		VLLM: map[string]any{"disable-custom-all-reduce": nil}, Env: map[string]*string{"NCCL_NVLS_ENABLE": nil}}
	launch, err := profile.Resolve(defaults, p)
	if err != nil {
		t.Fatal(err)
	}
	if contains(launch.Argv, "--disable-custom-all-reduce") {
		t.Error("null did not remove the option")
	}
	if _, ok := launch.Env["NCCL_NVLS_ENABLE"]; ok {
		t.Error("null did not remove the variable")
	}
}

func TestRefusals(t *testing.T) {
	_, defaults := loader(t)
	base := func() profile.Profile {
		return profile.Profile{Schema: profile.Schema, ID: "x", Model: profile.Model{ID: "m"}, Parallel: profile.Parallel{TP: 2}, VLLM: map[string]any{}, Env: map[string]*string{}}
	}
	secret := "s"
	cases := map[string]func(*profile.Profile){
		"pipeline":   func(p *profile.Profile) { p.VLLM["pipeline-parallel-size"] = 2 },
		"deepep":     func(p *profile.Profile) { p.VLLM["all2all-backend"] = "deepep_low_latency" },
		"credential": func(p *profile.Profile) { p.Env["HF_TOKEN"] = &secret },
		"owned":      func(p *profile.Profile) { p.Env["CUDA_VISIBLE_DEVICES"] = &secret },
		"image":      func(p *profile.Profile) { p.Image = "vllm/vllm-openai:latest" },
		"mount":      func(p *profile.Profile) { p.Model.ContainerPath = "model" },
		"engram": func(p *profile.Profile) {
			p.Parallel.DP = 2
			p.VLLM["engram-config"] = map[string]any{"cpu_offload": true}
		},
		"canary":    func(p *profile.Profile) { p.Canary.Mode = "strict" },
		"badflag":   func(p *profile.Profile) { p.VLLM["Max Len"] = 1 },
		"emptylist": func(p *profile.Profile) { p.VLLM["served-model-name"] = []any{} },
	}
	for name, mutate := range cases {
		p := base()
		mutate(&p)
		if _, err := profile.Resolve(defaults, p); err == nil {
			t.Errorf("%s: accepted", name)
		}
	}
	if _, err := profile.Resolve(defaults, base()); err != nil {
		t.Fatalf("base profile refused: %v", err)
	}
}

func TestWarmupGrids(t *testing.T) {
	l, defaults := loader(t)
	p, _ := l.Load("deepseek-v41-flash-prod-tp8")
	launch, err := profile.Resolve(defaults, p)
	if err != nil {
		t.Fatal(err)
	}
	cases := launch.WarmupCases()
	seen := map[profile.WarmupCase]bool{}
	for _, c := range cases {
		if seen[c] {
			t.Errorf("duplicate warmup case %+v", c)
		}
		seen[c] = true
	}
	for _, want := range []profile.WarmupCase{{32, 128, 32}, {32, 1, 32}, {512, 1, 32}, {12288, 1, 32}} {
		if !seen[want] {
			t.Errorf("missing %+v", want)
		}
	}
	if launch.Speculative != 5 || launch.CaptureSizes[0] != 6 || launch.CaptureSizes[12] != 768 || launch.Mode != "FULL_AND_PIECEWISE" {
		t.Errorf("graphs: k=%d sizes=%v mode=%s", launch.Speculative, launch.CaptureSizes, launch.Mode)
	}
	p, _ = l.Load("glm53-flash-prod-tp8")
	launch, _ = profile.Resolve(defaults, p)
	cases = launch.WarmupCases()
	if cases[0] != (profile.WarmupCase{4, 1, 32}) || !containsCase(cases, profile.WarmupCase{16384, 1, 32}) || !containsCase(cases, profile.WarmupCase{512, 8, 32}) || containsCase(cases, profile.WarmupCase{32, 1, 32}) == false {
		t.Errorf("glm warmup grid: %v", cases)
	}
	if n := len(cases); n < 70 || n > 90 {
		t.Errorf("glm warmup grid has %d cases", n)
	}
}

func containsCase(cases []profile.WarmupCase, c profile.WarmupCase) bool {
	for _, item := range cases {
		if item == c {
			return true
		}
	}
	return false
}

func TestSizes(t *testing.T) {
	for in, want := range map[string]int64{"1TiB": 1 << 40, "64GiB": 64 << 30, "16KiB": 16 << 10, "1024": 1024, "1.5GiB": 3 << 29, "2GB": 2e9} {
		if got, err := profile.ParseSize(in); err != nil || got != want {
			t.Errorf("%s: %d %v", in, got, err)
		}
	}
	for _, bad := range []string{"", "-1", "x", "1XB"} {
		if _, err := profile.ParseSize(bad); err == nil {
			t.Errorf("%q accepted", bad)
		}
	}
	if profile.FormatSize(1<<40) != "1TiB" || profile.FormatSize(1536) != "1536" {
		t.Error("format")
	}
}
