//go:build linux && e2e

// Package e2e_test runs the real warmctl binary through a whole
// snapshot and its restores on a CPU host, on the host and in the pod
// image as Kubernetes would run it: Docker, docker-init, the agent,
// CRIU, the archived trees, the content-addressed store, the manifest, the
// gateway and warmctl's Python extension, around a stand-in vLLM whose
// "GPU memory" is host memory. Only CUDA itself is left out.
//
// Needs Docker and root. Run through `make e2e-test`.
package e2e_test

import (
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/TensorRails/warmctl/v2/internal/docker"
	"github.com/TensorRails/warmctl/v2/internal/manifest"
)

var (
	warmctl  = flag.String("warmctl", "", "linux warmctl binary")
	image    = flag.String("image", "warmctl-e2e:local", "image with CRIU, Python and the stand-in vLLM")
	testdata = flag.String("testdata", "", "tests/integration/e2e/testdata")
	registry = flag.String("registry", "", "plain-HTTP registry for the push/pull leg, e.g. localhost:5000")
)

const listen = "127.0.0.1:18000"

type output struct {
	mu  sync.Mutex
	buf bytes.Buffer
}

func (o *output) Write(p []byte) (int, error) {
	o.mu.Lock()
	defer o.mu.Unlock()
	return o.buf.Write(p)
}

func (o *output) String() string {
	o.mu.Lock()
	defer o.mu.Unlock()
	return o.buf.String()
}

type env struct {
	t     *testing.T
	base  string
	state string
	path  string
	extra []string // environment for the next commands, after the defaults
}

func (e *env) command(ctx context.Context, args ...string) *exec.Cmd {
	cmd := exec.CommandContext(ctx, *warmctl, args...)
	cmd.Env = append(os.Environ(), "PATH="+e.path)
	if *registry != "" {
		cmd.Env = append(cmd.Env, "WARMCTL_REGISTRY="+*registry+"/e2e")
	}
	cmd.Env = append(cmd.Env, e.extra...)
	return cmd
}

func (e *env) run(args ...string) string {
	e.t.Helper()
	out, err := e.try(args...)
	if err != nil {
		e.t.Fatalf("warmctl %s: %v", args[0], err)
	}
	return out
}

func (e *env) try(args ...string) (string, error) {
	e.t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()
	out, err := e.command(ctx, args...).CombinedOutput()
	e.t.Logf("warmctl %s\n%s", strings.Join(args, " "), out)
	return string(out), err
}

func (e *env) snapshot(id, output string, extra ...string) (string, error) {
	args := append([]string{"build", "--test-cpu", "--profile", "fake-tp2", "--profile-dir",
		filepath.Join(e.base, "profiles"), "--model-dir", filepath.Join(e.base, "model"), "--gpus", "0,1",
		"--output", output, "--state-dir", e.state, "--id", id, "--part-bytes", "300000"}, extra...)
	return e.try(args...)
}

// v1 P202: a failed snapshot must leave no container, no dump, no partial
// output and no GPU claim, and must keep its logs.
func (e *env) failedSnapshot() {
	t := e.t
	out := filepath.Join(e.base, "snapshots", "failed")
	text, err := e.snapshot("e2e-failed", out, "--set", "env.FAKE_VLLM_FAIL=1")
	if err == nil || !strings.Contains(text, "engine exited") {
		t.Fatalf("a failing engine did not fail the snapshot: %v", err)
	}
	for _, gone := range []string{out, filepath.Join(e.state, "bakes", "e2e-failed")} {
		if _, err := os.Stat(gone); err == nil {
			t.Errorf("%s left behind", gone)
		}
	}
	for _, kept := range []string{"engine.log", "agent.log"} {
		if raw, err := os.ReadFile(filepath.Join(e.state, "logs", "e2e-failed", kept)); err != nil || len(raw) == 0 {
			t.Errorf("log %s not kept: %v", kept, err)
		}
	}
	e.noContainers()
}

func (e *env) noContainers() {
	e.t.Helper()
	dk, _ := docker.New(context.Background(), docker.DefaultSocket)
	if left, err := dk.List(context.Background(), docker.ManagedLabel); err != nil || len(left) != 0 {
		e.t.Errorf("containers left behind: %+v %v", left, err)
	}
}

func setup(t *testing.T) *env {
	if *warmctl == "" || *testdata == "" {
		t.Skip("needs -warmctl and -testdata")
	}
	ctx := context.Background()
	dk, err := docker.New(ctx, docker.DefaultSocket)
	if err != nil {
		t.Fatal(err)
	}
	img, err := dk.InspectImage(ctx, *image)
	if err != nil {
		t.Fatalf("image %s: %v (make e2e-test builds it)", *image, err)
	}
	base := t.TempDir()
	e := &env{t: t, base: base, state: filepath.Join(base, "state"), path: *testdata + ":" + os.Getenv("PATH")}
	write := func(name, body string) {
		if err := os.MkdirAll(filepath.Dir(name), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(name, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	write(filepath.Join(base, "model", "config.json"), `{"seed": "e2e"}`)
	hw, err := os.ReadFile(filepath.Join(*testdata, "hardware.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	write(filepath.Join(base, "profiles", "hardware.yaml"), string(hw))
	write(filepath.Join(base, "profiles", "fake-tp2.yaml"), fmt.Sprintf(`schema: warmctl.profile/v1
id: fake-tp2
image: %s
model: {id: fake/model, revision: "0", served_names: [fake]}
parallel: {tp: 2}
hardware: [fake-gpu]
warmup: {prefill_tokens: [64, 128], decode_prompt_tokens: 32}
canary: {mode: bitwise, prefix_tokens: 64}
resources: {memory: 2GiB, pids: 4096, shm: 64MiB, cpus: 2}
`, img.ID))
	return e
}

func TestSnapshotRestore(t *testing.T) {
	e := setup(t)
	e.failedSnapshot()
	snap := filepath.Join(e.base, "snapshots", "e2e")
	out, err := e.snapshot("e2e", snap, "--set", "env.FAKE_VLLM_HOLD_MODEL=1")
	if err != nil || !strings.Contains(out, "snapshot completed") {
		t.Fatalf("snapshot did not complete: %v", err)
	}
	m, err := manifest.Read(filepath.Join(snap, "manifest.json"))
	if err != nil {
		t.Fatal(err)
	}
	if len(m.Weights.Ranks) != 2 || len(m.Weights.Ranks[0].Parts) < 3 || len(m.Weights.Ranks[1].Allocations) != 3 {
		t.Errorf("weights: %d ranks, %d parts, %d allocations", len(m.Weights.Ranks), len(m.Weights.Ranks[0].Parts), len(m.Weights.Ranks[1].Allocations))
	}
	if !m.Process.NeedsModel || !m.Process.CollectivesDetached || m.Canary == nil || m.Launch.Model != "fake" {
		t.Errorf("process %+v, canary %v, model %q", m.Process, m.Canary != nil, m.Launch.Model)
	}
	r0, r1 := m.Weights.Ranks[0], m.Weights.Ranks[1]
	if r0.Allocations[0].Ptr != r1.Allocations[0].Ptr || r0.Parts[0].SHA256 != r1.Parts[0].SHA256 || r0.Parts[1].SHA256 == r1.Parts[1].SHA256 {
		t.Errorf("ranks should share addresses and the first part only: %+v %+v", r0.Parts, r1.Parts)
	}
	e.run("verify", snap, "--deep")
	e.restore(snap, "e2e-r1", "--gpus", "0,1")
	// v1 P149: listing the GPUs in another order must not reorder ranks.
	e.restore(snap, "e2e-r2", "--gpus", "1,0")
	if *registry != "" {
		ref := "oci://" + *registry + "/e2e/fake-tp2:v1"
		// 1 MiB pieces: the weight parts split, small files go in bundles.
		out := e.run("push", snap, ref, "--plain-http", "--piece-size", "1MiB")
		blobs, sent := transferred(t, out, "uploaded")
		if blobs != sent || blobs < 3 {
			t.Errorf("first push did not upload every distinct blob once")
		}
		if out := e.run("push", snap, ref, "--plain-http", "--piece-size", "1MiB"); !strings.Contains(out, ", 0 uploaded") {
			t.Errorf("second push uploaded blobs again")
		}
		if out := e.run("ls", "oci://"+*registry+"/e2e/fake-tp2", "--plain-http"); strings.TrimSpace(lastLine(out)) != "v1" {
			t.Errorf("tags: %q", out)
		}
		// Another host: an empty store downloads every blob and checks it.
		other := filepath.Join(e.base, "other-host")
		pulled := filepath.Join(other, "snapshots", "e2e")
		out = e.run("pull", ref, "--plain-http", "--state-dir", filepath.Join(other, "state"), "--output", pulled)
		if n, got := transferred(t, out, "downloaded"); n != blobs || got != blobs {
			t.Errorf("pull into an empty store did not download every distinct blob once")
		}
		e.run("verify", pulled, "--deep")
		// Without --id the instance is named after the repository.
		e.restore(ref, "", "--plain-http")
		// A short name: push files the snapshot under its platform tag in
		// the repository's index, and up finds it there.
		if out := e.run("push", snap, "fake-tp2-short", "--plain-http", "--piece-size", "1MiB"); !strings.Contains(out, "which lists 1 platforms") {
			t.Errorf("push by name did not index the snapshot")
		}
		e.restore("fake-tp2-short", "", "--plain-http")
		// The catalog lists the model with the platform this host restores.
		cat := filepath.Join(e.base, "catalog.yaml")
		if err := os.WriteFile(cat, []byte("models:\n  - {name: fake-tp2-short, title: Fake, license: Apache-2.0}\n"+
			"  - {name: absent, license: MIT}\n"), 0o644); err != nil {
			t.Fatal(err)
		}
		e.run("catalog", "--publish", cat, "--plain-http")
		if out := e.run("catalog", "--plain-http", "--state-dir", e.state); !strings.Contains(out, "fake-tp2-short") ||
			!strings.Contains(out, "2xfake-gpu") || !strings.Contains(out, "pulled") {
			t.Errorf("catalog does not list the pulled model: %q", out)
		}
		// A shared cache: mirror the model into a directory, then up from it
		// with the registry out of reach.
		cache := filepath.Join(e.base, "cache")
		if out := e.run("mirror", "fake-tp2-short", cache, "--plain-http"); !strings.Contains(out, "mirrored 1 snapshots") {
			t.Errorf("mirror into the cache: %q", out)
		}
		if out := e.run("mirror", "fake-tp2-short", cache, "--plain-http"); !strings.Contains(out, ", 0 sent") {
			t.Errorf("a second mirror sent blobs again: %q", out)
		}
		if err := os.RemoveAll(filepath.Join(e.state, "pulled")); err != nil {
			t.Fatal(err)
		}
		e.extra = []string{"WARMCTL_CACHE=" + cache, "WARMCTL_REGISTRY=localhost:1/e2e"}
		e.restore("fake-tp2-short", "", "--plain-http")
		e.extra = nil
	}
	e.podLegs(snap)
	e.noContainers()
}

// transferred reads "N files in B blobs, K <verb>" from a push or pull.
func transferred(t *testing.T, out, verb string) (int, int) {
	t.Helper()
	m := regexp.MustCompile(`(\d+) files in (\d+) blobs, (\d+) ` + verb).FindStringSubmatch(out)
	if m == nil {
		t.Fatalf("no transfer summary in %q", out)
	}
	blobs, _ := strconv.Atoi(m[2])
	n, _ := strconv.Atoi(m[3])
	return blobs, n
}

func lastLine(s string) string {
	lines := strings.Split(strings.TrimSpace(s), "\n")
	return lines[len(lines)-1]
}

// restore runs `up` until a completion went through the gateway, checks
// `ps`, and stops it with `down`. An empty id lets up name the instance.
func (e *env) restore(snap, id string, extra ...string) {
	t := e.t
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()
	var out output
	args := []string{"up", snap, "--test-cpu", "--state-dir", e.state, "--listen", listen}
	if id != "" {
		args = append(args, "--id", id)
	} else {
		id = filepath.Base(strings.SplitN(snap, ":v", 2)[0])
	}
	args = append(args, extra...)
	cmd := e.command(ctx, args...)
	cmd.Stdout, cmd.Stderr = &out, &out
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	exited := make(chan error, 1)
	go func() { exited <- cmd.Wait() }()
	defer func() { t.Logf("warmctl up %s\n%s", id, out.String()) }()
	await := func(marker string) {
		deadline := time.Now().Add(2 * time.Minute)
		for !strings.Contains(out.String(), marker) {
			select {
			case err := <-exited:
				t.Fatalf("up exited before %q: %v", marker, err)
			case <-time.After(100 * time.Millisecond):
			}
			if time.Now().After(deadline) {
				t.Fatalf("no %q", marker)
			}
		}
	}
	await("serving started")
	body := `{"model":"fake","prompt":[1,2,3,4],"max_tokens":4,"temperature":0}`
	resp, err := http.Post("http://"+listen+"/v1/completions", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	raw, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	var reply struct {
		Choices []json.RawMessage `json:"choices"`
	}
	if resp.StatusCode != 200 || json.Unmarshal(raw, &reply) != nil || len(reply.Choices) != 1 {
		t.Fatalf("completion through the gateway: %d %s", resp.StatusCode, raw)
	}
	await("first host-gateway token")
	if resp, err := http.Post("http://"+listen+"/sleep?level=1", "application/json", nil); err != nil || resp.StatusCode != 404 {
		t.Fatalf("gateway let a management route through: %v", err)
	}
	if ps := e.run("ps", "--state-dir", e.state); !strings.Contains(ps, id) || !strings.Contains(ps, "running") {
		t.Errorf("ps does not list %s running: %q", id, ps)
	}
	e.run("down", id, "--state-dir", e.state)
	select {
	case err := <-exited:
		if err != nil {
			t.Fatalf("up exited with %v", err)
		}
	case <-time.After(2 * time.Minute):
		t.Fatal("up did not exit after down")
	}
}
