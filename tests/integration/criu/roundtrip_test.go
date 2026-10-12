//go:build linux && criu

// Package criu_test is the CPU half of a snapshot, end to end: a donor
// container runs the agent, which starts a stand-in engine, freezes it,
// dumps it with CRIU and archives its trees; a second container installs
// the trees, restores the tree stopped and thaws it; the engine's memory,
// deleted file, shared mapping and listening socket must all have survived.
//
// Needs Docker, root (the snapshot files are root's) and the image built by
// `make criu-image`. Run through `make criu-test`.
package criu_test

import (
	"bufio"
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"net"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/TensorRails/warmctl/v2/internal/agent"
	"github.com/TensorRails/warmctl/v2/internal/docker"
)

var (
	binDir = flag.String("bin", "", "directory with the linux warmctl and witness binaries")
	image  = flag.String("image", "warmctl-criu-test:local", "image with CRIU")
	socket = flag.String("docker", docker.DefaultSocket, "Docker socket")
)

type observation struct {
	Nonce    string `json:"nonce"`
	Counter  int    `json:"counter"`
	Sequence int    `json:"sequence"`
	PID      int    `json:"pid"`
	Deleted  string `json:"deleted_sha256"`
	Shared   uint64 `json:"shared"`
}

type harness struct {
	t     *testing.T
	ctx   context.Context
	dk    *docker.Client
	image string
	base  string
	work  string
	tag   string
}

func TestRoundtrip(t *testing.T) {
	if *binDir == "" {
		t.Skip("needs -bin with the linux warmctl and witness binaries")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()
	h := setup(ctx, t)

	donorIO, donorWS := h.dir("donor-io"), h.dir("donor-ws")
	snap := agent.Config{
		Mode: agent.ModeSnapshot, Control: "/state/control.sock", CRIU: "/usr/local/sbin/criu", Trees: agent.DefaultTrees,
		Argv: []string{"/work/witness", "/workspace/witness"}, Env: []string{"PATH=/usr/bin:/bin"},
		Log: "/workspace/logs/engine.log", Out: "/state/out",
	}
	donor := h.start("donor", snap, []docker.Mount{{Source: donorIO, Target: "/state"}, {Source: donorWS, Target: "/workspace"}})
	c := h.dial(filepath.Join(donorIO, "control.sock"))
	witness := filepath.Join(donorWS, "witness")
	h.await(witness, "main", 0)
	h.request(witness, 1)
	before, beforeWorker := h.await(witness, "main", 1), h.await(witness, "worker", 1)

	var cudaRes agent.CUDAResult
	h.call(c, "cuda", &cudaRes)
	var frozen agent.FreezeResult
	h.call(c, "freeze", &frozen)
	if len(frozen.Tree.Processes) != 2 || frozen.Tree.Root != before.PID {
		t.Fatalf("frozen tree %+v, witness main %d", frozen.Tree, before.PID)
	}
	var dumped agent.DumpResult
	h.call(c, "dump", &dumped)
	var archived agent.FSResult
	h.call(c, "fs", &archived)
	h.call(c, "finish", nil)
	if code := h.exit(donor); code != 0 {
		t.Fatalf("donor agent exited %d", code)
	}
	t.Logf("donor: dump %.2fs (criu %s), trees %+v", dumped.Seconds, dumped.Version, archived.Trees)

	recvIO, recvWS := h.dir("recv-io"), h.dir("recv-ws")
	restore := agent.Config{
		Mode: agent.ModeRestore, Control: "/state/control.sock", CRIU: "/usr/local/sbin/criu", Trees: agent.DefaultTrees,
		Images: "/capture/criu", FS: "/capture/fs", Work: "/state/work", Root: frozen.Tree.Root, MaxID: frozen.Tree.MaxID(),
	}
	receiver := h.start("receiver", restore, []docker.Mount{
		{Source: filepath.Join(donorIO, "out"), Target: "/capture", ReadOnly: true},
		{Source: recvIO, Target: "/state"}, {Source: recvWS, Target: "/workspace"},
	})
	defer h.stop(receiver)
	c = h.dial(filepath.Join(recvIO, "control.sock"))
	start := time.Now()
	var installed agent.FSResult
	h.call(c, "install-fs", &installed)
	var restored agent.RestoreResult
	h.call(c, "criu", &restored)
	h.call(c, "thaw", nil)
	h.call(c, "cuda", nil)
	t.Logf("receiver: install %.2fs, criu restore %.2fs, total %.2fs", installed.Seconds, restored.Seconds, time.Since(start).Seconds())

	witness = filepath.Join(recvWS, "witness")
	h.request(witness, 2)
	after, afterWorker := h.await(witness, "main", 2), h.await(witness, "worker", 2)
	if after.Nonce != before.Nonce || after.PID != before.PID || after.Counter != 2 {
		t.Errorf("main lost its memory: before %+v after %+v", before, after)
	}
	if afterWorker.Nonce != beforeWorker.Nonce || afterWorker.PID != beforeWorker.PID || afterWorker.Counter != 2 {
		t.Errorf("worker lost its memory: before %+v after %+v", beforeWorker, afterWorker)
	}
	if after.Deleted == "" || after.Deleted != before.Deleted {
		t.Errorf("deleted file changed: %s -> %s", before.Deleted, after.Deleted)
	}
	if before.Shared != 1 || after.Shared != 2 {
		t.Errorf("shared mapping lost its counter: %d -> %d", before.Shared, after.Shared)
	}
	if got := h.ask(filepath.Join(witness, "witness.sock")); got != before.Nonce {
		t.Errorf("restored socket answered %q, want %q", got, before.Nonce)
	}
}

func setup(ctx context.Context, t *testing.T) *harness {
	dk, err := docker.New(ctx, *socket)
	if err != nil {
		t.Fatal(err)
	}
	img, err := dk.InspectImage(ctx, *image)
	if err != nil {
		t.Fatalf("image %s: %v (make criu-image)", *image, err)
	}
	tag := make([]byte, 4)
	_, _ = rand.Read(tag)
	h := &harness{t: t, ctx: ctx, dk: dk, image: img.ID, base: t.TempDir(), tag: hex.EncodeToString(tag)}
	h.work = h.dir("work")
	for _, name := range []string{"warmctl", "witness"} {
		raw, err := os.ReadFile(filepath.Join(*binDir, name))
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(h.work, name), raw, 0o755); err != nil {
			t.Fatal(err)
		}
	}
	return h
}

func (h *harness) dir(name string) string {
	d := filepath.Join(h.base, name)
	if err := os.MkdirAll(d, 0o755); err != nil {
		h.t.Fatal(err)
	}
	return d
}

func (h *harness) start(role string, cfg agent.Config, mounts []docker.Mount) string {
	h.t.Helper()
	name := role + ".json"
	if err := cfg.Write(filepath.Join(h.work, name)); err != nil {
		h.t.Fatal(err)
	}
	spec := docker.Spec{
		Name: "warmctl-it-" + role + "-" + h.tag, Image: h.image,
		Entrypoint: []string{"/work/warmctl", "agent", "--config", "/work/" + name},
		Mounts:     append([]docker.Mount{{Source: h.work, Target: "/work", ReadOnly: true}}, mounts...),
		Memory:     2 << 30, CPUs: 2, PIDs: 4096, SHM: 64 << 20, TmpSize: 256 << 20, Checkpoint: true,
	}
	id, err := h.dk.Create(h.ctx, spec)
	if err != nil {
		h.t.Fatal(err)
	}
	h.t.Cleanup(func() {
		if h.t.Failed() {
			h.dumpLogs(role, id)
		}
		_ = h.dk.Remove(context.Background(), id)
	})
	if err := h.dk.Start(h.ctx, id); err != nil {
		h.t.Fatal(err)
	}
	return id
}

func (h *harness) dumpLogs(role, id string) {
	var b strings.Builder
	_ = h.dk.Logs(context.Background(), id, 200, &b)
	h.t.Logf("--- %s container log\n%s", role, b.String())
	for _, log := range []string{"donor-io/out/criu-work/dump.log", "recv-io/work/restore.log", "donor-ws/logs/engine.log", "recv-ws/logs/engine.log"} {
		raw, err := os.ReadFile(filepath.Join(h.base, log))
		if err != nil {
			continue
		}
		if len(raw) > 6000 {
			raw = raw[len(raw)-6000:]
		}
		h.t.Logf("--- %s\n%s", log, raw)
	}
}

func (h *harness) dial(name string) *agent.Client {
	h.t.Helper()
	deadline := time.Now().Add(30 * time.Second)
	for {
		conn, err := net.Dial("unix", name)
		if err == nil {
			return agent.NewClient(conn)
		}
		if time.Now().After(deadline) {
			h.t.Fatalf("agent socket %s: %v", name, err)
		}
		time.Sleep(50 * time.Millisecond)
	}
}

func (h *harness) call(c *agent.Client, step string, out any) {
	h.t.Helper()
	if err := c.Call(h.ctx, step, nil, out); err != nil {
		h.t.Fatalf("step %s: %v", step, err)
	}
}

func (h *harness) request(dir string, seq int) {
	h.t.Helper()
	tmp := filepath.Join(dir, "request.tmp")
	if err := os.WriteFile(tmp, []byte(fmt.Sprint(seq)), 0o644); err != nil {
		h.t.Fatal(err)
	}
	if err := os.Rename(tmp, filepath.Join(dir, "request")); err != nil {
		h.t.Fatal(err)
	}
}

func (h *harness) await(dir, role string, seq int) observation {
	h.t.Helper()
	deadline := time.Now().Add(30 * time.Second)
	for {
		var o observation
		raw, err := os.ReadFile(filepath.Join(dir, role+".json"))
		if err == nil && json.Unmarshal(raw, &o) == nil && o.Sequence == seq {
			return o
		}
		if err != nil && !errors.Is(err, os.ErrNotExist) {
			h.t.Fatal(err)
		}
		if time.Now().After(deadline) {
			h.t.Fatalf("%s never answered request %d (last %s)", role, seq, raw)
		}
		time.Sleep(20 * time.Millisecond)
	}
}

func (h *harness) ask(name string) string {
	h.t.Helper()
	conn, err := net.DialTimeout("unix", name, 5*time.Second)
	if err != nil {
		h.t.Errorf("restored socket: %v", err)
		return ""
	}
	defer conn.Close()
	line, err := bufio.NewReader(conn).ReadString('\n')
	if err != nil && !errors.Is(err, io.EOF) {
		h.t.Errorf("restored socket: %v", err)
	}
	return strings.TrimSpace(line)
}

func (h *harness) exit(id string) int {
	h.t.Helper()
	ctx, cancel := context.WithTimeout(h.ctx, 60*time.Second)
	defer cancel()
	code, err := h.dk.Wait(ctx, id)
	if err != nil {
		h.t.Fatal(err)
	}
	return code
}

func (h *harness) stop(id string) {
	_ = h.dk.Stop(context.Background(), id, 10*time.Second)
}
