package dmtcp

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"
)

// The coordinator's list of a restarted `vllm serve` as DMTCP 5.0 prints it:
// API server and engine core, real PIDs of the new pod, the engine core
// still at a restart barrier.
const list = `Client List:
#, PROG[virtPID:realPID]@HOST, DMTCP-UNIQUEPID, STATE, BARRIER
2, VLLM::EngineCore[44000:59133]@b5266da1239b, 314ceaedf4104db7-44000-3942572dc4057, WorkerState::RESTARTING, Socket::Restart_Ns_Register_Data
1, python3.12[40000:59128]@b5266da1239b, 314ceaedf4104db7-40000-3941ee323f43e, WorkerState::RUNNING,
`

func TestParseList(t *testing.T) {
	peers, err := ParseList(list)
	if err != nil {
		t.Fatal(err)
	}
	want := []Peer{
		{Name: "python3.12", Virtual: 40000, Real: 59128, State: "RUNNING"},
		{Name: "VLLM::EngineCore", Virtual: 44000, Real: 59133, State: "RESTARTING"},
	}
	if !reflect.DeepEqual(peers, want) {
		t.Fatalf("got %+v", peers)
	}
	if _, err := ParseList("Client List:\n#, PROG[virtPID:realPID]@HOST, DMTCP-UNIQUEPID, STATE, BARRIER\n"); err == nil {
		t.Fatal("an empty list parsed")
	}
	m := NewMap(peers)
	if got, err := m.Real([]int{44000, 40000}); err != nil || !reflect.DeepEqual(got, []int{59133, 59128}) {
		t.Fatalf("real %v %v", got, err)
	}
	if virtual, err := m.Virtual([]int{59128}); err != nil || virtual[0] != 40000 {
		t.Fatalf("virtual %v %v", virtual, err)
	}
	if _, err := m.Real([]int{45000}); err == nil {
		t.Fatal("a PID outside the computation translated")
	}
	if got, err := m.Real(nil); err != nil || len(got) != 0 {
		t.Fatalf("nil %v %v", got, err)
	}
}

func TestWrap(t *testing.T) {
	tools := Tools{Dir: "/opt/warmctl/dmtcp/bin", Plugin: "/opt/warmctl/dmtcp/libdmtcp_fdkeep.so", Port: 7779}
	got := strings.Join(tools.Wrap([]string{"vllm", "serve", "/models/m"}, "/state/out/criu"), " ")
	want := "/opt/warmctl/dmtcp/bin/dmtcp_launch --coord-port 7779 --no-gzip --ckptdir /state/out/criu " +
		"--with-plugin /opt/warmctl/dmtcp/libdmtcp_fdkeep.so vllm serve /models/m"
	if got != want {
		t.Fatalf("wrap\n got %s\nwant %s", got, want)
	}
	restart := strings.Join(tools.RestartArgs([]string{"/capture/criu/ckpt_a.dmtcp"}), " ")
	if restart != "--coord-port 7779 --no-strict-checking /capture/criu/ckpt_a.dmtcp" {
		t.Fatalf("restart args %s", restart)
	}
	tools.Dir, tools.Port = "", 7790
	if got := tools.Wrap([]string{"x"}, "/d")[0:3]; got[0] != "dmtcp_launch" || got[2] != "7790" {
		t.Fatalf("PATH and port: %v", got)
	}
}

func TestImages(t *testing.T) {
	dir := t.TempDir()
	if _, err := Images(dir); err == nil {
		t.Fatal("no images found in an empty directory")
	}
	for _, name := range []string{"ckpt_python3.12_x-44000-b.dmtcp", "ckpt_python3.12_x-40000-a.dmtcp", "dmtcp_restart_script.sh"} {
		if err := os.WriteFile(filepath.Join(dir, name), nil, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	if err := os.Mkdir(filepath.Join(dir, "ckpt_python3.12_x-44000-b_files"), 0o755); err != nil {
		t.Fatal(err)
	}
	images, err := Images(dir)
	if err != nil || len(images) != 2 || filepath.Base(images[0]) != "ckpt_python3.12_x-40000-a.dmtcp" {
		t.Fatalf("%v %v", images, err)
	}
}

// A stand-in dmtcp_coordinator: it writes the port it "bound" to
// --port-file, as the real one does before it forks into the background.
const fakeCoordinator = `#!/bin/sh
while [ $# -gt 0 ]; do
	case "$1" in
	--port) [ "$2" = 0 ] || exit 1; shift ;;
	--port-file) printf 41234 > "$2"; shift ;;
	esac
	shift
done
`

func TestCoordinator(t *testing.T) {
	dir, tmp := t.TempDir(), t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "dmtcp_coordinator"), []byte(fakeCoordinator), 0o755); err != nil {
		t.Fatal(err)
	}
	tools := Tools{Dir: dir, TmpDir: tmp}
	port, err := tools.Coordinator(context.Background())
	if err != nil || port != 41234 {
		t.Fatalf("port %d %v", port, err)
	}
	if left, _ := os.ReadDir(tmp); len(left) != 0 {
		t.Fatalf("the port file stayed: %v", left)
	}
	// A coordinator that fails to bind the port it was given.
	tools.Port = 7779
	if _, err := tools.Coordinator(context.Background()); err == nil {
		t.Fatal("a failed coordinator returned a port")
	}
}

func TestWaitPeers(t *testing.T) {
	running := func(n int) []Peer {
		peers := make([]Peer, n)
		for i := range peers {
			peers[i] = Peer{Virtual: 40000 + 1000*i, Real: 100 + i, State: "RUNNING"}
		}
		return peers
	}
	restarting := running(2)
	restarting[1].State = "RESTARTING"
	// lists replays the coordinator's answers, then repeats the last one.
	lists := func(answers ...[]Peer) func(context.Context) ([]Peer, error) {
		return func(context.Context) ([]Peer, error) {
			peers := answers[0]
			if len(answers) > 1 {
				answers = answers[1:]
			}
			if peers == nil {
				return nil, errors.New("not yet")
			}
			return peers, nil
		}
	}
	ctx := context.Background()
	peers, err := waitPeers(ctx, lists(nil, running(1), restarting, running(2)), 2, time.Minute, nil)
	if err != nil || len(peers) != 2 {
		t.Fatalf("all running: %v %v", peers, err)
	}
	if _, err := waitPeers(ctx, lists(restarting, running(1)), 2, time.Minute, nil); err == nil ||
		!strings.Contains(err.Error(), "1 of 2 restarted processes left") {
		t.Fatalf("a peer left: %v", err)
	}
	died := errors.New("dmtcp_restart exited")
	if _, err := waitPeers(ctx, lists(restarting), 2, time.Minute, func() error { return died }); !errors.Is(err, died) {
		t.Fatalf("failed: %v", err)
	}
	if _, err := waitPeers(ctx, lists(restarting), 2, 100*time.Millisecond, nil); err == nil ||
		!strings.Contains(err.Error(), "1 of 2 peers running") {
		t.Fatalf("timeout: %v", err)
	}
}
