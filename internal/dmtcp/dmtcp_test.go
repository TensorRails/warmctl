package dmtcp

import (
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
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
	if real, err := m.Real([]int{44000, 40000}); err != nil || !reflect.DeepEqual(real, []int{59133, 59128}) {
		t.Fatalf("real %v %v", real, err)
	}
	if virtual, err := m.Virtual([]int{59128}); err != nil || virtual[0] != 40000 {
		t.Fatalf("virtual %v %v", virtual, err)
	}
	if _, err := m.Real([]int{45000}); err == nil {
		t.Fatal("a PID outside the computation translated")
	}
	if real, err := m.Real(nil); err != nil || len(real) != 0 {
		t.Fatalf("nil %v %v", real, err)
	}
}

func TestWrap(t *testing.T) {
	tools := Tools{Dir: "/opt/warmctl/dmtcp/bin", Plugin: "/opt/warmctl/dmtcp/libdmtcp_fdkeep.so"}
	got := strings.Join(tools.Wrap([]string{"vllm", "serve", "/models/m"}, "/state/out/criu"), " ")
	want := "/opt/warmctl/dmtcp/bin/dmtcp_launch --coord-port 7779 --no-gzip --ckptdir /state/out/criu " +
		"--with-plugin /opt/warmctl/dmtcp/libdmtcp_fdkeep.so vllm serve /models/m"
	if got != want {
		t.Fatalf("wrap\n got %s\nwant %s", got, want)
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
