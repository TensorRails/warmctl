package criu

import (
	"path/filepath"
	"strings"
	"testing"
)

func TestArgs(t *testing.T) {
	dump := strings.Join(DumpArgs(32768, "/state/images", "/state/work"), " ")
	want := "dump -v4 --tree 32768 --images-dir /state/images --work-dir /state/work --log-file /state/work/dump.log --shell-job --file-locks --ext-unix-sk --tcp-established --link-remap --ghost-limit 67108864 -L /state/work/plugins"
	if dump != want {
		t.Fatalf("dump argv\n got %s\nwant %s", dump, want)
	}
	restore := strings.Join(RestoreArgs("/capture/images", "/state/work"), " ")
	want = "restore -v4 --images-dir /capture/images --restore-detached --leave-stopped --work-dir /state/work --pidfile /state/work/root.pid --log-file /state/work/restore.log --shell-job --file-locks --ext-unix-sk --tcp-established --link-remap -L /state/work/plugins"
	if restore != want {
		t.Fatalf("restore argv\n got %s\nwant %s", restore, want)
	}
}

func TestTree(t *testing.T) {
	tree := Tree{Root: 40000, Processes: []Process{{PID: 40001, Threads: []int{40001, 40009}}, {PID: 40000, Threads: []int{40000}}}}.sorted()
	if tree.Processes[0].PID != 40000 || tree.MaxID() != 40009 || len(tree.PIDs()) != 2 {
		t.Fatalf("%+v", tree)
	}
}

func TestPrepareWork(t *testing.T) {
	work := filepath.Join(t.TempDir(), "work")
	if err := PrepareWork(work); err != nil {
		t.Fatal(err)
	}
	if err := PrepareWork(work); err == nil {
		t.Fatal("reused a work directory")
	}
	if err := PrepareWork("relative"); err == nil {
		t.Fatal("accepted a relative work directory")
	}
}
