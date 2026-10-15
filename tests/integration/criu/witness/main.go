// Command witness stands in for an inference engine in the CPU roundtrip.
// Its state lives only in memory: a random nonce and a counter in each of
// two processes in separate sessions, a deleted file held open, a shared
// mapping of a /dev/shm file and a listening unix socket. It answers a
// request number written to <dir>/request by writing <dir>/<role>.json.
package main

import (
	"crypto/rand"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"time"
)

type observation struct {
	Role     string `json:"role"`
	Nonce    string `json:"nonce"`
	Counter  int    `json:"counter"`
	Sequence int    `json:"sequence"`
	PID      int    `json:"pid"`
	Deleted  string `json:"deleted_sha256,omitempty"`
	Shared   uint64 `json:"shared,omitempty"`
}

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintln(os.Stderr, "usage: witness DIR [worker]")
		os.Exit(2)
	}
	dir, role := os.Args[1], "main"
	if len(os.Args) > 2 {
		role = os.Args[2]
	}
	// run answers requests until the test kills it; it returns only on failure.
	fmt.Fprintln(os.Stderr, "witness:", run(dir, role))
	os.Exit(1)
}

func run(dir, role string) error {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	nonce := make([]byte, 16)
	if _, err := rand.Read(nonce); err != nil {
		return err
	}
	obs := observation{Role: role, Nonce: hex.EncodeToString(nonce), PID: os.Getpid()}
	var res resources
	if role == "main" {
		var err error
		if res, err = startMain(dir, obs.Nonce); err != nil {
			return err
		}
	}
	fmt.Printf("witness %s pid %d nonce %s\n", role, obs.PID, obs.Nonce)
	report := filepath.Join(dir, role+".json")
	if err := write(report, obs); err != nil {
		return err
	}
	for {
		if seq := readInt(filepath.Join(dir, "request")); seq > obs.Sequence {
			obs.Sequence = seq
			obs.Counter++
			res.observe(&obs)
			if err := write(report, obs); err != nil {
				return err
			}
			fmt.Printf("witness %s answered %d\n", role, seq)
		}
		time.Sleep(10 * time.Millisecond)
	}
}

// resources are what only the main process holds.
type resources struct {
	deleted *os.File
	shared  []byte
}

func (r resources) observe(obs *observation) {
	if r.deleted != nil {
		obs.Deleted = digest(r.deleted)
	}
	if r.shared != nil {
		binary.LittleEndian.PutUint64(r.shared, binary.LittleEndian.Uint64(r.shared)+1)
		obs.Shared = binary.LittleEndian.Uint64(r.shared)
	}
}

// startMain starts the worker in its own session and opens the main
// process's resources.
func startMain(dir, nonce string) (resources, error) {
	worker := exec.Command(os.Args[0], dir, "worker")
	worker.Stdout, worker.Stderr = os.Stdout, os.Stderr
	worker.SysProcAttr = &syscall.SysProcAttr{Setsid: true}
	if err := worker.Start(); err != nil {
		return resources{}, err
	}
	go func() { _ = worker.Wait(); os.Exit(3) }()
	var res resources
	var err error
	if res.deleted, err = deletedFile("/tmp/witness-deleted"); err != nil {
		return res, err
	}
	if res.shared, err = sharedMapping("/dev/shm/witness-shared"); err != nil {
		return res, err
	}
	return res, listen(filepath.Join(dir, "witness.sock"), nonce)
}

// deletedFile leaves 1 MiB of random bytes reachable only through an open
// descriptor: CRIU must carry it as a link remap.
func deletedFile(name string) (*os.File, error) {
	f, err := os.OpenFile(name, os.O_CREATE|os.O_RDWR|os.O_TRUNC, 0o600)
	if err != nil {
		return nil, err
	}
	body := make([]byte, 1<<20)
	if _, err := rand.Read(body); err != nil {
		return nil, err
	}
	if _, err := f.Write(body); err != nil {
		return nil, err
	}
	return f, os.Remove(name)
}

func digest(f *os.File) string {
	body := make([]byte, 1<<20)
	n, _ := f.ReadAt(body, 0)
	sum := sha256.Sum256(body[:n])
	return hex.EncodeToString(sum[:])
}

// sharedMapping maps a /dev/shm file shared, the way NCCL and vLLM's
// message queues do; the descriptor is closed, only the mapping remains.
func sharedMapping(name string) ([]byte, error) {
	f, err := os.OpenFile(name, os.O_CREATE|os.O_RDWR, 0o600)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	if err := f.Truncate(4096); err != nil {
		return nil, err
	}
	return syscall.Mmap(int(f.Fd()), 0, 4096, syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_SHARED)
}

func listen(name, nonce string) error {
	l, err := net.Listen("unix", name)
	if err != nil {
		return err
	}
	go func() {
		for {
			c, err := l.Accept()
			if err != nil {
				return
			}
			_, _ = c.Write([]byte(nonce + "\n"))
			c.Close()
		}
	}()
	return nil
}

func readInt(name string) int {
	raw, err := os.ReadFile(name)
	if err != nil {
		return 0
	}
	n, _ := strconv.Atoi(strings.TrimSpace(string(raw)))
	return n
}

func write(name string, obs observation) error {
	raw, err := json.Marshal(obs)
	if err != nil {
		return err
	}
	tmp := name + ".tmp"
	if err := os.WriteFile(tmp, raw, 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, name)
}
