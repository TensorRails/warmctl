package docker

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"io"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
)

type fakeDaemon struct {
	mu      sync.Mutex
	created []createBody
	pulled  bool
	paths   []string
}

func (f *fakeDaemon) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.paths = append(f.paths, r.Method+" "+r.URL.Path)
	p := strings.TrimPrefix(r.URL.Path, "/v"+apiVersion)
	switch {
	case p == "/version":
		json.NewEncoder(w).Encode(Version{Version: "29.1", APIVersion: "1.51", MinAPIVersion: "1.24", Os: "linux", Arch: "amd64"})
	case strings.HasPrefix(p, "/images/") && strings.HasSuffix(p, "/json"):
		if !f.pulled {
			w.WriteHeader(404)
			w.Write([]byte(`{"message":"No such image"}`))
			return
		}
		json.NewEncoder(w).Encode(Image{ID: "sha256:img", RepoDigests: []string{"vllm/vllm-openai@sha256:abc"}, Os: "linux", Architecture: "amd64"})
	case p == "/images/create":
		if r.URL.Query().Get("fromImage") != "vllm/vllm-openai" || r.URL.Query().Get("tag") != "sha256:abc" {
			w.WriteHeader(400)
			return
		}
		f.pulled = true
		w.Write([]byte(`{"status":"Pulling"}` + "\n" + `{"status":"Done"}` + "\n"))
	case p == "/containers/create":
		var body createBody
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			w.WriteHeader(400)
			return
		}
		f.created = append(f.created, body)
		w.WriteHeader(201)
		w.Write([]byte(`{"Id":"c1"}`))
	case p == "/containers/c1/start":
		w.WriteHeader(204)
	case p == "/containers/c1/wait":
		w.Write([]byte(`{"StatusCode":3}`))
	case p == "/containers/c1/stop":
		w.WriteHeader(304)
	case strings.HasPrefix(p, "/containers/gone"):
		w.WriteHeader(404)
		w.Write([]byte(`{"message":"No such container"}`))
	case p == "/containers/c1/logs":
		for _, frame := range []struct {
			stream byte
			text   string
		}{{1, "out\n"}, {2, "err\n"}} {
			var h [8]byte
			h[0] = frame.stream
			binary.BigEndian.PutUint32(h[4:], uint32(len(frame.text)))
			w.Write(h[:])
			w.Write([]byte(frame.text))
		}
	case p == "/containers/json":
		w.Write([]byte(`[{"Id":"c1","Names":["/warmctl-x"],"State":"running","Labels":{"io.warmctl.managed":"true"}}]`))
	default:
		w.WriteHeader(500)
	}
}

func serve(t *testing.T) (*Client, *fakeDaemon) {
	t.Helper()
	dir, err := os.MkdirTemp("/tmp", "dk")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { os.RemoveAll(dir) })
	l, err := net.Listen("unix", filepath.Join(dir, "d.sock"))
	if err != nil {
		t.Fatal(err)
	}
	f := &fakeDaemon{}
	srv := &http.Server{Handler: f, ReadHeaderTimeout: time.Second}
	go srv.Serve(l)
	t.Cleanup(func() { srv.Close() })
	c, err := New(context.Background(), filepath.Join(dir, "d.sock"))
	if err != nil {
		t.Fatal(err)
	}
	return c, f
}

func TestLifecycle(t *testing.T) {
	ctx := context.Background()
	c, f := serve(t)
	if _, err := c.EnsureImage(ctx, "vllm/vllm-openai:tag", nil); err == nil {
		t.Fatal("unpinned image accepted")
	}
	img, err := c.EnsureImage(ctx, "vllm/vllm-openai:cu134@sha256:abc", io.Discard)
	if err != nil || img.ID != "sha256:img" || !f.pulled {
		t.Fatalf("ensure: %+v %v", img, err)
	}
	spec := Spec{Name: "warmctl-x", Image: img.ID, Entrypoint: []string{"/work/warmctl", "agent"}, Memory: 1 << 30, CPUs: 2, PIDs: 64, SHM: 1 << 20}
	id, err := c.Create(ctx, spec)
	if err != nil || id != "c1" {
		t.Fatalf("create: %s %v", id, err)
	}
	if err := c.Start(ctx, id); err != nil {
		t.Fatal(err)
	}
	if code, err := c.Wait(ctx, id); err != nil || code != 3 {
		t.Fatalf("wait: %d %v", code, err)
	}
	if err := c.Stop(ctx, id, time.Second); err != nil {
		t.Fatal(err)
	}
	if err := c.Remove(ctx, "gone"); err != nil {
		t.Fatal(err)
	}
	if _, err := c.Inspect(ctx, "gone"); !IsNotFound(err) {
		t.Fatalf("inspect missing: %v", err)
	}
	var logs bytes.Buffer
	if err := c.Logs(ctx, id, 10, &logs); err != nil || logs.String() != "out\nerr\n" {
		t.Fatalf("logs %q %v", logs.String(), err)
	}
	list, err := c.List(ctx, ManagedLabel)
	if err != nil || len(list) != 1 || list[0].ID != "c1" {
		t.Fatalf("list %+v %v", list, err)
	}
	if len(f.created) != 1 || f.created[0].Labels[ManagedLabel] != "true" {
		t.Fatalf("created %+v", f.created)
	}
}

func TestRenderCheckpoint(t *testing.T) {
	spec := Spec{
		Name: "warmctl-donor", Image: "sha256:img", Entrypoint: []string{"/work/warmctl", "agent", "snapshot"},
		Env:    map[string]string{"B": "2", "A": "1"},
		Mounts: []Mount{{Source: "/srv/model", Target: "/model", ReadOnly: true}, {Source: "/srv/io", Target: "/state"}},
		GPUs:   []string{"GPU-1", "GPU-2"}, Memory: 64 << 30, SHM: 16 << 30, CPUs: 16, PIDs: 32768, Checkpoint: true,
	}
	b, err := render(spec)
	if err != nil {
		t.Fatal(err)
	}
	raw, _ := json.Marshal(b)
	for _, want := range []string{
		`"Env":["A=1","B=2"]`, `"User":"0:0"`, `"NetworkMode":"none"`, `"Init":true`, `"ReadonlyRootfs":true`,
		`"ReadonlyPaths":[]`, `"seccomp=unconfined"`, `"CHECKPOINT_RESTORE"`, `"DAC_READ_SEARCH"`,
		`"DeviceIDs":["GPU-1","GPU-2"]`, `"NanoCpus":16000000000`, `"MemorySwap":68719476736`,
		`"/tmp":"rw,nosuid,nodev,exec,size=1073741824"`, `{"Type":"bind","Source":"/srv/model","Target":"/model","ReadOnly":true}`,
	} {
		if !strings.Contains(string(raw), want) {
			t.Errorf("rendered spec lacks %s\n%s", want, raw)
		}
	}
	plain := spec
	plain.Checkpoint, plain.GPUs = false, nil
	b, _ = render(plain)
	raw, _ = json.Marshal(b)
	if strings.Contains(string(raw), "CapAdd") || strings.Contains(string(raw), "ReadonlyPaths") || strings.Contains(string(raw), "DeviceRequests") {
		t.Errorf("plain spec carries checkpoint settings: %s", raw)
	}
	for name, bad := range map[string]func(*Spec){
		"name":   func(s *Spec) { s.Name = "../x" },
		"limits": func(s *Spec) { s.Memory = 0 },
		"env":    func(s *Spec) { s.Env = map[string]string{"A B": "1"} },
		"mount":  func(s *Spec) { s.Mounts = append(s.Mounts, Mount{Source: "/a", Target: "/model"}) },
		"entry":  func(s *Spec) { s.Entrypoint = []string{"warmctl"} },
	} {
		s := spec
		s.Mounts = append([]Mount(nil), spec.Mounts...)
		bad(&s)
		if _, err := render(s); err == nil {
			t.Errorf("%s accepted", name)
		}
	}
}

func TestCheckInit(t *testing.T) {
	for body, ok := range map[string]bool{
		`{"InitBinary":"docker-init","InitCommit":{"ID":"de40ad0"}}`: true,
		`{"InitBinary":"docker-init","InitCommit":{"ID":""}}`:        true,
		`{"InitBinary":"","InitCommit":{"ID":""}}`:                   false,
	} {
		dir, _ := os.MkdirTemp("/tmp", "di")
		l, err := net.Listen("unix", filepath.Join(dir, "d.sock"))
		if err != nil {
			t.Fatal(err)
		}
		reply := body
		srv := &http.Server{ReadHeaderTimeout: time.Second, Handler: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if strings.HasSuffix(r.URL.Path, "/version") {
				w.Write([]byte(`{"Version":"29","APIVersion":"1.51","MinAPIVersion":"1.24","Os":"linux"}`))
				return
			}
			w.Write([]byte(reply))
		})}
		go srv.Serve(l)
		c, err := New(context.Background(), filepath.Join(dir, "d.sock"))
		if err != nil {
			t.Fatal(err)
		}
		if err := c.CheckInit(context.Background()); (err == nil) != ok {
			t.Errorf("%s: %v", body, err)
		}
		srv.Close()
		os.RemoveAll(dir)
	}
}
