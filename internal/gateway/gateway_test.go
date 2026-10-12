package gateway

import (
	"context"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

func TestGateway(t *testing.T) {
	dir, _ := os.MkdirTemp("/tmp", "gw")
	defer os.RemoveAll(dir)
	sock := filepath.Join(dir, "inference.sock")
	l, err := net.Listen("unix", sock)
	if err != nil {
		t.Fatal(err)
	}
	upstream := &http.Server{Handler: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`{"choices":[{"text":"hi"}],"path":"` + r.URL.Path + `"}`))
	}), ReadHeaderTimeout: time.Second}
	go upstream.Serve(l)
	defer upstream.Close()
	var tokens atomic.Int32
	g := New(sock, func() { tokens.Add(1) })
	front := httptest.NewServer(g)
	defer front.Close()
	get := func(method, path string) (int, string) {
		req, _ := http.NewRequest(method, front.URL+path, strings.NewReader("{}"))
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()
		body, _ := io.ReadAll(resp.Body)
		return resp.StatusCode, string(body)
	}
	if code, _ := get("POST", "/v1/completions"); code != 503 {
		t.Fatalf("not-ready gateway answered %d", code)
	}
	if code, _ := get("GET", "/health"); code != 503 {
		t.Fatalf("not-ready health %d", code)
	}
	g.SetReady(true)
	for _, path := range []string{"/v1/completions", "/v1/chat/completions"} {
		if code, body := get("POST", path); code != 200 || !strings.Contains(body, path) {
			t.Fatalf("%s: %d %s", path, code, body)
		}
	}
	if tokens.Load() != 1 {
		t.Fatalf("first token marked %d times", tokens.Load())
	}
	for _, refused := range []string{"/sleep", "/collective_rpc", "/wake_up"} {
		if code, _ := get("POST", refused); code != 404 {
			t.Fatalf("%s: %d", refused, code)
		}
	}
	if code, _ := get("GET", "/health"); code != 200 {
		t.Fatal(code)
	}
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	if err := g.Drain(ctx); err != nil {
		t.Fatal(err)
	}
	if code, _ := get("POST", "/v1/completions"); code != 503 {
		t.Fatal("drained gateway admitted")
	}
}
