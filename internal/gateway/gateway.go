// Package gateway is the host's public endpoint of a restored engine: a
// TCP listener that forwards the inference routes, and only those, to the
// agent's inference socket. Until the restore has passed its checks every
// request gets 503, so a client never waits on a half-restored engine.
package gateway

import (
	"bytes"
	"context"
	"net"
	"net/http"
	"net/http/httputil"
	"net/url"
	"sync"
	"time"
)

// Gateway forwards inference requests once ready.
type Gateway struct {
	mu     sync.Mutex
	ready  bool
	active int
	idle   chan struct{}
	proxy  *httputil.ReverseProxy
	token  func()
	once   sync.Once
}

// New forwards to the engine behind socket; onToken runs once, when the
// first completion bytes go out.
func New(socket string, onToken func()) *Gateway {
	transport := &http.Transport{
		Proxy: nil,
		DialContext: func(ctx context.Context, _, _ string) (net.Conn, error) {
			return (&net.Dialer{}).DialContext(ctx, "unix", socket)
		},
		MaxIdleConns: 100, MaxIdleConnsPerHost: 100, IdleConnTimeout: 90 * time.Second,
	}
	proxy := httputil.NewSingleHostReverseProxy(&url.URL{Scheme: "http", Host: "engine"})
	proxy.Transport = transport
	proxy.FlushInterval = -1
	proxy.ErrorHandler = func(w http.ResponseWriter, _ *http.Request, _ error) {
		http.Error(w, "engine unavailable", http.StatusBadGateway)
	}
	idle := make(chan struct{})
	close(idle)
	return &Gateway{proxy: proxy, idle: idle, token: onToken}
}

// SetReady opens or closes admission.
func (g *Gateway) SetReady(ready bool) {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.ready = ready
}

func (g *Gateway) isReady() bool {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.ready
}

// Drain closes admission and waits for the admitted requests, streaming
// bodies included.
func (g *Gateway) Drain(ctx context.Context) error {
	g.mu.Lock()
	g.ready = false
	idle := g.idle
	g.mu.Unlock()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-idle:
		return nil
	}
}

func (g *Gateway) admit() bool {
	g.mu.Lock()
	defer g.mu.Unlock()
	if !g.ready {
		return false
	}
	if g.active == 0 {
		g.idle = make(chan struct{})
	}
	g.active++
	return true
}

func (g *Gateway) done() {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.active--
	if g.active == 0 {
		close(g.idle)
	}
}

func allowed(r *http.Request) bool {
	switch r.Method + " " + r.URL.Path {
	case "GET /v1/models", "GET /metrics", "POST /v1/completions", "POST /v1/chat/completions":
		return true
	}
	return false
}

// ServeHTTP admits the inference routes, answers /health from readiness and
// refuses the management routes (sleep, wake, collective RPC).
func (g *Gateway) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if r.Method == http.MethodGet && r.URL.Path == "/health" {
		if !g.isReady() {
			http.Error(w, "not ready", http.StatusServiceUnavailable)
			return
		}
		w.WriteHeader(http.StatusOK)
		return
	}
	if !allowed(r) {
		http.NotFound(w, r)
		return
	}
	if !g.admit() {
		http.Error(w, "not ready", http.StatusServiceUnavailable)
		return
	}
	defer g.done()
	r.Body = http.MaxBytesReader(w, r.Body, 16<<20)
	if r.Method == http.MethodPost && g.token != nil {
		w = &tokenWriter{ResponseWriter: w, g: g}
	}
	g.proxy.ServeHTTP(w, r)
}

// tokenWriter notices the first response body that carries a choice.
type tokenWriter struct {
	http.ResponseWriter
	g      *Gateway
	status int
}

func (w *tokenWriter) WriteHeader(code int) {
	w.status = code
	w.ResponseWriter.WriteHeader(code)
}

func (w *tokenWriter) Write(p []byte) (int, error) {
	if w.status == 0 {
		w.status = http.StatusOK
	}
	n, err := w.ResponseWriter.Write(p)
	if n > 0 && w.status == http.StatusOK && (bytes.Contains(p, []byte("choices")) || bytes.Contains(p, []byte("data:"))) {
		w.g.once.Do(w.g.token)
	}
	return n, err
}

func (w *tokenWriter) Flush() {
	if f, ok := w.ResponseWriter.(http.Flusher); ok {
		f.Flush()
	}
}

// Serve runs the gateway on l until ctx ends, then drains for up to grace.
func Serve(ctx context.Context, l net.Listener, g *Gateway, grace time.Duration) error {
	server := &http.Server{Handler: g, ReadHeaderTimeout: 10 * time.Second, IdleTimeout: 2 * time.Minute}
	done := make(chan error, 1)
	go func() { done <- server.Serve(l) }()
	select {
	case err := <-done:
		return err
	case <-ctx.Done():
	}
	shutdown, cancel := context.WithTimeout(context.Background(), grace)
	defer cancel()
	_ = g.Drain(shutdown)
	return server.Shutdown(shutdown)
}
