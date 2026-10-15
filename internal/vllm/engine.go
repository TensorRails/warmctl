// Package vllm drives a vLLM API server through the agent's inference
// socket: lifecycle (health, sleep, wake), the worker RPC that reaches
// warmctl's extension, and token-ID completions for warmup and canaries.
package vllm

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"strconv"
	"time"
)

const maxBody = 64 << 20

// RankHeader routes a request to one data-parallel engine replica.
const RankHeader = "X-data-parallel-rank"

// Engine is one API server, reached over a unix socket.
type Engine struct {
	http *http.Client
	// Ranks is the number of workers (TP x DP) every RPC must answer for.
	Ranks int
}

// New connects through socket. Keep-alive is off: an idle connection into
// the engine would be an established TCP socket in the dumped tree.
func New(socket string, ranks int) *Engine {
	transport := &http.Transport{
		Proxy: nil,
		DialContext: func(ctx context.Context, _, _ string) (net.Conn, error) {
			return (&net.Dialer{}).DialContext(ctx, "unix", socket)
		},
		DisableKeepAlives: true,
	}
	return &Engine{http: &http.Client{Transport: transport}, Ranks: ranks}
}

// HTTPError is a non-200 answer.
type HTTPError struct {
	Path   string
	Status int
	Body   string
}

func (e *HTTPError) Error() string {
	return fmt.Sprintf("vllm %s: HTTP %d: %.2048s", e.Path, e.Status, e.Body)
}

type rankKey struct{}

// WithRank routes the context's requests to one data-parallel replica.
func WithRank(ctx context.Context, rank int) context.Context {
	return context.WithValue(ctx, rankKey{}, rank)
}

func (e *Engine) raw(ctx context.Context, method, path string, body any) ([]byte, error) {
	var reader io.Reader
	if body != nil {
		data, err := json.Marshal(body)
		if err != nil {
			return nil, err
		}
		reader = bytes.NewReader(data)
	}
	req, err := http.NewRequestWithContext(ctx, method, "http://engine"+path, reader)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	if rank, ok := ctx.Value(rankKey{}).(int); ok {
		req.Header.Set(RankHeader, strconv.Itoa(rank))
	}
	resp, err := e.http.Do(req)
	if err != nil {
		return nil, fmt.Errorf("vllm %s: %w", path, err)
	}
	defer resp.Body.Close()
	data, err := io.ReadAll(io.LimitReader(resp.Body, maxBody+1))
	if err != nil {
		return nil, err
	}
	if len(data) > maxBody {
		return nil, fmt.Errorf("vllm %s: response exceeds %d bytes", path, maxBody)
	}
	if resp.StatusCode != http.StatusOK {
		return nil, &HTTPError{Path: path, Status: resp.StatusCode, Body: string(data)}
	}
	return data, nil
}

func (e *Engine) call(ctx context.Context, method, path string, body, out any) error {
	data, err := e.raw(ctx, method, path, body)
	if err != nil || out == nil {
		return err
	}
	if err := json.Unmarshal(data, out); err != nil {
		return fmt.Errorf("vllm %s: %w", path, err)
	}
	return nil
}

// Health is GET /health.
func (e *Engine) Health(ctx context.Context) error {
	_, err := e.raw(ctx, http.MethodGet, "/health", nil)
	return err
}

// WaitReady polls /health until it answers 200. alive is checked between
// polls; its error (the engine died) ends the wait.
func (e *Engine) WaitReady(ctx context.Context, timeout time.Duration, alive func() error) error {
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	ticker := time.NewTicker(time.Second)
	defer ticker.Stop()
	for {
		probe, done := context.WithTimeout(ctx, 10*time.Second)
		err := e.Health(probe)
		done()
		if err == nil {
			return nil
		}
		if alive != nil {
			if err := alive(); err != nil {
				return err
			}
		}
		select {
		case <-ctx.Done():
			return fmt.Errorf("engine not ready after %s: %w", timeout, err)
		case <-ticker.C:
		}
	}
}

// Sleeping is GET /is_sleeping.
func (e *Engine) Sleeping(ctx context.Context) (bool, error) {
	var out struct {
		Sleeping *bool `json:"is_sleeping"`
	}
	if err := e.call(ctx, http.MethodGet, "/is_sleeping", nil, &out); err != nil {
		return false, err
	}
	if out.Sleeping == nil {
		return false, errors.New("vllm /is_sleeping: no is_sleeping field")
	}
	return *out.Sleeping, nil
}

// Sleep puts the engine to sleep at level and checks it sleeps.
func (e *Engine) Sleep(ctx context.Context, level int) error {
	if err := e.call(ctx, http.MethodPost, fmt.Sprintf("/sleep?level=%d&mode=wait", level), nil, nil); err != nil {
		return err
	}
	return e.expectSleeping(ctx, true)
}

func (e *Engine) expectSleeping(ctx context.Context, want bool) error {
	got, err := e.Sleeping(ctx)
	if err != nil {
		return err
	}
	if got != want {
		return fmt.Errorf("vllm: engine sleeping=%t, expected %t", got, want)
	}
	return nil
}

// WakeUp maps the given tags back (all when none are given).
func (e *Engine) WakeUp(ctx context.Context, tags ...string) error {
	path := "/wake_up"
	if len(tags) > 0 {
		q := url.Values{}
		for _, t := range tags {
			q.Add("tags", t)
		}
		path += "?" + q.Encode()
	}
	return e.call(ctx, http.MethodPost, path, nil, nil)
}

// ResetPrefixCache empties the prefix cache.
func (e *Engine) ResetPrefixCache(ctx context.Context) error {
	return e.call(ctx, http.MethodPost, "/reset_prefix_cache", nil, nil)
}

// RPC calls method on every worker through /collective_rpc and decodes the
// per-rank results into out (a pointer to a slice), which must hold Ranks rows.
func (e *Engine) RPC(ctx context.Context, method string, kwargs any, timeout time.Duration, out any) error {
	if kwargs == nil {
		kwargs = map[string]any{}
	}
	body := map[string]any{"method": method, "kwargs": kwargs, "timeout": timeout.Seconds()}
	var reply struct {
		Results []json.RawMessage `json:"results"`
	}
	ctx, cancel := context.WithTimeout(ctx, timeout+30*time.Second)
	defer cancel()
	data, err := e.raw(ctx, http.MethodPost, "/collective_rpc", body)
	if err != nil {
		return fmt.Errorf("rpc %s: %w", method, err)
	}
	if len(bytes.TrimSpace(data)) == 0 {
		return fmt.Errorf("rpc %s: the engine returned no results", method)
	}
	if err := json.Unmarshal(data, &reply); err != nil {
		return fmt.Errorf("rpc %s: %w", method, err)
	}
	if len(reply.Results) != e.Ranks {
		return fmt.Errorf("rpc %s: %d results for %d ranks", method, len(reply.Results), e.Ranks)
	}
	if out == nil {
		return nil
	}
	raw, err := json.Marshal(reply.Results)
	if err != nil {
		return err
	}
	if err := json.Unmarshal(raw, out); err != nil {
		return fmt.Errorf("rpc %s: %w", method, err)
	}
	return nil
}
