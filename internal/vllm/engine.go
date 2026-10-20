// Package vllm drives a vLLM API server through the agent's inference
// socket: lifecycle (health, sleep, wake), the worker RPC that reaches
// warmctl's extension, and token-ID completions for warmup and canaries.
package vllm

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strconv"
	"strings"
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
	// replies, when set, carries every rank's RPC result in a file.
	replies *replies
}

// replies is a directory the engine shares with this process: host is
// where this process sees it, guest where the engine does.
type replies struct{ host, guest string }

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

// ReplyThrough sends every RPC through the adapter's warmctl_rpc, which
// publishes each rank's result under guest, seen here as host. A
// data-parallel engine needs it: /collective_rpc runs on every engine core
// but answers with the first engine's workers only (vLLM's
// DPLBAsyncMPClient.call_utility_async).
func (e *Engine) ReplyThrough(host, guest string) {
	e.replies = &replies{host: host, guest: guest}
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
	ctx, cancel := context.WithTimeout(ctx, timeout+30*time.Second)
	defer cancel()
	var results []json.RawMessage
	var err error
	if e.replies != nil {
		results, err = e.replyRPC(ctx, method, kwargs, timeout)
	} else {
		results, err = e.collective(ctx, method, kwargs, timeout)
	}
	if err != nil {
		return fmt.Errorf("rpc %s: %w", method, err)
	}
	if len(results) != e.Ranks {
		return fmt.Errorf("rpc %s: %d results for %d ranks", method, len(results), e.Ranks)
	}
	if out == nil {
		return nil
	}
	raw, err := json.Marshal(results)
	if err != nil {
		return err
	}
	if err := json.Unmarshal(raw, out); err != nil {
		return fmt.Errorf("rpc %s: %w", method, err)
	}
	return nil
}

// collective is one /collective_rpc call and the results it answers with.
func (e *Engine) collective(ctx context.Context, method string, kwargs any,
	timeout time.Duration) ([]json.RawMessage, error) {
	body := map[string]any{"method": method, "kwargs": kwargs, "timeout": timeout.Seconds()}
	data, err := e.raw(ctx, http.MethodPost, "/collective_rpc", body)
	if err != nil {
		return nil, err
	}
	if len(bytes.TrimSpace(data)) == 0 {
		return nil, errors.New("the engine returned no results")
	}
	var reply struct {
		Results []json.RawMessage `json:"results"`
	}
	if err := json.Unmarshal(data, &reply); err != nil {
		return nil, err
	}
	return reply.Results, nil
}

// replyRPC runs method through warmctl_rpc and reads every rank's result
// from its file. The engine answers once every engine core has run the
// call, so a missing file is a rank the call never reached.
func (e *Engine) replyRPC(ctx context.Context, method string, kwargs any,
	timeout time.Duration) ([]json.RawMessage, error) {
	nonce := make([]byte, 8)
	if _, err := rand.Read(nonce); err != nil {
		return nil, err
	}
	name := method + "-" + hex.EncodeToString(nonce)
	dir := filepath.Join(e.replies.host, name)
	defer os.RemoveAll(dir)
	wrapped := map[string]any{"method": method, "kwargs": kwargs, "reply": e.replies.guest + "/" + name}
	if _, err := e.collective(ctx, "warmctl_rpc", wrapped, timeout); err != nil {
		return nil, err
	}
	return readReplies(dir, e.Ranks)
}

// readReplies decodes <dir>/<rank>.json of ranks 0..ranks-1.
func readReplies(dir string, ranks int) ([]json.RawMessage, error) {
	entries, err := os.ReadDir(dir)
	if err != nil && !errors.Is(err, os.ErrNotExist) {
		return nil, err
	}
	results := make([]json.RawMessage, ranks)
	for _, entry := range entries {
		rank, err := strconv.Atoi(strings.TrimSuffix(entry.Name(), ".json"))
		if err != nil || rank < 0 || rank >= ranks || strconv.Itoa(rank)+".json" != entry.Name() {
			return nil, fmt.Errorf("reply %s is not one of ranks 0..%d", entry.Name(), ranks-1)
		}
		data, err := os.ReadFile(filepath.Join(dir, entry.Name()))
		if err != nil {
			return nil, err
		}
		var reply map[string]json.RawMessage
		if err := json.Unmarshal(data, &reply); err != nil {
			return nil, fmt.Errorf("rank %d reply: %w", rank, err)
		}
		result, ok := reply["result"]
		if !ok {
			return nil, fmt.Errorf("rank %d reply has no result", rank)
		}
		results[rank] = result
	}
	var missing []string
	for rank, result := range results {
		if result == nil {
			missing = append(missing, strconv.Itoa(rank))
		}
	}
	if len(missing) > 0 {
		return nil, fmt.Errorf("%d of %d ranks did not answer: %s", len(missing), ranks, strings.Join(missing, ","))
	}
	return results, nil
}
