package vllm

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/TensorRails/warmctl/v2/internal/profile"
)

type fake struct {
	mu       sync.Mutex
	sleeping bool
	cached   map[string]bool
	calls    []string
	ranks    []string
}

func (f *fake) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.calls = append(f.calls, r.Method+" "+r.URL.Path+"?"+r.URL.RawQuery)
	f.ranks = append(f.ranks, r.Header.Get(RankHeader))
	switch r.URL.Path {
	case "/health":
	case "/sleep":
		f.sleeping = true
	case "/wake_up":
		f.sleeping = false
	case "/is_sleeping":
		fmt.Fprintf(w, `{"is_sleeping":%t}`, f.sleeping)
	case "/reset_prefix_cache":
		f.cached = map[string]bool{}
	case "/collective_rpc":
		var body struct {
			Method string         `json:"method"`
			Kwargs map[string]any `json:"kwargs"`
		}
		json.NewDecoder(r.Body).Decode(&body)
		fmt.Fprintf(w, `{"results":[{"rank":0,"method":%q},{"rank":1,"method":%q}]}`, body.Method, body.Method)
	case "/v1/completions":
		var body struct {
			Prompt    json.RawMessage `json:"prompt"`
			MaxTokens int             `json:"max_tokens"`
		}
		json.NewDecoder(r.Body).Decode(&body)
		var batch [][]int
		if json.Unmarshal(body.Prompt, &batch) != nil {
			var one []int
			json.Unmarshal(body.Prompt, &one)
			batch = [][]int{one}
		}
		cached, prompt := 0, 0
		var choices []string
		for i, p := range batch {
			key := r.Header.Get(RankHeader) + fmt.Sprint(p)
			if f.cached[key] {
				cached += len(p)
			}
			f.cached[key] = true
			prompt += len(p)
			tokens := make([]string, body.MaxTokens)
			logprobs := make([]string, body.MaxTokens)
			for j := range tokens {
				tokens[j] = fmt.Sprintf(`"token_id:%d"`, 1000+j)
				logprobs[j] = fmt.Sprintf("%g", -0.5-float64(j)/10)
			}
			choices = append(choices, fmt.Sprintf(`{"index":%d,"logprobs":{"tokens":[%s],"token_logprobs":[%s]}}`, i, strings.Join(tokens, ","), strings.Join(logprobs, ",")))
		}
		fmt.Fprintf(w, `{"choices":[%s],"usage":{"prompt_tokens":%d,"prompt_tokens_details":{"cached_tokens":%d}}}`, strings.Join(choices, ","), prompt, cached)
	default:
		w.WriteHeader(404)
	}
}

func start(t *testing.T) (*Engine, *fake) {
	dir, err := os.MkdirTemp("/tmp", "vl")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { os.RemoveAll(dir) })
	sock := filepath.Join(dir, "inference.sock")
	l, err := net.Listen("unix", sock)
	if err != nil {
		t.Fatal(err)
	}
	f := &fake{cached: map[string]bool{}}
	srv := &http.Server{Handler: f, ReadHeaderTimeout: time.Second}
	go srv.Serve(l)
	t.Cleanup(func() { srv.Close() })
	return New(sock, 2), f
}

func TestLifecycleAndRPC(t *testing.T) {
	ctx := context.Background()
	e, f := start(t)
	if err := e.WaitReady(ctx, 5*time.Second, nil); err != nil {
		t.Fatal(err)
	}
	if err := e.Sleep(ctx, 2); err != nil {
		t.Fatal(err)
	}
	if err := e.WakeUp(ctx, "weights"); err != nil {
		t.Fatal(err)
	}
	if asleep, _ := e.Sleeping(ctx); asleep {
		t.Fatal("still sleeping")
	}
	var rows []struct {
		Rank   int    `json:"rank"`
		Method string `json:"method"`
	}
	if err := e.RPC(ctx, "warmctl_describe", nil, time.Minute, &rows); err != nil || len(rows) != 2 || rows[1].Method != "warmctl_describe" {
		t.Fatalf("%+v %v", rows, err)
	}
	e.Ranks = 3
	if err := e.RPC(ctx, "warmctl_describe", nil, time.Minute, nil); err == nil {
		t.Fatal("short RPC answer accepted")
	}
	if !strings.Contains(strings.Join(f.calls, " "), "POST /sleep?level=2&mode=wait") || !strings.Contains(strings.Join(f.calls, " "), "POST /wake_up?tags=weights") {
		t.Fatalf("calls %v", f.calls)
	}
}

func TestWarmupAndCanary(t *testing.T) {
	ctx := context.Background()
	e, f := start(t)
	cases := []profile.WarmupCase{{PromptTokens: 32, Batch: 4, GeneratedTokens: 8}, {PromptTokens: 512, Batch: 1, GeneratedTokens: 8}}
	var seen []int
	if err := e.Warmup(ctx, "m", 2, cases, func(i int, _ profile.WarmupCase, _ float64) { seen = append(seen, i) }); err != nil {
		t.Fatal(err)
	}
	if len(seen) != 2 || !strings.Contains(strings.Join(f.ranks, ","), "1") {
		t.Fatalf("cases %v ranks %v", seen, f.ranks)
	}
	if err := e.ResetPrefixCache(ctx); err != nil {
		t.Fatal(err)
	}
	donor, err := e.RunCanary(ctx, "m", 64, 16)
	if err != nil || donor.MissCached != 0 || donor.HitCached != 64 || len(donor.Miss.Tokens) != 16 {
		t.Fatalf("%+v %v", donor, err)
	}
	if s, err := CompareCanary(profile.Canary{Mode: "bitwise"}, donor, donor); err != nil || s != "canary matches" {
		t.Fatal(s, err)
	}
	bad := donor
	bad.Miss = Row{Tokens: append([]int{7}, donor.Miss.Tokens[1:]...), Logprobs: donor.Miss.Logprobs}
	if _, err := CompareCanary(profile.Canary{Mode: "bitwise"}, donor, bad); err == nil {
		t.Fatal("token mismatch passed bitwise")
	}
	if s, err := CompareCanary(profile.Canary{Mode: "warn"}, donor, bad); err != nil || !strings.Contains(s, "token 0") {
		t.Fatal(s, err)
	}
	late := donor
	late.Miss = Row{Tokens: append(append([]int{}, donor.Miss.Tokens[:8]...), 5, 5, 5, 5, 5, 5, 5, 5), Logprobs: donor.Miss.Logprobs}
	if _, err := CompareCanary(profile.Canary{Mode: "tolerance"}, donor, late); err != nil {
		t.Fatalf("late divergence failed tolerance: %v", err)
	}
	hybrid, uncached := donor, donor
	hybrid.HitCached, uncached.HitCached = 0, 0
	if _, err := CompareCanary(profile.Canary{Mode: "bitwise"}, hybrid, uncached); err != nil {
		t.Fatalf("a donor without prefix hits failed the restore: %v", err)
	}
	if _, err := CompareCanary(profile.Canary{Mode: "bitwise"}, donor, uncached); err == nil {
		t.Fatal("a lost prefix cache passed")
	}
	stale := donor
	stale.MissCached = 64
	if _, err := CompareCanary(profile.Canary{Mode: "tolerance"}, donor, stale); err == nil {
		t.Fatal("non-empty KV cache passed")
	}
	if p := Prompt(4, 2); p[0] != 102 || p[1] != 11 {
		t.Fatal(p)
	}
}

func serveHandler(t *testing.T, h http.HandlerFunc) *Engine {
	dir, err := os.MkdirTemp("/tmp", "vh")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { os.RemoveAll(dir) })
	l, err := net.Listen("unix", filepath.Join(dir, "s.sock"))
	if err != nil {
		t.Fatal(err)
	}
	srv := &http.Server{Handler: h, ReadHeaderTimeout: time.Second}
	go srv.Serve(l)
	t.Cleanup(func() { srv.Close() })
	return New(filepath.Join(dir, "s.sock"), 2)
}

// vLLM answers /collective_rpc with 200 and no body when the RPC returns
// nothing at all, and with nulls for methods that return None per rank.
func TestRPCBodies(t *testing.T) {
	ctx := context.Background()
	empty := serveHandler(t, func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(200) })
	if err := empty.RPC(ctx, "checkpoint_prepare", nil, time.Minute, nil); err == nil || !strings.Contains(err.Error(), "no results") {
		t.Fatalf("empty body: %v", err)
	}
	nulls := serveHandler(t, func(w http.ResponseWriter, _ *http.Request) { w.Write([]byte(`{"results":[null,null]}`)) })
	if err := nulls.RPC(ctx, "checkpoint_prepare", nil, time.Minute, nil); err != nil {
		t.Fatalf("null results: %v", err)
	}
}

// P199: an HTTP error must carry the engine's reason, not just the status.
func TestErrorCarriesBody(t *testing.T) {
	e := serveHandler(t, func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(400)
		w.Write([]byte(`{"error":"Out of range float values are not JSON compliant: nan"}`))
	})
	_, err := e.Complete(context.Background(), "m", 8, 1, 4)
	var he *HTTPError
	if !errors.As(err, &he) || he.Status != 400 || !strings.Contains(err.Error(), "not JSON compliant: nan") {
		t.Fatalf("error lost the body: %v", err)
	}
}
