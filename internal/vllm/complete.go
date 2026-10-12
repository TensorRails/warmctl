package vllm

import (
	"context"
	"errors"
	"fmt"
	"math"
	"net/http"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/TensorRails/warmctl/v2/internal/manifest"
	"github.com/TensorRails/warmctl/v2/internal/profile"
)

// Seed of every warmctl request: greedy, seeded, so two engines with the
// same weights produce the same tokens.
const Seed = 42

var pattern = []int{9707, 11, 358, 646, 1492, 264, 1614, 13}

// Prompt is a token-ID prompt of length n; variant makes the prompts of
// one batch distinct, so none of them hits another's prefix.
func Prompt(n, variant int) []int {
	out := make([]int, n)
	for i := range out {
		out[i] = pattern[i%len(pattern)]
	}
	if n > 0 && variant > 0 {
		out[0] = 100 + variant
	}
	return out
}

// Row is one choice: generated token IDs and their logprobs.
type Row = manifest.Row

// Completion is one /v1/completions answer.
type Completion struct {
	Rows         []Row   `json:"rows"`
	PromptTokens int     `json:"prompt_tokens"`
	CachedTokens int     `json:"cached_tokens"`
	Seconds      float64 `json:"seconds"`
}

// Complete runs batch prompts of promptTokens, generating maxTokens each.
func (e *Engine) Complete(ctx context.Context, model string, promptTokens, batch, maxTokens int) (Completion, error) {
	prompts := make([][]int, batch)
	for i := range prompts {
		prompts[i] = Prompt(promptTokens, i)
	}
	var input any = prompts
	if batch == 1 {
		input = prompts[0]
	}
	body := map[string]any{
		"model": model, "prompt": input, "max_tokens": maxTokens, "temperature": 0, "seed": Seed,
		"logprobs": 1, "ignore_eos": true, "return_tokens_as_token_ids": true,
	}
	started := time.Now()
	var resp struct {
		Choices []struct {
			Index    int `json:"index"`
			Logprobs struct {
				Tokens   []string  `json:"tokens"`
				Logprobs []float64 `json:"token_logprobs"`
			} `json:"logprobs"`
		} `json:"choices"`
		Usage struct {
			PromptTokens int `json:"prompt_tokens"`
			Details      *struct {
				CachedTokens *int `json:"cached_tokens"`
			} `json:"prompt_tokens_details"`
		} `json:"usage"`
	}
	if err := e.call(ctx, http.MethodPost, "/v1/completions", body, &resp); err != nil {
		return Completion{}, err
	}
	out := Completion{PromptTokens: resp.Usage.PromptTokens, Seconds: time.Since(started).Seconds(),
		Rows: make([]Row, batch)}
	if len(resp.Choices) != batch || resp.Usage.Details == nil || resp.Usage.Details.CachedTokens == nil {
		return out, errors.New("vllm: completion lacks choices or cached-token usage (--enable-prompt-tokens-details)")
	}
	out.CachedTokens = *resp.Usage.Details.CachedTokens
	for _, c := range resp.Choices {
		if c.Index < 0 || c.Index >= batch || len(c.Logprobs.Tokens) != maxTokens || len(c.Logprobs.Logprobs) != maxTokens {
			return out, errors.New("vllm: short or misplaced completion choice")
		}
		row := Row{Logprobs: c.Logprobs.Logprobs}
		for _, t := range c.Logprobs.Tokens {
			id, err := strconv.Atoi(strings.TrimPrefix(t, "token_id:"))
			if err != nil || !strings.HasPrefix(t, "token_id:") {
				return out, fmt.Errorf("vllm: token %q is not a token ID", t)
			}
			row.Tokens = append(row.Tokens, id)
		}
		out.Rows[c.Index] = row
	}
	return out, nil
}

// replicas runs fn on every data-parallel replica at once (expert-parallel
// replicas step together) and returns the slowest time.
func replicas(ctx context.Context, dp int, fn func(context.Context) (float64, error)) (float64, error) {
	if dp <= 1 {
		return fn(ctx)
	}
	seconds := make([]float64, dp)
	errs := make([]error, dp)
	var wg sync.WaitGroup
	for r := 0; r < dp; r++ {
		wg.Add(1)
		go func(r int) {
			defer wg.Done()
			seconds[r], errs[r] = fn(WithRank(ctx, r))
		}(r)
	}
	wg.Wait()
	slowest := 0.0
	for _, s := range seconds {
		slowest = math.Max(slowest, s)
	}
	return slowest, errors.Join(errs...)
}

// Warmup runs every case after a prefix-cache reset, so each prefill
// really runs and replays (or captures) its graphs. report is called after
// each case.
func (e *Engine) Warmup(ctx context.Context, model string, dp int, cases []profile.WarmupCase,
	report func(i int, c profile.WarmupCase, seconds float64)) error {
	for i, c := range cases {
		if err := e.ResetPrefixCache(ctx); err != nil {
			return err
		}
		seconds, err := replicas(ctx, dp, func(ctx context.Context) (float64, error) {
			out, err := e.Complete(ctx, model, c.PromptTokens, c.Batch, c.GeneratedTokens)
			if err == nil && out.CachedTokens != 0 {
				err = fmt.Errorf("warmup case %+v hit the prefix cache after a reset", c)
			}
			return out.Seconds, err
		})
		if err != nil {
			return err
		}
		if report != nil {
			report(i, c, seconds)
		}
	}
	return nil
}

// Canary is a miss and a hit of the same prompt on a fresh prefix cache.
type Canary = manifest.Canary

// RunCanary sends the canary prompt twice. On a restored engine the first
// must miss (the KV cache came back empty) and the second must hit.
func (e *Engine) RunCanary(ctx context.Context, model string, promptTokens, maxTokens int) (Canary, error) {
	c := Canary{PromptTokens: promptTokens, MaxTokens: maxTokens}
	miss, err := e.Complete(ctx, model, promptTokens, 1, maxTokens)
	if err != nil {
		return c, err
	}
	hit, err := e.Complete(ctx, model, promptTokens, 1, maxTokens)
	if err != nil {
		return c, err
	}
	c.Miss, c.MissCached, c.MissSeconds = miss.Rows[0], miss.CachedTokens, miss.Seconds
	c.Hit, c.HitCached, c.HitSeconds = hit.Rows[0], hit.CachedTokens, hit.Seconds
	return c, nil
}

// CompareCanary checks a restored canary against the donor's under the
// profile's policy. It returns a one-line summary; the error is set only
// when the policy fails the restore.
func CompareCanary(policy profile.Canary, want, got Canary) (string, error) {
	var problems []string
	if got.MissCached != 0 {
		problems = append(problems, fmt.Sprintf("first request hit %d cached tokens (KV cache not empty)", got.MissCached))
	}
	// Hybrid models cache prefixes in large aligned blocks: a short canary
	// may hit nothing on the donor too, and then there is nothing to check.
	if want.HitCached > 0 && got.HitCached <= 0 {
		problems = append(problems, "repeated request did not hit the prefix cache")
	}
	tol := policy.Tolerance
	if tol == 0 {
		tol = 1e-3
	}
	for _, pass := range []struct {
		name      string
		want, got Row
	}{{"miss", want.Miss, got.Miss}, {"hit", want.Hit, got.Hit}} {
		if p := compareRow(policy.Mode, tol, pass.want, pass.got); p != "" {
			problems = append(problems, pass.name+": "+p)
		}
	}
	summary := "canary matches"
	if len(problems) > 0 {
		summary = "canary differs: " + strings.Join(problems, "; ")
	}
	if len(problems) > 0 && policy.Mode != "warn" {
		return summary, errors.New(summary)
	}
	return summary, nil
}

// compareRow: bitwise needs every token equal and logprobs within tol;
// tolerance needs the first token equal and logprobs within tol up to the
// first divergence.
func compareRow(mode string, tol float64, want, got Row) string {
	n := min(len(want.Tokens), len(got.Tokens), len(want.Logprobs), len(got.Logprobs))
	if n == 0 || len(want.Tokens) != len(got.Tokens) {
		return "lengths differ"
	}
	worst, at := 0.0, 0
	for i := 0; i < n; i++ {
		if want.Tokens[i] != got.Tokens[i] {
			if mode == "bitwise" || i == 0 {
				return fmt.Sprintf("token %d is %d, donor had %d", i, got.Tokens[i], want.Tokens[i])
			}
			break
		}
		if d := math.Abs(want.Logprobs[i] - got.Logprobs[i]); d > worst {
			worst, at = d, i
		}
	}
	if worst > tol {
		return fmt.Sprintf("logprob %d off by %.3g (tolerance %.3g)", at, worst, tol)
	}
	return ""
}
