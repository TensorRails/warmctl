package profile

import "sort"

// WarmupCase is one bake warmup request shape: a batch of identical prompts
// of PromptTokens, decoding GeneratedTokens, with the prefix cache reset
// first so the prefill really runs.
type WarmupCase struct {
	PromptTokens    int `json:"prompt_tokens"`
	Batch           int `json:"batch"`
	GeneratedTokens int `json:"generated_tokens"`
}

// WarmupCases expands the profile's warmup generator. Without a warmup
// section it is prompts of 32, 128 and 512 tokens at every decode batch,
// which replays every full decode graph. With piecewise graphs one lone
// prompt is added for each capture size no case's prefill step lands on.
func (l Launch) WarmupCases() []WarmupCase {
	w := l.Profile.Warmup
	gen := w.GeneratedTokens
	if gen == 0 {
		gen = 32
	}
	batches := l.Profile.Graphs.decodeBatches()
	var cases []WarmupCase
	if isZeroWarmup(w) {
		cases = mixedCases([]Mixed{{PromptTokens: []int{32, 128, 512}, Batches: batches}}, gen)
	} else {
		cases = append(l.prefillCases(w, batches, gen), mixedCases(w.Mixed, gen)...)
	}
	return l.addPiecewiseFillers(cases, gen)
}

// prefillCases are the lone prefills (capture sizes, then the declared
// ranges, each length once) and one decode batch per decode graph; the
// decode prompt at batch 1 is a lone prefill too.
func (l Launch) prefillCases(w Warmup, batches []int, gen int) []WarmupCase {
	var lengths []int
	if w.PrefillCaptureSizes {
		lengths = append(lengths, l.CaptureSizes...)
	}
	for _, r := range w.PrefillTokens {
		lengths = append(lengths, r.Values()...)
	}
	var decodes []WarmupCase
	for _, b := range batches {
		if w.DecodePromptTokens == 0 {
			break
		}
		if b == 1 {
			lengths = append(lengths, w.DecodePromptTokens)
			continue
		}
		decodes = append(decodes, WarmupCase{w.DecodePromptTokens, b, gen})
	}
	var cases []WarmupCase
	seen := map[int]bool{}
	for _, n := range lengths {
		if n > 0 && !seen[n] {
			seen[n] = true
			cases = append(cases, WarmupCase{n, 1, gen})
		}
	}
	return append(cases, decodes...)
}

func mixedCases(blocks []Mixed, gen int) []WarmupCase {
	var cases []WarmupCase
	for _, m := range blocks {
		for _, n := range m.PromptTokens {
			for _, b := range m.Batches {
				cases = append(cases, WarmupCase{n, b, gen})
			}
		}
	}
	return cases
}

func (l Launch) piecewise() bool {
	return l.Mode == "FULL_AND_PIECEWISE" || l.Mode == "PIECEWISE"
}

func (l Launch) addPiecewiseFillers(cases []WarmupCase, gen int) []WarmupCase {
	if !l.piecewise() {
		return cases
	}
	maxBatched := asInt(l.Options["max-num-batched-tokens"])
	sizes := append([]int(nil), l.CaptureSizes...)
	sort.Ints(sizes)
	covered := map[int]bool{}
	for _, c := range cases {
		tokens := c.PromptTokens * c.Batch
		if maxBatched > 0 && tokens > maxBatched {
			continue
		}
		if i := sort.SearchInts(sizes, tokens); i < len(sizes) {
			covered[sizes[i]] = true
		}
	}
	for _, size := range sizes {
		if !covered[size] {
			cases = append(cases, WarmupCase{size, 1, gen})
		}
	}
	return cases
}
