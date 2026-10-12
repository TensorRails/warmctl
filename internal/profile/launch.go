package profile

import (
	"encoding/json"
	"fmt"
	"sort"
	"strconv"
	"strings"
)

// Lifecycle flags are what warmctl adds to a serving launch; a stock vLLM
// baseline runs the same launch without them.
var Lifecycle = []string{
	"--enable-sleep-mode",
	"--enable-nccl-comm-suspend",
	"--worker-extension-cls=warmctl_vllm.worker.Extension",
}

// Launch is the effective engine launch of a profile on its defaults.
type Launch struct {
	Profile      Profile           `json:"profile"`
	Options      map[string]any    `json:"options"`
	Argv         []string          `json:"argv"`
	Env          map[string]string `json:"env"`
	CaptureSizes []int             `json:"capture_sizes"`
	Mode         string            `json:"cudagraph_mode"`
	Speculative  int               `json:"speculative_tokens"`
	Memory       int64             `json:"memory_bytes"`
	SHM          int64             `json:"shm_bytes"`
}

// Resolve merges p over defaults and encodes the launch.
func Resolve(defaults, p Profile) (Launch, error) {
	m := Merge(defaults, p)
	if err := m.Validate(); err != nil {
		return Launch{}, err
	}
	l := Launch{Profile: m, Env: m.Environment(), Options: baseOptions(m)}
	l.Speculative = speculativeTokens(l.Options)
	l.setGraphs(m)
	argv, err := EncodeArgv(m.Model.MountPath(), l.Options, m.Argv.Negatable)
	if err != nil {
		return Launch{}, fmt.Errorf("profile %s: %w", m.ID, err)
	}
	l.Argv = argv
	if m.Resources.Memory != "" {
		l.Memory, _ = ParseSize(m.Resources.Memory)
	}
	if m.Resources.SHM != "" {
		l.SHM, _ = ParseSize(m.Resources.SHM)
	}
	return l, nil
}

// baseOptions are the vllm section plus what parallel and model declare.
func baseOptions(m Profile) map[string]any {
	options := map[string]any{}
	for k, v := range m.VLLM {
		options[k] = v
	}
	options["tensor-parallel-size"] = m.Parallel.TP
	if m.Parallel.DP > 1 {
		options["data-parallel-size"] = m.Parallel.DP
	}
	if m.Parallel.EP {
		options["enable-expert-parallel"] = true
	}
	if _, ok := options["served-model-name"]; !ok {
		names := m.Model.ServedNames
		if len(names) == 0 {
			names = []string{m.ID}
		}
		options["served-model-name"] = toAnyList(names)
	}
	return options
}

// setGraphs derives the CUDA graph capture sizes and, unless the profile
// writes vllm.compilation-config out, the compilation config itself.
func (l *Launch) setGraphs(m Profile) {
	batches := m.Graphs.decodeBatches()
	if _, ok := l.Options["max-num-seqs"]; !ok {
		l.Options["max-num-seqs"] = batches[len(batches)-1]
	}
	l.Mode = m.Graphs.Mode
	if l.Mode == "" {
		l.Mode = "FULL_DECODE_ONLY"
	}
	for _, b := range batches {
		l.CaptureSizes = append(l.CaptureSizes, b*(1+l.Speculative))
	}
	l.CaptureSizes = append(l.CaptureSizes, m.Graphs.ExtraSizes...)
	if explicit, ok := l.Options["compilation-config"].(map[string]any); ok {
		l.readExplicitGraphs(explicit)
		return
	}
	fuse := m.Graphs.FuseAllReduceRMS != nil && *m.Graphs.FuseAllReduceRMS
	mode := 3
	if m.Graphs.CompileMode != nil {
		mode = *m.Graphs.CompileMode
	}
	l.Options["compilation-config"] = map[string]any{
		"mode": mode, "cudagraph_mode": l.Mode, "cudagraph_capture_sizes": toAnyList(l.CaptureSizes),
		"pass_config": map[string]any{"fuse_allreduce_rms": fuse},
	}
}

func (l *Launch) readExplicitGraphs(explicit map[string]any) {
	if mode, _ := explicit["cudagraph_mode"].(string); mode != "" {
		l.Mode = mode
	}
	if sizes, ok := explicit["cudagraph_capture_sizes"].([]any); ok {
		l.CaptureSizes = nil
		for _, s := range sizes {
			l.CaptureSizes = append(l.CaptureSizes, asInt(s))
		}
	}
}

func speculativeTokens(options map[string]any) int {
	spec, ok := options["speculative-config"].(map[string]any)
	if !ok {
		return 0
	}
	return asInt(spec["num_speculative_tokens"])
}

func toAnyList[T any](in []T) []any {
	out := make([]any, len(in))
	for i, v := range in {
		out[i] = v
	}
	return out
}

// StockArgs are the argv flags without `vllm serve <model>` and without the
// lifecycle flags: what a stock vLLM A/B baseline runs.
func (l Launch) StockArgs() []string {
	skip := map[string]bool{}
	for _, f := range Lifecycle {
		skip[f] = true
	}
	var out []string
	for _, arg := range l.Argv[3:] {
		if !skip[arg] {
			out = append(out, arg)
		}
	}
	return out
}

// EncodeArgv turns options into `vllm serve <model> --flag ...`, keys sorted.
// The YAML type of a value picks the encoding: bool is a switch (`--x`, or
// `--no-x` when false and negatable), ints and floats take a separate value,
// strings use `--x=v`, lists repeat values after the flag, maps become
// canonical JSON.
func EncodeArgv(modelPath string, options map[string]any, negatable []string) ([]string, error) {
	neg := map[string]bool{}
	for _, n := range negatable {
		neg[n] = true
	}
	keys := make([]string, 0, len(options))
	for k := range options {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	args := []string{"vllm", "serve", modelPath}
	for _, key := range keys {
		if !flagName(key) {
			return nil, fmt.Errorf("invalid vllm option name %q", key)
		}
		part, err := encodeOption(key, options[key], neg[key])
		if err != nil {
			return nil, err
		}
		args = append(args, part...)
	}
	return args, nil
}

func flagName(key string) bool {
	if key == "" {
		return false
	}
	for _, c := range key {
		if (c < 'a' || c > 'z') && (c < '0' || c > '9') && c != '-' && c != '_' {
			return false
		}
	}
	return true
}

func encodeOption(key string, value any, negatable bool) ([]string, error) {
	flag := "--" + key
	switch v := value.(type) {
	case bool:
		if v {
			return []string{flag}, nil
		}
		if negatable {
			return []string{"--no-" + key}, nil
		}
		return nil, nil
	case int:
		return []string{flag, strconv.Itoa(v)}, nil
	case int64:
		return []string{flag, strconv.FormatInt(v, 10)}, nil
	case float64:
		return []string{flag, strconv.FormatFloat(v, 'g', -1, 64)}, nil
	case string:
		if v == "" || strings.ContainsRune(v, 0) {
			return nil, fmt.Errorf("option %s: empty string", key)
		}
		return []string{flag + "=" + v}, nil
	case []any:
		if len(v) == 0 {
			return nil, fmt.Errorf("option %s: empty list", key)
		}
		out := []string{flag}
		for _, item := range v {
			s, err := scalar(item)
			if err != nil || s == "" || strings.HasPrefix(s, "-") {
				return nil, fmt.Errorf("option %s: list items must be plain scalars", key)
			}
			out = append(out, s)
		}
		return out, nil
	case map[string]any:
		raw, err := json.Marshal(v)
		if err != nil {
			return nil, fmt.Errorf("option %s: %w", key, err)
		}
		return []string{flag, string(raw)}, nil
	}
	return nil, fmt.Errorf("option %s: unsupported value type %T", key, value)
}

func scalar(v any) (string, error) {
	switch s := v.(type) {
	case string:
		return s, nil
	case int:
		return strconv.Itoa(s), nil
	case int64:
		return strconv.FormatInt(s, 10), nil
	case float64:
		return strconv.FormatFloat(s, 'g', -1, 64), nil
	case bool:
		return strconv.FormatBool(s), nil
	}
	return "", fmt.Errorf("not a scalar: %T", v)
}
