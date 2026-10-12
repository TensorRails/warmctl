// Package profile loads model profiles and the hardware table from YAML and
// turns a profile into the effective vLLM launch: argv, environment, graph
// capture sizes, warmup grid, canary policy and resources.
//
// Nothing here knows a model or a GPU by name. A profile is data; the code
// only merges it over profiles/_defaults.yaml, validates the few layouts a
// snapshot cannot restore, and encodes options into vLLM flags.
package profile

import (
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path"
	"path/filepath"
	"regexp"
	"slices"
	"sort"
	"strings"

	"gopkg.in/yaml.v3"

	"github.com/TensorRails/warmctl/v2/internal/interpose"
)

// Schema identifies a profile file; DefaultsSchema the defaults file.
const (
	Schema         = "warmctl.profile/v1"
	DefaultsSchema = "warmctl.profile-defaults/v1"
)

// Model is the checkpoint a profile serves.
type Model struct {
	ID            string   `yaml:"id" json:"id"`
	Revision      string   `yaml:"revision" json:"revision"`
	ContainerPath string   `yaml:"container_path,omitempty" json:"container_path,omitempty"`
	ServedNames   []string `yaml:"served_names,omitempty" json:"served_names,omitempty"`
}

// MountPath is where the model directory is mounted inside the container.
func (m Model) MountPath() string {
	if m.ContainerPath == "" {
		return "/model"
	}
	return m.ContainerPath
}

// Parallel is the engine layout. EP shards experts whole across ranks.
type Parallel struct {
	TP int  `yaml:"tp" json:"tp"`
	DP int  `yaml:"dp,omitempty" json:"dp,omitempty"`
	EP bool `yaml:"ep,omitempty" json:"ep,omitempty"`
}

// Ranks is the number of GPU workers: TP x DP.
func (p Parallel) Ranks() int {
	return p.TP * max(p.DP, 1)
}

// IPCMode resolves the profile's ipc before the engine starts: auto stays
// auto (the donor checks the GPUs) unless the engine has one GPU process,
// which shares no memory and runs native.
func (p Profile) IPCMode() string {
	switch {
	case p.IPC == "" || p.IPC == interpose.Native:
		return interpose.Native
	case p.IPC == interpose.Auto && p.Parallel.Ranks() <= 1:
		return interpose.Native
	}
	return p.IPC
}

// Graphs describes CUDA graph capture when vllm.compilation-config is not
// written out explicitly. Capture sizes are batch x (1+k) tokens, k being the
// speculative token count declared in vllm.speculative-config.
type Graphs struct {
	Mode             string `yaml:"mode,omitempty" json:"mode,omitempty"`
	DecodeBatches    []int  `yaml:"decode_batches,omitempty" json:"decode_batches,omitempty"`
	ExtraSizes       []int  `yaml:"extra_capture_sizes,omitempty" json:"extra_capture_sizes,omitempty"`
	FuseAllReduceRMS *bool  `yaml:"fuse_allreduce_rms,omitempty" json:"fuse_allreduce_rms,omitempty"`
	CompileMode      *int   `yaml:"compile_mode,omitempty" json:"compile_mode,omitempty"`
}

func (g Graphs) decodeBatches() []int {
	if len(g.DecodeBatches) == 0 {
		return []int{1, 2, 4, 8}
	}
	return g.DecodeBatches
}

// Range is an inclusive token range with a step, usable where a plain
// integer is: `{from: 64, to: 2560, step: 64}` or `16384`.
type Range struct {
	From, To, Step int
}

// UnmarshalYAML accepts either an integer or a from/to/step mapping.
func (r *Range) UnmarshalYAML(node *yaml.Node) error {
	var single int
	if node.Kind == yaml.ScalarNode {
		if err := node.Decode(&single); err != nil {
			return err
		}
		*r = Range{single, single, 1}
		return nil
	}
	var v struct {
		From int `yaml:"from"`
		To   int `yaml:"to"`
		Step int `yaml:"step"`
	}
	if err := node.Decode(&v); err != nil {
		return err
	}
	if v.From < 1 || v.To < v.From || v.Step < 1 {
		return fmt.Errorf("invalid token range %d..%d step %d", v.From, v.To, v.Step)
	}
	*r = Range{v.From, v.To, v.Step}
	return nil
}

// Values expands the range.
func (r Range) Values() []int {
	var out []int
	for n := r.From; n <= r.To; n += r.Step {
		out = append(out, n)
	}
	return out
}

// Mixed is a prompt-length by batch-size block of warmup requests.
type Mixed struct {
	PromptTokens []int `yaml:"prompt_tokens" json:"prompt_tokens"`
	Batches      []int `yaml:"batches" json:"batches"`
}

// Warmup is the generator of the bake warmup grid. An empty section means
// prompts of 32/128/512 tokens at every decode batch.
type Warmup struct {
	PrefillCaptureSizes bool    `yaml:"prefill_capture_sizes,omitempty" json:"prefill_capture_sizes,omitempty"`
	PrefillTokens       []Range `yaml:"prefill_tokens,omitempty" json:"prefill_tokens,omitempty"`
	DecodePromptTokens  int     `yaml:"decode_prompt_tokens,omitempty" json:"decode_prompt_tokens,omitempty"`
	Mixed               []Mixed `yaml:"mixed,omitempty" json:"mixed,omitempty"`
	GeneratedTokens     int     `yaml:"generated_tokens,omitempty" json:"generated_tokens,omitempty"`
}

// Canary is the restore correctness policy: bitwise, tolerance or warn.
type Canary struct {
	Mode         string  `yaml:"mode,omitempty" json:"mode,omitempty"`
	Tolerance    float64 `yaml:"tolerance,omitempty" json:"tolerance,omitempty"`
	PrefixTokens int     `yaml:"prefix_tokens,omitempty" json:"prefix_tokens,omitempty"`
}

// Resources are the container limits.
type Resources struct {
	Memory string `yaml:"memory,omitempty" json:"memory,omitempty"`
	PIDs   int    `yaml:"pids,omitempty" json:"pids,omitempty"`
	SHM    string `yaml:"shm,omitempty" json:"shm,omitempty"`
	CPUs   int    `yaml:"cpus,omitempty" json:"cpus,omitempty"`
}

// Hooks are the optional adapter behaviours a profile turns on.
type Hooks struct {
	HostTables      string   `yaml:"host_tables,omitempty" json:"host_tables,omitempty"`
	MoETunedConfigs string   `yaml:"moe_tuned_configs,omitempty" json:"moe_tuned_configs,omitempty"`
	Plugins         []string `yaml:"plugins,omitempty" json:"plugins,omitempty"`
}

// RestorePolicy is what a restore does after the engine is awake.
type RestorePolicy struct {
	Warmup      string `yaml:"warmup,omitempty" json:"warmup,omitempty"`
	CUDARestore string `yaml:"cuda_restore,omitempty" json:"cuda_restore,omitempty"`
}

// Argv holds encoder settings that are data, not code.
type Argv struct {
	// Negatable flags take `--no-<flag>` when false; every other false
	// boolean is simply omitted.
	Negatable []string `yaml:"negatable,omitempty" json:"negatable,omitempty"`
}

// Profile is one YAML profile, or the defaults file.
type Profile struct {
	Schema    string             `yaml:"schema" json:"schema"`
	ID        string             `yaml:"id,omitempty" json:"id"`
	Image     string             `yaml:"image,omitempty" json:"image"`
	Model     Model              `yaml:"model,omitempty" json:"model"`
	Parallel  Parallel           `yaml:"parallel,omitempty" json:"parallel"`
	Hardware  []string           `yaml:"hardware,omitempty" json:"hardware"`
	VLLM      map[string]any     `yaml:"vllm,omitempty" json:"vllm"`
	Env       map[string]*string `yaml:"env,omitempty" json:"env"`
	Graphs    Graphs             `yaml:"graphs,omitempty" json:"graphs"`
	Warmup    Warmup             `yaml:"warmup,omitempty" json:"warmup"`
	Canary    Canary             `yaml:"canary,omitempty" json:"canary"`
	Resources Resources          `yaml:"resources,omitempty" json:"resources"`
	Hooks     Hooks              `yaml:"hooks,omitempty" json:"hooks"`
	// IPC is how cross-process CUDA memory is checkpointed: native,
	// interpose or auto (docs/interpose.md).
	IPC     string        `yaml:"ipc,omitempty" json:"ipc"`
	Restore RestorePolicy `yaml:"restore,omitempty" json:"restore"`
	Argv    Argv          `yaml:"argv,omitempty" json:"argv"`
}

// Loader finds profiles in the embedded assets and in extra directories.
type Loader struct {
	Assets fs.FS
	Dirs   []string
}

var idPattern = regexp.MustCompile(`^[a-z0-9][a-z0-9._-]{0,127}$`)

// Decode reads one strict YAML document.
func Decode(r io.Reader, out any) error {
	dec := yaml.NewDecoder(r)
	dec.KnownFields(true)
	if err := dec.Decode(out); err != nil && !errors.Is(err, io.EOF) {
		return err
	}
	return nil
}

// Defaults reads profiles/_defaults.yaml from the assets.
func (l Loader) Defaults() (Profile, error) {
	var d Profile
	if err := l.decodeAsset("profiles/_defaults.yaml", &d); err != nil {
		return d, err
	}
	if d.Schema != DefaultsSchema {
		return d, fmt.Errorf("profiles/_defaults.yaml: schema %q, want %s", d.Schema, DefaultsSchema)
	}
	return d, nil
}

// Load resolves ref: a path to a YAML file, or a profile id looked up in
// Dirs and then in the embedded profiles directory.
func (l Loader) Load(ref string) (Profile, error) {
	var p Profile
	if strings.ContainsAny(ref, "/\\") || strings.HasSuffix(ref, ".yaml") || strings.HasSuffix(ref, ".yml") {
		f, err := os.Open(ref)
		if err != nil {
			return p, err
		}
		defer f.Close()
		if err := Decode(f, &p); err != nil {
			return p, fmt.Errorf("%s: %w", ref, err)
		}
		if p.ID == "" {
			p.ID = strings.TrimSuffix(strings.TrimSuffix(filepath.Base(ref), ".yaml"), ".yml")
		}
		return p, p.check()
	}
	if !idPattern.MatchString(ref) {
		return p, fmt.Errorf("invalid profile id %q", ref)
	}
	for _, dir := range l.Dirs {
		f, err := os.Open(filepath.Join(dir, ref+".yaml"))
		if err != nil {
			continue
		}
		defer f.Close()
		if err := Decode(f, &p); err != nil {
			return p, fmt.Errorf("%s: %w", f.Name(), err)
		}
		return p, p.check()
	}
	if err := l.decodeAsset(path.Join("profiles", ref+".yaml"), &p); err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			return p, fmt.Errorf("unknown profile %q", ref)
		}
		return p, err
	}
	return p, p.check()
}

// IDs lists the embedded profiles.
func (l Loader) IDs() ([]string, error) {
	entries, err := fs.ReadDir(l.Assets, "profiles")
	if err != nil {
		return nil, err
	}
	var ids []string
	for _, e := range entries {
		name := e.Name()
		if strings.HasPrefix(name, "_") || !strings.HasSuffix(name, ".yaml") {
			continue
		}
		ids = append(ids, strings.TrimSuffix(name, ".yaml"))
	}
	sort.Strings(ids)
	return ids, nil
}

func (l Loader) decodeAsset(name string, out any) error {
	f, err := l.Assets.Open(name)
	if err != nil {
		return err
	}
	defer f.Close()
	if err := Decode(f, out); err != nil {
		return fmt.Errorf("%s: %w", name, err)
	}
	return nil
}

func (p Profile) check() error {
	if p.Schema != Schema {
		return fmt.Errorf("profile %s: schema %q, want %s", p.ID, p.Schema, Schema)
	}
	if !idPattern.MatchString(p.ID) {
		return fmt.Errorf("invalid profile id %q", p.ID)
	}
	return nil
}

// Merge lays p over the defaults: scalars win when set, vllm and env merge
// one level deep where a null value removes the default, lists replace.
func Merge(defaults, p Profile) Profile {
	out := defaults
	out.Schema, out.ID = Schema, p.ID
	if p.Image != "" {
		out.Image = p.Image
	}
	out.Model = p.Model
	out.Parallel = p.Parallel
	if len(p.Hardware) > 0 {
		out.Hardware = p.Hardware
	}
	out.VLLM = mergeAny(defaults.VLLM, p.VLLM)
	out.Env = mergeEnv(defaults.Env, p.Env)
	out.Graphs = mergeGraphs(defaults.Graphs, p.Graphs)
	if !isZeroWarmup(p.Warmup) {
		out.Warmup = p.Warmup
	}
	if p.Warmup.GeneratedTokens != 0 {
		out.Warmup.GeneratedTokens = p.Warmup.GeneratedTokens
	} else if out.Warmup.GeneratedTokens == 0 {
		out.Warmup.GeneratedTokens = defaults.Warmup.GeneratedTokens
	}
	out.Canary = mergeCanary(defaults.Canary, p.Canary)
	out.Resources = mergeResources(defaults.Resources, p.Resources)
	out.Hooks = p.Hooks
	if p.IPC != "" {
		out.IPC = p.IPC
	}
	if p.Restore.Warmup != "" {
		out.Restore.Warmup = p.Restore.Warmup
	}
	if p.Restore.CUDARestore != "" {
		out.Restore.CUDARestore = p.Restore.CUDARestore
	}
	if len(p.Argv.Negatable) > 0 {
		out.Argv = p.Argv
	}
	return out
}

func mergeAny(base, over map[string]any) map[string]any {
	out := map[string]any{}
	for k, v := range base {
		out[k] = v
	}
	for k, v := range over {
		if v == nil {
			delete(out, k)
			continue
		}
		out[k] = v
	}
	return out
}

func mergeEnv(base, over map[string]*string) map[string]*string {
	out := map[string]*string{}
	for k, v := range base {
		if v != nil {
			out[k] = v
		}
	}
	for k, v := range over {
		if v == nil {
			delete(out, k)
			continue
		}
		out[k] = v
	}
	return out
}

func mergeGraphs(base, over Graphs) Graphs {
	out := base
	if over.Mode != "" {
		out.Mode = over.Mode
	}
	if len(over.DecodeBatches) > 0 {
		out.DecodeBatches = over.DecodeBatches
	}
	if len(over.ExtraSizes) > 0 {
		out.ExtraSizes = over.ExtraSizes
	}
	if over.FuseAllReduceRMS != nil {
		out.FuseAllReduceRMS = over.FuseAllReduceRMS
	}
	if over.CompileMode != nil {
		out.CompileMode = over.CompileMode
	}
	return out
}

func isZeroWarmup(w Warmup) bool {
	return !w.PrefillCaptureSizes && len(w.PrefillTokens) == 0 && w.DecodePromptTokens == 0 && len(w.Mixed) == 0
}

func mergeCanary(base, over Canary) Canary {
	out := base
	if over.Mode != "" {
		out.Mode = over.Mode
	}
	if over.Tolerance != 0 {
		out.Tolerance = over.Tolerance
	}
	if over.PrefixTokens != 0 {
		out.PrefixTokens = over.PrefixTokens
	}
	return out
}

func mergeResources(base, over Resources) Resources {
	out := base
	if over.Memory != "" {
		out.Memory = over.Memory
	}
	if over.PIDs != 0 {
		out.PIDs = over.PIDs
	}
	if over.SHM != "" {
		out.SHM = over.SHM
	}
	if over.CPUs != 0 {
		out.CPUs = over.CPUs
	}
	return out
}

// Environment flattens the merged env map.
func (p Profile) Environment() map[string]string {
	out := map[string]string{}
	for k, v := range p.Env {
		if v != nil {
			out[k] = *v
		}
	}
	return out
}

var (
	envName      = regexp.MustCompile(`^[A-Za-z_][A-Za-z0-9_]*$`)
	imagePattern = regexp.MustCompile(`^([^\s@]+@)?sha256:[0-9a-f]{64}$`)
	// restorableAll2All are the expert-parallel dispatch backends whose
	// memory a snapshot can bring back: AgRs and passthrough ride the
	// suspended PyNccl communicators, FlashInfer one-sided rebuilds its
	// workspace at the same addresses. DeepEP and its relatives keep NVSHMEM
	// or cudaIpc peer buffers with no detach API.
	restorableAll2All = map[string]bool{
		"allgather_reducescatter":     true,
		"passthrough":                 true,
		"flashinfer_nvlink_one_sided": true,
	}
	// forbiddenEnv breaks cuda-checkpoint or changes the IPC protocol.
	forbiddenEnv = []string{"NCCL_CUMEM_ENABLE", "CUDA_CHECKPOINT_JOB_FILE", "CUDA_VISIBLE_DEVICES"}
)

// Validate refuses what a snapshot cannot capture or restore. Everything
// else is the user's choice; a bad vLLM flag fails at engine start.
func (p Profile) Validate() error {
	for _, check := range []func() error{p.check, p.validateLaunch, p.validateLayout, p.validateEnv, p.validateChoices} {
		if err := check(); err != nil {
			return err
		}
	}
	return nil
}

func (p Profile) validateLaunch() error {
	if !imagePattern.MatchString(p.Image) {
		return fmt.Errorf("profile %s: image must be pinned as name@sha256:<64 hex> or a local sha256 image id", p.ID)
	}
	if strings.TrimSpace(p.Model.ID) == "" {
		return fmt.Errorf("profile %s: model.id is required", p.ID)
	}
	mount := p.Model.MountPath()
	if !path.IsAbs(mount) || path.Clean(mount) != mount || mount == "/" {
		return fmt.Errorf("profile %s: model.container_path must be a clean absolute path", p.ID)
	}
	if p.Parallel.TP < 1 || p.Parallel.DP < 0 {
		return fmt.Errorf("profile %s: parallel.tp must be >= 1", p.ID)
	}
	for field, value := range map[string]string{"resources.memory": p.Resources.Memory, "resources.shm": p.Resources.SHM} {
		if _, err := ParseSize(value); value != "" && err != nil {
			return fmt.Errorf("profile %s: %s: %w", p.ID, field, err)
		}
	}
	return nil
}

// validateLayout refuses the parallel layouts whose memory no snapshot can
// bring back.
func (p Profile) validateLayout() error {
	if pp, ok := p.VLLM["pipeline-parallel-size"]; ok && asInt(pp) > 1 {
		return fmt.Errorf("profile %s: pipeline parallelism cannot be snapshotted", p.ID)
	}
	if backend, ok := p.VLLM["all2all-backend"]; ok {
		if s, _ := backend.(string); !restorableAll2All[s] {
			return fmt.Errorf("profile %s: all2all-backend %v keeps peer buffers no snapshot can restore", p.ID, backend)
		}
	}
	engram, ok := p.VLLM["engram-config"].(map[string]any)
	if !ok || p.Parallel.DP <= 1 {
		return nil
	}
	across, _ := engram["embedding_across_dp"].(bool)
	shared, hasShared := engram["dp_shared_memory"].(bool)
	if !across || !hasShared || shared {
		return fmt.Errorf("profile %s: data-parallel engram needs embedding_across_dp and no dp_shared_memory", p.ID)
	}
	return nil
}

func (p Profile) validateEnv() error {
	for _, key := range forbiddenEnv {
		if _, ok := p.Env[key]; ok {
			return fmt.Errorf("profile %s: env %s is owned by warmctl", p.ID, key)
		}
	}
	for key, value := range p.Env {
		if credential(key) {
			return fmt.Errorf("profile %s: credential variable %s cannot live in a profile", p.ID, key)
		}
		if !envName.MatchString(key) || (value != nil && strings.ContainsRune(*value, 0)) {
			return fmt.Errorf("profile %s: invalid env entry %q", p.ID, key)
		}
	}
	return nil
}

func credential(key string) bool {
	upper := strings.ToUpper(key)
	for _, suffix := range []string{"_TOKEN", "_PASSWORD", "_SECRET", "_API_KEY"} {
		if strings.HasSuffix(upper, suffix) {
			return true
		}
	}
	return false
}

func (p Profile) validateChoices() error {
	choices := []struct {
		field, value string
		allowed      []string
	}{
		{"canary.mode", p.Canary.Mode, []string{"", "bitwise", "tolerance", "warn"}},
		{"restore.cuda_restore", p.Restore.CUDARestore, []string{"", "auto", "serial", "parallel"}},
		{"restore.warmup", p.Restore.Warmup, []string{"", "none", "canary", "full"}},
		{"hooks.host_tables", p.Hooks.HostTables, []string{"", "off", "auto"}},
		{"ipc", p.IPC, []string{"", interpose.Native, interpose.Interpose, interpose.Auto}},
	}
	for _, c := range choices {
		if !slices.Contains(c.allowed, c.value) {
			return fmt.Errorf("profile %s: %s %q is not one of %v", p.ID, c.field, c.value, c.allowed[1:])
		}
	}
	return nil
}

func asInt(v any) int {
	switch n := v.(type) {
	case int:
		return n
	case int64:
		return int(n)
	case float64:
		return int(n)
	}
	return 0
}

// Set applies one command-line override, "vllm.<flag>=<yaml value>",
// "env.<NAME>=<value>" or "ipc=<mode>"; the value null removes the key, as
// in YAML.
func (p *Profile) Set(expr string) error {
	key, value, ok := strings.Cut(expr, "=")
	if ok && key == "ipc" {
		p.IPC = value
		return p.validateChoices()
	}
	section, name, dotted := strings.Cut(key, ".")
	if !ok || !dotted || name == "" {
		return fmt.Errorf("--set %q: want vllm.<flag>=<value> or env.<NAME>=<value>", expr)
	}
	var parsed any
	if err := yaml.Unmarshal([]byte(value), &parsed); err != nil {
		return fmt.Errorf("--set %q: %w", expr, err)
	}
	switch section {
	case "vllm":
		if p.VLLM == nil {
			p.VLLM = map[string]any{}
		}
		p.VLLM[name] = parsed
	case "env":
		if p.Env == nil {
			p.Env = map[string]*string{}
		}
		if parsed == nil {
			p.Env[name] = nil
		} else {
			v := value
			p.Env[name] = &v
		}
	default:
		return fmt.Errorf("--set %q: only vllm.* and env.* can be set", expr)
	}
	return nil
}
