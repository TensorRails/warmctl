// Package manifest defines manifest.json, the one description of a snapshot.
// A restore reads everything it needs from it and never derives a launch
// from code: the effective profile, the argv and environment the donor ran,
// the hardware it ran on, the processes that own CUDA state, the files of
// the snapshot and, per rank, how the weights pool was exported.
package manifest

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path"
	"path/filepath"
	"regexp"
	"strings"
	"time"

	"github.com/TensorRails/warmctl/v2/internal/profile"
)

// Schema identifies manifest.json.
const Schema = "warmctl.snapshot/v2"

// Manifest is manifest.json.
type Manifest struct {
	Schema    string          `json:"schema"`
	ID        string          `json:"id"`
	CreatedAt time.Time       `json:"created_at"`
	Tool      string          `json:"warmctl"`
	Profile   profile.Profile `json:"profile"`
	Launch    Launch          `json:"launch"`
	Hardware  Hardware        `json:"hardware"`
	Process   Process         `json:"process"`
	Files     []File          `json:"files"`
	Weights   Weights         `json:"weights"`
	Canary    *Canary         `json:"canary,omitempty"`
}

// Launch is the engine launch exactly as the donor ran it.
type Launch struct {
	Image string `json:"image"`
	// Model is the served name requests use.
	Model string `json:"model"`
	// ModelDir is the donor host's model directory.
	ModelDir     string            `json:"model_dir"`
	ModelSHA256  string            `json:"model_sha256"`
	Argv         []string          `json:"argv"`
	Env          map[string]string `json:"env"`
	CaptureSizes []int             `json:"capture_sizes"`
	Ranks        int               `json:"ranks"`
}

// GPU is one device of the donor, in rank order.
type GPU struct {
	Rank      int    `json:"rank"`
	Index     int    `json:"index"`
	UUID      string `json:"uuid"`
	PCIBusID  string `json:"pci_bus_id"`
	Name      string `json:"name"`
	MemoryMiB int64  `json:"memory_mib"`
}

// Hardware is what a restore host has to match.
type Hardware struct {
	Row            profile.Hardware `json:"row"`
	GPUs           []GPU            `json:"gpus"`
	Topology       string           `json:"topology"`
	TopologySHA256 string           `json:"topology_sha256"`
	Driver         string           `json:"driver"`
	Kernel         string           `json:"kernel"`
	CPUFlags       []string         `json:"cpu_flags"`
	VLLM           string           `json:"vllm"`
	CRIU           string           `json:"criu"`
	CUDACheckpoint string           `json:"cuda_checkpoint"`
	// DriverPortable: built with the driver's version check off, so the
	// snapshot restores under any release of the same driver branch.
	// DriverFiles are the host driver libraries the dump maps, kept under
	// driver/ and mounted back at these paths under another release.
	DriverPortable bool     `json:"driver_portable,omitempty"`
	DriverFiles    []string `json:"driver_files,omitempty"`
}

// DriverBranch is the branch of an NVIDIA driver version: 580 of 580.173.02.
func DriverBranch(v string) string {
	if i := strings.IndexByte(v, '.'); i > 0 {
		return v[:i]
	}
	return v
}

// Owner is a process of the dumped tree that holds CUDA state. Rank is -1
// for the engine core and the API server.
type Owner struct {
	PID        int    `json:"pid"`
	StartTicks uint64 `json:"start_ticks"`
	Rank       int    `json:"rank"`
	Role       string `json:"role"`
}

// Process describes the dumped process tree.
type Process struct {
	RootPID             int      `json:"root_pid"`
	MaxID               int      `json:"max_id"`
	PIDFloor            int      `json:"pid_floor"`
	NeedsModel          bool     `json:"needs_model"`
	Owners              []Owner  `json:"cuda_owners"`
	VisibleUUIDs        []string `json:"visible_uuids"`
	CollectivesDetached bool     `json:"collectives_detached"`
	NCCLMode            string   `json:"nccl_mode"`
	EngineURL           string   `json:"engine_url"`
	// IPC is "interpose" when the engine ran under cuInterpose: the restore
	// mounts the snapshot's ipc/ where the donor had the build and runs the
	// coordinator after the CUDA restore.
	IPC string `json:"ipc,omitempty"`
	// IPCProcesses are the processes the coordinator prepared and restores.
	IPCProcesses []int `json:"ipc_processes,omitempty"`
}

// File is one member of the snapshot directory, relative to it.
type File struct {
	Path   string `json:"path"`
	Size   int64  `json:"size"`
	SHA256 string `json:"sha256"`
	Role   string `json:"role"`
	Root   string `json:"root,omitempty"`
}

// File roles.
const (
	RoleCRIU      = "criu"
	RoleRuntimeFS = "runtimefs"
	RoleFSMeta    = "runtimefs-meta"
	RoleRemaps    = "remaps"
	RoleJITCache  = "jitcache"
	RoleWeights   = "weights"
	RoleHostTable = "host-table"
	RoleAdapter   = "adapter"
	RoleIPC       = "ipc"
	RoleDriver    = "driver"
)

// Allocation is one entry of the sleep-mode allocator's weights pool: the
// virtual address it must come back at, its size and where its bytes sit in
// the rank's part stream.
type Allocation struct {
	Index  int    `json:"index"`
	Ptr    string `json:"ptr"`
	Bytes  int64  `json:"bytes"`
	Tag    string `json:"tag"`
	Part   int    `json:"part"`
	Offset int64  `json:"offset"`
	// Sum is the allocation's fingerprint read from donor device memory:
	// sum, sum of squares and position-weighted sum of its bytes.
	Sum [3]int64 `json:"sum"`
}

// Part is one exported file. The parts of a rank, concatenated in order,
// are its pool stream: every allocation's bytes in address order.
type Part struct {
	Index  int    `json:"index"`
	Path   string `json:"path"`
	Size   int64  `json:"size"`
	SHA256 string `json:"sha256"`
}

// Fingerprint totals a rank's allocation fingerprints, for the record.
type Fingerprint struct {
	Sum         [3]int64 `json:"sum"`
	Allocations int      `json:"allocations"`
	Bytes       int64    `json:"bytes"`
}

// HostTable is a pinned host table released before the dump.
type HostTable struct {
	Name  string `json:"name"`
	Bytes int64  `json:"bytes"`
	Parts []Part `json:"parts"`
}

// Rank is one worker's export.
type Rank struct {
	Rank        int          `json:"rank"`
	Allocations []Allocation `json:"allocations"`
	Parts       []Part       `json:"parts"`
	Fingerprint Fingerprint  `json:"fingerprint"`
	HostTables  []HostTable  `json:"host_tables,omitempty"`
}

// Weights is the whole export.
type Weights struct {
	PartBytes int64  `json:"part_bytes"`
	Ranks     []Rank `json:"ranks"`
}

// Row is one generated sequence: token IDs and their logprobs.
type Row struct {
	Tokens   []int     `json:"tokens"`
	Logprobs []float64 `json:"logprobs"`
}

// Canary is a miss and a hit of the same token-ID prompt: the donor's
// baseline, and what a restored engine must reproduce.
type Canary struct {
	PromptTokens int     `json:"prompt_tokens"`
	MaxTokens    int     `json:"max_tokens"`
	Miss         Row     `json:"miss"`
	Hit          Row     `json:"hit"`
	MissCached   int     `json:"miss_cached_tokens"`
	HitCached    int     `json:"hit_cached_tokens"`
	MissSeconds  float64 `json:"miss_seconds"`
	HitSeconds   float64 `json:"hit_seconds"`
}

var digestPattern = regexp.MustCompile(`^[0-9a-f]{64}$`)

// IsDigest reports a lowercase hex SHA-256.
func IsDigest(s string) bool { return digestPattern.MatchString(s) }

// Read loads and validates manifest.json.
func Read(name string) (Manifest, error) {
	var m Manifest
	raw, err := os.ReadFile(name)
	if err != nil {
		return m, err
	}
	if err := json.Unmarshal(raw, &m); err != nil {
		return m, fmt.Errorf("%s: %w", name, err)
	}
	return m, m.Validate()
}

// Write stores the manifest atomically.
func (m Manifest) Write(name string) error {
	if err := m.Validate(); err != nil {
		return err
	}
	raw, err := json.MarshalIndent(m, "", "  ")
	if err != nil {
		return err
	}
	tmp, err := os.CreateTemp(filepath.Dir(name), ".manifest-*")
	if err != nil {
		return err
	}
	defer os.Remove(tmp.Name())
	_, err = tmp.Write(append(raw, '\n'))
	if err == nil {
		err = tmp.Sync()
	}
	if closeErr := tmp.Close(); err == nil {
		err = closeErr
	}
	if err != nil {
		return err
	}
	if err := os.Chmod(tmp.Name(), 0o644); err != nil {
		return err
	}
	return os.Rename(tmp.Name(), name)
}

// Validate checks what a restore depends on.
func (m Manifest) Validate() error {
	if m.Schema != Schema {
		return fmt.Errorf("manifest schema %q, want %s", m.Schema, Schema)
	}
	if m.ID == "" || m.Launch.Ranks < 1 || len(m.Launch.Argv) < 3 || m.Process.RootPID < 1 {
		return errors.New("manifest: id, launch and process are required")
	}
	files := map[string]File{}
	for _, f := range m.Files {
		if f.Path == "" || path.IsAbs(f.Path) || path.Clean(f.Path) != f.Path || f.Path == "." {
			return fmt.Errorf("manifest: file path %q is not clean and relative", f.Path)
		}
		if _, dup := files[f.Path]; dup {
			return fmt.Errorf("manifest: file %s listed twice", f.Path)
		}
		if !IsDigest(f.SHA256) || f.Size < 0 {
			return fmt.Errorf("manifest: file %s has no digest", f.Path)
		}
		files[f.Path] = f
	}
	if len(m.Weights.Ranks) != m.Launch.Ranks {
		return fmt.Errorf("manifest: %d weight ranks for %d launch ranks", len(m.Weights.Ranks), m.Launch.Ranks)
	}
	for i, r := range m.Weights.Ranks {
		if err := r.validate(i, files); err != nil {
			return err
		}
	}
	return nil
}

func (r Rank) validate(index int, files map[string]File) error {
	if r.Rank != index {
		return fmt.Errorf("manifest: rank %d at position %d", r.Rank, index)
	}
	if err := r.validateParts(files); err != nil {
		return err
	}
	if err := r.validateAllocations(); err != nil {
		return err
	}
	return r.validateHostTables(files)
}

func (r Rank) validateParts(files map[string]File) error {
	for i, p := range r.Parts {
		if p.Index != i {
			return fmt.Errorf("manifest: rank %d part %d at position %d", r.Rank, p.Index, i)
		}
		listed, ok := files[p.Path]
		if !ok || listed.SHA256 != p.SHA256 || listed.Size != p.Size {
			return fmt.Errorf("manifest: rank %d part %s is not a listed file", r.Rank, p.Path)
		}
	}
	return nil
}

func (r Rank) validateAllocations() error {
	var stream int64
	for _, p := range r.Parts {
		stream += p.Size
	}
	for i, a := range r.Allocations {
		if a.Index != i || a.Bytes < 0 || a.Offset < 0 || a.Part < 0 || a.Part >= len(r.Parts) {
			return fmt.Errorf("manifest: rank %d allocation %d is malformed", r.Rank, i)
		}
		if a.Bytes > 0 && a.Offset >= r.Parts[a.Part].Size {
			return fmt.Errorf("manifest: rank %d allocation %s starts outside its part", r.Rank, a.Ptr)
		}
		if r.streamStart(a)+a.Bytes > stream {
			return fmt.Errorf("manifest: rank %d allocation %s runs past the pool stream", r.Rank, a.Ptr)
		}
	}
	return nil
}

func (r Rank) validateHostTables(files map[string]File) error {
	for _, t := range r.HostTables {
		for _, p := range t.Parts {
			if listed, ok := files[p.Path]; !ok || listed.SHA256 != p.SHA256 {
				return fmt.Errorf("manifest: rank %d host table %s part %s is not a listed file", r.Rank, t.Name, p.Path)
			}
		}
	}
	return nil
}

func (r Rank) streamStart(a Allocation) int64 {
	var start int64
	for _, p := range r.Parts[:a.Part] {
		start += p.Size
	}
	return start + a.Offset
}

// File finds a listed file.
func (m Manifest) File(name string) (File, bool) {
	for _, f := range m.Files {
		if f.Path == name {
			return f, true
		}
	}
	return File{}, false
}

// Bytes is the size of every listed file.
func (m Manifest) Bytes() int64 {
	var n int64
	for _, f := range m.Files {
		n += f.Size
	}
	return n
}

// WeightBytes is the size of the exported pools.
func (m Manifest) WeightBytes() int64 {
	var n int64
	for _, r := range m.Weights.Ranks {
		for _, p := range r.Parts {
			n += p.Size
		}
	}
	return n
}

// LaunchLane are the files a restore reads before its receiver starts: the
// adapter and the cuInterpose build it mounts, the donor's driver libraries.
func (m Manifest) LaunchLane() []File {
	var out []File
	for _, f := range m.Files {
		if f.Role == RoleAdapter || f.Role == RoleIPC || f.Role == RoleDriver {
			out = append(out, f)
		}
	}
	return out
}

// FirstLane are the files a restore needs before the weights: everything
// but the exported parts.
func (m Manifest) FirstLane() []File {
	var out []File
	for _, f := range m.Files {
		if f.Role != RoleWeights && f.Role != RoleHostTable {
			out = append(out, f)
		}
	}
	return out
}

// WeightLane are the exported parts and host tables, in rank order.
func (m Manifest) WeightLane() []File {
	var out []File
	for _, f := range m.Files {
		if f.Role == RoleWeights || f.Role == RoleHostTable {
			out = append(out, f)
		}
	}
	return out
}
