package agent

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path"

	"github.com/TensorRails/warmctl/v2/internal/criu"
	"github.com/TensorRails/warmctl/v2/internal/interpose"
)

// Modes of the agent.
const (
	ModeSnapshot = "snapshot"
	ModeRestore  = "restore"
)

// Tree is a directory the snapshot carries as a runtimefs archive.
type Tree struct {
	Name string `json:"name"`
	Path string `json:"path"`
}

// DefaultTrees are what an engine writes after the image: compile and JIT
// caches under /workspace, scratch in /tmp, shared memory in /dev/shm.
var DefaultTrees = []Tree{
	{Name: "workspace", Path: "/workspace"},
	{Name: "tmp", Path: "/tmp"},
	{Name: "shm", Path: "/dev/shm"},
}

// Config is the agent's whole input, written by the host into the
// read-only control directory.
type Config struct {
	Mode string `json:"mode"`
	// Control is the unix socket the host drives the agent through.
	Control string `json:"control"`
	// Inference, when set, is a unix socket bridged to EngineAddr: the
	// container has no network, the engine listens on its loopback.
	Inference  string `json:"inference,omitempty"`
	EngineAddr string `json:"engine_addr,omitempty"`
	CRIU       string `json:"criu"`
	// CUDACheckpoint is NVIDIA's cuda-checkpoint.
	CUDACheckpoint string `json:"cuda_checkpoint"`
	Trees          []Tree `json:"trees"`
	// Watch are paths the freeze step looks for among the tree's open
	// files and mappings, so the host knows which mounts a restore needs.
	Watch []string `json:"watch,omitempty"`
	// IPC is the profile's resolved ipc: native, interpose or auto. Under
	// interpose or auto the snapshot agent may start the engine under
	// cuInterpose (docs/interpose.md); IPCCoordinator runs its checkpoints.
	IPC            string `json:"ipc,omitempty"`
	IPCCoordinator string `json:"ipc_coordinator,omitempty"`
	// Hostname, when set, is the host name the agent gives the UTS
	// namespace of its own it was started in, before anything runs: a
	// restore in place, where the container's host name is the pod's (or,
	// with hostNetwork, the node's) and the engine ran under a receiver's.
	Hostname string `json:"hostname,omitempty"`

	// Snapshot: the engine to start and where results go.
	Argv []string `json:"argv,omitempty"`
	Env  []string `json:"env,omitempty"`
	Dir  string   `json:"dir,omitempty"`
	Log  string   `json:"log,omitempty"`
	Out  string   `json:"out,omitempty"`

	// Restore: where the snapshot is and which tree to expect.
	Images string `json:"images,omitempty"`
	FS     string `json:"fs,omitempty"`
	Work   string `json:"work,omitempty"`
	Root   int    `json:"root,omitempty"`
	MaxID  int    `json:"max_id,omitempty"`
}

// ReadConfig loads and checks a config file.
func ReadConfig(name string) (Config, error) {
	var c Config
	raw, err := os.ReadFile(name)
	if err != nil {
		return c, err
	}
	if err := json.Unmarshal(raw, &c); err != nil {
		return c, fmt.Errorf("%s: %w", name, err)
	}
	return c, c.Validate()
}

// Write stores the config.
func (c Config) Write(name string) error {
	if err := c.Validate(); err != nil {
		return err
	}
	raw, err := json.MarshalIndent(c, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(name, append(raw, '\n'), 0o644)
}

func abs(paths ...string) bool {
	for _, p := range paths {
		if !path.IsAbs(p) || path.Clean(p) != p {
			return false
		}
	}
	return true
}

// Validate checks the fields the mode needs.
func (c Config) Validate() error {
	if !abs(c.Control) || c.CRIU == "" || len(c.Trees) == 0 {
		return errors.New("agent: control socket, criu and trees are required")
	}
	if c.Inference != "" && (!abs(c.Inference) || c.EngineAddr == "") {
		return errors.New("agent: inference socket needs an absolute path and an engine address")
	}
	for _, t := range c.Trees {
		if t.Name == "" || !abs(t.Path) {
			return fmt.Errorf("agent: invalid tree %+v", t)
		}
	}
	if c.IPC != "" && c.IPC != interpose.Native && !abs(c.IPCCoordinator) {
		return fmt.Errorf("agent: ipc %s needs the coordinator's path", c.IPC)
	}
	switch c.Mode {
	case ModeSnapshot:
		if len(c.Argv) == 0 || !abs(c.Log, c.Out) {
			return errors.New("agent: snapshot needs argv, log and out")
		}
	case ModeRestore:
		if !abs(c.Images, c.FS, c.Work) || c.Root < criu.PIDFloor || c.MaxID < c.Root {
			return errors.New("agent: restore needs images, fs, work, root and max_id")
		}
	default:
		return fmt.Errorf("agent: unknown mode %q", c.Mode)
	}
	return nil
}

// Results of the steps, shared with the host.
type (
	// Status answers "status" at any time.
	Status struct {
		Mode      string `json:"mode"`
		Step      string `json:"step"`
		EnginePID int    `json:"engine_pid"`
		Running   bool   `json:"running"`
		Exit      string `json:"exit,omitempty"`
		Failed    string `json:"failed,omitempty"`
	}
	// CUDAResult lists the processes whose CUDA state moved.
	CUDAResult struct {
		Owners  []int              `json:"owners"`
		Seconds float64            `json:"seconds"`
		Phases  map[string]float64 `json:"phases,omitempty"`
	}
	// FreezeResult is the stopped tree and how many of its open files and
	// mappings sit under each watched path.
	FreezeResult struct {
		Tree       criu.Tree      `json:"tree"`
		References map[string]int `json:"references"`
	}
	// CUDARestoreArgs drive the restore's cuda step.
	CUDARestoreArgs struct {
		Owners     []int  `json:"owners"`
		Parallel   []int  `json:"parallel,omitempty"`
		Concurrent bool   `json:"concurrent,omitempty"`
		DeviceMap  string `json:"device_map"`
	}
	// DumpResult reports the CRIU dump.
	DumpResult struct {
		Seconds float64 `json:"seconds"`
		Version string  `json:"criu_version"`
	}
	// FSTree is one archived or installed tree.
	FSTree struct {
		Name    string `json:"name"`
		Path    string `json:"path"`
		Entries int    `json:"entries"`
		Bytes   int64  `json:"bytes"`
	}
	// FSResult reports the trees.
	FSResult struct {
		Trees   []FSTree `json:"trees"`
		Seconds float64  `json:"seconds"`
	}
	// IPCResult reports the cross-process memory step: whether the engine
	// runs under cuInterpose and why.
	IPCResult struct {
		Interposed bool   `json:"interposed"`
		Reason     string `json:"reason,omitempty"`
		// Participants are the processes the coordinator prepared.
		Participants []int   `json:"participants,omitempty"`
		Seconds      float64 `json:"seconds"`
	}
	// IPCRestoreArgs drive the restore's ipc step: the processes the
	// coordinator prepared.
	IPCRestoreArgs struct {
		Processes []int `json:"processes"`
	}
	// RestoreResult reports the CRIU restore.
	RestoreResult struct {
		Tree    criu.Tree `json:"tree"`
		Seconds float64   `json:"seconds"`
	}
)

// Steps are the agent's fixed sequences. A step may be skipped, never
// repeated or run out of order; "status" is allowed at any time.
var Steps = map[string][]string{
	ModeSnapshot: {"ipc", "cuda", "freeze", "dump", "fs", "finish"},
	ModeRestore:  {"install-fs", "criu", "thaw", "cuda", "ipc"},
}
