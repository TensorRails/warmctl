package docker

import (
	"errors"
	"fmt"
	"path"
	"regexp"
	"sort"
	"strings"
)

// Spec is a container warmctl runs: the donor of a snapshot or the
// receiver of a restore.
type Spec struct {
	Name       string
	Image      string
	Entrypoint []string
	Env        map[string]string
	Labels     map[string]string
	WorkingDir string
	Mounts     []Mount
	// GPUs are device UUIDs in rank order.
	GPUs   []string
	Memory int64
	SHM    int64
	CPUs   int
	PIDs   int64
	// TmpSize bounds the /tmp tmpfs.
	TmpSize int64
	// Checkpoint grants what CRIU and cuda-checkpoint need inside the
	// container: root, the checkpoint capabilities, no seccomp or AppArmor
	// profile and a writable /proc/sys for the PID allocator.
	Checkpoint bool
}

// Mount is a bind mount.
type Mount struct {
	Source   string
	Target   string
	ReadOnly bool
}

// Hostname is every warmctl container's host name: a restored engine finds
// the one it was snapshotted under.
const Hostname = "warmctl"

// DefaultTmpSize bounds a container's /tmp tmpfs.
const DefaultTmpSize = 1 << 30

var (
	namePattern = regexp.MustCompile(`^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}$`)
	envPattern  = regexp.MustCompile(`^[A-Za-z_][A-Za-z0-9_]*$`)
)

// CheckpointCaps are the capabilities of a checkpoint container. CRIU's
// link remaps use linkat(AT_EMPTY_PATH) on unlinked inodes, which needs
// DAC_READ_SEARCH even as root.
var CheckpointCaps = []string{
	"SYS_ADMIN", "SYS_PTRACE", "CHECKPOINT_RESTORE", "SYS_RESOURCE", "NET_ADMIN", "SYS_TIME", "DAC_READ_SEARCH",
}

type bindMount struct {
	Type     string
	Source   string
	Target   string
	ReadOnly bool
}

type deviceRequest struct {
	Driver       string
	DeviceIDs    []string
	Capabilities [][]string
}

type logConfig struct {
	Type   string
	Config map[string]string
}

type hostConfig struct {
	Mounts         []bindMount
	DeviceRequests []deviceRequest `json:",omitempty"`
	NetworkMode    string
	IpcMode        string
	Init           bool
	ReadonlyRootfs bool
	Memory         int64
	MemorySwap     int64
	NanoCpus       int64
	PidsLimit      int64
	ShmSize        int64
	Tmpfs          map[string]string
	CapAdd         []string  `json:",omitempty"`
	SecurityOpt    []string  `json:",omitempty"`
	ReadonlyPaths  *[]string `json:",omitempty"`
	LogConfig      logConfig
}

type createBody struct {
	Image      string
	User       string `json:",omitempty"`
	Entrypoint []string
	Cmd        []string
	Env        []string
	Labels     map[string]string
	Hostname   string
	WorkingDir string
	HostConfig hostConfig
}

func (s Spec) validate() error {
	if !namePattern.MatchString(s.Name) {
		return fmt.Errorf("docker: invalid container name %q", s.Name)
	}
	if s.Image == "" || len(s.Entrypoint) == 0 || !path.IsAbs(s.Entrypoint[0]) {
		return errors.New("docker: image and an absolute entrypoint are required")
	}
	if s.Memory < 64<<20 || s.CPUs < 1 || s.PIDs < 16 || s.SHM < 1<<20 {
		return errors.New("docker: memory, cpus, pids and shm limits are required")
	}
	for key, value := range s.Env {
		if !envPattern.MatchString(key) || strings.ContainsRune(value, 0) {
			return fmt.Errorf("docker: invalid environment entry %q", key)
		}
	}
	targets := map[string]bool{}
	for _, m := range s.Mounts {
		if !path.IsAbs(m.Source) || !path.IsAbs(m.Target) || path.Clean(m.Target) != m.Target || m.Target == "/" {
			return fmt.Errorf("docker: invalid mount %s -> %s", m.Source, m.Target)
		}
		if targets[m.Target] {
			return fmt.Errorf("docker: two mounts at %s", m.Target)
		}
		targets[m.Target] = true
	}
	return nil
}

func render(s Spec) (createBody, error) {
	if err := s.validate(); err != nil {
		return createBody{}, err
	}
	b := createBody{
		Image:      s.Image,
		Entrypoint: append([]string(nil), s.Entrypoint...),
		Cmd:        []string{},
		Labels:     map[string]string{ManagedLabel: "true"},
		Hostname:   Hostname,
		WorkingDir: s.WorkingDir,
	}
	if b.WorkingDir == "" {
		b.WorkingDir = "/"
	}
	for k, v := range s.Labels {
		b.Labels[k] = v
	}
	keys := make([]string, 0, len(s.Env))
	for k := range s.Env {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		b.Env = append(b.Env, k+"="+s.Env[k])
	}
	h := &b.HostConfig
	for _, m := range s.Mounts {
		h.Mounts = append(h.Mounts, bindMount{Type: "bind", Source: m.Source, Target: m.Target, ReadOnly: m.ReadOnly})
	}
	// No network: the engine listens on loopback in its own namespace and
	// the agent bridges a unix socket to it. A snapshot must not hold a
	// socket bound to a host address.
	h.NetworkMode = "none"
	h.IpcMode = "private"
	// docker-init is PID 1 and reaps orphans; the agent is its only child.
	h.Init = true
	// Everything the engine writes lands in the trees the snapshot carries
	// (/workspace, /tmp, /dev/shm); a write anywhere else fails at bake
	// time instead of going missing at restore.
	h.ReadonlyRootfs = true
	h.Memory, h.MemorySwap = s.Memory, s.Memory
	h.NanoCpus = int64(s.CPUs) * 1e9
	h.PidsLimit = s.PIDs
	h.ShmSize = s.SHM
	tmp := s.TmpSize
	if tmp == 0 {
		tmp = DefaultTmpSize
	}
	// exec: JIT compilers load what they build in a temporary directory.
	h.Tmpfs = map[string]string{"/tmp": fmt.Sprintf("rw,nosuid,nodev,exec,size=%d", tmp)}
	h.LogConfig = logConfig{Type: "json-file", Config: map[string]string{"max-size": "16m", "max-file": "2"}}
	if s.Checkpoint {
		b.User = "0:0"
		h.CapAdd = append([]string(nil), CheckpointCaps...)
		h.SecurityOpt = []string{"seccomp=unconfined", "apparmor=unconfined"}
		// An explicit empty list lifts Docker's read-only /proc/sys, so the
		// PID allocator (ns_last_pid) can be moved.
		empty := []string{}
		h.ReadonlyPaths = &empty
	}
	if len(s.GPUs) > 0 {
		h.DeviceRequests = []deviceRequest{{
			Driver: "nvidia", DeviceIDs: append([]string(nil), s.GPUs...), Capabilities: [][]string{{"gpu"}},
		}}
	}
	return b, nil
}
