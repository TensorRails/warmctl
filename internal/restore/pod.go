package restore

// A restore in place, inside a Kubernetes pod (docs/kubernetes.md).
// Kubernetes made the receiver container, so this process checks that it
// is one a restore works in, bind-mounts what Docker would have mounted and
// runs the agent as its child. The agent's steps are the same.

import (
	"context"
	"errors"
	"fmt"
	"net"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	"github.com/TensorRails/warmctl/v2/internal/agent"
	"github.com/TensorRails/warmctl/v2/internal/criu"
	"github.com/TensorRails/warmctl/v2/internal/docker"
	"github.com/TensorRails/warmctl/v2/internal/host"
	"github.com/TensorRails/warmctl/v2/internal/interpose"
	"github.com/TensorRails/warmctl/v2/internal/manifest"
	"github.com/TensorRails/warmctl/v2/internal/pyext"
	"github.com/TensorRails/warmctl/v2/internal/run"
	"github.com/TensorRails/warmctl/v2/internal/runtimefs"
	"github.com/TensorRails/warmctl/v2/internal/snapshot"
)

// Pod defaults.
const (
	// DefaultRunDir holds a pod restore's agent config, sockets and CRIU
	// work: a volume of the pod's own.
	DefaultRunDir = "/run/warmctl"
	// DefaultPodListen is the gateway in a pod. The restored engine has
	// 127.0.0.1:8000 of the pod's network namespace.
	DefaultPodListen = ":8080"
)

// inits reap orphans, as the restored tree's root is one: CRIU restores it
// detached and it is reparented to PID 1.
var inits = []string{"tini", "docker-init", "dumb-init", "catatonit", "pause"}

// capBits are the capability numbers of docker.CheckpointCaps.
var capBits = map[string]uint{
	"DAC_READ_SEARCH": 2, "NET_ADMIN": 12, "SYS_PTRACE": 19, "SYS_ADMIN": 21, "SYS_RESOURCE": 24, "SYS_TIME": 25,
	"CHECKPOINT_RESTORE": 40,
}

// checkInit refuses to run as PID 1, where nothing would reap the restored
// tree, or in the host's PID namespace, where the trees are the host's.
func checkInit(comm string, self int) error {
	if self == 1 {
		return errors.New("warmctl is PID 1: run it under tini (the pod image's entrypoint), which reaps the restored engine")
	}
	if self >= criu.PIDFloor {
		return fmt.Errorf("warmctl is PID %d, in the snapshot's PID range: the pod needs its own PID namespace", self)
	}
	comm = strings.TrimSpace(comm)
	for _, name := range inits {
		if comm == name || strings.HasPrefix(comm, name+"-") {
			return nil
		}
	}
	return fmt.Errorf("PID 1 is %q, not a container init (%s): an in-place restore runs in a pod's "+
		"container, without hostPID", comm, strings.Join(inits, ", "))
}

// checkStatus reads /proc/self/status: the checkpoint capabilities and no
// seccomp filter, as a receiver container has.
func checkStatus(status string) error {
	var eff uint64
	seccomp := -1
	for _, line := range strings.Split(status, "\n") {
		key, value, _ := strings.Cut(line, ":")
		value = strings.TrimSpace(value)
		switch key {
		case "CapEff":
			eff, _ = strconv.ParseUint(value, 16, 64)
		case "Seccomp":
			seccomp, _ = strconv.Atoi(value)
		}
	}
	var missing []string
	for _, name := range docker.CheckpointCaps {
		if eff&(1<<capBits[name]) == 0 {
			missing = append(missing, name)
		}
	}
	if len(missing) > 0 {
		return fmt.Errorf("the pod lacks %s: CRIU and the mounts need privileged: true", strings.Join(missing, ", "))
	}
	if seccomp != 0 {
		return fmt.Errorf("seccomp mode %d: the pod needs seccompProfile Unconfined", seccomp)
	}
	return nil
}

// fsInfo is what statfs says about a tree's file system.
type fsInfo struct {
	Free     int64
	ReadOnly bool
}

// checkTree checks that a tree's file system is writable and takes its
// archive (need bytes).
func checkTree(t agent.Tree, info fsInfo, need int64) error {
	switch {
	case info.ReadOnly:
		return fmt.Errorf("%s is read-only: with readOnlyRootFilesystem, mount an emptyDir there", t.Path)
	case info.Free < need:
		return fmt.Errorf("%s has %s free, the snapshot's tree needs %s", t.Path, gib(info.Free), gib(need))
	}
	return nil
}

func gib(n int64) string { return fmt.Sprintf("%.2f GiB", float64(n)/(1<<30)) }

// treeBytes is the size of a tree's archive in the snapshot.
func treeBytes(m manifest.Manifest, name string) int64 {
	for _, f := range m.Files {
		if f.Path == "fs/"+name+".tar" {
			return f.Size
		}
	}
	return 0
}

// checkImage compares the image the pod image was built FROM with the one
// the snapshot's engine ran in: every library the restored processes map
// comes from it.
func checkImage(base, snapshot string) error {
	if base == snapshot || (digest(base) != "" && digest(base) == digest(snapshot)) {
		return nil
	}
	return fmt.Errorf("the snapshot's engine ran in %s, this pod's image is built FROM %s: "+
		"build it with --build-arg BASE=%s", snapshot, base, snapshot)
}

func digest(ref string) string {
	if strings.HasPrefix(ref, "sha256:") {
		return ref
	}
	if _, d, ok := strings.Cut(ref, "@"); ok && strings.HasPrefix(d, "sha256:") {
		return d
	}
	return ""
}

// memoryLimit reads a cgroup v2 memory.max: bytes, or false for "max".
func memoryLimit(raw string) (int64, bool) {
	n, err := strconv.ParseInt(strings.TrimSpace(raw), 10, 64)
	return n, err == nil
}

// hostUTS is /proc/self/ns/uts in the host's own UTS namespace, whose
// inode the kernel fixes (PROC_UTS_INIT_INO).
const hostUTS = "uts:[4026531838]"

// checkUTS refuses the node's UTS namespace: containerd puts a hostNetwork
// pod there, and its network namespace is the node's too, where the
// restored engine's 127.0.0.1:8000 would be the node's loopback.
func checkUTS(link string) error {
	if link == hostUTS {
		return errors.New("this pod shares the node's UTS and network namespaces (hostNetwork): " +
			"the restored engine would listen on the node's loopback; run it without hostNetwork")
	}
	return nil
}

// checkListen refuses an address that takes the restored engine's own
// socket, its port on loopback or on every address. The engine's port on
// the pod's IP is free: KServe sends probes and traffic there.
func checkListen(listen string) error {
	host, port, err := net.SplitHostPort(listen)
	if err != nil {
		return fmt.Errorf("--listen %q: %w", listen, err)
	}
	if _, engine, _ := net.SplitHostPort(snapshot.EngineAddr); port == engine {
		ip := net.ParseIP(host)
		if host == "" || host == "localhost" || ip == nil || ip.IsLoopback() || ip.IsUnspecified() {
			return fmt.Errorf("--listen %s: port %s is the restored engine's on the pod's loopback; use %s, "+
				"or the pod's IP with port %s", listen, port, DefaultPodListen, port)
		}
	}
	return nil
}

// PodGPUs is the --gpus list of the GPUs the NVIDIA device plugin gave this
// pod, from NVIDIA_VISIBLE_DEVICES, or "" when it names none: then the
// restore takes the first free GPUs of the snapshot's model, as on a host.
func PodGPUs(ctx context.Context) (string, error) {
	value := strings.TrimSpace(os.Getenv("NVIDIA_VISIBLE_DEVICES"))
	var listed []string
	switch {
	case value == "" || value == "all" || value == "void" || value == "none":
		return "", nil
	case strings.HasPrefix(value, "/"):
		// The device plugin's volume-mounts strategy: one file per device.
		entries, err := os.ReadDir(value)
		if err != nil {
			return "", fmt.Errorf("NVIDIA_VISIBLE_DEVICES: %w", err)
		}
		for _, e := range entries {
			listed = append(listed, e.Name())
		}
	default:
		listed = strings.Split(value, ",")
	}
	inventory, err := host.Inventory(ctx)
	if err != nil {
		return "", err
	}
	return VisibleGPUs(listed, inventory)
}

// VisibleGPUs turns device plugin names (UUIDs or indices) into indices.
func VisibleGPUs(listed []string, inventory []host.GPU) (string, error) {
	var out []string
	for _, item := range listed {
		item = strings.TrimSpace(item)
		if item == "" {
			continue
		}
		if _, err := strconv.Atoi(item); err == nil {
			out = append(out, item)
			continue
		}
		found := false
		for _, g := range inventory {
			if g.UUID == item {
				out, found = append(out, strconv.Itoa(g.Index)), true
				break
			}
		}
		if !found {
			return "", fmt.Errorf("the device plugin gave %s, which nvidia-smi does not list (MIG instances "+
				"are not supported)", item)
		}
	}
	return strings.Join(out, ","), nil
}

// podEnv is the agent's environment in a pod: the container's, with the
// snapshot's launch environment over it, as a receiver container has.
func podEnv(base []string, launch map[string]string) []string {
	out := make([]string, 0, len(base)+len(launch))
	for _, kv := range base {
		if key, _, _ := strings.Cut(kv, "="); !hasKey(launch, key) {
			out = append(out, kv)
		}
	}
	keys := make([]string, 0, len(launch))
	for key := range launch {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	for _, key := range keys {
		out = append(out, key+"="+launch[key])
	}
	return out
}

func hasKey(m map[string]string, key string) bool {
	_, ok := m[key]
	return ok
}

// engineMounts are what the restored engine finds where the donor had it:
// the snapshot's Python package, its cuInterpose build (mapped by every
// interposed process) and the model directory, when there is one to mount.
func (r *restore) engineMounts(model string) []docker.Mount {
	mounts := []docker.Mount{{Source: filepath.Join(r.snapshot, "adapter"), Target: pyext.GuestPath, ReadOnly: true}}
	if r.interposed() {
		mounts = append(mounts, docker.Mount{Source: filepath.Join(r.snapshot, "ipc"), Target: interpose.GuestDir,
			ReadOnly: true})
	}
	if model != "" && model != r.m.Profile.Model.MountPath() {
		mounts = append(mounts, docker.Mount{Source: model, Target: r.m.Profile.Model.MountPath(), ReadOnly: true})
	}
	return mounts
}

// podModel is the model directory to mount in a pod: the operator's, or
// none when the pod mounts the model where the profile wants it.
func podModel(given string, m manifest.Manifest) (string, error) {
	if given != "" || !m.Process.NeedsModel {
		return given, nil
	}
	mount := m.Profile.Model.MountPath()
	if entries, err := os.ReadDir(mount); err != nil || len(entries) == 0 {
		return "", fmt.Errorf("the snapshot's processes keep files of %s open: mount the model there (%s "+
			"on the donor host) or pass --model-dir", mount, m.Launch.ModelDir)
	}
	return "", nil
}

// checkPod is the pod half of the preflight: what Docker would have given
// a receiver container, Kubernetes must have given this one.
func (r *restore) checkPod() error {
	comm, _ := os.ReadFile("/proc/1/comm")
	if err := checkInit(string(comm), os.Getpid()); err != nil {
		return err
	}
	uts, err := os.Readlink("/proc/self/ns/uts")
	if err != nil {
		return err
	}
	if err := checkUTS(uts); err != nil {
		return err
	}
	status, err := os.ReadFile("/proc/self/status")
	if err != nil {
		return err
	}
	if err := checkStatus(string(status)); err != nil {
		return err
	}
	if err := writable("/proc/sys/kernel/ns_last_pid"); err != nil {
		return fmt.Errorf("/proc/sys is read-only (%v): the pod needs privileged: true, so the agent can move "+
			"the PID allocator", err)
	}
	model, err := podModel(r.o.ModelDir, r.m)
	if err != nil {
		return err
	}
	for _, m := range r.engineMounts(model) {
		if info, err := os.Stat(m.Target); err != nil || !info.IsDir() {
			return fmt.Errorf("no mount point %s in this image: build it from deploy/kubernetes/Dockerfile", m.Target)
		}
	}
	if err := r.checkTrees(); err != nil {
		return err
	}
	r.warnPod()
	return r.checkBase()
}

// checkTrees checks the trees warmctl does not mount itself: /workspace,
// in the container's own layer or on a volume.
func (r *restore) checkTrees() error {
	for _, t := range agent.DefaultTrees {
		if _, ours := r.tmpfs(t.Path); ours {
			continue
		}
		info, err := statTree(t.Path)
		if err == nil {
			err = writable(t.Path)
		}
		if err != nil {
			return fmt.Errorf("tree %s: %w", t.Path, err)
		}
		if err := checkTree(t, info, treeBytes(r.m, t.Name)); err != nil {
			return err
		}
	}
	return nil
}

// tmpfs is the size of the tmpfs a receiver container has at path: /tmp
// and /dev/shm, which a pod's defaults (a 64 MiB /dev/shm) do not match.
func (r *restore) tmpfs(path string) (int64, bool) {
	switch path {
	case "/tmp":
		return docker.DefaultTmpSize, true
	case "/dev/shm":
		shm, _ := parseSize(r.m.Profile.Resources.SHM)
		return max(shm, 64<<20), true
	}
	return 0, false
}

func (r *restore) checkBase() error {
	rt, ok, err := ReadRuntime(RuntimeFile)
	if err != nil {
		return err
	}
	if !ok {
		r.p.Printf("warning: no %s: this is not the warmctl runtime image, nothing says what it was built FROM",
			RuntimeFile)
		return nil
	}
	return checkImage(rt.Base, r.m.Launch.Image)
}

// warnPod reports what may still restore but differs from a receiver
// container.
func (r *restore) warnPod() {
	want, _ := parseSize(r.m.Profile.Resources.Memory)
	raw, _ := os.ReadFile("/sys/fs/cgroup/memory.max")
	if limit, ok := memoryLimit(string(raw)); ok && limit < want {
		r.p.Printf("warning: the pod's memory limit is %s, the engine ran with %s", gib(limit), gib(want))
	}
}

// claimPod makes this container what a receiver container is: fresh
// tmpfs at /tmp and /dev/shm of the receiver's sizes, and nothing in
// /workspace or the run directory but what an earlier container of this
// pod left there (a restarted container keeps its layer and volumes). The
// host name is the agent's to set, in a UTS namespace of its own.
func (r *restore) claimPod() error {
	r.dir = r.o.RunDir
	if err := claimDir(r.dir); err != nil {
		return err
	}
	for _, t := range agent.DefaultTrees {
		size, ours := r.tmpfs(t.Path)
		if !ours {
			if err := claimDir(t.Path); err != nil {
				return err
			}
			continue
		}
		if err := mountTmpfs(t.Path, size, t.Path == "/tmp"); err != nil {
			return fmt.Errorf("tmpfs at %s: %w", t.Path, err)
		}
		r.mounts = append(r.mounts, t.Path)
	}
	for _, d := range []string{"work", "io"} {
		if err := os.MkdirAll(filepath.Join(r.dir, d), 0o755); err != nil {
			return err
		}
	}
	return nil
}

// claimDir makes dir this restore's: it may be empty or hold what an
// earlier restore left under its mark, which goes; anything else may be
// the operator's data on a volume and is refused.
func claimDir(dir string) error {
	entries, err := os.ReadDir(dir)
	if errors.Is(err, os.ErrNotExist) {
		entries, err = nil, os.MkdirAll(dir, 0o755)
	}
	if err != nil {
		return err
	}
	marked := false
	for _, e := range entries {
		marked = marked || e.Name() == runtimefs.Mark
	}
	if len(entries) > 0 && !marked {
		return fmt.Errorf("%s holds files no restore left there (no %s): mount an emptyDir at %s, "+
			"or leave it to the container's layer", dir, runtimefs.Mark, dir)
	}
	for _, e := range entries {
		if err := os.RemoveAll(filepath.Join(dir, e.Name())); err != nil {
			return err
		}
	}
	return os.WriteFile(filepath.Join(dir, runtimefs.Mark), []byte("a warmctl restore's directory\n"), 0o644)
}

// startPod mounts what the restored engine needs and starts the agent as
// this process's child.
func (r *restore) startPod(ctx context.Context) error {
	model, err := podModel(r.o.ModelDir, r.m)
	if err != nil {
		return err
	}
	drivers, err := r.driverMounts()
	if err != nil {
		return err
	}
	if len(drivers) == 0 && !r.o.CPUTest {
		r.warnDriver()
	}
	for _, m := range append(r.engineMounts(model), drivers...) {
		if err := mountPoint(m.Source, m.Target); err != nil {
			return err
		}
		if err := bindMount(m.Source, m.Target, m.ReadOnly); err != nil {
			return fmt.Errorf("mount %s at %s: %w", m.Source, m.Target, err)
		}
		r.mounts = append(r.mounts, m.Target)
	}
	self, err := os.Executable()
	if err != nil {
		return err
	}
	r.c, err = run.Spawn(ctx, run.Child{
		Name: r.o.ID, Argv: []string{self, "agent", "--config", filepath.Join(r.dir, "work", "agent.json")},
		Env: podEnv(os.Environ(), r.m.Launch.Env), IO: filepath.Join(r.dir, "io"), Log: snapshot.EngineLog,
		AgentLog: filepath.Join(r.dir, "agent.log"), Output: os.Stderr, Ranks: r.m.Launch.Ranks, NewUTS: true,
	})
	if err != nil {
		return err
	}
	r.p.Printf("receiver agent started in this pod")
	return nil
}

// warnDriver names the driver libraries the dump maps that this pod lacks
// under the same driver: the NVIDIA runtime put them elsewhere (a CDI spec
// of another distribution), and CRIU will not map them back.
func (r *restore) warnDriver() {
	for _, p := range r.m.Hardware.DriverFiles {
		if _, err := os.Stat(p); err != nil {
			r.p.Printf("warning: %s, which the restored processes map, is missing in this pod", p)
		}
	}
}

// mountPoint makes the target of a file's bind mount, a driver library
// under another release's name, as Docker would.
func mountPoint(source, target string) error {
	info, err := os.Stat(source)
	if err != nil || info.IsDir() {
		return err
	}
	if _, err := os.Stat(target); !errors.Is(err, os.ErrNotExist) {
		return err
	}
	return os.WriteFile(target, nil, 0o644)
}

// unmountPod undoes startPod's mounts, innermost first.
func (r *restore) unmountPod() {
	for i := len(r.mounts) - 1; i >= 0; i-- {
		_ = unmount(r.mounts[i])
	}
	r.mounts = nil
}
