package cli

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"path"
	"path/filepath"
	"strconv"
	"strings"
	"text/tabwriter"
	"time"

	"github.com/TensorRails/warmctl/v2/internal/config"
	"github.com/TensorRails/warmctl/v2/internal/docker"
	"github.com/TensorRails/warmctl/v2/internal/gateway"
	"github.com/TensorRails/warmctl/v2/internal/host"
	"github.com/TensorRails/warmctl/v2/internal/manifest"
	"github.com/TensorRails/warmctl/v2/internal/oci"
	"github.com/TensorRails/warmctl/v2/internal/restore"
	"github.com/TensorRails/warmctl/v2/internal/run"
	"github.com/TensorRails/warmctl/v2/internal/snapshot"
	"github.com/TensorRails/warmctl/v2/internal/store"
)

// DefaultState is where warmctl keeps its store, runs and instances.
const DefaultState = "/var/lib/warmctl"

type list []string

func (l *list) String() string     { return strings.Join(*l, ",") }
func (l *list) Set(v string) error { *l = append(*l, v); return nil }

type common struct {
	state, dockerSock, progress, criu, helper, scratch, cudaExec, cudaBackend string
	noVerify, cpuTest                                                         bool
	threads                                                                   int
}

func (c *common) register(fs *flag.FlagSet) {
	fs.StringVar(&c.state, "state-dir", DefaultState, "state directory")
	fs.StringVar(&c.dockerSock, "docker", docker.DefaultSocket, "Docker socket")
	fs.StringVar(&c.progress, "progress", "plain", "progress format: plain or jsonl")
	fs.StringVar(&c.criu, "criu", "", "host CRIU binary to use instead of the image's")
	fs.StringVar(&c.helper, "cuda-checkpoint", "", "host cuda-checkpoint to use instead of the image's")
	fs.StringVar(&c.cudaBackend, "cuda-backend", "auto",
		"how CUDA state moves: driver (libcuda's checkpoint API), cli (cuda-checkpoint), or auto (driver when usable)")
	fs.StringVar(&c.scratch, "scratch", "", "accepted for compatibility")
	fs.StringVar(&c.cudaExec, "cuda-execution", "auto",
		"CUDA restore: auto (everything concurrent when no collective links the processes), serial, "+
			"parallel (only the workers concurrent) or parallel-all")
	fs.BoolVar(&c.noVerify, "no-verify", false, "canary only warns")
	fs.IntVar(&c.threads, "threads", 0, "pool copy threads per rank (default 4)")
	fs.BoolVar(&c.cpuTest, "test-cpu", false, "end-to-end test on a CPU host: no GPU devices, no io_uring check")
}

// parse accepts the positional argument before, after or between the
// flags: `up NAME --plain-http` or `up --mode in-place NAME`.
func parse(fs *flag.FlagSet, args []string) (string, error) {
	var positional string
	if len(args) > 0 && !strings.HasPrefix(args[0], "-") {
		positional, args = args[0], args[1:]
	}
	if err := fs.Parse(args); err != nil {
		return "", err
	}
	if positional == "" && fs.NArg() > 0 {
		positional = fs.Arg(0)
		if err := fs.Parse(fs.Args()[1:]); err != nil {
			return "", err
		}
	}
	if fs.NArg() > 0 {
		return "", fmt.Errorf("unexpected arguments %v", fs.Args())
	}
	return positional, nil
}

func runBuild(ctx context.Context, args []string, stdout, stderr io.Writer, version string) error {
	fs := flag.NewFlagSet("build", flag.ContinueOnError)
	fs.SetOutput(stderr)
	var c common
	c.register(fs)
	var o snapshot.Options
	var sets, dirs list
	fs.StringVar(&o.Profile, "profile", "", "profile id or YAML file")
	fs.Var(&dirs, "profile-dir", "extra directory of profile YAML files (repeatable)")
	fs.StringVar(&o.ModelDir, "model-dir", "", "model directory on this host")
	fs.StringVar(&o.GPUs, "gpus", "", "GPU indices, e.g. 0-7")
	fs.StringVar(&o.Output, "output", "", "snapshot directory to create")
	fs.StringVar(&o.ID, "id", "", "run id (default: the output's name)")
	fs.Var(&sets, "set", "profile override vllm.<flag>=<value>, env.<NAME>=<value> or ipc=<mode> (repeatable)")
	fs.Int64Var(&o.PartBytes, "part-bytes", 1<<30, "size of a weights part")
	fs.DurationVar(&o.ReadyTimeout, "ready-timeout", 90*time.Minute, "how long the engine may take to load")
	fs.StringVar(&o.Cuinterpose, "cuinterpose", "", "cuInterpose build directory (default: cuinterpose/ next to warmctl)")
	if _, err := parse(fs, args); err != nil {
		return err
	}
	if o.Profile == "" || o.ModelDir == "" || o.Output == "" {
		return errors.New("build needs --profile, --model-dir and --output")
	}
	if o.ID == "" {
		o.ID = filepath.Base(o.Output)
	}
	o.Output, _ = filepath.Abs(o.Output)
	o.ModelDir, _ = filepath.Abs(o.ModelDir)
	o.Set, o.ProfileDirs, o.StateDir, o.DockerSocket = sets, dirs, c.state, c.dockerSock
	o.CRIU, o.CUDACheckpoint, o.NoVerify, o.Threads, o.Version = c.criu, c.helper, c.noVerify, c.threads, version
	o.CPUTest, o.CUDABackend = c.cpuTest, c.cudaBackend
	o.Progress = run.NewProgress(stdout, c.progress)
	m, err := snapshot.Run(ctx, o)
	if err != nil {
		o.Progress.Printf("snapshot failed: %v", err)
		return err
	}
	_ = recordSnapshot(c.state, o.Output)
	o.Progress.Printf("snapshot completed: %s", snapshot.Describe(m))
	return nil
}

func runUp(ctx context.Context, args []string, stdout, stderr io.Writer) error {
	fs := flag.NewFlagSet("up", flag.ContinueOnError)
	fs.SetOutput(stderr)
	var c common
	c.register(fs)
	var o restore.Options
	mode := registerUp(fs, &o)
	accept := fs.Bool("accept-license", false, "accept the license of a model the catalog marks gated")
	var of ociFlags
	of.register(fs)
	snap, err := parse(fs, args)
	if err != nil {
		return err
	}
	if snap == "" {
		return errors.New("up needs a snapshot directory, a NAME[:TAG] or an oci:// reference")
	}
	o.Progress = run.NewProgress(stdout, c.progress)
	if o.TLSCert != "" || o.TLSKey != "" {
		if o.TLS, err = gateway.LoadTLS(o.TLSCert, o.TLSKey); err != nil {
			return err
		}
	}
	gate, err := chooseMode(ctx, fs, &o, *mode, c.dockerSock)
	if err != nil {
		return err
	}
	if gate != nil {
		defer func() { _ = gate.Close() }()
	}
	cfg, err := config.Load()
	if err != nil {
		return err
	}
	snap, stopPull, err := upFrom(ctx, snap, &o, c.state, *accept, of, cfg)
	if err != nil {
		return err
	}
	defer stopPull()
	o.Snapshot, _ = filepath.Abs(snap)
	if o.ID == "" {
		o.ID = filepath.Base(o.Snapshot)
	}
	o.StateDir, o.DockerSocket, o.CUDAExecution, o.NoVerify = c.state, c.dockerSock, c.cudaExec, c.noVerify
	o.CRIU, o.CUDACheckpoint, o.Threads, o.Serve, o.CPUTest = c.criu, c.helper, c.threads, true, c.cpuTest
	o.CUDABackend = c.cudaBackend
	if err := restore.Run(ctx, o); err != nil {
		o.Progress.Printf("up failed: %v", err)
		return err
	}
	o.Progress.Printf("up completed")
	return nil
}

// upFrom is the snapshot directory up restores from: snap itself, or, for a
// reference, the directory a pull fills (startPull). The engine may serve
// before the pull has synced every blob; stop, once up ends, ends a pull
// still running and waits for it to clean up.
func upFrom(ctx context.Context, snap string, o *restore.Options, state string, accept bool, of ociFlags,
	cfg config.Config) (dir string, stop func(), err error) {
	ref := reference(snap, cfg)
	if ref == "" {
		return snap, func() {}, nil
	}
	if err := acceptLicense(ctx, ref, cfg, state, accept, of.options(o.Progress)); err != nil {
		return "", nil, err
	}
	pctx, cancel := context.WithCancel(ctx)
	dir, pulling, err := startPull(pctx, ref, o, state, of.options(o.Progress), cfg)
	if err != nil || pulling == nil {
		cancel()
		return dir, func() {}, err
	}
	return dir, func() {
		cancel()
		switch p := <-pulling; {
		case p.err == nil:
		case errors.Is(p.err, context.Canceled):
			o.Progress.Printf("the pull stopped before it finished; the next up resumes it")
		default:
			o.Progress.Printf("the pull failed: %v; the snapshot is not kept", p.err)
		}
	}, nil
}

type pulled struct {
	dir string
	err error
}

// startPull pulls ref for up and returns as soon as the restore can start.
// That is when the pull has the snapshot's manifest and its directory, with
// o.Arrival set and pulling open until the pull ends; or when the pull has
// ended, or found the snapshot pulled before, with pulling nil.
func startPull(ctx context.Context, ref string, o *restore.Options, state string, po oci.Options,
	cfg config.Config) (string, <-chan pulled, error) {
	arrivals := oci.NewArrivals()
	po.Arrivals = arrivals
	done := make(chan pulled, 1)
	go func() {
		dir, err := pullForUp(ctx, ref, o, state, po, cfg)
		done <- pulled{dir, err}
	}()
	select {
	case <-arrivals.Begun():
		o.Arrival = arrivals
		return arrivals.Dir(), done, nil
	case p := <-done:
		return p.dir, nil, p.err
	}
}

func runDown(ctx context.Context, args []string, stdout, stderr io.Writer) error {
	fs := flag.NewFlagSet("down", flag.ContinueOnError)
	fs.SetOutput(stderr)
	state := fs.String("state-dir", DefaultState, "state directory")
	sock := fs.String("docker", docker.DefaultSocket, "Docker socket")
	grace := fs.Duration("grace", 30*time.Second, "how long the engine gets to stop")
	id, err := parse(fs, args)
	if err != nil || id == "" {
		return errors.Join(err, errors.New("down needs an instance id"))
	}
	name := run.ContainerName(id)
	if raw, err := os.ReadFile(filepath.Join(*state, "restores", id, "instance.json")); err == nil {
		var inst restore.Instance
		if json.Unmarshal(raw, &inst) == nil && inst.Container != "" {
			name = inst.Container
		}
	}
	dk, err := docker.New(ctx, *sock)
	if err != nil {
		return err
	}
	if err := dk.Stop(ctx, name, *grace); err != nil {
		return err
	}
	fmt.Fprintf(stdout, "stopped %s\n", name)
	return nil
}

func registerUp(fs *flag.FlagSet, o *restore.Options) *string {
	fs.StringVar(&o.ID, "id", "", "instance id (default: the snapshot's or the repository's name)")
	fs.StringVar(&o.GPUs, "gpus", "", "GPU indices (default: the first free GPUs of the snapshot's model)")
	fs.StringVar(&o.Listen, "listen", "127.0.0.1:8000", "gateway address")
	fs.StringVar(&o.TLSCert, "tls-cert", "", "serve HTTPS with this PEM certificate, read again when it changes")
	fs.StringVar(&o.TLSKey, "tls-key", "", "the PEM private key of --tls-cert")
	fs.StringVar(&o.Canary, "canary", "", "override the canary policy: bitwise, tolerance, warn or off")
	fs.StringVar(&o.ModelDir, "model-dir", "", "model directory, when the snapshot needs it")
	fs.BoolVar(&o.VerifyWeights, "verify-weights", false, "check each weights part's SHA-256 while reloading")
	fs.StringVar(&o.RunDir, "run-dir", restore.DefaultRunDir, "in place: the agent's config, sockets and CRIU work")
	fs.Bool("wait", false, "accepted for compatibility")
	fs.Duration("wait-timeout", 0, "accepted for compatibility")
	return fs.String("mode", restore.ModeAuto, "where to restore: auto (in place in the warmctl runtime image, "+
		"else through Docker), host or in-place (docs/kubernetes.md); in place the gateway defaults to "+
		restore.DefaultPodListen)
}

// chooseMode decides between a host restore and one in place. In place it
// opens the gateway before anything is pulled, so a pod's probes get 503
// rather than no answer, and takes the GPUs the device plugin gave the pod.
func chooseMode(ctx context.Context, fs *flag.FlagSet, o *restore.Options, mode, sock string) (*restore.Gate, error) {
	inPlace, reason, err := restore.Choose(ctx, mode, restore.Probe{
		Runtime: func() (restore.Runtime, bool, error) { return restore.ReadRuntime(restore.RuntimeFile) },
		Docker: func(ctx context.Context) error {
			_, err := docker.New(ctx, sock)
			return err
		},
		Kubernetes: restore.KubernetesSigns,
	})
	if err != nil || !inPlace {
		return nil, err
	}
	o.InPlace = true
	o.Progress.Printf("%s", reason)
	listenSet := false
	fs.Visit(func(f *flag.Flag) { listenSet = listenSet || f.Name == "listen" })
	if !listenSet {
		o.Listen = restore.DefaultPodListen
	}
	if o.GPUs == "" {
		if o.GPUs, err = restore.PodGPUs(ctx); err != nil {
			return nil, err
		}
	}
	// The restored engine is told paths under it (RPC replies): absolute,
	// whatever its working directory.
	o.RunDir, _ = filepath.Abs(o.RunDir)
	o.Gate, err = restore.OpenGate(o.Listen, o.TLS, o.RunDir, o.Progress)
	return o.Gate, err
}

// pullForUp pulls ref for o, naming the instance after the repository
// unless o.ID is set. The host check runs on the snapshot's manifest before
// any blob is fetched and picks o.GPUs, so the restore takes the GPUs it
// checked.
func pullForUp(ctx context.Context, ref string, o *restore.Options, state string, po oci.Options,
	cfg config.Config) (string, error) {
	if o.ID == "" {
		o.ID = repoName(ref)
	}
	po.Check = func(m manifest.Manifest) error {
		gpus, err := restore.CheckHost(ctx, m.Hardware, o.GPUs, filepath.Join(state, "locks"))
		if err == nil && o.GPUs == "" {
			o.GPUs = indices(gpus)
		}
		return err
	}
	return pullSnapshot(ctx, ref, "", state, po, po.Check, cfg)
}

// repoName is the last path element of an oci:// reference's repository:
// the instance id of `up oci://host/warmctl/glm53-flash:8xh100`.
func repoName(ref string) string {
	repo := strings.TrimPrefix(ref, oci.Scheme)
	if i := strings.Index(repo, "@"); i >= 0 {
		repo = repo[:i]
	}
	if i := strings.LastIndex(repo, ":"); i > strings.LastIndex(repo, "/") {
		repo = repo[:i]
	}
	return path.Base(repo)
}

// indices lists GPUs as --gpus takes them.
func indices(gpus []host.GPU) string {
	out := make([]string, len(gpus))
	for i, g := range gpus {
		out[i] = strconv.Itoa(g.Index)
	}
	return strings.Join(out, ",")
}

// runPs lists the instances recorded under <state>/restores with their
// container's state; stopped ones only with --all.
func runPs(ctx context.Context, args []string, stdout, stderr io.Writer) error {
	fs := flag.NewFlagSet("ps", flag.ContinueOnError)
	fs.SetOutput(stderr)
	state := fs.String("state-dir", DefaultState, "state directory")
	sock := fs.String("docker", docker.DefaultSocket, "Docker socket")
	all := fs.Bool("all", false, "also list instances whose container is not running")
	if _, err := parse(fs, args); err != nil {
		return err
	}
	files, _ := filepath.Glob(filepath.Join(*state, "restores", "*", "instance.json"))
	dk, err := docker.New(ctx, *sock)
	if err != nil {
		return err
	}
	tw := tabwriter.NewWriter(stdout, 0, 4, 2, ' ', 0)
	fmt.Fprintln(tw, "ID\tSTATUS\tLISTEN\tUP\tSNAPSHOT")
	for _, name := range files {
		var inst restore.Instance
		raw, err := os.ReadFile(name)
		if err != nil || json.Unmarshal(raw, &inst) != nil {
			continue
		}
		status := "gone"
		if st, err := dk.Inspect(ctx, inst.Container); err == nil {
			status = st.State.Status
		}
		if status != "running" && !*all {
			continue
		}
		fmt.Fprintf(tw, "%s\t%s\t%s\t%s\t%s\n", inst.ID, status, inst.Listen,
			time.Since(inst.Started).Round(time.Second), inst.Snapshot)
	}
	return tw.Flush()
}

func runVerify(ctx context.Context, args []string, stdout, stderr io.Writer) error {
	fs := flag.NewFlagSet("verify", flag.ContinueOnError)
	fs.SetOutput(stderr)
	deep := fs.Bool("deep", false, "hash every file")
	dir, err := parse(fs, args)
	if err != nil || dir == "" {
		return errors.Join(err, errors.New("verify needs a snapshot directory"))
	}
	m, err := manifest.Read(filepath.Join(dir, "manifest.json"))
	if err != nil {
		return err
	}
	for _, f := range m.Files {
		name := filepath.Join(dir, filepath.FromSlash(f.Path))
		info, err := os.Stat(name)
		if err != nil || info.Size() != f.Size {
			return fmt.Errorf("%s: missing or wrong size", f.Path)
		}
		if *deep {
			sum, _, err := store.HashFile(ctx, name)
			if err != nil || sum != f.SHA256 {
				return fmt.Errorf("%s: content differs", f.Path)
			}
		}
	}
	fmt.Fprintf(stdout, "ok %s (%d files)\n", snapshot.Describe(m), len(m.Files))
	return nil
}

// recordSnapshot remembers a published snapshot for gc.
func recordSnapshot(state, dir string) error {
	name := filepath.Join(state, "snapshots.txt")
	f, err := os.OpenFile(name, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
	if err != nil {
		return err
	}
	defer f.Close()
	_, err = fmt.Fprintln(f, dir)
	return err
}

func runGC(ctx context.Context, args []string, stdout, stderr io.Writer) error {
	fs := flag.NewFlagSet("gc", flag.ContinueOnError)
	fs.SetOutput(stderr)
	state := fs.String("state-dir", DefaultState, "state directory")
	dry := fs.Bool("dry-run", false, "only report")
	if _, err := parse(fs, args); err != nil {
		return err
	}
	keep := map[string]bool{}
	raw, _ := os.ReadFile(filepath.Join(*state, "snapshots.txt"))
	for _, dir := range strings.Fields(string(raw)) {
		m, err := manifest.Read(filepath.Join(dir, "manifest.json"))
		if err != nil {
			continue
		}
		for _, f := range m.Files {
			keep[f.SHA256] = true
		}
	}
	s := store.Store{Root: *state}
	// Pieces a pull left a week ago will not be resumed.
	if stale, _ := filepath.Glob(filepath.Join(s.Blobs(), ".assemble-*")); !*dry {
		for _, name := range stale {
			if info, err := os.Stat(name); err == nil && time.Since(info.ModTime()) > 7*24*time.Hour {
				_ = os.Remove(name)
			}
		}
	}
	all, err := s.List()
	if err != nil {
		return err
	}
	if *dry {
		fmt.Fprintf(stdout, "%d blobs, %d referenced\n", len(all), len(keep))
		return nil
	}
	freed, err := s.GC(ctx, keep)
	fmt.Fprintf(stdout, "freed %.2f GiB\n", float64(freed)/(1<<30))
	return err
}
