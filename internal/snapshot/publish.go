package snapshot

import (
	"context"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/TensorRails/warmctl/v2/internal/criu"
	"github.com/TensorRails/warmctl/v2/internal/host"
	"github.com/TensorRails/warmctl/v2/internal/interpose"
	"github.com/TensorRails/warmctl/v2/internal/manifest"
	"github.com/TensorRails/warmctl/v2/internal/run"
	"github.com/TensorRails/warmctl/v2/internal/store"
)

// source is one file to publish: where it is, where it goes in the
// snapshot, its role and, when the writer already hashed it, its digest.
type source struct {
	path, rel, role, digest string
}

// publish writes the snapshot: every file into the content-addressed store
// (a hard link when it is on the same filesystem), linked from there into
// the output directory, and manifest.json last.
func (b *bake) publish(ctx context.Context) error {
	started := time.Now()
	paths, err := driverFiles(filepath.Join(b.dir, "io", "out", "criu"), b.gpus[0].Driver, filepath.Join(b.dir, "driver"))
	if err != nil {
		b.p.Printf("warning: the snapshot restores only under driver %s: %v", b.gpus[0].Driver, err)
	} else {
		b.driverPaths = paths
	}
	sources, err := b.sources()
	if err != nil {
		return err
	}
	if err := os.MkdirAll(b.o.Output, 0o755); err != nil {
		return err
	}
	// A half-published output is not a snapshot: remove it on failure so
	// a rerun with the same output can proceed.
	published := false
	defer func() {
		if !published {
			_ = os.RemoveAll(b.o.Output)
		}
	}()
	files, err := Publish(ctx, store.Store{Root: b.o.StateDir}, sources, b.o.Output, 8)
	if err != nil {
		return err
	}
	b.manifest = b.assemble(files)
	if err := b.manifest.Write(filepath.Join(b.o.Output, "manifest.json")); err != nil {
		return err
	}
	if err := run.SyncFS(b.o.Output); err != nil {
		return err
	}
	published = true
	b.p.Printf("published %s: %d files, %.2f GiB (weights %.2f GiB) in %.1fs", b.o.Output, len(files),
		float64(b.manifest.Bytes())/(1<<30), float64(b.manifest.WeightBytes())/(1<<30), time.Since(started).Seconds())
	return nil
}

// tree is a directory published under prefix; role names each file's role
// from its path in the tree, "" leaves it out.
type tree struct {
	root, prefix string
	role         func(rel string) string
}

func fixed(role string) func(string) string { return func(string) string { return role } }

func fsRole(rel string) string {
	if strings.HasSuffix(rel, ".json") {
		return manifest.RoleFSMeta
	}
	return manifest.RoleRuntimeFS
}

func adapterRole(rel string) string {
	if strings.HasPrefix(rel, ".") {
		return ""
	}
	return manifest.RoleAdapter
}

func (b *bake) trees() []tree {
	out := filepath.Join(b.dir, "io", "out")
	trees := []tree{
		{filepath.Join(out, "criu"), "criu", fixed(manifest.RoleCRIU)},
		{filepath.Join(out, "fs"), "fs", fsRole},
		{b.adapter, "adapter", adapterRole},
	}
	if b.driverPaths != nil {
		trees = append(trees, tree{filepath.Join(b.dir, "driver"), "driver", fixed(manifest.RoleDriver)})
	}
	if b.ipc.Interposed {
		// The cuInterpose build the engine has mapped and the coordinator's state.
		trees = append(trees, tree{b.ipcDir, "ipc", fixed(manifest.RoleIPC)},
			tree{filepath.Join(out, "ipc"), "ipc", fixed(manifest.RoleIPC)})
	}
	return trees
}

func (t tree) sources() ([]source, error) {
	var list []source
	err := filepath.WalkDir(t.root, func(p string, d fs.DirEntry, err error) error {
		if err != nil || !d.Type().IsRegular() {
			return err
		}
		rel, _ := filepath.Rel(t.root, p)
		if role := t.role(rel); role != "" {
			list = append(list, source{path: p, rel: filepath.ToSlash(filepath.Join(t.prefix, rel)), role: role})
		}
		return nil
	})
	return list, err
}

func (b *bake) sources() ([]source, error) {
	var list []source
	for _, t := range b.trees() {
		files, err := t.sources()
		if err != nil {
			return nil, err
		}
		list = append(list, files...)
	}
	out := filepath.Join(b.dir, "io", "out")
	for _, x := range b.exports {
		for _, p := range x.Parts {
			list = append(list, source{path: filepath.Join(out, "weights", p.Path), rel: "weights/" + p.Path,
				role: manifest.RoleWeights, digest: p.SHA256})
		}
	}
	return list, nil
}

// Publish stores and links every source with up to workers files at once
// and returns the manifest's file list, sorted by path.
func Publish(ctx context.Context, s store.Store, sources []source, output string,
	workers int) ([]manifest.File, error) {
	files := make([]manifest.File, len(sources))
	errs := make([]error, len(sources))
	jobs := make(chan int)
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range jobs {
				files[i], errs[i] = publishOne(ctx, s, sources[i], output)
			}
		}()
	}
	for i := range sources {
		jobs <- i
	}
	close(jobs)
	wg.Wait()
	if err := errors.Join(errs...); err != nil {
		return nil, err
	}
	sort.Slice(files, func(i, j int) bool { return files[i].Path < files[j].Path })
	return files, nil
}

func publishOne(ctx context.Context, s store.Store, src source, output string) (manifest.File, error) {
	info, err := os.Stat(src.path)
	if err != nil {
		return manifest.File{}, err
	}
	digest := src.digest
	if digest == "" {
		if digest, _, err = s.Add(ctx, src.path); err != nil {
			return manifest.File{}, err
		}
	} else if err := s.AddHashed(ctx, src.path, digest); err != nil {
		return manifest.File{}, err
	}
	if err := s.Link(ctx, digest, filepath.Join(output, filepath.FromSlash(src.rel))); err != nil {
		return manifest.File{}, err
	}
	return manifest.File{Path: src.rel, Size: info.Size(), SHA256: digest, Role: src.role}, nil
}

func (b *bake) assemble(files []manifest.File) manifest.Manifest {
	ranks := make([]manifest.Rank, len(b.exports))
	for i, x := range b.exports {
		r := manifest.Rank{Rank: x.Rank, Allocations: x.Allocations}
		for _, p := range x.Parts {
			p.Path = "weights/" + p.Path
			r.Parts = append(r.Parts, p)
		}
		for _, a := range r.Allocations {
			for k := range a.Sum {
				r.Fingerprint.Sum[k] += a.Sum[k]
			}
			r.Fingerprint.Bytes += a.Bytes
		}
		r.Fingerprint.Allocations = len(r.Allocations)
		ranks[i] = r
	}
	return manifest.Manifest{
		Schema: manifest.Schema, ID: b.o.ID, CreatedAt: time.Now().UTC(), Tool: b.o.Version,
		Profile: b.launch.Profile,
		Launch: manifest.Launch{
			Image: b.launch.Profile.Image, Model: b.served, ModelDir: b.o.ModelDir, ModelSHA256: b.model,
			Argv: b.cfg.Argv, Env: b.launch.Env,
			CaptureSizes: b.launch.CaptureSizes, Ranks: len(b.workers),
		},
		Hardware: b.hardware(),
		Process:  b.process(),
		Files:    files,
		Weights:  manifest.Weights{PartBytes: b.partBytes(), Ranks: ranks},
		Canary:   &b.canary,
	}
}

func (b *bake) partBytes() int64 {
	if b.o.PartBytes > 0 {
		return b.o.PartBytes
	}
	return 1 << 30
}

func (b *bake) hardware() manifest.Hardware {
	h := manifest.Hardware{
		Row: b.row, Topology: b.topology, TopologySHA256: b.topoSHA, Driver: b.gpus[0].Driver,
		Kernel: host.Kernel(), CPUFlags: host.CPUFlags(), VLLM: b.workers[0].VLLM, CRIU: b.dumped.Version,
	}
	if b.driverPaths != nil && b.launch.Env[DriverCheckEnv] == "1" {
		h.DriverPortable, h.DriverFiles = true, b.driverPaths
	}
	for i, g := range b.gpus {
		h.GPUs = append(h.GPUs, manifest.GPU{Rank: i, Index: g.Index, UUID: g.UUID, PCIBusID: g.PCIBusID, Name: g.Name,
			MemoryMiB: g.MemoryMiB})
	}
	return h
}

func (b *bake) process() manifest.Process {
	rankOf := map[int]int{}
	for _, w := range b.workers {
		rankOf[w.PID] = w.Rank
	}
	p := manifest.Process{
		RootPID: b.frozen.Tree.Root, MaxID: b.frozen.Tree.MaxID(), PIDFloor: criu.PIDFloor,
		NeedsModel:          b.frozen.References[b.launch.Profile.Model.MountPath()] > 0,
		CollectivesDetached: b.detached() || b.ipc.Interposed, EngineURL: EngineAddr,
	}
	if b.ipc.Interposed {
		// After the coordinator's prepare no process maps another's memory.
		p.IPC, p.IPCProcesses = interpose.Interpose, b.ipc.Participants
	}
	if len(b.prepared) > 0 {
		p.NCCLMode = b.prepared[0].NCCLMode
	}
	for _, g := range b.gpus {
		p.VisibleUUIDs = append(p.VisibleUUIDs, g.UUID)
	}
	for _, pid := range b.cudaRes.Owners {
		o := manifest.Owner{PID: pid, Rank: -1, Role: "engine"}
		if r, ok := rankOf[pid]; ok {
			o.Rank, o.Role = r, "worker"
		}
		p.Owners = append(p.Owners, o)
	}
	return p
}

// Describe is a one-line summary of a published snapshot.
func Describe(m manifest.Manifest) string {
	return fmt.Sprintf("%s: profile %s, %d ranks on %s, %.2f GiB", m.ID, m.Profile.ID, m.Launch.Ranks,
		m.Hardware.Row.ID, float64(m.Bytes())/(1<<30))
}
