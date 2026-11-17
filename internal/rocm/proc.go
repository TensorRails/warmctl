package rocm

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/TensorRails/warmctl/v2/internal/manifest"
)

// Proc is where processes are read; tests point it at a fixture.
var Proc = "/proc"

// runtimeLibraries mark a process that runs ROCm: it holds GPU state.
var runtimeLibraries = []string{"libamdhip64.so", "libhsa-runtime64.so"}

// device reports a path of an AMD GPU device: the compute device or a DRM node.
func device(p string) bool {
	return p == "/dev/kfd" || strings.HasPrefix(p, "/dev/dri/")
}

// procFiles are the files a process maps (first) and holds open.
type procFiles struct {
	maps []string
	fds  []string
}

func readProc(pid int) (procFiles, error) {
	var out procFiles
	raw, err := os.ReadFile(filepath.Join(Proc, strconv.Itoa(pid), "maps"))
	if err != nil {
		return out, err
	}
	for _, line := range strings.Split(string(raw), "\n") {
		// address perms offset dev inode path
		fields := strings.Fields(line)
		if len(fields) >= 6 {
			// /proc/maps preserves spaces in the pathname. Strip only the
			// five header columns; the file name can contain repeated spaces.
			path := line
			for range 5 {
				end := strings.IndexAny(path, " \t")
				path = strings.TrimLeft(path[end:], " \t")
			}
			out.maps = append(out.maps, path)
		}
	}
	dir := filepath.Join(Proc, strconv.Itoa(pid), "fd")
	fds, err := os.ReadDir(dir)
	if err != nil {
		return out, err
	}
	for _, fd := range fds {
		if target, err := os.Readlink(filepath.Join(dir, fd.Name())); err == nil {
			out.fds = append(out.fds, target)
		}
	}
	return out, nil
}

func (f procFiles) mapsAny(names ...string) bool {
	for _, m := range f.maps {
		for _, n := range names {
			if strings.Contains(filepath.Base(m), n) || m == n {
				return true
			}
		}
	}
	return false
}

// Owners are the processes among pids that run ROCm.
func Owners(pids []int) ([]int, error) {
	var out []int
	for _, pid := range pids {
		f, err := readProc(pid)
		if os.IsNotExist(err) {
			continue
		}
		if err != nil {
			return nil, err
		}
		if f.mapsAny(runtimeLibraries...) {
			out = append(out, pid)
		}
	}
	sort.Ints(out)
	return out, nil
}

// VerifyReleased checks a tree after every worker's GPU checkpoint, before
// DMTCP dumps it, and returns the processes that run ROCm. A process that
// still holds or maps an AMD device kept GPU state no checkpoint took (one
// warmctl did not ask, as an API server that initialized ROCm): the dump is
// refused rather than taken without it. A process that runs ROCm without
// lib cannot be restored.
func VerifyReleased(pids []int, lib string) ([]int, error) {
	var owners []int
	for _, pid := range pids {
		f, err := readProc(pid)
		if os.IsNotExist(err) {
			continue
		}
		if err != nil {
			return nil, err
		}
		for _, p := range append(append([]string(nil), f.maps...), f.fds...) {
			if device(p) {
				return nil, fmt.Errorf("process %d still holds %s after the GPU checkpoint: GPU state no worker "+
					"checkpointed (a process other than vLLM's workers that initialized ROCm?)", pid, p)
			}
		}
		if !f.mapsAny(runtimeLibraries...) {
			continue
		}
		if !f.mapsAny(lib) {
			return nil, fmt.Errorf("process %d runs ROCm without %s", pid, lib)
		}
		owners = append(owners, pid)
	}
	sort.Ints(owners)
	return owners, nil
}

// VerifyPreloaded checks that every restored process that runs ROCm maps lib:
// they replay their GPU state through it next.
func VerifyPreloaded(pids []int, lib string) error {
	for _, pid := range pids {
		f, err := readProc(pid)
		if os.IsNotExist(err) {
			continue
		}
		if err != nil {
			return err
		}
		if f.mapsAny(runtimeLibraries...) && !f.mapsAny(lib) {
			return fmt.Errorf("restored process %d runs ROCm without %s", pid, lib)
		}
	}
	return nil
}

// Coverage describes the dependencies outside the archived trees. Mapped
// entries must be present at restart. Uncovered entries could not be tracked.
type Coverage struct {
	Uncovered []string
	Mapped    []manifest.MappedFile
}

// Cover checks the files of pids against the archived trees, the model
// directory and the mounts visible to each process.
func Cover(pids []int, trees []string, modelDir, lib string, started time.Time) (Coverage, error) {
	var c Coverage
	mapped := map[string]manifest.MappedFile{}
	uncovered := map[string]bool{}
	for _, pid := range pids {
		f, err := readProc(pid)
		if os.IsNotExist(err) {
			continue
		}
		if err != nil {
			return c, err
		}
		mounts := processMounts(pid)
		for _, p := range append(f.maps, f.fds...) {
			if !strings.HasPrefix(p, "/") || strings.HasSuffix(p, " (deleted)") || pseudo(p) ||
				p == lib || under(p, trees) {
				continue
			}
			isModel := modelDir != "" && under(p, []string{modelDir})
			// A read-only container root can supply its image files. A bind
			// mount below it is a separate dependency even if also read-only.
			mount := mounts.at(p)
			if !isModel && mount.path == "/" && mount.readOnly {
				continue
			}
			info, err := os.Stat(p)
			if err != nil {
				uncovered[p+" ("+err.Error()+")"] = true
				continue
			}
			// Files newly created on the writable container layer (JIT
			// caches, for example) belong in an archived tree. A mapped
			// model or a separate volume can be supplied at restore.
			onVolume := mount.path != "" && mount.path != "/"
			if !isModel && (!info.Mode().IsRegular() || (!onVolume && !info.ModTime().Before(started))) {
				uncovered[p] = true
				continue
			}
			// /proc names the actual file, including the blobs behind
			// Hugging Face snapshot links. For a writable or unknown root,
			// age alone cannot prove the base image contains the file.
			mapped[p] = manifest.MappedFile{Path: p, Size: info.Size(), MtimeNS: info.ModTime().UnixNano()}
		}
	}
	for p := range uncovered {
		c.Uncovered = append(c.Uncovered, p)
	}
	for _, f := range mapped {
		c.Mapped = append(c.Mapped, f)
	}
	sort.Strings(c.Uncovered)
	sort.Slice(c.Mapped, func(i, j int) bool { return c.Mapped[i].Path < c.Mapped[j].Path })
	return c, nil
}

type mountPoint struct {
	path     string
	readOnly bool
}

type mountTable []mountPoint

// An incomplete or inaccessible mount table gives no file an image exemption.
func processMounts(pid int) mountTable {
	raw, err := os.ReadFile(filepath.Join(Proc, strconv.Itoa(pid), "mountinfo"))
	if err != nil {
		return nil
	}
	escape := strings.NewReplacer(`\040`, " ", `\011`, "\t", `\012`, "\n", `\134`, `\`)
	var mounts mountTable
	hasRoot := false
	for _, line := range strings.Split(strings.TrimSuffix(string(raw), "\n"), "\n") {
		before, _, ok := strings.Cut(line, " - ")
		fields := strings.Fields(before)
		if !ok || len(fields) < 6 || !filepath.IsAbs(fields[4]) {
			return nil
		}
		p := escape.Replace(fields[4])
		mounts = append(mounts, mountPoint{path: p, readOnly: strings.Contains(","+fields[5]+",", ",ro,")})
		hasRoot = hasRoot || p == "/"
	}
	if !hasRoot {
		return nil
	}
	return mounts
}

// at returns the deepest containing mount, including file bind mounts.
func (m mountTable) at(p string) mountPoint {
	var best mountPoint
	for _, mount := range m {
		if len(mount.path) >= len(best.path) && (mount.path == "/" || p == mount.path ||
			strings.HasPrefix(p, mount.path+"/")) {
			best = mount
		}
	}
	return best
}

func pseudo(p string) bool {
	for _, prefix := range []string{"/dev/", "/proc/", "/sys/", "/memfd:", "/anon_hugepage", "/SYSV"} {
		if strings.HasPrefix(p, prefix) {
			return true
		}
	}
	return false
}

func under(p string, dirs []string) bool {
	for _, d := range dirs {
		d = strings.TrimSuffix(d, "/")
		if p == d || strings.HasPrefix(p, d+"/") {
			return true
		}
	}
	return false
}

// stat is a file's size and modification time.
func stat(p string) (manifest.MappedFile, error) {
	info, err := os.Stat(p)
	if err != nil {
		return manifest.MappedFile{}, err
	}
	return manifest.MappedFile{Path: p, Size: info.Size(), MtimeNS: info.ModTime().UnixNano()}, nil
}

// SameFiles checks recorded files before a restart: same size and mtime.
func SameFiles(files []manifest.MappedFile) error {
	for _, want := range files {
		got, err := stat(want.Path)
		if err != nil {
			return fmt.Errorf("%s, which the snapshot's processes mapped: %w", want.Path, err)
		}
		if got.Size != want.Size || got.MtimeNS != want.MtimeNS {
			return fmt.Errorf("%s changed since the snapshot (size %d, mtime %d ns; the snapshot's %d, %d): a "+
				"restart maps it again", want.Path, got.Size, got.MtimeNS, want.Size, want.MtimeNS)
		}
	}
	return nil
}
