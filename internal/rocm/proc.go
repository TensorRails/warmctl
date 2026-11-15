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
			out.maps = append(out.maps, strings.Join(fields[5:], " "))
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

// Coverage is what a snapshot carries of the files a tree maps or holds
// open. A restart maps them again from their paths and checks their size
// and modification time, so each must be one of:
//   - under a tree the snapshot archives (its mtime is kept to the ns);
//   - under the model's directory (the same volume at the same path);
//   - a file of the image older than the engine (the same image digest);
//   - the GPU library itself.
//
// Uncovered lists the others; Model the model's mapped files, recorded so a
// restore can check them before it starts.
type Coverage struct {
	Uncovered []string
	Model     []manifest.MappedFile
}

// imageDirs hold the image's files.
var imageDirs = []string{"/usr/", "/opt/", "/lib/", "/lib64/", "/etc/", "/bin/", "/sbin/"}

// Cover checks the files of pids against the trees, the model directory and
// the image (files not modified since started).
func Cover(pids []int, trees []string, modelDir, lib string, started time.Time) (Coverage, error) {
	var c Coverage
	seen := map[string]bool{}
	for _, pid := range pids {
		f, err := readProc(pid)
		if os.IsNotExist(err) {
			continue
		}
		if err != nil {
			return c, err
		}
		for _, p := range append(append([]string(nil), f.maps...), f.fds...) {
			if seen[p] || !strings.HasPrefix(p, "/") || strings.HasSuffix(p, " (deleted)") || pseudo(p) {
				continue
			}
			seen[p] = true
			switch {
			case p == lib, under(p, trees):
			case modelDir != "" && under(p, []string{modelDir}):
				if m, err := stat(p); err == nil {
					c.Model = append(c.Model, m)
				} else {
					c.Uncovered = append(c.Uncovered, p+" ("+err.Error()+")")
				}
			case imageFile(p, started):
			default:
				c.Uncovered = append(c.Uncovered, p)
			}
		}
	}
	sort.Strings(c.Uncovered)
	sort.Slice(c.Model, func(i, j int) bool { return c.Model[i].Path < c.Model[j].Path })
	return c, nil
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

// imageFile reports a file of the image: one older than the container
// (its PID 1), wherever it is (the ROCm image keeps Python under
// /root/.local), or under the image's usual directories and older than the
// engine.
func imageFile(p string, started time.Time) bool {
	info, err := os.Stat(p)
	if err != nil || !info.Mode().IsRegular() {
		return false
	}
	if c, ok := containerStart(); ok && info.ModTime().Before(c) {
		return true
	}
	return under(p, trimSlash(imageDirs)) && info.ModTime().Before(started)
}

// containerStart is when PID 1 started: the boot time from /proc/stat plus
// its start time in clock ticks (USER_HZ, 100 on Linux).
func containerStart() (time.Time, bool) {
	stat, err := os.ReadFile(filepath.Join(Proc, "stat"))
	if err != nil {
		return time.Time{}, false
	}
	var boot int64
	for _, line := range strings.Split(string(stat), "\n") {
		if v, ok := strings.CutPrefix(line, "btime "); ok {
			boot, _ = strconv.ParseInt(strings.TrimSpace(v), 10, 64)
		}
	}
	raw, err := os.ReadFile(filepath.Join(Proc, "1", "stat"))
	if err != nil || boot == 0 {
		return time.Time{}, false
	}
	// The command may hold spaces and parentheses: fields follow the last ")".
	i := strings.LastIndexByte(string(raw), ')')
	if i < 0 {
		return time.Time{}, false
	}
	fields := strings.Fields(string(raw)[i+1:])
	// starttime is field 22 of the line; fields here start at field 3.
	if len(fields) < 20 {
		return time.Time{}, false
	}
	ticks, err := strconv.ParseInt(fields[19], 10, 64)
	if err != nil {
		return time.Time{}, false
	}
	return time.Unix(boot, 0).Add(time.Duration(ticks) * time.Second / 100), true
}

func trimSlash(dirs []string) []string {
	out := make([]string, len(dirs))
	for i, d := range dirs {
		out[i] = strings.TrimSuffix(d, "/")
	}
	return out
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
