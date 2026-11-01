package run

import (
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"syscall"

	"github.com/TensorRails/warmctl/v2/internal/runtimefs"
)

// MergeEnv is base with launch's variables over it, launch's sorted by name:
// the environment of an engine started in this container rather than in one
// warmctl made, as a container's is the image's plus the profile's.
func MergeEnv(base []string, launch map[string]string) []string {
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

// SetAside is the suffix Place gives what it found at a path.
const SetAside = ".warmctl-image"

// Place makes path lead to dir without a mount, for a container that may
// not mount (the dmtcp backend): a symbolic link replaces an earlier link or
// an empty directory, and anything else at path is moved aside, once, to
// path+SetAside. Processes that open files through path see dir's real
// paths, which a DMTCP restart needs to find again.
func Place(dir, path string) error {
	if target, err := os.Readlink(path); err == nil {
		if target == dir {
			return nil
		}
		if err := os.Remove(path); err != nil {
			return err
		}
	} else if info, err := os.Lstat(path); err == nil {
		entries, _ := os.ReadDir(path)
		switch {
		case info.IsDir() && len(entries) == 0:
			if err := os.Remove(path); err != nil {
				return err
			}
		default:
			if _, err := os.Lstat(path + SetAside); err == nil {
				return fmt.Errorf("%s exists and %s is taken: remove one", path, path+SetAside)
			}
			if err := setAside(path); err != nil {
				return err
			}
		}
	} else if !errors.Is(err, os.ErrNotExist) {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	return os.Symlink(dir, path)
}

// rename is os.Rename; a test makes it fail as overlayfs does.
var rename = os.Rename

// setAside moves path to path+SetAside. Overlayfs without redirect_dir,
// Docker's and containerd's default, refuses to rename a directory of an
// image's layer (EXDEV): it is copied aside and removed instead.
func setAside(path string) error {
	err := rename(path, path+SetAside)
	if !errors.Is(err, syscall.EXDEV) {
		return err
	}
	if err := copyTree(path, path+SetAside); err != nil {
		return fmt.Errorf("set %s aside: %w", path, err)
	}
	return os.RemoveAll(path)
}

// copyTree copies the directories, files and links under src to dst.
func copyTree(src, dst string) error {
	return filepath.WalkDir(src, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		rel, err := filepath.Rel(src, p)
		if err != nil {
			return err
		}
		to := filepath.Join(dst, rel)
		info, err := d.Info()
		if err != nil {
			return err
		}
		switch {
		case d.IsDir():
			return os.MkdirAll(to, info.Mode().Perm())
		case d.Type()&fs.ModeSymlink != 0:
			target, err := os.Readlink(p)
			if err != nil {
				return err
			}
			return os.Symlink(target, to)
		default:
			return CopyFile(p, to, info.Mode().Perm())
		}
	})
}

// ClaimDir makes dir a tree of this run in a container warmctl did not make:
// it may be empty or hold what an earlier run left under its mark, which
// goes; anything else may be the operator's data on a volume and is refused.
func ClaimDir(dir string) error {
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
		var names []string
		for _, e := range entries[:min(len(entries), 3)] {
			names = append(names, e.Name())
		}
		if len(entries) > 3 {
			names = append(names, "...")
		}
		return fmt.Errorf("%s holds files no warmctl run left there (%s; no %s): remove them, mount an emptyDir "+
			"at %s, or leave it to the container's layer", dir, strings.Join(names, ", "), runtimefs.Mark, dir)
	}
	for _, e := range entries {
		if err := os.RemoveAll(filepath.Join(dir, e.Name())); err != nil {
			return err
		}
	}
	return os.WriteFile(filepath.Join(dir, runtimefs.Mark), []byte("a warmctl run's directory\n"), 0o644)
}
