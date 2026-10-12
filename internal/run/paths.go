package run

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
)

var idPattern = regexp.MustCompile(`^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,63}$`)

// CheckID refuses ids that would not make clean container names and paths.
func CheckID(id string) error {
	if !idPattern.MatchString(id) {
		return fmt.Errorf("invalid id %q: letters, digits, '.', '_' and '-', at most 64", id)
	}
	return nil
}

// ContainerName is the container of run id.
func ContainerName(id string) string { return "warmctl-" + id }

// CopyExecutable copies this program into dir, for the container to run.
func CopyExecutable(dir, name string) error {
	self, err := os.Executable()
	if err != nil {
		return err
	}
	return CopyFile(self, filepath.Join(dir, name), 0o755)
}

// CopyFile copies a file with mode.
func CopyFile(src, dst string, mode os.FileMode) error {
	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer in.Close()
	out, err := os.OpenFile(dst, os.O_CREATE|os.O_EXCL|os.O_WRONLY, mode)
	if err != nil {
		return err
	}
	if _, err := io.Copy(out, in); err != nil {
		out.Close()
		return err
	}
	return out.Close()
}

// ModelSHA256 identifies a model directory by its file names, sizes and the
// bytes of every file under 4 MiB (configs, tokenizers). Weight files are
// identified by name and size: hashing a 500 GB checkpoint per bake buys
// nothing the restored weights do not prove.
func ModelSHA256(dir string) (string, error) {
	var names []string
	err := filepath.WalkDir(dir, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.Type().IsRegular() || d.Type()&fs.ModeSymlink != 0 {
			rel, _ := filepath.Rel(dir, p)
			if !strings.HasPrefix(rel, ".") {
				names = append(names, rel)
			}
		}
		return nil
	})
	if err != nil {
		return "", err
	}
	if len(names) == 0 {
		return "", errors.New("model directory is empty")
	}
	sort.Strings(names)
	h := sha256.New()
	for _, name := range names {
		info, err := os.Stat(filepath.Join(dir, name))
		if err != nil {
			return "", err
		}
		fmt.Fprintf(h, "%s\x00%d\x00", name, info.Size())
		if info.Size() < 4<<20 {
			raw, err := os.ReadFile(filepath.Join(dir, name))
			if err != nil {
				return "", err
			}
			h.Write(raw)
		}
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}
