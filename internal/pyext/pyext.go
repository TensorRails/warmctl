// Package pyext materializes the embedded vLLM worker extension on the host,
// where containers mount it read-only at /opt/warmctl/adapter.
package pyext

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"io/fs"
	"os"
	"path/filepath"
	"sort"

	"github.com/TensorRails/warmctl/v2"
)

// GuestPath is where containers see the package root (on PYTHONPATH).
const GuestPath = "/opt/warmctl/adapter"

type file struct {
	name string
	data []byte
}

func files() ([]file, error) {
	var out []file
	err := fs.WalkDir(warmctl.Adapter, "python", func(name string, d fs.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			return err
		}
		data, err := fs.ReadFile(warmctl.Adapter, name)
		if err != nil {
			return err
		}
		rel, _ := filepath.Rel("python", name)
		out = append(out, file{rel, data})
		return nil
	})
	sort.Slice(out, func(i, j int) bool { return out[i].name < out[j].name })
	return out, err
}

// SHA256 identifies the package: every file's name and bytes in name order.
func SHA256() (string, error) {
	list, err := files()
	if err != nil {
		return "", err
	}
	h := sha256.New()
	for _, f := range list {
		h.Write([]byte(f.name))
		h.Write([]byte{0})
		h.Write(f.data)
		h.Write([]byte{0})
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

// Materialize writes the package under root/<sha> once and returns that
// directory and the digest. An existing complete copy is reused.
func Materialize(root string) (string, string, error) {
	sum, err := SHA256()
	if err != nil {
		return "", "", err
	}
	dir := filepath.Join(root, sum)
	if _, err := os.Stat(filepath.Join(dir, ".complete")); err == nil {
		return dir, sum, nil
	}
	tmp, err := os.MkdirTemp(root, ".adapter-")
	if errors.Is(err, fs.ErrNotExist) {
		if err = os.MkdirAll(root, 0o755); err == nil {
			tmp, err = os.MkdirTemp(root, ".adapter-")
		}
	}
	if err != nil {
		return "", "", err
	}
	defer os.RemoveAll(tmp)
	list, err := files()
	if err != nil {
		return "", "", err
	}
	for _, f := range list {
		target := filepath.Join(tmp, f.name)
		if err := os.MkdirAll(filepath.Dir(target), 0o755); err != nil {
			return "", "", err
		}
		if err := os.WriteFile(target, f.data, 0o644); err != nil {
			return "", "", err
		}
	}
	if err := os.WriteFile(filepath.Join(tmp, ".complete"), []byte(sum+"\n"), 0o644); err != nil {
		return "", "", err
	}
	if err := os.Chmod(tmp, 0o755); err != nil {
		return "", "", err
	}
	if err := os.Rename(tmp, dir); err != nil && !errors.Is(err, fs.ErrExist) {
		if _, statErr := os.Stat(filepath.Join(dir, ".complete")); statErr != nil {
			return "", "", err
		}
	}
	return dir, sum, nil
}
