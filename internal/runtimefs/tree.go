// Package runtimefs archives the filesystem trees a process snapshot
// depends on: /workspace (compile and JIT caches, mapped .so files), /tmp
// and /dev/shm (what the engine wrote after the image), taken after the
// CRIU dump so the link_remap files CRIU leaves next to deleted-but-open
// files are ordinary members. A PAX tar carries the data; a JSON manifest
// carries what tar cannot: unix sockets (CRIU recreates them), xattrs and
// per-file digests.
package runtimefs

import (
	"archive/tar"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path"
	"strings"
	"time"
)

// Schema identifies a runtimefs manifest.
const Schema = "warmctl.runtimefs/v2"

// MaxEntries bounds one tree.
const MaxEntries = 262144

// Entry is one member of the tree.
type Entry struct {
	Path   string            `json:"path"`
	Type   string            `json:"type"`
	Link   string            `json:"link,omitempty"`
	Size   int64             `json:"size,omitempty"`
	SHA256 string            `json:"sha256,omitempty"`
	Mode   uint32            `json:"mode"`
	UID    uint32            `json:"uid"`
	GID    uint32            `json:"gid"`
	Mtime  int64             `json:"mtime_ns"`
	Xattrs map[string][]byte `json:"xattrs,omitempty"`
}

// Entry types.
const (
	TypeDir      = "directory"
	TypeFile     = "file"
	TypeHardlink = "hardlink"
	TypeSymlink  = "symlink"
	TypeFifo     = "fifo"
	TypeSocket   = "socket"
)

// Manifest describes one archived tree and where it lives in the container.
type Manifest struct {
	Schema  string  `json:"schema"`
	Root    string  `json:"root"`
	Entries []Entry `json:"entries"`
}

type observed struct {
	Entry
	dev, ino, nlink uint64
}

func confined(s string) bool {
	return s == "." || fs.ValidPath(s) && !strings.ContainsAny(s, "\\\x00")
}

// Validate checks the manifest's internal consistency.
func (m Manifest) Validate() error {
	if m.Schema != Schema || !path.IsAbs(m.Root) || len(m.Entries) == 0 || len(m.Entries) > MaxEntries {
		return errors.New("runtimefs: invalid manifest")
	}
	if m.Entries[0].Path != "." || m.Entries[0].Type != TypeDir {
		return errors.New("runtimefs: manifest must start with the root directory")
	}
	seen := map[string]Entry{}
	for _, e := range m.Entries {
		if err := e.validate(seen); err != nil {
			return err
		}
		seen[e.Path] = e
	}
	return nil
}

// validate checks one entry against the entries before it.
func (e Entry) validate(seen map[string]Entry) error {
	if !confined(e.Path) || e.Mode&^uint32(0o7777) != 0 || e.Size < 0 {
		return fmt.Errorf("runtimefs: invalid entry %q", e.Path)
	}
	if _, dup := seen[e.Path]; dup {
		return fmt.Errorf("runtimefs: duplicate entry %q", e.Path)
	}
	if e.Path != "." && seen[path.Dir(e.Path)].Type != TypeDir {
		return fmt.Errorf("runtimefs: %q has no parent directory", e.Path)
	}
	for name, value := range e.Xattrs {
		if !supportedXattr(name) || len(value) > 65536 {
			return fmt.Errorf("runtimefs: unsupported xattr %q on %q", name, e.Path)
		}
	}
	switch e.Type {
	case TypeFile:
		if len(e.SHA256) != 64 {
			return fmt.Errorf("runtimefs: file %q has no digest", e.Path)
		}
	case TypeHardlink:
		if seen[e.Link].Type != TypeFile {
			return fmt.Errorf("runtimefs: hardlink %q must follow its file", e.Path)
		}
	case TypeSymlink:
		if e.Link == "" || strings.ContainsRune(e.Link, 0) {
			return fmt.Errorf("runtimefs: symlink %q has no target", e.Path)
		}
	case TypeDir, TypeFifo, TypeSocket:
	default:
		return fmt.Errorf("runtimefs: unsupported type %q at %q", e.Type, e.Path)
	}
	return nil
}

func supportedXattr(s string) bool {
	return len(s) <= 255 && !strings.ContainsRune(s, 0) &&
		(strings.HasPrefix(s, "user.") && len(s) > 5 || s == "system.posix_acl_access" || s == "system.posix_acl_default")
}

// Write stores the manifest as JSON.
func (m Manifest) Write(name string) error {
	if err := m.Validate(); err != nil {
		return err
	}
	raw, err := json.Marshal(m)
	if err != nil {
		return err
	}
	return os.WriteFile(name, append(raw, '\n'), 0o644)
}

// Read loads and validates a manifest.
func Read(name string) (Manifest, error) {
	var m Manifest
	raw, err := os.ReadFile(name)
	if err != nil {
		return m, err
	}
	if err := json.Unmarshal(raw, &m); err != nil {
		return m, fmt.Errorf("%s: %w", name, err)
	}
	return m, m.Validate()
}

// Files lists the regular files and hardlinks.
func (m Manifest) Files() []Entry {
	var out []Entry
	for _, e := range m.Entries {
		if e.Type == TypeFile || e.Type == TypeHardlink {
			out = append(out, e)
		}
	}
	return out
}

func scan(ctx context.Context, source string) ([]observed, error) {
	var list []observed
	err := fs.WalkDir(os.DirFS(source), ".", func(name string, _ fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if err := ctx.Err(); err != nil {
			return err
		}
		if len(list) >= MaxEntries {
			return errors.New("runtimefs: too many entries")
		}
		o, err := observe(source, name)
		if err != nil {
			return err
		}
		if o.Type == TypeSymlink {
			if o.Link, err = os.Readlink(source + "/" + name); err != nil {
				return err
			}
		}
		list = append(list, o)
		return nil
	})
	return list, err
}

// Snapshot archives source into a new tar at archive and returns the
// manifest, with root as the tree's path inside the container.
func Snapshot(ctx context.Context, source, archive, root string) (m Manifest, err error) {
	m = Manifest{Schema: Schema, Root: root}
	entries, err := scan(ctx, source)
	if err != nil {
		return m, err
	}
	f, err := os.OpenFile(archive, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o644)
	if err != nil {
		return m, err
	}
	keep := false
	defer func() {
		err = errors.Join(err, f.Close())
		if !keep {
			_ = os.Remove(archive)
		}
	}()
	w := &archiver{ctx: ctx, source: source, tw: tar.NewWriter(f), first: map[inode]int{}}
	for _, o := range entries {
		if err = w.add(&m, o); err != nil {
			return m, err
		}
	}
	if err = m.Validate(); err != nil {
		return m, err
	}
	if err = w.tw.Close(); err != nil {
		return m, err
	}
	if err = f.Sync(); err != nil {
		return m, err
	}
	keep = true
	return m, nil
}

type inode struct{ dev, ino uint64 }

type archiver struct {
	ctx    context.Context
	source string
	tw     *tar.Writer
	first  map[inode]int
}

// add writes one entry: a second name of an inode becomes a hardlink, a
// socket is recorded and left for CRIU to bind again.
func (w *archiver) add(m *Manifest, o observed) error {
	e := o.Entry
	if e.Type == TypeSocket {
		m.Entries = append(m.Entries, e)
		return nil
	}
	h := &tar.Header{Name: e.Path, Mode: int64(e.Mode), Uid: int(e.UID), Gid: int(e.GID),
		ModTime: time.Unix(0, e.Mtime), Format: tar.FormatPAX}
	switch e.Type {
	case TypeDir:
		h.Typeflag = tar.TypeDir
	case TypeFifo:
		h.Typeflag = tar.TypeFifo
	case TypeSymlink:
		h.Typeflag, h.Linkname = tar.TypeSymlink, e.Link
	case TypeFile:
		if index, ok := w.first[inode{o.dev, o.ino}]; ok && o.nlink > 1 {
			e.Type, e.Link, e.SHA256, e.Size = TypeHardlink, m.Entries[index].Path, "", 0
			h.Typeflag, h.Linkname = tar.TypeLink, e.Link
		} else {
			w.first[inode{o.dev, o.ino}] = len(m.Entries)
			h.Typeflag, h.Size = tar.TypeReg, e.Size
		}
	default:
		return fmt.Errorf("runtimefs: unsupported object at %s", e.Path)
	}
	if err := w.tw.WriteHeader(h); err != nil {
		return err
	}
	if e.Type == TypeFile {
		sum, err := copyFile(w.ctx, w.source+"/"+e.Path, w.tw, e.Size)
		if err != nil {
			return err
		}
		e.SHA256 = sum
	}
	m.Entries = append(m.Entries, e)
	return nil
}

func copyFile(ctx context.Context, name string, w io.Writer, size int64) (string, error) {
	in, err := os.Open(name)
	if err != nil {
		return "", err
	}
	defer in.Close()
	h := sha256.New()
	n, err := io.Copy(io.MultiWriter(w, h), &contextReader{ctx, io.LimitReader(in, size)})
	if err != nil {
		return "", err
	}
	if n != size {
		return "", fmt.Errorf("runtimefs: %s shrank while archiving", name)
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

// Mark is a receiver's claim on a tree's directory: written before the
// install, it is the one entry Install accepts in its destination, and it
// tells a later restore in the same place that what the directory holds is
// a restore's, not someone's data on a volume.
const Mark = ".warmctl-receiver"

// Install unpacks archive into destination, checking every file against
// the manifest. destination is created, or must be an empty directory (a
// container's /tmp, /dev/shm or a fresh bind mount) but for Mark. Sockets
// are left to CRIU, which binds them again.
func Install(ctx context.Context, archive, destination string, m Manifest) error {
	if err := m.Validate(); err != nil {
		return err
	}
	f, err := os.Open(archive)
	if err != nil {
		return err
	}
	defer f.Close()
	created, err := prepareDestination(destination)
	if err != nil {
		return err
	}
	if err := extract(ctx, tar.NewReader(&contextReader{ctx, f}), destination, m); err != nil {
		if created {
			_ = os.RemoveAll(destination)
		}
		return err
	}
	return nil
}

func extract(ctx context.Context, tr *tar.Reader, destination string, m Manifest) error {
	root, err := os.OpenRoot(destination)
	if err != nil {
		return err
	}
	defer root.Close()
	for _, e := range m.Entries {
		if e.Type == TypeSocket {
			continue
		}
		h, err := tr.Next()
		if err != nil {
			return fmt.Errorf("runtimefs: archive ends before %s: %w", e.Path, err)
		}
		if h.Name != e.Path {
			return fmt.Errorf("runtimefs: archive has %s where the manifest has %s", h.Name, e.Path)
		}
		if err := install(ctx, root, destination, e, h, tr); err != nil {
			return fmt.Errorf("runtimefs: %s: %w", e.Path, err)
		}
	}
	if _, err := tr.Next(); !errors.Is(err, io.EOF) {
		return errors.New("runtimefs: archive has entries the manifest lacks")
	}
	return applyAll(destination, m)
}

// applyAll sets owners, modes, xattrs and times, deepest first so that a
// directory's mtime survives the entries created in it.
func applyAll(destination string, m Manifest) error {
	for i := len(m.Entries) - 1; i >= 0; i-- {
		e := m.Entries[i]
		if e.Type == TypeHardlink || e.Type == TypeSocket {
			continue
		}
		if err := applyMetadata(destination+"/"+e.Path, e); err != nil {
			return fmt.Errorf("runtimefs: %s: %w", e.Path, err)
		}
	}
	return nil
}

func prepareDestination(destination string) (bool, error) {
	err := os.Mkdir(destination, 0o700)
	if err == nil {
		return true, nil
	}
	if !errors.Is(err, fs.ErrExist) {
		return false, err
	}
	entries, err := os.ReadDir(destination)
	if err != nil {
		return false, err
	}
	for _, e := range entries {
		if e.Name() != Mark {
			return false, fmt.Errorf("runtimefs: %s is not empty", destination)
		}
	}
	return false, nil
}

func install(ctx context.Context, root *os.Root, destination string, e Entry, h *tar.Header, tr *tar.Reader) error {
	switch e.Type {
	case TypeDir:
		if e.Path == "." {
			return nil
		}
		return root.Mkdir(e.Path, 0o700)
	case TypeSymlink:
		return root.Symlink(e.Link, e.Path)
	case TypeHardlink:
		return root.Link(e.Link, e.Path)
	case TypeFifo:
		return mkfifo(destination+"/"+e.Path, e.Mode)
	case TypeFile:
		if h.Typeflag != tar.TypeReg || h.Size != e.Size {
			return errors.New("archive header differs from manifest")
		}
		out, err := root.OpenFile(e.Path, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o600)
		if err != nil {
			return err
		}
		sum := sha256.New()
		n, err := io.Copy(io.MultiWriter(out, sum), &contextReader{ctx, tr})
		if closeErr := out.Close(); err == nil {
			err = closeErr
		}
		if err != nil {
			return err
		}
		if n != e.Size || hex.EncodeToString(sum.Sum(nil)) != e.SHA256 {
			return errors.New("content differs from manifest digest")
		}
		return nil
	}
	return fmt.Errorf("unsupported type %q", e.Type)
}

type contextReader struct {
	ctx context.Context
	r   io.Reader
}

func (r *contextReader) Read(p []byte) (int, error) {
	if err := r.ctx.Err(); err != nil {
		return 0, err
	}
	return r.r.Read(p)
}
