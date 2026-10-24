// Package store keeps snapshot blobs content-addressed under
// <state>/blobs/sha256/<digest>. Snapshot and instance directories hold hard
// links into it, so a part shared by two snapshots exists once on disk and
// pulling a snapshot that shares parts with one already present copies
// nothing. GC removes the blobs no snapshot refers to.
package store

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
)

// Store is a blob directory.
type Store struct {
	Root string
}

var digestPattern = regexp.MustCompile(`^[0-9a-f]{64}$`)

// Blobs is the blob directory.
func (s Store) Blobs() string { return filepath.Join(s.Root, "blobs", "sha256") }

// BlobPath is where digest lives.
func (s Store) BlobPath(digest string) (string, error) {
	if !digestPattern.MatchString(digest) {
		return "", fmt.Errorf("store: invalid digest %q", digest)
	}
	return filepath.Join(s.Blobs(), digest), nil
}

// Has reports whether the blob is present.
func (s Store) Has(digest string) bool {
	name, err := s.BlobPath(digest)
	if err != nil {
		return false
	}
	info, err := os.Stat(name)
	return err == nil && info.Mode().IsRegular()
}

// Add hashes the file and makes it a blob: a hard link when the file is on
// the store's filesystem, a copy otherwise. The blob, and through the link
// the file itself, becomes read-only.
func (s Store) Add(ctx context.Context, name string) (string, int64, error) {
	digest, size, err := HashFile(ctx, name)
	if err != nil {
		return "", 0, err
	}
	return digest, size, s.AddHashed(ctx, name, digest)
}

// AddHashed records a file whose digest the caller already computed while
// writing it, without reading it again.
func (s Store) AddHashed(ctx context.Context, name, digest string) error {
	blob, err := s.BlobPath(digest)
	if err != nil {
		return err
	}
	if s.Has(digest) {
		return nil
	}
	if err := os.MkdirAll(s.Blobs(), 0o700); err != nil {
		return err
	}
	tmp, err := os.CreateTemp(s.Blobs(), ".add-*")
	if err != nil {
		return err
	}
	tmpName := tmp.Name()
	tmp.Close()
	os.Remove(tmpName)
	if err := os.Link(name, tmpName); err != nil {
		got, _, err := CopyFile(ctx, name, tmpName)
		if err != nil {
			return err
		}
		if got != digest {
			os.Remove(tmpName)
			return fmt.Errorf("store: %s hashed to %s while copying, expected %s", name, got, digest)
		}
	}
	if err := os.Chmod(tmpName, 0o444); err != nil {
		os.Remove(tmpName)
		return err
	}
	return claim(tmpName, blob)
}

// claim makes tmp the blob unless another writer got there first: a link
// never replaces an existing name, so every path linked from a blob keeps
// pointing at the one copy (rename would swap the inode under them).
func claim(tmp, blob string) error {
	defer os.Remove(tmp)
	if err := os.Link(tmp, blob); err != nil && !errors.Is(err, fs.ErrExist) {
		return err
	}
	return nil
}

// Link materializes the blob at dst: a hard link, or a verified copy when
// dst is on another filesystem.
func (s Store) Link(ctx context.Context, digest, dst string) error {
	blob, err := s.BlobPath(digest)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(dst), 0o700); err != nil {
		return err
	}
	err = os.Link(blob, dst)
	if err == nil || errors.Is(err, fs.ErrExist) {
		return err
	}
	got, _, err := CopyFile(ctx, blob, dst)
	if err != nil {
		return err
	}
	if got != digest {
		os.Remove(dst)
		return fmt.Errorf("store: blob %s is corrupt (%s)", digest, got)
	}
	return nil
}

// List returns every blob digest.
func (s Store) List() ([]string, error) {
	entries, err := os.ReadDir(s.Blobs())
	if errors.Is(err, fs.ErrNotExist) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	var out []string
	for _, e := range entries {
		if digestPattern.MatchString(e.Name()) {
			out = append(out, e.Name())
		}
	}
	return out, nil
}

// GC removes blobs not in keep and returns the bytes freed. A blob still
// linked from a snapshot directory keeps its data there; only the store's
// name goes.
func (s Store) GC(ctx context.Context, keep map[string]bool) (int64, error) {
	digests, err := s.List()
	if err != nil {
		return 0, err
	}
	var freed int64
	for _, d := range digests {
		if keep[d] {
			continue
		}
		if err := ctx.Err(); err != nil {
			return freed, err
		}
		name := filepath.Join(s.Blobs(), d)
		info, err := os.Stat(name)
		if err != nil {
			continue
		}
		if err := os.Remove(name); err != nil {
			return freed, err
		}
		freed += info.Size()
	}
	return freed, nil
}

// HashFile is the SHA-256 and size of a file.
func HashFile(ctx context.Context, name string) (string, int64, error) {
	f, err := os.Open(name)
	if err != nil {
		return "", 0, err
	}
	defer f.Close()
	h := sha256.New()
	n, err := copyChecked(ctx, h, f)
	if err != nil {
		return "", 0, err
	}
	return hex.EncodeToString(h.Sum(nil)), n, nil
}

// CopyFile creates dst (exclusively) from src, hashing the bytes as they
// pass and syncing the result.
func CopyFile(ctx context.Context, src, dst string) (string, int64, error) {
	in, err := os.Open(src)
	if err != nil {
		return "", 0, err
	}
	defer in.Close()
	out, err := os.OpenFile(dst, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o644)
	if err != nil {
		return "", 0, err
	}
	h := sha256.New()
	n, err := copyChecked(ctx, io.MultiWriter(out, h), in)
	if err == nil {
		err = out.Sync()
	}
	if closeErr := out.Close(); err == nil {
		err = closeErr
	}
	if err != nil {
		os.Remove(dst)
		return "", 0, err
	}
	return hex.EncodeToString(h.Sum(nil)), n, nil
}

func copyChecked(ctx context.Context, w io.Writer, r io.Reader) (int64, error) {
	buf := make([]byte, 4<<20)
	var total int64
	for {
		if err := ctx.Err(); err != nil {
			return total, err
		}
		n, err := r.Read(buf)
		if n > 0 {
			if _, werr := w.Write(buf[:n]); werr != nil {
				return total, werr
			}
			total += int64(n)
		}
		if errors.Is(err, io.EOF) {
			return total, nil
		}
		if err != nil {
			return total, err
		}
	}
}

// Ingest streams r into the store as digest, checking the bytes hash to it
// and number size. A blob already present is not read again.
func (s Store) Ingest(ctx context.Context, r io.Reader, digest string, size int64) error {
	if s.Has(digest) {
		return nil
	}
	name, err := s.Write(ctx, r, digest, size)
	if err != nil {
		return err
	}
	if err := syncFile(name); err != nil {
		os.Remove(name)
		return err
	}
	return s.Claim(name, digest)
}

// Write streams r into a new file next to the blobs, checking that its
// bytes hash to digest and number size, and returns its name. The file is
// read-only already, as a blob is (CRIU checks the mode of a file a restored
// process maps), but not a blob yet, nor synced: Claim makes it one once it
// is on disk.
func (s Store) Write(ctx context.Context, r io.Reader, digest string, size int64) (string, error) {
	if _, err := s.BlobPath(digest); err != nil {
		return "", err
	}
	if err := os.MkdirAll(s.Blobs(), 0o700); err != nil {
		return "", err
	}
	tmp, err := os.CreateTemp(s.Blobs(), ".ingest-*")
	if err != nil {
		return "", err
	}
	h := sha256.New()
	n, err := copyChecked(ctx, io.MultiWriter(tmp, h), r)
	if closeErr := tmp.Close(); err == nil {
		err = closeErr
	}
	if err == nil {
		if got := hex.EncodeToString(h.Sum(nil)); got != digest || n != size {
			err = fmt.Errorf("store: blob %s arrived as %s (%d of %d bytes)", digest, got, n, size)
		}
	}
	if err == nil {
		err = os.Chmod(tmp.Name(), 0o444)
	}
	if err != nil {
		os.Remove(tmp.Name())
		return "", err
	}
	return tmp.Name(), nil
}

// Claim makes name, which Write returned and which is on disk now (synced,
// or written before Sync), the read-only blob of digest. Its name goes;
// hard links to it stay.
func (s Store) Claim(name, digest string) error {
	blob, err := s.BlobPath(digest)
	if err != nil {
		return err
	}
	if err := os.Chmod(name, 0o444); err != nil {
		os.Remove(name)
		return err
	}
	return claim(name, blob)
}

// Sync writes everything on the store's filesystem to disk, the files
// Write left among it.
func (s Store) Sync() error { return syncFS(s.Blobs()) }

func syncFile(name string) error {
	f, err := os.Open(name)
	if err != nil {
		return err
	}
	defer f.Close()
	return f.Sync()
}
