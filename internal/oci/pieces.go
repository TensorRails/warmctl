package oci

import (
	"bufio"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"

	"github.com/opencontainers/go-digest"
	ocispec "github.com/opencontainers/image-spec/specs-go/v1"
	"oras.land/oras-go/v2"

	"github.com/TensorRails/warmctl/v2/internal/manifest"
	"github.com/TensorRails/warmctl/v2/internal/store"
)

// A snapshot's files map onto layers three ways. A file larger than the
// piece size travels as consecutive piece layers: pieces transfer in
// parallel and stay under registries' blob and request limits and CDN
// caches (Cloudflare caches objects up to 512 MB). A run of consecutive
// files each under a quarter piece travels as one bundle layer, their bytes
// back to back: CRIU writes thousands of small images, and a layer each
// would swell the artifact manifest past registries' 4 MiB limit. Every
// other file is a layer of its own, whose digest is the file's SHA-256.
// Pieces are checked against their digests, bundled files against the
// SHA-256 manifest.json records.
const (
	PieceType  = "application/vnd.warmctl.file.piece"
	BundleType = "application/vnd.warmctl.file.bundle"
	// AnnotationPieceOf names the file a piece belongs to, and
	// AnnotationPieceOffset where in it the piece starts.
	AnnotationPieceOf     = "io.warmctl.piece.of"
	AnnotationPieceOffset = "io.warmctl.piece.offset"
	// AnnotationBundleOf names a bundle's first file, AnnotationBundleFiles
	// how many consecutive files of manifest.json it holds.
	AnnotationBundleOf    = "io.warmctl.bundle.of"
	AnnotationBundleFiles = "io.warmctl.bundle.files"
	// DefaultPieceSize keeps every layer under the 100 MB request limit of
	// a registry on Cloudflare Workers, so each goes in one request.
	DefaultPieceSize = 64 << 20
)

func (o Options) pieceSize() int64 {
	if o.PieceSize > 0 {
		return o.PieceSize
	}
	return DefaultPieceSize
}

// describe lists a snapshot's layers. Bundles and pieces are hashed here,
// each on its own worker, so one large file is hashed in parallel too,
// unless the digests file of an earlier push already records them.
func describe(ctx context.Context, src sources, o Options) ([]ocispec.Descriptor, error) {
	var out []ocispec.Descriptor
	var pending []int
	size := o.pieceSize()
	for i := 0; i < len(src.files); {
		f := src.files[i]
		if n, total := bundleRun(src.files[i:], size); n > 1 {
			pending = append(pending, len(out))
			out = append(out, ocispec.Descriptor{MediaType: BundleType, Size: total, Annotations: map[string]string{
				AnnotationBundleOf: f.Path, AnnotationBundleFiles: strconv.Itoa(n)}})
			i += n
			continue
		}
		i++
		if f.Size <= size {
			out = append(out, layer(f))
			continue
		}
		for off := int64(0); off < f.Size; off += size {
			pending = append(pending, len(out))
			out = append(out, ocispec.Descriptor{MediaType: PieceType, Size: min(size, f.Size-off),
				Annotations: map[string]string{AnnotationPieceOf: f.Path, AnnotationPieceOffset: strconv.FormatInt(off, 10)}})
		}
	}
	known := readDigests(src.dir)
	var todo []int
	for _, i := range pending {
		if dg := known[src.key(out[i])]; dg.Validate() == nil {
			out[i].Digest = dg
		} else {
			todo = append(todo, i)
		}
	}
	err := parallel(ctx, len(todo), o.workers(), func(i int) error {
		d := &out[todo[i]]
		sum, err := hashLayer(ctx, src.open(*d), d.Size)
		d.Digest = sum
		return err
	})
	if err == nil && len(todo) > 0 {
		if err := writeDigests(src, out, pending); err != nil {
			o.logf("digests not saved, the next push hashes again: %v", err)
		}
	}
	return out, err
}

// DigestsFile in a snapshot directory records the digests of its pieces and
// bundles, so that a repeat push does not read the snapshot to hash them.
// Each is keyed by what manifest.json records of its bytes, so a file whose
// SHA-256 changed, or another piece size, misses and is hashed again.
const DigestsFile = ".push-digests.json"

func readDigests(dir string) map[string]digest.Digest {
	var known map[string]digest.Digest
	raw, err := os.ReadFile(filepath.Join(dir, DigestsFile))
	if err != nil || json.Unmarshal(raw, &known) != nil {
		return nil // none or damaged: hash again
	}
	return known
}

// writeDigests replaces the digests file with those of layers[pending].
func writeDigests(src sources, layers []ocispec.Descriptor, pending []int) error {
	known := make(map[string]digest.Digest, len(pending))
	for _, i := range pending {
		known[src.key(layers[i])] = layers[i].Digest
	}
	raw, err := json.Marshal(known)
	if err != nil {
		return err
	}
	tmp, err := os.CreateTemp(src.dir, DigestsFile+"-*")
	if err != nil {
		return err
	}
	defer os.Remove(tmp.Name())
	_, err = tmp.Write(raw)
	if closeErr := tmp.Close(); err == nil {
		err = closeErr
	}
	if err != nil {
		return err
	}
	return os.Rename(tmp.Name(), filepath.Join(src.dir, DigestsFile))
}

// bundleRun counts the files at the head of files that go in one bundle:
// each under a quarter piece, together at most a piece.
func bundleRun(files []manifest.File, piece int64) (int, int64) {
	n, total := 0, int64(0)
	for _, f := range files {
		if f.Size >= piece/4 || total+f.Size > piece {
			break
		}
		n, total = n+1, total+f.Size
	}
	return n, total
}

func hashLayer(ctx context.Context, open func() (io.ReadCloser, error), size int64) (digest.Digest, error) {
	r, err := open()
	if err != nil {
		return "", err
	}
	defer r.Close()
	h := sha256.New()
	n, err := copyBuffered(ctx, h, r)
	if err == nil && n != size {
		err = fmt.Errorf("oci: read %d of %d bytes", n, size)
	}
	return digest.NewDigestFromBytes(digest.SHA256, h.Sum(nil)), err
}

// buffers are the copy buffers of every transfer: 4 MiB reads keep syscalls
// and hashing calls few at multi-GB/s rates without churning the heap.
var buffers = sync.Pool{New: func() any { b := make([]byte, 4<<20); return &b }}

func copyBuffered(ctx context.Context, w io.Writer, r io.Reader) (int64, error) {
	buf := buffers.Get().(*[]byte)
	defer buffers.Put(buf)
	var total int64
	for {
		if err := ctx.Err(); err != nil {
			return total, err
		}
		n, err := r.Read(*buf)
		if n > 0 {
			if _, werr := w.Write((*buf)[:n]); werr != nil {
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

// sources read layers' bytes from the snapshot in dir.
type sources struct {
	dir   string
	files []manifest.File
	at    map[string]int
}

func newSources(dir string, files []manifest.File) sources {
	at := make(map[string]int, len(files))
	for i, f := range files {
		at[f.Path] = i
	}
	return sources{dir, files, at}
}

func (s sources) path(p string) string { return filepath.Join(s.dir, filepath.FromSlash(p)) }

func (s sources) open(d ocispec.Descriptor) func() (io.ReadCloser, error) {
	switch d.MediaType {
	case PieceType:
		off, _ := strconv.ParseInt(d.Annotations[AnnotationPieceOffset], 10, 64)
		name := s.path(d.Annotations[AnnotationPieceOf])
		return func() (io.ReadCloser, error) {
			f, err := os.Open(name)
			if err != nil {
				return nil, err
			}
			return struct {
				io.Reader
				io.Closer
			}{io.NewSectionReader(f, off, d.Size), f}, nil
		}
	case BundleType:
		files := s.bundle(d)
		return func() (io.ReadCloser, error) { return &concat{s: s, files: files}, nil }
	default:
		name := s.path(d.Annotations[ocispec.AnnotationTitle])
		return func() (io.ReadCloser, error) { return os.Open(name) }
	}
}

func (s sources) bundle(d ocispec.Descriptor) []manifest.File {
	first := s.at[d.Annotations[AnnotationBundleOf]]
	n, _ := strconv.Atoi(d.Annotations[AnnotationBundleFiles])
	return s.files[first:min(first+n, len(s.files))]
}

// key names the bytes of a piece by its file's SHA-256 and range, those of
// a bundle by the SHA-256 of its files' SHA-256s.
func (s sources) key(d ocispec.Descriptor) string {
	if d.MediaType == PieceType {
		f := s.files[s.at[d.Annotations[AnnotationPieceOf]]]
		return fmt.Sprintf("%s@%s+%d", f.SHA256, d.Annotations[AnnotationPieceOffset], d.Size)
	}
	h := sha256.New()
	for _, f := range s.bundle(d) {
		h.Write([]byte(f.SHA256))
	}
	return "bundle:" + hex.EncodeToString(h.Sum(nil))
}

// concat reads files back to back, opening each when the last one ends.
type concat struct {
	s     sources
	files []manifest.File
	cur   *os.File
}

func (c *concat) Read(p []byte) (int, error) {
	for {
		if c.cur == nil {
			if len(c.files) == 0 {
				return 0, io.EOF
			}
			f, err := os.Open(c.s.path(c.files[0].Path))
			if err != nil {
				return 0, err
			}
			c.cur, c.files = f, c.files[1:]
		}
		n, err := c.cur.Read(p)
		if errors.Is(err, io.EOF) {
			c.cur.Close()
			c.cur = nil
			if n == 0 {
				continue
			}
			err = nil
		}
		return n, err
	}
}

func (c *concat) Close() error {
	if c.cur == nil {
		return nil
	}
	return c.cur.Close()
}

// fileLayers are the layers that carry files: one file in its own layer or
// in pieces, or the files of a bundle.
type fileLayers struct {
	files  []manifest.File
	layers []ocispec.Descriptor
}

// plan maps the artifact's layers onto the snapshot's files, in order.
func plan(m manifest.Manifest, layers []ocispec.Descriptor) ([]fileLayers, error) {
	var out []fileLayers
	i := 0
	for k := 0; k < len(m.Files); {
		var fl fileLayers
		var err error
		switch {
		case i < len(layers) && layers[i].MediaType == BundleType:
			fl, err = planBundle(m.Files[k:], layers[i])
			i++
		case i < len(layers) && layers[i].MediaType != PieceType:
			fl, err = planWhole(m.Files[k], layers[i])
			i++
		default:
			fl, i, err = planPieces(m.Files[k], layers, i)
		}
		if err != nil {
			return nil, err
		}
		out = append(out, fl)
		k += len(fl.files)
	}
	if i != len(layers) {
		return nil, fmt.Errorf("oci: %d layers for %d files", len(layers), len(m.Files))
	}
	return out, nil
}

func planWhole(f manifest.File, got ocispec.Descriptor) (fileLayers, error) {
	want := layer(f)
	if got.Digest != want.Digest || got.Size != want.Size || got.Annotations[ocispec.AnnotationTitle] != f.Path {
		return fileLayers{}, fmt.Errorf("oci: layer %s is not %s", got.Annotations[ocispec.AnnotationTitle], f.Path)
	}
	return fileLayers{[]manifest.File{f}, []ocispec.Descriptor{got}}, nil
}

func planBundle(files []manifest.File, d ocispec.Descriptor) (fileLayers, error) {
	n, err := strconv.Atoi(d.Annotations[AnnotationBundleFiles])
	if err != nil || n < 1 || n > len(files) || d.Annotations[AnnotationBundleOf] != files[0].Path {
		return fileLayers{}, fmt.Errorf("oci: bundle of %q files from %s does not follow manifest.json",
			d.Annotations[AnnotationBundleFiles], d.Annotations[AnnotationBundleOf])
	}
	total := int64(0)
	for _, f := range files[:n] {
		total += f.Size
	}
	if total != d.Size {
		return fileLayers{}, fmt.Errorf("oci: bundle from %s holds %d bytes, its files %d", files[0].Path, d.Size, total)
	}
	return fileLayers{files[:n], []ocispec.Descriptor{d}}, nil
}

// planPieces takes the consecutive pieces of f from layers[i:], which must
// cover it exactly, and returns the index past them.
func planPieces(f manifest.File, layers []ocispec.Descriptor, i int) (fileLayers, int, error) {
	start, off := i, int64(0)
	for i < len(layers) && layers[i].MediaType == PieceType && layers[i].Annotations[AnnotationPieceOf] == f.Path {
		if layers[i].Size <= 0 || layers[i].Annotations[AnnotationPieceOffset] != strconv.FormatInt(off, 10) {
			return fileLayers{}, i, fmt.Errorf("oci: piece %d of %s does not start at %d", i-start, f.Path, off)
		}
		off += layers[i].Size
		i++
	}
	if i == start || off != f.Size {
		return fileLayers{}, i, fmt.Errorf("oci: layers cover %d of the %d bytes of %s", off, f.Size, f.Path)
	}
	return fileLayers{[]manifest.File{f}, layers[start:i]}, i, nil
}

// fetchBundle splits a bundle into its files and stores those in want, each
// checked against its SHA-256 in manifest.json; the others are skipped.
func fetchBundle(ctx context.Context, target oras.ReadOnlyTarget, d ocispec.Descriptor, files []manifest.File,
	want map[string]bool, has func(sha string) bool, write func(sha string, r io.Reader, size int64) error) error {
	rc, err := target.Fetch(ctx, d)
	if err != nil {
		return err
	}
	defer rc.Close()
	r := bufio.NewReaderSize(rc, 1<<20)
	stored := map[string]bool{}
	for _, f := range files {
		part := io.LimitReader(r, f.Size)
		// Read past a file stored already: by an earlier attempt of this
		// bundle, or as a second copy within it.
		if want[f.SHA256] && !stored[f.SHA256] && !has(f.SHA256) {
			stored[f.SHA256] = true
			if err := write(f.SHA256, part, f.Size); err != nil {
				return err
			}
			continue
		}
		if n, err := io.Copy(io.Discard, part); err != nil || n != f.Size {
			return fmt.Errorf("oci: bundle %s ends inside %s: %v", d.Digest, f.Path, err)
		}
	}
	if n, _ := r.Read(make([]byte, 1)); n != 0 {
		return fmt.Errorf("oci: bundle %s is longer than its %d files", d.Digest, len(files))
	}
	return nil
}

// assembly is a file being written from its pieces next to the store's
// blobs, so that it becomes a blob by a link. It outlives a failed pull:
// .assemble-<sha256> keeps the bytes and .assemble-<sha256>.pieces lists the
// pieces written and checked, so that the next pull fetches only the rest.
type assembly struct {
	file    manifest.File
	tmp     *os.File
	journal *os.File
	mu      sync.Mutex
	done    map[int64]bool // offsets an earlier pull left in tmp
	left    atomic.Int64   // pieces still to write, and one while they are scheduled
}

func newAssembly(ctx context.Context, s store.Store, f manifest.File, pieces []ocispec.Descriptor) (*assembly, error) {
	if err := os.MkdirAll(s.Blobs(), 0o700); err != nil {
		return nil, err
	}
	name := filepath.Join(s.Blobs(), ".assemble-"+f.SHA256)
	// A pull that failed after the file was whole left it read-only.
	_ = os.Chmod(name, 0o600)
	tmp, err := os.OpenFile(name, os.O_RDWR|os.O_CREATE, 0o600)
	if err != nil {
		return nil, err
	}
	// Two pulls on one host may want the same file; one assembles it.
	if err := lockFile(tmp); err != nil {
		tmp.Close()
		return nil, fmt.Errorf("oci: another pull is assembling %s: %w", f.Path, err)
	}
	if err := preallocate(tmp, f.Size); err != nil {
		tmp.Close()
		return nil, err
	}
	a := &assembly{file: f, tmp: tmp, done: map[int64]bool{}}
	a.resume(ctx, pieces)
	if a.journal, err = os.OpenFile(name+".pieces", os.O_WRONLY|os.O_CREATE|os.O_APPEND, 0o600); err != nil {
		tmp.Close()
		return nil, err
	}
	return a, nil
}

// resume takes the pieces an earlier pull recorded, each only once its
// bytes in tmp hash to its digest again: the journal is written without
// syncing the data, so a crash can leave a recorded piece unwritten.
func (a *assembly) resume(ctx context.Context, pieces []ocispec.Descriptor) {
	raw, err := os.ReadFile(a.tmp.Name() + ".pieces")
	if err != nil {
		return
	}
	recorded := map[string]bool{}
	for _, line := range strings.Fields(string(raw)) {
		recorded[line] = true
	}
	for _, d := range pieces {
		off := pieceOffset(d)
		if !recorded[pieceKey(d)] {
			continue
		}
		open := func() (io.ReadCloser, error) { return io.NopCloser(io.NewSectionReader(a.tmp, off, d.Size)), nil }
		if sum, err := hashLayer(ctx, open, d.Size); err == nil && sum == d.Digest {
			a.done[off] = true
		}
	}
}

func pieceOffset(d ocispec.Descriptor) int64 {
	off, _ := strconv.ParseInt(d.Annotations[AnnotationPieceOffset], 10, 64)
	return off
}

func pieceKey(d ocispec.Descriptor) string {
	return fmt.Sprintf("%s@%d", d.Digest.Encoded(), pieceOffset(d))
}

// fetchPiece writes one piece at its offset, checking it against its
// digest, and records it.
func (a *assembly) fetchPiece(ctx context.Context, target oras.ReadOnlyTarget, d ocispec.Descriptor) error {
	off, err := strconv.ParseInt(d.Annotations[AnnotationPieceOffset], 10, 64)
	if err != nil {
		return err
	}
	rc, err := target.Fetch(ctx, d)
	if err != nil {
		return err
	}
	defer rc.Close()
	h := sha256.New()
	n, err := copyBuffered(ctx, io.MultiWriter(io.NewOffsetWriter(a.tmp, off), h), io.LimitReader(rc, d.Size+1))
	if err != nil {
		return err
	}
	if got := "sha256:" + hex.EncodeToString(h.Sum(nil)); got != d.Digest.String() || n != d.Size {
		return fmt.Errorf("oci: piece of %s at %d arrived as %s (%d of %d bytes)", a.file.Path, off, got, n, d.Size)
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	_, err = fmt.Fprintln(a.journal, pieceKey(d))
	return err
}

// finish makes the assembled file the blob of its SHA-256 once the pull has
// synced it: every byte was checked against a piece digest, and the pieces
// cover the file exactly.
//
// The file stays open, and so locked, until it is linked into the store
// and its name removed.
func (a *assembly) finish(ctx context.Context, s store.Store) error {
	name := a.tmp.Name()
	defer a.tmp.Close()
	a.journal.Close()
	if err := s.AddHashed(ctx, name, a.file.SHA256); err != nil {
		return err
	}
	os.Remove(name + ".pieces")
	return os.Remove(name)
}

// abort leaves the file and its journal for the next pull to resume.
func (a *assembly) abort() {
	a.journal.Close()
	a.tmp.Close()
}

// saveDigests records the pulled snapshot's piece and bundle digests in its
// digests file, so that pushing it on, to a cache or a mirror, hashes
// nothing. A write that fails only costs that push the hashing.
func saveDigests(dir string, files []manifest.File, layers []ocispec.Descriptor) {
	var pending []int
	for i, d := range layers {
		if d.MediaType == PieceType || d.MediaType == BundleType {
			pending = append(pending, i)
		}
	}
	if len(pending) > 0 {
		_ = writeDigests(newSources(dir, files), layers, pending)
	}
}
