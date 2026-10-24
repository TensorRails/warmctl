// Package oci moves snapshots through an OCI registry. A file of a snapshot
// is one blob whose digest is the SHA-256 the manifest already records, or,
// above the piece size, consecutive piece blobs (pieces.go). Blobs two
// snapshots share are stored and transferred once, and a pull checks every
// blob as it streams in. manifest.json is the artifact's config.
package oci

import (
	"bytes"
	"context"
	"crypto/tls"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/opencontainers/go-digest"
	ocispec "github.com/opencontainers/image-spec/specs-go/v1"
	"oras.land/oras-go/v2"
	"oras.land/oras-go/v2/content"
	"oras.land/oras-go/v2/errdef"
	"oras.land/oras-go/v2/registry/remote"
	"oras.land/oras-go/v2/registry/remote/auth"
	"oras.land/oras-go/v2/registry/remote/credentials"
	"oras.land/oras-go/v2/registry/remote/retry"

	"github.com/TensorRails/warmctl/v2/internal/manifest"
	"github.com/TensorRails/warmctl/v2/internal/store"
)

// Media types of a warmctl snapshot artifact.
const (
	ArtifactType = "application/vnd.warmctl.snapshot.v2"
	ConfigType   = "application/vnd.warmctl.snapshot.manifest.v2+json"
	PartType     = "application/vnd.warmctl.weights.part"
	FileType     = "application/vnd.warmctl.file"
	// Scheme prefixes a registry reference on the command line.
	Scheme = "oci://"
)

// Options of a transfer.
type Options struct {
	PlainHTTP   bool
	Concurrency int
	// ChunkSize uploads blobs larger than it in chunks of that size. Zero
	// sends every blob whole, and DefaultChunkSize chunks once a registry
	// refuses a whole blob as too large.
	ChunkSize int64
	// PieceSize splits files larger than it into pieces of that size; zero
	// means DefaultPieceSize.
	PieceSize int64
	// Check, when set, vets the snapshot's manifest after a pull resolves
	// it and before any blob is downloaded: a host that cannot restore it
	// should not fetch hundreds of GiB first.
	Check func(manifest.Manifest) error
	// Arrivals, when set, makes a pull write the output directory in place,
	// file by file, so that a restore can start before the pull ends.
	Arrivals *Arrivals
	Log      func(format string, args ...any)
}

func (o Options) logf(format string, args ...any) {
	if o.Log != nil {
		o.Log(format, args...)
	}
}

func (o Options) workers() int {
	if o.Concurrency > 0 {
		return o.Concurrency
	}
	return 64
}

// sharedTransport is the one transport of the process, so that connections
// to a registry are reused across its repositories.
var sharedTransport = sync.OnceValue(transport)

// transport is tuned for bulk blob transfer. HTTP/1.1, not HTTP/2: HTTP/2
// multiplexes every parallel stream to a host over one TCP connection,
// which caps a transfer at one connection's throughput. Connections stay
// pooled between blobs, so TLS is set up once per worker, and 1 MiB socket
// buffers keep a few large writes per request.
func transport() http.RoundTripper {
	t := http.DefaultTransport.(*http.Transport).Clone()
	t.ForceAttemptHTTP2 = false
	t.TLSNextProto = map[string]func(string, *tls.Conn) http.RoundTripper{}
	// Offer only HTTP/1.1 in ALPN, or an HTTP/2 server answers in frames.
	if t.TLSClientConfig == nil {
		t.TLSClientConfig = &tls.Config{}
	} else {
		t.TLSClientConfig = t.TLSClientConfig.Clone()
	}
	t.TLSClientConfig.NextProtos = []string{"http/1.1"}
	t.MaxIdleConns, t.MaxIdleConnsPerHost = 1024, 1024
	t.ReadBufferSize, t.WriteBufferSize = 1<<20, 1<<20
	t.DisableCompression = true
	return retry.NewTransport(t)
}

// IsRef reports an oci:// reference.
func IsRef(s string) bool { return strings.HasPrefix(s, Scheme) }

// Repository opens the repository of an oci:// reference with the Docker
// credentials of this user, and returns the tag or digest it names.
func Repository(ref string, o Options) (*remote.Repository, string, error) {
	name := strings.TrimPrefix(ref, Scheme)
	repo, err := remote.NewRepository(name)
	if err != nil {
		return nil, "", fmt.Errorf("oci: %w", err)
	}
	repo.PlainHTTP = o.PlainHTTP
	client := &auth.Client{Client: &http.Client{Transport: sharedTransport()}, Cache: auth.NewCache()}
	if creds, err := credentials.NewStoreFromDocker(credentials.StoreOptions{}); err == nil {
		client.Credential = credentials.Credential(creds)
	}
	repo.Client = client
	return repo, repo.Reference.Reference, nil
}

// Push uploads the snapshot in dir and tags it.
func Push(ctx context.Context, target oras.Target, dir, tag string, o Options) (ocispec.Descriptor, error) {
	raw, err := os.ReadFile(filepath.Join(dir, "manifest.json"))
	if err != nil {
		return ocispec.Descriptor{}, err
	}
	m, err := manifest.Read(filepath.Join(dir, "manifest.json"))
	if err != nil {
		return ocispec.Descriptor{}, err
	}
	src := newSources(dir, m.Files)
	layers, err := describe(ctx, src, o)
	if err != nil {
		return ocispec.Descriptor{}, err
	}
	p := newPusher(target, o)
	sent, err := upload(ctx, p, src, layers, o)
	if err != nil {
		return ocispec.Descriptor{}, err
	}
	config := content.NewDescriptorFromBytes(ConfigType, raw)
	open := func() (io.ReadCloser, error) { return io.NopCloser(bytes.NewReader(raw)), nil }
	if _, err := pushIfMissing(ctx, p, config, open); err != nil {
		return ocispec.Descriptor{}, err
	}
	desc, err := oras.PackManifest(ctx, target, oras.PackManifestVersion1_1, ArtifactType, oras.PackManifestOptions{
		Layers: layers, ConfigDescriptor: &config,
		ManifestAnnotations: map[string]string{ocispec.AnnotationCreated: m.CreatedAt.UTC().Format(time.RFC3339)},
	})
	if err != nil {
		return ocispec.Descriptor{}, err
	}
	if tag != "" {
		if err := target.Tag(ctx, desc, tag); err != nil {
			return ocispec.Descriptor{}, err
		}
	}
	o.logf("pushed %s: %d files in %d blobs, %d uploaded, %.2f GiB sent", desc.Digest, len(m.Files),
		len(distinct(layers)), sent.blobs,
		float64(sent.bytes)/(1<<30))
	return desc, nil
}

func layer(f manifest.File) ocispec.Descriptor {
	media := FileType
	if f.Role == manifest.RoleWeights {
		media = PartType
	}
	return ocispec.Descriptor{
		MediaType: media, Digest: digest.Digest("sha256:" + f.SHA256), Size: f.Size,
		Annotations: map[string]string{ocispec.AnnotationTitle: f.Path},
	}
}

type tally struct {
	mu           sync.Mutex
	blobs, bytes int64
}

func (t *tally) add(n int64) {
	t.mu.Lock()
	t.blobs++
	t.bytes += n
	t.mu.Unlock()
}

func upload(ctx context.Context, p *pusher, src sources, layers []ocispec.Descriptor,
	o Options) (*tally, error) {
	sent, checked := &tally{}, &tally{}
	unique := distinct(layers)
	var total int64
	for _, d := range unique {
		total += d.Size
	}
	defer meter(o, "pushed", total, checked)()
	err := parallel(ctx, len(unique), o.workers(), func(i int) error {
		d := unique[i]
		uploaded, err := pushIfMissing(ctx, p, d, src.open(d))
		if uploaded {
			sent.add(d.Size)
		}
		if err == nil {
			checked.add(d.Size)
		}
		return err
	})
	return sent, err
}

// distinct keeps the first layer of every digest: two ranks can export
// byte-identical parts, and one transfer serves both.
func distinct(layers []ocispec.Descriptor) []ocispec.Descriptor {
	seen := map[digest.Digest]bool{}
	var out []ocispec.Descriptor
	for _, d := range layers {
		if !seen[d.Digest] {
			seen[d.Digest] = true
			out = append(out, d)
		}
	}
	return out
}

// pushIfMissing uploads a blob unless the target has it.
func pushIfMissing(ctx context.Context, p *pusher, d ocispec.Descriptor,
	open func() (io.ReadCloser, error)) (bool, error) {
	if ok, err := p.target.Exists(ctx, d); err != nil || ok {
		return false, err
	}
	err := p.push(ctx, d, open)
	if errors.Is(err, errdef.ErrAlreadyExists) {
		return false, nil
	}
	return err == nil, err
}

// Pull fetches the snapshot ref names into the store and links it into
// output, which must not exist. It returns the snapshot's manifest.
func Pull(ctx context.Context, target oras.Target, ref, output string, s store.Store,
	o Options) (m manifest.Manifest, err error) {
	defer func() { o.Arrivals.end(err) }()
	desc, am, m, config, err := resolve(ctx, target, ref)
	if err != nil {
		return m, err
	}
	files, err := plan(m, am.Layers)
	if err != nil {
		return m, err
	}
	if o.Check != nil {
		if err := o.Check(m); err != nil {
			return m, err
		}
	}
	if err := roomFor(s, m.Files); err != nil {
		return m, err
	}
	if err := os.MkdirAll(filepath.Dir(output), 0o755); err != nil {
		return m, err
	}
	release, err := claimOutput(output)
	if err != nil {
		return m, err
	}
	defer release()
	// A pull that a restore follows writes the snapshot's directory in
	// place; any other writes a temporary one and renames it at the end.
	dir := output
	if o.Arrivals == nil {
		dir, err = os.MkdirTemp(filepath.Dir(output), ".pull-")
	} else {
		err = os.Mkdir(output, 0o755)
	}
	if err != nil {
		return m, err
	}
	pulled := false
	defer func() {
		if !pulled {
			os.RemoveAll(dir)
		}
	}()
	o.Arrivals.begin(output, m, config)
	got := &tally{}
	if err := download(ctx, target, s, arrivalOrder(m, files), dir, got, o); err != nil {
		return m, err
	}
	saveDigests(dir, m.Files, am.Layers)
	if err := complete(dir, output, config); err != nil {
		return m, err
	}
	pulled = true
	o.logf("pulled %s into %s: %d files in %d blobs, %d downloaded, %.2f GiB", desc.Digest, output,
		len(m.Files), len(distinct(am.Layers)), got.blobs,
		float64(got.bytes)/(1<<30))
	return m, nil
}

// download brings every file the store lacks into it, all at once: whole-file
// blobs stream into the store, bundles split into it, pieces go into their
// file's assembly. Each file is linked into dir as soon as its blob is
// stored, in the order of files.
func download(ctx context.Context, target oras.Target, s store.Store, files []fileLayers, dir string, got *tally,
	o Options) error {
	dl := &downloader{ctx: ctx, target: stallGuard{target, stallAfter}, s: s, dir: dir, arrivals: o.Arrivals,
		got: got, seen: map[string]bool{}, waiting: map[string][]manifest.File{}, written: map[string]string{}}
	defer dl.abort()
	for _, fl := range files {
		if err := dl.schedule(fl); err != nil {
			return err
		}
	}
	if dl.resumed > 0 {
		o.logf("resuming: %.2f GiB of pieces are left from an earlier pull", gib(dl.resumed))
	}
	stop := meter(o, "pulled", dl.total, got)
	err := parallel(ctx, len(dl.jobs), o.workers(), func(i int) error { return dl.jobs[i]() })
	stop()
	if err != nil {
		return err
	}
	start := time.Now()
	if err := dl.commit(); err != nil {
		return err
	}
	o.logf("synced %.2f GiB to disk in %.1fs", gib(got.bytes), time.Since(start).Seconds())
	return nil
}

// complete writes manifest.json, which makes dir a snapshot, and moves a
// temporary dir into place as output.
func complete(dir, output string, config []byte) error {
	if err := os.WriteFile(filepath.Join(dir, "manifest.json"), config, 0o644); err != nil {
		return err
	}
	if dir == output {
		return nil
	}
	if err := os.Chmod(dir, 0o755); err != nil {
		return err
	}
	return os.Rename(dir, output)
}

type downloader struct {
	ctx      context.Context
	target   oras.ReadOnlyTarget
	s        store.Store
	dir      string
	arrivals *Arrivals
	got      *tally
	seen     map[string]bool
	jobs     []func() error
	asms     []*assembly
	total    int64
	resumed  int64
	mu       sync.Mutex
	waiting  map[string][]manifest.File // by SHA-256: files to link once their bytes are checked
	written  map[string]string          // by SHA-256: files written and checked, for the store to claim
	finished []*assembly                // assembled and checked, for the store to claim
}

// A pull syncs nothing until every blob is written, then the filesystem
// once: a sync per blob would make each wait for the writeback of all the
// others (ext4 commits in order), and with them any other process's, such
// as `docker create` for a receiver that starts during the pull. The store
// claims a blob only after that sync, so it never holds a blob that is not
// on disk.

// write streams a blob's bytes next to the store, checked, and links the
// files waiting for them. A second copy, from a retried bundle, is dropped.
func (dl *downloader) write(sha string, r io.Reader, size int64) error {
	name, err := dl.s.Write(dl.ctx, r, sha, size)
	if err != nil {
		return err
	}
	dl.mu.Lock()
	_, dup := dl.written[sha]
	if !dup {
		dl.written[sha] = name
	}
	dl.mu.Unlock()
	if dup {
		return os.Remove(name)
	}
	return dl.checked(sha, name)
}

// has reports a blob this pull wrote or the store holds.
func (dl *downloader) has(sha string) bool {
	dl.mu.Lock()
	_, ok := dl.written[sha]
	dl.mu.Unlock()
	return ok || dl.s.Has(sha)
}

// commit syncs what the pull wrote, makes it blobs of the store, and links
// the files that waited for the store.
func (dl *downloader) commit() error {
	if err := dl.s.Sync(); err != nil {
		return err
	}
	for sha, name := range dl.written {
		if err := dl.s.Claim(name, sha); err != nil {
			return err
		}
		delete(dl.written, sha)
	}
	for len(dl.finished) > 0 {
		if err := dl.finished[0].finish(dl.ctx, dl.s); err != nil {
			return err
		}
		dl.finished = dl.finished[1:]
	}
	for sha := range dl.waiting {
		if err := dl.stored(sha); err != nil {
			return err
		}
	}
	for sha, fs := range dl.waiting {
		return fmt.Errorf("oci: %s (%s) was never stored", fs[0].Path, sha)
	}
	return nil
}

// need claims a file the store lacks, once per SHA-256. A file the store
// has is linked at once; any other waits for its blob.
func (dl *downloader) need(f manifest.File) (bool, error) {
	if !dl.seen[f.SHA256] && dl.s.Has(f.SHA256) {
		return false, dl.place(f)
	}
	dl.waiting[f.SHA256] = append(dl.waiting[f.SHA256], f)
	if dl.seen[f.SHA256] {
		return false, nil
	}
	dl.seen[f.SHA256] = true
	return true, nil
}

// checked links the files waiting for sha to name, which holds their bytes,
// checked, before the pull syncs them and the store claims them: a restore
// that follows the pull reads them from memory while the disk catches up. A
// file it cannot link there, on another filesystem, waits for the blob
// (stored).
func (dl *downloader) checked(sha, name string) error {
	dl.mu.Lock()
	files := dl.waiting[sha]
	delete(dl.waiting, sha)
	dl.mu.Unlock()
	var left []manifest.File
	for _, f := range files {
		dst, err := under(dl.dir, f.Path)
		if err != nil {
			return err
		}
		if err := os.MkdirAll(filepath.Dir(dst), 0o700); err != nil {
			return err
		}
		if os.Link(name, dst) != nil {
			left = append(left, f)
			continue
		}
		dl.arrivals.arrived(f.Path)
	}
	if len(left) > 0 {
		dl.mu.Lock()
		dl.waiting[sha] = append(dl.waiting[sha], left...)
		dl.mu.Unlock()
	}
	return nil
}

// stored links the files still waiting for the blob of sha, once it is one.
func (dl *downloader) stored(sha string) error {
	dl.mu.Lock()
	files := dl.waiting[sha]
	delete(dl.waiting, sha)
	dl.mu.Unlock()
	for _, f := range files {
		if err := dl.place(f); err != nil {
			return err
		}
	}
	return nil
}

// place links a stored file into the snapshot directory.
func (dl *downloader) place(f manifest.File) error {
	if err := link(dl.ctx, dl.s, f, dl.dir); err != nil {
		return err
	}
	dl.arrivals.arrived(f.Path)
	return nil
}

// add queues a transfer of a blob of size, counted once it is done, and
// then done after it.
func (dl *downloader) add(size int64, fn, then func() error) {
	dl.total += size
	dl.jobs = append(dl.jobs, func() error {
		err := retryBlob(dl.ctx, fn)
		if err == nil {
			dl.got.add(size)
			err = then()
		}
		return err
	})
}

// retryBlob runs a blob's transfer again after a transient failure: a 5xx,
// a reset or stalled connection, a body cut short. Each attempt starts the
// blob over; a piece is rewritten in place, a bundle skips the files it
// stored.
func retryBlob(ctx context.Context, fn func() error) error {
	for failures := 0; ; {
		err := fn()
		if err == nil || ctx.Err() != nil || !retryable(err) {
			return err
		}
		if failures++; failures > chunkRetries {
			return err
		}
		if err := sleep(ctx, chunkBackoff<<(failures-1)); err != nil {
			return err
		}
	}
}

func (dl *downloader) schedule(fl fileLayers) error {
	switch fl.layers[0].MediaType {
	case BundleType:
		return dl.scheduleBundle(fl)
	case PieceType:
		return dl.schedulePieces(fl)
	}
	d, f := fl.layers[0], fl.files[0]
	if need, err := dl.need(f); err != nil || !need {
		return err
	}
	dl.add(d.Size, func() error { return fetch(dl.ctx, dl.target, d, dl.write) }, func() error { return nil })
	return nil
}

func (dl *downloader) scheduleBundle(fl fileLayers) error {
	want := map[string]bool{}
	for _, f := range fl.files {
		need, err := dl.need(f)
		if err != nil {
			return err
		}
		if need {
			want[f.SHA256] = true
		}
	}
	if len(want) == 0 {
		return nil
	}
	d := fl.layers[0]
	dl.add(d.Size, func() error { return fetchBundle(dl.ctx, dl.target, d, fl.files, want, dl.has, dl.write) },
		func() error { return nil })
	return nil
}

func (dl *downloader) schedulePieces(fl fileLayers) error {
	f := fl.files[0]
	if need, err := dl.need(f); err != nil || !need {
		return err
	}
	a, err := newAssembly(dl.ctx, dl.s, f, fl.layers)
	if err != nil {
		return err
	}
	dl.asms = append(dl.asms, a)
	// The job that writes the file's last piece hands it on, checked; a
	// file an earlier pull left whole is handed on now.
	finish := func() error {
		if a.left.Add(-1) > 0 {
			return nil
		}
		dl.mu.Lock()
		dl.finished = append(dl.finished, a)
		dl.mu.Unlock()
		return dl.checked(f.SHA256, a.tmp.Name())
	}
	a.left.Store(1)
	for _, d := range fl.layers {
		if a.done[pieceOffset(d)] {
			dl.resumed += d.Size
			continue
		}
		a.left.Add(1)
		dl.add(d.Size, func() error { return a.fetchPiece(dl.ctx, dl.target, d) }, finish)
	}
	return finish()
}

// abort keeps what a failed pull got for the next one: the blobs written,
// once synced, become the store's, and each assembly stays to resume.
func (dl *downloader) abort() {
	synced := len(dl.written) > 0 && dl.s.Sync() == nil
	for sha, name := range dl.written {
		if !synced || dl.s.Claim(name, sha) != nil {
			os.Remove(name)
		}
	}
	for _, a := range dl.asms {
		a.abort()
	}
}

// resolve fetches and checks the artifact manifest and its config, the
// snapshot's manifest.json.
func resolve(ctx context.Context, target oras.ReadOnlyTarget, ref string) (ocispec.Descriptor, ocispec.Manifest,
	manifest.Manifest, []byte, error) {
	var am ocispec.Manifest
	var m manifest.Manifest
	desc, err := target.Resolve(ctx, ref)
	if err != nil {
		return desc, am, m, nil, fmt.Errorf("oci: resolve %s: %w", ref, err)
	}
	raw, err := content.FetchAll(ctx, target, desc)
	if err != nil {
		return desc, am, m, nil, err
	}
	if err := json.Unmarshal(raw, &am); err != nil {
		return desc, am, m, nil, err
	}
	if am.ArtifactType != ArtifactType || am.Config.MediaType != ConfigType {
		return desc, am, m, nil, fmt.Errorf("oci: %s is a %q, not a warmctl snapshot", ref, am.ArtifactType)
	}
	config, err := content.FetchAll(ctx, target, am.Config)
	if err != nil {
		return desc, am, m, nil, err
	}
	if err := json.Unmarshal(config, &m); err != nil {
		return desc, am, m, nil, err
	}
	if err := m.Validate(); err != nil {
		return desc, am, m, nil, err
	}
	_, err = plan(m, am.Layers)
	return desc, am, m, config, err
}

// fetch streams a blob into the store.
// fetch hands a whole-file blob to write.
func fetch(ctx context.Context, target oras.ReadOnlyTarget, d ocispec.Descriptor,
	write func(sha string, r io.Reader, size int64) error) error {
	rc, err := target.Fetch(ctx, d)
	if err != nil {
		return err
	}
	defer rc.Close()
	return write(d.Digest.Encoded(), rc, d.Size)
}

// link places a stored file at its path under dir.
func link(ctx context.Context, s store.Store, f manifest.File, dir string) error {
	dst, err := under(dir, f.Path)
	if err != nil {
		return err
	}
	return s.Link(ctx, f.SHA256, dst)
}

// under is where a snapshot file of path p lives in dir.
func under(dir, p string) (string, error) {
	if p == "" || path.IsAbs(p) || path.Clean(p) != p || strings.HasPrefix(p, "../") {
		return "", fmt.Errorf("oci: file path %q", p)
	}
	return filepath.Join(dir, filepath.FromSlash(p)), nil
}

// Tags lists a repository's tags.
func Tags(ctx context.Context, repo *remote.Repository) ([]string, error) {
	var out []string
	err := repo.Tags(ctx, "", func(tags []string) error {
		out = append(out, tags...)
		return nil
	})
	return out, err
}

func parallel(ctx context.Context, n, workers int, fn func(int) error) error {
	jobs := make(chan int)
	errs := make([]error, n)
	var wg sync.WaitGroup
	for w := 0; w < min(workers, max(n, 1)); w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range jobs {
				errs[i] = fn(i)
			}
		}()
	}
	for i := 0; i < n && ctx.Err() == nil; i++ {
		jobs <- i
	}
	close(jobs)
	wg.Wait()
	if err := ctx.Err(); err != nil {
		return err
	}
	return errors.Join(errs...)
}
