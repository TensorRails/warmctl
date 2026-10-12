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
	Log   func(format string, args ...any)
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
	o Options) (manifest.Manifest, error) {
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
	tmp, err := os.MkdirTemp(filepath.Dir(output), ".pull-")
	if err != nil {
		return m, err
	}
	defer os.RemoveAll(tmp)
	got := &tally{}
	if err := download(ctx, target, s, files, got, o); err != nil {
		return m, err
	}
	for _, f := range m.Files {
		if err := link(ctx, s, f, tmp); err != nil {
			return m, err
		}
	}
	saveDigests(tmp, m.Files, am.Layers)
	if err := os.WriteFile(filepath.Join(tmp, "manifest.json"), config, 0o644); err != nil {
		return m, err
	}
	if err := os.Chmod(tmp, 0o755); err != nil {
		return m, err
	}
	if err := os.Rename(tmp, output); err != nil {
		return m, err
	}
	o.logf("pulled %s into %s: %d files in %d blobs, %d downloaded, %.2f GiB", desc.Digest, output,
		len(m.Files), len(distinct(am.Layers)), got.blobs,
		float64(got.bytes)/(1<<30))
	return m, nil
}

// download brings every file the store lacks: whole-file blobs stream into
// the store, bundles split into it, pieces go into their file's assembly,
// all at once.
func download(ctx context.Context, target oras.Target, s store.Store, files []fileLayers, got *tally,
	o Options) error {
	dl := &downloader{ctx: ctx, target: stallGuard{target, stallAfter}, s: s, got: got, seen: map[string]bool{}}
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
	for len(dl.asms) > 0 {
		if err := dl.asms[0].finish(ctx, s); err != nil {
			return err
		}
		dl.asms = dl.asms[1:]
	}
	return nil
}

type downloader struct {
	ctx     context.Context
	target  oras.ReadOnlyTarget
	s       store.Store
	got     *tally
	seen    map[string]bool
	jobs    []func() error
	asms    []*assembly
	total   int64
	resumed int64
}

// need claims a file the store lacks, once per SHA-256.
func (dl *downloader) need(f manifest.File) bool {
	if dl.seen[f.SHA256] || dl.s.Has(f.SHA256) {
		return false
	}
	dl.seen[f.SHA256] = true
	return true
}

// add queues a transfer of a blob of size, counted once it is done.
func (dl *downloader) add(size int64, fn func() error) {
	dl.total += size
	dl.jobs = append(dl.jobs, func() error {
		err := retryBlob(dl.ctx, fn)
		if err == nil {
			dl.got.add(size)
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
	d := fl.layers[0]
	switch d.MediaType {
	case BundleType:
		want := map[string]bool{}
		for _, f := range fl.files {
			if dl.need(f) {
				want[f.SHA256] = true
			}
		}
		if len(want) > 0 {
			dl.add(d.Size, func() error { return fetchBundle(dl.ctx, dl.target, dl.s, d, fl.files, want) })
		}
	case PieceType:
		if !dl.need(fl.files[0]) {
			return nil
		}
		a, err := newAssembly(dl.ctx, dl.s, fl.files[0], fl.layers)
		if err != nil {
			return err
		}
		dl.asms = append(dl.asms, a)
		for _, d := range fl.layers {
			if a.done[pieceOffset(d)] {
				dl.resumed += d.Size
				continue
			}
			dl.add(d.Size, func() error { return a.fetchPiece(dl.ctx, dl.target, d) })
		}
	default:
		if dl.need(fl.files[0]) {
			dl.add(d.Size, func() error { return fetch(dl.ctx, dl.target, dl.s, d) })
		}
	}
	return nil
}

func (dl *downloader) abort() {
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
func fetch(ctx context.Context, target oras.ReadOnlyTarget, s store.Store, d ocispec.Descriptor) error {
	rc, err := target.Fetch(ctx, d)
	if err != nil {
		return err
	}
	defer rc.Close()
	return s.Ingest(ctx, rc, d.Digest.Encoded(), d.Size)
}

// link places a stored file at its path under dir.
func link(ctx context.Context, s store.Store, f manifest.File, dir string) error {
	if f.Path == "" || path.IsAbs(f.Path) || path.Clean(f.Path) != f.Path || strings.HasPrefix(f.Path, "../") {
		return fmt.Errorf("oci: file path %q", f.Path)
	}
	return s.Link(ctx, f.SHA256, filepath.Join(dir, filepath.FromSlash(f.Path)))
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
