package oci

// Chunked blob uploads, OCI distribution spec v1.1 "Pushing a blob in
// chunks": POST opens an upload session, every PATCH appends one
// Content-Range of the blob, and PUT ?digest= closes it. A registry behind
// a request size limit needs it for weights parts of several GiB (an R2
// registry on Cloudflare Workers takes 100-500 MB per request). Every other
// registry keeps the monolithic upload oras-go does, unless the user asks
// for chunks or the registry refuses a whole blob as too large (413).

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"sync/atomic"
	"time"

	ocispec "github.com/opencontainers/image-spec/specs-go/v1"
	"oras.land/oras-go/v2"
	"oras.land/oras-go/v2/errdef"
	"oras.land/oras-go/v2/registry/remote"
	"oras.land/oras-go/v2/registry/remote/auth"
	"oras.land/oras-go/v2/registry/remote/errcode"
)

// DefaultChunkSize is the chunk a push falls back to after a registry
// refused a whole blob: under the smallest Workers request limit, above the
// 5 MiB minimum part of S3-style multipart stores.
const DefaultChunkSize = 64 << 20

var (
	fallbackChunk int64 = DefaultChunkSize
	chunkRetries        = 5
	chunkBackoff        = time.Second
	errLocation         = errors.New("oci: unusable upload Location")
)

// pusher uploads blobs to a target, in chunks where the registry needs them.
type pusher struct {
	target  oras.Target
	repo    *remote.Repository // nil when the target is not a registry
	chunk   int64
	chunked atomic.Bool // a registry refused a whole blob: chunk the rest
	o       Options
}

func newPusher(target oras.Target, o Options) *pusher {
	p := &pusher{target: target, chunk: o.ChunkSize, o: o}
	p.repo, _ = target.(*remote.Repository)
	return p
}

func (p *pusher) chunkSize() int64 {
	if p.chunk > 0 {
		return p.chunk
	}
	return fallbackChunk
}

// push uploads one blob whole, or in chunks when it is larger than the
// chunk size asked for or the registry has refused a whole blob.
func (p *pusher) push(ctx context.Context, d ocispec.Descriptor, open func() (io.ReadCloser, error)) error {
	if p.repo != nil && (p.chunked.Load() || p.chunk > 0 && d.Size > p.chunk) {
		return p.pushChunked(ctx, d, open)
	}
	err := p.pushWhole(ctx, d, open)
	var resp *errcode.ErrorResponse
	if p.repo == nil || !errors.As(err, &resp) || resp.StatusCode != http.StatusRequestEntityTooLarge {
		return err
	}
	if !p.chunked.Swap(true) {
		p.o.logf("registry refused a %s blob as too large: uploading in %s chunks",
			bytesText(d.Size), bytesText(p.chunkSize()))
	}
	return p.pushChunked(ctx, d, open)
}

// pushWhole uploads a blob in one request, again from the start after a
// transient failure: a streamed body cannot be replayed by the transport,
// and one 5xx among thousands of blobs must not fail a push.
func (p *pusher) pushWhole(ctx context.Context, d ocispec.Descriptor, open func() (io.ReadCloser, error)) error {
	for failures := 0; ; {
		r, err := open()
		if err != nil {
			return err
		}
		err = p.target.Push(ctx, d, r)
		r.Close()
		if err == nil || errors.Is(err, errdef.ErrAlreadyExists) || !retryable(err) {
			return err
		}
		if failures++; failures > chunkRetries {
			return fmt.Errorf("oci: upload %s: %w", d.Digest, err)
		}
		p.o.logf("upload %s failed, attempt %d of %d: %v", d.Digest, failures+1, chunkRetries+1, err)
		if err := sleep(ctx, chunkBackoff<<(failures-1)); err != nil {
			return err
		}
	}
}

// pushChunked uploads a blob in chunks. A chunk that fails is resumed from
// the offset the registry reports for the session.
func (p *pusher) pushChunked(ctx context.Context, d ocispec.Descriptor, open func() (io.ReadCloser, error)) error {
	rc, err := open()
	if err != nil {
		return err
	}
	defer rc.Close()
	src, ok := rc.(io.ReaderAt)
	if !ok { // in-memory content such as the config: small, so hold it
		raw, err := io.ReadAll(io.LimitReader(rc, d.Size+1))
		if err != nil {
			return err
		}
		if int64(len(raw)) != d.Size {
			return fmt.Errorf("oci: %s is %d bytes, the descriptor says %d", d.Digest, len(raw), d.Size)
		}
		src = bytes.NewReader(raw)
	}
	ctx = auth.AppendRepositoryScope(ctx, p.repo.Reference, auth.ActionPull, auth.ActionPush)
	loc, minChunk, err := p.startUpload(ctx)
	if err != nil {
		return err
	}
	size := max(p.chunkSize(), minChunk)
	var off int64
	for failures := 0; off < d.Size; {
		next, got, err := p.patchChunk(ctx, loc, src, off, min(size, d.Size-off))
		if err == nil {
			loc, off, failures = next, got, 0
			continue
		}
		if failures++; failures > chunkRetries || !retryable(err) {
			return fmt.Errorf("oci: upload %s at byte %d: %w", d.Digest, off, err)
		}
		if err := sleep(ctx, chunkBackoff<<(failures-1)); err != nil {
			return err
		}
		if next, got, err := p.uploadStatus(ctx, loc); err == nil {
			loc, off = next, got
		}
	}
	return p.finishUpload(ctx, loc, d)
}

// startUpload opens a session and returns its URL and the registry's
// minimum chunk (OCI-Chunk-Min-Length), zero when it has none.
func (p *pusher) startUpload(ctx context.Context) (string, int64, error) {
	scheme := "https"
	if p.repo.PlainHTTP {
		scheme = "http"
	}
	ref := p.repo.Reference
	u := fmt.Sprintf("%s://%s/v2/%s/blobs/uploads/", scheme, ref.Host(), ref.Repository)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, u, nil)
	if err != nil {
		return "", 0, err
	}
	resp, err := p.repo.Client.Do(req)
	if err != nil {
		return "", 0, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusAccepted {
		return "", 0, responseError(resp)
	}
	loc, err := uploadLocation(resp, "")
	if err != nil {
		return "", 0, err
	}
	minChunk, _ := strconv.ParseInt(resp.Header.Get("OCI-Chunk-Min-Length"), 10, 64)
	return loc, max(minChunk, 0), nil
}

// patchChunk sends n bytes at off and returns the session's next URL and
// how many bytes the registry now holds.
func (p *pusher) patchChunk(ctx context.Context, loc string, src io.ReaderAt, off, n int64) (string, int64, error) {
	body := func() io.ReadCloser { return io.NopCloser(io.NewSectionReader(src, off, n)) }
	req, err := http.NewRequestWithContext(ctx, http.MethodPatch, loc, body())
	if err != nil {
		return "", 0, err
	}
	req.GetBody = func() (io.ReadCloser, error) { return body(), nil }
	req.ContentLength = n
	req.Header.Set("Content-Type", "application/octet-stream")
	req.Header.Set("Content-Range", fmt.Sprintf("%d-%d", off, off+n-1))
	resp, err := p.repo.Client.Do(req)
	if err != nil {
		return "", 0, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusAccepted {
		return "", 0, responseError(resp)
	}
	next, err := uploadLocation(resp, loc)
	if err != nil {
		return "", 0, err
	}
	got, ok := held(resp.Header.Get("Range"))
	if !ok {
		got = off + n
	}
	return next, got, nil
}

// uploadStatus asks how many bytes of the session the registry holds.
func (p *pusher) uploadStatus(ctx context.Context, loc string) (string, int64, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, loc, nil)
	if err != nil {
		return "", 0, err
	}
	resp, err := p.repo.Client.Do(req)
	if err != nil {
		return "", 0, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusNoContent && resp.StatusCode != http.StatusAccepted {
		return "", 0, responseError(resp)
	}
	next, err := uploadLocation(resp, loc)
	if err != nil {
		return "", 0, err
	}
	// distribution answers "0-0" for an empty session as well as for one
	// byte; resending from zero is the safe reading.
	r := resp.Header.Get("Range")
	got, ok := held(r)
	if !ok || strings.TrimPrefix(r, "bytes=") == "0-0" {
		got = 0
	}
	return next, got, nil
}

// finishUpload closes the session with the blob's digest.
func (p *pusher) finishUpload(ctx context.Context, loc string, d ocispec.Descriptor) error {
	u, err := url.Parse(loc)
	if err != nil {
		return err
	}
	q := u.Query()
	q.Set("digest", d.Digest.String())
	u.RawQuery = q.Encode()
	req, err := http.NewRequestWithContext(ctx, http.MethodPut, u.String(), http.NoBody)
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/octet-stream")
	resp, err := p.repo.Client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusCreated {
		return responseError(resp)
	}
	return nil
}

// uploadLocation is the session URL a response points to. Like oras-go for
// its own uploads, it stays on the registry's host and never leaves https,
// so credentials are not sent elsewhere.
func uploadLocation(resp *http.Response, current string) (string, error) {
	if resp.Header.Get("Location") == "" {
		if current == "" {
			return "", fmt.Errorf("%w: %s %s answered without one", errLocation,
				resp.Request.Method, resp.Request.URL.Redacted())
		}
		return current, nil
	}
	loc, err := resp.Location()
	if err != nil {
		return "", fmt.Errorf("%w: %v", errLocation, err)
	}
	req := resp.Request.URL
	if req.Port() == "443" && loc.Hostname() == req.Hostname() && loc.Port() == "" {
		loc.Host = loc.Hostname() + ":443"
	}
	if loc.Hostname() != req.Hostname() || port(loc) != port(req) {
		return "", fmt.Errorf("%w: %q is on a different host than the registry %q", errLocation, loc.Host, req.Host)
	}
	if req.Scheme == "https" && loc.Scheme != "https" {
		return "", fmt.Errorf("%w: %q leaves https", errLocation, loc.Host)
	}
	return loc.String(), nil
}

func port(u *url.URL) string {
	if p := u.Port(); p != "" {
		return p
	}
	if u.Scheme == "https" {
		return "443"
	}
	return "80"
}

// held reads a Range header, "0-<last byte>", as the bytes it covers.
func held(r string) (int64, bool) {
	start, last, ok := strings.Cut(strings.TrimPrefix(r, "bytes="), "-")
	if !ok || start != "0" {
		return 0, false
	}
	n, err := strconv.ParseInt(last, 10, 64)
	if err != nil || n < 0 {
		return 0, false
	}
	return n + 1, true
}

// responseError reads a registry's error the way oras-go reports its own.
func responseError(resp *http.Response) error {
	e := &errcode.ErrorResponse{Method: resp.Request.Method, URL: resp.Request.URL, StatusCode: resp.StatusCode}
	var body struct {
		Errors errcode.Errors `json:"errors"`
	}
	if json.NewDecoder(io.LimitReader(resp.Body, 8<<10)).Decode(&body) == nil {
		e.Errors = body.Errors
	}
	return e
}

// retryable: the network, a registry under load, or a chunk it took only
// part of (416). Any other refusal is final.
func retryable(err error) bool {
	if errors.Is(err, errLocation) || errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		return false
	}
	var resp *errcode.ErrorResponse
	if !errors.As(err, &resp) {
		return true
	}
	switch resp.StatusCode {
	case http.StatusRequestedRangeNotSatisfiable, http.StatusRequestTimeout, http.StatusTooManyRequests:
		return true
	}
	return resp.StatusCode >= 500
}

func sleep(ctx context.Context, d time.Duration) error {
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-t.C:
		return nil
	}
}

func bytesText(n int64) string {
	switch {
	case n >= 1<<30:
		return fmt.Sprintf("%.2f GiB", float64(n)/(1<<30))
	case n >= 1<<20:
		return fmt.Sprintf("%.0f MiB", float64(n)/(1<<20))
	}
	return fmt.Sprintf("%d B", n)
}
