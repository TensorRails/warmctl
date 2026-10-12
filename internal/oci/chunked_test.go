package oci

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/opencontainers/go-digest"

	"github.com/TensorRails/warmctl/v2/internal/store"
)

// fakeRegistry serves the OCI distribution API from memory: blobs, upload
// sessions (whole and chunked), manifests and tags, with the limits and
// failures a push has to get through.
type fakeRegistry struct {
	mu        sync.Mutex
	blobs     map[string][]byte
	manifests map[string][]byte
	types     map[string]string
	uploads   map[string][]byte
	maxBody   int64 // 413 for a request body above it, like a Workers limit
	minChunk  int64 // advertised as OCI-Chunk-Min-Length
	failPatch int   // the n-th PATCH keeps half its bytes and answers 500
	failPut   int   // the n-th closing PUT reads its body and answers 502
	puts      int
	patches   []int64
	tooLarge  int
	sessions  int
}

func newFakeRegistry() *fakeRegistry {
	return &fakeRegistry{blobs: map[string][]byte{}, manifests: map[string][]byte{},
		types: map[string]string{}, uploads: map[string][]byte{}}
}

func registryError(w http.ResponseWriter, status int, code string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	fmt.Fprintf(w, `{"errors":[{"code":%q,"message":"fake"}]}`, code)
}

func (f *fakeRegistry) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.maxBody > 0 && r.ContentLength > f.maxBody {
		f.tooLarge++
		registryError(w, http.StatusRequestEntityTooLarge, "SIZE_INVALID")
		return
	}
	p := strings.TrimPrefix(r.URL.Path, "/v2/")
	if repo, id, ok := strings.Cut(p, "/blobs/uploads/"); ok {
		f.upload(w, r, repo, id)
	} else if _, dgst, ok := strings.Cut(p, "/blobs/"); ok {
		b, ok := f.blobs[dgst]
		if !ok {
			registryError(w, http.StatusNotFound, "BLOB_UNKNOWN")
			return
		}
		w.Header().Set("Content-Length", strconv.Itoa(len(b)))
		w.Header().Set("Docker-Content-Digest", dgst)
		if r.Method == http.MethodGet {
			w.Write(b)
		}
	} else if _, ref, ok := strings.Cut(p, "/manifests/"); ok {
		f.manifest(w, r, ref)
	} else {
		registryError(w, http.StatusNotFound, "NAME_UNKNOWN")
	}
}

func (f *fakeRegistry) upload(w http.ResponseWriter, r *http.Request, repo, id string) {
	session := func(id string) {
		// relative, with a query the client has to carry along
		w.Header().Set("Location", "/v2/"+repo+"/blobs/uploads/"+id+"?_state=s-"+id)
	}
	if r.Method == http.MethodPost && id == "" {
		f.sessions++
		id = strconv.Itoa(f.sessions)
		f.uploads[id] = []byte{}
		session(id)
		if f.minChunk > 0 {
			w.Header().Set("OCI-Chunk-Min-Length", strconv.FormatInt(f.minChunk, 10))
		}
		w.WriteHeader(http.StatusAccepted)
		return
	}
	buf, ok := f.uploads[id]
	if !ok || r.URL.Query().Get("_state") != "s-"+id {
		registryError(w, http.StatusNotFound, "BLOB_UPLOAD_UNKNOWN")
		return
	}
	switch r.Method {
	case http.MethodPatch:
		var first, last int
		if _, err := fmt.Sscanf(r.Header.Get("Content-Range"), "%d-%d", &first, &last); err != nil || first != len(buf) {
			w.Header().Set("Range", fmt.Sprintf("0-%d", max(len(buf)-1, 0)))
			registryError(w, http.StatusRequestedRangeNotSatisfiable, "BLOB_UPLOAD_INVALID")
			return
		}
		body, _ := io.ReadAll(r.Body)
		if len(body) != last-first+1 {
			registryError(w, http.StatusBadRequest, "BLOB_UPLOAD_INVALID")
			return
		}
		f.patches = append(f.patches, int64(len(body)))
		if len(f.patches) == f.failPatch {
			f.uploads[id] = append(buf, body[:len(body)/2]...)
			registryError(w, http.StatusInternalServerError, "UNKNOWN")
			return
		}
		f.uploads[id] = append(buf, body...)
		session(id)
		w.Header().Set("Range", fmt.Sprintf("0-%d", len(f.uploads[id])-1))
		w.WriteHeader(http.StatusAccepted)
	case http.MethodGet:
		session(id)
		w.Header().Set("Range", fmt.Sprintf("0-%d", max(len(buf)-1, 0)))
		w.WriteHeader(http.StatusNoContent)
	case http.MethodPut:
		body, _ := io.ReadAll(r.Body)
		if f.puts++; f.puts == f.failPut {
			registryError(w, http.StatusBadGateway, "UNKNOWN")
			return
		}
		buf = append(buf, body...)
		want := r.URL.Query().Get("digest")
		if digest.FromBytes(buf).String() != want {
			registryError(w, http.StatusBadRequest, "DIGEST_INVALID")
			return
		}
		f.blobs[want] = buf
		delete(f.uploads, id)
		w.Header().Set("Location", "/v2/"+repo+"/blobs/"+want)
		w.Header().Set("Docker-Content-Digest", want)
		w.WriteHeader(http.StatusCreated)
	default:
		registryError(w, http.StatusMethodNotAllowed, "UNSUPPORTED")
	}
}

func (f *fakeRegistry) manifest(w http.ResponseWriter, r *http.Request, ref string) {
	if r.Method == http.MethodPut {
		body, _ := io.ReadAll(r.Body)
		d := digest.FromBytes(body).String()
		for _, k := range []string{d, ref} {
			f.manifests[k], f.types[k] = body, r.Header.Get("Content-Type")
		}
		w.Header().Set("Docker-Content-Digest", d)
		w.WriteHeader(http.StatusCreated)
		return
	}
	b, ok := f.manifests[ref]
	if !ok {
		registryError(w, http.StatusNotFound, "MANIFEST_UNKNOWN")
		return
	}
	w.Header().Set("Content-Type", f.types[ref])
	w.Header().Set("Docker-Content-Digest", digest.FromBytes(b).String())
	w.Header().Set("Content-Length", strconv.Itoa(len(b)))
	if r.Method == http.MethodGet {
		w.Write(b)
	}
}

// bigPart is a weights part of 100 KiB.
var bigPart = strings.Repeat("0123456789abcdef", 6400)

// pushToFake pushes a snapshot with a 100 KiB weights part to f and pulls it
// back, checking every file.
func pushToFake(t *testing.T, f *fakeRegistry, o Options) {
	t.Helper()
	srv := httptest.NewServer(f)
	defer srv.Close()
	roundTrip(t, "oci://"+strings.TrimPrefix(srv.URL, "http://")+"/snaps/x:v1", o)
}

// TestPushRealRegistry runs the round trip against a plain-HTTP registry at
// $WARMCTL_TEST_REGISTRY, e.g. `docker run -p 5000:5000 registry:2` and
// WARMCTL_TEST_REGISTRY=localhost:5000.
func TestPushRealRegistry(t *testing.T) {
	host := os.Getenv("WARMCTL_TEST_REGISTRY")
	if host == "" {
		t.Skip("WARMCTL_TEST_REGISTRY is not set")
	}
	// A repository per run: a registry keeps blobs per repository, so a
	// second push into the same one would find every blob and send nothing.
	for _, chunk := range []int64{0, 16 << 10} {
		ref := fmt.Sprintf("oci://%s/warmctl-test/c%d-%d:v1", host, chunk, time.Now().UnixNano())
		roundTrip(t, ref, Options{ChunkSize: chunk})
	}
}

func roundTrip(t *testing.T, ref string, o Options) {
	t.Helper()
	root := t.TempDir()
	snapshotDir(t, filepath.Join(root, "snap"), bigPart)
	o.PlainHTTP = true
	if o.PieceSize == 0 {
		o.PieceSize = int64(len(bigPart)) // the part is one blob, neither bundled nor split
	}
	repo, tag, err := Repository(ref, o)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := Push(context.Background(), repo, filepath.Join(root, "snap"), tag, o); err != nil {
		t.Fatal(err)
	}
	out := filepath.Join(root, "pulled")
	m, err := Pull(context.Background(), repo, tag, out, store.Store{Root: filepath.Join(root, "state")}, o)
	if err != nil {
		t.Fatal(err)
	}
	for _, file := range m.Files {
		raw, err := os.ReadFile(filepath.Join(out, file.Path))
		sum := sha256.Sum256(raw)
		if err != nil || hex.EncodeToString(sum[:]) != file.SHA256 {
			t.Fatalf("%s differs after pull: %v", file.Path, err)
		}
	}
}

func TestPushWholeBlobsByDefault(t *testing.T) {
	f := newFakeRegistry()
	pushToFake(t, f, Options{})
	if len(f.patches) != 0 || f.tooLarge != 0 {
		t.Fatalf("a registry without limits got %d PATCHes and %d 413s", len(f.patches), f.tooLarge)
	}
}

func TestPushChunked(t *testing.T) {
	f := newFakeRegistry()
	pushToFake(t, f, Options{ChunkSize: 16 << 10})
	// Only the 100 KiB part is above the chunk size: six full chunks and a tail.
	want := []int64{16384, 16384, 16384, 16384, 16384, 16384, 4096}
	if fmt.Sprint(f.patches) != fmt.Sprint(want) {
		t.Fatalf("PATCH sizes %v, want %v", f.patches, want)
	}
}

func TestPushChunkedHonoursMinimum(t *testing.T) {
	f := newFakeRegistry()
	f.minChunk = 40000
	pushToFake(t, f, Options{ChunkSize: 16 << 10})
	if fmt.Sprint(f.patches) != fmt.Sprint([]int64{40000, 40000, 22400}) {
		t.Fatalf("PATCH sizes %v with a 40000 byte minimum", f.patches)
	}
}

func TestPushChunksAfterTooLarge(t *testing.T) {
	defer func(v int64) { fallbackChunk = v }(fallbackChunk)
	fallbackChunk = 16 << 10
	f := newFakeRegistry()
	f.maxBody = 32 << 10
	var logs []string
	pushToFake(t, f, Options{Log: func(format string, a ...any) { logs = append(logs, fmt.Sprintf(format, a...)) }})
	// One refusal, then the part and the config go in chunks.
	if f.tooLarge != 1 || len(f.patches) < 7 {
		t.Fatalf("%d 413s, PATCH sizes %v", f.tooLarge, f.patches)
	}
	if !strings.Contains(strings.Join(logs, "\n"), "refused") {
		t.Fatalf("logs %q", logs)
	}
}

func TestPushChunkedResumes(t *testing.T) {
	defer func(v time.Duration) { chunkBackoff = v }(chunkBackoff)
	chunkBackoff = time.Millisecond
	f := newFakeRegistry()
	f.failPatch = 3
	pushToFake(t, f, Options{ChunkSize: 16 << 10})
	var sent int64
	for _, n := range f.patches {
		sent += n
	}
	// The failed chunk's kept half is not sent again.
	if sent != int64(len(bigPart))+16384/2 {
		t.Fatalf("sent %d bytes in %v", sent, f.patches)
	}
}

// A whole blob that meets a 5xx is uploaded again from its start.
func TestPushRetriesWholeBlob(t *testing.T) {
	defer func(v time.Duration) { chunkBackoff = v }(chunkBackoff)
	chunkBackoff = time.Millisecond
	f := newFakeRegistry()
	f.failPut = 2
	var logs []string
	pushToFake(t, f, Options{Log: func(format string, a ...any) { logs = append(logs, fmt.Sprintf(format, a...)) }})
	if f.sessions != f.puts || !strings.Contains(strings.Join(logs, "\n"), "attempt 2 of 6") {
		t.Fatalf("%d sessions, %d PUTs, logs %q", f.sessions, f.puts, logs)
	}
}

func TestUploadLocation(t *testing.T) {
	resp := func(req, loc string) *http.Response {
		u, _ := url.Parse(req)
		r := &http.Response{Header: http.Header{}, Request: &http.Request{Method: http.MethodPost, URL: u}}
		if loc != "" {
			r.Header.Set("Location", loc)
		}
		return r
	}
	for _, c := range []struct{ req, loc, want string }{
		{"https://reg.io/v2/x/blobs/uploads/", "/v2/x/blobs/uploads/1?s=a", "https://reg.io/v2/x/blobs/uploads/1?s=a"},
		{"https://reg.io:443/v2/x/blobs/uploads/", "https://reg.io/up/1", "https://reg.io:443/up/1"},
		{"https://reg.io/v2/x/blobs/uploads/", "https://evil.io/up/1", ""},
		{"https://reg.io/v2/x/blobs/uploads/", "http://reg.io/up/1", ""},
		{"https://reg.io/v2/x/blobs/uploads/", "", ""},
	} {
		got, err := uploadLocation(resp(c.req, c.loc), "")
		if got != c.want || (c.want == "") != (err != nil) {
			t.Errorf("Location %q after %s: %q %v, want %q", c.loc, c.req, got, err, c.want)
		}
	}
	if got, err := uploadLocation(resp("https://reg.io/a", ""), "https://reg.io/b"); err != nil || got != "https://reg.io/b" {
		t.Errorf("no Location keeps the session: %q %v", got, err)
	}
}

func TestHeld(t *testing.T) {
	for r, want := range map[string]int64{"0-99": 100, "bytes=0-9": 10, "0-0": 1, "5-9": -1, "": -1, "0-x": -1} {
		got, ok := held(r)
		if !ok {
			got = -1
		}
		if got != want {
			t.Errorf("held(%q) = %d, want %d", r, got, want)
		}
	}
}
