package oci

import (
	"context"
	"errors"
	"io"
	"sync"
	"time"

	ocispec "github.com/opencontainers/image-spec/specs-go/v1"
	"oras.land/oras-go/v2"
)

// stallAfter is how long a blob's body may send nothing before its
// connection is dropped and the blob retried. Some paths to a CDN stall a
// response mid-body for good instead of failing it.
var stallAfter = 30 * time.Second

// stallGuard fetches through target and closes a body that stays silent
// for idle, so that the read fails and the blob can be fetched again.
type stallGuard struct {
	oras.ReadOnlyTarget
	idle time.Duration
}

func (g stallGuard) Fetch(ctx context.Context, d ocispec.Descriptor) (io.ReadCloser, error) {
	rc, err := g.ReadOnlyTarget.Fetch(ctx, d)
	if err != nil {
		return nil, err
	}
	w := &watched{rc: rc}
	w.timer = time.AfterFunc(g.idle, w.stall)
	return w, nil
}

type watched struct {
	rc      io.ReadCloser
	timer   *time.Timer
	mu      sync.Mutex
	stalled bool
}

func (w *watched) stall() {
	w.mu.Lock()
	w.stalled = true
	w.mu.Unlock()
	w.rc.Close()
}

func (w *watched) Read(p []byte) (int, error) {
	n, err := w.rc.Read(p)
	if n > 0 {
		w.timer.Reset(stallAfter)
	}
	w.mu.Lock()
	stalled := w.stalled
	w.mu.Unlock()
	if stalled && err != nil && !errors.Is(err, io.EOF) {
		err = errStalled
	}
	return n, err
}

func (w *watched) Close() error {
	w.timer.Stop()
	return w.rc.Close()
}

var errStalled = errors.New("oci: the registry stopped sending, connection dropped")
