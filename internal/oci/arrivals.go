package oci

import (
	"context"
	"fmt"
	"os"
	"path"
	"path/filepath"
	"sync"

	"github.com/TensorRails/warmctl/v2/internal/manifest"
)

// Arrivals follows a pull that a restore starts on before the pull ends.
// Once the manifest passed the host check, the pull names the directory it
// writes, the snapshot's own; then it reports every file as it is linked
// there, every byte checked, before the disk has it all; once every blob is
// synced and in the store, manifest.json comes last, and the pull ends.
type Arrivals struct {
	begun   chan struct{}
	dir     string
	m       manifest.Manifest
	raw     []byte
	mu      sync.Mutex
	in      map[string]bool
	ended   bool
	err     error
	changed chan struct{}
}

// NewArrivals is for one pull, set as its Options.Arrivals.
func NewArrivals() *Arrivals {
	return &Arrivals{begun: make(chan struct{}), in: map[string]bool{}, changed: make(chan struct{})}
}

// Begun is closed once Dir and Manifest are known.
func (a *Arrivals) Begun() <-chan struct{} { return a.begun }

// Dir is the snapshot directory the pull fills.
func (a *Arrivals) Dir() string { return a.dir }

// Manifest is the snapshot's manifest.json, before the file is written.
func (a *Arrivals) Manifest() manifest.Manifest { return a.m }

// ManifestJSON is the bytes manifest.json will hold.
func (a *Arrivals) ManifestJSON() []byte { return a.raw }

// Wait returns once every file of paths is in place, or with the error that
// ended the pull, or ctx's.
func (a *Arrivals) Wait(ctx context.Context, paths []string) error {
	i := 0
	for {
		a.mu.Lock()
		for i < len(paths) && a.in[paths[i]] {
			i++
		}
		ended, err, changed := a.ended, a.err, a.changed
		a.mu.Unlock()
		switch {
		case i == len(paths):
			return nil
		case err != nil:
			return err
		case ended:
			return fmt.Errorf("oci: the pull ended without %s", paths[i])
		}
		select {
		case <-changed:
		case <-ctx.Done():
			return ctx.Err()
		}
	}
}

// Done waits for the pull to end and returns its error: nil once every file
// and manifest.json are in place.
func (a *Arrivals) Done(ctx context.Context) error {
	for {
		a.mu.Lock()
		ended, err, changed := a.ended, a.err, a.changed
		a.mu.Unlock()
		if ended {
			return err
		}
		select {
		case <-changed:
		case <-ctx.Done():
			return ctx.Err()
		}
	}
}

func (a *Arrivals) begin(dir string, m manifest.Manifest, raw []byte) {
	if a != nil {
		a.dir, a.m, a.raw = dir, m, raw
		close(a.begun)
	}
}

func (a *Arrivals) arrived(p string) {
	if a != nil {
		a.update(func() { a.in[p] = true })
	}
}

func (a *Arrivals) end(err error) {
	if a != nil {
		a.update(func() { a.ended, a.err = true, err })
	}
}

func (a *Arrivals) update(fn func()) {
	a.mu.Lock()
	defer a.mu.Unlock()
	fn()
	close(a.changed)
	a.changed = make(chan struct{})
}

// arrivalOrder puts the files a restore reads to start its receiver first,
// then the rest it needs before the weights, then the weights lane
// round-robin across ranks: the receiver starts before the disk is busy,
// the restore steps start early, and every rank's reload finds its first
// parts. A layer goes as early as the earliest file it holds.
func arrivalOrder(m manifest.Manifest, files []fileLayers) []fileLayers {
	const launch, first, weights = 0, 1, 2
	lane := map[string]int{}
	for _, f := range m.LaunchLane() {
		lane[f.Path] = launch
	}
	for _, f := range m.WeightLane() {
		lane[f.Path] = weights
	}
	var head [weights][]fileLayers
	var ranks []string
	byRank := map[string][]fileLayers{}
	for _, fl := range files {
		tier := weights
		for _, f := range fl.files {
			t, ok := lane[f.Path]
			if !ok {
				t = first
			}
			tier = min(tier, t)
		}
		if tier < weights {
			head[tier] = append(head[tier], fl)
			continue
		}
		d := path.Dir(fl.files[0].Path)
		if _, seen := byRank[d]; !seen {
			ranks = append(ranks, d)
		}
		byRank[d] = append(byRank[d], fl)
	}
	out := append(append([]fileLayers(nil), head[launch]...), head[first]...)
	for i := 0; len(out) < len(files); i++ {
		for _, d := range ranks {
			if i < len(byRank[d]) {
				out = append(out, byRank[d][i])
			}
		}
	}
	return out
}

// claimOutput takes output for this pull: <output>.pulling, locked, marks
// it. A snapshot already there is refused; what a pull that died left there
// is removed. release drops the mark.
func claimOutput(output string) (release func(), err error) {
	marker := output + ".pulling"
	f, err := os.OpenFile(marker, os.O_RDWR|os.O_CREATE|os.O_EXCL, 0o644)
	leftover := os.IsExist(err)
	if leftover {
		f, err = os.OpenFile(marker, os.O_RDWR, 0o644)
	}
	if err != nil {
		return nil, err
	}
	if err := lockFile(f); err != nil {
		f.Close()
		return nil, fmt.Errorf("oci: another pull is writing %s: %w", output, err)
	}
	release = func() {
		os.Remove(marker)
		f.Close()
	}
	if _, err := os.Stat(filepath.Join(output, "manifest.json")); err == nil {
		release()
		return nil, fmt.Errorf("oci: %s holds a snapshot already", output)
	}
	if leftover {
		if err := os.RemoveAll(output); err != nil {
			release()
			return nil, err
		}
	}
	return release, nil
}
