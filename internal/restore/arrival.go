package restore

import (
	"context"
	"time"

	"github.com/TensorRails/warmctl/v2/internal/docker"
)

type imagePull struct {
	image docker.Image
	err   error
}

// launching, while the snapshot is pulled, waits for the few files the
// receiver starts with, which the pull fetches first, and pulls a
// receiver's engine image meanwhile, if this host lacks it. The receiver
// then starts before the pull has filled the page cache: `docker create`
// syncs, and a sync waits for the dirty pages the pull leaves (v1 P138).
func (r *restore) launching(ctx context.Context) error {
	if r.o.Arrival == nil {
		return nil
	}
	if !r.o.InPlace {
		r.pulling = make(chan imagePull, 1)
		go func() {
			img, err := r.dk.EnsureImage(ctx, r.m.Launch.Image, nil)
			r.pulling <- imagePull{img, err}
		}()
	}
	var paths []string
	for _, f := range r.m.LaunchLane() {
		paths = append(paths, f.Path)
	}
	return r.o.Arrival.Wait(ctx, paths)
}

// arrive, while the snapshot is pulled, waits for the files the process
// restore reads: all but the weights lane.
func (r *restore) arrive(ctx context.Context) error {
	if r.o.Arrival == nil {
		return nil
	}
	start := time.Now()
	var paths []string
	var bytes int64
	for _, f := range r.m.FirstLane() {
		paths = append(paths, f.Path)
		bytes += f.Size
	}
	if err := r.o.Arrival.Wait(ctx, paths); err != nil {
		return err
	}
	r.p.Printf("%d files for the process restore arrived (%.2f GiB) after %.1fs; the weights are still being pulled",
		len(paths), float64(bytes)/(1<<30), time.Since(start).Seconds())
	return nil
}

// arrived waits for the weights lane before the engine wakes, which reloads
// every part. The pull may still be syncing them to disk; the engine reads
// the checked bytes from memory.
func (r *restore) arrived(ctx context.Context) error {
	if r.o.Arrival == nil {
		return nil
	}
	start := time.Now()
	var paths []string
	for _, f := range r.m.WeightLane() {
		paths = append(paths, f.Path)
	}
	if err := r.o.Arrival.Wait(ctx, paths); err != nil {
		return err
	}
	r.p.Printf("the weights arrived, %.1fs after the process restore", time.Since(start).Seconds())
	return nil
}

// manifestPath is the manifest.json the weights reload reads, as the engine
// sees it: the snapshot's, or during a pull, which writes the snapshot's
// last, the copy prepare made in the work directory.
func (r *restore) manifestPath() string {
	if r.o.Arrival != nil {
		return r.guest().work + "/manifest.json"
	}
	return r.guest().capture + "/manifest.json"
}

// ensureImage makes the engine image present: the pull arrive started, or
// one now.
func (r *restore) ensureImage(ctx context.Context) (docker.Image, error) {
	if r.pulling == nil {
		return r.dk.EnsureImage(ctx, r.m.Launch.Image, nil)
	}
	select {
	case p := <-r.pulling:
		return p.image, p.err
	case <-ctx.Done():
		return docker.Image{}, ctx.Err()
	}
}
