package restore

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"time"

	"github.com/TensorRails/warmctl/v2/internal/manifest"
	"github.com/TensorRails/warmctl/v2/internal/zframes"
)

// stageWorkers is how many ranges a stage copies at once: object storage
// behind FUSE reaches its bandwidth only with many requests in flight.
const stageWorkers = 32

// staging names a file a stage is still writing: it takes its own name only
// once complete, as a pull links a file in, so the engine's reload may
// open a weights part as soon as it shows up.
const staging = ".staging"

// stageStall is how long a stage may go without finishing a range before it
// gives up: a read of a network file system can hang without failing, and
// the engine's reload waits for the weights without a deadline of its own.
var stageStall = 5 * time.Minute

// arrivalWait bounds how long a stage waits for a file the manifest lists:
// a RunPod Global Volume shows another pod's writes seconds late, and its
// manifest.json may show before the files.
var arrivalWait = 2 * time.Minute

// Stage copies the snapshot in dir to dst, every file of m and its
// manifest.json, many ranges in flight, and returns how many bytes it read
// from dir. A file stored as zstd frames is decoded frame by frame on the
// workers, so dst always holds the files as they are. It is for a snapshot
// directory on a FUSE file system (an S3 bucket behind geesefs, as a RunPod
// Global Volume, or MooseFS, as its network volumes): FUSE keeps no page
// cache the prefetch could fill for later readers, and the restore's readers
// (CRIU or dmtcp_restart, the weights reload) read one stream each. The files
// go lane by lane, those the receiver starts with, those the process restore
// reads, the weights, and done, when set, hears of each file once it is
// complete: the restore runs while the weights still come. A copy whose
// manifest.json matches is complete and is reused: manifest.json is written
// last. An encoded snapshot is staged wherever it is.
func Stage(ctx context.Context, dir, dst string, m manifest.Manifest, done func(path string)) (int64, error) {
	raw, err := os.ReadFile(filepath.Join(dir, "manifest.json"))
	if err != nil {
		return 0, err
	}
	if staged(dst, raw) {
		return 0, nil
	}
	if err := os.RemoveAll(dst); err != nil {
		return 0, err
	}
	if done == nil {
		done = func(string) {}
	}
	first, weights := lanes(m)
	left, err := prepare(ctx, dir, dst, append(append([]manifest.File(nil), first...), weights...), done)
	if err != nil {
		return 0, err
	}
	read, err := copyJobs(ctx, dir, dst, stagePlan(first, weights), left, done)
	if err != nil {
		return read, err
	}
	return read, os.WriteFile(filepath.Join(dst, "manifest.json"), raw, 0o644)
}

// prepare waits for every file to show up in dir and makes it in dst at its
// size under its staging name; an empty file is done at once. It returns how
// many bytes each file still lacks.
func prepare(ctx context.Context, dir, dst string, files []manifest.File, done func(string)) (
	map[string]*atomic.Int64, error) {
	left := map[string]*atomic.Int64{}
	for _, f := range files {
		info, err := arrived(ctx, filepath.Join(dir, f.Path), f.Stored())
		if err != nil {
			return nil, err
		}
		if f.Size == 0 {
			if err := create(info, filepath.Join(dst, f.Path), 0); err != nil {
				return nil, err
			}
			done(f.Path)
			continue
		}
		if err := create(info, filepath.Join(dst, f.Path)+staging, f.Size); err != nil {
			return nil, err
		}
		left[f.Path] = new(atomic.Int64)
		left[f.Path].Store(f.Size)
	}
	return left, nil
}

// copyJobs runs jobs on stageWorkers workers and returns the bytes they read
// from dir; the first error stops them all. When no job finishes for
// stageStall it fails at once, leaving the hung reads behind.
func copyJobs(ctx context.Context, dir, dst string, jobs []prefetchJob, left map[string]*atomic.Int64,
	done func(string)) (int64, error) {
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	ch := feed(ctx, jobs)
	var read, last atomic.Int64 // last: when a job last finished, in Unix nanoseconds
	var once sync.Once
	var first error
	fail := func(err error) { once.Do(func() { first = err; cancel() }) }
	last.Store(time.Now().UnixNano())
	var wg sync.WaitGroup
	for range stageWorkers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			w := &stager{buf: make([]byte, 8<<20)}
			defer w.close()
			for j := range ch {
				got, err := w.stage(ctx, dir, dst, j, left[j.path], done)
				read.Add(got)
				last.Store(time.Now().UnixNano())
				if err != nil {
					fail(err)
				}
			}
		}()
	}
	finished := make(chan struct{})
	go func() { wg.Wait(); close(finished) }()
	tick := time.NewTicker(stageStall / 10)
	defer tick.Stop()
	for {
		select {
		case <-finished:
			if first != nil {
				return read.Load(), first
			}
			return read.Load(), ctx.Err()
		case <-tick.C:
			if idle := time.Since(time.Unix(0, last.Load())); idle > stageStall {
				fail(fmt.Errorf("no range of %s copied in %s: its reads hang", dir, idle.Round(time.Second)))
				return read.Load(), first
			}
		}
	}
}

// stage does j; when j wrote the file's last bytes, the file takes its own
// name and done hears of it, unless the stage has failed meanwhile. It
// returns the bytes it read.
func (w *stager) stage(ctx context.Context, dir, dst string, j prefetchJob, left *atomic.Int64,
	done func(string)) (int64, error) {
	to := filepath.Join(dst, j.path)
	n, got, err := w.run(filepath.Join(dir, j.path), to+staging, j)
	if err != nil || left.Add(-n) != 0 || ctx.Err() != nil {
		return got, err
	}
	if err := os.Rename(to+staging, to); err != nil {
		return got, err
	}
	done(j.path)
	return got, nil
}

// stagePlan is prefetchPlan's order, an encoded file cut at its frames.
func stagePlan(first, weights []manifest.File) []prefetchJob {
	var jobs []prefetchJob
	for _, f := range append(append([]manifest.File(nil), first...), byRankRoundRobin(weights)...) {
		if f.Encoding == "" {
			jobs = append(jobs, prefetchPlan([]manifest.File{f}, nil, prefetchChunk, 0)...)
			continue
		}
		for i := range f.Frames {
			at, n, offset, size := zframes.Frame(f.Frames, f.Size, i)
			jobs = append(jobs, prefetchJob{path: f.Path, offset: offset, size: size, stored: at, storedSize: n,
				split: f.Split})
		}
	}
	return jobs
}

// stager is one stage worker's buffers and frame decoder.
type stager struct {
	buf, frame, out []byte
	dec             *zframes.Decoder
}

// run does j and returns how many bytes of the file it wrote and how many
// it read from src.
func (w *stager) run(src, dst string, j prefetchJob) (int64, int64, error) {
	if j.storedSize == 0 {
		n, err := copyRange(src, dst, j.offset, j.size, w.buf)
		return n, n, err
	}
	if w.dec == nil {
		var err error
		if w.dec, err = zframes.NewDecoder(); err != nil {
			return 0, 0, err
		}
	}
	if int64(cap(w.frame)) < j.storedSize {
		w.frame = make([]byte, j.storedSize)
	}
	frame := w.frame[:j.storedSize]
	in, err := os.Open(src)
	if err != nil {
		return 0, 0, err
	}
	n, err := in.ReadAt(frame, j.stored)
	in.Close()
	if errors.Is(err, io.EOF) && int64(n) == j.storedSize {
		err = nil
	} else if errors.Is(err, io.EOF) {
		err = io.ErrUnexpectedEOF
	}
	if err != nil {
		return 0, int64(n), err
	}
	decoded, err := w.dec.Decode(frame, w.out, j.size, j.split)
	if err != nil {
		return 0, int64(n), fmt.Errorf("%s at %d: %w", src, j.stored, err)
	}
	if j.split > 1 {
		w.out = decoded // ours to reuse; unsplit bytes are the decoder's
	}
	out, err := os.OpenFile(dst, os.O_WRONLY, 0)
	if err != nil {
		return 0, int64(n), err
	}
	if _, err := out.WriteAt(decoded, j.offset); err != nil {
		return 0, int64(n), errors.Join(err, out.Close())
	}
	return j.size, int64(n), out.Close()
}

func (w *stager) close() {
	if w.dec != nil {
		w.dec.Close()
	}
}

// staged is whether dst holds a complete copy of the snapshot whose
// manifest.json is raw.
func staged(dst string, raw []byte) bool {
	have, err := os.ReadFile(filepath.Join(dst, "manifest.json"))
	return err == nil && bytes.Equal(have, raw)
}

// lanes orders m's files as a restore needs them: the receiver's launch
// files, the rest the process restore reads, then the weights.
func lanes(m manifest.Manifest) (first, weights []manifest.File) {
	launch := map[string]bool{}
	for _, f := range m.LaunchLane() {
		launch[f.Path] = true
		first = append(first, f)
	}
	for _, f := range m.FirstLane() {
		if !launch[f.Path] {
			first = append(first, f)
		}
	}
	return first, m.WeightLane()
}

// arrived waits, up to arrivalWait, until src exists with size bytes.
func arrived(ctx context.Context, src string, size int64) (fs.FileInfo, error) {
	deadline := time.Now().Add(arrivalWait)
	for {
		info, err := os.Stat(src)
		switch {
		case err == nil && info.Size() >= size:
			return info, nil
		case err != nil && !errors.Is(err, fs.ErrNotExist):
			return nil, err
		case time.Now().After(deadline):
			if err == nil {
				err = fmt.Errorf("%d of %d bytes", info.Size(), size)
			}
			return nil, fmt.Errorf("%s not there after %s: %w", src, arrivalWait, err)
		}
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-time.After(500 * time.Millisecond):
		}
	}
}

// create makes dst at size, with src's mode.
func create(info fs.FileInfo, dst string, size int64) error {
	if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
		return err
	}
	f, err := os.OpenFile(dst, os.O_CREATE|os.O_EXCL|os.O_WRONLY, info.Mode().Perm()|0o200)
	if err != nil {
		return err
	}
	return errors.Join(f.Truncate(size), f.Close())
}

// copyRange copies size bytes of src at offset to the same offset of dst.
func copyRange(src, dst string, offset, size int64, buf []byte) (int64, error) {
	in, err := os.Open(src)
	if err != nil {
		return 0, err
	}
	defer in.Close()
	out, err := os.OpenFile(dst, os.O_WRONLY, 0)
	if err != nil {
		return 0, err
	}
	done := int64(0)
	for done < size {
		n, err := in.ReadAt(buf[:min(int64(len(buf)), size-done)], offset+done)
		if n > 0 {
			if _, werr := out.WriteAt(buf[:n], offset+done); werr != nil {
				return done, errors.Join(werr, out.Close())
			}
			done += int64(n)
		}
		if errors.Is(err, io.EOF) && done < size {
			err = io.ErrUnexpectedEOF // the source is shorter than the manifest says
		} else if errors.Is(err, io.EOF) {
			err = nil
		}
		if err != nil {
			return done, errors.Join(err, out.Close())
		}
	}
	return done, out.Close()
}
