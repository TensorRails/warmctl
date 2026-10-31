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

// arrivalWait bounds how long a stage waits for a file the manifest lists:
// a RunPod Global Volume shows another pod's writes seconds late, and its
// manifest.json may show before the files.
var arrivalWait = 2 * time.Minute

// Stage copies the snapshot in dir to dst, every file of m and its
// manifest.json, many ranges in flight, and returns how many bytes it read
// from dir. A file stored as zstd frames is decoded frame by frame on the
// workers, so dst always holds the files as they are. It is for a snapshot directory on a FUSE file system (an S3
// bucket behind geesefs, as a RunPod Global Volume, or MooseFS, as its
// network volumes): FUSE keeps no page cache the prefetch could fill for
// later readers, and the restore's readers (CRIU or dmtcp_restart, the
// weights reload) read one stream each. The files go lane by lane, those
// the receiver starts with, those the process restore reads, the weights,
// and done, when set, hears of each file once it is complete: the restore
// runs while the weights still come. A copy whose manifest.json matches is
// complete and is reused: manifest.json is written last. An encoded
// snapshot is staged wherever it is.
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
	left := map[string]*atomic.Int64{}
	for _, f := range append(append([]manifest.File(nil), first...), weights...) {
		info, err := arrived(ctx, filepath.Join(dir, f.Path), f.Stored())
		if err != nil {
			return 0, err
		}
		if err := create(info, filepath.Join(dst, f.Path), f.Size); err != nil {
			return 0, err
		}
		if f.Size == 0 {
			done(f.Path)
			continue
		}
		left[f.Path] = new(atomic.Int64)
		left[f.Path].Store(f.Size)
	}
	jobs := stagePlan(first, weights)

	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	ch := feed(ctx, jobs)
	var read atomic.Int64
	var once sync.Once
	var firstErr error
	var wg sync.WaitGroup
	for range stageWorkers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			w := &stager{buf: make([]byte, 8<<20)}
			defer w.close()
			for j := range ch {
				n, got, err := w.run(filepath.Join(dir, j.path), filepath.Join(dst, j.path), j)
				read.Add(got)
				if err != nil {
					once.Do(func() { firstErr = err; cancel() })
					continue
				}
				if left[j.path].Add(-n) == 0 {
					done(j.path)
				}
			}
		}()
	}
	wg.Wait()
	if firstErr != nil {
		return read.Load(), firstErr
	}
	if err := ctx.Err(); err != nil {
		return read.Load(), err
	}
	return read.Load(), os.WriteFile(filepath.Join(dst, "manifest.json"), raw, 0o644)
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
			jobs = append(jobs, prefetchJob{path: f.Path, offset: offset, size: size, stored: at, storedSize: n})
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
	if w.out, err = w.dec.Decode(frame, w.out, j.size); err != nil {
		return 0, int64(n), fmt.Errorf("%s at %d: %w", src, j.stored, err)
	}
	out, err := os.OpenFile(dst, os.O_WRONLY, 0)
	if err != nil {
		return 0, int64(n), err
	}
	if _, err := out.WriteAt(w.out, j.offset); err != nil {
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
