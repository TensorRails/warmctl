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
)

// stageWorkers is how many ranges a stage copies at once: object storage
// behind FUSE reaches its bandwidth only with many requests in flight.
const stageWorkers = 32

// arrivalWait bounds how long a stage waits for a file the manifest lists:
// a RunPod Global Volume shows another pod's writes seconds late, and its
// manifest.json may show before the files.
var arrivalWait = 2 * time.Minute

// Stage copies the snapshot in dir to dst, every file of m and its
// manifest.json, many ranges in flight, and returns how many bytes it
// copied. It is for a snapshot directory on a FUSE file system (an S3
// bucket behind geesefs, as a RunPod Global Volume): FUSE keeps no page
// cache the prefetch could fill for later readers, and the restore's readers
// (CRIU or dmtcp_restart, the weights reload) read one stream each. The
// process images go first. A copy whose manifest.json matches is complete
// and is reused: manifest.json is written last.
func Stage(ctx context.Context, dir, dst string, m manifest.Manifest) (int64, error) {
	raw, err := os.ReadFile(filepath.Join(dir, "manifest.json"))
	if err != nil {
		return 0, err
	}
	if have, err := os.ReadFile(filepath.Join(dst, "manifest.json")); err == nil && bytes.Equal(have, raw) {
		return 0, nil
	}
	if err := os.RemoveAll(dst); err != nil {
		return 0, err
	}
	var images, rest []manifest.File
	for _, f := range m.Files {
		if f.Role == manifest.RoleCRIU {
			images = append(images, f)
		} else {
			rest = append(rest, f)
		}
		info, err := arrived(ctx, filepath.Join(dir, f.Path), f.Size)
		if err != nil {
			return 0, err
		}
		if err := create(info, filepath.Join(dst, f.Path), f.Size); err != nil {
			return 0, err
		}
	}
	jobs := prefetchPlan(append(images, rest...), nil, prefetchChunk, 0)

	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	ch := feed(ctx, jobs)
	var copied atomic.Int64
	var once sync.Once
	var first error
	var wg sync.WaitGroup
	for range stageWorkers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			buf := make([]byte, 8<<20)
			for j := range ch {
				n, err := copyRange(filepath.Join(dir, j.path), filepath.Join(dst, j.path), j.offset, j.size, buf)
				copied.Add(n)
				if err != nil {
					once.Do(func() { first = err; cancel() })
				}
			}
		}()
	}
	wg.Wait()
	if first != nil {
		return copied.Load(), first
	}
	if err := ctx.Err(); err != nil {
		return copied.Load(), err
	}
	return copied.Load(), os.WriteFile(filepath.Join(dst, "manifest.json"), raw, 0o644)
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
