package oci

import (
	"fmt"
	"os"
	"time"

	"github.com/TensorRails/warmctl/v2/internal/manifest"
	"github.com/TensorRails/warmctl/v2/internal/store"
)

// meterEvery is how often a transfer reports its progress.
var meterEvery = 5 * time.Second

// meter logs every meterEvery how much of total bytes t has counted, the
// average rate and the time left, until the returned stop is called.
func meter(o Options, verb string, total int64, t *tally) (stop func()) {
	if o.Log == nil || total == 0 {
		return func() {}
	}
	start, done := time.Now(), make(chan struct{})
	go func() {
		tick := time.NewTicker(meterEvery)
		defer tick.Stop()
		for {
			select {
			case <-done:
				return
			case <-tick.C:
				t.mu.Lock()
				n := t.bytes
				t.mu.Unlock()
				rate := float64(n) / time.Since(start).Seconds()
				left := "unknown"
				if rate > 0 {
					left = (time.Duration(float64(total-n)/rate) * time.Second).Round(time.Second).String()
				}
				o.logf("%s %.1f of %.1f GiB, %.0f MB/s, ETA %s", verb, gib(n), gib(total), rate/1e6, left)
			}
		}
	}()
	return func() { close(done) }
}

func gib(n int64) float64 { return float64(n) / (1 << 30) }

// roomFor checks, before anything is downloaded, that the store's file
// system can hold the files it lacks, with 5% to spare.
func roomFor(s store.Store, files []manifest.File) error {
	var need int64
	seen := map[string]bool{}
	for _, f := range files {
		if !seen[f.SHA256] && !s.Has(f.SHA256) {
			need += f.Size
		}
		seen[f.SHA256] = true
	}
	if need == 0 {
		return nil
	}
	if err := os.MkdirAll(s.Blobs(), 0o700); err != nil {
		return err
	}
	free, err := freeBytes(s.Blobs())
	if err != nil {
		return nil // unknown: the copy itself will tell
	}
	if free < need+need/20 {
		return fmt.Errorf("oci: the snapshot needs %.1f GiB more in %s, %.1f GiB are free", gib(need), s.Root, gib(free))
	}
	return nil
}
