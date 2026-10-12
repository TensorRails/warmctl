package restore

import (
	"bufio"
	"context"
	"fmt"
	"os"
	"path"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/TensorRails/warmctl/v2/internal/manifest"
)

const (
	// prefetchChunk is how much one prefetch read covers: big files are
	// read by several workers at once.
	prefetchChunk = 64 << 20
	// prefetchWorkers is the default number of parallel readers; network
	// file systems reach their bandwidth only with many requests in flight
	// (one stream of a 10 GiB/s NFS reads below 1 GiB/s).
	prefetchWorkers = 32
	// prefetchHeadroom is memory left to the restored processes and the
	// rest of the host when the prefetch budget is computed.
	prefetchHeadroom = 16 << 30
)

type prefetchJob struct {
	path   string
	offset int64
	size   int64
}

// prefetchPlan orders the reads: the CRIU images first, then the weights
// round-robin across ranks so every rank's reload finds its first parts in
// the page cache, in chunks of at most chunk bytes, stopping at budget bytes
// (budget <= 0: no limit).
func prefetchPlan(criu, weights []manifest.File, chunk, budget int64) []prefetchJob {
	var jobs []prefetchJob
	var total int64
	for _, f := range append(append([]manifest.File(nil), criu...), byRankRoundRobin(weights)...) {
		for off := int64(0); off < f.Size; off += chunk {
			n := min(chunk, f.Size-off)
			if budget > 0 && total+n > budget {
				return jobs
			}
			jobs = append(jobs, prefetchJob{f.Path, off, n})
			total += n
		}
	}
	return jobs
}

// byRankRoundRobin orders files one per rank directory in turn, keeping each
// rank's own order.
func byRankRoundRobin(files []manifest.File) []manifest.File {
	var order []string
	byRank := map[string][]manifest.File{}
	for _, f := range files {
		d := path.Dir(f.Path)
		if _, seen := byRank[d]; !seen {
			order = append(order, d)
		}
		byRank[d] = append(byRank[d], f)
	}
	out := make([]manifest.File, 0, len(files))
	for i := 0; len(out) < len(files); i++ {
		for _, d := range order {
			if i < len(byRank[d]) {
				out = append(out, byRank[d][i])
			}
		}
	}
	return out
}

// prefetchStats tells how the read-ahead went: bytes read, when the CRIU
// images were all read and when it finished, from its start.
type prefetchStats struct {
	start     time.Time
	bytes     atomic.Int64
	criuBytes int64
	criuDone  atomic.Int64 // nanoseconds after start, 0 until done
	done      atomic.Int64
}

func (s *prefetchStats) String() string {
	elapsed := time.Since(s.start)
	state := "still reading"
	if d := s.done.Load(); d > 0 {
		elapsed, state = time.Duration(d), "done"
	}
	gib := float64(s.bytes.Load()) / (1 << 30)
	out := fmt.Sprintf("%.1f GiB in %.1fs (%.2f GiB/s, %s)", gib, elapsed.Seconds(), gib/elapsed.Seconds(), state)
	if d := s.criuDone.Load(); d > 0 {
		out += fmt.Sprintf(", CRIU images (%.1f GiB) read by %.1fs", float64(s.criuBytes)/(1<<30), time.Duration(d).Seconds())
	}
	return out
}

// Prefetch reads a snapshot's CRIU images and its weights into the page
// cache with parallel readers while CRIU and cuda-checkpoint restore, so a
// cold restore from remote storage overlaps reading with the restore steps
// and the weights reload reads from memory. It reads at most what the host
// can cache besides the restored processes. By default every reader starts
// on the CRIU images, which CRIU needs first; WARMCTL_PREFETCH_ORDER=together
// starts half of them on the weights at once.
func Prefetch(ctx context.Context, dir string, m manifest.Manifest, st *prefetchStats) {
	var criu []manifest.File
	for _, f := range m.Files {
		if f.Role == manifest.RoleCRIU {
			criu = append(criu, f)
			st.criuBytes += f.Size
		}
	}
	budget := memAvailable() - st.criuBytes - prefetchHeadroom
	if budget < st.criuBytes {
		budget = st.criuBytes
	}
	workers := prefetchWorkers
	if n, err := strconv.Atoi(os.Getenv("WARMCTL_PREFETCH_WORKERS")); err == nil && n > 0 {
		workers = n
	}
	criuFirst := workers
	if os.Getenv("WARMCTL_PREFETCH_ORDER") == "together" {
		criuFirst = max(1, workers/2)
	}
	criuJobs := prefetchPlan(criu, nil, prefetchChunk, budget)
	weightJobs := prefetchPlan(nil, m.WeightLane(), prefetchChunk, budget-st.criuBytes)
	r := &reader{ctx: ctx, dir: dir, st: st}
	r.criuLeft.Store(int64(len(criuJobs)))
	if len(criuJobs) == 0 {
		st.criuDone.Store(1)
	}
	criuCh, weightCh := feed(ctx, criuJobs), feed(ctx, weightJobs)
	var wg sync.WaitGroup
	for i := range workers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			r.run(i < criuFirst, criuCh, weightCh)
		}()
	}
	wg.Wait()
	st.done.Store(int64(time.Since(st.start)))
}

// reader is one prefetch's shared state.
type reader struct {
	ctx      context.Context
	dir      string
	st       *prefetchStats
	criuLeft atomic.Int64
}

// run reads the CRIU images first when criuFirst, then weights, until both
// channels close.
func (r *reader) run(criuFirst bool, criuCh, weightCh <-chan prefetchJob) {
	buf := make([]byte, 8<<20)
	if criuFirst {
		for j := range criuCh {
			r.st.bytes.Add(readRange(r.ctx, filepath.Join(r.dir, j.path), j.offset, j.size, buf))
			if r.criuLeft.Add(-1) == 0 {
				r.st.criuDone.Store(int64(time.Since(r.st.start)))
			}
		}
	}
	for j := range weightCh {
		r.st.bytes.Add(readRange(r.ctx, filepath.Join(r.dir, j.path), j.offset, j.size, buf))
	}
}

// feed hands jobs out on a channel until they run out or ctx ends.
func feed(ctx context.Context, jobs []prefetchJob) <-chan prefetchJob {
	ch := make(chan prefetchJob)
	go func() {
		defer close(ch)
		for _, j := range jobs {
			select {
			case ch <- j:
			case <-ctx.Done():
				return
			}
		}
	}()
	return ch
}

// readRange reads size bytes of name from offset and drops them, returning
// how many it read. It first asks the kernel to read the whole range ahead,
// which puts many requests in flight instead of one per reader.
func readRange(ctx context.Context, name string, offset, size int64, buf []byte) int64 {
	f, err := os.Open(name)
	if err != nil {
		return 0
	}
	defer f.Close()
	willNeed(f, offset, size)
	done := int64(0)
	for done < size && ctx.Err() == nil {
		n, err := f.ReadAt(buf[:min(int64(len(buf)), size-done)], offset+done)
		done += int64(n)
		if err != nil {
			break
		}
	}
	return done
}

// memAvailable is MemAvailable of /proc/meminfo in bytes, 0 if unknown.
func memAvailable() int64 {
	f, err := os.Open("/proc/meminfo")
	if err != nil {
		return 0
	}
	defer f.Close()
	s := bufio.NewScanner(f)
	for s.Scan() {
		if rest, ok := strings.CutPrefix(s.Text(), "MemAvailable:"); ok {
			kb, _ := strconv.ParseInt(strings.TrimSuffix(strings.TrimSpace(rest), " kB"), 10, 64)
			return kb << 10
		}
	}
	return 0
}
