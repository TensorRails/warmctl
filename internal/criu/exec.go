package criu

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/TensorRails/warmctl/v2/internal/process"
)

// Timeout bounds one CRIU invocation.
const Timeout = 10 * time.Minute

// PrepareWork creates a fresh work directory with the empty plugin
// directory CRIU is pointed at. Reusing one would hide an older log.
func PrepareWork(dir string) error {
	if !filepath.IsAbs(dir) {
		return errors.New("criu: work directory must be absolute")
	}
	if err := os.Mkdir(dir, 0o700); err != nil {
		return err
	}
	return os.Mkdir(filepath.Join(dir, "plugins"), 0o700)
}

// Dump checkpoints the tree rooted at tree into images with the CRIU at
// program, whose version is version (criu --version). CRIU kills the tree
// afterwards; the caller reaps it.
func Dump(ctx context.Context, program, version string, tree int, images, work string) error {
	args := DumpArgs(tree, images, work, SkipsNetworkLock(version))
	if _, err := process.Capture(ctx, Timeout, program, args...); err != nil {
		return fmt.Errorf("criu dump: %w\n%s", err, logTail(filepath.Join(work, "dump.log")))
	}
	return nil
}

// Restore brings images back, every task stopped, and returns the root PID.
func Restore(ctx context.Context, program, images, work string) (int, error) {
	if _, err := process.Capture(ctx, Timeout, program, RestoreArgs(images, work)...); err != nil {
		return 0, fmt.Errorf("criu restore: %w\n%s", err, logTail(filepath.Join(work, "restore.log")))
	}
	raw, err := os.ReadFile(filepath.Join(work, "root.pid"))
	if err != nil {
		return 0, err
	}
	pid, err := strconv.Atoi(strings.TrimSpace(string(raw)))
	if err != nil || pid < 2 {
		return 0, fmt.Errorf("criu restore: bad pidfile %q", raw)
	}
	return pid, nil
}

func logTail(name string) string {
	raw, err := os.ReadFile(name)
	if err != nil {
		return ""
	}
	if len(raw) > 8000 {
		raw = raw[len(raw)-8000:]
	}
	return string(raw)
}
