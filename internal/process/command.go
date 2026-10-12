// Package process runs bounded external tools without shell interpretation.
package process

import (
	"context"
	"errors"
	"fmt"
	"os/exec"
	"strings"
	"syscall"
	"time"
)

// Output is the trimmed stdout and stderr of a finished tool.
type Output struct {
	Stdout string `json:"stdout"`
	Stderr string `json:"stderr"`
}

type buffer struct {
	bytes     []byte
	limit     int
	truncated bool
}

// ErrOutputLimit means the captured output is incomplete and must not be
// parsed as a successful tool response.
var ErrOutputLimit = errors.New("external tool output exceeded capture limit")

func (b *buffer) Write(p []byte) (int, error) {
	n := len(p)
	if n > b.limit-len(b.bytes) {
		b.truncated = true
	}
	if left := b.limit - len(b.bytes); left > 0 {
		b.bytes = append(b.bytes, p[:min(left, n)]...)
	}
	return n, nil
}

// Capture runs executable with args under timeout. The child gets its own
// process group so cancellation kills forked helpers too; output is capped
// at 64 KiB stdout and 16 KiB stderr.
func Capture(ctx context.Context, timeout time.Duration, executable string, args ...string) (Output, error) {
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	cmd := exec.CommandContext(ctx, executable, args...)
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	cmd.Cancel = func() error {
		err := syscall.Kill(-cmd.Process.Pid, syscall.SIGKILL)
		if errors.Is(err, syscall.ESRCH) {
			return nil
		}
		return err
	}
	cmd.WaitDelay = time.Second
	stdout, stderr := buffer{limit: 64 << 10}, buffer{limit: 16 << 10}
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	err := cmd.Run()
	if errors.Is(err, exec.ErrWaitDelay) && cmd.Process != nil {
		_ = syscall.Kill(-cmd.Process.Pid, syscall.SIGKILL)
	}
	out := Output{strings.TrimSpace(string(stdout.bytes)), strings.TrimSpace(string(stderr.bytes))}
	if ctx.Err() != nil {
		return out, ctx.Err()
	}
	if stdout.truncated || stderr.truncated {
		return out, errors.Join(ErrOutputLimit, err)
	}
	if err != nil {
		return out, fmt.Errorf("%s: %w", executable, err)
	}
	return out, nil
}
