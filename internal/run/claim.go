package run

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"

	"github.com/TensorRails/warmctl/v2/internal/docker"
)

// Claim makes dir this process's working directory for a run. A directory
// left by a warmctl process that is gone (it crashed or was killed) is
// removed first; one whose owner still runs is refused.
func Claim(dir string) error {
	if err := os.MkdirAll(filepath.Dir(dir), 0o755); err != nil {
		return err
	}
	if raw, err := os.ReadFile(filepath.Join(dir, "owner.pid")); err == nil {
		if pid, _ := strconv.Atoi(strings.TrimSpace(string(raw))); pid > 0 && alive(pid) {
			return fmt.Errorf("%s belongs to running process %d", dir, pid)
		}
		if err := os.RemoveAll(dir); err != nil {
			return err
		}
	} else if _, err := os.Stat(dir); err == nil {
		return fmt.Errorf("%s exists and has no owner", dir)
	}
	if err := os.Mkdir(dir, 0o700); err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(dir, "owner.pid"), []byte(strconv.Itoa(os.Getpid())+"\n"), 0o644)
}

func alive(pid int) bool {
	err := syscall.Kill(pid, 0)
	return err == nil || errors.Is(err, syscall.EPERM)
}

// ClearName removes a stopped container that holds name (left by a crashed
// run) and refuses a running one.
func ClearName(ctx context.Context, dk *docker.Client, name string) error {
	st, err := dk.Inspect(ctx, name)
	if docker.IsNotFound(err) {
		return nil
	}
	if err != nil {
		return err
	}
	if st.State.Running {
		return fmt.Errorf("container %s is running: another warmctl run uses this id", name)
	}
	return dk.Remove(ctx, st.ID)
}
