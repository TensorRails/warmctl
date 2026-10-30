package cuda

import (
	"context"
	"encoding/hex"
	"fmt"
	"strconv"
	"strings"

	"github.com/TensorRails/warmctl/v2/internal/process"
)

// Backend moves the CUDA state of one process. Session keeps the order of
// the processes, the barriers between phases and the state checks.
type Backend interface {
	Name() string
	State(ctx context.Context, pid int) (string, error)
	Lock(ctx context.Context, pid, timeoutMS int) error
	Checkpoint(ctx context.Context, pid int) error
	Restore(ctx context.Context, pid int, pairs []Pair) error
	Unlock(ctx context.Context, pid int) error
}

// Backend names: Select takes them, and auto picks the driver when it can.
const (
	BackendAuto   = "auto"
	BackendDriver = "driver"
	BackendCLI    = "cli"
)

// Select returns the backend a session should use. auto takes the driver
// API when this host's libcuda has it with a struct layout known here, else
// the cuda-checkpoint binary at helper; driver fails instead of falling
// back. why says what auto saw.
func Select(mode, helper string) (b Backend, why string, err error) {
	switch mode {
	case BackendCLI:
		return NewCLI(helper), "", nil
	case BackendDriver:
		d, err := NewDriver()
		if err != nil {
			return nil, "", err
		}
		return d, "", nil
	case BackendAuto, "":
		d, err := NewDriver()
		if err != nil {
			return NewCLI(helper), err.Error(), nil
		}
		return d, "", nil
	}
	return nil, "", fmt.Errorf("cuda: backend %q, want auto, driver or cli", mode)
}

// CLI runs NVIDIA's cuda-checkpoint once per process and call.
type CLI struct {
	Helper string
	run    func(ctx context.Context, args ...string) (string, error)
}

// NewCLI uses the helper binary at path.
func NewCLI(helper string) *CLI {
	c := &CLI{Helper: helper}
	c.run = func(ctx context.Context, args ...string) (string, error) {
		out, err := process.Capture(ctx, CallTimeout, c.Helper, args...)
		if err != nil {
			return "", fmt.Errorf("%w: %s", err, out.Stderr)
		}
		return strings.TrimSpace(out.Stdout), nil
	}
	return c
}

// Name is the backend as the logs show it.
func (c *CLI) Name() string { return BackendCLI }

func (c *CLI) call(ctx context.Context, pid int, args ...string) (string, error) {
	out, err := c.run(ctx, append([]string{"--pid", strconv.Itoa(pid)}, args...)...)
	if err != nil {
		return "", fmt.Errorf("cuda-checkpoint %s on %d: %w", strings.Join(args, " "), pid, err)
	}
	return out, nil
}

// State is the process's CUDA state: running, locked, checkpointed or failed.
func (c *CLI) State(ctx context.Context, pid int) (string, error) {
	return c.call(ctx, pid, "--get-state")
}

// Lock blocks the process's CUDA calls, waiting up to timeoutMS for work in flight.
func (c *CLI) Lock(ctx context.Context, pid, timeoutMS int) error {
	_, err := c.call(ctx, pid, "--action", "lock", "--timeout", strconv.Itoa(timeoutMS))
	return err
}

// Checkpoint moves the locked process's device memory to host memory.
func (c *CLI) Checkpoint(ctx context.Context, pid int) error {
	_, err := c.call(ctx, pid, "--action", "checkpoint")
	return err
}

// Restore puts the checkpointed process's memory back, on the pairs' target GPUs.
func (c *CLI) Restore(ctx context.Context, pid int, pairs []Pair) error {
	m, err := DeviceMap(pairs)
	if err != nil {
		return err
	}
	_, err = c.call(ctx, pid, "--action", "restore", "--device-map", m)
	return err
}

// Unlock lets the restored process's CUDA calls run again.
func (c *CLI) Unlock(ctx context.Context, pid int) error {
	_, err := c.call(ctx, pid, "--action", "unlock")
	return err
}

// ParseDeviceMap is DeviceMap's inverse.
func ParseDeviceMap(m string) ([]Pair, error) {
	var pairs []Pair
	for _, part := range strings.Split(m, ",") {
		source, target, ok := strings.Cut(part, "=")
		if !ok {
			return nil, fmt.Errorf("cuda: device map entry %q is not SOURCE=TARGET", part)
		}
		pairs = append(pairs, Pair{Source: source, Target: target})
	}
	if _, err := DeviceMap(pairs); err != nil {
		return nil, err
	}
	return pairs, nil
}

// uuidBytes is the CUuuid of a GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx name.
func uuidBytes(name string) ([16]byte, error) {
	var out [16]byte
	digits := strings.ReplaceAll(strings.TrimPrefix(name, "GPU-"), "-", "")
	if !strings.HasPrefix(name, "GPU-") || len(digits) != 32 {
		return out, fmt.Errorf("cuda: %q is not a GPU UUID", name)
	}
	if _, err := hex.Decode(out[:], []byte(digits)); err != nil {
		return out, fmt.Errorf("cuda: %q is not a GPU UUID: %w", name, err)
	}
	return out, nil
}
