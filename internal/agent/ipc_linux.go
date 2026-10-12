//go:build linux

package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/TensorRails/warmctl/v2/internal/criu"
	"github.com/TensorRails/warmctl/v2/internal/cuda"
	"github.com/TensorRails/warmctl/v2/internal/interpose"
	"github.com/TensorRails/warmctl/v2/internal/process"
)

// CoordinatorTimeout bounds one cuInterpose checkpoint or restore: it moves
// the shared allocations' bytes, tens to hundreds of MiB per process.
const CoordinatorTimeout = 5 * time.Minute

// chooseIPC decides, before the engine starts, whether it runs under
// cuInterpose: ipc interpose requires it and refuses GPUs it cannot serve;
// auto takes it when every GPU can hold it and otherwise runs native.
func (a *agent) chooseIPC(ctx context.Context) (bool, string, error) {
	if a.cfg.IPC == "" || a.cfg.IPC == interpose.Native {
		return false, "ipc native", nil
	}
	out, err := process.Capture(ctx, time.Minute, "python3", "-m", "warmctl_vllm.devices")
	if err != nil {
		return false, "", fmt.Errorf("device probe: %w: %s", err, out.Stderr)
	}
	devices, err := interpose.ParseDevices([]byte(out.Stdout))
	if err != nil {
		return false, "", err
	}
	if err := interpose.Supported(devices); err != nil {
		if a.cfg.IPC == interpose.Interpose {
			return false, "", fmt.Errorf("ipc interpose: %w", err)
		}
		return false, "ipc auto: " + err.Error(), nil
	}
	return true, fmt.Sprintf("ipc %s: %d GPUs with VMM", a.cfg.IPC, len(devices)), nil
}

// ipcPrepare has the coordinator save the bytes of every cross-process
// allocation into host memory and release the sharing, every address kept,
// so each process can be checkpointed on its own.
func (a *agent) ipcPrepare(ctx context.Context) (any, error) {
	if !a.interposed {
		return IPCResult{Reason: a.ipcReason}, nil
	}
	start := time.Now()
	tree, err := criu.VisibleTree(a.enginePID())
	if err != nil {
		return nil, err
	}
	owners, err := cuda.Owners(tree.PIDs())
	if err != nil {
		return nil, err
	}
	res := IPCResult{Interposed: true, Reason: a.ipcReason}
	res.Participants = interpose.Participants(owners, func(p string) bool {
		_, err := os.Stat(p)
		return err == nil
	})
	if len(res.Participants) > 0 {
		dir := filepath.Join(a.cfg.Out, "ipc")
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return nil, err
		}
		if err := a.coordinate(ctx, "prepare", dir, res.Participants); err != nil {
			return nil, err
		}
	}
	res.Seconds = time.Since(start).Seconds()
	return res, nil
}

// ipcRestore re-creates the shared allocations at their addresses after
// the CUDA restore and loads their bytes back.
func (a *agent) ipcRestore(ctx context.Context, raw json.RawMessage) (any, error) {
	var args IPCRestoreArgs
	if err := json.Unmarshal(raw, &args); err != nil {
		return nil, err
	}
	start := time.Now()
	if err := a.coordinate(ctx, "restore", interpose.GuestDir, args.Processes); err != nil {
		return nil, err
	}
	return IPCResult{Interposed: true, Seconds: time.Since(start).Seconds()}, nil
}

func (a *agent) coordinate(ctx context.Context, action, dir string, pids []int) error {
	argv := interpose.Command(a.cfg.IPCCoordinator, action, dir, pids)
	out, err := process.Capture(ctx, CoordinatorTimeout, argv[0], argv[1:]...)
	if err != nil {
		return fmt.Errorf("cuinterpose %s: %w: %s", action, err, out.Stderr)
	}
	return nil
}
