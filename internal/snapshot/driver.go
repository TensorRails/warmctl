package snapshot

import (
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"

	"github.com/TensorRails/warmctl/v2/internal/manifest"
	"github.com/TensorRails/warmctl/v2/internal/run"
)

// DriverCheckEnv is set in the engine at build time: it makes the NVIDIA driver
// skip its client/kernel version comparison. libcuda remembers the kernel
// module version it met at init and presents it again when cuda-checkpoint
// restores, so without it a snapshot restores only under the exact driver it
// was built on. With it, a restore under another release of the same branch
// works; another branch is still refused by the kernel module.
const DriverCheckEnv = "__RM_NO_VERSION_CHECK"

// hostUMDLibraryPath puts the host's user-mode driver (the libcuda the
// container toolkit mounts) ahead of the image's forward-compatible one.
const hostUMDLibraryPath = "/usr/lib/x86_64-linux-gnu:/usr/local/nvidia/lib64:/usr/local/cuda/lib64"

// hostUMD reports whether the engine should run on the host's libcuda rather
// than the image's forward-compatible one: for multi-GPU engines under driver
// branch 595 and later. Under a 610 kernel module the image's 615 libcuda
// could not checkpoint or restore processes that share GPU memory (custom
// all-reduce: CUDA_ERROR_UNKNOWN) while the host's 610 could. Single-GPU
// engines keep the image's libcuda, which restores under 610 and also JITs
// PTX built with the image's newer CUDA (MiMo's kernels do not load under the
// 610 libcuda). Under 580 the image's libcuda is the proven one, and the r580
// user-mode driver cannot restore multicast groups (ai-dynamo/snapshot#525).
func hostUMD(driver string, ranks int) bool {
	branch, err := strconv.Atoi(manifest.DriverBranch(driver))
	return err == nil && branch >= 595 && ranks > 1
}

// driverFiles copies the host driver libraries the dumped processes map
// (the container toolkit mounts them under the host's version, NVML among
// them) into dir, so a restore under another driver release can put them
// back at the same paths. It returns those paths.
func driverFiles(criuDir, driver, dir string) ([]string, error) {
	img, err := os.ReadFile(filepath.Join(criuDir, "files.img"))
	if err != nil {
		return nil, err
	}
	re := regexp.MustCompile(`/[A-Za-z0-9_./+-]+\.so\.` + regexp.QuoteMeta(driver))
	seen := map[string]bool{}
	for _, p := range re.FindAll(img, -1) {
		seen[string(p)] = true
	}
	paths := make([]string, 0, len(seen))
	for p := range seen {
		paths = append(paths, p)
	}
	sort.Strings(paths)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, err
	}
	for _, p := range paths {
		if err := run.CopyFile(p, filepath.Join(dir, filepath.Base(p)), 0o644); err != nil {
			return nil, fmt.Errorf("driver library %s the engine maps: %w", p, err)
		}
	}
	return paths, nil
}
