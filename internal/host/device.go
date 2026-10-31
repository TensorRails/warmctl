package host

import (
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

// DevicePath is the GPU's device node, /dev/nvidiaN. N is the minor number the
// driver gives it in /proc/driver/nvidia/gpus/<bus>/information, in the
// host's order: the same GPU model is /dev/nvidia3 in one pod and
// /dev/nvidia0 in the next.
func DevicePath(busID string) (string, error) {
	raw, err := os.ReadFile(filepath.Join("/proc/driver/nvidia/gpus", normalizeBus(busID), "information"))
	if err != nil {
		return "", err
	}
	minor, err := parseDeviceMinor(string(raw))
	if err != nil {
		return "", fmt.Errorf("GPU %s: %w", busID, err)
	}
	return "/dev/nvidia" + strconv.Itoa(minor), nil
}

func parseDeviceMinor(info string) (int, error) {
	for _, line := range strings.Split(info, "\n") {
		if key, value, ok := strings.Cut(line, ":"); ok && strings.TrimSpace(key) == "Device Minor" {
			return strconv.Atoi(strings.TrimSpace(value))
		}
	}
	return 0, fmt.Errorf("no Device Minor line")
}
