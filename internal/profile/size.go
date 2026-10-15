package profile

import (
	"fmt"
	"strconv"
	"strings"
)

// ParseSize reads "1TiB", "64GiB", "512MiB", "16KiB", "1024" (bytes) and the
// decimal forms "1TB", "64GB". Case-insensitive, spaces allowed.
func ParseSize(s string) (int64, error) {
	t := strings.ToLower(strings.ReplaceAll(strings.TrimSpace(s), " ", ""))
	if t == "" {
		return 0, fmt.Errorf("empty size")
	}
	units := []struct {
		suffix string
		mult   int64
	}{
		{"tib", 1 << 40}, {"gib", 1 << 30}, {"mib", 1 << 20}, {"kib", 1 << 10},
		{"tb", 1e12}, {"gb", 1e9}, {"mb", 1e6}, {"kb", 1e3},
		{"t", 1 << 40}, {"g", 1 << 30}, {"m", 1 << 20}, {"k", 1 << 10}, {"b", 1},
	}
	for _, u := range units {
		if strings.HasSuffix(t, u.suffix) {
			n, err := strconv.ParseFloat(strings.TrimSuffix(t, u.suffix), 64)
			if err != nil || n < 0 {
				return 0, fmt.Errorf("invalid size %q", s)
			}
			return int64(n * float64(u.mult)), nil
		}
	}
	n, err := strconv.ParseInt(t, 10, 64)
	if err != nil || n < 0 {
		return 0, fmt.Errorf("invalid size %q", s)
	}
	return n, nil
}
