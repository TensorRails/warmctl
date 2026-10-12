package cli

import "testing"

func TestParseSize(t *testing.T) {
	for in, want := range map[string]int64{"0": 0, "1048576": 1 << 20, "64MiB": 64 << 20, "512 KiB": 512 << 10, "2GiB": 2 << 30} {
		if got, err := parseSize(in); err != nil || got != want {
			t.Errorf("parseSize(%q) = %d, %v; want %d", in, got, err, want)
		}
	}
	for _, in := range []string{"", "64MB", "-1", "1.5GiB", "MiB", "9223372036854775807GiB"} {
		if _, err := parseSize(in); err == nil {
			t.Errorf("parseSize(%q) accepted", in)
		}
	}
}
