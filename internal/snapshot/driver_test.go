package snapshot

import "testing"

// Multi-GPU engines run on the host's libcuda from driver branch 595 on;
// single-GPU ones and older branches keep the image's forward-compatible one.
func TestHostUMD(t *testing.T) {
	cases := []struct {
		driver string
		ranks  int
		want   bool
	}{
		{"580.173.02", 8, false}, {"570.195.03", 2, false}, {"595.91.07", 2, true}, {"610.57.04", 8, true},
		{"610.57.04", 1, false}, {"615.71.09", 4, true}, {"", 8, false},
	}
	for _, c := range cases {
		if got := hostUMD(c.driver, c.ranks); got != c.want {
			t.Errorf("hostUMD(%q, %d) = %v, want %v", c.driver, c.ranks, got, c.want)
		}
	}
}
