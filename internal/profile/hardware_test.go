package profile_test

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/TensorRails/warmctl/v2"
	"github.com/TensorRails/warmctl/v2/internal/profile"
)

// A profile directory may repeat a shipped row word for word, as one written
// before the release that shipped it does; a different row under the same id
// is still an error.
func TestHardwareDirRepeatsShippedRow(t *testing.T) {
	row := `schema: warmctl.hardware/v1
rows:
  - id: rtx-pro-6000-96gb
    nvidia_smi_name: "NVIDIA RTX PRO 6000 Blackwell Server Edition"
    memory_mib: %s
    arch: sm_120
    nvlink: false
    topology_class: pcie
    device_config_name: NVIDIA_RTX_PRO_6000_Blackwell_Server_Edition
    driver_min: "580"
`
	for _, tc := range []struct {
		memory string
		ok     bool
	}{{"97887", true}, {"98304", false}} {
		dir := t.TempDir()
		if err := os.WriteFile(filepath.Join(dir, "hardware.yaml"), []byte(strings.Replace(row, "%s", tc.memory, 1)), 0o644); err != nil {
			t.Fatal(err)
		}
		table, err := profile.Loader{Assets: warmctl.Assets, Dirs: []string{dir}}.Hardware()
		if (err == nil) != tc.ok {
			t.Fatalf("memory %s: err %v", tc.memory, err)
		}
		if !tc.ok {
			continue
		}
		n := 0
		for _, r := range table.Rows {
			if r.ID == "rtx-pro-6000-96gb" {
				n++
			}
		}
		if n != 1 {
			t.Errorf("rtx-pro-6000-96gb listed %d times", n)
		}
	}
}
