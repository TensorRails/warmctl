package profile

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// HardwareSchema identifies hardware/hardware.yaml.
const HardwareSchema = "warmctl.hardware/v1"

// Hardware is one qualified GPU model. Code branches only on these fields,
// never on a product name.
type Hardware struct {
	ID               string `yaml:"id" json:"id"`
	NvidiaSMIName    string `yaml:"nvidia_smi_name" json:"nvidia_smi_name"`
	MemoryMiB        int64  `yaml:"memory_mib" json:"memory_mib"`
	Arch             string `yaml:"arch" json:"arch"`
	NVLink           bool   `yaml:"nvlink" json:"nvlink"`
	TopologyClass    string `yaml:"topology_class" json:"topology_class"`
	DeviceConfigName string `yaml:"device_config_name" json:"device_config_name"`
	DriverMin        string `yaml:"driver_min" json:"driver_min"`
}

// HardwareTable is hardware/hardware.yaml.
type HardwareTable struct {
	Schema string     `yaml:"schema" json:"schema"`
	Rows   []Hardware `yaml:"rows" json:"rows"`
}

// Hardware reads the embedded table plus a hardware.yaml in each profile
// directory, whose rows extend it.
func (l Loader) Hardware() (HardwareTable, error) {
	var t HardwareTable
	if err := l.decodeAsset("hardware/hardware.yaml", &t); err != nil {
		return t, err
	}
	if t.Schema != HardwareSchema {
		return t, fmt.Errorf("hardware.yaml: schema %q, want %s", t.Schema, HardwareSchema)
	}
	for _, dir := range l.Dirs {
		f, err := os.Open(filepath.Join(dir, "hardware.yaml"))
		if err != nil {
			continue
		}
		var extra HardwareTable
		err = Decode(f, &extra)
		f.Close()
		if err != nil || extra.Schema != HardwareSchema {
			return t, fmt.Errorf("%s/hardware.yaml: %v (schema %q)", dir, err, extra.Schema)
		}
		t.Rows = append(t.Rows, extra.Rows...)
	}
	seen := map[string]bool{}
	for _, r := range t.Rows {
		if r.ID == "" || r.NvidiaSMIName == "" || r.MemoryMiB <= 0 || r.Arch == "" || seen[r.ID] {
			return t, fmt.Errorf("hardware.yaml: invalid or duplicate row %q", r.ID)
		}
		seen[r.ID] = true
	}
	return t, nil
}

// ByID finds a row.
func (t HardwareTable) ByID(id string) (Hardware, bool) {
	for _, r := range t.Rows {
		if r.ID == id {
			return r, true
		}
	}
	return Hardware{}, false
}

// ByName finds the row for an nvidia-smi product name.
func (t HardwareTable) ByName(name string) (Hardware, bool) {
	for _, r := range t.Rows {
		if strings.EqualFold(strings.TrimSpace(r.NvidiaSMIName), strings.TrimSpace(name)) {
			return r, true
		}
	}
	return Hardware{}, false
}

// Supports reports whether a profile lists the hardware row.
func (p Profile) Supports(row Hardware) bool {
	for _, id := range p.Hardware {
		if id == row.ID {
			return true
		}
	}
	return false
}
