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
	// Vendor is "amd" for an AMD Instinct row, which names its model in
	// ProductName (as warmctl reads it from the KFD topology) instead of
	// NvidiaSMIName; empty for NVIDIA's rows.
	Vendor      string `yaml:"vendor,omitempty" json:"vendor,omitempty"`
	ProductName string `yaml:"product_name,omitempty" json:"product_name,omitempty"`
}

// Name is the product name a GPU of the row reports.
func (r Hardware) Name() string {
	if r.Vendor == "amd" {
		return r.ProductName
	}
	return r.NvidiaSMIName
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
		for _, r := range extra.Rows {
			// A row written for an older release that later shipped it
			// verbatim is not a conflict.
			if known, ok := t.ByID(r.ID); ok && known == r {
				continue
			}
			t.Rows = append(t.Rows, r)
		}
	}
	seen := map[string]bool{}
	for _, r := range t.Rows {
		if r.ID == "" || r.Name() == "" || r.MemoryMiB <= 0 || r.Arch == "" || seen[r.ID] ||
			(r.Vendor != "" && r.Vendor != "amd") {
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

// ByName finds the row for a product name: nvidia-smi's, or an AMD GPU's.
func (t HardwareTable) ByName(name string) (Hardware, bool) {
	for _, r := range t.Rows {
		if strings.EqualFold(strings.TrimSpace(r.Name()), strings.TrimSpace(name)) {
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

// VendorOf is the GPU vendor a profile is written for: "amd" when every
// hardware row it lists is an AMD one, else "" (NVIDIA).
func (t HardwareTable) VendorOf(p Profile) string {
	if len(p.Hardware) == 0 {
		return ""
	}
	for _, id := range p.Hardware {
		if r, ok := t.ByID(id); !ok || r.Vendor != "amd" {
			return ""
		}
	}
	return "amd"
}

// DefaultsFor are the defaults p is resolved over when inspected (warmctl
// profile show, argv, check): the ROCm layer for a profile of AMD rows. A
// bake takes the vendor from its host instead.
func (l Loader) DefaultsFor(p Profile) (Profile, error) {
	table, err := l.Hardware()
	if err != nil {
		return Profile{}, err
	}
	l.Vendor = table.VendorOf(p)
	return l.Defaults()
}
