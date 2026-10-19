package cli

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/TensorRails/warmctl/v2/internal/host"
)

// fakeGPUs is an NVML session on a host with one GPU per name.
type fakeGPUs []string

func (f fakeGPUs) Driver() (string, error) { return "580.95.05", nil }
func (f fakeGPUs) Count() (int, error)     { return len(f), nil }
func (f fakeGPUs) GPU(i int) (host.GPU, error) {
	return host.GPU{UUID: fmt.Sprintf("GPU-%d", i), Name: f[i], PCIBusID: fmt.Sprintf("00000000:%02X:00.0", 0x18+i)}, nil
}
func (f fakeGPUs) Processes(int) (int, error)    { return 0, nil }
func (f fakeGPUs) NVLink(int, int) (bool, error) { return false, nil }
func (f fakeGPUs) Path(int, int) (string, error) { return "PIX", nil }
func (f fakeGPUs) Close() error                  { return nil }

func useGPUs(t *testing.T, names ...string) {
	t.Helper()
	old := host.Open
	host.Open = func() (host.Session, error) {
		if len(names) == 0 {
			return nil, errors.New("nvml: no driver")
		}
		return fakeGPUs(names), nil
	}
	t.Cleanup(func() { host.Open = old })
}

// rtx4090Dir is a profile directory whose hardware.yaml adds a GPU model.
func rtx4090Dir(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	row := `schema: warmctl.hardware/v1
rows:
  - id: rtx-4090-24gb
    nvidia_smi_name: "NVIDIA GeForce RTX 4090"
    memory_mib: 24564
    arch: sm_89
    nvlink: false
    topology_class: pcie
    device_config_name: NVIDIA_GeForce_RTX_4090
    driver_min: "580"
`
	if err := os.WriteFile(filepath.Join(dir, "hardware.yaml"), []byte(row), 0o644); err != nil {
		t.Fatal(err)
	}
	return dir
}

// hardwareRow is the line of the hardware table whose first field is id.
func hardwareRow(out, id string) string {
	for _, l := range strings.Split(out, "\n") {
		if f := strings.Fields(l); len(f) > 0 && f[0] == id {
			return l
		}
	}
	return ""
}

// The table lists the shipped rows and a --dir's, and marks this host's.
func TestHardware(t *testing.T) {
	dir := rtx4090Dir(t)
	for _, c := range []struct {
		gpus             []string
		args             []string
		marked, footnote string
	}{
		{nil, nil, "", ""},
		{[]string{"NVIDIA RTX PRO 6000 Blackwell Server Edition"}, nil, "rtx-pro-6000-96gb", ""},
		{[]string{"NVIDIA GeForce RTX 4090", "NVIDIA GeForce RTX 4090"}, nil, "", "2 x NVIDIA GeForce RTX 4090, no row"},
		{[]string{"NVIDIA GeForce RTX 4090", "NVIDIA GeForce RTX 4090"}, []string{"--dir", dir}, "rtx-4090-24gb", ""},
	} {
		useGPUs(t, c.gpus...)
		var stdout, stderr bytes.Buffer
		if code := Run(context.Background(), append([]string{"hardware"}, c.args...), &stdout, &stderr, "test"); code != 0 {
			t.Fatalf("%v %v: exit %d, %q", c.gpus, c.args, code, stderr.String())
		}
		out := stdout.String()
		if !strings.HasPrefix(out, "ID ") || strings.Contains(out, "THIS HOST") != (c.marked != "") {
			t.Fatalf("header:\n%s", out)
		}
		if strings.Contains(out, " \n") {
			t.Errorf("trailing blanks:\n%q", out)
		}
		want := []string{"h100-80gb-sxm", "h200-141gb", "l40s-48gb", "b200-180gb", "rtx-pro-6000-96gb"}
		if len(c.args) > 0 {
			want = append(want, "rtx-4090-24gb")
		}
		for _, id := range want {
			row := hardwareRow(out, id)
			if row == "" {
				t.Errorf("%v: no row %s:\n%s", c.args, id, out)
			}
			if mark := strings.HasSuffix(row, "GPU") || strings.HasSuffix(row, "GPUs"); mark != (id == c.marked) {
				t.Errorf("%v on %v: row %q marked %t", c.args, c.gpus, row, mark)
			}
		}
		got := hardwareRow(out, "rtx-pro-6000-96gb")
		if !strings.Contains(got, "NVIDIA RTX PRO 6000 Blackwell Server Edition") ||
			!strings.Contains(got, "97887 MiB") || !strings.Contains(got, "sm_120") || !strings.Contains(got, "pcie") {
			t.Errorf("rtx-pro-6000-96gb row %q", got)
		}
		if c.marked == "rtx-4090-24gb" && !strings.HasSuffix(hardwareRow(out, c.marked), "2 GPUs") {
			t.Errorf("rtx-4090-24gb row %q", hardwareRow(out, c.marked))
		}
		if strings.Contains(out, "this host:") != (c.footnote != "") || !strings.Contains(out, c.footnote) {
			t.Errorf("%v %v: footnote, want %q:\n%s", c.gpus, c.args, c.footnote, out)
		}
	}
}

// doctor names the host's hardware row, or warns with the nvidia-smi name
// of a GPU that has none and passes.
func TestDoctorHardware(t *testing.T) {
	dir := rtx4090Dir(t)
	for _, c := range []struct {
		gpu  string
		args []string
		want string
	}{
		{"NVIDIA RTX PRO 6000 Blackwell Server Edition", nil,
			"ok   hardware         rtx-pro-6000-96gb (1 x NVIDIA RTX PRO 6000 Blackwell Server Edition)"},
		{"NVIDIA GeForce RTX 4090", nil,
			`warn hardware         GPU "NVIDIA GeForce RTX 4090" has no row in hardware/hardware.yaml; ` +
				"add a row to a hardware.yaml next to the profile"},
		{"NVIDIA GeForce RTX 4090", []string{"--profile-dir", dir},
			"ok   hardware         rtx-4090-24gb (1 x NVIDIA GeForce RTX 4090)"},
	} {
		useGPUs(t, c.gpu)
		var stdout, stderr bytes.Buffer
		args := append([]string{"doctor", "--docker", filepath.Join(t.TempDir(), "none.sock")}, c.args...)
		Run(context.Background(), args, &stdout, &stderr, "test")
		if !strings.Contains(stdout.String(), c.want) {
			t.Errorf("%s %v: want %q in\n%s", c.gpu, c.args, c.want, stdout.String())
		}
		if strings.Contains(stdout.String(), "FAIL hardware") {
			t.Errorf("%s: hardware failed doctor:\n%s", c.gpu, stdout.String())
		}
	}
}
