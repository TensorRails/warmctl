package host

import "testing"

func TestParseDeviceMinor(t *testing.T) {
	info := "Model: \t\t NVIDIA L40S\nBus Location: \t 0000:61:00.0\nDevice Minor: \t 5\nGPU Excluded:\t No\n"
	if m, err := parseDeviceMinor(info); err != nil || m != 5 {
		t.Fatalf("minor %d: %v", m, err)
	}
	if _, err := parseDeviceMinor("Model: NVIDIA L40S\n"); err == nil {
		t.Fatal("a file without the minor parsed")
	}
}
