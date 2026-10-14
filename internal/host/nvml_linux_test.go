package host

import (
	"testing"
	"unsafe"
)

// The structs NVML fills must match nvml.h byte for byte.
func TestNVMLLayouts(t *testing.T) {
	for _, c := range []struct {
		name      string
		got, want uintptr
	}{
		{"nvmlPciInfo_t", unsafe.Sizeof(pciInfo{}), 68},
		{"nvmlPciInfo_t.busId offset", unsafe.Offsetof(pciInfo{}.busID), 36},
		{"nvmlMemory_t", unsafe.Sizeof(memoryInfo{}), 24},
		{"nvmlProcessInfo_t", unsafe.Sizeof(processInfo{}), 24},
	} {
		if c.got != c.want {
			t.Errorf("%s: %d bytes, nvml.h has %d", c.name, c.got, c.want)
		}
	}
	if cString([]byte("580.95.05\x00junk")) != "580.95.05" || cString([]byte("full")) != "full" {
		t.Error("cString")
	}
}
