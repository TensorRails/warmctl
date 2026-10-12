//go:build linux

package restore

import (
	"os"

	"golang.org/x/sys/unix"
)

// willNeed asks the kernel to start reading a file range into the page
// cache asynchronously.
func willNeed(f *os.File, offset, size int64) {
	_ = unix.Fadvise(int(f.Fd()), offset, size, unix.FADV_WILLNEED)
}
