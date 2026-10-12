//go:build !linux

package restore

import "os"

func willNeed(*os.File, int64, int64) {}
