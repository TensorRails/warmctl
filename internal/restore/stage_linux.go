package restore

import "golang.org/x/sys/unix"

// fuseSuperMagic is a FUSE file system's statfs type.
const fuseSuperMagic = 0x65735546

// onFUSE is whether path is on a FUSE file system.
func onFUSE(path string) bool {
	var st unix.Statfs_t
	return unix.Statfs(path, &st) == nil && st.Type == fuseSuperMagic
}
