//go:build linux

package runtimefs

import (
	"errors"
	"fmt"
	"strings"

	"golang.org/x/sys/unix"
)

func observe(source, name string) (observed, error) {
	p := source + "/" + name
	var st unix.Stat_t
	if err := unix.Lstat(p, &st); err != nil {
		return observed{}, err
	}
	e := Entry{Path: name, Mode: st.Mode & 0o7777, UID: st.Uid, GID: st.Gid, Mtime: st.Mtim.Nano()}
	switch st.Mode & unix.S_IFMT {
	case unix.S_IFDIR:
		e.Type = TypeDir
	case unix.S_IFREG:
		e.Type, e.Size = TypeFile, st.Size
	case unix.S_IFLNK:
		e.Type = TypeSymlink
	case unix.S_IFSOCK:
		e.Type = TypeSocket
	case unix.S_IFIFO:
		e.Type = TypeFifo
	default:
		return observed{}, fmt.Errorf("runtimefs: unsupported object at %s", name)
	}
	xattrs, err := listXattrs(p)
	if err != nil {
		return observed{}, err
	}
	e.Xattrs = xattrs
	return observed{e, st.Dev, st.Ino, uint64(st.Nlink)}, nil
}

func listXattrs(p string) (map[string][]byte, error) {
	n, err := unix.Llistxattr(p, nil)
	if err != nil || n == 0 {
		if errors.Is(err, unix.ENOTSUP) || errors.Is(err, unix.EOPNOTSUPP) {
			err = nil
		}
		return nil, err
	}
	if n > 1<<20 {
		return nil, errors.New("runtimefs: xattr list too large")
	}
	buf := make([]byte, n)
	if n, err = unix.Llistxattr(p, buf); err != nil {
		return nil, err
	}
	out := map[string][]byte{}
	for _, key := range strings.Split(string(buf[:n]), "\x00") {
		if key == "" || !supportedXattr(key) {
			continue
		}
		size, err := unix.Lgetxattr(p, key, nil)
		if err != nil || size > 65536 {
			return nil, fmt.Errorf("runtimefs: xattr %s on %s: %w", key, p, err)
		}
		value := make([]byte, size)
		if size, err = unix.Lgetxattr(p, key, value); err != nil {
			return nil, err
		}
		out[key] = value[:size]
	}
	if len(out) == 0 {
		return nil, nil
	}
	return out, nil
}

func applyMetadata(p string, e Entry) error {
	if err := unix.Lchown(p, int(e.UID), int(e.GID)); err != nil {
		return err
	}
	if e.Type != TypeSymlink {
		if err := unix.Chmod(p, e.Mode); err != nil {
			return err
		}
	}
	for key, value := range e.Xattrs {
		if err := unix.Lsetxattr(p, key, value, 0); err != nil {
			return err
		}
	}
	times := []unix.Timespec{unix.NsecToTimespec(e.Mtime), unix.NsecToTimespec(e.Mtime)}
	return unix.UtimesNanoAt(unix.AT_FDCWD, p, times, unix.AT_SYMLINK_NOFOLLOW)
}

func mkfifo(p string, mode uint32) error {
	return unix.Mkfifo(p, mode)
}
