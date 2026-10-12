//go:build !linux

package runtimefs

import (
	"fmt"
	"os"
	"syscall"
	"time"
)

func observe(source, name string) (observed, error) {
	p := source + "/" + name
	info, err := os.Lstat(p)
	if err != nil {
		return observed{}, err
	}
	st, ok := info.Sys().(*syscall.Stat_t)
	if !ok {
		return observed{}, fmt.Errorf("runtimefs: no stat for %s", name)
	}
	e := Entry{Path: name, Mode: uint32(info.Mode().Perm()), UID: st.Uid, GID: st.Gid, Mtime: info.ModTime().UnixNano()}
	switch {
	case info.IsDir():
		e.Type = TypeDir
	case info.Mode().IsRegular():
		e.Type, e.Size = TypeFile, info.Size()
	case info.Mode()&os.ModeSymlink != 0:
		e.Type = TypeSymlink
	case info.Mode()&os.ModeSocket != 0:
		e.Type = TypeSocket
	case info.Mode()&os.ModeNamedPipe != 0:
		e.Type = TypeFifo
	default:
		return observed{}, fmt.Errorf("runtimefs: unsupported object at %s", name)
	}
	return observed{e, uint64(st.Dev), uint64(st.Ino), uint64(st.Nlink)}, nil
}

func applyMetadata(p string, e Entry) error {
	if e.Type == TypeSymlink {
		return nil
	}
	if err := os.Chmod(p, os.FileMode(e.Mode)); err != nil {
		return err
	}
	t := time.Unix(0, e.Mtime)
	return os.Chtimes(p, t, t)
}

func mkfifo(p string, mode uint32) error {
	return syscall.Mkfifo(p, mode)
}
