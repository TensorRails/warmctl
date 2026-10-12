//go:build linux

package run

import "syscall"

func procAttr(newUTS bool) (*syscall.SysProcAttr, error) {
	attr := &syscall.SysProcAttr{Setpgid: true}
	if newUTS {
		attr.Cloneflags = syscall.CLONE_NEWUTS
	}
	return attr, nil
}
