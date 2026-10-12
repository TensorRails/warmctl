//go:build !linux

package run

import (
	"errors"
	"syscall"
)

func procAttr(newUTS bool) (*syscall.SysProcAttr, error) {
	if newUTS {
		return nil, errors.New("run: UTS namespaces are Linux's")
	}
	return &syscall.SysProcAttr{Setpgid: true}, nil
}
