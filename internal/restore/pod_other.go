//go:build !linux

package restore

import "errors"

var errPod = errors.New("restores in a pod run only on Linux")

func statTree(string) (fsInfo, error)      { return fsInfo{}, errPod }
func writable(string) error                { return errPod }
func bindMount(string, string, bool) error { return errPod }
func unmount(string) error                 { return errPod }
func mountTmpfs(string, int64, bool) error { return errPod }
