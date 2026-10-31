//go:build !linux

package host

import (
	"errors"
	"io"
)

// RunPeerProbe runs on linux only.
func RunPeerProbe([]string, io.Writer) error {
	return errors.New("the peer probe runs on linux only")
}
