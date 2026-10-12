//go:build !linux

package agent

import (
	"context"
	"errors"
	"io"
)

// Run is Linux-only: the agent lives in a container.
func Run(context.Context, string, io.Writer) error {
	return errors.New("agent: runs only inside a Linux container")
}
