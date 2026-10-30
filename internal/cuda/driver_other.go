//go:build !linux

package cuda

import (
	"context"
	"errors"
)

var errNoDriver = errors.New("cuda driver API: Linux only")

// Driver calls the CUDA checkpoint API of libcuda; Linux only.
type Driver struct{}

// NewDriver fails off Linux.
func NewDriver() (*Driver, error) { return nil, errNoDriver }

// Name is the backend as the logs show it.
func (*Driver) Name() string { return BackendDriver }

// State fails off Linux.
func (*Driver) State(context.Context, int) (string, error) { return "", errNoDriver }

// Lock fails off Linux.
func (*Driver) Lock(context.Context, int, int) error { return errNoDriver }

// Checkpoint fails off Linux.
func (*Driver) Checkpoint(context.Context, int) error { return errNoDriver }

// Restore fails off Linux.
func (*Driver) Restore(context.Context, int, []Pair) error { return errNoDriver }

// Unlock fails off Linux.
func (*Driver) Unlock(context.Context, int) error { return errNoDriver }
