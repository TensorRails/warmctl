//go:build !linux

package criu

import (
	"context"
	"errors"
)

// ErrUnsupported is returned on hosts without Linux process namespaces.
var ErrUnsupported = errors.New("criu: only Linux can snapshot a process tree")

// ErrNotIsolated is returned when a namespace-wide operation would touch the
// host's PID namespace.
var ErrNotIsolated = ErrUnsupported

// Frozen is a stopped tree.
type Frozen struct{ Tree Tree }

// Freeze is unsupported here.
func Freeze(context.Context, int) (*Frozen, error) { return nil, ErrUnsupported }

// RestoreStopped is unsupported here.
func RestoreStopped(context.Context, string, string, string, int, int) (*Frozen, error) {
	return nil, ErrUnsupported
}

// ReservePIDFloor is unsupported here.
func ReservePIDFloor() error { return ErrUnsupported }

// PrepareRestorePIDs is unsupported here.
func PrepareRestorePIDs(int) error { return ErrUnsupported }

// Thaw is unsupported here.
func (f *Frozen) Thaw(context.Context) error { return ErrUnsupported }

// Stopped is unsupported here.
func (f *Frozen) Stopped() error { return ErrUnsupported }

// Kill does nothing here.
func (f *Frozen) Kill() {}

// Close does nothing here.
func (f *Frozen) Close() error { return nil }
