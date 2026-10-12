//go:build !linux

// Package cuda finds the processes of a tree that hold CUDA state and moves
// that state with NVIDIA's cuda-checkpoint.
package cuda

// Owners finds nothing outside Linux.
func Owners([]int) ([]int, error) { return nil, nil }
