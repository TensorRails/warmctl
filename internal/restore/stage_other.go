//go:build !linux

package restore

// onFUSE is false where restores do not run.
func onFUSE(string) bool { return false }
