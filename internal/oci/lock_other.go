//go:build !unix

package oci

import "os"

func lockFile(*os.File) error { return nil }
