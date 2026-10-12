// Package warmctl embeds what ships inside the binary: the curated model
// profiles, the profile defaults, the hardware table and the vLLM worker
// extension. Nothing in the Go code knows a model or a GPU by name.
package warmctl

import "embed"

// Assets holds profiles/*.yaml and hardware/hardware.yaml.
//
//go:embed profiles/*.yaml hardware/hardware.yaml
var Assets embed.FS

// Adapter holds the Python package mounted into every engine container.
//
//go:embed python/warmctl_vllm/*.py
var Adapter embed.FS
