package run

import (
	"os"
	"path/filepath"
	"testing"
)

func TestRuntimeMarker(t *testing.T) {
	dir := t.TempDir()
	name := filepath.Join(dir, "RUNTIME")
	if _, ok, err := ReadRuntime(name); ok || err != nil {
		t.Fatalf("no marker: %t %v", ok, err)
	}
	os.WriteFile(name, []byte("warmctl=v2.0.0-beta.4\nbase=vllm/vllm-openai@sha256:ad9d\n"), 0o644)
	rt, ok, err := ReadRuntime(name)
	if !ok || err != nil || rt.Warmctl != "v2.0.0-beta.4" || rt.Base != "vllm/vllm-openai@sha256:ad9d" {
		t.Fatalf("marker: %+v %t %v", rt, ok, err)
	}
	os.WriteFile(name, []byte("base=x\n"), 0o644)
	if _, ok, err := ReadRuntime(name); ok || err == nil {
		t.Fatalf("a marker without the version: %t %v", ok, err)
	}
}
