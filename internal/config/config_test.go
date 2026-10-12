package config

import (
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

func TestLoad(t *testing.T) {
	dir := t.TempDir()
	system, user := filepath.Join(dir, "etc.yaml"), filepath.Join(dir, "user.yaml")
	defer func(f func() []string) { Files = f }(Files)
	Files = func() []string { return []string{system, user, filepath.Join(dir, "missing.yaml")} }
	for _, k := range []string{"WARMCTL_REGISTRY", "WARMCTL_MIRRORS", "WARMCTL_CACHE"} {
		t.Setenv(k, "")
	}
	if c, err := Load(); err != nil || c.Registry != DefaultRegistry || c.Cache != "" {
		t.Fatalf("defaults: %+v %v", c, err)
	}
	os.WriteFile(system, []byte("registry: oci://harbor.acme/warmctl/\nmirrors: [m1.acme/warmctl]\ncache: /mnt/a\n"), 0o644)
	os.WriteFile(user, []byte("cache: /mnt/b\n"), 0o644)
	c, err := Load()
	if err != nil || c.Registry != "harbor.acme/warmctl" || c.Cache != "/mnt/b" || !reflect.DeepEqual(c.Mirrors, []string{"m1.acme/warmctl"}) {
		t.Fatalf("files: %+v %v", c, err)
	}
	t.Setenv("WARMCTL_REGISTRY", "localhost:5000/e2e")
	t.Setenv("WARMCTL_MIRRORS", "a.io/x, b.io/y")
	if c, _ = Load(); c.Registry != "localhost:5000/e2e" || !reflect.DeepEqual(c.Mirrors, []string{"a.io/x", "b.io/y"}) {
		t.Fatalf("environment: %+v", c)
	}
	os.WriteFile(user, []byte("unknown_key: x\n"), 0o644)
	if _, err := Load(); err == nil {
		t.Fatal("an unknown key was accepted")
	}
}
