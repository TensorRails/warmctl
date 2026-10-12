// Package config reads warmctl's host settings: where short image names
// resolve, which mirrors and cache to try first. /etc/warmctl/config.yaml
// is read first, ~/.config/warmctl/config.yaml overrides it, and the
// environment overrides both.
package config

import (
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"strings"

	"gopkg.in/yaml.v3"
)

// DefaultRegistry is where short names resolve when nothing says otherwise.
const DefaultRegistry = "registry.warmctl.io/warmctl"

// Config is the merged settings.
type Config struct {
	// Registry is a registry host and namespace: `up glm53-flash` pulls
	// oci://<Registry>/glm53-flash:latest.
	Registry string `yaml:"registry"`
	// Mirrors hold copies of Registry's repositories, tried before it.
	Mirrors []string `yaml:"mirrors"`
	// Cache is a directory, usually on a shared file system, holding OCI
	// image layouts of pulled snapshots, tried before any registry.
	Cache string `yaml:"cache"`
}

// Files are read in order; a later one overrides the fields it sets.
var Files = func() []string {
	files := []string{"/etc/warmctl/config.yaml"}
	if dir, err := os.UserConfigDir(); err == nil {
		files = append(files, filepath.Join(dir, "warmctl", "config.yaml"))
	}
	return files
}

// Load merges the files and then WARMCTL_REGISTRY, WARMCTL_MIRRORS (comma
// separated) and WARMCTL_CACHE.
func Load() (Config, error) {
	c := Config{Registry: DefaultRegistry}
	for _, name := range Files() {
		raw, err := os.ReadFile(name)
		if errors.Is(err, fs.ErrNotExist) {
			continue
		}
		if err != nil {
			return c, err
		}
		var f Config
		dec := yaml.NewDecoder(strings.NewReader(string(raw)))
		dec.KnownFields(true)
		if err := dec.Decode(&f); err != nil && !errors.Is(err, io.EOF) {
			return c, fmt.Errorf("%s: %w", name, err)
		}
		c.merge(f)
	}
	c.merge(Config{Registry: os.Getenv("WARMCTL_REGISTRY"), Cache: os.Getenv("WARMCTL_CACHE"),
		Mirrors: split(os.Getenv("WARMCTL_MIRRORS"))})
	c.Registry = strings.TrimSuffix(strings.TrimPrefix(c.Registry, "oci://"), "/")
	return c, nil
}

func (c *Config) merge(f Config) {
	if f.Registry != "" {
		c.Registry = f.Registry
	}
	if f.Mirrors != nil {
		c.Mirrors = f.Mirrors
	}
	if f.Cache != "" {
		c.Cache = f.Cache
	}
}

func split(s string) []string {
	var out []string
	for _, v := range strings.Split(s, ",") {
		if v = strings.TrimSpace(v); v != "" {
			out = append(out, v)
		}
	}
	return out
}
