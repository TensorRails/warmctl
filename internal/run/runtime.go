package run

import (
	"bufio"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"strings"
)

// RuntimeFile marks the warmctl runtime image (deploy/kubernetes/Dockerfile):
// the warmctl it carries and the image it was built FROM.
const RuntimeFile = "/opt/warmctl/RUNTIME"

// Runtime is RuntimeFile: `key=value` lines.
type Runtime struct {
	// Warmctl is the version built into the image.
	Warmctl string
	// Base is the image it was built FROM, the one a snapshot's engine
	// must have run in.
	Base string
}

// ParseRuntime reads RuntimeFile's text.
func ParseRuntime(text string) (Runtime, error) {
	var r Runtime
	s := bufio.NewScanner(strings.NewReader(text))
	for s.Scan() {
		key, value, ok := strings.Cut(strings.TrimSpace(s.Text()), "=")
		switch {
		case !ok:
			continue
		case key == "warmctl":
			r.Warmctl = value
		case key == "base":
			r.Base = value
		}
	}
	if r.Warmctl == "" || r.Base == "" {
		return r, fmt.Errorf("%s: want warmctl= and base= lines", RuntimeFile)
	}
	return r, nil
}

// ReadRuntime reads the marker; false when this is not the runtime image.
func ReadRuntime(name string) (Runtime, bool, error) {
	raw, err := os.ReadFile(name)
	if errors.Is(err, fs.ErrNotExist) {
		return Runtime{}, false, nil
	}
	if err != nil {
		return Runtime{}, false, err
	}
	r, err := ParseRuntime(string(raw))
	return r, err == nil, err
}
