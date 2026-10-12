package restore

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"strings"
)

// Where `up` restores.
const (
	// ModeAuto decides: in place in the warmctl runtime image, else
	// through Docker.
	ModeAuto = "auto"
	// ModeHost creates a receiver container through the Docker API.
	ModeHost = "host"
	// ModeInPlace restores in this container, which a runtime (Kubernetes)
	// made as a receiver container (docs/kubernetes.md).
	ModeInPlace = "in-place"
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

// Probe is what the choice of mode looks at.
type Probe struct {
	// Runtime reads the runtime image's marker.
	Runtime func() (Runtime, bool, error)
	// Docker is nil when the Docker API answers.
	Docker func(context.Context) error
	// Kubernetes lists the signs of a Kubernetes pod, for the log line
	// only: a Kubernetes node may run Docker too.
	Kubernetes func() []string
}

// Choose decides where to restore: an explicit mode wins; the runtime
// image's marker means in place; else the Docker API, as on any host.
// Missing Docker alone proves nothing, so without either it fails. The
// reason is a log line for an in-place restore, empty for a host one.
func Choose(ctx context.Context, mode string, p Probe) (bool, string, error) {
	switch mode {
	case ModeHost:
		return false, "", nil
	case ModeInPlace:
		return true, "restoring in place (--mode in-place)", nil
	case "", ModeAuto:
	default:
		return false, "", fmt.Errorf("--mode %q: want %s, %s or %s", mode, ModeAuto, ModeHost, ModeInPlace)
	}
	rt, ok, err := p.Runtime()
	if err != nil {
		return false, "", err
	}
	if ok {
		where := "warmctl runtime image"
		if signs := p.Kubernetes(); len(signs) > 0 {
			where = "Kubernetes pod (" + strings.Join(signs, ", ") + "), warmctl runtime image"
		}
		return true, fmt.Sprintf("%s %s on %s: restoring in place", where, rt.Warmctl, rt.Base), nil
	}
	dockerErr := p.Docker(ctx)
	if dockerErr == nil {
		return false, "", nil
	}
	hint := ""
	if signs := p.Kubernetes(); len(signs) > 0 {
		hint = "; this is a Kubernetes pod (" + strings.Join(signs, ", ") + ") running another image"
	}
	return false, "", fmt.Errorf("docker not found (%v) and this is not the warmctl runtime image (no %s): "+
		"install Docker, or run the warmctl image in Kubernetes%s", dockerErr, RuntimeFile, hint)
}

// KubernetesSigns are what kubelet leaves in a pod's container.
func KubernetesSigns() []string {
	var signs []string
	if os.Getenv("KUBERNETES_SERVICE_HOST") != "" {
		signs = append(signs, "KUBERNETES_SERVICE_HOST")
	}
	if _, err := os.Stat("/var/run/secrets/kubernetes.io/serviceaccount"); err == nil {
		signs = append(signs, "service account")
	}
	if raw, err := os.ReadFile("/proc/self/cgroup"); err == nil && strings.Contains(string(raw), "kubepods") {
		signs = append(signs, "kubepods cgroup")
	}
	return signs
}
