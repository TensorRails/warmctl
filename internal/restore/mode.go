package restore

import (
	"context"
	"fmt"
	"os"
	"strings"

	"github.com/TensorRails/warmctl/v2/internal/run"
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

// Probe is what the choice of mode looks at.
type Probe struct {
	// Runtime reads the runtime image's marker.
	Runtime func() (run.Runtime, bool, error)
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
		"install Docker, or run the warmctl image in Kubernetes%s", dockerErr, run.RuntimeFile, hint)
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
