//go:build linux && e2e

package e2e_test

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
)

var podImage = flag.String("pod-image", "", "deploy/kubernetes/Dockerfile built on -image")

const podGateway = "127.0.0.1:18080"

// pod is one restore in the runtime image, run the way
// deploy/kubernetes/warmctl.yaml runs it: privileged and nothing else. A
// container runtime's defaults (a 64 MiB /dev/shm, a host name of its own)
// are warmctl's to fix; a network of its own with the gateway published, no
// Docker socket, tini as PID 1 from the image's entrypoint.
type pod struct {
	e    *env
	name string
	args []string
}

func (e *env) newPod(name string, extra ...string) *pod {
	p := &pod{e: e, name: "warmctl-e2e-pod-" + name}
	p.args = append([]string{"run", "-d", "--name", p.name, "--privileged", "-p", podGateway + ":8080",
		"-v", filepath.Join(e.base, "model") + ":/model:ro",
		// The NVIDIA runtime puts nvidia-smi in a GPU pod; here the fake one.
		"-v", *testdata + ":/opt/fake-nvidia:ro",
		"-e", "PATH=/opt/fake-nvidia:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"}, extra...)
	e.t.Cleanup(p.remove)
	return p
}

// hardened adds what a locked-down pod has: a read-only root, so the
// state, the run directory and /workspace are emptyDir-like volumes, and
// memory and PID limits.
func (p *pod) hardened() *pod {
	for _, v := range []string{"workspace", "run", "state"} {
		dir := filepath.Join(p.e.base, "pods", p.name, v)
		if err := os.MkdirAll(dir, 0o755); err != nil {
			p.e.t.Fatal(err)
		}
		target := map[string]string{"workspace": "/workspace", "run": "/run/warmctl", "state": "/var/lib/warmctl"}[v]
		p.args = append(p.args, "-v", dir+":"+target)
	}
	p.args = append(p.args, "--read-only", "--memory", "2g", "--pids-limit", "4096")
	return p
}

func (p *pod) remove() { _, _ = dockerCLI("rm", "-f", p.name) }

func dockerCLI(args ...string) (string, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
	defer cancel()
	out, err := exec.CommandContext(ctx, "docker", args...).CombinedOutput()
	return string(out), err
}

func (p *pod) logs() string {
	out, _ := dockerCLI("logs", p.name)
	return out
}

// up starts the pod with the host's command, `up` and these arguments, and
// serves: the runtime image's marker makes it restore in place.
func (p *pod) up(args ...string) {
	t := p.e.t
	all := append(append(p.args, *podImage, "up", "--test-cpu"), args...)
	if out, err := dockerCLI(all...); err != nil {
		t.Fatalf("docker run: %v\n%s", err, out)
	}
	p.serve(1)
}

// serve waits for the n-th restore of this container's life to admit,
// checking that the gate answered 503 before, then sends a completion.
func (p *pod) serve(n int) {
	t := p.e.t
	statuses := p.awaitReady()
	if !strings.Contains(statuses, "503") {
		t.Errorf("the gate did not answer 503 before admitting: %s", statuses)
	}
	if got := strings.Count(p.logs(), "serving started"); got != n {
		t.Fatalf("%d restores served, want %d:\n%s", got, n, p.logs())
	}
	if got := strings.Count(p.logs(), "warmctl runtime image dev on sha256:"); got != n {
		t.Fatalf("%d restores chose in place from the image's marker, want %d", got, n)
	}
	if got := strings.Count(p.logs(), "host name warmctl in the agent's own UTS namespace"); got != n {
		t.Fatalf("%d restores named the agent's UTS namespace, want %d", got, n)
	}
	// The restored engine runs under the receiver's host name; the pod
	// keeps its own (with hostNetwork it would be the node's).
	if own, _ := dockerCLI("exec", p.name, "cat", "/proc/sys/kernel/hostname"); strings.TrimSpace(own) == "warmctl" {
		t.Fatal("the pod's own host name was changed")
	}
	engine, err := dockerCLI("exec", p.name, "sh", "-c", `for d in /proc/[0-9]*; do p=${d#/proc/}; `+
		`if [ "$p" -ge 32768 ]; then exec nsenter -t "$p" -u cat /proc/sys/kernel/hostname; fi; done`)
	if err != nil || strings.TrimSpace(engine) != "warmctl" {
		t.Fatalf("the restored engine's host name: %q %v", engine, err)
	}
	body := `{"model":"fake","prompt":[1,2,3,4],"max_tokens":4,"temperature":0}`
	resp, err := http.Post("http://"+podGateway+"/v1/completions", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	raw, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	var reply struct {
		Choices []json.RawMessage `json:"choices"`
	}
	if resp.StatusCode != 200 || json.Unmarshal(raw, &reply) != nil || len(reply.Choices) != 1 {
		t.Fatalf("completion through the pod's gateway: %d %s", resp.StatusCode, raw)
	}
	if resp, err := http.Post("http://"+podGateway+"/sleep?level=1", "application/json", nil); err != nil ||
		resp.StatusCode != 404 {
		t.Fatalf("the pod's gateway let a management route through: %v", err)
	}
	// The engine's own log reaches the pod's output (kubectl logs).
	deadline := time.Now().Add(10 * time.Second)
	for strings.Count(p.logs(), `"POST /v1/completions`) < n {
		if time.Now().After(deadline) {
			t.Fatalf("the engine's log did not reach the pod's output:\n%s", p.logs())
		}
		time.Sleep(100 * time.Millisecond)
	}
}

// unprivileged drops the privileges, as a pod under the baseline Pod
// Security Standard runs.
func (p *pod) unprivileged() *pod {
	var args []string
	for _, a := range p.args {
		if a != "--privileged" {
			args = append(args, a)
		}
	}
	p.args = args
	return p
}

// refused runs the pod in the foreground and expects up to fail, saying
// want.
func (p *pod) refused(want string, args ...string) {
	t := p.e.t
	var fg []string
	for _, a := range p.args {
		if a != "-d" {
			fg = append(fg, a)
		}
	}
	out, err := dockerCLI(append(append(fg, *podImage, "up", "--test-cpu"), args...)...)
	if err == nil || !strings.Contains(out, want) || !strings.Contains(out, "up failed") {
		t.Fatalf("pod %s was not refused with %q: %v\n%s", p.name, want, err, out)
	}
	t.Logf("pod %s refused\n%s", p.name, out)
}

// awaitReady polls /health as a readiness probe does until it answers 200
// and returns the answers it saw.
func (p *pod) awaitReady() string {
	t := p.e.t
	var seen []string
	deadline := time.Now().Add(3 * time.Minute)
	client := &http.Client{Timeout: 2 * time.Second}
	for time.Now().Before(deadline) {
		resp, err := client.Get("http://" + podGateway + "/health")
		status := "error"
		if err == nil {
			resp.Body.Close()
			status = fmt.Sprint(resp.StatusCode)
		}
		if len(seen) == 0 || seen[len(seen)-1] != status {
			seen = append(seen, status)
		}
		if status == "200" {
			return strings.Join(seen, " ")
		}
		if state, _ := dockerCLI("inspect", "-f", "{{.State.Status}}", p.name); strings.TrimSpace(state) == "exited" {
			t.Fatalf("the pod exited before it was ready:\n%s", p.logs())
		}
		time.Sleep(50 * time.Millisecond)
	}
	t.Fatalf("the pod was not ready in time (%s):\n%s", strings.Join(seen, " "), p.logs())
	return ""
}

// restart restarts the container in place, as kubelet restarts a crashed
// one: the volumes keep what the last restore left in them.
func (p *pod) restart() {
	if out, err := dockerCLI("restart", "-t", "60", p.name); err != nil {
		p.e.t.Fatalf("docker restart: %v\n%s", err, out)
	}
	p.serve(2)
}

// stop sends SIGTERM through tini, as kubelet does, and expects a clean
// exit within the grace period.
func (p *pod) stop() {
	t := p.e.t
	if out, err := dockerCLI("stop", "-t", "60", p.name); err != nil {
		t.Fatalf("docker stop: %v\n%s", err, out)
	}
	code, _ := dockerCLI("inspect", "-f", "{{.State.ExitCode}}", p.name)
	logs := p.logs()
	t.Logf("pod %s\n%s", p.name, logs)
	if strings.TrimSpace(code) != "0" || !strings.Contains(logs, "up completed") {
		t.Fatalf("the pod did not stop cleanly: exit %s", code)
	}
}

// podLegs restores the snapshot in the runtime image three ways: from a
// read-only volume, restarted in place; by name from a shared cache; by
// name from the registry in a hardened pod, restarted with its volumes.
// A pod without privileges, and one with someone's data at /workspace, are
// refused first.
func (e *env) podLegs(snap string) {
	if *podImage == "" {
		e.t.Log("no -pod-image: pod legs skipped")
		return
	}
	p := e.newPod("unprivileged", "-v", snap+":/snapshots/e2e:ro").unprivileged()
	p.refused("need privileged: true", "/snapshots/e2e")
	// A volume of the operator's at /workspace is never emptied.
	foreign := filepath.Join(e.base, "pods", "foreign-workspace")
	if err := os.MkdirAll(foreign, 0o755); err != nil {
		e.t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(foreign, "notes.txt"), []byte("mine"), 0o644); err != nil {
		e.t.Fatal(err)
	}
	p = e.newPod("foreign", "-v", snap+":/snapshots/e2e:ro", "-v", foreign+":/workspace")
	p.refused("/workspace holds files no restore left there", "/snapshots/e2e")
	if raw, err := os.ReadFile(filepath.Join(foreign, "notes.txt")); err != nil || string(raw) != "mine" {
		e.t.Fatalf("the operator's /workspace was touched: %v", err)
	}
	// kubelet sets KUBERNETES_SERVICE_HOST in every container: only the
	// log line says so, the marker chose the mode. A restart keeps the
	// container's layer: the second restore finds the first's /workspace.
	p = e.newPod("dir", "-v", snap+":/snapshots/e2e:ro", "-e", "KUBERNETES_SERVICE_HOST=10.96.0.1")
	p.up("/snapshots/e2e")
	if !strings.Contains(p.logs(), "Kubernetes pod (KUBERNETES_SERVICE_HOST), warmctl runtime image") {
		e.t.Errorf("no Kubernetes pod in the log")
	}
	p.restart()
	p.stop()
	if *registry == "" {
		return
	}
	cache := filepath.Join(e.base, "cache")
	p = e.newPod("cache", "-v", cache+":/mnt/cache", "-e", "WARMCTL_CACHE=/mnt/cache",
		"-e", "WARMCTL_REGISTRY=localhost:1/e2e")
	p.up("fake-tp2-short", "--plain-http")
	p.stop()
	if !strings.Contains(p.logs(), "pulling from dir:/mnt/cache/fake-tp2-short") {
		t := e.t
		t.Errorf("the pod did not pull from the shared cache")
	}
	ip, err := dockerCLI("inspect", "-f", "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}",
		"warmctl-e2e-registry")
	if err != nil || strings.TrimSpace(ip) == "" {
		e.t.Fatalf("registry address: %v %s", err, ip)
	}
	p = e.newPod("registry", "-e", "WARMCTL_REGISTRY="+strings.TrimSpace(ip)+":5000/e2e").hardened()
	p.up("fake-tp2-short", "--plain-http")
	p.restart()
	p.stop()
}
