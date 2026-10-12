package cli

import (
	"context"
	"flag"
	"fmt"
	"io"
	"os/exec"
	"strings"
	"time"

	"github.com/TensorRails/warmctl/v2/internal/docker"
	"github.com/TensorRails/warmctl/v2/internal/host"
)

// runDoctor checks what snapshots need from this host and prints one line
// per check.
func runDoctor(ctx context.Context, args []string, stdout, stderr io.Writer) error {
	fs := flag.NewFlagSet("doctor", flag.ContinueOnError)
	fs.SetOutput(stderr)
	sock := fs.String("docker", docker.DefaultSocket, "Docker socket")
	if _, err := positionals(fs, args); err != nil {
		return err
	}
	ctx, cancel := context.WithTimeout(ctx, time.Minute)
	defer cancel()
	failed := 0
	check := func(name string, ok bool, detail string) {
		mark := "ok  "
		if !ok {
			mark, failed = "FAIL", failed+1
		}
		fmt.Fprintf(stdout, "%s %-16s %s\n", mark, name, detail)
	}
	dk, err := docker.New(ctx, *sock)
	if err == nil {
		check("docker", true, fmt.Sprintf("%s (API %s)", dk.Version.Version, dk.Version.APIVersion))
	} else {
		check("docker", false, err.Error())
	}
	gpus, err := host.Inventory(ctx)
	if err == nil {
		check("gpus", true, fmt.Sprintf("%d x %s, driver %s", len(gpus), gpus[0].Name, gpus[0].Driver))
	} else {
		check("gpus", false, err.Error())
	}
	disabled, err := host.IOUringDisabled()
	check("io_uring", err == nil && disabled == 2, fmt.Sprintf("kernel.io_uring_disabled=%d (snapshots need 2)", disabled))
	_, err = exec.LookPath("nvidia-ctk")
	check("nvidia-runtime", err == nil, "nvidia-ctk on PATH: "+fmt.Sprint(err == nil))
	check("kernel", true, host.Kernel())
	if failed > 0 {
		return fmt.Errorf("%d checks failed: %s", failed, strings.TrimSpace("see above"))
	}
	return nil
}
