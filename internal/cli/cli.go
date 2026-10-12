// Package cli is the warmctl command line.
package cli

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"strings"

	"github.com/TensorRails/warmctl/v2"
	"github.com/TensorRails/warmctl/v2/internal/agent"
	"github.com/TensorRails/warmctl/v2/internal/profile"
)

const usage = `usage: warmctl <command> [flags]

  build         bake a warmed engine into a snapshot (aliases: snapshot, bake)
  up            serve a snapshot: a directory, a NAME[:TAG] of the registry or
                oci://REGISTRY/REPO[:TAG], pulled first when needed (alias: restore)
  ps            list the instances serving on this host
  down          stop an instance (alias: stop)
  images        list the snapshots on this host, or oci://REPO's tags (alias: ls)
  catalog       list the registry's models this host can restore
  push|pull     move snapshots through an OCI registry or a directory (dir:PATH)
  mirror        copy models into another registry or a directory (a shared cache)
  verify        check a snapshot directory against its manifest
  gc            remove blobs no recorded snapshot uses
  doctor        check this host
  profile       inspect profiles: list, check, show ID, argv ID
  agent         container side of build and up (internal)
  version
`

// Run executes args and returns the process exit code.
func Run(ctx context.Context, args []string, stdout, stderr io.Writer, version string) int {
	if len(args) == 0 {
		fmt.Fprint(stderr, usage)
		return 2
	}
	var err error
	switch args[0] {
	case "version":
		fmt.Fprintln(stdout, "warmctl", version)
	case "profile":
		err = runProfile(args[1:], stdout)
	case "agent":
		err = runAgent(ctx, args[1:], stderr)
	case "build", "snapshot", "bake":
		err = runBuild(ctx, args[1:], stdout, stderr, version)
	case "up", "restore":
		err = runUp(ctx, args[1:], stdout, stderr)
	case "down", "stop":
		err = runDown(ctx, args[1:], stdout, stderr)
	case "ps":
		err = runPs(ctx, args[1:], stdout, stderr)
	case "verify":
		err = runVerify(ctx, args[1:], stdout, stderr)
	case "gc":
		err = runGC(ctx, args[1:], stdout, stderr)
	case "push":
		err = runPush(ctx, args[1:], stdout, stderr)
	case "pull":
		err = runPull(ctx, args[1:], stdout, stderr)
	case "mirror":
		err = runMirror(ctx, args[1:], stdout, stderr)
	case "catalog":
		err = runCatalog(ctx, args[1:], stdout, stderr)
	case "images", "ls":
		err = runImages(ctx, args[1:], stdout, stderr)
	case "doctor":
		err = runDoctor(ctx, args[1:], stdout, stderr)
	case "help", "-h", "--help":
		fmt.Fprint(stdout, usage)
	default:
		fmt.Fprintf(stderr, "unknown command %q\n%s", args[0], usage)
		return 2
	}
	if err != nil {
		fmt.Fprintln(stderr, "error:", err)
		return 1
	}
	return 0
}

func runProfile(args []string, stdout io.Writer) error {
	fs := flag.NewFlagSet("profile", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	dir := fs.String("dir", "", "extra directory of profile YAML files")
	if err := fs.Parse(args); err != nil {
		return err
	}
	rest := fs.Args()
	if len(rest) == 0 {
		return errors.New("profile list|check|show ID|argv ID")
	}
	var dirs []string
	if *dir != "" {
		dirs = []string{*dir}
	}
	loader := profileLoader(dirs)
	defaults, err := loader.Defaults()
	if err != nil {
		return err
	}
	switch rest[0] {
	case "list":
		ids, err := loader.IDs()
		if err != nil {
			return err
		}
		for _, id := range ids {
			fmt.Fprintln(stdout, id)
		}
		return nil
	case "check":
		return checkProfiles(loader, defaults, stdout)
	case "show", "argv":
		if len(rest) != 2 {
			return fmt.Errorf("profile %s needs a profile id or YAML path", rest[0])
		}
		p, err := loader.Load(rest[1])
		if err != nil {
			return err
		}
		launch, err := profile.Resolve(defaults, p)
		if err != nil {
			return err
		}
		if rest[0] == "argv" {
			fmt.Fprintln(stdout, strings.Join(launch.Argv, " "))
			return nil
		}
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		return enc.Encode(struct {
			profile.Launch
			Warmup []profile.WarmupCase `json:"warmup_cases"`
		}{launch, launch.WarmupCases()})
	}
	return fmt.Errorf("unknown profile command %q", rest[0])
}

func checkProfiles(loader profile.Loader, defaults profile.Profile, stdout io.Writer) error {
	table, err := loader.Hardware()
	if err != nil {
		return err
	}
	ids, err := loader.IDs()
	if err != nil {
		return err
	}
	for _, id := range ids {
		p, err := loader.Load(id)
		if err != nil {
			return err
		}
		launch, err := profile.Resolve(defaults, p)
		if err != nil {
			return err
		}
		for _, hw := range launch.Profile.Hardware {
			if _, ok := table.ByID(hw); !ok {
				return fmt.Errorf("profile %s: unknown hardware %q", id, hw)
			}
		}
		fmt.Fprintf(stdout, "ok %-32s ranks=%-2d graphs=%-3d warmup=%-3d argv=%d\n", id,
			launch.Profile.Parallel.Ranks(), len(launch.CaptureSizes), len(launch.WarmupCases()), len(launch.Argv))
	}
	return nil
}

func runAgent(ctx context.Context, args []string, stderr io.Writer) error {
	fs := flag.NewFlagSet("agent", flag.ContinueOnError)
	fs.SetOutput(stderr)
	config := fs.String("config", "", "agent config written by the host")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *config == "" {
		return errors.New("agent needs --config")
	}
	return agent.Run(ctx, *config, stderr)
}

func profileLoader(dirs []string) profile.Loader {
	return profile.Loader{Assets: warmctl.Assets, Dirs: dirs}
}
