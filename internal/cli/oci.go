package cli

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"oras.land/oras-go/v2"

	"github.com/TensorRails/warmctl/v2/internal/config"
	"github.com/TensorRails/warmctl/v2/internal/manifest"
	"github.com/TensorRails/warmctl/v2/internal/oci"
	"github.com/TensorRails/warmctl/v2/internal/restore"
	"github.com/TensorRails/warmctl/v2/internal/run"
	"github.com/TensorRails/warmctl/v2/internal/snapshot"
	"github.com/TensorRails/warmctl/v2/internal/store"
)

type ociFlags struct {
	plainHTTP   bool
	concurrency int
}

func (f *ociFlags) register(fs *flag.FlagSet) {
	fs.BoolVar(&f.plainHTTP, "plain-http", false, "talk to the registry over plain HTTP")
	fs.IntVar(&f.concurrency, "concurrency", 0, "blobs transferred at once (0: 64)")
}

func (f *ociFlags) options(p *run.Progress) oci.Options {
	return oci.Options{PlainHTTP: f.plainHTTP, Concurrency: f.concurrency, Log: p.Printf}
}

// positionals splits leading and trailing positional arguments from flags.
func positionals(fs *flag.FlagSet, args []string) ([]string, error) {
	var out []string
	for len(args) > 0 && !strings.HasPrefix(args[0], "-") {
		out, args = append(out, args[0]), args[1:]
	}
	if err := fs.Parse(args); err != nil {
		return nil, err
	}
	return append(out, fs.Args()...), nil
}

func runPush(ctx context.Context, args []string, stdout, stderr io.Writer) error {
	fs := flag.NewFlagSet("push", flag.ContinueOnError)
	fs.SetOutput(stderr)
	var f ociFlags
	f.register(fs)
	chunk := fs.String("chunk-size", "0", "upload blobs larger than this in chunks of this size, e.g. 64MiB "+
		"(0: whole blobs, chunks only after the registry refuses one as too large)")
	piece := fs.String("piece-size", "64MiB", "split files larger than this into layers of this size")
	pos, err := positionals(fs, args)
	if err != nil {
		return err
	}
	cfg, err := config.Load()
	if err != nil {
		return err
	}
	if len(pos) != 2 || reference(pos[1], cfg) == "" {
		return errors.New("usage: warmctl push SNAPSHOT_DIR NAME|oci://REGISTRY/REPO[:TAG] [--chunk-size SIZE]")
	}
	size, err := parseSize(*chunk)
	if err != nil {
		return fmt.Errorf("--chunk-size: %w", err)
	}
	pieceSize, err := parseSize(*piece)
	if err != nil || pieceSize <= 0 {
		return fmt.Errorf("--piece-size: %q is not a positive size", *piece)
	}
	p := run.NewProgress(stdout, "plain")
	o := f.options(p)
	o.ChunkSize, o.PieceSize = size, pieceSize
	repo, tag, err := oci.Open(ctx, reference(pos[1], cfg), o)
	if err != nil {
		return err
	}
	// Without a tag the snapshot goes under its platform tag and into the
	// repository's index, where `up NAME` finds it.
	var m manifest.Manifest
	indexed := tag == ""
	if indexed {
		if m, err = manifest.Read(filepath.Join(pos[0], "manifest.json")); err != nil {
			return err
		}
		tag = oci.PlatformTag(m)
	}
	desc, err := oci.Push(ctx, repo, pos[0], tag, o)
	if err != nil || !indexed {
		return err
	}
	n, err := oci.AddToIndex(ctx, repo, desc, m, tag)
	if err == nil {
		p.Printf("%s:%s is tagged %s, which lists %d platforms", repoName(reference(pos[1], cfg)), tag, oci.IndexTag, n)
	}
	return err
}

// reference turns what was typed into a reference to pull or push, or ""
// for a snapshot directory. oci://… and dir:… stay as they are; an existing
// directory or a path is local; a first element naming a host, as in
// Docker (a dot, a colon or localhost), is a registry; anything else is a
// NAME[:TAG] in the configured registry: `glm53-flash` is
// oci://registry.warmctl.io/warmctl/glm53-flash.
func reference(arg string, cfg config.Config) string {
	if oci.IsRef(arg) || strings.HasPrefix(arg, oci.DirScheme) {
		return arg
	}
	if _, err := os.Stat(arg); err == nil || arg == "" || strings.HasPrefix(arg, "/") ||
		strings.HasPrefix(arg, ".") {
		return ""
	}
	if first, _, ok := strings.Cut(arg, "/"); ok && (strings.ContainsAny(first, ".:") || first == "localhost") {
		return oci.Scheme + arg
	}
	return oci.Scheme + cfg.Registry + "/" + arg
}

// parseSize reads a byte count with an optional KiB, MiB or GiB suffix.
func parseSize(s string) (int64, error) {
	n, mult := strings.TrimSpace(s), int64(1)
	for _, u := range []struct {
		suffix string
		mult   int64
	}{{"KiB", 1 << 10}, {"MiB", 1 << 20}, {"GiB", 1 << 30}} {
		if strings.HasSuffix(n, u.suffix) {
			n, mult = strings.TrimSpace(strings.TrimSuffix(n, u.suffix)), u.mult
			break
		}
	}
	v, err := strconv.ParseInt(n, 10, 64)
	if err != nil || v < 0 || v > math.MaxInt64/mult {
		return 0, fmt.Errorf("size %q: want bytes, or a count of KiB, MiB or GiB", s)
	}
	return v * mult, nil
}

func runPull(ctx context.Context, args []string, stdout, stderr io.Writer) error {
	fs := flag.NewFlagSet("pull", flag.ContinueOnError)
	fs.SetOutput(stderr)
	var f ociFlags
	f.register(fs)
	state := fs.String("state-dir", DefaultState, "state directory (blob store)")
	output := fs.String("output", "", "snapshot directory to create (default: <state>/pulled/<digest>)")
	pos, err := positionals(fs, args)
	if err != nil {
		return err
	}
	cfg, err := config.Load()
	if err != nil {
		return err
	}
	if len(pos) != 1 || reference(pos[0], cfg) == "" {
		return errors.New("usage: warmctl pull NAME[:TAG]|oci://REGISTRY/REPO[:TAG] [--output DIR]")
	}
	// From an index, the snapshot whose hardware this host has, busy or not.
	fits := func(m manifest.Manifest) error { return restore.Fits(ctx, m.Hardware) }
	po := f.options(run.NewProgress(stdout, "plain"))
	dir, err := pullSnapshot(ctx, reference(pos[0], cfg), *output, *state, po, fits, cfg)
	if err == nil {
		fmt.Fprintln(stdout, dir)
	}
	return err
}

// pullSnapshot fetches ref into output, or into <state>/pulled/<digest>
// where a complete earlier pull is reused. A ref without a tag means the
// repository's index, of which choose picks the entry. It tries the
// sources of ref in order, and a snapshot that came from a registry is
// copied into the cache directory.
func pullSnapshot(ctx context.Context, ref, output, state string, o oci.Options,
	choose func(manifest.Manifest) error, cfg config.Config) (string, error) {
	var tried []string
	for i, src := range sources(ref, cfg) {
		// A source without this host's platform, as a cache filled by
		// other hardware, gives way to the next one.
		target, tag, err := openSource(ctx, src, o)
		if err == nil {
			tag, err = oci.Choose(ctx, target, tag, choose)
		}
		if err != nil {
			tried = append(tried, fmt.Sprintf("%s: %v", src, err))
			continue
		}
		if src != ref {
			o.Log("pulling from %s", src)
		}
		dir, pulled, err := pullFrom(ctx, target, tag, output, state, o)
		fromCache := i == 0 && strings.HasPrefix(src, oci.DirScheme)
		if err == nil && pulled && cfg.Cache != "" && !fromCache {
			cacheSnapshot(ctx, ref, dir, cfg, o)
		}
		return dir, err
	}
	return "", fmt.Errorf("cannot pull %s:\n  %s", ref, strings.Join(tried, "\n  "))
}

// openSource opens a source that has the tag ref names, or the index.
func openSource(ctx context.Context, src string, o oci.Options) (oras.Target, string, error) {
	if path, ok := strings.CutPrefix(src, oci.DirScheme); ok {
		if _, err := os.Stat(strings.SplitN(path, ":", 2)[0]); err != nil {
			return nil, "", errors.New("not in the cache")
		}
	}
	target, tag, err := oci.Open(ctx, src, o)
	if err != nil {
		return nil, "", err
	}
	if tag == "" {
		tag = oci.IndexTag
	}
	if _, err := target.Resolve(ctx, tag); err != nil {
		return nil, "", err
	}
	return target, tag, nil
}

// pullFrom pulls the snapshot tag from target, unless an earlier pull left
// it in output.
func pullFrom(ctx context.Context, target oras.Target, tag, output, state string, o oci.Options) (
	string, bool, error) {
	if output == "" {
		desc, err := target.Resolve(ctx, tag)
		if err != nil {
			return "", false, fmt.Errorf("oci: resolve %s: %w", tag, err)
		}
		output = filepath.Join(state, "pulled", desc.Digest.Encoded()[:16])
		if _, err := os.Stat(filepath.Join(output, "manifest.json")); err == nil {
			o.Log("using %s, pulled before", output)
			return output, false, nil
		}
	}
	output, _ = filepath.Abs(output)
	if _, err := oci.Pull(ctx, target, tag, output, store.Store{Root: state}, o); err != nil {
		return "", false, err
	}
	return output, true, recordSnapshot(state, output)
}

// sources lists where to pull ref from, cheapest first. A repository of the
// configured registry has copies under the same name in the cache
// directory and on the mirrors; any other ref comes from itself.
func sources(ref string, cfg config.Config) []string {
	rest, ok := underRegistry(ref, cfg)
	if !ok {
		return []string{ref}
	}
	var out []string
	if cfg.Cache != "" {
		out = append(out, oci.DirScheme+filepath.Join(cfg.Cache, rest))
	}
	for _, m := range cfg.Mirrors {
		out = append(out, oci.Scheme+strings.TrimSuffix(strings.TrimPrefix(m, oci.Scheme), "/")+"/"+rest)
	}
	return append(out, ref)
}

// underRegistry returns NAME[:TAG] of a ref in the configured registry.
func underRegistry(ref string, cfg config.Config) (string, bool) {
	if !oci.IsRef(ref) {
		return "", false
	}
	return strings.CutPrefix(strings.TrimPrefix(ref, oci.Scheme), cfg.Registry+"/")
}

// cacheSnapshot copies a snapshot pulled from a registry into the cache
// directory, under its platform tag and in the index, so that the next host
// on the same file system pulls it from there. The pull saved its digests,
// so nothing is hashed; a failure only costs the next host the download.
func cacheSnapshot(ctx context.Context, ref, dir string, cfg config.Config, o oci.Options) {
	err := func() error {
		target, _, err := oci.Open(ctx, oci.DirScheme+filepath.Join(cfg.Cache, repoName(ref)), o)
		if err != nil {
			return err
		}
		m, err := manifest.Read(filepath.Join(dir, "manifest.json"))
		if err != nil {
			return err
		}
		tag := oci.PlatformTag(m)
		desc, err := oci.Push(ctx, target, dir, tag, o)
		if err != nil {
			return err
		}
		_, err = oci.AddToIndex(ctx, target, desc, m, tag)
		return err
	}()
	if err != nil {
		o.Log("not cached in %s: %v", cfg.Cache, err)
		return
	}
	o.Log("cached in %s", filepath.Join(cfg.Cache, repoName(ref)))
}

// runImages lists the snapshots this host recorded, or a repository's tags.
func runImages(ctx context.Context, args []string, stdout, stderr io.Writer) error {
	fs := flag.NewFlagSet("images", flag.ContinueOnError)
	fs.SetOutput(stderr)
	var f ociFlags
	f.register(fs)
	state := fs.String("state-dir", DefaultState, "state directory")
	pos, err := positionals(fs, args)
	if err != nil {
		return err
	}
	if len(pos) == 1 && oci.IsRef(pos[0]) {
		repo, _, err := oci.Repository(pos[0], f.options(run.NewProgress(stderr, "plain")))
		if err != nil {
			return err
		}
		tags, err := oci.Tags(ctx, repo)
		for _, t := range tags {
			fmt.Fprintln(stdout, t)
		}
		return err
	}
	raw, _ := os.ReadFile(filepath.Join(*state, "snapshots.txt"))
	seen := map[string]bool{}
	for _, dir := range strings.Fields(string(raw)) {
		if seen[dir] {
			continue
		}
		seen[dir] = true
		if m, err := manifest.Read(filepath.Join(dir, "manifest.json")); err == nil {
			fmt.Fprintf(stdout, "%s\t%s\n", dir, snapshot.Describe(m))
		}
	}
	return nil
}
