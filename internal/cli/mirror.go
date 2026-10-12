package cli

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"path/filepath"
	"strings"

	"github.com/TensorRails/warmctl/v2/internal/config"
	"github.com/TensorRails/warmctl/v2/internal/host"
	"github.com/TensorRails/warmctl/v2/internal/manifest"
	"github.com/TensorRails/warmctl/v2/internal/oci"
	"github.com/TensorRails/warmctl/v2/internal/restore"
	"github.com/TensorRails/warmctl/v2/internal/run"
)

const mirrorUsage = "usage: warmctl mirror NAME|oci://REPO[:TAG] DESTINATION | warmctl mirror --catalog DESTINATION\n" +
	"  DESTINATION: a directory (dir:/mnt/shared/warmctl or /mnt/shared/warmctl) or a registry namespace " +
	"(oci://harbor.acme/warmctl)"

// runMirror copies a model, or every model of the catalog, into another
// registry or a directory: the platforms this host can restore, or all
// with --all-platforms.
func runMirror(ctx context.Context, args []string, stdout, stderr io.Writer) error {
	fs := flag.NewFlagSet("mirror", flag.ContinueOnError)
	fs.SetOutput(stderr)
	var f ociFlags
	f.register(fs)
	all := fs.Bool("all-platforms", false, "copy every platform, not only those this host can restore")
	catalog := fs.Bool("catalog", false, "copy every model of the registry's catalog, and the catalog")
	pos, err := positionals(fs, args)
	if err != nil {
		return err
	}
	cfg, err := config.Load()
	if err != nil {
		return err
	}
	o := f.options(run.NewProgress(stdout, "plain"))
	var choose func(manifest.Manifest) error
	if !*all {
		if _, err := host.Inventory(ctx); err != nil {
			return fmt.Errorf("%w; --all-platforms copies every platform", err)
		}
		choose = func(m manifest.Manifest) error { return restore.Fits(ctx, m.Hardware) }
	}
	switch {
	case *catalog && len(pos) == 1:
		return mirrorCatalog(ctx, oci.Scheme+cfg.Registry, namespace(pos[0]), choose, o)
	case !*catalog && len(pos) == 2 && reference(pos[0], cfg) != "":
		src := reference(pos[0], cfg)
		_, err := mirrorOne(ctx, src, namespace(pos[1])+"/"+repoName(src), choose, o)
		return err
	}
	return errors.New(mirrorUsage)
}

// namespace reads a destination: dir:PATH or a path is a directory of OCI
// layouts, oci://HOST/NS or HOST/NS a registry namespace.
func namespace(arg string) string {
	arg = strings.TrimSuffix(arg, "/")
	switch {
	case strings.HasPrefix(arg, oci.DirScheme), oci.IsRef(arg):
		return arg
	case strings.HasPrefix(arg, "/"), strings.HasPrefix(arg, "."):
		abs, _ := filepath.Abs(arg)
		return oci.DirScheme + abs
	}
	return oci.Scheme + arg
}

func mirrorOne(ctx context.Context, src, dst string, choose func(manifest.Manifest) error, o oci.Options) (int, error) {
	from, tag, err := oci.Open(ctx, src, o)
	if err != nil {
		return 0, err
	}
	if tag == "" {
		tag = oci.IndexTag
	}
	to, _, err := oci.Open(ctx, dst, o)
	if err != nil {
		return 0, err
	}
	o.Log("mirroring %s:%s to %s", src, tag, dst)
	return oci.Mirror(ctx, from, tag, to, choose, o)
}

// mirrorCatalog copies every model of the source namespace's catalog that
// has a platform to copy, then the catalog itself.
func mirrorCatalog(ctx context.Context, src, dst string, choose func(manifest.Manifest) error, o oci.Options) error {
	from, _, err := oci.Open(ctx, src+"/"+oci.CatalogRepository, o)
	if err != nil {
		return err
	}
	cat, err := oci.FetchCatalog(ctx, from)
	if err != nil {
		return err
	}
	copied := 0
	for _, model := range cat.Models {
		n, err := mirrorOne(ctx, src+"/"+model.Name, dst+"/"+model.Name, choose, o)
		if err != nil {
			o.Log("%s skipped: %v", model.Name, err)
			continue
		}
		copied += n
	}
	if copied == 0 {
		return errors.New("no snapshot of the catalog was copied")
	}
	to, _, err := oci.Open(ctx, dst+"/"+oci.CatalogRepository, o)
	if err != nil {
		return err
	}
	_, err = oci.PushCatalog(ctx, to, cat)
	return err
}
