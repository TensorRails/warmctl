package cli

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"text/tabwriter"
	"time"

	ocispec "github.com/opencontainers/image-spec/specs-go/v1"
	"gopkg.in/yaml.v3"
	"oras.land/oras-go/v2/registry/remote"

	"github.com/TensorRails/warmctl/v2/internal/config"
	"github.com/TensorRails/warmctl/v2/internal/host"
	"github.com/TensorRails/warmctl/v2/internal/oci"
	"github.com/TensorRails/warmctl/v2/internal/restore"
	"github.com/TensorRails/warmctl/v2/internal/run"
	"github.com/TensorRails/warmctl/v2/internal/store"
)

// catalogRow is one platform of one model as `catalog` shows it.
type catalogRow struct {
	Model    string  `json:"model"`
	Platform string  `json:"platform"`
	Tag      string  `json:"tag"`
	GiB      float64 `json:"size_gib"`
	Status   string  `json:"status"`
	License  string  `json:"license"`
	fits     bool
}

// runCatalog lists the platforms of the registry's models that this host
// can restore, or publishes a catalog file.
func runCatalog(ctx context.Context, args []string, stdout, stderr io.Writer) error {
	fs := flag.NewFlagSet("catalog", flag.ContinueOnError)
	fs.SetOutput(stderr)
	var f ociFlags
	f.register(fs)
	all := fs.Bool("all", false, "also list the platforms this host cannot restore, with the reason")
	asJSON := fs.Bool("json", false, "print JSON")
	publish := fs.String("publish", "", "publish this catalog YAML file to the registry")
	state := fs.String("state-dir", DefaultState, "state directory")
	pos, err := positionals(fs, args)
	if err != nil {
		return err
	}
	cfg, err := config.Load()
	if err != nil {
		return err
	}
	ns := cfg.Registry
	if len(pos) == 1 {
		ns = strings.TrimSuffix(strings.TrimPrefix(pos[0], oci.Scheme), "/")
	} else if len(pos) > 1 {
		return errors.New("usage: warmctl catalog [REGISTRY/NAMESPACE] [--all] [--json] [--publish FILE]")
	}
	o := f.options(run.NewProgress(stderr, "plain"))
	repo, _, err := oci.Repository(oci.Scheme+ns+"/"+oci.CatalogRepository, o)
	if err != nil {
		return err
	}
	if *publish != "" {
		return publishCatalog(ctx, repo, *publish, ns, stdout)
	}
	gpus, err := host.Inventory(ctx)
	if err != nil && !*all {
		return fmt.Errorf("%w; --all lists every platform", err)
	}
	rows, err := catalogRows(ctx, ns, repo, gpus, store.Store{Root: *state}, o)
	if err != nil {
		return err
	}
	return printCatalog(stdout, rows, *all, *asJSON)
}

func publishCatalog(ctx context.Context, repo *remote.Repository, file, ns string, stdout io.Writer) error {
	raw, err := os.ReadFile(file)
	if err != nil {
		return err
	}
	var c oci.Catalog
	dec := yaml.NewDecoder(strings.NewReader(string(raw)))
	dec.KnownFields(true)
	if err := dec.Decode(&c); err != nil {
		return fmt.Errorf("%s: %w", file, err)
	}
	seen := map[string]bool{}
	for _, m := range c.Models {
		if m.Name == "" || seen[m.Name] || strings.ContainsAny(m.Name, ":@ ") {
			return fmt.Errorf("%s: model name %q is empty, repeated or not a repository name", file, m.Name)
		}
		seen[m.Name] = true
	}
	if _, err := oci.PushCatalog(ctx, repo, c); err != nil {
		return err
	}
	fmt.Fprintf(stdout, "published %d models to %s/%s\n", len(c.Models), ns, oci.CatalogRepository)
	return nil
}

// catalogRows reads every model's index, 16 models at a time; rows keep
// the catalog's order. An entry for GPUs this host lacks is ruled out by its
// annotations; the others get the host check on their manifest.json.
func catalogRows(ctx context.Context, ns string, repo *remote.Repository, gpus []host.GPU, s store.Store,
	o oci.Options) ([]catalogRow, error) {
	cat, err := oci.FetchCatalog(ctx, repo)
	if err != nil {
		return nil, err
	}
	count := map[string]int{}
	for _, g := range gpus {
		count[g.Name]++
	}
	rows := make([][]catalogRow, len(cat.Models))
	errs := make([]error, len(cat.Models))
	var wg sync.WaitGroup
	sem := make(chan struct{}, 16)
	for i, model := range cat.Models {
		wg.Add(1)
		go func() {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()
			rows[i], errs[i] = modelRows(ctx, ns, model, count, s, o)
		}()
	}
	wg.Wait()
	if err := errors.Join(errs...); err != nil {
		return nil, err
	}
	var out []catalogRow
	for _, r := range rows {
		out = append(out, r...)
	}
	return out, nil
}

func modelRows(ctx context.Context, ns string, model oci.Model, count map[string]int, s store.Store,
	o oci.Options) ([]catalogRow, error) {
	mrepo, _, err := oci.Repository(oci.Scheme+ns+"/"+model.Name, o)
	if err != nil {
		return nil, err
	}
	entries, err := oci.Platforms(ctx, mrepo)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", model.Name, err)
	}
	var rows []catalogRow
	for _, e := range entries {
		row := catalogRow{Model: model.Name, Platform: e.Annotations[oci.AnnotationPlatform],
			Tag: e.Annotations[ocispec.AnnotationRefName], License: model.License}
		size, _ := strconv.ParseInt(e.Annotations[oci.AnnotationSize], 10, 64)
		row.GiB = float64(size) / (1 << 30)
		row.Status, row.fits = entryStatus(ctx, mrepo, e, count, s)
		rows = append(rows, row)
	}
	return rows, nil
}

func entryStatus(ctx context.Context, repo *remote.Repository, e ocispec.Descriptor, count map[string]int,
	s store.Store) (string, bool) {
	gpu := e.Annotations[oci.AnnotationGPU]
	n, _ := strconv.Atoi(e.Annotations[oci.AnnotationGPUs])
	if count[gpu] < n {
		return fmt.Sprintf("needs %d× %s", n, gpu), false
	}
	m, err := oci.ReadManifest(ctx, repo, e.Annotations[ocispec.AnnotationRefName])
	if err == nil {
		err = restore.Fits(ctx, m.Hardware)
	}
	if err != nil {
		return err.Error(), false
	}
	for _, f := range m.Files {
		if !s.Has(f.SHA256) {
			return "ready", true
		}
	}
	return "pulled", true
}

func printCatalog(stdout io.Writer, rows []catalogRow, all, asJSON bool) error {
	var shown []catalogRow
	for _, r := range rows {
		if r.fits || all {
			shown = append(shown, r)
		}
	}
	if asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		return enc.Encode(shown)
	}
	tw := tabwriter.NewWriter(stdout, 0, 4, 2, ' ', 0)
	fmt.Fprintln(tw, "MODEL\tPLATFORM\tSIZE\tSTATUS\tLICENSE")
	for _, r := range shown {
		fmt.Fprintf(tw, "%s\t%s\t%.0f GiB\t%s\t%s\n", r.Model, r.Platform, r.GiB, r.Status, r.License)
	}
	if len(shown) == 0 {
		fmt.Fprintln(tw, "(no model of this registry restores on this host; --all lists them)")
	}
	return tw.Flush()
}

// acceptLicense lets up pull a model the catalog marks gated only once its
// license was accepted on this host, with --accept-license the first time.
func acceptLicense(ctx context.Context, ref string, cfg config.Config, state string, accept bool,
	o oci.Options) error {
	name, ok := strings.CutPrefix(strings.TrimPrefix(ref, oci.Scheme), cfg.Registry+"/")
	if !ok {
		return nil
	}
	name = repoName(oci.Scheme + name)
	repo, _, err := oci.Repository(oci.Scheme+cfg.Registry+"/"+oci.CatalogRepository, o)
	if err != nil {
		return nil
	}
	cat, err := oci.FetchCatalog(ctx, repo)
	model, found := cat.Find(name)
	if err != nil || !found || !model.Gated {
		return nil
	}
	marker := filepath.Join(state, "licenses", name)
	if _, err := os.Stat(marker); err == nil {
		return nil
	}
	if !accept {
		return fmt.Errorf("%s is under the %s: read it, then run up again with --accept-license", name, model.License)
	}
	if err := os.MkdirAll(filepath.Dir(marker), 0o755); err != nil {
		return err
	}
	return os.WriteFile(marker, []byte(model.License+"\naccepted "+time.Now().UTC().Format(time.RFC3339)+"\n"), 0o644)
}
