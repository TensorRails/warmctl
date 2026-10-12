package oci

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"strings"

	ocispec "github.com/opencontainers/image-spec/specs-go/v1"
	"oras.land/oras-go/v2"
	"oras.land/oras-go/v2/content"
	layout "oras.land/oras-go/v2/content/oci"
	"oras.land/oras-go/v2/errdef"

	"github.com/TensorRails/warmctl/v2/internal/manifest"
)

// DirScheme prefixes a repository kept as an OCI image layout on a file
// system, usually a shared one: dir:/mnt/shared/warmctl/glm53-flash:latest.
const DirScheme = "dir:"

// Open returns the repository ref names, a registry (oci://) or an OCI
// image layout (dir:), and the tag or digest it names, "" for none.
func Open(ctx context.Context, ref string, o Options) (oras.Target, string, error) {
	path, ok := strings.CutPrefix(ref, DirScheme)
	if !ok {
		repo, tag, err := Repository(ref, o)
		return repo, tag, err
	}
	tag := ""
	if i := strings.LastIndex(path, ":"); i > strings.LastIndex(path, "/") {
		path, tag = path[:i], path[i+1:]
	}
	store, err := layout.NewWithContext(ctx, path)
	return store, tag, err
}

// mirrored is one snapshot to copy: its tag, manifest and blobs.
type mirrored struct {
	tag  string
	desc ocispec.Descriptor
	raw  []byte
	m    manifest.Manifest
	am   ocispec.Manifest
}

// Mirror copies the snapshot src names under tag, or, when tag is an index,
// every entry choose accepts (nil: all), into dst: blobs dst lacks stream
// from src without touching the local disk, then each snapshot's manifest
// and tag, and dst's index lists them. It returns how many snapshots it
// copied.
func Mirror(ctx context.Context, src oras.ReadOnlyTarget, tag string, dst oras.Target,
	choose func(manifest.Manifest) error, o Options) (int, error) {
	snaps, err := mirrorSet(ctx, src, tag, choose)
	if err != nil {
		return 0, err
	}
	var blobs []ocispec.Descriptor
	for _, s := range snaps {
		blobs = append(blobs, s.am.Config)
		blobs = append(blobs, s.am.Layers...)
	}
	unique := distinct(blobs)
	var total int64
	for _, d := range unique {
		total += d.Size
	}
	p, sent, checked := newPusher(dst, o), &tally{}, &tally{}
	stop := meter(o, "mirrored", total, checked)
	err = parallel(ctx, len(unique), o.workers(), func(i int) error {
		d := unique[i]
		uploaded, err := pushIfMissing(ctx, p, d, func() (io.ReadCloser, error) { return src.Fetch(ctx, d) })
		if uploaded {
			sent.add(d.Size)
		}
		if err == nil {
			checked.add(d.Size)
		}
		return err
	})
	stop()
	if err != nil {
		return 0, err
	}
	for _, s := range snaps {
		if err := dst.Push(ctx, s.desc, bytes.NewReader(s.raw)); err != nil && !errors.Is(err, errdef.ErrAlreadyExists) {
			return 0, err
		}
		if err := dst.Tag(ctx, s.desc, s.tag); err != nil {
			return 0, err
		}
		if _, err := AddToIndex(ctx, dst, s.desc, s.m, s.tag); err != nil {
			return 0, err
		}
	}
	o.logf("mirrored %d snapshots: %d blobs, %d sent, %.2f GiB sent", len(snaps), len(unique), sent.blobs, gib(sent.bytes))
	return len(snaps), nil
}

// mirrorSet resolves the snapshots to copy.
func mirrorSet(ctx context.Context, src oras.ReadOnlyTarget, tag string, choose func(manifest.Manifest) error) (
	[]mirrored, error) {
	desc, err := src.Resolve(ctx, tag)
	if err != nil {
		return nil, fmt.Errorf("oci: resolve %s: %w", tag, err)
	}
	tags := []string{tag}
	if desc.MediaType == ocispec.MediaTypeImageIndex {
		tags = nil
		entries, err := Platforms(ctx, src)
		if err != nil {
			return nil, err
		}
		for _, e := range entries {
			tags = append(tags, e.Annotations[ocispec.AnnotationRefName])
		}
	}
	var out []mirrored
	var why []string
	for _, t := range tags {
		desc, am, m, _, err := resolve(ctx, src, t)
		if err == nil && choose != nil && desc.MediaType != ocispec.MediaTypeImageIndex {
			if err = choose(m); err != nil {
				why = append(why, fmt.Sprintf("%s: %v", t, err))
				continue
			}
		}
		if err != nil {
			return nil, err
		}
		raw, err := content.FetchAll(ctx, src, desc)
		if err != nil {
			return nil, err
		}
		out = append(out, mirrored{tag: t, desc: desc, raw: raw, m: m, am: am})
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("no snapshot under %s fits:\n  %s", tag, strings.Join(why, "\n  "))
	}
	return out, nil
}
