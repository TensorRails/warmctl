package oci

import (
	"bytes"
	"context"
	"encoding/json"
	"path/filepath"
	"testing"

	ocispec "github.com/opencontainers/image-spec/specs-go/v1"
	"oras.land/oras-go/v2/content"
	"oras.land/oras-go/v2/content/memory"

	"github.com/TensorRails/warmctl/v2/internal/manifest"
)

// retagWithSchema replaces the snapshot under tag by the same snapshot whose
// manifest.json carries schema: what a newer warmctl publishes for another
// GPU vendor (warmctl.snapshot/v2-rocm).
func retagWithSchema(t *testing.T, reg *memory.Store, tag, schema string) {
	t.Helper()
	ctx := context.Background()
	desc, err := reg.Resolve(ctx, tag)
	if err != nil {
		t.Fatal(err)
	}
	raw, err := content.FetchAll(ctx, reg, desc)
	if err != nil {
		t.Fatal(err)
	}
	var am ocispec.Manifest
	if err := json.Unmarshal(raw, &am); err != nil {
		t.Fatal(err)
	}
	config, err := content.FetchAll(ctx, reg, am.Config)
	if err != nil {
		t.Fatal(err)
	}
	var doc map[string]any
	if err := json.Unmarshal(config, &doc); err != nil {
		t.Fatal(err)
	}
	doc["schema"] = schema
	config, err = json.Marshal(doc)
	if err != nil {
		t.Fatal(err)
	}
	am.Config = content.NewDescriptorFromBytes(ConfigType, config)
	if err := reg.Push(ctx, am.Config, bytes.NewReader(config)); err != nil {
		t.Fatal(err)
	}
	raw, err = json.Marshal(am)
	if err != nil {
		t.Fatal(err)
	}
	d := content.NewDescriptorFromBytes(desc.MediaType, raw)
	if err := reg.Push(ctx, d, bytes.NewReader(raw)); err != nil {
		t.Fatal(err)
	}
	if err := reg.Tag(ctx, d, tag); err != nil {
		t.Fatal(err)
	}
}

// A repository may come to hold snapshots of another GPU vendor next to the
// NVIDIA ones. Such a snapshot carries its own manifest schema, so that this
// warmctl, and every warmctl released before, skips its index entry whatever
// the order of the tags, and keeps choosing the NVIDIA snapshot it can
// restore — even with a check that would accept anything of the right
// backend, as restore.CanRestore does for DMTCP on an unprivileged pod.
func TestChooseSkipsForeignSchema(t *testing.T) {
	ctx := context.Background()
	reg := memory.New()
	// "1xa…" sorts before "1xh…": the foreign entry comes first in the index.
	foreign, mf := platformSnapshot(t, "amd-instinct-288gb", "7.1.3")
	nvidia, mn := platformSnapshot(t, "h100-80gb-sxm", "580.95.05")
	for _, s := range []struct {
		dir string
		m   *manifest.Manifest
	}{{foreign, &mf}, {nvidia, &mn}} {
		s.m.Process.Backend = manifest.BackendDMTCP
		if err := s.m.Write(filepath.Join(s.dir, "manifest.json")); err != nil {
			t.Fatal(err)
		}
		pushIndexed(t, reg, s.dir, *s.m)
	}
	foreignTag, nvidiaTag := PlatformTag(mf), PlatformTag(mn)
	anything := func(manifest.Manifest) error { return nil }
	// Why the schema matters: under the NVIDIA schema the first entry would
	// win, and up would then fail its host check with a fitting snapshot in
	// the index.
	if tag, err := Choose(ctx, reg, IndexTag, anything); err != nil || tag != foreignTag {
		t.Fatalf("control: chose %q (%v), want the first entry %s", tag, err, foreignTag)
	}
	retagWithSchema(t, reg, foreignTag, "warmctl.snapshot/v2-rocm")

	idx, err := readIndex(ctx, reg)
	if err != nil {
		t.Fatal(err)
	}
	if got := idx.Manifests[0].Annotations[ocispec.AnnotationRefName]; got != foreignTag {
		t.Fatalf("the foreign entry should come first, index starts with %s", got)
	}
	tag, err := Choose(ctx, reg, IndexTag, anything)
	if err != nil || tag != nvidiaTag {
		t.Fatalf("chose %q (%v), want the NVIDIA snapshot %s", tag, err, nvidiaTag)
	}
	// Asked for the foreign snapshot by its tag, the refusal names the schema.
	if _, _, _, _, err := resolve(ctx, reg, foreignTag); err == nil {
		t.Fatal("a foreign schema must not resolve")
	}
}
