package oci

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"

	ocispec "github.com/opencontainers/image-spec/specs-go/v1"
	"oras.land/oras-go/v2"
	"oras.land/oras-go/v2/content"
	"oras.land/oras-go/v2/errdef"

	"github.com/TensorRails/warmctl/v2/internal/manifest"
)

// A registry namespace publishes its models in a catalog: the artifact
// <namespace>/catalog:latest, whose one layer is the JSON list. Each model's
// repository, <namespace>/<name>, holds its platform index.
const (
	CatalogRepository   = "catalog"
	CatalogArtifactType = "application/vnd.warmctl.catalog.v1"
	CatalogType         = "application/vnd.warmctl.catalog.v1+json"
)

// Catalog is the list of published models.
type Catalog struct {
	Models []Model `json:"models" yaml:"models"`
}

// Model is one entry of the catalog.
type Model struct {
	Name        string `json:"name" yaml:"name"`
	Title       string `json:"title" yaml:"title"`
	License     string `json:"license" yaml:"license"`
	Gated       bool   `json:"gated,omitempty" yaml:"gated"`
	Description string `json:"description,omitempty" yaml:"description"`
}

// Find returns the model called name.
func (c Catalog) Find(name string) (Model, bool) {
	for _, m := range c.Models {
		if m.Name == name {
			return m, true
		}
	}
	return Model{}, false
}

// PushCatalog publishes c in target, the catalog repository, as IndexTag.
func PushCatalog(ctx context.Context, target oras.Target, c Catalog) (ocispec.Descriptor, error) {
	raw, err := json.Marshal(c)
	if err != nil {
		return ocispec.Descriptor{}, err
	}
	layer := content.NewDescriptorFromBytes(CatalogType, raw)
	for _, b := range []struct {
		d    ocispec.Descriptor
		data []byte
	}{{ocispec.DescriptorEmptyJSON, ocispec.DescriptorEmptyJSON.Data}, {layer, raw}} {
		if err := target.Push(ctx, b.d, bytes.NewReader(b.data)); err != nil && !errors.Is(err, errdef.ErrAlreadyExists) {
			return ocispec.Descriptor{}, err
		}
	}
	empty := ocispec.DescriptorEmptyJSON
	empty.Data = nil
	desc, err := oras.PackManifest(ctx, target, oras.PackManifestVersion1_1, CatalogArtifactType,
		oras.PackManifestOptions{Layers: []ocispec.Descriptor{layer}, ConfigDescriptor: &empty})
	if err != nil {
		return desc, err
	}
	return desc, target.Tag(ctx, desc, IndexTag)
}

// FetchCatalog reads the catalog from target, the catalog repository.
func FetchCatalog(ctx context.Context, target oras.ReadOnlyTarget) (Catalog, error) {
	var c Catalog
	desc, err := target.Resolve(ctx, IndexTag)
	if err != nil {
		return c, fmt.Errorf("oci: no catalog: %w", err)
	}
	raw, err := content.FetchAll(ctx, target, desc)
	if err != nil {
		return c, err
	}
	var am ocispec.Manifest
	if err := json.Unmarshal(raw, &am); err != nil {
		return c, err
	}
	if am.ArtifactType != CatalogArtifactType || len(am.Layers) != 1 || am.Layers[0].MediaType != CatalogType {
		return c, fmt.Errorf("oci: %s is not a warmctl catalog", IndexTag)
	}
	if raw, err = content.FetchAll(ctx, target, am.Layers[0]); err != nil {
		return c, err
	}
	return c, json.Unmarshal(raw, &c)
}

// Platforms lists the entries of a repository's index, none when it has no
// index.
func Platforms(ctx context.Context, target oras.ReadOnlyTarget) ([]ocispec.Descriptor, error) {
	idx, err := readIndex(ctx, target)
	return idx.Manifests, err
}

// ReadManifest reads the manifest.json of the snapshot ref names, without
// any of its files.
func ReadManifest(ctx context.Context, target oras.ReadOnlyTarget, ref string) (manifest.Manifest, error) {
	_, _, m, _, err := resolve(ctx, target, ref)
	return m, err
}
