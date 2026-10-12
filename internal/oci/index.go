package oci

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"strconv"
	"strings"
	"time"

	specs "github.com/opencontainers/image-spec/specs-go"
	ocispec "github.com/opencontainers/image-spec/specs-go/v1"
	"oras.land/oras-go/v2"
	"oras.land/oras-go/v2/content"
	"oras.land/oras-go/v2/errdef"

	"github.com/TensorRails/warmctl/v2/internal/manifest"
)

// A model's snapshots for several platforms share one repository, each
// under its platform tag (8xh100-80gb-sxm-580.95.05). The repository's
// IndexTag is an OCI image index listing them, so that, as with a
// multi-arch image, `up NAME` takes the one this host can restore.
const (
	IndexTag          = "latest"
	IndexArtifactType = "application/vnd.warmctl.snapshot.index.v2"
	// AnnotationPlatform is the GPU count and hardware row: 8xh100-80gb-sxm.
	AnnotationPlatform = "io.warmctl.platform"
	// AnnotationGPU is the GPU model as nvidia-smi names it, AnnotationGPUs
	// how many: enough for a host to rule an entry out without its manifest.
	AnnotationGPU     = "io.warmctl.gpu"
	AnnotationGPUs    = "io.warmctl.gpus"
	AnnotationDriver  = "io.warmctl.driver"
	AnnotationProfile = "io.warmctl.profile"
	AnnotationVLLM    = "io.warmctl.vllm"
	AnnotationSize    = "io.warmctl.size"
)

func platform(m manifest.Manifest) string {
	return fmt.Sprintf("%dx%s", m.Launch.Ranks, m.Hardware.Row.ID)
}

func gpuName(m manifest.Manifest) string {
	if len(m.Hardware.GPUs) == 0 {
		return ""
	}
	return m.Hardware.GPUs[0].Name
}

// PlatformTag is the tag a snapshot is pushed under next to the index: its
// platform and driver, since a snapshot restores only on the same driver.
func PlatformTag(m manifest.Manifest) string {
	return platform(m) + "-" + m.Hardware.Driver
}

// AddToIndex records the snapshot m, pushed as desc under tag, in the
// repository's index, in place of an entry of the same platform and
// driver, and returns how many entries the index lists. Two pushes racing
// on one repository may lose an entry; the last write wins.
func AddToIndex(ctx context.Context, target oras.Target, desc ocispec.Descriptor, m manifest.Manifest,
	tag string) (int, error) {
	idx, err := readIndex(ctx, target)
	if err != nil {
		return 0, err
	}
	entry := ocispec.Descriptor{MediaType: desc.MediaType, Digest: desc.Digest, Size: desc.Size,
		ArtifactType: ArtifactType, Annotations: map[string]string{
			ocispec.AnnotationRefName: tag, AnnotationPlatform: platform(m), AnnotationDriver: m.Hardware.Driver,
			AnnotationGPU: gpuName(m), AnnotationGPUs: strconv.Itoa(m.Launch.Ranks),
			AnnotationProfile: m.Profile.ID, AnnotationVLLM: m.Hardware.VLLM,
			AnnotationSize: strconv.FormatInt(m.Bytes(), 10), ocispec.AnnotationCreated: m.CreatedAt.UTC().Format(time.RFC3339),
		}}
	kept := []ocispec.Descriptor{entry}
	for _, e := range idx.Manifests {
		if e.Annotations[AnnotationPlatform] != platform(m) || e.Annotations[AnnotationDriver] != m.Hardware.Driver {
			kept = append(kept, e)
		}
	}
	sort.Slice(kept, func(i, j int) bool {
		return kept[i].Annotations[ocispec.AnnotationRefName] < kept[j].Annotations[ocispec.AnnotationRefName]
	})
	idx.Manifests = kept
	raw, err := json.Marshal(idx)
	if err != nil {
		return 0, err
	}
	d := content.NewDescriptorFromBytes(ocispec.MediaTypeImageIndex, raw)
	if err := target.Push(ctx, d, bytes.NewReader(raw)); err != nil && !errors.Is(err, errdef.ErrAlreadyExists) {
		return 0, err
	}
	return len(kept), target.Tag(ctx, d, IndexTag)
}

// readIndex fetches the repository's index, or an empty one.
func readIndex(ctx context.Context, target oras.ReadOnlyTarget) (ocispec.Index, error) {
	idx := ocispec.Index{Versioned: specs.Versioned{SchemaVersion: 2}, MediaType: ocispec.MediaTypeImageIndex,
		ArtifactType: IndexArtifactType}
	desc, err := target.Resolve(ctx, IndexTag)
	if errors.Is(err, errdef.ErrNotFound) {
		return idx, nil
	}
	if err != nil {
		return idx, err
	}
	if desc.MediaType != ocispec.MediaTypeImageIndex {
		return idx, fmt.Errorf("oci: %s is a %s, not a warmctl index", IndexTag, desc.MediaType)
	}
	raw, err := content.FetchAll(ctx, target, desc)
	if err != nil {
		return idx, err
	}
	return idx, json.Unmarshal(raw, &idx)
}

// Choose returns the tag of the snapshot ref stands for: ref itself when it
// names a snapshot, or, when it names an index, the first entry whose
// manifest.json check accepts (nil accepts any). Otherwise the error lists
// every platform and why it does not fit.
func Choose(ctx context.Context, target oras.ReadOnlyTarget, ref string, check func(manifest.Manifest) error) (
	string, error) {
	desc, err := target.Resolve(ctx, ref)
	if err != nil {
		return "", fmt.Errorf("oci: resolve %s: %w", ref, err)
	}
	if desc.MediaType != ocispec.MediaTypeImageIndex {
		return ref, nil
	}
	raw, err := content.FetchAll(ctx, target, desc)
	if err != nil {
		return "", err
	}
	var idx ocispec.Index
	if err := json.Unmarshal(raw, &idx); err != nil {
		return "", err
	}
	var why []string
	for _, e := range idx.Manifests {
		tag := e.Annotations[ocispec.AnnotationRefName]
		_, _, m, _, err := resolve(ctx, target, tag)
		if err == nil && check != nil {
			err = check(m)
		}
		if err == nil {
			return tag, nil
		}
		why = append(why, fmt.Sprintf("%s: %v", tag, err))
	}
	if len(why) == 0 {
		return "", fmt.Errorf("oci: %s lists no snapshots", ref)
	}
	return "", fmt.Errorf("no snapshot under %s restores on this host:\n  %s", ref, strings.Join(why, "\n  "))
}
