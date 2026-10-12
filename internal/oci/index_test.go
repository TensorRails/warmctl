package oci

import (
	"context"
	"errors"
	"path/filepath"
	"strings"
	"testing"

	"oras.land/oras-go/v2/content/memory"

	"github.com/TensorRails/warmctl/v2/internal/manifest"
)

// platformSnapshot writes a test snapshot taken on row with driver.
func platformSnapshot(t *testing.T, row, driver string) (string, manifest.Manifest) {
	t.Helper()
	dir := filepath.Join(t.TempDir(), row+"-"+driver)
	snapshotDir(t, dir, "weights-of-"+row)
	m, err := manifest.Read(filepath.Join(dir, "manifest.json"))
	if err != nil {
		t.Fatal(err)
	}
	m.Hardware.Row.ID, m.Hardware.Driver = row, driver
	if err := m.Write(filepath.Join(dir, "manifest.json")); err != nil {
		t.Fatal(err)
	}
	return dir, m
}

func pushIndexed(t *testing.T, reg *memory.Store, dir string, m manifest.Manifest) int {
	t.Helper()
	ctx := context.Background()
	desc, err := Push(ctx, reg, dir, PlatformTag(m), Options{})
	if err != nil {
		t.Fatal(err)
	}
	n, err := AddToIndex(ctx, reg, desc, m, PlatformTag(m))
	if err != nil {
		t.Fatal(err)
	}
	return n
}

// The index keeps one entry per platform and driver, and Choose takes the
// entry the host check accepts or says why none fits.
func TestIndexChoose(t *testing.T) {
	ctx := context.Background()
	reg := memory.New()
	h100, mh := platformSnapshot(t, "h100-80gb-sxm", "580.95.05")
	b200, mb := platformSnapshot(t, "b200-180gb", "580.95.05")
	if n := pushIndexed(t, reg, h100, mh); n != 1 {
		t.Fatalf("%d entries", n)
	}
	if n := pushIndexed(t, reg, b200, mb); n != 2 {
		t.Fatalf("%d entries", n)
	}
	if n := pushIndexed(t, reg, b200, mb); n != 2 {
		t.Fatalf("a re-push added an entry: %d", n)
	}
	onlyB200 := func(m manifest.Manifest) error {
		if m.Hardware.Row.ID != "b200-180gb" {
			return errors.New("GPU is not a B200")
		}
		return nil
	}
	tag, err := Choose(ctx, reg, IndexTag, onlyB200)
	if err != nil || tag != "1xb200-180gb-580.95.05" {
		t.Fatalf("chose %q, %v", tag, err)
	}
	none := func(manifest.Manifest) error { return errors.New("driver differs") }
	_, err = Choose(ctx, reg, IndexTag, none)
	if err == nil || !strings.Contains(err.Error(), "1xh100-80gb-sxm-580.95.05: driver differs") ||
		!strings.Contains(err.Error(), "1xb200-180gb-580.95.05: driver differs") {
		t.Fatalf("refusal does not list the platforms: %v", err)
	}
	if tag, err := Choose(ctx, reg, "1xh100-80gb-sxm-580.95.05", none); err != nil || tag != "1xh100-80gb-sxm-580.95.05" {
		t.Fatalf("a snapshot tag must pass through: %q %v", tag, err)
	}
}

func TestCatalog(t *testing.T) {
	ctx := context.Background()
	reg := memory.New()
	if _, err := FetchCatalog(ctx, reg); err == nil {
		t.Fatal("a missing catalog was read")
	}
	want := Catalog{Models: []Model{{Name: "glm53-flash", Title: "GLM-5.3-Flash", License: "MIT"},
		{Name: "llama31-8b", License: "Llama 3.1 Community License", Gated: true}}}
	if _, err := PushCatalog(ctx, reg, want); err != nil {
		t.Fatal(err)
	}
	got, err := FetchCatalog(ctx, reg)
	if err != nil || len(got.Models) != 2 || got.Models[1] != want.Models[1] {
		t.Fatalf("%+v %v", got, err)
	}
	if m, ok := got.Find("llama31-8b"); !ok || !m.Gated {
		t.Fatal("Find")
	}
}
