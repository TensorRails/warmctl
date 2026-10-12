package run

import (
	"os"
	"path/filepath"
	"strconv"
	"testing"
)

func TestClaim(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "restores", "a")
	if err := Claim(dir); err != nil {
		t.Fatal(err)
	}
	if err := Claim(dir); err == nil {
		t.Fatal("claimed a directory this live process owns")
	}
	os.WriteFile(filepath.Join(dir, "owner.pid"), []byte(strconv.Itoa(1<<22-3)), 0o644)
	os.WriteFile(filepath.Join(dir, "junk"), []byte("x"), 0o644)
	if err := Claim(dir); err != nil {
		t.Fatalf("stale directory not reclaimed: %v", err)
	}
	if _, err := os.Stat(filepath.Join(dir, "junk")); err == nil {
		t.Fatal("stale contents kept")
	}
	plain := filepath.Join(t.TempDir(), "x")
	os.Mkdir(plain, 0o755)
	if err := Claim(plain); err == nil {
		t.Fatal("claimed a directory without an owner file")
	}
}
