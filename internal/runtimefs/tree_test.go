package runtimefs

import (
	"bytes"
	"context"
	"net"
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

func TestSnapshotInstall(t *testing.T) {
	ctx := context.Background()
	root := t.TempDir()
	src := filepath.Join(root, "src")
	for _, dir := range []string{"a/b", "empty"} {
		if err := os.MkdirAll(filepath.Join(src, dir), 0o755); err != nil {
			t.Fatal(err)
		}
	}
	pattern := bytes.Repeat([]byte("warmctl-runtimefs-"), 4096)
	files := map[string][]byte{"a/f.txt": []byte("hello\n"), "a/b/g.bin": pattern}
	for name, body := range files {
		if err := os.WriteFile(filepath.Join(src, name), body, 0o640); err != nil {
			t.Fatal(err)
		}
	}
	if err := os.Link(filepath.Join(src, "a/f.txt"), filepath.Join(src, "a/h.txt")); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink("b/g.bin", filepath.Join(src, "a/l")); err != nil {
		t.Fatal(err)
	}
	if err := mkfifo(filepath.Join(src, "a/p"), 0o600); err != nil {
		t.Fatal(err)
	}
	if runtime.GOOS == "linux" {
		l, err := net.Listen("unix", filepath.Join(src, "a/s.sock"))
		if err != nil {
			t.Fatal(err)
		}
		defer l.Close()
	}
	archive := filepath.Join(root, "fs.tar")
	m, err := Snapshot(ctx, src, archive, "/workspace")
	if err != nil {
		t.Fatal(err)
	}
	types := map[string]string{}
	for _, e := range m.Entries {
		types[e.Path] = e.Type
	}
	want := map[string]string{".": TypeDir, "a": TypeDir, "a/b": TypeDir, "empty": TypeDir, "a/f.txt": TypeFile, "a/h.txt": TypeHardlink, "a/b/g.bin": TypeFile, "a/l": TypeSymlink, "a/p": TypeFifo}
	if runtime.GOOS == "linux" {
		want["a/s.sock"] = TypeSocket
	}
	for p, ty := range want {
		if types[p] != ty {
			t.Errorf("%s: %q, want %q", p, types[p], ty)
		}
	}
	if len(types) != len(want) {
		t.Errorf("entries %v", types)
	}
	dst := filepath.Join(root, "dst")
	if err := Install(ctx, archive, dst, m); err != nil {
		t.Fatal(err)
	}
	for name, body := range files {
		got, err := os.ReadFile(filepath.Join(dst, name))
		if err != nil || !bytes.Equal(got, body) {
			t.Errorf("%s differs: %v", name, err)
		}
	}
	a, _ := os.Stat(filepath.Join(dst, "a/f.txt"))
	b, _ := os.Stat(filepath.Join(dst, "a/h.txt"))
	if !os.SameFile(a, b) || a.Mode().Perm() != 0o640 {
		t.Errorf("hardlink or mode lost: %v %v", a.Mode(), os.SameFile(a, b))
	}
	if target, err := os.Readlink(filepath.Join(dst, "a/l")); err != nil || target != "b/g.bin" {
		t.Errorf("symlink %q %v", target, err)
	}
	if info, err := os.Lstat(filepath.Join(dst, "a/p")); err != nil || info.Mode()&os.ModeNamedPipe == 0 {
		t.Errorf("fifo lost: %v", err)
	}
	if _, err := os.Lstat(filepath.Join(dst, "a/s.sock")); err == nil {
		t.Error("socket was materialized")
	}
	srcInfo, _ := os.Stat(filepath.Join(src, "a/b/g.bin"))
	dstInfo, _ := os.Stat(filepath.Join(dst, "a/b/g.bin"))
	if runtime.GOOS == "linux" && !srcInfo.ModTime().Equal(dstInfo.ModTime()) {
		t.Errorf("mtime %v != %v", srcInfo.ModTime(), dstInfo.ModTime())
	}
	if err := Install(ctx, archive, dst, m); err == nil {
		t.Error("installed over a non-empty directory")
	}
	empty := filepath.Join(root, "empty")
	if err := os.Mkdir(empty, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := Install(ctx, archive, empty, m); err != nil {
		t.Errorf("install into an empty directory: %v", err)
	}
	marked := filepath.Join(root, "marked")
	os.Mkdir(marked, 0o755)
	os.WriteFile(filepath.Join(marked, Mark), nil, 0o644)
	if err := Install(ctx, archive, marked, m); err != nil {
		t.Errorf("install into a directory a receiver marked: %v", err)
	}
	raw, _ := os.ReadFile(archive)
	i := bytes.Index(raw, pattern[:64])
	raw[i+100] ^= 1
	tampered := filepath.Join(root, "bad.tar")
	os.WriteFile(tampered, raw, 0o644)
	if err := Install(ctx, tampered, filepath.Join(root, "dst2"), m); err == nil {
		t.Error("tampered archive installed")
	}
	if _, err := os.Stat(filepath.Join(root, "dst2")); err == nil {
		t.Error("failed install left its directory")
	}
	name := filepath.Join(root, "fs.json")
	if err := m.Write(name); err != nil {
		t.Fatal(err)
	}
	if back, err := Read(name); err != nil || len(back.Entries) != len(m.Entries) || back.Root != "/workspace" {
		t.Fatalf("manifest roundtrip: %v", err)
	}
}

func TestValidate(t *testing.T) {
	good := Manifest{Schema: Schema, Root: "/tmp", Entries: []Entry{{Path: ".", Type: TypeDir}, {Path: "f", Type: TypeFile, SHA256: string(make([]byte, 64))}, {Path: "h", Type: TypeHardlink, Link: "f"}}}
	if err := good.Validate(); err != nil {
		t.Fatal(err)
	}
	for name, bad := range map[string]Manifest{
		"root":     {Schema: Schema, Root: "tmp", Entries: good.Entries},
		"escape":   {Schema: Schema, Root: "/tmp", Entries: []Entry{{Path: ".", Type: TypeDir}, {Path: "../x", Type: TypeDir}}},
		"orphan":   {Schema: Schema, Root: "/tmp", Entries: []Entry{{Path: ".", Type: TypeDir}, {Path: "d/x", Type: TypeDir}}},
		"nodigest": {Schema: Schema, Root: "/tmp", Entries: []Entry{{Path: ".", Type: TypeDir}, {Path: "f", Type: TypeFile}}},
		"link":     {Schema: Schema, Root: "/tmp", Entries: []Entry{{Path: ".", Type: TypeDir}, {Path: "h", Type: TypeHardlink, Link: "f"}}},
	} {
		if err := bad.Validate(); err == nil {
			t.Errorf("%s accepted", name)
		}
	}
}
