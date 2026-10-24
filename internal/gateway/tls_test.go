package gateway

import (
	"context"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/tls"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"math/big"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"testing"
	"time"
)

// selfSigned writes a certificate for 127.0.0.1 with the given serial and
// its key into dir as tls.crt and tls.key.
func selfSigned(t *testing.T, dir string, serial int64) *x509.Certificate {
	t.Helper()
	key, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	tmpl := &x509.Certificate{
		SerialNumber: big.NewInt(serial), Subject: pkix.Name{CommonName: "gateway"},
		NotBefore: time.Now().Add(-time.Hour), NotAfter: time.Now().Add(time.Hour),
		IPAddresses: []net.IP{net.IPv4(127, 0, 0, 1)}, KeyUsage: x509.KeyUsageDigitalSignature,
		ExtKeyUsage: []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth},
	}
	der, err := x509.CreateCertificate(rand.Reader, tmpl, tmpl, &key.PublicKey, key)
	if err != nil {
		t.Fatal(err)
	}
	keyDER, err := x509.MarshalECPrivateKey(key)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	write := func(name, kind string, b []byte) {
		if err := os.WriteFile(filepath.Join(dir, name), pem.EncodeToMemory(&pem.Block{Type: kind, Bytes: b}), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	write("tls.crt", "CERTIFICATE", der)
	write("tls.key", "EC PRIVATE KEY", keyDER)
	cert, err := x509.ParseCertificate(der)
	if err != nil {
		t.Fatal(err)
	}
	return cert
}

// The gateway serves HTTPS from the first probe, and a mounted secret
// rotated the way Kubernetes does it (a new directory behind the ..data
// symlink) is presented on the next handshake.
func TestTLS(t *testing.T) {
	dir := t.TempDir()
	first := selfSigned(t, filepath.Join(dir, "v1"), 1)
	if err := os.Symlink("v1", filepath.Join(dir, "..data")); err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"tls.crt", "tls.key"} {
		if err := os.Symlink(filepath.Join("..data", name), filepath.Join(dir, name)); err != nil {
			t.Fatal(err)
		}
	}
	certFile, keyFile := filepath.Join(dir, "tls.crt"), filepath.Join(dir, "tls.key")
	if _, err := LoadTLS(certFile, ""); err == nil {
		t.Fatal("a certificate without its key loaded")
	}
	if _, err := LoadTLS(filepath.Join(dir, "v1", "tls.crt"), filepath.Join(dir, "missing.key")); err == nil {
		t.Fatal("a missing key loaded")
	}
	tl, err := LoadTLS(certFile, keyFile)
	if err != nil {
		t.Fatal(err)
	}
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() { done <- Serve(ctx, tl.Listen(l), New(filepath.Join(dir, "none.sock"), nil), time.Second) }()
	defer func() { cancel(); <-done }()
	url := tl.URL(l.Addr()) + "/health"
	if url[:8] != "https://" {
		t.Fatalf("URL %s", url)
	}

	roots := x509.NewCertPool()
	roots.AddCert(first)
	// health answers over a new connection, so every call is a handshake,
	// and returns the serial of the certificate the gateway presented.
	health := func(roots *x509.CertPool) (int, int64) {
		t.Helper()
		client := &http.Client{Timeout: 5 * time.Second, Transport: &http.Transport{
			TLSClientConfig: &tls.Config{RootCAs: roots}, DisableKeepAlives: true}}
		resp, err := client.Get(url)
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()
		return resp.StatusCode, resp.TLS.PeerCertificates[0].SerialNumber.Int64()
	}
	if code, serial := health(roots); code != http.StatusServiceUnavailable || serial != 1 {
		t.Fatalf("not-ready gateway over TLS: %d, serial %d", code, serial)
	}
	if resp, err := http.Get("http://" + l.Addr().String() + "/health"); err == nil && resp.StatusCode == http.StatusOK {
		t.Fatal("plain HTTP answered 200")
	}

	second := selfSigned(t, filepath.Join(dir, "v2"), 2)
	if err := os.Symlink("v2", filepath.Join(dir, "..data_tmp")); err != nil {
		t.Fatal(err)
	}
	if err := os.Rename(filepath.Join(dir, "..data_tmp"), filepath.Join(dir, "..data")); err != nil {
		t.Fatal(err)
	}
	roots.AddCert(second)
	if _, serial := health(roots); serial != 2 {
		t.Fatalf("rotated secret: serial %d", serial)
	}

	// A certificate replaced before its key keeps the last good pair.
	selfSigned(t, filepath.Join(dir, "v3"), 3)
	if err := os.Rename(filepath.Join(dir, "v3", "tls.crt"), filepath.Join(dir, "v2", "tls.crt")); err != nil {
		t.Fatal(err)
	}
	if _, serial := health(roots); serial != 2 {
		t.Fatalf("half-rotated secret: serial %d", serial)
	}
}
