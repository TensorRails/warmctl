package gateway

import (
	"crypto/tls"
	"errors"
	"fmt"
	"net"
	"os"
	"sync"
)

// TLS is the gateway's certificate, read from PEM files and read again
// when they change: Kubernetes rotates a mounted secret by swapping a
// symlink, and the next handshake gets the new pair without a restart.
type TLS struct {
	certFile, keyFile string
	mu                sync.Mutex
	cert              *tls.Certificate
	seen              [2]os.FileInfo
}

// LoadTLS reads the pair once, so that a bad certificate fails the start
// rather than every handshake.
func LoadTLS(certFile, keyFile string) (*TLS, error) {
	if certFile == "" || keyFile == "" {
		return nil, errors.New("--tls-cert and --tls-key go together")
	}
	t := &TLS{certFile: certFile, keyFile: keyFile}
	seen, err := t.stat()
	if err != nil {
		return nil, err
	}
	if err := t.load(seen); err != nil {
		return nil, err
	}
	return t, nil
}

func (t *TLS) stat() ([2]os.FileInfo, error) {
	var seen [2]os.FileInfo
	for i, name := range []string{t.certFile, t.keyFile} {
		fi, err := os.Stat(name)
		if err != nil {
			return seen, err
		}
		seen[i] = fi
	}
	return seen, nil
}

func (t *TLS) load(seen [2]os.FileInfo) error {
	cert, err := tls.LoadX509KeyPair(t.certFile, t.keyFile)
	if err != nil {
		return fmt.Errorf("--tls-cert %s, --tls-key %s: %w", t.certFile, t.keyFile, err)
	}
	t.cert, t.seen = &cert, seen
	return nil
}

// certificate is the pair to present, reloaded when either file is another
// file or was rewritten. A pair that fails to load, say the certificate
// replaced before its key, keeps the last good one until the next handshake.
func (t *TLS) certificate(*tls.ClientHelloInfo) (*tls.Certificate, error) {
	t.mu.Lock()
	defer t.mu.Unlock()
	seen, err := t.stat()
	if err == nil && changed(t.seen, seen) {
		_ = t.load(seen)
	}
	return t.cert, nil
}

func changed(old, cur [2]os.FileInfo) bool {
	for i := range old {
		if !os.SameFile(old[i], cur[i]) || !old[i].ModTime().Equal(cur[i].ModTime()) || old[i].Size() != cur[i].Size() {
			return true
		}
	}
	return false
}

// Listen wraps l to serve TLS 1.2 or later over HTTP/1.1, or returns l
// as it is when t is nil.
func (t *TLS) Listen(l net.Listener) net.Listener {
	if t == nil {
		return l
	}
	return tls.NewListener(l, &tls.Config{
		MinVersion:     tls.VersionTLS12,
		GetCertificate: t.certificate,
		NextProtos:     []string{"http/1.1"},
	})
}

// URL is the gateway's address as a client reaches it.
func (t *TLS) URL(addr net.Addr) string {
	if t == nil {
		return "http://" + addr.String()
	}
	return "https://" + addr.String()
}
