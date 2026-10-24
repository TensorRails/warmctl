package restore

import (
	"context"
	"net"
	"path/filepath"
	"time"

	"github.com/TensorRails/warmctl/v2/internal/gateway"
	"github.com/TensorRails/warmctl/v2/internal/run"
)

// Gate is a pod restore's gateway, open before the snapshot is pulled so
// that the pod's probes get an answer from the start: /health is 503 and
// every inference route refused until the restore admits.
type Gate struct {
	addr   net.Addr
	gw     *gateway.Gateway
	cancel context.CancelFunc
	done   chan struct{}
	err    error
}

// OpenGate listens on listen, with TLS from the first probe when t is set,
// and will forward to the agent's inference socket under runDir.
func OpenGate(listen string, t *gateway.TLS, runDir string, p *run.Progress) (*Gate, error) {
	if err := checkListen(listen); err != nil {
		return nil, err
	}
	l, err := net.Listen("tcp", listen)
	if err != nil {
		return nil, err
	}
	l = t.Listen(l)
	gw := gateway.New(filepath.Join(runDir, "io", "inference.sock"), func() {
		p.Marker("first host-gateway token %.1fs", p.Elapsed())
	})
	ctx, cancel := context.WithCancel(context.Background())
	g := &Gate{addr: l.Addr(), gw: gw, cancel: cancel, done: make(chan struct{})}
	go func() {
		g.err = gateway.Serve(ctx, l, gw, 30*time.Second)
		close(g.done)
	}()
	p.Printf("gateway on %s: not ready", t.URL(g.addr))
	return g, nil
}

// Addr is where the gate listens.
func (g *Gate) Addr() net.Addr { return g.addr }

// Close drains the admitted requests and stops listening.
func (g *Gate) Close() error {
	g.cancel()
	<-g.done
	return g.err
}
