package agent

import (
	"context"
	"errors"
	"io"
	"net"
	"os"
	"time"
)

// listenUnix replaces a stale socket file and opens the socket to the
// host; the private directory around it is what keeps others out.
func listenUnix(name string) (net.Listener, error) {
	if err := os.Remove(name); err != nil && !errors.Is(err, os.ErrNotExist) {
		return nil, err
	}
	l, err := net.Listen("unix", name)
	if err != nil {
		return nil, err
	}
	if err := os.Chmod(name, 0o666); err != nil {
		l.Close()
		return nil, err
	}
	return l, nil
}

// Proxy bridges connections on a unix socket to a TCP address until ctx ends.
func Proxy(ctx context.Context, socket, addr string) error {
	l, err := listenUnix(socket)
	if err != nil {
		return err
	}
	stop := context.AfterFunc(ctx, func() { l.Close() })
	defer stop()
	for {
		c, err := l.Accept()
		if err != nil {
			if ctx.Err() != nil {
				return nil
			}
			return err
		}
		go bridge(c, addr)
	}
}

func bridge(c net.Conn, addr string) {
	defer c.Close()
	u, err := net.DialTimeout("tcp", addr, 5*time.Second)
	if err != nil {
		return
	}
	defer u.Close()
	done := make(chan struct{}, 2)
	half := func(dst, src net.Conn) {
		_, _ = io.Copy(dst, src)
		if cw, ok := dst.(interface{ CloseWrite() error }); ok {
			_ = cw.CloseWrite()
		}
		done <- struct{}{}
	}
	go half(u, c)
	go half(c, u)
	<-done
	<-done
}
