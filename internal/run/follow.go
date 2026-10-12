package run

import (
	"bytes"
	"context"
	"io"
	"os"
	"time"
)

// Follow copies the whole lines appended to name from now on to w until
// ctx ends: the restored engine's log, which it keeps writing in the
// workspace, to a pod's output.
func Follow(ctx context.Context, name string, w io.Writer) {
	f, err := os.Open(name)
	if err != nil {
		return
	}
	defer f.Close()
	if _, err := f.Seek(0, io.SeekEnd); err != nil {
		return
	}
	buf := make([]byte, 64<<10)
	var pending []byte
	tick := time.NewTicker(500 * time.Millisecond)
	defer tick.Stop()
	for {
		n, err := f.Read(buf)
		pending = append(pending, buf[:n]...)
		if i := bytes.LastIndexByte(pending, '\n'); i >= 0 {
			_, _ = w.Write(pending[:i+1])
			pending = append(pending[:0], pending[i+1:]...)
		}
		if n > 0 && err == nil {
			continue
		}
		select {
		case <-ctx.Done():
			return
		case <-tick.C:
		}
	}
}
