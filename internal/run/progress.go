// Package run holds what snapshot and restore share: the progress log,
// the run's container with its agent and engine, and the staging of files.
package run

import (
	"encoding/json"
	"fmt"
	"io"
	"sync"
	"time"
)

// Progress writes one line per event, with the seconds since the start.
// Format is "plain" or "jsonl".
type Progress struct {
	mu     sync.Mutex
	w      io.Writer
	start  time.Time
	format string
}

// NewProgress starts the clock.
func NewProgress(w io.Writer, format string) *Progress {
	return &Progress{w: w, start: time.Now(), format: format}
}

// Elapsed is the seconds since the start.
func (p *Progress) Elapsed() float64 { return time.Since(p.start).Seconds() }

// Printf logs an event.
func (p *Progress) Printf(format string, args ...any) {
	p.mu.Lock()
	defer p.mu.Unlock()
	msg := fmt.Sprintf(format, args...)
	if p.format == "jsonl" {
		raw, _ := json.Marshal(map[string]any{"seconds": p.Elapsed(), "event": msg})
		fmt.Fprintln(p.w, string(raw))
		return
	}
	fmt.Fprintf(p.w, "[%7.1fs] %s\n", p.Elapsed(), msg)
}

// Marker writes a line as is, at the start of the line, for scripts that
// grep for it ("serving started 41.2s").
func (p *Progress) Marker(format string, args ...any) {
	p.mu.Lock()
	defer p.mu.Unlock()
	fmt.Fprintf(p.w, format+"\n", args...)
}
