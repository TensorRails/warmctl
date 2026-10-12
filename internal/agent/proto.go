// Package agent is PID 1 of the donor and receiver containers. The host
// drives it through a fixed sequence of steps over a unix socket; this file
// is the wire protocol: one JSON object per line, requests carrying an id and
// a step name, responses echoing the id. There is no journal and no replay:
// after a failed step the agent refuses every further one and the host
// removes the container.
package agent

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"sync"
	"time"
)

// MaxLine bounds one protocol line.
const MaxLine = 16 << 20

// Request asks the agent to run one step.
type Request struct {
	ID   int64           `json:"id"`
	Step string          `json:"step"`
	Args json.RawMessage `json:"args,omitempty"`
}

// Response answers the request with the same id.
type Response struct {
	ID     int64           `json:"id"`
	OK     bool            `json:"ok"`
	Result json.RawMessage `json:"result,omitempty"`
	Error  string          `json:"error,omitempty"`
}

// ErrPoisoned is reported for every step after a failed one.
var ErrPoisoned = errors.New("an earlier step failed")

// StepError is a step's failure as the agent reported it.
type StepError struct {
	Step    string
	Message string
}

func (e *StepError) Error() string { return "agent step " + e.Step + ": " + e.Message }

// Client sends steps over one connection, one at a time.
type Client struct {
	mu   sync.Mutex
	conn net.Conn
	r    *bufio.Reader
	next int64
}

// NewClient wraps a connected socket.
func NewClient(conn net.Conn) *Client {
	return &Client{conn: conn, r: bufio.NewReaderSize(conn, 64<<10)}
}

// Call runs step with args and decodes the agent's result into result,
// which may be nil. Cancelling ctx aborts the wait.
func (c *Client) Call(ctx context.Context, step string, args, result any) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.next++
	req := Request{ID: c.next, Step: step}
	if args != nil {
		raw, err := json.Marshal(args)
		if err != nil {
			return err
		}
		req.Args = raw
	}
	line, err := json.Marshal(req)
	if err != nil {
		return err
	}
	_ = c.conn.SetDeadline(time.Time{})
	stop := context.AfterFunc(ctx, func() { _ = c.conn.SetDeadline(time.Now()) })
	defer stop()
	if _, err := c.conn.Write(append(line, '\n')); err != nil {
		return wrap(ctx, err)
	}
	raw, err := readLine(c.r, MaxLine)
	if err != nil {
		return wrap(ctx, err)
	}
	var resp Response
	if err := json.Unmarshal(raw, &resp); err != nil {
		return fmt.Errorf("agent: malformed response: %w", err)
	}
	if resp.ID != req.ID {
		return fmt.Errorf("agent: response %d for request %d", resp.ID, req.ID)
	}
	if !resp.OK {
		return &StepError{Step: step, Message: resp.Error}
	}
	if result != nil && len(resp.Result) > 0 {
		return json.Unmarshal(resp.Result, result)
	}
	return nil
}

// Close closes the connection.
func (c *Client) Close() error { return c.conn.Close() }

func wrap(ctx context.Context, err error) error {
	if ctx.Err() != nil {
		return ctx.Err()
	}
	return fmt.Errorf("agent: %w", err)
}

// Handler runs one step and returns its JSON-encodable result.
type Handler func(ctx context.Context, step string, args json.RawMessage) (any, error)

// Final wraps a step's result to end the session once the answer is sent.
type Final struct{ Result any }

// Serve answers requests on conn until the peer closes it. The first failed
// step poisons the session: later steps get ErrPoisoned without running.
func Serve(ctx context.Context, conn net.Conn, handle Handler) error {
	defer conn.Close()
	stop := context.AfterFunc(ctx, func() { _ = conn.SetDeadline(time.Now()) })
	defer stop()
	r := bufio.NewReaderSize(conn, 64<<10)
	var poisoned error
	for {
		raw, err := readLine(r, MaxLine)
		if errors.Is(err, io.EOF) {
			return nil
		}
		if err != nil {
			return wrap(ctx, err)
		}
		var req Request
		if err := json.Unmarshal(raw, &req); err != nil {
			return fmt.Errorf("agent: malformed request: %w", err)
		}
		resp, final := answer(ctx, req, handle, poisoned)
		if !resp.OK && poisoned == nil {
			poisoned = fmt.Errorf("%w: %s: %s", ErrPoisoned, req.Step, resp.Error)
		}
		line, err := json.Marshal(resp)
		if err != nil {
			return err
		}
		if _, err := conn.Write(append(line, '\n')); err != nil {
			return wrap(ctx, err)
		}
		if final {
			return nil
		}
	}
}

func answer(ctx context.Context, req Request, handle Handler, poisoned error) (Response, bool) {
	resp := Response{ID: req.ID}
	if poisoned != nil {
		resp.Error = poisoned.Error()
		return resp, false
	}
	result, err := handle(ctx, req.Step, req.Args)
	if err != nil {
		resp.Error = err.Error()
		return resp, false
	}
	final := false
	if f, ok := result.(Final); ok {
		result, final = f.Result, true
	}
	if result != nil {
		if resp.Result, err = json.Marshal(result); err != nil {
			resp.Error = err.Error()
			return resp, false
		}
	}
	resp.OK = true
	return resp, final
}

func readLine(r *bufio.Reader, limit int) ([]byte, error) {
	var out []byte
	for {
		chunk, err := r.ReadSlice('\n')
		out = append(out, chunk...)
		if len(out) > limit {
			return nil, fmt.Errorf("line exceeds %d bytes", limit)
		}
		if err == nil {
			return out[:len(out)-1], nil
		}
		if errors.Is(err, bufio.ErrBufferFull) {
			continue
		}
		if errors.Is(err, io.EOF) && len(out) > 0 {
			return nil, io.ErrUnexpectedEOF
		}
		return nil, err
	}
}
