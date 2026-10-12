package agent

import (
	"context"
	"encoding/json"
	"errors"
	"net"
	"strings"
	"testing"
	"time"
)

func echoHandler(ctx context.Context, step string, args json.RawMessage) (any, error) {
	switch step {
	case "echo":
		var v map[string]any
		if err := json.Unmarshal(args, &v); err != nil {
			return nil, err
		}
		return v, nil
	case "slow":
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-time.After(5 * time.Second):
			return "late", nil
		}
	case "fail":
		return nil, errors.New("boom")
	}
	return nil, errors.New("unknown step " + step)
}

func TestCallAndPoison(t *testing.T) {
	a, b := net.Pipe()
	done := make(chan error, 1)
	go func() { done <- Serve(context.Background(), b, echoHandler) }()
	c := NewClient(a)
	var got map[string]any
	if err := c.Call(context.Background(), "echo", map[string]any{"x": 1.5, "big": strings.Repeat("y", 200<<10)}, &got); err != nil {
		t.Fatal(err)
	}
	if got["x"] != 1.5 || len(got["big"].(string)) != 200<<10 {
		t.Fatalf("echo returned %v", got["x"])
	}
	err := c.Call(context.Background(), "fail", nil, nil)
	var se *StepError
	if !errors.As(err, &se) || se.Step != "fail" || se.Message != "boom" {
		t.Fatalf("fail returned %v", err)
	}
	err = c.Call(context.Background(), "echo", map[string]any{}, nil)
	if !errors.As(err, &se) || !strings.Contains(se.Message, ErrPoisoned.Error()) {
		t.Fatalf("poisoned agent answered %v", err)
	}
	c.Close()
	if err := <-done; err != nil {
		t.Fatalf("serve: %v", err)
	}
}

func TestCancel(t *testing.T) {
	a, b := net.Pipe()
	go func() { _ = Serve(context.Background(), b, echoHandler) }()
	c := NewClient(a)
	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()
	start := time.Now()
	err := c.Call(ctx, "slow", nil, nil)
	if !errors.Is(err, context.DeadlineExceeded) || time.Since(start) > 2*time.Second {
		t.Fatalf("cancel returned %v after %s", err, time.Since(start))
	}
	c.Close()
}

func TestLineLimit(t *testing.T) {
	a, b := net.Pipe()
	done := make(chan error, 1)
	go func() { done <- Serve(context.Background(), b, echoHandler) }()
	go func() {
		_, _ = a.Write([]byte(strings.Repeat("x", MaxLine+2)))
		a.Close()
	}()
	if err := <-done; err == nil || !strings.Contains(err.Error(), "exceeds") {
		t.Fatalf("oversized line: %v", err)
	}
}
