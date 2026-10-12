package process

import (
	"context"
	"errors"
	"testing"
	"time"
)

func TestCapture(t *testing.T) {
	out, err := Capture(context.Background(), 5*time.Second, "sh", "-c", "echo hi; echo err >&2")
	if err != nil || out.Stdout != "hi" || out.Stderr != "err" {
		t.Fatalf("%+v %v", out, err)
	}
	_, err = Capture(context.Background(), 5*time.Second, "sh", "-c", "exit 3")
	if err == nil {
		t.Fatal("exit status ignored")
	}
}

func TestCaptureTimeout(t *testing.T) {
	start := time.Now()
	_, err := Capture(context.Background(), 200*time.Millisecond, "sh", "-c", "sleep 5")
	if !errors.Is(err, context.DeadlineExceeded) || time.Since(start) > 3*time.Second {
		t.Fatalf("%v after %s", err, time.Since(start))
	}
}

func TestCaptureLimit(t *testing.T) {
	_, err := Capture(context.Background(), 10*time.Second, "sh", "-c", "head -c 200000 /dev/zero | tr '\\0' a")
	if !errors.Is(err, ErrOutputLimit) {
		t.Fatalf("%v", err)
	}
}
