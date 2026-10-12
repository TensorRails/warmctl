// Command warmctl snapshots warmed vLLM engines and restores them.
package main

import (
	"context"
	"os"
	"os/signal"
	"syscall"

	"github.com/TensorRails/warmctl/v2/internal/cli"
)

var version = "dev"

func main() {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	code := cli.Run(ctx, os.Args[1:], os.Stdout, os.Stderr, version)
	stop()
	os.Exit(code)
}
