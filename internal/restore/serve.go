package restore

import (
	"context"
	"errors"
	"fmt"
	"net"
	"os"
	"path/filepath"
	"time"

	"github.com/TensorRails/warmctl/v2/internal/gateway"
	"github.com/TensorRails/warmctl/v2/internal/profile"
	"github.com/TensorRails/warmctl/v2/internal/run"
	"github.com/TensorRails/warmctl/v2/internal/vllm"
)

func parseSize(s string) (int64, error) {
	if s == "" {
		return 0, nil
	}
	return profile.ParseSize(s)
}

// wake runs the engine side: vLLM's and warmctl's restore hooks on the
// sleeping engine, the weights back to their addresses and checked, then
// the KV cache.
func (r *restore) wake(ctx context.Context) error {
	e := r.c.Engine
	if asleep, err := e.Sleeping(ctx); err != nil || !asleep {
		return fmt.Errorf("restored engine is not asleep (%t, %v)", asleep, err)
	}
	if err := e.FinishRestore(ctx); err != nil {
		return err
	}
	if err := e.WakeUp(ctx, "weights"); err != nil {
		return err
	}
	threads := r.o.Threads
	if threads == 0 {
		threads = 4
	}
	reloaded, err := e.ReloadPool(ctx, r.manifestPath(), r.guest().capture, threads, r.o.VerifyWeights,
		r.o.SyncWeights)
	if err != nil {
		return err
	}
	r.p.Printf("weights reloaded and fingerprints match: %s", reloadSummary(reloaded))
	if r.pf != nil {
		r.p.Printf("prefetch: %s", r.pf)
	}
	if err := e.WakeUp(ctx, "kv_cache"); err != nil {
		return err
	}
	if asleep, err := e.Sleeping(ctx); err != nil || asleep {
		return fmt.Errorf("engine still asleep after wake (%t, %v)", asleep, err)
	}
	return e.Health(ctx)
}

// verify runs the canary under the snapshot's policy (or the operator's).
func (r *restore) verify(ctx context.Context) error {
	policy := r.m.Profile.Canary
	if r.o.NoVerify {
		policy.Mode = "warn"
	}
	if r.o.Canary != "" {
		policy.Mode = r.o.Canary
	}
	if policy.Mode == "off" || r.m.Profile.Restore.Warmup == "none" || r.m.Canary == nil {
		r.p.Printf("canary skipped")
		return nil
	}
	model := r.m.Launch.Model
	if r.m.Profile.Restore.Warmup == "full" {
		if err := r.fullWarmup(ctx, model); err != nil {
			return err
		}
	}
	got, err := r.c.Engine.RunCanary(ctx, model, r.m.Canary.PromptTokens, r.m.Canary.MaxTokens)
	if err != nil {
		return err
	}
	summary, err := vllm.CompareCanary(policy, *r.m.Canary, got)
	r.p.Printf("%s (policy %s; miss %.3fs, hit %.3fs)", summary, policy.Mode, got.MissSeconds, got.HitSeconds)
	return err
}

func (r *restore) fullWarmup(ctx context.Context, model string) error {
	defaults := r.m.Profile
	launch, err := profile.Resolve(profile.Profile{Schema: profile.DefaultsSchema}, defaults)
	if err != nil {
		return err
	}
	if err := r.c.Engine.Warmup(ctx, model, max(defaults.Parallel.DP, 1), launch.WarmupCases(), nil); err != nil {
		return err
	}
	return r.c.Engine.ResetPrefixCache(ctx)
}

// admit opens the host gateway, or a pod's gate.
func (r *restore) admit(context.Context) error {
	if g := r.o.Gate; g != nil {
		r.gw = g.gw
		r.gw.SetReady(true)
		r.p.Marker("serving started %.1fs", r.p.Elapsed())
		r.p.Printf("serving %s on %s", r.m.Profile.ID, r.o.TLS.URL(g.Addr()))
		return nil
	}
	listen := r.o.Listen
	if listen == "" {
		listen = "127.0.0.1:8000"
	}
	l, err := net.Listen("tcp", listen)
	if err != nil {
		return err
	}
	r.listener = r.o.TLS.Listen(l)
	r.gw = gateway.New(filepath.Join(r.dir, "io", "inference.sock"), func() {
		r.p.Marker("first host-gateway token %.1fs", r.p.Elapsed())
	})
	r.gw.SetReady(true)
	r.p.Marker("serving started %.1fs", r.p.Elapsed())
	r.p.Printf("serving %s on %s", r.m.Profile.ID, r.o.TLS.URL(l.Addr()))
	return nil
}

// serve forwards requests until ctx ends or the engine exits.
func (r *restore) serve(ctx context.Context) error {
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	exited := make(chan error, 1)
	go func() {
		code, err := r.c.Wait(context.Background())
		if err == nil {
			err = fmt.Errorf("%w (container exit %d)", ErrStopped, code)
		}
		exited <- err
		cancel()
	}()
	if r.o.Gate != nil {
		return r.servePod(ctx, exited)
	}
	done := make(chan error, 1)
	listener := r.listener
	r.listener = nil
	go func() { done <- gateway.Serve(ctx, listener, r.gw, 30*time.Second) }()
	select {
	case err := <-exited:
		<-done
		r.p.Printf("engine stopped: %v", err)
		return nil
	case <-ctx.Done():
		err := <-done
		r.p.Printf("stopping")
		if err != nil && !errors.Is(err, net.ErrClosed) {
			return err
		}
		return nil
	}
}

// servePod waits on a pod's gate, copying the engine's log to the pod's
// output, and drains the gate when the engine exits or ctx ends.
func (r *restore) servePod(ctx context.Context, exited <-chan error) error {
	go run.Follow(ctx, r.c.Log, os.Stderr)
	select {
	case err := <-exited:
		r.p.Printf("engine stopped: %v", err)
	case <-ctx.Done():
		r.p.Printf("stopping")
	case <-r.o.Gate.done:
		return fmt.Errorf("gateway: %w", r.o.Gate.err)
	}
	return r.o.Gate.Close()
}

// reloadSummary is the weights phase in one line: the bytes, the slowest
// rank and, from adapters that split it, where that rank's time went.
func reloadSummary(ranks []vllm.Reloaded) string {
	var bytes int64
	var slow vllm.Reloaded
	for _, x := range ranks {
		bytes += x.Bytes
		if x.Seconds >= slow.Seconds {
			slow = x
		}
	}
	out := fmt.Sprintf("%.2f GiB, slowest rank %.1fs", float64(bytes)/(1<<30), slow.Seconds)
	if slow.Reload == 0 && slow.Fingerprint == 0 {
		return out
	}
	mode := "sync"
	if slow.Pipelined {
		mode = "pipelined"
	}
	if slow.Fused {
		mode += ", fingerprinted as it landed"
	}
	gib := float64(slow.Bytes) / (1 << 30)
	return out + fmt.Sprintf(" (rank %d: copy %.1fs at %.1f GiB/s, %s, %d threads spent read %.1fs and copy wait %.1fs; "+
		"fingerprint %.1fs)", slow.Rank, slow.Reload, gib/max(slow.Reload, 1e-9), mode, slow.Threads, slow.Read,
		slow.Copy, slow.Fingerprint)
}
