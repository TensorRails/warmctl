// Package docker drives the local Docker Engine over its unix socket with
// the handful of calls a snapshot and a restore make: ensure an image by
// digest, create a container from a Spec, start, wait, stop, remove,
// inspect, read logs, list by label. Nothing else.
package docker

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"
)

const (
	apiVersion  = "1.45"
	maxResponse = 4 << 20
	// DefaultSocket is where Docker listens on Linux.
	DefaultSocket = "/var/run/docker.sock"
	// ManagedLabel marks every container warmctl creates.
	ManagedLabel = "io.warmctl.managed"
)

// Client talks to one local daemon.
type Client struct {
	http    *http.Client
	Version Version
}

// Version is the daemon's /version answer.
type Version struct {
	Version       string
	APIVersion    string
	MinAPIVersion string
	Os            string
	Arch          string
}

// APIError is a non-2xx answer.
type APIError struct {
	Status  int
	Message string
}

func (e *APIError) Error() string { return fmt.Sprintf("docker: HTTP %d: %s", e.Status, e.Message) }

// IsNotFound reports a 404 answer.
func IsNotFound(err error) bool {
	var e *APIError
	return errors.As(err, &e) && e.Status == http.StatusNotFound
}

func isStatus(err error, code int) bool {
	var e *APIError
	return errors.As(err, &e) && e.Status == code
}

// New connects to the daemon at socket and checks it speaks API 1.45.
func New(ctx context.Context, socket string) (*Client, error) {
	if socket == "" {
		socket = DefaultSocket
	}
	transport := &http.Transport{
		Proxy: nil,
		DialContext: func(ctx context.Context, _, _ string) (net.Conn, error) {
			return (&net.Dialer{Timeout: 5 * time.Second}).DialContext(ctx, "unix", socket)
		},
		MaxIdleConns:    8,
		IdleConnTimeout: 30 * time.Second,
	}
	c := &Client{http: &http.Client{Transport: transport}}
	if err := c.call(ctx, http.MethodGet, "/version", nil, &c.Version); err != nil {
		return nil, err
	}
	if c.Version.Os != "linux" {
		return nil, fmt.Errorf("docker: daemon runs %s, not linux", c.Version.Os)
	}
	if !between(c.Version.MinAPIVersion, apiVersion, c.Version.APIVersion) {
		return nil, fmt.Errorf("docker: daemon API %s..%s lacks %s",
			c.Version.MinAPIVersion, c.Version.APIVersion, apiVersion)
	}
	return c, nil
}

func minor(v string) int {
	major, rest, ok := strings.Cut(v, ".")
	n, err := strconv.Atoi(rest)
	if !ok || major != "1" || err != nil {
		return -1
	}
	return n
}

func between(low, v, high string) bool {
	return minor(low) >= 0 && minor(low) <= minor(v) && minor(v) <= minor(high)
}

func endpoint(path string) string { return "/v" + apiVersion + path }

func (c *Client) request(ctx context.Context, method, path string, body any) (*http.Response, error) {
	var reader io.Reader
	if body != nil {
		raw, err := json.Marshal(body)
		if err != nil {
			return nil, err
		}
		reader = bytes.NewReader(raw)
	}
	req, err := http.NewRequestWithContext(ctx, method, "http://docker"+path, reader)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("User-Agent", "warmctl")
	return c.http.Do(req)
}

func (c *Client) call(ctx context.Context, method, path string, body, out any) error {
	resp, err := c.request(ctx, method, path, body)
	if err != nil {
		return fmt.Errorf("docker: %w", err)
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(io.LimitReader(resp.Body, maxResponse+1))
	if err != nil {
		return err
	}
	if len(raw) > maxResponse {
		return errors.New("docker: response exceeds limit")
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return apiError(resp.StatusCode, raw)
	}
	if out == nil || len(raw) == 0 {
		return nil
	}
	return json.Unmarshal(raw, out)
}

func apiError(code int, raw []byte) error {
	var v struct {
		Message string `json:"message"`
	}
	_ = json.Unmarshal(raw, &v)
	if v.Message == "" {
		v.Message = http.StatusText(code)
	}
	return &APIError{Status: code, Message: v.Message}
}

// Info is the part of /info warmctl reads.
type Info struct {
	InitBinary string
	Runtimes   map[string]any
}

// Info reads the daemon's facts.
func (c *Client) Info(ctx context.Context) (Info, error) {
	var info Info
	err := c.call(ctx, http.MethodGet, endpoint("/info"), nil, &info)
	return info, err
}

// CheckInit fails when the daemon names no init binary: warmctl's
// containers run docker-init as PID 1 (HostConfig.Init). The init commit
// is not checked: distribution builds leave it empty with a working init.
func (c *Client) CheckInit(ctx context.Context) error {
	info, err := c.Info(ctx)
	if err != nil {
		return err
	}
	if info.InitBinary == "" {
		return errors.New("docker: the daemon has no init binary; install docker-init (tini)")
	}
	return nil
}

// Image is what EnsureImage found.
type Image struct {
	ID           string `json:"Id"`
	RepoDigests  []string
	Os           string
	Architecture string
}

// EnsureImage makes ref (name@sha256:...) present, pulling it when missing,
// and returns the local image. A tag without a digest is refused.
func (c *Client) EnsureImage(ctx context.Context, ref string, progress io.Writer) (Image, error) {
	if strings.HasPrefix(ref, "sha256:") {
		return c.inspectImage(ctx, ref)
	}
	name, digest, ok := strings.Cut(ref, "@")
	if !ok || !strings.HasPrefix(digest, "sha256:") {
		return Image{}, fmt.Errorf("docker: image %q is not pinned by digest", ref)
	}
	img, err := c.inspectImage(ctx, ref)
	if IsNotFound(err) {
		if err := c.pull(ctx, name, digest, progress); err != nil {
			return Image{}, err
		}
		img, err = c.inspectImage(ctx, ref)
	}
	if err != nil {
		return Image{}, err
	}
	for _, d := range img.RepoDigests {
		if strings.HasSuffix(d, "@"+digest) {
			return img, nil
		}
	}
	return Image{}, fmt.Errorf("docker: local image %s does not carry digest %s", img.ID, digest)
}

// InspectImage reads a local image by reference or ID.
func (c *Client) InspectImage(ctx context.Context, ref string) (Image, error) {
	return c.inspectImage(ctx, ref)
}

func (c *Client) inspectImage(ctx context.Context, ref string) (Image, error) {
	var img Image
	err := c.call(ctx, http.MethodGet, endpoint("/images/"+url.PathEscape(ref)+"/json"), nil, &img)
	return img, err
}

func (c *Client) pull(ctx context.Context, name, digest string, progress io.Writer) error {
	if i := strings.LastIndex(name, ":"); i > strings.LastIndex(name, "/") {
		name = name[:i]
	}
	query := url.Values{"fromImage": {name}, "tag": {digest}}
	resp, err := c.request(ctx, http.MethodPost, endpoint("/images/create?")+query.Encode(), nil)
	if err != nil {
		return fmt.Errorf("docker: pull: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		raw, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return apiError(resp.StatusCode, raw)
	}
	dec := json.NewDecoder(resp.Body)
	for {
		var event struct {
			Status string `json:"status"`
			ID     string `json:"id"`
			Error  string `json:"error"`
		}
		if err := dec.Decode(&event); errors.Is(err, io.EOF) {
			return nil
		} else if err != nil {
			return fmt.Errorf("docker: pull: %w", err)
		}
		if event.Error != "" {
			return fmt.Errorf("docker: pull %s: %s", name, event.Error)
		}
		if progress != nil && event.ID == "" {
			fmt.Fprintln(progress, event.Status)
		}
	}
}

// Create makes a container from spec and returns its ID.
func (c *Client) Create(ctx context.Context, spec Spec) (string, error) {
	body, err := render(spec)
	if err != nil {
		return "", err
	}
	var out struct {
		ID string `json:"Id"`
	}
	path := endpoint("/containers/create?name=") + url.QueryEscape(spec.Name)
	if err := c.call(ctx, http.MethodPost, path, body, &out); err != nil {
		return "", err
	}
	return out.ID, nil
}

// Start starts a created container.
func (c *Client) Start(ctx context.Context, id string) error {
	err := c.call(ctx, http.MethodPost, endpoint("/containers/"+url.PathEscape(id)+"/start"), nil, nil)
	if isStatus(err, http.StatusNotModified) {
		return nil
	}
	return err
}

// Wait blocks until the container stops and returns its exit code.
func (c *Client) Wait(ctx context.Context, id string) (int, error) {
	var out struct {
		StatusCode int
		Error      *struct{ Message string }
	}
	path := endpoint("/containers/" + url.PathEscape(id) + "/wait?condition=not-running")
	err := c.call(ctx, http.MethodPost, path, nil, &out)
	if err != nil {
		return -1, err
	}
	if out.Error != nil && out.Error.Message != "" {
		return out.StatusCode, fmt.Errorf("docker: wait: %s", out.Error.Message)
	}
	return out.StatusCode, nil
}

// Stop sends SIGTERM, then SIGKILL after grace. A stopped or missing
// container is not an error.
func (c *Client) Stop(ctx context.Context, id string, grace time.Duration) error {
	path := fmt.Sprintf("/containers/%s/stop?t=%d", url.PathEscape(id), int(grace.Seconds()))
	err := c.call(ctx, http.MethodPost, endpoint(path), nil, nil)
	if isStatus(err, http.StatusNotModified) || IsNotFound(err) {
		return nil
	}
	return err
}

// Remove force-removes the container. A missing container is not an error.
func (c *Client) Remove(ctx context.Context, id string) error {
	err := c.call(ctx, http.MethodDelete, endpoint("/containers/"+url.PathEscape(id)+"?force=true"), nil, nil)
	if IsNotFound(err) {
		return nil
	}
	return err
}

// State is the part of an inspection warmctl reads.
type State struct {
	ID     string `json:"Id"`
	Name   string
	Image  string
	Config struct{ Labels map[string]string }
	State  struct {
		Status    string
		Running   bool
		Pid       int
		ExitCode  int
		Error     string
		OOMKilled bool
	}
}

// Inspect reads a container's state.
func (c *Client) Inspect(ctx context.Context, id string) (State, error) {
	var s State
	err := c.call(ctx, http.MethodGet, endpoint("/containers/"+url.PathEscape(id)+"/json"), nil, &s)
	return s, err
}

// Summary is one row of List.
type Summary struct {
	ID     string `json:"Id"`
	Names  []string
	State  string
	Labels map[string]string
}

// List returns the containers carrying label (key or key=value), running or not.
func (c *Client) List(ctx context.Context, label string) ([]Summary, error) {
	filters, err := json.Marshal(map[string][]string{"label": {label}})
	if err != nil {
		return nil, err
	}
	var out []Summary
	path := endpoint("/containers/json?all=1&filters=") + url.QueryEscape(string(filters))
	err = c.call(ctx, http.MethodGet, path, nil, &out)
	return out, err
}

// Logs copies the container's stdout and stderr, the last tail lines when
// tail > 0, into w.
func (c *Client) Logs(ctx context.Context, id string, tail int, w io.Writer) error {
	query := "?stdout=1&stderr=1"
	if tail > 0 {
		query += "&tail=" + strconv.Itoa(tail)
	}
	resp, err := c.request(ctx, http.MethodGet, endpoint("/containers/"+url.PathEscape(id)+"/logs"+query), nil)
	if err != nil {
		return fmt.Errorf("docker: logs: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		raw, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return apiError(resp.StatusCode, raw)
	}
	return demux(resp.Body, w)
}

// demux strips Docker's 8-byte stream frame headers.
func demux(r io.Reader, w io.Writer) error {
	var header [8]byte
	for {
		if _, err := io.ReadFull(r, header[:]); errors.Is(err, io.EOF) {
			return nil
		} else if err != nil {
			return err
		}
		size := binary.BigEndian.Uint32(header[4:])
		if header[0] > 2 || size > 4<<20 {
			return errors.New("docker: malformed log frame")
		}
		if _, err := io.CopyN(w, r, int64(size)); err != nil {
			return err
		}
	}
}
