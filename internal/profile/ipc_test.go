package profile

import "testing"

func TestIPCMode(t *testing.T) {
	for _, c := range []struct {
		ipc  string
		tp   int
		want string
	}{
		{"", 8, "native"}, {"native", 8, "native"}, {"interpose", 1, "interpose"},
		{"auto", 1, "native"}, {"auto", 2, "auto"},
	} {
		p := Profile{IPC: c.ipc, Parallel: Parallel{TP: c.tp}}
		if got := p.IPCMode(); got != c.want {
			t.Errorf("ipc %q tp %d: %s, want %s", c.ipc, c.tp, got, c.want)
		}
	}
}
