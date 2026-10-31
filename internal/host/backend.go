package host

import (
	"fmt"
	"os"
	"strconv"
	"strings"

	"golang.org/x/sys/unix"
)

// Process snapshot backends, as agent.BackendCRIU and agent.BackendDMTCP.
const (
	BackendAuto  = "auto"
	BackendCRIU  = "criu"
	BackendDMTCP = "dmtcp"
)

// Capability numbers CRIU's restore needs one of.
const (
	capSysAdmin          = 21
	capCheckpointRestore = 40
)

// BackendProbe is what decides whether this container may run CRIU.
type BackendProbe struct {
	// CapEff is the effective capability set of this process.
	CapEff uint64
	// LastPIDWritable: /proc/sys/kernel/ns_last_pid can be written, so the
	// agent can move the PID allocator and CRIU can restore PIDs.
	LastPIDWritable bool
	// Seccomp is the seccomp mode: 0 none, 2 a filter (Docker's default
	// refuses unshare and clone3, which CRIU's restore needs).
	Seccomp int
}

// ProbeBackend reads this process's capabilities and seccomp mode and tries
// ns_last_pid for writing.
func ProbeBackend() (BackendProbe, error) {
	status, err := os.ReadFile("/proc/self/status")
	if err != nil {
		return BackendProbe{}, err
	}
	p := ParseStatus(string(status))
	p.LastPIDWritable = unix.Access("/proc/sys/kernel/ns_last_pid", unix.W_OK) == nil
	return p, nil
}

// ParseStatus takes CapEff and Seccomp from /proc/self/status.
func ParseStatus(status string) BackendProbe {
	p := BackendProbe{Seccomp: -1}
	for _, line := range strings.Split(status, "\n") {
		key, value, _ := strings.Cut(line, ":")
		value = strings.TrimSpace(value)
		switch key {
		case "CapEff":
			p.CapEff, _ = strconv.ParseUint(value, 16, 64)
		case "Seccomp":
			p.Seccomp, _ = strconv.Atoi(value)
		}
	}
	return p
}

// CRIUReason is why CRIU cannot run here, or "" when it can.
func (p BackendProbe) CRIUReason() string {
	var why []string
	if p.CapEff&(1<<capSysAdmin) == 0 && p.CapEff&(1<<capCheckpointRestore) == 0 {
		why = append(why, "no CAP_SYS_ADMIN or CAP_CHECKPOINT_RESTORE")
	}
	if !p.LastPIDWritable {
		why = append(why, "ns_last_pid is read-only")
	}
	if p.Seccomp != 0 {
		why = append(why, fmt.Sprintf("seccomp mode %d", p.Seccomp))
	}
	return strings.Join(why, ", ")
}
