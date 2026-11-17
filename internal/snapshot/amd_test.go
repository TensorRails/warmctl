package snapshot

import (
	"strings"
	"testing"

	"github.com/TensorRails/warmctl/v2/internal/host"
	"github.com/TensorRails/warmctl/v2/internal/manifest"
	"github.com/TensorRails/warmctl/v2/internal/vllm"
)

func TestImagePythonPath(t *testing.T) {
	for _, c := range []struct{ profile, image, want string }{
		{"/opt/warmctl/adapter", "/opt/python/lib/python3.14/site-packages/_rocm_sdk_core/share/amd_smi:",
			"/opt/warmctl/adapter:/opt/python/lib/python3.14/site-packages/_rocm_sdk_core/share/amd_smi"},
		{"/opt/warmctl/adapter", "", "/opt/warmctl/adapter"},
		{"/opt/warmctl/adapter", "/opt/warmctl/adapter:/x", "/opt/warmctl/adapter:/x"},
	} {
		env := map[string]string{"PYTHONPATH": c.profile}
		imagePythonPath(env, c.image)
		if env["PYTHONPATH"] != c.want {
			t.Errorf("%q + %q = %q, want %q", c.profile, c.image, env["PYTHONPATH"], c.want)
		}
	}
}

func TestAMDWorkerPlacement(t *testing.T) {
	b := bake{gpus: []host.GPU{{PCIBusID: "0000:c1:00.1"}}, gpuLib: manifest.MappedFile{Version: "0.1.1"}}
	for _, address := range []string{"0000:C1:00.1", "0000:c1:00.0", ""} {
		w := vllm.Worker{KFDReplay: "0.1.1", PCIBusID: address}
		err := b.checkWorkers([]vllm.Worker{w})
		if (err == nil) != strings.EqualFold(address, b.gpus[0].PCIBusID) {
			t.Errorf("worker PCI address %q: %v", address, err)
		}
	}
}
