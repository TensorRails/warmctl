package snapshot

import "testing"

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
