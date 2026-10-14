// A stand-in libnvidia-ml.so.1 for the CPU e2e test (make e2e-test): two
// idle GPUs joined by NVLink, matching testdata/hardware.yaml's fake-gpu
// row. It calls nothing from libc, so it loads in any container.

typedef struct {
	char busIdLegacy[16];
	unsigned int domain, bus, device, pciDeviceId, pciSubSystemId;
	char busId[32];
} nvmlPciInfo_t;

typedef struct {
	unsigned long long total, free, used;
} nvmlMemory_t;

typedef void *nvmlDevice_t;

enum { SUCCESS = 0, INVALID_ARGUMENT = 2, INSUFFICIENT_SIZE = 7 };

static int put(char *dst, unsigned int size, const char *src) {
	unsigned int i;
	for (i = 0; src[i]; i++) {
		if (i + 1 >= size)
			return INSUFFICIENT_SIZE;
		dst[i] = src[i];
	}
	dst[i] = 0;
	return SUCCESS;
}

static int index_of(nvmlDevice_t d) { return (int)(unsigned long)d - 1; }

int nvmlInit_v2(void) { return SUCCESS; }
int nvmlShutdown(void) { return SUCCESS; }

const char *nvmlErrorString(int result) {
	return result == INSUFFICIENT_SIZE ? "Insufficient Size" : "Fake Error";
}

int nvmlSystemGetDriverVersion(char *version, unsigned int size) { return put(version, size, "1.0"); }

int nvmlDeviceGetCount_v2(unsigned int *count) {
	*count = 2;
	return SUCCESS;
}

int nvmlDeviceGetHandleByIndex_v2(unsigned int index, nvmlDevice_t *device) {
	if (index > 1)
		return INVALID_ARGUMENT;
	*device = (nvmlDevice_t)(unsigned long)(index + 1);
	return SUCCESS;
}

int nvmlDeviceGetUUID(nvmlDevice_t d, char *uuid, unsigned int size) {
	return put(uuid, size, index_of(d) ? "GPU-11111111-1111-1111-1111-111111111111"
	                                   : "GPU-00000000-0000-0000-0000-000000000000");
}

int nvmlDeviceGetName(nvmlDevice_t d, char *name, unsigned int size) {
	(void)d;
	return put(name, size, "Fake GPU");
}

int nvmlDeviceGetPciInfo_v3(nvmlDevice_t d, nvmlPciInfo_t *pci) {
	pci->domain = 0;
	pci->bus = (unsigned int)index_of(d) + 1;
	pci->device = 0;
	return put(pci->busId, sizeof pci->busId, index_of(d) ? "00000000:02:00.0" : "00000000:01:00.0");
}

int nvmlDeviceGetMemoryInfo(nvmlDevice_t d, nvmlMemory_t *memory) {
	(void)d;
	memory->total = 1024ULL << 20;
	memory->free = memory->total;
	memory->used = 0;
	return SUCCESS;
}

int nvmlDeviceGetComputeRunningProcesses_v3(nvmlDevice_t d, unsigned int *count, void *infos) {
	(void)d;
	(void)infos;
	*count = 0;
	return SUCCESS;
}

int nvmlDeviceGetP2PStatus(nvmlDevice_t a, nvmlDevice_t b, int caps, int *status) {
	(void)a;
	(void)b;
	*status = caps == 2 ? 0 : 5; // NVLink OK, anything else not supported
	return SUCCESS;
}

int nvmlDeviceGetTopologyCommonAncestor(nvmlDevice_t a, nvmlDevice_t b, int *level) {
	(void)a;
	(void)b;
	*level = 30; // PHB
	return SUCCESS;
}
