// A stand-in libcuda.so.1 for TestProbe: an H100 with VMM, POSIX handles and
// multicast, and a vGPU with none. Device handles are not ordinals, so a
// probe that confuses the two fails. It calls nothing from libc.

typedef int CUdevice;

enum { SUCCESS = 0, INVALID_VALUE = 1, NOT_INITIALIZED = 3, INVALID_DEVICE = 101 };

static int initialized;
static const char *names[] = {"NVIDIA H100 80GB HBM3", "Fake vGPU"};

static int index_of(CUdevice dev) { return dev - 10; }

int cuInit(unsigned int flags) {
	(void)flags;
	initialized = 1;
	return SUCCESS;
}

int cuDeviceGetCount(int *count) {
	if (!initialized)
		return NOT_INITIALIZED;
	*count = 2;
	return SUCCESS;
}

int cuDeviceGet(CUdevice *device, int ordinal) {
	if (!initialized)
		return NOT_INITIALIZED;
	if (ordinal < 0 || ordinal > 1)
		return INVALID_DEVICE;
	*device = ordinal + 10;
	return SUCCESS;
}

int cuDeviceGetName(char *name, int len, CUdevice dev) {
	int i = index_of(dev), k;
	const char *src;
	if (i < 0 || i > 1)
		return INVALID_DEVICE;
	src = names[i];
	for (k = 0; src[k] && k + 1 < len; k++)
		name[k] = src[k];
	name[k] = 0;
	return SUCCESS;
}

int cuDeviceGetAttribute(int *value, int attribute, CUdevice dev) {
	int i = index_of(dev);
	if (i < 0 || i > 1)
		return INVALID_DEVICE;
	switch (attribute) {
	case 75: /* compute capability major */
		*value = i == 0 ? 9 : 8;
		return SUCCESS;
	case 76: /* minor */
		*value = i == 0 ? 0 : 9;
		return SUCCESS;
	case 102: /* VMM */
	case 103: /* POSIX file descriptor handles */
	case 132: /* multicast */
		*value = i == 0;
		return SUCCESS;
	default:
		return INVALID_VALUE;
	}
}

int cuGetErrorString(int error, const char **str) {
	switch (error) {
	case SUCCESS:
		*str = "no error";
		return SUCCESS;
	case NOT_INITIALIZED:
		*str = "initialization error";
		return SUCCESS;
	case INVALID_DEVICE:
		*str = "invalid device ordinal";
		return SUCCESS;
	default:
		*str = 0;
		return INVALID_VALUE;
	}
}
