// A stand-in libcuda.so.1 for the driver backend's tests: the checkpoint
// API of CUDA 13 with a state per PID. It checks what a real driver would
// refuse (reserved fields set, a lock without a timeout, a restore without
// GPU pairs, a call out of order) and records the GPU pairs it was given.
// The version comes from VERSION at build time. It calls nothing from libc.

#ifndef VERSION
#define VERSION 13000
#endif

typedef unsigned long long u64;
typedef struct { unsigned char bytes[16]; } CUuuid;
typedef struct { unsigned int timeoutMs, reserved0; u64 reserved1[7]; } LockArgs;
typedef struct { u64 reserved[8]; } PlainArgs;
typedef struct { CUuuid oldUuid, newUuid; } GpuPair;
typedef struct { GpuPair *gpuPairs; unsigned int gpuPairsCount; char reserved[44]; u64 reserved1; } RestoreArgs;

enum { SUCCESS = 0, INVALID_VALUE = 1, ILLEGAL_STATE = 401, NOT_FOUND = 500, UNKNOWN = 999 };
enum { RUNNING = 0, LOCKED = 1, CHECKPOINTED = 2 };

// PIDs 100..115 exist; the fakeLast functions report the last lock's
// timeout and the last restore's pairs.
static int states[16];
static GpuPair lastPairs[8];
static unsigned int lastCount, lastTimeout;

unsigned int fakeLastTimeout(void) { return lastTimeout; }

unsigned int fakeLastPairs(GpuPair *out) {
	for (unsigned int k = 0; k < lastCount; k++)
		out[k] = lastPairs[k];
	return lastCount;
}

static int slot(int pid) { return pid >= 100 && pid < 116 ? pid - 100 : -1; }

static int zero(const void *p, int n) {
	const unsigned char *b = p;
	for (int i = 0; i < n; i++)
		if (b[i])
			return 0;
	return 1;
}

int cuDriverGetVersion(int *v) {
	*v = VERSION;
	return SUCCESS;
}

int cuGetErrorString(int e, const char **s) {
	switch (e) {
	case INVALID_VALUE: *s = "invalid argument"; return SUCCESS;
	case ILLEGAL_STATE: *s = "operation not permitted when stream is capturing"; return SUCCESS;
	case NOT_FOUND: *s = "named symbol not found"; return SUCCESS;
	}
	return INVALID_VALUE;
}

int cuCheckpointProcessGetState(int pid, int *state) {
	int i = slot(pid);
	if (i < 0)
		return NOT_FOUND;
	*state = states[i];
	return SUCCESS;
}

static int step(int pid, int from, int to) {
	int i = slot(pid);
	if (i < 0)
		return NOT_FOUND;
	if (states[i] != from)
		return ILLEGAL_STATE;
	states[i] = to;
	return SUCCESS;
}

int cuCheckpointProcessLock(int pid, LockArgs *a) {
	if (!a || a->timeoutMs == 0 || a->reserved0 || !zero(a->reserved1, sizeof a->reserved1))
		return INVALID_VALUE;
	lastTimeout = a->timeoutMs;
	return step(pid, RUNNING, LOCKED);
}

int cuCheckpointProcessCheckpoint(int pid, PlainArgs *a) {
	if (a && !zero(a, sizeof *a))
		return INVALID_VALUE;
	return step(pid, LOCKED, CHECKPOINTED);
}

int cuCheckpointProcessRestore(int pid, RestoreArgs *a) {
	if (!a || !a->gpuPairs || a->gpuPairsCount == 0 || a->gpuPairsCount > 8 ||
	    !zero(a->reserved, sizeof a->reserved) || a->reserved1)
		return INVALID_VALUE;
	for (unsigned int k = 0; k < a->gpuPairsCount; k++)
		lastPairs[k] = a->gpuPairs[k];
	lastCount = a->gpuPairsCount;
	return step(pid, CHECKPOINTED, LOCKED);
}

int cuCheckpointProcessUnlock(int pid, PlainArgs *a) {
	if (a && !zero(a, sizeof *a))
		return INVALID_VALUE;
	return step(pid, LOCKED, RUNNING);
}
