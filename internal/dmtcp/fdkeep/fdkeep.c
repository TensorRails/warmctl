// DMTCP plugin: keep eventfds and pipes that DMTCP did not track.
// The CUDA driver creates the eventfds and pipes its helper threads poll during
// cuInit; DMTCP drops them on restart, the helper threads see POLLNVAL and exit,
// and cuda-checkpoint can no longer find the restore thread. Before a checkpoint
// we record every eventfd and pipe end; on restart we recreate the ones that are
// missing at the same fd numbers, with the same flags, counter and pipe bytes.
#define _GNU_SOURCE
#include <dirent.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/epoll.h>
#include <sys/eventfd.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <sys/prctl.h>
#include <sys/socket.h>
#include <unistd.h>
#include "dmtcp.h"

#define MAXFD 512
#define PIPEBUF 65536

enum kind { NONE, EVENTFD, PIPE, LISTEN };

struct rec {
  enum kind kind;
  int flags;                // file status flags from fdinfo (octal)
  int cloexec;
  unsigned long long count; // eventfd counter
  unsigned long inode;      // pipe inode, pairs read and write ends
  int len;                  // pipe bytes pending (read end only)
  char *data;
};

static struct rec recs[MAXFD];

static void logf_(const char *fmt, ...)
{
  char buf[512];
  va_list ap;
  va_start(ap, fmt);
  int n = vsnprintf(buf, sizeof buf, fmt, ap);
  va_end(ap);
  int fd = open("/tmp/fdkeep.log", O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0644);
  if (fd >= 0) {
    if (write(fd, buf, n) < 0) {}
    close(fd);
  }
}

static int fdinfo(int fd, int *flags, unsigned long long *count)
{
  char path[64], line[256];
  snprintf(path, sizeof path, "/proc/self/fdinfo/%d", fd);
  FILE *f = fopen(path, "re");
  if (!f) return -1;
  while (fgets(line, sizeof line, f)) {
    if (sscanf(line, "flags: %o", flags) == 1) continue;
    sscanf(line, "eventfd-count: %llx", count);
  }
  fclose(f);
  return 0;
}

static int nwatches;
static void record_epoll(int epfd);

static void record(void)
{
  memset(recs, 0, sizeof recs);
  nwatches = 0;
  DIR *d = opendir("/proc/self/fd");
  if (!d) return;
  int self = dirfd(d);
  struct dirent *e;
  while ((e = readdir(d))) {
    int fd = atoi(e->d_name);
    if (e->d_name[0] == '.' || fd == self || fd >= MAXFD) continue;
    char path[64], target[128];
    snprintf(path, sizeof path, "/proc/self/fd/%d", fd);
    ssize_t n = readlink(path, target, sizeof target - 1);
    if (n <= 0) continue;
    target[n] = 0;
    struct rec *r = &recs[fd];
    if (strcmp(target, "anon_inode:[eventfd]") == 0) {
      r->kind = EVENTFD;
    } else if (sscanf(target, "pipe:[%lu]", &r->inode) == 1) {
      r->kind = PIPE;
    } else if (strcmp(target, "anon_inode:[eventpoll]") == 0) {
      record_epoll(fd);
      continue;
    } else if (strncmp(target, "socket:", 7) == 0) {
      // DMTCP sometimes misses listen() and restores the socket bound but not listening.
      int on = 0;
      socklen_t len = sizeof on;
      if (getsockopt(fd, SOL_SOCKET, SO_ACCEPTCONN, &on, &len) == 0 && on) r->kind = LISTEN;
      continue;
    } else {
      continue;
    }
    fdinfo(fd, &r->flags, &r->count);
    r->cloexec = (fcntl(fd, F_GETFD) & FD_CLOEXEC) != 0;
    if (r->kind == PIPE && (r->flags & O_ACCMODE) == O_RDONLY) {
      int avail = 0;
      if (ioctl(fd, FIONREAD, &avail) == 0 && avail > 0) {
        // Take the bytes out to save them, then put them back for the live process.
        r->data = malloc(avail);
        int fl = fcntl(fd, F_GETFL);
        fcntl(fd, F_SETFL, fl | O_NONBLOCK);
        r->len = read(fd, r->data, avail);
        fcntl(fd, F_SETFL, fl);
        if (r->len < 0) r->len = 0;
        for (int w = 0; w < MAXFD && r->len > 0; w++) {
          if (recs[w].kind == PIPE && recs[w].inode == r->inode && (recs[w].flags & O_ACCMODE) == O_WRONLY) {
            if (write(w, r->data, r->len) < 0) {}
            break;
          }
        }
      }
    }
  }
  closedir(d);
  int ne = 0, np = 0, nl = 0;
  for (int fd = 0; fd < MAXFD; fd++) {
    if (recs[fd].kind == EVENTFD) ne++;
    if (recs[fd].kind == PIPE) np++;
    if (recs[fd].kind == LISTEN) nl++;
  }
  logf_("pid %d precheckpoint: %d eventfds, %d pipe ends, %d listening sockets, %d epoll watches\n", getpid(), ne,
        np, nl, nwatches);
}

static long raw_pipe2(int *p, int flags)
{
  long ret;
  __asm__ volatile("syscall" : "=a"(ret) : "0"(293L /* SYS_pipe2 */), "D"(p), "S"((long)flags) : "rcx", "r11", "memory");
  return ret;
}

// Epoll watches. A forked child shares its parent's epoll fds and sockets, and DMTCP refills a shared
// epoll from the watch list of whichever process restores it: when that is the child, the watches the
// parent added after the fork (uvloop's listening socket in vLLM's API server) are gone, and the
// server never accepts. Record the kernel's own list, keeping only watches whose fd in this process
// is still the watched file, and add back whatever is missing after restart.
#define MAXWATCH 4096
struct watch {
  int epfd, tfd;
  unsigned int events;
  unsigned long long data;
};
static struct watch watches[MAXWATCH];

static void record_epoll(int epfd)
{
  char path[64], line[256];
  snprintf(path, sizeof path, "/proc/self/fdinfo/%d", epfd);
  FILE *f = fopen(path, "re");
  if (!f) return;
  while (fgets(line, sizeof line, f) && nwatches < MAXWATCH) {
    int tfd;
    unsigned int events;
    unsigned long long data;
    unsigned long ino;
    if (sscanf(line, "tfd: %d events: %x data: %llx pos:%*d ino:%lx", &tfd, &events, &data, &ino) != 4) continue;
    struct stat st;
    if (fstat(tfd, &st) != 0 || st.st_ino != ino) continue; // registered through another process's fd table
    watches[nwatches++] = (struct watch){epfd, tfd, events, data};
  }
  fclose(f);
}

static long raw_epoll_ctl(int epfd, int op, int fd, struct epoll_event *ev)
{
  long ret;
  register long r10 __asm__("r10") = (long)ev;
  __asm__ volatile("syscall" : "=a"(ret) : "0"(233L /* SYS_epoll_ctl */), "D"((long)epfd), "S"((long)op), "d"((long)fd), "r"(r10) : "rcx", "r11", "memory");
  return ret;
}

static void rewatch(void)
{
  int added = 0, failed = 0;
  for (int i = 0; i < nwatches; i++) {
    struct epoll_event ev = {.events = watches[i].events, .data.u64 = watches[i].data};
    long r = raw_epoll_ctl(watches[i].epfd, EPOLL_CTL_ADD, watches[i].tfd, &ev);
    if (r == 0) added++;
    else if (r != -17 /* EEXIST */) failed++;
  }
  logf_("pid %d restart: %d of %d epoll watches were missing and added back, %d failed\n", getpid(), added, nwatches, failed);
}

// DMTCP's listen() wrapper takes a lock the checkpoint thread may hold while threads resume.
static long raw_listen(int fd, int backlog)
{
  long ret;
  __asm__ volatile("syscall" : "=a"(ret) : "0"(50L /* SYS_listen */), "D"((long)fd), "S"((long)backlog) : "rcx", "r11", "memory");
  return ret;
}

// Move a fresh fd above every slot we may fill, so placing one never clobbers another.
static int high(int fd)
{
  int h = fcntl(fd, F_DUPFD_CLOEXEC, MAXFD + 100);
  close(fd);
  return h;
}

static void place(int src, int fd, const struct rec *r)
{
  dup2(src, fd);
  fcntl(fd, F_SETFL, r->flags & (O_NONBLOCK | O_APPEND));
  fcntl(fd, F_SETFD, r->cloexec ? FD_CLOEXEC : 0);
}

// Is fd open and of the recorded kind? DMTCP restores the pipes it tracked as
// socketpairs, and cuda-checkpoint wakes the restore thread by opening
// /proc/PID/fd/N of its pipe, which fails with ENXIO on a socket.
static int same_kind(int fd, enum kind kind)
{
  char path[64], target[128];
  snprintf(path, sizeof path, "/proc/self/fd/%d", fd);
  ssize_t n = readlink(path, target, sizeof target - 1);
  if (n <= 0) return 0;
  target[n] = 0;
  if (kind == EVENTFD) return strcmp(target, "anon_inode:[eventfd]") == 0;
  return strncmp(target, "pipe:", 5) == 0;
}

static void recreate(void)
{
  int made = 0, kept = 0;
  // A pipe is redone whole when either end is missing or not a pipe.
  for (int fd = 0; fd < MAXFD; fd++) {
    struct rec *r = &recs[fd];
    if (r->kind != PIPE || r->inode == 0 || same_kind(fd, PIPE)) continue;
    for (int g = 0; g < MAXFD; g++) {
      if (recs[g].kind == PIPE && recs[g].inode == r->inode && fcntl(g, F_GETFD) != -1) {
        close(g);
      }
    }
  }
  for (int fd = 0; fd < MAXFD; fd++) {
    struct rec *r = &recs[fd];
    if (r->kind == NONE) continue;
    if (r->kind == LISTEN) {
      int on = 0;
      socklen_t len = sizeof on;
      if (getsockopt(fd, SOL_SOCKET, SO_ACCEPTCONN, &on, &len) == 0 && !on) {
        logf_("  fd %d was listening, restored without listen: listen -> %ld\n", fd, raw_listen(fd, 4096));
        made++;
      }
      continue;
    }
    if (fcntl(fd, F_GETFD) != -1) { // DMTCP restored it, or the slot is taken
      if (r->kind == EVENTFD || r->inode != 0) kept++;
      continue;
    }
    if (r->kind == EVENTFD) {
      int e = high(eventfd(r->count, 0));
      place(e, fd, r);
      close(e);
      made++;
      continue;
    }
    // Pipe: create once per inode, then place both ends wherever they lived.
    if (r->inode == 0) continue;
    unsigned long ino = r->inode;
    int p[2];
    // DMTCP's pipe() and syscall() wrappers return a socketpair; we need a real pipe.
    if (raw_pipe2(p, 0) != 0) continue;
    p[0] = high(p[0]);
    p[1] = high(p[1]);
    for (int g = fd; g < MAXFD; g++) {
      struct rec *q = &recs[g];
      if (q->kind != PIPE || q->inode != ino || fcntl(g, F_GETFD) != -1) continue;
      int rd = (q->flags & O_ACCMODE) == O_RDONLY;
      if (rd && q->len > 0 && write(p[1], q->data, q->len) < 0) {}
      place(rd ? p[0] : p[1], g, q);
      q->inode = 0; // done
      made++;
    }
    close(p[0]);
    close(p[1]);
  }
  logf_("pid %d restart: recreated %d fds, %d already open\n", getpid(), made, kept);
  for (int fd = 0; fd < MAXFD; fd++) {
    if (recs[fd].kind == PIPE && !same_kind(fd, PIPE)) logf_("  fd %d is not a pipe after fixup\n", fd);
  }
}

// Thread names. DMTCP restarts every thread as "DMTCP:<name>" cut to 15 bytes,
// and cuda-checkpoint finds the driver's restore thread by its name
// (cuda<digits>). Save the names before a checkpoint and put them back after.
#define MAXTHREADS 256
static char names[MAXTHREADS][16];
static int nnames;

static int readcomm(const char *tid, char *out)
{
  char path[64];
  snprintf(path, sizeof path, "/proc/self/task/%s/comm", tid);
  int fd = open(path, O_RDONLY | O_CLOEXEC);
  if (fd < 0) return -1;
  int n = read(fd, out, 15);
  close(fd);
  if (n <= 0) return -1;
  out[n] = 0;
  out[strcspn(out, "\n")] = 0;
  return 0;
}

static void savenames(void)
{
  nnames = 0;
  DIR *d = opendir("/proc/self/task");
  if (!d) return;
  struct dirent *e;
  while ((e = readdir(d)) && nnames < MAXTHREADS) {
    if (e->d_name[0] == '.') continue;
    if (readcomm(e->d_name, names[nnames]) == 0) nnames++;
  }
  closedir(d);
}

static void restorenames(void)
{
  char used[MAXTHREADS] = {0};
  int fixed = 0;
  DIR *d = opendir("/proc/self/task");
  if (!d) return;
  struct dirent *e;
  while ((e = readdir(d))) {
    char cur[16];
    if (e->d_name[0] == '.' || readcomm(e->d_name, cur) != 0) continue;
    if (strncmp(cur, "DMTCP:", 6) != 0) continue;
    const char *stem = cur + 6;
    for (int i = 0; i < nnames; i++) {
      if (used[i] || strncmp(names[i], stem, strlen(stem)) != 0) continue;
      used[i] = 1;
      char path[64];
      snprintf(path, sizeof path, "/proc/self/task/%s/comm", e->d_name);
      int fd = open(path, O_WRONLY | O_CLOEXEC);
      if (fd >= 0) {
        if (write(fd, names[i], strlen(names[i])) > 0) fixed++;
        close(fd);
      }
      break;
    }
  }
  closedir(d);
  logf_("pid %d restart: renamed %d of %d threads\n", getpid(), fixed, nnames);
}

// RESTART runs plugins in reverse order, so ours runs before DMTCP's file and
// socket plugins, which then dup their socketpairs over our pipes. THREAD_RESUME
// comes after them and before any thread returns to application code: the first
// thread there does the fixup, the rest wait for it.
static volatile int restarted, fixing, fixed;

static void hook(DmtcpEvent_t event, DmtcpEventData_t *data)
{
  (void)data;
  switch (event) {
  case DMTCP_EVENT_INIT:
  case DMTCP_EVENT_ATFORK_CHILD:
    // Yama ptrace_scope=1 lets only ancestors attach; with FDKEEP_PTRACER_ANY set, gdb can attach to a
    // process stuck in a checkpoint from any shell.
    if (getenv("FDKEEP_PTRACER_ANY")) prctl(PR_SET_PTRACER, PR_SET_PTRACER_ANY, 0, 0, 0);
    break;
  case DMTCP_EVENT_PRECHECKPOINT:
    record();
    savenames();
    break;
  case DMTCP_EVENT_RESTART:
    fixing = fixed = 0;
    __atomic_store_n(&restarted, 1, __ATOMIC_SEQ_CST);
    break;
  case DMTCP_EVENT_THREAD_RESUME:
    if (!__atomic_load_n(&restarted, __ATOMIC_SEQ_CST)) break;
    if (__atomic_exchange_n(&fixing, 1, __ATOMIC_SEQ_CST) == 0) {
      recreate();
      rewatch();
      restorenames();
      __atomic_store_n(&restarted, 0, __ATOMIC_SEQ_CST);
      __atomic_store_n(&fixed, 1, __ATOMIC_SEQ_CST);
    } else {
      while (!__atomic_load_n(&fixed, __ATOMIC_SEQ_CST)) usleep(100);
    }
    break;
  default:
    break;
  }
}

DmtcpPluginDescriptor_t fdkeep_plugin = {
  DMTCP_PLUGIN_API_VERSION,
  DMTCP_PACKAGE_VERSION,
  "fdkeep",
  "warmctl",
  "",
  "Recreate eventfds and pipes DMTCP did not track (CUDA driver helper threads)",
  hook
};

DMTCP_DECL_PLUGIN(fdkeep_plugin);
