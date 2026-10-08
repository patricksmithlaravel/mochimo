/**
 * @file _cuda_rt_emu.h
 * @brief Fake CUDA runtime for CPU tests of CUDA host code.
 * @details Include AFTER test/_cuda_emu.h (device code emulation). The
 * two headers together let a CPU test compile CUDA host code, such as
 * src/peach.cu, as plain C and run it without a GPU:
 * - device memory is host memory: every cudaMalloc() is its own shared
 *   anonymous mapping, filled with a garbage pattern and followed by a
 *   PROT_NONE guard page; allocations are tracked (leaks, double frees,
 *   frees of unknown pointers, copies outside an allocation);
 * - kernels launched with CUDA_KERNEL() run synchronously at launch, one
 *   emulated thread at a time (EMU_RT_KERNEL() may redirect a launch,
 *   e.g. to run it in worker processes with emu_rt_run_parallel());
 * - streams: all work executes immediately, but a stream stays busy
 *   (cudaStreamQuery() returns cudaErrorNotReady) for emu_rt.busy_ticks
 *   ticks after work was enqueued. A tick is one poll of the host loop
 *   under test: the test advances it with emu_rt_sleep() (or
 *   emu_rt_tick()). Device-to-host copies into pinned memory land when
 *   the stream completes, not at enqueue time. cudaStreamSynchronize(),
 *   cudaEventSynchronize(), cudaDeviceSynchronize(), cudaMemcpy() and
 *   cudaFree() complete work like the real runtime;
 * - events record the fake time (milliseconds, advanced by the test and
 *   by per-launch costs) at which their stream reaches them;
 * - errors: cudaGetLastError() / cudaPeekAtLastError() report the last
 *   error of ANY call (as the real runtime does), per host thread: a
 *   single threaded test selects the emulated host thread with
 *   emu_rt_thread(); sticky errors persist;
 * - fault injection: fail the Nth call of an API (emu_rt_fault(), e.g.
 *   "cudaMalloc", or "launch:<kernel>" for kernel launches), sticky
 *   device errors, streams that stay busy (emu_rt_stream_stuck()), and
 *   hooks after every kernel launch and every tick to corrupt device
 *   memory between polls.
 * <br />
 * Single threaded: call the runtime from one host thread only. Kernel
 * code run in worker processes (emu_rt_run_parallel()) may only write
 * device memory. Linux/POSIX only (mmap, fork).
 * @copyright Adequate Systems LLC, 2018-2025. All Rights Reserved.
 * <br />For license information, please refer to ../../LICENSE.md
*/

/* include guard */
#ifndef TEST_CUDA_RT_EMU_H
#define TEST_CUDA_RT_EMU_H


#ifndef TEST_CUDA_EMU_H
   #error "include test/_cuda_emu.h before test/_cuda_rt_emu.h"
#endif

#include <errno.h>      /* for errno, EINTR */
#include <stdarg.h>     /* for va_list */
#include <stdint.h>     /* for uint64_t, uintptr_t */
#include <stdio.h>      /* for printf() */
#include <stdlib.h>     /* for calloc(), abort() */
#include <string.h>     /* for memcpy(), strcmp() */
#include <time.h>       /* for time() */
#include <sys/mman.h>   /* for mmap(), mprotect(), munmap() */
#include <sys/types.h>  /* for pid_t */
#include <sys/wait.h>   /* for waitpid() */
#include <unistd.h>     /* for fork(), sysconf(), _exit() */

/* error codes (values as in the CUDA runtime; cudaSuccess: _cuda_emu.h,
 * whose cudaError_t enumerates cudaSuccess only) */
#define cudaErrorInvalidValue             ((cudaError_t) 1)
#define cudaErrorMemoryAllocation         ((cudaError_t) 2)
#define cudaErrorInitializationError      ((cudaError_t) 3)
#define cudaErrorInvalidConfiguration     ((cudaError_t) 9)
#define cudaErrorInvalidDevicePointer     ((cudaError_t) 17)
#define cudaErrorInvalidMemcpyDirection   ((cudaError_t) 21)
#define cudaErrorNoDevice                 ((cudaError_t) 100)
#define cudaErrorInvalidDevice            ((cudaError_t) 101)
#define cudaErrorInvalidResourceHandle    ((cudaError_t) 400)
#define cudaErrorNotReady                 ((cudaError_t) 600)
#define cudaErrorIllegalAddress           ((cudaError_t) 700)
#define cudaErrorLaunchOutOfResources     ((cudaError_t) 701)
#define cudaErrorLaunchTimeout            ((cudaError_t) 702)
#define cudaErrorLaunchFailure            ((cudaError_t) 719)
#define cudaErrorNotSupported             ((cudaError_t) 801)
#define cudaErrorUnknown                  ((cudaError_t) 999)

/* flags */
#define cudaStreamDefault           0x00
#define cudaStreamNonBlocking       0x01
#define cudaEventDefault            0x00
#define cudaEventBlockingSync       0x01
#define cudaEventDisableTiming      0x02
#define cudaEventInterprocess       0x04
#define cudaHostAllocDefault        0x00
#define cudaHostAllocPortable       0x01
#define cudaHostAllocMapped         0x02
#define cudaHostAllocWriteCombined  0x04
#define cudaDeviceScheduleAuto      0x00
#define cudaDeviceScheduleSpin      0x01
#define cudaDeviceScheduleYield     0x02
#define cudaDeviceScheduleBlockingSync 0x04
#define cudaDeviceMapHost           0x08
#define cudaDeviceLmemResizeToMax   0x10

/* enumerations (tag AND typedef name, so C and C++ spellings work) */
enum cudaMemcpyKind {
   cudaMemcpyHostToHost = 0, cudaMemcpyHostToDevice = 1,
   cudaMemcpyDeviceToHost = 2, cudaMemcpyDeviceToDevice = 3,
   cudaMemcpyDefault = 4
};
typedef enum cudaMemcpyKind cudaMemcpyKind;

enum cudaDeviceAttr {
   cudaDevAttrMaxThreadsPerBlock = 1, cudaDevAttrMaxBlockDimX = 2,
   cudaDevAttrMaxGridDimX = 5, cudaDevAttrMaxSharedMemoryPerBlock = 8,
   cudaDevAttrWarpSize = 10, cudaDevAttrMaxRegistersPerBlock = 12,
   cudaDevAttrClockRate = 13, cudaDevAttrMultiProcessorCount = 16,
   cudaDevAttrKernelExecTimeout = 17, cudaDevAttrIntegrated = 18,
   cudaDevAttrCanMapHostMemory = 19, cudaDevAttrComputeMode = 20,
   cudaDevAttrConcurrentKernels = 31, cudaDevAttrPciBusId = 33,
   cudaDevAttrPciDeviceId = 34, cudaDevAttrMemoryClockRate = 36,
   cudaDevAttrGlobalMemoryBusWidth = 37, cudaDevAttrL2CacheSize = 38,
   cudaDevAttrMaxThreadsPerMultiProcessor = 39,
   cudaDevAttrAsyncEngineCount = 40, cudaDevAttrUnifiedAddressing = 41,
   cudaDevAttrComputeCapabilityMajor = 75,
   cudaDevAttrComputeCapabilityMinor = 76,
   cudaDevAttrMaxSharedMemoryPerMultiprocessor = 81,
   cudaDevAttrMaxRegistersPerMultiprocessor = 82,
   cudaDevAttrMaxBlocksPerMultiprocessor = 106,
   cudaDevAttrMaxPersistingL2CacheSize = 108,
   cudaDevAttrMaxAccessPolicyWindowSize = 109
};
typedef enum cudaDeviceAttr cudaDeviceAttr;

enum cudaLimit {
   cudaLimitStackSize = 0x00, cudaLimitPrintfFifoSize = 0x01,
   cudaLimitMallocHeapSize = 0x02, cudaLimitDevRuntimeSyncDepth = 0x03,
   cudaLimitDevRuntimePendingLaunchCount = 0x04,
   cudaLimitMaxL2FetchGranularity = 0x05,
   cudaLimitPersistingL2CacheSize = 0x06
};
typedef enum cudaLimit cudaLimit;

enum cudaFuncCache {
   cudaFuncCachePreferNone = 0, cudaFuncCachePreferShared = 1,
   cudaFuncCachePreferL1 = 2, cudaFuncCachePreferEqual = 3
};
typedef enum cudaFuncCache cudaFuncCache;

enum cudaAccessProperty {
   cudaAccessPropertyNormal = 0, cudaAccessPropertyStreaming = 1,
   cudaAccessPropertyPersisting = 2
};
typedef enum cudaAccessProperty cudaAccessProperty;

enum cudaLaunchAttributeID {
   cudaLaunchAttributeIgnore = 0,
   cudaLaunchAttributeAccessPolicyWindow = 1,
   cudaLaunchAttributeCooperative = 2,
   cudaLaunchAttributeSynchronizationPolicy = 3,
   cudaLaunchAttributePriority = 8
};
typedef enum cudaLaunchAttributeID cudaLaunchAttributeID;
typedef enum cudaLaunchAttributeID cudaStreamAttrID;
#define cudaStreamAttributeAccessPolicyWindow \
   cudaLaunchAttributeAccessPolicyWindow
#define cudaStreamAttributeSynchronizationPolicy \
   cudaLaunchAttributeSynchronizationPolicy
#define cudaStreamAttributePriority  cudaLaunchAttributePriority

struct cudaAccessPolicyWindow {
   void *base_ptr;
   size_t num_bytes;
   float hitRatio;
   enum cudaAccessProperty hitProp;
   enum cudaAccessProperty missProp;
};
typedef struct cudaAccessPolicyWindow cudaAccessPolicyWindow;

union cudaLaunchAttributeValue {
   char pad[64];
   struct cudaAccessPolicyWindow accessPolicyWindow;
   int cooperative;
   int priority;
};
typedef union cudaLaunchAttributeValue cudaLaunchAttributeValue;
typedef union cudaLaunchAttributeValue cudaStreamAttrValue;

/* device properties: the commonly used subset of the real structure
 * (no warpSize: a macro of the device emulation, test/_cuda_emu.h) */
struct cudaDeviceProp {
   char name[256];
   size_t totalGlobalMem, sharedMemPerBlock, totalConstMem;
   size_t sharedMemPerMultiprocessor;
   int regsPerBlock, regsPerMultiprocessor, maxThreadsPerBlock;
   int maxThreadsDim[3], maxGridSize[3];
   int clockRate, memoryClockRate, memoryBusWidth;
   int major, minor, multiProcessorCount, maxThreadsPerMultiProcessor;
   int maxBlocksPerMultiProcessor, l2CacheSize, persistingL2CacheMaxSize;
   int accessPolicyMaxWindowSize, concurrentKernels, asyncEngineCount;
   int unifiedAddressing, canMapHostMemory, integrated;
   int pciBusID, pciDeviceID, pciDomainID;
};
typedef struct cudaDeviceProp cudaDeviceProp;

/* kernel attributes */
struct cudaFuncAttributes {
   size_t sharedSizeBytes, constSizeBytes, localSizeBytes;
   int maxThreadsPerBlock, numRegs, ptxVersion, binaryVersion;
   int cacheModeCA, maxDynamicSharedSizeBytes, preferredShmemCarveout;
};
typedef struct cudaFuncAttributes cudaFuncAttributes;

enum cudaFuncAttribute {
   cudaFuncAttributeMaxDynamicSharedMemorySize = 8,
   cudaFuncAttributePreferredSharedMemoryCarveout = 9
};
typedef enum cudaFuncAttribute cudaFuncAttribute;

/* emulator limits */
#define EMU_RT_MAXPROC     16    /**< max. worker processes */
#define EMU_RT_MAXKCFG     64    /**< max. kernel configurations */
#define EMU_RT_MAXFAULT    16    /**< max. active fault rules */
#define EMU_RT_MAXCOUNT    128   /**< max. counted API names */
#define EMU_RT_MAXTHREAD   4     /**< emulated host threads */
#define EMU_RT_SPIN_LIMIT  10000000UL  /**< queries within one tick that
   count as a host busy-wait loop (aborts: it would never end) */

/* allocation kinds */
#define EMU_RT_DEVMEM   1  /**< cudaMalloc() */
#define EMU_RT_HOSTMEM  2  /**< cudaMallocHost() / cudaHostAlloc() */

/** Fake device model (one model for every device). */
typedef struct {
   int count;              /**< number of devices (0: no device) */
   char name[256];         /**< device name */
   int sms;                /**< multiprocessors */
   int max_threads_sm;     /**< max. resident threads per SM */
   int max_threads_block;  /**< max. threads per block */
   int max_blocks_sm;      /**< max. resident blocks per SM */
   int cc_major, cc_minor; /**< compute capability */
   int l2_bytes;           /**< L2 cache size */
   int persist_l2_max;     /**< max. persisting L2 size */
   int max_window;         /**< max. access policy window */
   int regs_sm;            /**< registers per SM */
   int smem_sm;            /**< shared memory per SM */
   int smem_block;         /**< shared memory per block */
   int clock_khz;          /**< clock rate */
   size_t total_mem;       /**< global memory */
} EMU_RT_DEVICE;

/** Per kernel configuration and statistics (name may end in '*'). */
typedef struct {
   char name[64];          /**< kernel name, or prefix followed by '*' */
   int min_grid;           /**< occupancy: min. grid for full device
                              (0: SMs x active_sm) */
   int block;              /**< occupancy: suggested block size */
   int active_sm;          /**< occupancy: active blocks per SM */
   int max_block;          /**< launch: max. threads per block (as a
                              __launch_bounds__), 0: device limit */
   double ms;              /**< fake run time per launch (< 0: auto) */
   unsigned long launches; /**< successful launches */
   unsigned long failed;   /**< failed launches */
} EMU_RT_KCFG;

/** Fault rule: fail the Nth call of an API. */
typedef struct {
   char api[80];           /**< API name, or "launch:<kernel>" */
   unsigned long countdown;   /**< calls until the failure (1 = next) */
   cudaError_t err;        /**< error to return */
   int repeat;             /**< keep failing every later call */
   int active;             /**< rule armed */
   unsigned long hits;     /**< injected failures */
} EMU_RT_FAULT;

/** Call counter. */
typedef struct {
   char api[80];
   unsigned long calls;
} EMU_RT_COUNT;

/** Tracked allocation. */
typedef struct {
   void *ptr;              /**< pointer returned to the caller */
   size_t size;            /**< requested size */
   void *base;             /**< mapping base */
   size_t maplen;          /**< mapping length */
   int kind;               /**< EMU_RT_DEVMEM or EMU_RT_HOSTMEM */
   int live;               /**< not freed yet */
   unsigned long seq;      /**< allocation number (1, 2, ...) */
   unsigned long call;     /**< call number of its API (emu_rt_calls()) */
   const char *api;        /**< allocating API */
   const char *tag;        /**< caller's pointer expression */
} EMU_RT_ALLOC;

/** Deferred host write (device-to-host copy into pinned memory). */
typedef struct emu_rt__pending {
   unsigned long seq;      /**< operation number */
   void *dst;              /**< host destination */
   void *data;             /**< copied device data */
   size_t n;               /**< bytes */
   struct emu_rt__pending *next;
} EMU_RT__PENDING;

/** Stream. */
typedef struct emu_rt__stream {
   int live;               /**< not destroyed */
   int nonblocking;        /**< cudaStreamNonBlocking */
   int stuck;              /**< fault: never completes */
   int id;                 /**< creation number (0: legacy stream) */
   unsigned long done_tick;   /**< busy while tick < done_tick */
   unsigned long last_seq;    /**< last enqueued operation */
   unsigned long done_seq;    /**< operations completed up to here */
   double t_ready;         /**< fake time when the enqueued work ends */
   EMU_RT__PENDING *head, *tail;  /**< deferred host writes */
   cudaStreamAttrValue window;   /**< access policy window */
} EMU_RT__STREAM;

/** Event (cudaEvent_t). */
struct emu_rt__event {
   int live;               /**< not destroyed */
   int recorded;           /**< recorded at least once */
   int timing;             /**< timing enabled */
   EMU_RT__STREAM *s;      /**< stream of the last record */
   unsigned long done_tick;   /**< tick when the record point completes */
   unsigned long seq;      /**< stream operation at the record */
   double t;               /**< fake time of the record point */
};
typedef struct emu_rt__event *cudaEvent_t;

/** Current kernel launch. */
typedef struct {
   const char *name;       /**< kernel name */
   EMU_RT__STREAM *s;      /**< stream */
   EMU_RT_KCFG *cfg;       /**< kernel configuration */
   unsigned long serial;   /**< launch number */
   int active;             /**< inside CUDA_KERNEL() */
   int ok;                 /**< launch accepted */
   int started;            /**< first thread started */
   int done;               /**< all threads handled (emu_rt_kernel_done) */
} EMU_RT__LAUNCH;

/** Fake runtime state. */
typedef struct {
   int initialized;
   /* configuration (set by the test, see emu_rt_init()) */
   EMU_RT_DEVICE dev;      /**< device model */
   unsigned long busy_ticks;  /**< ticks a stream stays busy */
   int nproc;              /**< worker processes, emu_rt_run_parallel() */
   int garbage;            /**< fill new memory with garbage */
   int verbose;            /**< log injected faults and kernel launches */
   /* clock */
   double now_ms;          /**< fake time, milliseconds */
   time_t base_time;       /**< real time at emu_rt_init() */
   unsigned long tick;     /**< host polls */
   unsigned long spin_tick, spins;  /**< busy-wait detection */
   /* runtime state */
   int device;             /**< current device */
   unsigned int device_flags;
   size_t limit_persist_l2;
   cudaError_t last_error[EMU_RT_MAXTHREAD];   /**< cudaGetLastError(),
                              per emulated host thread */
   int thread;             /**< current emulated host thread */
   cudaError_t sticky;     /**< sticky (device) error */
   unsigned long seq;      /**< operation numbers */
   EMU_RT_ALLOC *alloc;    /**< allocations (incl. freed ones) */
   size_t nalloc, maxalloc;
   unsigned long alloc_seq;
   size_t live_dev_bytes;
   EMU_RT__STREAM **streams;  /**< created streams (incl. destroyed) */
   size_t nstreams, maxstreams;
   EMU_RT__STREAM null_stream;   /**< legacy default stream (0) */
   struct emu_rt__event **events;   /**< created events */
   size_t nevents, maxevents;
   EMU_RT_KCFG kcfg[EMU_RT_MAXKCFG];
   int nkcfg;
   EMU_RT_FAULT fault[EMU_RT_MAXFAULT];
   int nfault;
   EMU_RT_COUNT count[EMU_RT_MAXCOUNT];
   int ncount;
   EMU_RT__LAUNCH launch;  /**< current launch */
   /* statistics */
   unsigned long misuse;   /**< API misuse (see emu_rt__misuse()) */
   unsigned long hangs;    /**< synchronizations a real GPU never ends */
   unsigned long crashes;  /**< kernels that crashed a worker process */
   unsigned long launches; /**< successful kernel launches */
   unsigned long launch_fail;    /**< failed kernel launches */
   /* hooks */
   void (*post_launch)(const char *name, cudaStream_t s, void *user);
   void *post_launch_user;
   void (*on_tick)(unsigned long tick, void *user);
   void *on_tick_user;
} EMU_RT;

static EMU_RT emu_rt;

/**
 * @private
 * Report (and count) a misuse of the runtime API by the code under test.
*/
static inline void emu_rt__misuse(const char *fmt, ...)
   __attribute__((format(printf, 1, 2)));
static inline void emu_rt__misuse(const char *fmt, ...)
{
   va_list ap;

   emu_rt.misuse++;
   printf("EMU MISUSE: ");
   va_start(ap, fmt);
   vprintf(fmt, ap);
   va_end(ap);
   printf("\n");
   fflush(stdout);
}  /* end emu_rt__misuse() */

/**
 * Initialize (or reset the configuration of) the fake runtime with a
 * small default device model. Existing allocations, streams and events
 * are kept. Called automatically by the first API call.
*/
static inline void emu_rt_init(void)
{
   long n;

   memset(&emu_rt.dev, 0, sizeof(emu_rt.dev));
   emu_rt.dev.count = 1;
   snprintf(emu_rt.dev.name, sizeof(emu_rt.dev.name),
      "Emulated CUDA device");
   emu_rt.dev.sms = 2;
   emu_rt.dev.max_threads_sm = 64;
   emu_rt.dev.max_threads_block = 1024;
   emu_rt.dev.max_blocks_sm = 16;
   emu_rt.dev.cc_major = 8;
   emu_rt.dev.cc_minor = 6;
   emu_rt.dev.l2_bytes = 4 << 20;
   emu_rt.dev.persist_l2_max = 3 << 20;
   emu_rt.dev.max_window = 128 << 20;
   emu_rt.dev.regs_sm = 65536;
   emu_rt.dev.smem_sm = 100 << 10;
   emu_rt.dev.smem_block = 48 << 10;
   emu_rt.dev.clock_khz = 1700000;
   emu_rt.dev.total_mem = (size_t) 8 << 30;
   emu_rt.busy_ticks = 2;
   n = sysconf(_SC_NPROCESSORS_ONLN);
   emu_rt.nproc = n < 1 ? 1 : n > 4 ? 4 : (int) n;
   emu_rt.garbage = 1;
   if (!emu_rt.initialized) {
      emu_rt.base_time = time(NULL);
      emu_rt.null_stream.live = 1;
      emu_rt.initialized = 1;
   }
}  /* end emu_rt_init() */

/**
 * @private
 * Count a call of @a api.
*/
static inline void emu_rt__count(const char *api)
{
   int i;

   for (i = 0; i < emu_rt.ncount; i++) {
      if (strcmp(emu_rt.count[i].api, api) == 0) {
         emu_rt.count[i].calls++;
         return;
      }
   }
   if (emu_rt.ncount < EMU_RT_MAXCOUNT) {
      snprintf(emu_rt.count[i].api, sizeof(emu_rt.count[i].api), "%s", api);
      emu_rt.count[i].calls = 1;
      emu_rt.ncount++;
   }
}  /* end emu_rt__count() */

/**
 * Number of calls of @a api so far (e.g. "cudaMalloc", or
 * "launch:<kernel>" for successful and failed launch attempts).
*/
static inline unsigned long emu_rt_calls(const char *api)
{
   int i;

   for (i = 0; i < emu_rt.ncount; i++) {
      if (strcmp(emu_rt.count[i].api, api) == 0) return emu_rt.count[i].calls;
   }
   return 0;
}  /* end emu_rt_calls() */

/**
 * @private
 * Apply the fault rules to a call of @a api.
 * @returns the injected error, or cudaSuccess
*/
static inline cudaError_t emu_rt__fault(const char *api)
{
   EMU_RT_FAULT *f;
   int i;

   for (i = 0; i < emu_rt.nfault; i++) {
      f = &emu_rt.fault[i];
      if (!f->active || strcmp(f->api, api) != 0) continue;
      if (f->countdown > 1) {
         f->countdown--;
         continue;
      }
      if (!f->repeat) f->active = 0;
      f->hits++;
      if (emu_rt.verbose) {
         printf("EMU: injected failure of %s: error %d\n", api, (int) f->err);
      }
      return f->err;
   }
   return cudaSuccess;
}  /* end emu_rt__fault() */

/**
 * @private
 * Common entry of every API function: count the call, then report a
 * sticky error or an injected failure.
*/
static inline cudaError_t emu_rt__enter(const char *api)
{
   if (!emu_rt.initialized) emu_rt_init();
   emu_rt__count(api);
   if (emu_rt.sticky != cudaSuccess) return emu_rt.sticky;
   return emu_rt__fault(api);
}  /* end emu_rt__enter() */

/**
 * @private
 * Common return of every API function: an error (but not
 * cudaErrorNotReady) becomes the last error, as in the real runtime.
*/
static inline cudaError_t emu_rt__ret(cudaError_t err)
{
   if (err != cudaSuccess && err != cudaErrorNotReady) {
      emu_rt.last_error[emu_rt.thread] = err;
   }
   return err;
}  /* end emu_rt__ret() */

/**
 * Fail the @a nth call (1 = the next call) of @a api from now on with
 * @a err; with @a repeat, also every later call (until cleared). Kernel
 * launches: api "launch:<kernel name>" or "launch:*" (any kernel).
 * @returns 0 on success, -1 if too many rules are armed
*/
static inline int emu_rt_fault(const char *api, unsigned long nth,
   cudaError_t err, int repeat)
{
   EMU_RT_FAULT *f;
   int i;

   if (!emu_rt.initialized) emu_rt_init();
   for (i = 0; i < emu_rt.nfault && emu_rt.fault[i].active; i++);
   if (i >= EMU_RT_MAXFAULT) return (-1);
   f = &emu_rt.fault[i];
   memset(f, 0, sizeof(*f));
   snprintf(f->api, sizeof(f->api), "%s", api);
   f->countdown = nth ? nth : 1;
   f->err = err;
   f->repeat = repeat;
   f->active = 1;
   if (i == emu_rt.nfault) emu_rt.nfault++;
   return 0;
}  /* end emu_rt_fault() */

/**
 * Disarm every fault rule.
 * @returns the number of failures injected by the disarmed rules
*/
static inline unsigned long emu_rt_fault_clear(void)
{
   unsigned long hits = 0;
   int i;

   for (i = 0; i < emu_rt.nfault; i++) hits += emu_rt.fault[i].hits;
   emu_rt.nfault = 0;
   return hits;
}  /* end emu_rt_fault_clear() */

/**
 * Set (or clear, with cudaSuccess) a sticky device error: from now on
 * every call fails with @a err, as after a kernel fault on a real GPU.
*/
static inline void emu_rt_sticky(cudaError_t err)
{
   int i;

   if (!emu_rt.initialized) emu_rt_init();
   emu_rt.sticky = err;
   if (err != cudaSuccess) {
      for (i = 0; i < EMU_RT_MAXTHREAD; i++) emu_rt.last_error[i] = err;
   }
}  /* end emu_rt_sticky() */

/**
 * Select the emulated host thread @a t (0 .. EMU_RT_MAXTHREAD - 1) of the
 * following calls: the real runtime keeps the last error per host
 * thread (e.g. gpuminer initializes devices in its main thread and
 * solves in a device thread).
*/
static inline void emu_rt_thread(int t)
{
   if (!emu_rt.initialized) emu_rt_init();
   emu_rt.thread = t < 0 ? 0 : t >= EMU_RT_MAXTHREAD ? EMU_RT_MAXTHREAD - 1 : t;
}  /* end emu_rt_thread() */

/**
 * Find (or create) the configuration of kernel @a name. A configuration
 * named "<prefix>*" applies to every kernel starting with <prefix>.
 * @param create create an exact entry when no entry matches exactly
 * @returns the configuration, or NULL (none, or table full)
*/
static inline EMU_RT_KCFG *emu_rt_kcfg(const char *name, int create)
{
   EMU_RT_KCFG *k, *wild = NULL;
   size_t len;
   int i;

   if (!emu_rt.initialized) emu_rt_init();
   for (i = 0; i < emu_rt.nkcfg; i++) {
      k = &emu_rt.kcfg[i];
      if (strcmp(k->name, name) == 0) return k;
      len = strlen(k->name);
      if (wild == NULL && len > 0 && k->name[len - 1] == '*' &&
            strncmp(k->name, name, len - 1) == 0) wild = k;
   }
   if (!create || emu_rt.nkcfg >= EMU_RT_MAXKCFG) return wild;
   k = &emu_rt.kcfg[emu_rt.nkcfg++];
   if (wild) *k = *wild;
   else {
      memset(k, 0, sizeof(*k));
      k->block = 128;
      k->active_sm = 4;
      k->ms = -1.0;
   }
   snprintf(k->name, sizeof(k->name), "%s", name);
   k->launches = k->failed = 0;
   return k;
}  /* end emu_rt_kcfg() */

/**
 * Fake time, as time(): seconds since the epoch, advanced by the test
 * (emu_rt_sleep(), emu_rt_advance()) and by synchronizations.
*/
static inline time_t emu_rt_time(time_t *t)
{
   time_t now;

   if (!emu_rt.initialized) emu_rt_init();
   now = emu_rt.base_time + (time_t) (emu_rt.now_ms / 1000.0);
   if (t) *t = now;
   return now;
}  /* end emu_rt_time() */

/**
 * @private
 * Is stream @a s busy (its enqueued work not complete)?
*/
static inline int emu_rt__busy(const EMU_RT__STREAM *s)
{
   return s->stuck || emu_rt.tick < s->done_tick;
}  /* end emu_rt__busy() */

/**
 * @private
 * Find the live allocation of @a kind (0: any) containing @a p.
*/
static inline EMU_RT_ALLOC *emu_rt__owner(const void *p, int kind)
{
   uintptr_t a = (uintptr_t) p, b;
   size_t i;

   for (i = 0; i < emu_rt.nalloc; i++) {
      if (!emu_rt.alloc[i].live) continue;
      if (kind && emu_rt.alloc[i].kind != kind) continue;
      b = (uintptr_t) emu_rt.alloc[i].ptr;
      if (a >= b && a < b + emu_rt.alloc[i].size) return &emu_rt.alloc[i];
   }
   return NULL;
}  /* end emu_rt__owner() */

/**
 * @private
 * Complete the work of stream @a s up to operation @a upto: apply its
 * deferred host writes (into live pinned memory only).
*/
static inline void emu_rt__flush(EMU_RT__STREAM *s, unsigned long upto)
{
   EMU_RT__PENDING *p;
   EMU_RT_ALLOC *a;
   uintptr_t end;

   while ((p = s->head) != NULL && p->seq <= upto) {
      s->head = p->next;
      if (s->head == NULL) s->tail = NULL;
      a = emu_rt__owner(p->dst, EMU_RT_HOSTMEM);
      end = a ? (uintptr_t) a->ptr + a->size : 0;
      if (a == NULL || (uintptr_t) p->dst + p->n > end) {
         emu_rt__misuse("device-to-host copy of %lu bytes completed into"
            " host memory %p that is no longer allocated",
            (unsigned long) p->n, p->dst);
      } else memcpy(p->dst, p->data, p->n);
      free(p->data);
      free(p);
   }
   if (upto > s->last_seq) upto = s->last_seq;
   if (s->done_seq < upto) s->done_seq = upto;
}  /* end emu_rt__flush() */

/**
 * @private
 * Complete ALL work of stream @a s now (a synchronization), advancing
 * the fake time to its end. A stuck stream would never complete: the
 * hang is counted, the work is completed regardless.
*/
static inline void emu_rt__complete(EMU_RT__STREAM *s, const char *api)
{
   if (s->stuck && s->done_seq < s->last_seq) {
      emu_rt.hangs++;
      printf("EMU: %s() waits for a stuck stream: a real GPU would hang\n",
         api);
      fflush(stdout);
   }
   emu_rt__flush(s, ~0UL);
   if (s->done_tick > emu_rt.tick) s->done_tick = emu_rt.tick;
   if (s->t_ready > emu_rt.now_ms) emu_rt.now_ms = s->t_ready;
}  /* end emu_rt__complete() */

/**
 * @private
 * Complete all streams (incl. destroyed streams with work left), or only
 * the legacy stream and the blocking streams when @a blocking_only.
*/
static inline void emu_rt__complete_all(int blocking_only, const char *api)
{
   size_t i;

   emu_rt__complete(&emu_rt.null_stream, api);
   for (i = 0; i < emu_rt.nstreams; i++) {
      if (blocking_only && emu_rt.streams[i]->nonblocking) continue;
      emu_rt__complete(emu_rt.streams[i], api);
   }
}  /* end emu_rt__complete_all() */

/**
 * @private
 * Enqueue an operation of @a cost_ms fake time on stream @a s: the
 * stream is busy for emu_rt.busy_ticks ticks (and after the work it
 * depends on: the legacy stream waits for all blocking streams, and
 * blocking streams wait for the legacy stream).
 * @returns the operation's sequence number
*/
static inline unsigned long emu_rt__enqueue(EMU_RT__STREAM *s,
   double cost_ms)
{
   EMU_RT__STREAM *o;
   unsigned long dt = emu_rt.tick + emu_rt.busy_ticks;
   double t0 = emu_rt.now_ms;
   size_t i;

   if (s == &emu_rt.null_stream) {
      for (i = 0; i < emu_rt.nstreams; i++) {
         o = emu_rt.streams[i];
         if (o->nonblocking || !emu_rt__busy(o)) continue;
         if (o->done_tick > dt) dt = o->done_tick;
         if (o->t_ready > t0) t0 = o->t_ready;
      }
   } else if (!s->nonblocking && emu_rt__busy(&emu_rt.null_stream)) {
      o = &emu_rt.null_stream;
      if (o->done_tick > dt) dt = o->done_tick;
      if (o->t_ready > t0) t0 = o->t_ready;
   }
   if (s->done_tick < dt) s->done_tick = dt;
   if (s->t_ready > t0) t0 = s->t_ready;
   s->t_ready = t0 + cost_ms;
   s->last_seq = ++emu_rt.seq;
   return s->last_seq;
}  /* end emu_rt__enqueue() */

/**
 * @private
 * Resolve a stream handle (0 = legacy default stream).
 * @returns cudaSuccess, or cudaErrorInvalidResourceHandle (misuse)
*/
static inline cudaError_t emu_rt__stream(cudaStream_t h, const char *api,
   EMU_RT__STREAM **sp)
{
   size_t i;

   if (h == NULL) {
      *sp = &emu_rt.null_stream;
      return cudaSuccess;
   }
   for (i = 0; i < emu_rt.nstreams; i++) {
      if ((void *) emu_rt.streams[i] != h) continue;
      if (!emu_rt.streams[i]->live) {
         emu_rt__misuse("%s() on a destroyed stream", api);
         return cudaErrorInvalidResourceHandle;
      }
      *sp = emu_rt.streams[i];
      return cudaSuccess;
   }
   emu_rt__misuse("%s() on an unknown stream %p", api, h);
   return cudaErrorInvalidResourceHandle;
}  /* end emu_rt__stream() */

/**
 * @private
 * Resolve an event handle.
*/
static inline cudaError_t emu_rt__event(cudaEvent_t e, const char *api)
{
   size_t i;

   for (i = 0; i < emu_rt.nevents; i++) {
      if (emu_rt.events[i] != e) continue;
      if (!e->live) {
         emu_rt__misuse("%s() on a destroyed event", api);
         return cudaErrorInvalidResourceHandle;
      }
      return cudaSuccess;
   }
   emu_rt__misuse("%s() on an unknown event %p", api, (void *) e);
   return cudaErrorInvalidResourceHandle;
}  /* end emu_rt__event() */

/**
 * Advance the host poll counter by one tick: streams whose busy time has
 * passed complete (their deferred host writes land), then the tick hook
 * runs (e.g. to corrupt device memory between polls).
*/
static inline void emu_rt_tick(void)
{
   size_t i;

   if (!emu_rt.initialized) emu_rt_init();
   emu_rt.tick++;
   if (!emu_rt__busy(&emu_rt.null_stream)) {
      emu_rt__flush(&emu_rt.null_stream, ~0UL);
   }
   for (i = 0; i < emu_rt.nstreams; i++) {
      if (!emu_rt__busy(emu_rt.streams[i])) {
         emu_rt__flush(emu_rt.streams[i], ~0UL);
      }
   }
   if (emu_rt.on_tick) emu_rt.on_tick(emu_rt.tick, emu_rt.on_tick_user);
}  /* end emu_rt_tick() */

/**
 * Advance the fake time by @a ms milliseconds, without a tick.
*/
static inline void emu_rt_advance(double ms)
{
   if (!emu_rt.initialized) emu_rt_init();
   emu_rt.now_ms += ms;
}  /* end emu_rt_advance() */

/**
 * Sleep between two host polls (replaces millisleep()): advance the fake
 * time by @a ms milliseconds and the poll counter by one tick.
*/
static inline void emu_rt_sleep(double ms)
{
   emu_rt_advance(ms);
   emu_rt_tick();
}  /* end emu_rt_sleep() */

/**
 * Make stream @a h stay busy (@a stuck = 1) until released (0). Work
 * enqueued meanwhile also waits; synchronizing with it counts a hang.
 * @returns 0, or -1 for an unknown stream
*/
static inline int emu_rt_stream_stuck(cudaStream_t h, int stuck)
{
   size_t i;

   if (h == NULL) {
      emu_rt.null_stream.stuck = stuck;
      return 0;
   }
   for (i = 0; i < emu_rt.nstreams; i++) {
      if ((void *) emu_rt.streams[i] == h) {
         emu_rt.streams[i]->stuck = stuck;
         return 0;
      }
   }
   return (-1);
}  /* end emu_rt_stream_stuck() */

/**
 * Is stream @a h busy (as the next cudaStreamQuery() would report)?
 * @returns 1 if busy, 0 if idle, -1 for an unknown stream
*/
static inline int emu_rt_stream_busy(cudaStream_t h)
{
   size_t i;

   if (h == NULL) return emu_rt__busy(&emu_rt.null_stream);
   for (i = 0; i < emu_rt.nstreams; i++) {
      if ((void *) emu_rt.streams[i] == h) {
         return emu_rt__busy(emu_rt.streams[i]);
      }
   }
   return (-1);
}  /* end emu_rt_stream_busy() */

/**
 * Number of live allocations of @a kind (0: any); total bytes in
 * @a bytes (if not NULL). With @a print, list them.
*/
static inline unsigned long emu_rt_live(int kind, size_t *bytes, int print)
{
   unsigned long n = 0;
   size_t i, sum = 0;

   for (i = 0; i < emu_rt.nalloc; i++) {
      if (!emu_rt.alloc[i].live) continue;
      if (kind && emu_rt.alloc[i].kind != kind) continue;
      n++;
      sum += emu_rt.alloc[i].size;
      if (print) {
         printf("   live %s #%lu: %lu bytes at %p (%s)\n",
            emu_rt.alloc[i].kind == EMU_RT_DEVMEM ? "device" : "pinned",
            emu_rt.alloc[i].seq, (unsigned long) emu_rt.alloc[i].size,
            emu_rt.alloc[i].ptr, emu_rt.alloc[i].tag);
      }
   }
   if (bytes) *bytes = sum;
   return n;
}  /* end emu_rt_live() */

/**
 * Number of live (not destroyed) streams and events.
*/
static inline unsigned long emu_rt_live_handles(void)
{
   unsigned long n = 0;
   size_t i;

   for (i = 0; i < emu_rt.nstreams; i++) n += emu_rt.streams[i]->live;
   for (i = 0; i < emu_rt.nevents; i++) n += emu_rt.events[i]->live;
   return n;
}  /* end emu_rt_live_handles() */

/**
 * Find the live allocation containing @a p (any kind), e.g. to classify
 * the buffers of the code under test.
 * @returns the allocation record (do not modify), or NULL
*/
static inline const EMU_RT_ALLOC *emu_rt_find(const void *p)
{
   return emu_rt__owner(p, 0);
}  /* end emu_rt_find() */

/**
 * Install the hook called after every successful kernel launch (after
 * its threads ran, before the next host call), or NULL.
*/
static inline void emu_rt_on_launch(void (*fn)(const char *name,
   cudaStream_t s, void *user), void *user)
{
   emu_rt.post_launch = fn;
   emu_rt.post_launch_user = user;
}  /* end emu_rt_on_launch() */

/**
 * Install the hook called at the end of every tick, or NULL.
*/
static inline void emu_rt_on_tick(void (*fn)(unsigned long tick,
   void *user), void *user)
{
   emu_rt.on_tick = fn;
   emu_rt.on_tick_user = user;
}  /* end emu_rt_on_tick() */

/****************************************************************
 * ERRORS AND DEVICE MANAGEMENT
 ****************************************************************/

/**
 * Error name of @a err.
*/
static inline const char *cudaGetErrorName(cudaError_t err)
{
   switch ((int) err) {
      case 0: return "cudaSuccess";
      case 1: return "cudaErrorInvalidValue";
      case 2: return "cudaErrorMemoryAllocation";
      case 3: return "cudaErrorInitializationError";
      case 9: return "cudaErrorInvalidConfiguration";
      case 17: return "cudaErrorInvalidDevicePointer";
      case 21: return "cudaErrorInvalidMemcpyDirection";
      case 100: return "cudaErrorNoDevice";
      case 101: return "cudaErrorInvalidDevice";
      case 400: return "cudaErrorInvalidResourceHandle";
      case 600: return "cudaErrorNotReady";
      case 700: return "cudaErrorIllegalAddress";
      case 701: return "cudaErrorLaunchOutOfResources";
      case 702: return "cudaErrorLaunchTimeout";
      case 719: return "cudaErrorLaunchFailure";
      case 801: return "cudaErrorNotSupported";
      default: return "cudaErrorUnknown";
   }
}  /* end cudaGetErrorName() */

/**
 * Error description of @a err.
*/
static inline const char *cudaGetErrorString(cudaError_t err)
{
   switch ((int) err) {
      case 0: return "no error";
      case 1: return "invalid argument";
      case 2: return "out of memory";
      case 3: return "initialization error";
      case 9: return "invalid configuration argument";
      case 17: return "invalid device pointer";
      case 21: return "invalid copy direction for memcpy";
      case 100: return "no CUDA-capable device is detected";
      case 101: return "invalid device ordinal";
      case 400: return "invalid resource handle";
      case 600: return "device not ready";
      case 700: return "an illegal memory access was encountered";
      case 701: return "too many resources requested for launch";
      case 702: return "the launch timed out and was terminated";
      case 719: return "unspecified launch failure";
      case 801: return "operation not supported";
      default: return "unknown error";
   }
}  /* end cudaGetErrorString() */

/* _cuda_emu.h defines a cudaGetLastError() that always succeeds */
#define cudaGetLastError()    emu_rt_get_last_error()

/**
 * cudaGetLastError(): last error of any call, then reset (a sticky error
 * is never reset).
*/
static inline cudaError_t emu_rt_get_last_error(void)
{
   cudaError_t err;

   if (!emu_rt.initialized) emu_rt_init();
   emu_rt__count("cudaGetLastError");
   err = emu_rt.last_error[emu_rt.thread];
   emu_rt.last_error[emu_rt.thread] = emu_rt.sticky;
   return err;
}  /* end emu_rt_get_last_error() */

/**
 * Last error of any call, not reset.
*/
static inline cudaError_t cudaPeekAtLastError(void)
{
   if (!emu_rt.initialized) emu_rt_init();
   emu_rt__count("cudaPeekAtLastError");
   return emu_rt.last_error[emu_rt.thread];
}  /* end cudaPeekAtLastError() */

static inline cudaError_t cudaGetDeviceCount(int *count)
{
   cudaError_t err = emu_rt__enter("cudaGetDeviceCount");

   if (count == NULL) return emu_rt__ret(cudaErrorInvalidValue);
   if (err == cudaSuccess && emu_rt.dev.count < 1) err = cudaErrorNoDevice;
   *count = err == cudaSuccess ? emu_rt.dev.count : 0;
   return emu_rt__ret(err);
}  /* end cudaGetDeviceCount() */

static inline cudaError_t cudaSetDevice(int device)
{
   cudaError_t err = emu_rt__enter("cudaSetDevice");

   if (err == cudaSuccess && (device < 0 || device >= emu_rt.dev.count)) {
      err = cudaErrorInvalidDevice;
   }
   if (err == cudaSuccess) emu_rt.device = device;
   return emu_rt__ret(err);
}  /* end cudaSetDevice() */

static inline cudaError_t cudaGetDevice(int *device)
{
   cudaError_t err = emu_rt__enter("cudaGetDevice");

   if (device == NULL) return emu_rt__ret(cudaErrorInvalidValue);
   if (err == cudaSuccess) *device = emu_rt.device;
   return emu_rt__ret(err);
}  /* end cudaGetDevice() */

static inline cudaError_t cudaSetDeviceFlags(unsigned int flags)
{
   cudaError_t err = emu_rt__enter("cudaSetDeviceFlags");

   if (err == cudaSuccess) emu_rt.device_flags = flags;
   return emu_rt__ret(err);
}  /* end cudaSetDeviceFlags() */

static inline cudaError_t cudaGetDeviceFlags(unsigned int *flags)
{
   cudaError_t err = emu_rt__enter("cudaGetDeviceFlags");

   if (flags == NULL) return emu_rt__ret(cudaErrorInvalidValue);
   if (err == cudaSuccess) *flags = emu_rt.device_flags;
   return emu_rt__ret(err);
}  /* end cudaGetDeviceFlags() */

static inline cudaError_t cudaDeviceGetAttribute(int *value,
   enum cudaDeviceAttr attr, int device)
{
   const EMU_RT_DEVICE *d = &emu_rt.dev;
   cudaError_t err = emu_rt__enter("cudaDeviceGetAttribute");
   int v;

   if (value == NULL) return emu_rt__ret(cudaErrorInvalidValue);
   if (err != cudaSuccess) return emu_rt__ret(err);
   if (device < 0 || device >= d->count) {
      return emu_rt__ret(cudaErrorInvalidDevice);
   }
   switch (attr) {
      case cudaDevAttrMaxThreadsPerBlock: v = d->max_threads_block; break;
      case cudaDevAttrMaxBlockDimX: v = d->max_threads_block; break;
      case cudaDevAttrMaxGridDimX: v = 2147483647; break;
      case cudaDevAttrMaxSharedMemoryPerBlock: v = d->smem_block; break;
      case cudaDevAttrWarpSize: v = 32; break;
      case cudaDevAttrMaxRegistersPerBlock: v = d->regs_sm; break;
      case cudaDevAttrClockRate: v = d->clock_khz; break;
      case cudaDevAttrMultiProcessorCount: v = d->sms; break;
      case cudaDevAttrKernelExecTimeout: v = 0; break;
      case cudaDevAttrIntegrated: v = 0; break;
      case cudaDevAttrCanMapHostMemory: v = 1; break;
      case cudaDevAttrComputeMode: v = 0; break;
      case cudaDevAttrConcurrentKernels: v = 1; break;
      case cudaDevAttrPciBusId: v = device + 1; break;
      case cudaDevAttrPciDeviceId: v = 0; break;
      case cudaDevAttrMemoryClockRate: v = 9500000; break;
      case cudaDevAttrGlobalMemoryBusWidth: v = 256; break;
      case cudaDevAttrL2CacheSize: v = d->l2_bytes; break;
      case cudaDevAttrMaxThreadsPerMultiProcessor:
         v = d->max_threads_sm;
         break;
      case cudaDevAttrAsyncEngineCount: v = 2; break;
      case cudaDevAttrUnifiedAddressing: v = 1; break;
      case cudaDevAttrComputeCapabilityMajor: v = d->cc_major; break;
      case cudaDevAttrComputeCapabilityMinor: v = d->cc_minor; break;
      case cudaDevAttrMaxSharedMemoryPerMultiprocessor: v = d->smem_sm; break;
      case cudaDevAttrMaxRegistersPerMultiprocessor: v = d->regs_sm; break;
      case cudaDevAttrMaxBlocksPerMultiprocessor: v = d->max_blocks_sm; break;
      case cudaDevAttrMaxPersistingL2CacheSize: v = d->persist_l2_max; break;
      case cudaDevAttrMaxAccessPolicyWindowSize: v = d->max_window; break;
      default:
         printf("EMU: cudaDeviceGetAttribute(): attribute %d is not"
            " emulated (add it to test/_cuda_rt_emu.h)\n", (int) attr);
         return emu_rt__ret(cudaErrorInvalidValue);
   }
   *value = v;
   return cudaSuccess;
}  /* end cudaDeviceGetAttribute() */

static inline cudaError_t cudaGetDeviceProperties(cudaDeviceProp *prop,
   int device)
{
   const EMU_RT_DEVICE *d = &emu_rt.dev;
   cudaError_t err = emu_rt__enter("cudaGetDeviceProperties");

   if (prop == NULL) return emu_rt__ret(cudaErrorInvalidValue);
   if (err != cudaSuccess) return emu_rt__ret(err);
   if (device < 0 || device >= d->count) {
      return emu_rt__ret(cudaErrorInvalidDevice);
   }
   memset(prop, 0, sizeof(*prop));
   snprintf(prop->name, sizeof(prop->name), "%s", d->name);
   prop->totalGlobalMem = d->total_mem;
   prop->sharedMemPerBlock = (size_t) d->smem_block;
   prop->sharedMemPerMultiprocessor = (size_t) d->smem_sm;
   prop->totalConstMem = 65536;
   prop->regsPerBlock = prop->regsPerMultiprocessor = d->regs_sm;
   prop->maxThreadsPerBlock = d->max_threads_block;
   prop->maxThreadsDim[0] = prop->maxThreadsDim[1] = d->max_threads_block;
   prop->maxThreadsDim[2] = 64;
   prop->maxGridSize[0] = 2147483647;
   prop->maxGridSize[1] = prop->maxGridSize[2] = 65535;
   prop->clockRate = d->clock_khz;
   prop->memoryClockRate = 9500000;
   prop->memoryBusWidth = 256;
   prop->major = d->cc_major;
   prop->minor = d->cc_minor;
   prop->multiProcessorCount = d->sms;
   prop->maxThreadsPerMultiProcessor = d->max_threads_sm;
   prop->maxBlocksPerMultiProcessor = d->max_blocks_sm;
   prop->l2CacheSize = d->l2_bytes;
   prop->persistingL2CacheMaxSize = d->persist_l2_max;
   prop->accessPolicyMaxWindowSize = d->max_window;
   prop->concurrentKernels = 1;
   prop->asyncEngineCount = 2;
   prop->unifiedAddressing = 1;
   prop->canMapHostMemory = 1;
   prop->pciBusID = device + 1;
   return cudaSuccess;
}  /* end cudaGetDeviceProperties() */

static inline cudaError_t cudaDeviceSetLimit(enum cudaLimit limit,
   size_t value)
{
   cudaError_t err = emu_rt__enter("cudaDeviceSetLimit");

   if (err == cudaSuccess && limit == cudaLimitPersistingL2CacheSize) {
      if (value > (size_t) emu_rt.dev.persist_l2_max) {
         value = (size_t) emu_rt.dev.persist_l2_max;
      }
      emu_rt.limit_persist_l2 = value;
   }
   return emu_rt__ret(err);
}  /* end cudaDeviceSetLimit() */

static inline cudaError_t cudaDeviceGetLimit(size_t *value,
   enum cudaLimit limit)
{
   cudaError_t err = emu_rt__enter("cudaDeviceGetLimit");

   if (value == NULL) return emu_rt__ret(cudaErrorInvalidValue);
   if (err == cudaSuccess) {
      *value = limit == cudaLimitPersistingL2CacheSize ?
         emu_rt.limit_persist_l2 : 0;
   }
   return emu_rt__ret(err);
}  /* end cudaDeviceGetLimit() */

static inline cudaError_t cudaCtxResetPersistingL2Cache(void)
{
   return emu_rt__ret(emu_rt__enter("cudaCtxResetPersistingL2Cache"));
}  /* end cudaCtxResetPersistingL2Cache() */

static inline cudaError_t cudaDeviceSetCacheConfig(enum cudaFuncCache c)
{
   (void) c;
   return emu_rt__ret(emu_rt__enter("cudaDeviceSetCacheConfig"));
}  /* end cudaDeviceSetCacheConfig() */

static inline cudaError_t cudaDriverGetVersion(int *version)
{
   cudaError_t err = emu_rt__enter("cudaDriverGetVersion");

   if (version == NULL) return emu_rt__ret(cudaErrorInvalidValue);
   *version = 12090;
   return emu_rt__ret(err);
}  /* end cudaDriverGetVersion() */

static inline cudaError_t cudaRuntimeGetVersion(int *version)
{
   cudaError_t err = emu_rt__enter("cudaRuntimeGetVersion");

   if (version == NULL) return emu_rt__ret(cudaErrorInvalidValue);
   *version = 12090;
   return emu_rt__ret(err);
}  /* end cudaRuntimeGetVersion() */

/****************************************************************
 * MEMORY
 ****************************************************************/

/**
 * @private
 * Fill @a n bytes at @a p with a deterministic garbage pattern.
*/
static inline void emu_rt__garbage(void *p, size_t n, uint64_t seed)
{
   unsigned char block[4096], *b = (unsigned char *) p;
   uint64_t z, x = seed;
   size_t i, k;

   for (i = 0; i < sizeof(block); i += 8) {
      x += 0x9e3779b97f4a7c15ULL;
      z = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
      z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
      z ^= z >> 31;
      memcpy(&block[i], &z, 8);
   }
   for (i = 0; i < n; i += k) {
      k = n - i < sizeof(block) ? n - i : sizeof(block);
      memcpy(&b[i], block, k);
      /* make every block differ */
      z = (uint64_t) i * 0x9e3779b97f4a7c15ULL;
      if (k >= 8) memcpy(&b[i], &z, 8);
   }
}  /* end emu_rt__garbage() */

/**
 * @private
 * Allocate @a size bytes of @a kind: a shared anonymous mapping (visible
 * to worker processes) with the buffer ending at a PROT_NONE guard page
 * (256-byte aligned, so at most 255 bytes of slack before the guard).
*/
static inline cudaError_t emu_rt__alloc(void **pp, size_t size, int kind,
   const char *api, const char *tag)
{
   EMU_RT_ALLOC *a;
   cudaError_t err = emu_rt__enter(api);
   size_t page, body, data, len;
   unsigned char *base;
   void *grow;

   if (pp == NULL) return emu_rt__ret(cudaErrorInvalidValue);
   if (err != cudaSuccess) return emu_rt__ret(err);
   if (size == 0) {
      *pp = NULL;
      return cudaSuccess;
   }
   if (kind == EMU_RT_DEVMEM && (size > emu_rt.dev.total_mem ||
         emu_rt.live_dev_bytes > emu_rt.dev.total_mem - size)) {
      return emu_rt__ret(cudaErrorMemoryAllocation);
   }
   if (emu_rt.nalloc == emu_rt.maxalloc) {
      grow = realloc(emu_rt.alloc, (emu_rt.maxalloc + 256) *
         sizeof(EMU_RT_ALLOC));
      if (grow == NULL) return emu_rt__ret(cudaErrorMemoryAllocation);
      emu_rt.alloc = (EMU_RT_ALLOC *) grow;
      emu_rt.maxalloc += 256;
   }
   page = (size_t) sysconf(_SC_PAGESIZE);
   body = (size + 255) & ~((size_t) 255);
   data = (body + page - 1) / page * page;
   len = data + (2 * page);
   base = (unsigned char *) mmap(NULL, len, PROT_READ | PROT_WRITE,
      MAP_SHARED | MAP_ANONYMOUS, -1, 0);
   if (base == (unsigned char *) MAP_FAILED) {
      return emu_rt__ret(cudaErrorMemoryAllocation);
   }
   if (mprotect(base, page, PROT_NONE) != 0 ||
         mprotect(base + page + data, page, PROT_NONE) != 0) {
      munmap(base, len);
      return emu_rt__ret(cudaErrorMemoryAllocation);
   }
   a = &emu_rt.alloc[emu_rt.nalloc++];
   a->base = base;
   a->maplen = len;
   a->ptr = base + page + data - body;
   a->size = size;
   a->kind = kind;
   a->live = 1;
   a->seq = ++emu_rt.alloc_seq;
   a->call = emu_rt_calls(api);
   a->api = api;
   a->tag = tag;
   if (emu_rt.garbage) emu_rt__garbage(a->ptr, body, a->seq);
   if (kind == EMU_RT_DEVMEM) emu_rt.live_dev_bytes += size;
   *pp = a->ptr;
   return cudaSuccess;
}  /* end emu_rt__alloc() */

/**
 * @private
 * Free an allocation of @a kind (implicit device synchronization).
*/
static inline cudaError_t emu_rt__free(void *p, int kind, const char *api)
{
   EMU_RT_ALLOC *a, *dead = NULL;
   cudaError_t err = emu_rt__enter(api);
   size_t i;

   if (err != cudaSuccess) return emu_rt__ret(err);
   if (p == NULL) return cudaSuccess;
   emu_rt__complete_all(0, api);
   for (i = 0; i < emu_rt.nalloc; i++) {
      a = &emu_rt.alloc[i];
      if (a->ptr != p) continue;
      if (!a->live) {
         dead = a;
         continue;
      }
      if (a->kind != kind) {
         emu_rt__misuse("%s() of %s memory %p (%s)", api,
            a->kind == EMU_RT_DEVMEM ? "device" : "pinned host", p, a->tag);
         return emu_rt__ret(cudaErrorInvalidValue);
      }
      munmap(a->base, a->maplen);
      a->live = 0;
      if (kind == EMU_RT_DEVMEM) emu_rt.live_dev_bytes -= a->size;
      return cudaSuccess;
   }
   if (dead) {
      emu_rt__misuse("%s() of %p twice (double free of %s)", api, p,
         dead->tag);
   } else if (emu_rt__owner(p, 0)) {
      emu_rt__misuse("%s() of an interior pointer %p", api, p);
   } else emu_rt__misuse("%s() of an unknown pointer %p", api, p);
   return emu_rt__ret(cudaErrorInvalidValue);
}  /* end emu_rt__free() */

/* typed allocation entry points take any T ** (as the C++ templates) */
#define cudaMalloc(PP, SIZE) \
   emu_rt__alloc((void **) (PP), (size_t) (SIZE), EMU_RT_DEVMEM, \
      "cudaMalloc", #PP)
#define cudaMallocHost(PP, SIZE) \
   emu_rt__alloc((void **) (PP), (size_t) (SIZE), EMU_RT_HOSTMEM, \
      "cudaMallocHost", #PP)
#define cudaHostAlloc(PP, SIZE, FLAGS) \
   ((void) (FLAGS), emu_rt__alloc((void **) (PP), (size_t) (SIZE), \
      EMU_RT_HOSTMEM, "cudaHostAlloc", #PP))

static inline cudaError_t cudaFree(void *p)
{
   return emu_rt__free(p, EMU_RT_DEVMEM, "cudaFree");
}  /* end cudaFree() */

static inline cudaError_t cudaFreeHost(void *p)
{
   return emu_rt__free(p, EMU_RT_HOSTMEM, "cudaFreeHost");
}  /* end cudaFreeHost() */

static inline cudaError_t cudaHostGetDevicePointer(void **pdev, void *phost,
   unsigned int flags)
{
   cudaError_t err = emu_rt__enter("cudaHostGetDevicePointer");

   (void) flags;
   if (pdev == NULL || emu_rt__owner(phost, EMU_RT_HOSTMEM) == NULL) {
      return emu_rt__ret(cudaErrorInvalidValue);
   }
   if (err == cudaSuccess) *pdev = phost;
   return emu_rt__ret(err);
}  /* end cudaHostGetDevicePointer() */

static inline cudaError_t cudaMemGetInfo(size_t *mfree, size_t *mtotal)
{
   cudaError_t err = emu_rt__enter("cudaMemGetInfo");

   if (mfree == NULL || mtotal == NULL) {
      return emu_rt__ret(cudaErrorInvalidValue);
   }
   if (err != cudaSuccess) return emu_rt__ret(err);
   *mtotal = emu_rt.dev.total_mem;
   *mfree = emu_rt.dev.total_mem - emu_rt.live_dev_bytes;
   return cudaSuccess;
}  /* end cudaMemGetInfo() */

/**
 * @private
 * Check a range of @a n bytes at @a p: device memory must lie within one
 * live device allocation; host memory may not overlap device memory, and
 * must lie within its pinned allocation if it starts in one.
 * @param want 1 = device, 0 = host, -1 = either (cudaMemcpyDefault)
 * @returns 1 if @a p is device memory, 0 if host memory, -1 if invalid
*/
static inline int emu_rt__range(const void *p, size_t n, int want,
   const char *api, const char *what)
{
   EMU_RT_ALLOC *a;
   size_t off;

   if (p == NULL) {
      emu_rt__misuse("%s(): NULL %s pointer", api, what);
      return (-1);
   }
   a = emu_rt__owner(p, 0);
   if (a && a->kind == EMU_RT_DEVMEM) {
      off = (size_t) ((uintptr_t) p - (uintptr_t) a->ptr);
      if (want == 0) {
         emu_rt__misuse("%s(): %s %p is device memory (%s), host memory"
            " expected", api, what, p, a->tag);
         return (-1);
      }
      if (n > a->size - off) {
         emu_rt__misuse("%s(): %s range of %lu bytes at offset %lu"
            " exceeds the device allocation of %lu bytes (%s)", api, what,
            (unsigned long) n, (unsigned long) off, (unsigned long) a->size,
            a->tag);
         return (-1);
      }
      return 1;
   }
   if (want == 1) {
      emu_rt__misuse("%s(): %s %p is not device memory", api, what, p);
      return (-1);
   }
   if (a) {
      off = (size_t) ((uintptr_t) p - (uintptr_t) a->ptr);
      if (n > a->size - off) {
         emu_rt__misuse("%s(): %s range of %lu bytes exceeds the pinned"
            " allocation of %lu bytes (%s)", api, what, (unsigned long) n,
            (unsigned long) a->size, a->tag);
         return (-1);
      }
   }
   return 0;
}  /* end emu_rt__range() */

/**
 * @private
 * Validate a copy; @returns its effective direction (as cudaMemcpyKind),
 * or -1 if invalid.
*/
static inline int emu_rt__copykind(void *dst, const void *src, size_t n,
   enum cudaMemcpyKind kind, const char *api)
{
   int d, s;

   switch (kind) {
      case cudaMemcpyHostToHost: d = 0; s = 0; break;
      case cudaMemcpyHostToDevice: d = 1; s = 0; break;
      case cudaMemcpyDeviceToHost: d = 0; s = 1; break;
      case cudaMemcpyDeviceToDevice: d = 1; s = 1; break;
      case cudaMemcpyDefault: d = -1; s = -1; break;
      default:
         emu_rt__misuse("%s(): invalid copy kind %d", api, (int) kind);
         return (-1);
   }
   if (n == 0) return (int) kind;
   d = emu_rt__range(dst, n, d, api, "destination");
   s = emu_rt__range(src, n, s, api, "source");
   if (d < 0 || s < 0) return (-1);
   return d ? (s ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice) :
      (s ? cudaMemcpyDeviceToHost : cudaMemcpyHostToHost);
}  /* end emu_rt__copykind() */

/**
 * @private
 * Fake transfer time of @a n bytes, milliseconds.
*/
static inline double emu_rt__copy_ms(size_t n)
{
   return 0.005 + (double) n / 12.0e6;
}  /* end emu_rt__copy_ms() */

static inline cudaError_t cudaMemcpy(void *dst, const void *src, size_t n,
   enum cudaMemcpyKind kind)
{
   cudaError_t err = emu_rt__enter("cudaMemcpy");

   if (err != cudaSuccess) return emu_rt__ret(err);
   if (emu_rt__copykind(dst, src, n, kind, "cudaMemcpy") < 0) {
      return emu_rt__ret(cudaErrorInvalidValue);
   }
   /* legacy default stream, host synchronous: waits for blocking streams */
   emu_rt__complete_all(1, "cudaMemcpy");
   if (n) memmove(dst, src, n);
   emu_rt.now_ms += emu_rt__copy_ms(n);
   return cudaSuccess;
}  /* end cudaMemcpy() */

static inline cudaError_t cudaMemcpyAsync(void *dst, const void *src,
   size_t n, enum cudaMemcpyKind kind, cudaStream_t stream)
{
   EMU_RT__PENDING *p;
   EMU_RT__STREAM *s;
   cudaError_t err = emu_rt__enter("cudaMemcpyAsync");
   unsigned long seq;
   int k;

   if (err == cudaSuccess) err = emu_rt__stream(stream, "cudaMemcpyAsync", &s);
   if (err != cudaSuccess) return emu_rt__ret(err);
   k = emu_rt__copykind(dst, src, n, kind, "cudaMemcpyAsync");
   if (k < 0) return emu_rt__ret(cudaErrorInvalidValue);
   if (n == 0) return cudaSuccess;
   if (k == cudaMemcpyDeviceToHost && emu_rt__owner(dst, EMU_RT_HOSTMEM)) {
      /* pinned destination: the data lands when the stream completes */
      p = (EMU_RT__PENDING *) calloc(1, sizeof(*p));
      if (p) p->data = malloc(n);
      if (p == NULL || p->data == NULL) {
         free(p);
         return emu_rt__ret(cudaErrorMemoryAllocation);
      }
      memcpy(p->data, src, n);
      p->dst = dst;
      p->n = n;
      seq = emu_rt__enqueue(s, emu_rt__copy_ms(n));
      p->seq = seq;
      if (s->tail) s->tail->next = p;
      else s->head = p;
      s->tail = p;
      return cudaSuccess;
   }
   if (k == cudaMemcpyDeviceToHost || k == cudaMemcpyHostToHost) {
      /* pageable destination: synchronous after the stream's work */
      emu_rt__complete(s, "cudaMemcpyAsync");
      memmove(dst, src, n);
      emu_rt__enqueue(s, emu_rt__copy_ms(n));
      emu_rt__complete(s, "cudaMemcpyAsync");
      return cudaSuccess;
   }
   memmove(dst, src, n);
   emu_rt__enqueue(s, emu_rt__copy_ms(n));
   return cudaSuccess;
}  /* end cudaMemcpyAsync() */

static inline cudaError_t cudaMemset(void *p, int value, size_t n)
{
   cudaError_t err = emu_rt__enter("cudaMemset");

   if (err != cudaSuccess) return emu_rt__ret(err);
   if (n == 0) return cudaSuccess;
   if (emu_rt__range(p, n, 1, "cudaMemset", "destination") < 0) {
      return emu_rt__ret(cudaErrorInvalidValue);
   }
   memset(p, value, n);
   emu_rt__enqueue(&emu_rt.null_stream, emu_rt__copy_ms(n) / 4);
   return cudaSuccess;
}  /* end cudaMemset() */

static inline cudaError_t cudaMemsetAsync(void *p, int value, size_t n,
   cudaStream_t stream)
{
   EMU_RT__STREAM *s;
   cudaError_t err = emu_rt__enter("cudaMemsetAsync");

   if (err == cudaSuccess) err = emu_rt__stream(stream, "cudaMemsetAsync", &s);
   if (err != cudaSuccess) return emu_rt__ret(err);
   if (n == 0) return cudaSuccess;
   if (emu_rt__range(p, n, 1, "cudaMemsetAsync", "destination") < 0) {
      return emu_rt__ret(cudaErrorInvalidValue);
   }
   memset(p, value, n);
   emu_rt__enqueue(s, emu_rt__copy_ms(n) / 4);
   return cudaSuccess;
}  /* end cudaMemsetAsync() */

/****************************************************************
 * STREAMS AND EVENTS
 ****************************************************************/

static inline cudaError_t cudaStreamCreateWithFlags(cudaStream_t *stream,
   unsigned int flags)
{
   EMU_RT__STREAM *s;
   cudaError_t err = emu_rt__enter("cudaStreamCreate");
   void *grow;

   if (stream == NULL) return emu_rt__ret(cudaErrorInvalidValue);
   if (err != cudaSuccess) return emu_rt__ret(err);
   if (emu_rt.nstreams == emu_rt.maxstreams) {
      grow = realloc(emu_rt.streams, (emu_rt.maxstreams + 64) *
         sizeof(EMU_RT__STREAM *));
      if (grow == NULL) return emu_rt__ret(cudaErrorMemoryAllocation);
      emu_rt.streams = (EMU_RT__STREAM **) grow;
      emu_rt.maxstreams += 64;
   }
   s = (EMU_RT__STREAM *) calloc(1, sizeof(*s));
   if (s == NULL) return emu_rt__ret(cudaErrorMemoryAllocation);
   s->live = 1;
   s->nonblocking = (flags & cudaStreamNonBlocking) ? 1 : 0;
   s->id = (int) emu_rt.nstreams + 1;
   s->t_ready = emu_rt.now_ms;
   emu_rt.streams[emu_rt.nstreams++] = s;
   *stream = (cudaStream_t) s;
   return cudaSuccess;
}  /* end cudaStreamCreateWithFlags() */

static inline cudaError_t cudaStreamCreate(cudaStream_t *stream)
{
   return cudaStreamCreateWithFlags(stream, cudaStreamDefault);
}  /* end cudaStreamCreate() */

static inline cudaError_t cudaStreamCreateWithPriority(cudaStream_t *stream,
   unsigned int flags, int priority)
{
   (void) priority;
   return cudaStreamCreateWithFlags(stream, flags);
}  /* end cudaStreamCreateWithPriority() */

static inline cudaError_t cudaDeviceGetStreamPriorityRange(int *least,
   int *greatest)
{
   cudaError_t err = emu_rt__enter("cudaDeviceGetStreamPriorityRange");

   if (least) *least = 0;
   if (greatest) *greatest = -5;
   return emu_rt__ret(err);
}  /* end cudaDeviceGetStreamPriorityRange() */

static inline cudaError_t cudaStreamDestroy(cudaStream_t stream)
{
   EMU_RT__STREAM *s;
   cudaError_t err = emu_rt__enter("cudaStreamDestroy");

   if (err == cudaSuccess && stream == NULL) {
      emu_rt__misuse("cudaStreamDestroy() of the legacy default stream");
      err = cudaErrorInvalidResourceHandle;
   }
   if (err == cudaSuccess) {
      err = emu_rt__stream(stream, "cudaStreamDestroy", &s);
   }
   if (err != cudaSuccess) return emu_rt__ret(err);
   /* returns at once; remaining work still completes (later ticks) */
   s->live = 0;
   return cudaSuccess;
}  /* end cudaStreamDestroy() */

static inline cudaError_t cudaStreamQuery(cudaStream_t stream)
{
   EMU_RT__STREAM *s;
   cudaError_t err = emu_rt__enter("cudaStreamQuery");

   if (err == cudaSuccess) err = emu_rt__stream(stream, "cudaStreamQuery", &s);
   if (err != cudaSuccess) return emu_rt__ret(err);
   if (emu_rt__busy(s)) {
      if (emu_rt.spin_tick != emu_rt.tick) {
         emu_rt.spin_tick = emu_rt.tick;
         emu_rt.spins = 0;
      }
      if (++emu_rt.spins > EMU_RT_SPIN_LIMIT) {
         printf("EMU: the code under test busy-waits on cudaStreamQuery()"
            " (the fake clock never advances inside a call)\n");
         fflush(stdout);
         abort();
      }
      return cudaErrorNotReady;
   }
   emu_rt__flush(s, ~0UL);
   return cudaSuccess;
}  /* end cudaStreamQuery() */

static inline cudaError_t cudaStreamSynchronize(cudaStream_t stream)
{
   EMU_RT__STREAM *s;
   cudaError_t err = emu_rt__enter("cudaStreamSynchronize");

   if (err == cudaSuccess) {
      err = emu_rt__stream(stream, "cudaStreamSynchronize", &s);
   }
   if (err != cudaSuccess) return emu_rt__ret(err);
   if (s == &emu_rt.null_stream) {
      emu_rt__complete_all(1, "cudaStreamSynchronize");
   } else emu_rt__complete(s, "cudaStreamSynchronize");
   return cudaSuccess;
}  /* end cudaStreamSynchronize() */

static inline cudaError_t cudaDeviceSynchronize(void)
{
   cudaError_t err = emu_rt__enter("cudaDeviceSynchronize");

   if (err != cudaSuccess) return emu_rt__ret(err);
   emu_rt__complete_all(0, "cudaDeviceSynchronize");
   return cudaSuccess;
}  /* end cudaDeviceSynchronize() */

static inline cudaError_t cudaStreamSetAttribute(cudaStream_t stream,
   cudaStreamAttrID attr, const cudaStreamAttrValue *value)
{
   EMU_RT__STREAM *s;
   cudaError_t err = emu_rt__enter("cudaStreamSetAttribute");

   if (err == cudaSuccess) {
      err = emu_rt__stream(stream, "cudaStreamSetAttribute", &s);
   }
   if (err != cudaSuccess) return emu_rt__ret(err);
   if (value == NULL) return emu_rt__ret(cudaErrorInvalidValue);
   if (attr == cudaStreamAttributeAccessPolicyWindow) {
      if (value->accessPolicyWindow.num_bytes >
            (size_t) emu_rt.dev.max_window ||
            value->accessPolicyWindow.hitRatio < 0.0f ||
            value->accessPolicyWindow.hitRatio > 1.0f) {
         return emu_rt__ret(cudaErrorInvalidValue);
      }
      if (value->accessPolicyWindow.num_bytes &&
            emu_rt__range(value->accessPolicyWindow.base_ptr,
            value->accessPolicyWindow.num_bytes, 1, "cudaStreamSetAttribute",
            "access policy window") < 0) {
         return emu_rt__ret(cudaErrorInvalidValue);
      }
      s->window = *value;
   }
   return cudaSuccess;
}  /* end cudaStreamSetAttribute() */

static inline cudaError_t cudaStreamGetAttribute(cudaStream_t stream,
   cudaStreamAttrID attr, cudaStreamAttrValue *value)
{
   EMU_RT__STREAM *s;
   cudaError_t err = emu_rt__enter("cudaStreamGetAttribute");

   if (err == cudaSuccess) {
      err = emu_rt__stream(stream, "cudaStreamGetAttribute", &s);
   }
   if (err != cudaSuccess) return emu_rt__ret(err);
   if (value == NULL) return emu_rt__ret(cudaErrorInvalidValue);
   if (attr == cudaStreamAttributeAccessPolicyWindow) *value = s->window;
   else memset(value, 0, sizeof(*value));
   return cudaSuccess;
}  /* end cudaStreamGetAttribute() */

/**
 * @private
 * Has the record point of event @a e completed?
*/
static inline int emu_rt__event_done(const struct emu_rt__event *e)
{
   if (!e->recorded || e->seq <= e->s->done_seq) return 1;
   return !e->s->stuck && emu_rt.tick >= e->done_tick;
}  /* end emu_rt__event_done() */

static inline cudaError_t cudaEventCreateWithFlags(cudaEvent_t *event,
   unsigned int flags)
{
   struct emu_rt__event *e;
   cudaError_t err = emu_rt__enter("cudaEventCreate");
   void *grow;

   if (event == NULL) return emu_rt__ret(cudaErrorInvalidValue);
   if (err != cudaSuccess) return emu_rt__ret(err);
   if (emu_rt.nevents == emu_rt.maxevents) {
      grow = realloc(emu_rt.events, (emu_rt.maxevents + 64) *
         sizeof(struct emu_rt__event *));
      if (grow == NULL) return emu_rt__ret(cudaErrorMemoryAllocation);
      emu_rt.events = (struct emu_rt__event **) grow;
      emu_rt.maxevents += 64;
   }
   e = (struct emu_rt__event *) calloc(1, sizeof(*e));
   if (e == NULL) return emu_rt__ret(cudaErrorMemoryAllocation);
   e->live = 1;
   e->timing = (flags & cudaEventDisableTiming) ? 0 : 1;
   emu_rt.events[emu_rt.nevents++] = e;
   *event = e;
   return cudaSuccess;
}  /* end cudaEventCreateWithFlags() */

static inline cudaError_t cudaEventCreate(cudaEvent_t *event)
{
   return cudaEventCreateWithFlags(event, cudaEventDefault);
}  /* end cudaEventCreate() */

static inline cudaError_t cudaEventDestroy(cudaEvent_t event)
{
   cudaError_t err = emu_rt__enter("cudaEventDestroy");

   if (err == cudaSuccess) err = emu_rt__event(event, "cudaEventDestroy");
   if (err != cudaSuccess) return emu_rt__ret(err);
   event->live = 0;
   return cudaSuccess;
}  /* end cudaEventDestroy() */

static inline cudaError_t cudaEventRecord(cudaEvent_t event,
   cudaStream_t stream)
{
   EMU_RT__STREAM *s;
   cudaError_t err = emu_rt__enter("cudaEventRecord");

   if (err == cudaSuccess) err = emu_rt__event(event, "cudaEventRecord");
   if (err == cudaSuccess) err = emu_rt__stream(stream, "cudaEventRecord", &s);
   if (err != cudaSuccess) return emu_rt__ret(err);
   event->recorded = 1;
   event->s = s;
   event->seq = s->last_seq;
   event->done_tick = s->done_tick > emu_rt.tick ? s->done_tick :
      emu_rt.tick;
   event->t = s->t_ready > emu_rt.now_ms ? s->t_ready : emu_rt.now_ms;
   return cudaSuccess;
}  /* end cudaEventRecord() */

static inline cudaError_t cudaEventQuery(cudaEvent_t event)
{
   cudaError_t err = emu_rt__enter("cudaEventQuery");

   if (err == cudaSuccess) err = emu_rt__event(event, "cudaEventQuery");
   if (err != cudaSuccess) return emu_rt__ret(err);
   if (!emu_rt__event_done(event)) return cudaErrorNotReady;
   if (event->recorded) emu_rt__flush(event->s, event->seq);
   return cudaSuccess;
}  /* end cudaEventQuery() */

static inline cudaError_t cudaEventSynchronize(cudaEvent_t event)
{
   cudaError_t err = emu_rt__enter("cudaEventSynchronize");

   if (err == cudaSuccess) err = emu_rt__event(event, "cudaEventSynchronize");
   if (err != cudaSuccess) return emu_rt__ret(err);
   if (!event->recorded) return cudaSuccess;
   if (!emu_rt__event_done(event) && event->s->stuck) {
      emu_rt.hangs++;
      printf("EMU: cudaEventSynchronize() waits for a stuck stream: a real"
         " GPU would hang\n");
      fflush(stdout);
   }
   emu_rt__flush(event->s, event->seq);
   if (event->t > emu_rt.now_ms) emu_rt.now_ms = event->t;
   return cudaSuccess;
}  /* end cudaEventSynchronize() */

static inline cudaError_t cudaEventElapsedTime(float *ms, cudaEvent_t start,
   cudaEvent_t stop)
{
   cudaError_t err = emu_rt__enter("cudaEventElapsedTime");

   if (err == cudaSuccess) err = emu_rt__event(start, "cudaEventElapsedTime");
   if (err == cudaSuccess) err = emu_rt__event(stop, "cudaEventElapsedTime");
   if (err != cudaSuccess) return emu_rt__ret(err);
   if (ms == NULL) return emu_rt__ret(cudaErrorInvalidValue);
   if (!start->recorded || !stop->recorded || !start->timing ||
         !stop->timing) {
      return emu_rt__ret(cudaErrorInvalidResourceHandle);
   }
   if (!emu_rt__event_done(start) || !emu_rt__event_done(stop)) {
      return cudaErrorNotReady;
   }
   *ms = (float) (stop->t - start->t);
   return cudaSuccess;
}  /* end cudaEventElapsedTime() */

static inline cudaError_t cudaStreamWaitEvent(cudaStream_t stream,
   cudaEvent_t event, unsigned int flags)
{
   EMU_RT__STREAM *s;
   cudaError_t err = emu_rt__enter("cudaStreamWaitEvent");

   (void) flags;
   if (err == cudaSuccess) err = emu_rt__event(event, "cudaStreamWaitEvent");
   if (err == cudaSuccess) {
      err = emu_rt__stream(stream, "cudaStreamWaitEvent", &s);
   }
   if (err != cudaSuccess) return emu_rt__ret(err);
   if (event->recorded && !emu_rt__event_done(event)) {
      if (s->done_tick < event->done_tick) s->done_tick = event->done_tick;
      if (s->t_ready < event->t) s->t_ready = event->t;
   }
   return cudaSuccess;
}  /* end cudaStreamWaitEvent() */

/****************************************************************
 * KERNELS: LAUNCH, OCCUPANCY, ATTRIBUTES
 ****************************************************************/

/**
 * @private
 * Begin a kernel launch (CUDA_KERNEL()): validate the stream and the
 * launch shape, apply fault rules ("launch:<name>", "launch:*"); on
 * failure no thread runs and the error is reported by
 * cudaGetLastError(), as for a real launch.
*/
static inline void emu_rt__kbegin(const char *name, long long grid,
   long long block, size_t shmem, cudaStream_t stream)
{
   EMU_RT__LAUNCH *L = &emu_rt.launch;
   EMU_RT__STREAM *s = NULL;
   EMU_RT_KCFG *k;
   cudaError_t err;
   char key[96];
   long long maxb;

   if (L->active) {
      printf("EMU: kernel %s launched inside kernel %s\n", name, L->name);
      fflush(stdout);
      abort();
   }
   memset(L, 0, sizeof(*L));
   L->active = 1;
   L->name = name;
   snprintf(key, sizeof(key), "launch:%s", name);
   err = emu_rt__enter(key);
   if (err == cudaSuccess) err = emu_rt__fault("launch:*");
   if (err == cudaSuccess) err = emu_rt__stream(stream, name, &s);
   k = emu_rt_kcfg(name, 1);
   maxb = k && k->max_block > 0 ? k->max_block : emu_rt.dev.max_threads_block;
   if (err == cudaSuccess && (grid < 1 || grid > 2147483647LL ||
         block < 1 || block > maxb)) {
      printf("EMU: kernel %s: invalid launch <<<%lld, %lld>>>\n", name,
         grid, block);
      err = cudaErrorInvalidConfiguration;
   }
   if (err == cudaSuccess && shmem > (size_t) emu_rt.dev.smem_block) {
      err = cudaErrorInvalidValue;
   }
   if (err != cudaSuccess) {
      emu_rt__ret(err);
      emu_rt.launch_fail++;
      if (k) k->failed++;
      return;
   }
   L->ok = 1;
   L->s = s;
   L->cfg = k;
   L->serial = ++emu_rt.launches;
   if (k) k->launches++;
   emu__launch((unsigned int) grid, (unsigned int) block);
   if (emu_rt.verbose > 1) {
      printf("EMU: launch #%lu %s<<<%lld, %lld>>> on stream %d\n",
         L->serial, name, grid, block, s->id);
   }
}  /* end emu_rt__kbegin() */

/**
 * @private
 * End a kernel launch: the stream becomes busy, then the launch hook.
*/
static inline void emu_rt__kend(void)
{
   EMU_RT__LAUNCH *L = &emu_rt.launch;
   double ms;

   L->active = 0;
   if (!L->ok) return;
   ms = L->cfg && L->cfg->ms >= 0.0 ? L->cfg->ms :
      0.005 + ((double) gridDim.x * (double) blockDim.x) * 1.0e-6;
   emu_rt__enqueue(L->s, ms);
   if (emu_rt.post_launch) {
      emu_rt.post_launch(L->name, L->s == &emu_rt.null_stream ? NULL :
         (cudaStream_t) L->s, emu_rt.post_launch_user);
   }
}  /* end emu_rt__kend() */

/**
 * @private
 * Advance the current launch to its next thread (block by block).
 * @returns non-zero while threads remain
*/
static inline int emu_rt__knext(void)
{
   EMU_RT__LAUNCH *L = &emu_rt.launch;

   if (!L->active) return 0;
   if (L->ok && !L->done) {
      if (!L->started) {
         L->started = 1;
         threadIdx.x = blockIdx.x = 0;
         return 1;
      }
      if (++threadIdx.x < blockDim.x) return 1;
      threadIdx.x = 0;
      if (++blockIdx.x < gridDim.x) return 1;
   }
   emu_rt__kend();
   return 0;
}  /* end emu_rt__knext() */

/**
 * Mark the current launch as complete after the running emulated thread
 * (for a launch redirected by EMU_RT_KERNEL() that handled all of its
 * threads at once).
*/
static inline void emu_rt_kernel_done(void)
{
   emu_rt.launch.done = 1;
}  /* end emu_rt_kernel_done() */

/**
 * Serial number of the current (or last) kernel launch.
*/
static inline unsigned long emu_rt_launch_serial(void)
{
   return emu_rt.launch.serial;
}  /* end emu_rt_launch_serial() */

/**
 * Run @a fn(@a user) once per thread of the current launch (all blocks),
 * with the emulated built-in variables set, split by blocks over
 * emu_rt.nproc processes (fork()). Only device memory (shared mappings)
 * written by @a fn is visible afterwards: use it for kernels whose
 * threads write device memory only, without atomics. A crashed worker
 * sets a sticky cudaErrorIllegalAddress.
 * @returns 0 on success, -1 if a worker failed
*/
static inline int emu_rt_run_parallel(void (*fn)(void *user), void *user)
{
   pid_t pid[EMU_RT_MAXPROC];
   unsigned int b0[EMU_RT_MAXPROC], b1[EMU_RT_MAXPROC], b, t;
   unsigned int grid = gridDim.x, n, c;
   int own[EMU_RT_MAXPROC], st, bad = 0;

   n = emu_rt.nproc < 1 ? 1 : emu_rt.nproc > EMU_RT_MAXPROC ?
      EMU_RT_MAXPROC : (unsigned int) emu_rt.nproc;
   if (n > grid) n = grid;
   if (n < 1) return 0;
   fflush(stdout);
   fflush(stderr);
   for (c = 0; c < n; c++) {
      b0[c] = (unsigned int) (((unsigned long long) grid * c) / n);
      b1[c] = (unsigned int) (((unsigned long long) grid * (c + 1)) / n);
      own[c] = 1;
      pid[c] = 0;
      if (c == 0) continue;
      pid[c] = fork();
      if (pid[c] == 0) {
         /* worker: own range only, then leave without cleanup */
         for (b = b0[c]; b < b1[c]; b++) {
            for (t = 0; t < blockDim.x; t++) {
               blockIdx.x = b;
               threadIdx.x = t;
               fn(user);
            }
         }
         _exit(0);
      }
      if (pid[c] > 0) own[c] = 0;
   }
   /* this process: range 0, and every range a fork() failed for */
   for (c = 0; c < n; c++) {
      if (!own[c]) continue;
      for (b = b0[c]; b < b1[c]; b++) {
         for (t = 0; t < blockDim.x; t++) {
            blockIdx.x = b;
            threadIdx.x = t;
            fn(user);
         }
      }
   }
   for (c = 1; c < n; c++) {
      if (own[c]) continue;
      while (waitpid(pid[c], &st, 0) < 0) {
         if (errno != EINTR) {
            st = -1;
            break;
         }
      }
      if (!(st >= 0 && WIFEXITED(st) && WEXITSTATUS(st) == 0)) {
         printf("EMU: kernel %s: worker process %u failed (status 0x%x)\n",
            emu_rt.launch.name ? emu_rt.launch.name : "?", c, (unsigned) st);
         bad = 1;
      }
   }
   blockIdx.x = threadIdx.x = 0;
   if (bad) {
      emu_rt.crashes++;
      emu_rt_sticky(cudaErrorIllegalAddress);
      return (-1);
   }
   return 0;
}  /* end emu_rt_run_parallel() */

/* kernel redirection hook (a test may define it before this header, e.g.
 * as a _Generic() selection over the kernel's type) */
#ifndef EMU_RT_KERNEL
   #define EMU_RT_KERNEL(FN)  FN
#endif

/* CUDA_KERNEL(FN, grid, block[, shmem, stream])(args...): replaces the
 * definition of _cuda_emu.h; same call shape as peach.cuh */
#undef CUDA_KERNEL
#define CUDA_KERNEL(FN, ...) \
   for (emu_rt__kbegin(#FN, EMU_RT__KARGS(__VA_ARGS__, 0, 0, 0)); \
      emu_rt__knext(); ) EMU_RT_KERNEL(FN)
#define EMU_RT__KARGS(GRID, BLOCK, SHMEM, STREAM, ...) \
   (long long) (GRID), (long long) (BLOCK), (size_t) (SHMEM), \
   (cudaStream_t) (STREAM)

/**
 * @private
 * cudaOccupancyMaxPotentialBlockSize() of kernel @a name.
*/
static inline cudaError_t emu_rt__occ_potential(int *min_grid, int *block,
   const char *name, size_t smem, int limit)
{
   EMU_RT_KCFG *k;
   cudaError_t err = emu_rt__enter("cudaOccupancyMaxPotentialBlockSize");
   int b;

   (void) smem;
   if (min_grid == NULL || block == NULL) {
      return emu_rt__ret(cudaErrorInvalidValue);
   }
   if (err != cudaSuccess) return emu_rt__ret(err);
   k = emu_rt_kcfg(name, 1);
   b = k ? k->block : 128;
   if (limit > 0 && b > limit) b = (limit & ~31) ? (limit & ~31) : limit;
   *block = b;
   *min_grid = k && k->min_grid > 0 ? k->min_grid :
      emu_rt.dev.sms * (k ? k->active_sm : 4);
   return cudaSuccess;
}  /* end emu_rt__occ_potential() */

/**
 * @private
 * cudaOccupancyMaxActiveBlocksPerMultiprocessor() of kernel @a name.
*/
static inline cudaError_t emu_rt__occ_active(int *num, const char *name,
   int block, size_t smem)
{
   EMU_RT_KCFG *k;
   cudaError_t err =
      emu_rt__enter("cudaOccupancyMaxActiveBlocksPerMultiprocessor");

   (void) smem;
   if (num == NULL) return emu_rt__ret(cudaErrorInvalidValue);
   if (err != cudaSuccess) return emu_rt__ret(err);
   k = emu_rt_kcfg(name, 1);
   if (block < 1 || block > emu_rt.dev.max_threads_block ||
         (k && k->max_block > 0 && block > k->max_block)) {
      *num = 0;
      return cudaSuccess;
   }
   *num = k ? k->active_sm : 4;
   return cudaSuccess;
}  /* end emu_rt__occ_active() */

/**
 * @private
 * cudaFuncGetAttributes() of kernel @a name.
*/
static inline cudaError_t emu_rt__func_attr(cudaFuncAttributes *attr,
   const char *name)
{
   EMU_RT_KCFG *k;
   cudaError_t err = emu_rt__enter("cudaFuncGetAttributes");

   if (attr == NULL) return emu_rt__ret(cudaErrorInvalidValue);
   if (err != cudaSuccess) return emu_rt__ret(err);
   k = emu_rt_kcfg(name, 1);
   memset(attr, 0, sizeof(*attr));
   attr->maxThreadsPerBlock = k && k->max_block > 0 ? k->max_block :
      emu_rt.dev.max_threads_block;
   attr->numRegs = 64;
   attr->ptxVersion = attr->binaryVersion = 10 * emu_rt.dev.cc_major +
      emu_rt.dev.cc_minor;
   attr->maxDynamicSharedSizeBytes = emu_rt.dev.smem_block;
   attr->preferredShmemCarveout = -1;
   return cudaSuccess;
}  /* end emu_rt__func_attr() */

/**
 * @private
 * Kernel configuration calls without effect (cache config, attributes).
*/
static inline cudaError_t emu_rt__func_set(const char *api,
   const char *name)
{
   (void) name;
   return emu_rt__ret(emu_rt__enter(api));
}  /* end emu_rt__func_set() */

/* kernel functions are passed as in CUDA C++ (templates) and identified
 * by name; the function itself is only evaluated as a void expression */
#define cudaOccupancyMaxPotentialBlockSize(MINGRID, BLOCK, ...) \
   emu_rt__occ_potential((MINGRID), (BLOCK), \
      EMU_RT__OCCP(__VA_ARGS__, 0, 0, 0))
#define EMU_RT__OCCP(FN, SMEM, LIMIT, ...) \
   ((void) (FN), #FN), (size_t) (SMEM), (int) (LIMIT)
#define cudaOccupancyMaxActiveBlocksPerMultiprocessor(NUM, FN, ...) \
   emu_rt__occ_active((NUM), ((void) (FN), #FN), \
      EMU_RT__OCCA(__VA_ARGS__, 0, 0))
#define EMU_RT__OCCA(BLOCK, SMEM, ...)  (int) (BLOCK), (size_t) (SMEM)
#define cudaFuncGetAttributes(ATTR, FN) \
   emu_rt__func_attr((ATTR), ((void) (FN), #FN))
#define cudaFuncSetCacheConfig(FN, CFG) \
   ((void) (CFG), emu_rt__func_set("cudaFuncSetCacheConfig", \
      ((void) (FN), #FN)))
#define cudaFuncSetAttribute(FN, ATTR, VALUE) \
   ((void) (ATTR), (void) (VALUE), emu_rt__func_set( \
      "cudaFuncSetAttribute", ((void) (FN), #FN)))

/* end include guard */
#endif
