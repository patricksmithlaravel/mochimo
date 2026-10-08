/**
 * @file peach-host.c
 * @brief Peach CUDA host code (src/peach.cu) end to end on the CPU.
 * @details OPT-IN (minutes, ~3 GiB of memory): without PEACH_TEST_HOST=1
 * the test prints a SKIP line and succeeds. Run it with:
 * <br />`make test-peach-host NO_CUDA=1 PEACH_TEST_HOST=1`
 * <br />(this file always compiles optimized, see the pragma below)
 * <br />
 * Compiles the real src/peach.cu as C against the device emulation
 * (test/_cuda_emu.h) and a fake CUDA runtime (test/_cuda_rt_emu.h), and
 * drives peach_init_cuda_device(), peach_solve_cuda() and
 * peach_free_cuda_device() like src/bin/gpuminer.c: a poll loop with one
 * shared solve output (btout), millisleep() replaced by the fake clock
 * (one tick per poll; a stream stays busy for a few ticks after work was
 * enqueued, and device-to-host copies into pinned memory land when it
 * completes). Every reported solve must pass peach_check() on the CPU,
 * belong to the current block (phash, bnum) and to a trailer presented
 * for it. Every map build must leave the full map of the current phash
 * (sampled tiles vs the reference peach_generate(), every tile vs the
 * map cache below); pipeline mode also checks the transition table.
 * Scenarios, in legacy mode (MCM_PEACH_LEGACY=1) and, when the build
 * has the pipeline solver, in pipeline mode:
 * (0) peach_checkhash_cuda() vs peach_checkhash() (CPU);
 * (1) legacy: map build, solves at a low difficulty;
 * (2) pipeline: map + T build and self-test, solves, ctx->work grows by
 *     completed nonces (not launched slots), adaptive batch size (also
 *     with batches of the old size still in flight after a halving,
 *     which slow down the batches next to them: one halving only);
 * (3) block changes (new phash) while batches are in flight, after they
 *     completed unharvested, while a stream is stuck, and during a map
 *     build: no solve for an old block, the map is rebuilt; a stream
 *     stalling during the map build delays its completion (and the
 *     transition table);
 * (4) bnum solved (by another device) / tcount 0 / BRIDGEv3 expiry:
 *     DEV_IDLE without reporting or launching, then resume;
 * (5) pipeline faults: T corrupted before / after the self-test, a
 *     forged device result, a queue overflow, a stale result and a
 *     failed kernel launch: legacy fallback; a sticky device error: the
 *     device fails; btout never receives an invalid solve;
 * (6) every cudaMalloc / cudaMallocHost / cudaStreamCreate /
 *     cudaEventCreate failing once during initialization: a legacy
 *     allocation fails the device (cleanly), a pipeline-only one falls
 *     back to legacy; afterwards the device still works, also when
 *     initialized in the host thread of the failed initialization (GPU
 *     A/B test), and a second device initialized after a failed one
 *     (gpuminer) gets the configured solver;
 * (7) peach_free_cuda_device() + re-initialization with another
 *     configuration; no leaks, double frees or invalid handles; the
 *     automatic skip masks per compute capability (8.x, 12.x) and
 *     explicit masks, with their names in the init log line, and the
 *     slots per batch they lead to (doubled when the round 0 mask drops
 *     an algorithm of every pair a, a + 4); the MD2
 *     grid (at most PEACH_CUDA_MD2_BLOCKS blocks per SM with several
 *     batch contexts, its occupancy with one); MCM_PEACH_STREAMS = 1 to 4
 *     and invalid values (automatic);
 * (8) pipeline: a trailer with difficulty 0: every final hash solves (as
 *     consensus and the legacy solver), solves are verified;
 * (9) pipeline: polling intervals of seconds (gpuminer -d); a pause of
 *     the caller while the trailer changes twice, and a change without a
 *     pause: solves only for a trailer gpuminer can send;
 * (10) one device fails on a CUDA error during a poll: the next device,
 *     polled from the same host thread, keeps working.
 * In pipeline mode, every reported solve must be for gpuminer's current
 * or previous trailer (the only ones it sends), and every batch launch
 * is checked against the trailer presented to peach_solve_cuda() and the
 * map: phash, clamped difficulty, a new epoch, the first nonce half
 * (valid haiku without a NaN replacement, q), midstate, tail, skip masks
 * and slots; the transition table must be built from the complete map
 * once every other stream is idle (map chunks and batches finished), and
 * must be finished before a batch starts. peach_pipeline_cuda_device()
 * must agree with the mode. Pipeline mode runs with the automatic number
 * of batch contexts (streams), PEACH_CUDA_NCTX_AUTO on the fake device;
 * scenario (7) also runs 1 to 4 contexts.
 * <br />
 * Map builds: the first build of a phash runs the real kcu_peach_build()
 * on all tiles in worker processes; the map is cached (per phash), and
 * later builds copy each launched chunk from the cache. The transition
 * table kernel always runs (in worker processes).
 * <br />
 * Environment: PEACH_TEST_HOST_PROCS (worker processes, default 4),
 * PEACH_TEST_HOST_CACHE (cached maps, 0..2, default 2; 1 GiB each),
 * PEACH_TEST_HOST_VERBOSE=1 (log of the code under test, launches),
 * PEACH_TEST_HOST_ONLY=legacy|pipeline (only these scenarios),
 * PEACH_TEST_HOST_STREAMS=<n> (MCM_PEACH_STREAMS of every scenario but
 * the checks of (7) that set it themselves; default unset: automatic).
 * @copyright Adequate Systems LLC, 2018-2025. All Rights Reserved.
 * <br />For license information, please refer to ../../LICENSE.md
*/

/* emulated kernels are the hot path: optimize this file whatever the
 * global flags are (CCARGS=-O2 cannot build trigg.c), but without strict
 * aliasing: the legacy device code type-puns words and floats as nvcc
 * allows */
#if defined(__GNUC__) && !defined(__clang__)
   #pragma GCC optimize ("O2", "no-strict-aliasing")
#endif

#include <math.h>    /* for isnan() (peach.cu) */
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* the copy of src/peach.cu below defines the public CUDA functions:
 * rename them (libmochimo.a has peach.cu.o in CUDA builds) */
#define peach_checkhash_cuda     peach_host_checkhash_cuda
#define peach_free_cuda_device   peach_host_free_cuda_device
#define peach_init_cuda_device   peach_host_init_cuda_device
#define peach_pipeline_cuda_device  peach_host_pipeline_cuda_device
#define peach_solve_cuda         peach_host_solve_cuda

#include "extint.h"
#include "extlib.h"
#include "extmath.h"
#include "error.h"
#include "peach.h"
#include "trigg.h"
#include "sha256.h"

/* reference access: the reference itself with ALL its external
 * definitions renamed (libmochimo.a has peach.o) */
#define peach_checkhash ref_peach_checkhash
#define peach_init ref_peach_init
#define peach_solve ref_peach_solve
#include "peach.c"
#undef peach_checkhash
#undef peach_init
#undef peach_solve

#include "_cuda_emu.h"

/* kernel redirection (see peach_host_build() / peach_host_transitions()):
 * selected by the kernel's type, any other kernel runs as is */
static void peach_host_build(word32 offset, word64 *d_map,
   word32 *d_phash);
static void peach_host_transitions(const uint4 *d_map, word16 *d_T,
   word32 offset, word32 count);
#define EMU_RT_KERNEL(FN) _Generic((FN), \
   void (*)(word32, word64 *, word32 *): peach_host_build, \
   void (*)(const uint4 *, word16 *, word32, word32): \
      peach_host_transitions, \
   default: (FN))

#include "_cuda_rt_emu.h"

/* crypto-c's sha256.cu cannot compile as C (its test function launches a
 * kernel with <<< >>>): the device SHA-256 is crypto-c's CPU SHA-256 */
#define CRYPTO_SHA256_CU
static inline void cu_sha256_init(SHA256_CTX *ctx)
   { sha256_init(ctx); }
static inline void cu_sha256_update(SHA256_CTX *ctx, const void *in,
   size_t inlen) { sha256_update(ctx, in, inlen); }
static inline void cu_sha256_final(SHA256_CTX *ctx, void *out)
   { sha256_final(ctx, out); }
static inline void cu_sha256(const void *in, size_t inlen, void *out)
   { sha256(in, inlen, out); }

/* peach.cuh: CUDA headers and the <<< >>> CUDA_KERNEL(); the emulation
 * provides both */
#define MOCHIMO_PEACH_CUH

/* the host code under test runs on the fake clock and logs to the test */
static void peach_host_plogx(int ll, const char *file, int line,
   const char *fmt, ...) __attribute__((format(printf, 4, 5)));
#define time(T)   emu_rt_time(T)
#define plogx     peach_host_plogx

#if defined(__GNUC__) && !defined(__clang__)
   #pragma GCC diagnostic push
   /* upstream code: `#pragma unroll`, `cuCONSTn860 static`, a comment
    * opener inside the file comment */
   #pragma GCC diagnostic ignored "-Wunknown-pragmas"
   #pragma GCC diagnostic ignored "-Wold-style-declaration"
   #pragma GCC diagnostic ignored "-Wcomment"
#endif
#include "peach.cu"
#if defined(__GNUC__) && !defined(__clang__)
   #pragma GCC diagnostic pop
#endif

#undef time
#undef plogx

/****************************************************************
 * TEST CONFIGURATION AND REPORTING
 ****************************************************************/

#define DYNASLEEP       10.0     /* ms between polls (gpuminer default) */
#define BUSY_TICKS      2        /* polls a stream stays busy */
#define BUILD_POLLS     400      /* max. polls for a map build */
#define SOLVE_POLLS     200      /* max. polls for the next solve */
#define MAXFAMILY       256      /* trailers presented per block */
#define MAXCACHE        2        /* cached maps */
#define NTSAMPLE        2048     /* sampled transition table entries */
#define DIFF_SOLVE      8        /* low difficulty: frequent solves */
#define DIFF_ALWAYS     1        /* every batch solves (~1/2 per nonce) */
#define DIFF_NEVER      48       /* practically never solves */
#define MAIN_THREAD     0        /* gpuminer: initializes the devices */
#define DEVICE_THREAD   1        /* gpuminer: device loop (solving) */

/* ms between polls of rig_poll() (gpuminer -d, default DYNASLEEP) */
static double PollMs = DYNASLEEP;

/* failed checks */
static unsigned long Fails;
static unsigned long SectionFails;
static const char *Section = "";
static int Verbose;

#define CHECK(COND, ...) \
   do { \
      if (!(COND)) { \
         Fails++; \
         SectionFails++; \
         printf("FAIL [%s] line %d: ", Section, __LINE__); \
         printf(__VA_ARGS__); \
         printf("\n"); \
         fflush(stdout); \
      } \
   } while (0)

/* scenario results */
#define MAXRESULT 32
static struct {
   const char *name;
   const char *result;
} Results[MAXRESULT];
static int Nresults;

/* captured log of the code under test */
static unsigned long Nlog[PLOG_DEBUG + 1];
static char LastAlert[512];
static char LastWarn[512];
static char LastInfo[512];

/**
 * Log function of the code under test (replaces plogx()): counts every
 * level, keeps the last alert, prints alerts, errors and warnings (all
 * levels with PEACH_TEST_HOST_VERBOSE=1).
*/
static void peach_host_plogx(int ll, const char *file, int line,
   const char *fmt, ...)
{
   static const char *name[] = {
      "ALERT", "ERRNO", "ERROR", "WARN", "INFO", "DEBUG"
   };
   char msg[512];
   va_list ap;

   (void) file;
   (void) line;
   va_start(ap, fmt);
   vsnprintf(msg, sizeof(msg), fmt, ap);
   va_end(ap);
   if (ll < 0 || ll > PLOG_DEBUG) ll = PLOG_DEBUG;
   Nlog[ll]++;
   if (ll == PLOG_ALERT) snprintf(LastAlert, sizeof(LastAlert), "%s", msg);
   if (ll == PLOG_WARN) snprintf(LastWarn, sizeof(LastWarn), "%s", msg);
   if (ll == PLOG_INFO) snprintf(LastInfo, sizeof(LastInfo), "%s", msg);
   if (ll <= PLOG_WARN || Verbose) {
      printf("   [peach.cu %s] %s\n", name[ll], msg);
      fflush(stdout);
   }
}  /* end peach_host_plogx() */

/**
 * Start a scenario section.
*/
static void section(const char *name)
{
   Section = name;
   SectionFails = 0;
   printf("\n== %s\n", name);
   fflush(stdout);
}  /* end section() */

/**
 * End a scenario section with its result (or "SKIP: reason").
*/
static void section_end(const char *skip)
{
   const char *res = skip ? skip : SectionFails ? "FAIL" : "PASS";

   printf("-- %s: %s\n", Section, res);
   if (Nresults < MAXRESULT) {
      Results[Nresults].name = Section;
      Results[Nresults].result = res;
      Nresults++;
   }
   fflush(stdout);
}  /* end section_end() */

/**
 * Small non-negative integer from the environment.
*/
static long env_long(const char *name, long def, long min, long max)
{
   const char *str = getenv(name);
   char *end;
   long v;

   if (str == NULL || *str == '\0') return def;
   v = strtol(str, &end, 10);
   if (*end != '\0' || v < min || v > max) return def;
   return v;
}  /* end env_long() */

/**
 * Set (value != NULL) or unset an environment variable.
*/
static void env_set(const char *name, const char *value)
{
   if (value) setenv(name, value, 1);
   else unsetenv(name);
}  /* end env_set() */

/* MCM_PEACH_STREAMS of the next peach_init_cuda_device() (NULL: unset) */
static const char *Streams;

/**
 * Solver configuration for the next peach_init_cuda_device().
*/
static void env_config(const char *legacy, const char *skip,
   const char *batch)
{
   env_set("MCM_PEACH_LEGACY", legacy);
   env_set("MCM_PEACH_SKIP", skip);
   env_set("MCM_PEACH_BATCH", batch);
   env_set("MCM_PEACH_STREAMS", Streams);
}  /* end env_config() */

/* deterministic test RNG (SplitMix64) */
static word64 Rs = WORD64_C(0x5DEECE66D1234567);

static word32 rnd32(void)
{
   word64 z = (Rs += WORD64_C(0x9e3779b97f4a7c15));
   z = (z ^ (z >> 30)) * WORD64_C(0xbf58476d1ce4e5b9);
   z = (z ^ (z >> 27)) * WORD64_C(0x94d049bb133111eb);
   return (word32) ((z ^ (z >> 31)) >> 16);
}  /* end rnd32() */

/****************************************************************
 * MAP CACHE AND KERNEL REDIRECTION
 ****************************************************************/

/* a cached map (one phash) */
typedef struct {
   word8 phash[HASHLEN];   /* map seed */
   word8 *map;             /* PEACHMAPLEN bytes */
   word8 *have;            /* tile present, PEACHCACHELEN flags */
   word32 ntiles;          /* tiles present */
   unsigned long stamp;    /* last use (LRU) */
} MAPCACHE;

static MAPCACHE Cache[MAXCACHE];
static int Ncache = MAXCACHE;
static unsigned long Stamp;

/* map build statistics */
static struct {
   word64 computed;        /* tiles computed by the real kernel */
   word64 copied;          /* tiles copied from the cache */
   unsigned long launches; /* kcu_peach_build launches */
   unsigned long tlaunches;   /* kcu_peach_pipe_transitions launches */
} Bstat;

/* arguments of the redirected launches, for the worker threads */
static struct {
   word32 offset, count;
   word64 *d_map;
   word32 *d_phash;
   const uint4 *map;
   word16 *T;
} Karg;

/* map build and transition table order (see bmap_chunk() and
 * check_transitions_launch()): which tiles of which phash the map build
 * launched since it (re)started, and the map the transition table was
 * last built from */
static struct {
   word8 phash[HASHLEN];   /* phash of the chunks launched */
   const void *d_map;      /* map the chunks were launched for */
   word8 *built;           /* tile launched (PEACHCACHELEN flags) */
   word32 nbuilt;          /* tiles launched */
   unsigned long serial;   /* launch serial of the last chunk */
   word8 t_phash[HASHLEN]; /* phash of the (complete) map of T */
   unsigned long t_serial; /* launch serial of the last T build */
   cudaStream_t t_stream;  /* stream of the last T build */
   unsigned long t_seq;    /* stream operation number of that build */
} Bmap;

/* pipeline batch launch checks (check_batch_launch()): epoch of the last
 * batch of the current device context, batches checked */
static struct {
   word32 epoch;
   unsigned long batches;
} Bchk;

/* checks of the transition table launch (after the miner emulation) */
static void check_transitions_launch(const uint4 *d_map, word16 *d_T,
   word32 offset, word32 count);

/**
 * Fake runtime stream of handle @a h (NULL: the legacy default stream).
 * @returns the stream, or NULL for an unknown handle
*/
static EMU_RT__STREAM *emu_stream(cudaStream_t h)
{
   size_t i;

   if (h == NULL) return &emu_rt.null_stream;
   for (i = 0; i < emu_rt.nstreams; i++) {
      if ((void *) emu_rt.streams[i] == h) return emu_rt.streams[i];
   }
   return NULL;
}  /* end emu_stream() */

/**
 * Clear the last error of every emulated host thread (a test helper, so
 * that one scenario's errors do not leak into the next), then select
 * the main thread.
*/
static void emu_clear_errors(void)
{
   int t;

   for (t = 0; t < EMU_RT_MAXTHREAD; t++) {
      emu_rt_thread(t);
      (void) cudaGetLastError();
   }
   emu_rt_thread(0);
}  /* end emu_clear_errors() */

/**
 * Track a kcu_peach_build() launch: tiles offset .. offset + n - 1 of the
 * map of phash *d_phash. A chunk at offset 0, another phash or another
 * map restarts the tracking (a new map build).
*/
static void bmap_chunk(word32 offset, word64 n, const word64 *d_map,
   const word32 *d_phash)
{
   word64 t, end;

   if (Bmap.built == NULL) {
      Bmap.built = (word8 *) calloc(PEACHCACHELEN, 1);
      if (Bmap.built == NULL) return;
   }
   if (offset == 0 || Bmap.d_map != (const void *) d_map ||
         memcmp(Bmap.phash, d_phash, HASHLEN) != 0) {
      memcpy(Bmap.phash, d_phash, HASHLEN);
      Bmap.d_map = (const void *) d_map;
      memset(Bmap.built, 0, PEACHCACHELEN);
      Bmap.nbuilt = 0;
   }
   Bmap.serial = emu_rt_launch_serial();
   end = (word64) offset + n;
   if (end > PEACHCACHELEN) end = PEACHCACHELEN;
   for (t = offset; t < end; t++) {
      if (!Bmap.built[t]) Bmap.nbuilt++;
      Bmap.built[t] = 1;
   }
}  /* end bmap_chunk() */

/**
 * Find the cache entry of @a phash (or NULL).
*/
static MAPCACHE *cache_find(const void *phash)
{
   int i;

   for (i = 0; i < Ncache; i++) {
      if (Cache[i].map && memcmp(Cache[i].phash, phash, HASHLEN) == 0) {
         Cache[i].stamp = ++Stamp;
         return &Cache[i];
      }
   }
   return NULL;
}  /* end cache_find() */

/**
 * New (empty) cache entry for @a phash, evicting the least recently used
 * one; NULL when caching is disabled or memory is short.
*/
static MAPCACHE *cache_new(const void *phash)
{
   MAPCACHE *c = NULL;
   int i;

   for (i = 0; i < Ncache; i++) {
      if (c == NULL || Cache[i].map == NULL ||
            (c->map && Cache[i].stamp < c->stamp)) c = &Cache[i];
      if (c->map == NULL) break;
   }
   if (c == NULL) return NULL;
   if (c->map == NULL) {
      c->map = (word8 *) malloc(PEACHMAPLEN);
      c->have = (word8 *) malloc(PEACHCACHELEN);
      if (c->map == NULL || c->have == NULL) {
         free(c->map);
         free(c->have);
         c->map = c->have = NULL;
         return NULL;
      }
   }
   memcpy(c->phash, phash, HASHLEN);
   memset(c->have, 0, PEACHCACHELEN);
   c->ntiles = 0;
   c->stamp = ++Stamp;
   return c;
}  /* end cache_new() */

/**
 * Worker thread body: one thread of kcu_peach_build().
*/
static void build_thread(void *user)
{
   (void) user;
   kcu_peach_build(Karg.offset, Karg.d_map, Karg.d_phash);
}  /* end build_thread() */

/**
 * Worker thread body: one thread of kcu_peach_pipe_transitions().
*/
static void transitions_thread(void *user)
{
   (void) user;
   kcu_peach_pipe_transitions(Karg.map, Karg.T, Karg.offset, Karg.count);
}  /* end transitions_thread() */

/**
 * Redirected kcu_peach_build() launch (runs for its first emulated
 * thread and handles the whole launch): tiles offset .. offset + grid x
 * block - 1 of the map of phash *d_phash are copied from the cache when
 * present, else computed by the real kernel in worker processes (and
 * cached).
*/
static void peach_host_build(word32 offset, word64 *d_map,
   word32 *d_phash)
{
   MAPCACHE *c;
   word64 n, end, t;

   emu_rt_kernel_done();
   Bstat.launches++;
   n = (word64) gridDim.x * blockDim.x;
   bmap_chunk(offset, n, d_map, d_phash);
   if (offset >= PEACHCACHELEN) return;
   end = (word64) offset + n;
   if (end > PEACHCACHELEN) end = PEACHCACHELEN;
   c = cache_find(d_phash);
   if (c) {
      for (t = offset; t < end && c->have[t]; t++);
      if (t == end) {
         memcpy((word8 *) d_map + ((size_t) offset * PEACHTILELEN),
            c->map + ((size_t) offset * PEACHTILELEN),
            (size_t) (end - offset) * PEACHTILELEN);
         Bstat.copied += end - offset;
         return;
      }
   }
   Karg.offset = offset;
   Karg.d_map = d_map;
   Karg.d_phash = d_phash;
   if (emu_rt_run_parallel(build_thread, NULL) != 0) return;
   Bstat.computed += end - offset;
   if (c == NULL) c = cache_new(d_phash);
   if (c == NULL) return;
   memcpy(c->map + ((size_t) offset * PEACHTILELEN),
      (word8 *) d_map + ((size_t) offset * PEACHTILELEN),
      (size_t) (end - offset) * PEACHTILELEN);
   for (t = offset; t < end; t++) {
      if (!c->have[t]) c->ntiles++;
      c->have[t] = 1;
   }
}  /* end peach_host_build() */

/**
 * Redirected kcu_peach_pipe_transitions() launch: all threads of the
 * real kernel, in worker processes.
*/
static void peach_host_transitions(const uint4 *d_map, word16 *d_T,
   word32 offset, word32 count)
{
   emu_rt_kernel_done();
   Bstat.tlaunches++;
   check_transitions_launch(d_map, d_T, offset, count);
   Karg.map = d_map;
   Karg.T = d_T;
   Karg.offset = offset;
   Karg.count = count;
   emu_rt_run_parallel(transitions_thread, NULL);
}  /* end peach_host_transitions() */

/****************************************************************
 * MINER EMULATION (src/bin/gpuminer.c device loop)
 ****************************************************************/

/* Block 0x1 trailer data taken directly from the Mochimo Blockchain Tfile
 * (as src/test/peach-mining-cu.c): a solved trailer, tcount = 1 */
static const word8 Block1[sizeof(BTRAILER)] = {
   0x00, 0x17, 0x0c, 0x67, 0x11, 0xb9, 0xdc, 0x3c, 0xa7, 0x46,
   0xc4, 0x6c, 0xc2, 0x81, 0xbc, 0x69, 0xe3, 0x03, 0xdf, 0xad,
   0x2f, 0x33, 0x3b, 0xa3, 0x97, 0xba, 0x06, 0x1e, 0xcc, 0xef,
   0xde, 0x03, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
   0xf4, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00,
   0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
   0xf7, 0x2d, 0x1f, 0xae, 0xa8, 0x7f, 0x5b, 0x8f, 0x3c, 0xa9,
   0xce, 0x6c, 0xdd, 0x5a, 0xe6, 0xf1, 0xb0, 0x81, 0xe5, 0x70,
   0xc1, 0xf8, 0xe9, 0x63, 0x90, 0xb1, 0x25, 0x38, 0x8e, 0x48,
   0x46, 0x73, 0x10, 0xf9, 0x01, 0x05, 0xf1, 0x01, 0x26, 0x00,
   0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x56, 0xdf,
   0x01, 0x11, 0x05, 0x4b, 0xb7, 0x03, 0x01, 0x56, 0x00, 0x00,
   0x00, 0x00, 0x00, 0x00, 0xb1, 0x0d, 0x31, 0x5b, 0x78, 0x49,
   0x1f, 0x37, 0xaa, 0xa7, 0x54, 0xef, 0x7d, 0xb8, 0x1a, 0x96,
   0x42, 0xd4, 0xba, 0x1c, 0xf7, 0x2f, 0x6e, 0x37, 0xff, 0x92,
   0x99, 0x9a, 0xa0, 0x32, 0x55, 0x51, 0xbc, 0xf1, 0x5f, 0x69
};

/* phashes used by the scenarios (Block1's phash with byte 31 ^ id) */
#define PHASH_A   0
#define PHASH_B   0x5a

/* the emulated miner: one device, the shared trailer and solve output */
static struct {
   DEVICE_CTX dev;               /* the device */
   BTRAILER bt;                  /* BT_curr (the network thread's) */
   BTRAILER btout;               /* bt_solve (shared by all devices) */
   BTRAILER family[MAXFAMILY];   /* trailers presented for the block */
   int nfamily;
   word64 solves;                /* reported solves, all checks passed */
   word64 bad;                   /* reported solves failing a check */
   word64 prev_candidate;        /* solves for an earlier candidate of
                                    the same block (allowed) */
   BTRAILER last;                /* the last verified solve */
   word64 polls;                 /* polls */
   int pause;                    /* gpuminer: no solving while expired
                                    or without transactions */
   int keep;                     /* clear btout after a solve (keep
                                    solving the same block) */
   int thread;                   /* emulated host thread of the polls */
} R;

#define PCTX   ((PEACH_CUDA_CTX *) R.dev.peach)

static int pipeline_active(void);

/**
 * Pipeline batch context of stream @a s (P->stream[0 .. nctx - 1]).
 * @returns the context number, or (-1) for another stream
*/
static int ctx_of_stream(const PEACH_CUDA_CTX *P, cudaStream_t s)
{
   int id;

   for (id = 0; id < P->nctx; id++) {
      if (P->stream[id] == s) return id;
   }
   return (-1);
}  /* end ctx_of_stream() */

/**
 * Make a fresh candidate trailer of block @a bnum on phash variant
 * @a phash_id at difficulty @a diff (time0 = now, tcount = 1).
*/
static void trailer_make(BTRAILER *bt, int phash_id, word32 bnum,
   word8 diff)
{
   memcpy(bt, Block1, sizeof(*bt));
   bt->phash[31] ^= (word8) phash_id;
   memset(bt->bnum, 0, sizeof(bt->bnum));
   put32(bt->bnum, bnum);
   put32(bt->tcount, 1);
   put32(bt->time0, (word32) emu_rt_time(NULL));
   memset(bt->difficulty, 0, sizeof(bt->difficulty));
   bt->difficulty[0] = diff;
   bt->mroot[0] ^= (word8) bnum;
   memset(bt->nonce, 0, sizeof(bt->nonce));
   memset(bt->stime, 0, sizeof(bt->stime));
   memset(bt->bhash, 0, sizeof(bt->bhash));
}  /* end trailer_make() */

/**
 * The network thread presents R.bt (a new block, or a new candidate of
 * the same block): remember it as a valid solve target.
*/
static void rig_present(void)
{
   if (R.nfamily > 0 && (memcmp(R.family[0].phash, R.bt.phash, HASHLEN) ||
         memcmp(R.family[0].bnum, R.bt.bnum, sizeof(R.bt.bnum)))) {
      R.nfamily = 0;
   }
   if (R.nfamily == MAXFAMILY) {
      memmove(&R.family[1], &R.family[2],
         sizeof(BTRAILER) * (MAXFAMILY - 2));
      R.nfamily--;
   }
   R.family[R.nfamily++] = R.bt;
}  /* end rig_present() */

/**
 * New block: fresh trailer (rig_present() included).
*/
static void rig_block(int phash_id, word32 bnum, word8 diff)
{
   trailer_make(&R.bt, phash_id, bnum, diff);
   rig_present();
}  /* end rig_block() */

/**
 * New candidate of the current block: different mroot, fresh time0,
 * difficulty @a diff.
*/
static void rig_candidate(word8 diff)
{
   R.bt.mroot[1]++;
   put32(R.bt.time0, (word32) emu_rt_time(NULL));
   R.bt.difficulty[0] = diff;
   rig_present();
}  /* end rig_candidate() */

/**
 * A device reported a solve (VEOK): check it as gpuminer does
 * (peach_check()), and that it belongs to the current block and to a
 * trailer presented for it.
*/
static void rig_solved(void)
{
   const BTRAILER *s = &R.btout;
   int i, ok = 0;

   if (peach_check(s) != VEOK) {
      CHECK(0, "reported solve fails peach_check() (bnum %u)",
         (unsigned) get32(s->bnum));
   } else if (memcmp(s->phash, R.bt.phash, HASHLEN) != 0 ||
         memcmp(s->bnum, R.bt.bnum, sizeof(s->bnum)) != 0) {
      CHECK(0, "reported solve is for an OLD block (bnum %u, current %u)",
         (unsigned) get32(s->bnum), (unsigned) get32(R.bt.bnum));
   } else {
      for (i = 0; i < R.nfamily && !ok; i++) {
         ok = memcmp(s, &R.family[i], 92) == 0;
      }
      CHECK(ok, "reported solve is for a trailer never presented");
   }
   if (ok) {
      R.solves++;
      R.last = *s;
      if (memcmp(s, &R.bt, 92) != 0) R.prev_candidate++;
      /* gpuminer sends a solve only if it matches its current or its
       * previous trailer (BT_curr, BT_prev; presented in this order),
       * else it pauses for the rest of the block: the pipeline solver
       * must report nothing else */
      if (memcmp(s, &R.bt, 92) != 0 && (R.nfamily < 2 ||
            memcmp(s, &R.family[R.nfamily - 2], 92) != 0) &&
            pipeline_active()) {
         CHECK(0, "pipeline reported a solve for a trailer gpuminer cannot"
            " send (neither its current nor its previous one)");
      }
   } else R.bad++;
   if (R.keep) memset(&R.btout, 0, sizeof(R.btout));
}  /* end rig_solved() */

/**
 * One iteration of gpuminer's device loop: sleep (fake clock), pause
 * check, peach_solve_cuda(), check a reported solve.
 * @returns the solver's return value (VERROR while paused)
*/
static int rig_poll(void)
{
   int ecode;

   emu_rt_sleep(PollMs);
   R.polls++;
   if (R.pause && (get32(R.bt.tcount) == 0 || difftime(emu_rt_time(NULL),
         get32(R.bt.time0)) >= BRIDGEv3)) return VERROR;
   emu_rt_thread(R.thread);
   ecode = peach_solve_cuda(&R.dev, &R.bt, 0, &R.btout);
   emu_rt_thread(MAIN_THREAD);
   if (ecode == VEOK) rig_solved();
   return ecode;
}  /* end rig_poll() */

/**
 * Poll until @a pred() holds (checked after each poll), at most @a max
 * polls. @returns the polls used, or -1
*/
static int rig_until(int (*pred)(void), int max)
{
   int n;

   for (n = 1; n <= max; n++) {
      rig_poll();
      if (pred()) return n;
      if (R.dev.status < DEV_NULL) return (-1);
   }
   return (-1);
}  /* end rig_until() */

/* predicates */
static word64 Target;
static int pred_solved(void) { return R.solves >= Target; }
static int pred_ready(void)
{
   return R.dev.status == DEV_WORK && PCTX != NULL &&
      memcmp(PCTX->map_phash, R.bt.phash, HASHLEN) == 0;
}

/**
 * Was the device initialized for the pipeline solver?
*/
static int pipeline_mode(void)
{
   return PCTX != NULL && PCTX->mode == PEACH_CUDA_MODE_PIPELINE;
}  /* end pipeline_mode() */

/**
 * Does the device (still) use the pipeline solver (no fallback)?
*/
static int pipeline_active(void)
{
   int active = pipeline_mode() && !PCTX->fallback;
   int api = peach_pipeline_cuda_device(&R.dev);

   CHECK(api == (R.dev.peach ? active : -1), "peach_pipeline_cuda_device()"
      " = %d, expected %d", api, R.dev.peach ? active : -1);
   return active;
}  /* end pipeline_active() */

/**
 * Check the device's map (and transition table in pipeline mode) is the
 * one of the current phash: sampled tiles vs peach_generate(), every
 * tile vs the map cache (every chunk was built).
*/
static void check_map(void)
{
   PEACH_CUDA_CTX *P = PCTX;
   static word8 ref[PEACHTILELEN];
   const word8 *map;
   MAPCACHE *c;
   word32 idx, i, bad;

   if (P == NULL || P->d_map == NULL) {
      CHECK(0, "no map");
      return;
   }
   map = (const word8 *) P->d_map;
   CHECK(memcmp(P->map_phash, R.bt.phash, HASHLEN) == 0,
      "map_phash is not the current phash");
   for (i = bad = 0; i < 12; i++) {
      idx = i < 4 ? (i < 3 ? i : PEACHCACHELEN_M1) :
         rnd32() & PEACHCACHELEN_M1;
      peach_generate(idx, R.bt.phash, ref);
      if (memcmp(map + ((size_t) idx * PEACHTILELEN), ref, PEACHTILELEN)) {
         if (bad++ < 3) CHECK(0, "map tile %u differs from peach_generate()",
            (unsigned) idx);
      }
   }
   c = cache_find(R.bt.phash);
   if (c != NULL) {
      CHECK(c->ntiles == PEACHCACHELEN, "map build launched only %u of %u"
         " tiles for this phash", (unsigned) c->ntiles, PEACHCACHELEN);
      CHECK(memcmp(map, c->map, PEACHMAPLEN) == 0,
         "map differs from the map of this phash");
   } else if (Ncache > 0) CHECK(0, "map build launched no tiles");
   if (pipeline_active()) {
      CHECK(P->d_T != NULL, "no transition table");
      if (P->d_T == NULL) return;
      for (i = bad = 0; i < NTSAMPLE; i++) {
         idx = i < 3 ? (i < 2 ? i : PEACHCACHELEN_M1) :
            rnd32() & PEACHCACHELEN_M1;
         if (P->d_T[idx] != peach_transition_tile((const word32 *)
               (map + ((size_t) idx * PEACHTILELEN)), idx)) {
            if (bad++ < 3) CHECK(0, "T[%u] differs from"
               " peach_transition_tile()", (unsigned) idx);
         }
      }
   }
}  /* end check_map() */

/**
 * Poll until the map of the current phash is built and the device works
 * (DEV_WORK), then check the map.
 * @returns polls used, or -1
*/
static int rig_build(void)
{
   int n = rig_until(pred_ready, BUILD_POLLS);

   CHECK(n >= 0, "no map build / DEV_WORK within %d polls (status %d)",
      BUILD_POLLS, R.dev.status);
   if (n >= 0) check_map();
   return n;
}  /* end rig_build() */

/**
 * Poll until @a n more solves were reported (and verified).
 * @returns polls used, or -1
*/
static int rig_solve(int n)
{
   int polls;

   Target = R.solves + (word64) n;
   polls = rig_until(pred_solved, SOLVE_POLLS * n);
   CHECK(polls >= 0, "%d solve(s) not found within %d polls (status %d,"
      " %llu solves)", n, SOLVE_POLLS * n, R.dev.status,
      (unsigned long long) R.solves);
   return polls;
}  /* end rig_solve() */

/**
 * peach_init_cuda_device() of the emulated device.
 * @returns its return value
*/
static int dev_init(void)
{
   int rc;

   R.dev.id = 0;
   R.dev.type = CUDA_DEVICE;
   snprintf(R.dev.info, sizeof(R.dev.info), "%s", emu_rt.dev.name);
   emu_rt_thread(MAIN_THREAD);
   Bchk.epoch = 0;   /* a new context starts its own epochs */
   rc = peach_init_cuda_device(&R.dev);
   return rc;
}  /* end dev_init() */

/**
 * peach_free_cuda_device(), then check that nothing is left: no device
 * or pinned memory, streams or events; freeing again is harmless.
*/
static void dev_free(void)
{
   int rc;

   emu_rt_thread(MAIN_THREAD);
   rc = peach_free_cuda_device(&R.dev);
   CHECK(rc == VEOK, "peach_free_cuda_device() = %d", rc);
   CHECK(R.dev.peach == NULL && R.dev.status == DEV_NULL,
      "device context not released (status %d)", R.dev.status);
   CHECK(emu_rt_live(0, NULL, 1) == 0, "memory leaked (listed above)");
   CHECK(emu_rt_live_handles() == 0, "%lu streams/events leaked",
      emu_rt_live_handles());
   rc = peach_free_cuda_device(&R.dev);
   CHECK(rc == VEOK, "second peach_free_cuda_device() = %d", rc);
   rc = peach_pipeline_cuda_device(&R.dev);
   CHECK(rc == -1, "peach_pipeline_cuda_device() = %d after free", rc);
}  /* end dev_free() */

/**
 * Initialize the device in the given configuration and expect success.
 * @returns 0 on success, else -1
*/
static int dev_start(const char *legacy, const char *skip,
   const char *batch)
{
   int rc;

   env_config(legacy, skip, batch);
   rc = dev_init();
   CHECK(rc == VEOK, "peach_init_cuda_device() = %d", rc);
   CHECK(R.dev.status == DEV_INIT, "status %d after init", R.dev.status);
   return rc == VEOK ? 0 : (-1);
}  /* end dev_start() */

/****************************************************************
 * FAULT INJECTION AND ACCOUNTING HOOKS (after kernel launches)
 ****************************************************************/

static struct {
   int corrupt_T;          /* after transitions: corrupt every T entry */
   int forge;              /* after final: forge a found result */
   int overflow;           /* after final: report a queue overflow */
   int stale;              /* after final: a captured older result */
   int capture;            /* after final: capture a found result */
   int injected;           /* faults injected */
   PEACH_PIPE_RESULT captured;   /* captured result */
   int have_captured;
   int track;              /* sum up completed nonces */
   word64 fin_sum;         /* completed, finished batches (tracked) */
   word64 last[PEACH_CUDA_NCTX_MAX];   /* completed of each context's
                                          last batch */
   unsigned long fin;      /* finished batches (tracked) */
   word32 contend;         /* batch time model (contend_batch()): slots
                              of a slow batch, 0 = off */
} Fx;

/**
 * Corrupt every transition table entry.
*/
static void corrupt_T(void)
{
   PEACH_CUDA_CTX *P = PCTX;
   word32 t;

   if (P == NULL || P->d_T == NULL) return;
   for (t = 0; t < PEACHCACHELEN; t++) P->d_T[t] ^= 0x5555;
   Fx.injected++;
}  /* end corrupt_T() */

/**
 * Checks of the transition table launch (called by its redirection,
 * before it runs): the map is complete for the phash recorded by the
 * solver, every other stream (map chunks on streams 0 and 1, batches on
 * the stream of every batch context) is idle, the launch covers the
 * whole table of the device. Records the map of the table.
*/
static void check_transitions_launch(const uint4 *d_map, word16 *d_T,
   word32 offset, word32 count)
{
   PEACH_CUDA_CTX *P = PCTX;
   EMU_RT__STREAM *ls = emu_rt.launch.s;
   cudaStream_t s = ls == &emu_rt.null_stream ? NULL : (cudaStream_t) ls;
   int i;

   Bmap.t_serial = emu_rt_launch_serial();
   Bmap.t_stream = s;
   Bmap.t_seq = ~0UL;   /* set when enqueued (on_launch()) */
   memcpy(Bmap.t_phash, Bmap.phash, HASHLEN);
   if (Bmap.nbuilt != PEACHCACHELEN) {
      CHECK(0, "transition table launched for an incomplete map (%u of %u"
         " tiles launched)", (unsigned) Bmap.nbuilt, PEACHCACHELEN);
      memset(Bmap.t_phash, 0, HASHLEN);
   }
   CHECK(offset == 0 && count >= PEACHCACHELEN, "transition table launch"
      " covers %u tiles from %u only", (unsigned) count, (unsigned) offset);
   if (P == NULL) return;
   CHECK((const void *) d_map == (const void *) P->d_map && d_T == P->d_T &&
      Bmap.d_map == (const void *) P->d_map, "transition table launch on"
      " other buffers than the device's map and table");
   CHECK(memcmp(Bmap.phash, P->map_phash, HASHLEN) == 0, "transition table"
      " launched for a map of another phash than map_phash");
   for (i = 0; i < (P->nctx > 2 ? P->nctx : 2); i++) {
      if (P->stream[i] == s) continue;
      CHECK(emu_rt_stream_busy(P->stream[i]) == 0, "transition table"
         " launched while stream %d may still build the map or run a"
         " batch", i);
   }
}  /* end check_transitions_launch() */

/**
 * Checks of a pipeline batch launch (after its init kernel was enqueued
 * on stream @a s): the batch belongs to the current trailer (as presented
 * on entry of peach_solve_cuda()) and to the phash of the map, the
 * transition table was built from that complete map after its last
 * chunk and has finished, and the batch parameters derive from the
 * batch's own trailer snapshot (h_bt[id]): clamped difficulty,
 * first nonce half (valid haiku, no NaN replacement, q), midstate, tail,
 * skip masks, slots, a new epoch; no unharvested batch is overwritten.
*/
static void check_batch_launch(cudaStream_t s)
{
   PEACH_CUDA_CTX *P = PCTX;
   const PEACH_PIPE_PARAMS *p;
   const EMU_RT__STREAM *ts;
   const BTRAILER *hb;
   SHA256_CTX sctx;
   word32 nlo[4], q;
   int id, nanf;

   id = ctx_of_stream(P, s);
   CHECK(id >= 0, "pipeline batch launched on an unknown stream");
   if (id < 0) return;
   Bchk.batches++;
   p = &(P->params[id]);
   hb = P->h_bt[id];
   CHECK(memcmp(hb, &R.bt, 92) == 0, "batch %d launched for a trailer other"
      " than the current one (bnum %u, current %u)", id,
      (unsigned) get32(hb->bnum), (unsigned) get32(R.bt.bnum));
   CHECK(memcmp(hb->phash, P->map_phash, HASHLEN) == 0, "batch %d launched"
      " for a phash other than the map's", id);
   CHECK(Bmap.t_serial > Bmap.serial &&
      memcmp(Bmap.t_phash, hb->phash, HASHLEN) == 0, "batch %d launched"
      " without a transition table built from the complete current map",
      id);
   ts = emu_stream(Bmap.t_stream);
   CHECK(Bmap.t_stream == s || (ts != NULL && (ts->done_seq >= Bmap.t_seq ||
      !emu_rt__busy(ts))), "batch %d launched while the transition table"
      " may still be built (on the other stream)", id);
   /* gpuminer passes diff 0: the clamped difficulty is difficulty[0] */
   CHECK(p->diff == (word32) hb->difficulty[0], "batch %d difficulty %u,"
      " trailer difficulty %u (expected equal)", id, (unsigned) p->diff,
      (unsigned) hb->difficulty[0]);
   memcpy(nlo, hb->nonce, sizeof(nlo));
   CHECK(memcmp(p->nonce_lo, nlo, sizeof(nlo)) == 0, "batch %d nonce_lo is"
      " not the first nonce half of its trailer", id);
   CHECK(trigg_syntax(nlo) == VEOK, "batch %d first nonce half fails"
      " trigg_syntax()", id);
   nanf = 0;
   q = peach_prefix_words(nlo, 4, 0, 0, &nanf);
   CHECK(nanf == 0 && q == p->q, "batch %d first nonce half: NaN"
      " replacement %d, q %u (params %u)", id, nanf, (unsigned) q,
      (unsigned) p->q);
   sha256_init(&sctx);
   sha256_update(&sctx, hb, 64);
   CHECK(memcmp(p->mid, sctx.state, sizeof(p->mid)) == 0, "batch %d"
      " midstate is not the one of its trailer", id);
   CHECK(memcmp(p->tail, (const word8 *) hb + 64, sizeof(p->tail)) == 0,
      "batch %d tail is not bytes 64..91 of its trailer", id);
   CHECK(p->skip == peach_pipe_skip_pack(P->cfg_skip), "batch %d skip"
      " masks 0x%016llx differ from the configuration", id,
      (unsigned long long) p->skip);
   CHECK(p->nslots >= P->nslots_min && p->nslots <= P->cap &&
      p->nslots % PEACH_PIPE_BLOCK == 0 && p->nslots == P->nslots,
      "batch %d slots %u (min %u, cap %u, current %u)", id,
      (unsigned) p->nslots, (unsigned) P->nslots_min, (unsigned) P->cap,
      (unsigned) P->nslots);
   CHECK(p->epoch != 0 && p->epoch > Bchk.epoch, "batch %d epoch %u is not"
      " new (previous batch %u)", id, (unsigned) p->epoch,
      (unsigned) Bchk.epoch);
   Bchk.epoch = p->epoch;
   CHECK(!P->inflight[id], "batch %d launched over an unharvested batch",
      id);
}  /* end check_batch_launch() */

/**
 * Batch time model of the adaptive batch size check with batches of two
 * sizes in flight (Fx.contend, called after the init kernel of a batch
 * on stream @a s was enqueued): a batch of Fx.contend slots or more
 * takes 400 ms (slow); a smaller batch takes 300 ms (slow) if another
 * stream is still busy with a larger batch, which shares the GPU with
 * it, else 150 ms (neither slow nor fast enough to grow). Sets the time
 * of the final kernel, which the same peach_pipe_enqueue() call enqueues
 * next on @a s.
*/
static void contend_batch(cudaStream_t s)
{
   PEACH_CUDA_CTX *P = PCTX;
   const EMU_RT__STREAM *o;
   double ms;
   int id, i;

   id = ctx_of_stream(P, s);
   if (id < 0) return;
   ms = P->params[id].nslots >= Fx.contend ? 400.0 : 150.0;
   for (i = 0; i < P->nctx && ms < 300.0; i++) {
      o = emu_stream(P->stream[i]);
      if (i != id && o != NULL && emu_rt__busy(o) &&
            P->params[i].nslots > P->params[id].nslots) ms = 300.0;
   }
   emu_rt_kcfg("kcu_peach_pipe_final", 1)->ms = ms;
}  /* end contend_batch() */

/**
 * Post-launch hook of the fake runtime.
*/
static void on_launch(const char *name, cudaStream_t s, void *user)
{
   PEACH_CUDA_CTX *P = PCTX;
   PEACH_PIPE_RESULT *res;
   word32 epoch;
   int id, i;

   (void) user;
   if (P == NULL) return;
   if (strcmp(name, "kcu_peach_pipe_init") == 0) {
      check_batch_launch(s);
      if (Fx.contend) contend_batch(s);
      return;
   }
   if (strcmp(name, "kcu_peach_pipe_transitions") == 0 &&
         emu_stream(s) != NULL) {
      Bmap.t_seq = emu_stream(s)->last_seq;
   }
   if (strcmp(name, "kcu_peach_pipe_transitions") == 0 && Fx.corrupt_T) {
      Fx.corrupt_T = 0;
      corrupt_T();
      return;
   }
   if (strcmp(name, "kcu_peach_pipe_final") != 0) return;
   id = ctx_of_stream(P, s);
   if (id < 0 || P->bufs[id].d_res == NULL) return;
   res = P->bufs[id].d_res;
   Fx.last[id] = res->completed;
   if (Fx.track) {
      Fx.fin_sum += res->completed;
      Fx.fin++;
   }
   if (Fx.capture && res->found && !Fx.have_captured) {
      Fx.captured = *res;
      Fx.have_captured = 1;
   }
   if (Fx.forge) {
      Fx.forge--;
      res->found = 1;
      for (i = 0; i < 4; i++) res->nonce_hi[i] = rnd32();
      for (i = 0; i < 8; i++) res->hash[i] = rnd32() & ~0xFFU;
      Fx.injected++;
   }
   if (Fx.overflow) {
      Fx.overflow--;
      res->overflow = 1;
      Fx.injected++;
   }
   if (Fx.stale && Fx.have_captured) {
      Fx.stale--;
      epoch = res->epoch;
      *res = Fx.captured;
      if (res->epoch == epoch) res->epoch = epoch ^ WORD32_C(0x80000000);
      Fx.injected++;
   }
}  /* end on_launch() */

/****************************************************************
 * SCENARIOS
 ****************************************************************/

/**
 * (0) peach_checkhash_cuda() (legacy checkhash kernel) vs the CPU.
*/
static void scenario_checkhash(void)
{
   BTRAILER bt[2];
   word8 out[2 * SHA256LEN], ref[SHA256LEN];
   int rc, expect;

   section("S0 peach_checkhash_cuda vs CPU");
   memcpy(&bt[0], Block1, sizeof(BTRAILER));
   memcpy(&bt[1], Block1, sizeof(BTRAILER));
   bt[1].mroot[0] ^= 1;    /* another trailer, same (valid) haiku */
   expect = 0;
   CHECK(peach_checkhash(&bt[0], bt[0].difficulty[0], ref) == VEOK,
      "block 1 fails peach_checkhash()");
   rc = peach_checkhash_cuda(2, bt, out);
   CHECK(memcmp(out, ref, SHA256LEN) == 0, "final hash 0 differs");
   if (peach_checkhash(&bt[1], bt[1].difficulty[0], ref) != VEOK) {
      expect = 1;
   }
   CHECK(memcmp(out + SHA256LEN, ref, SHA256LEN) == 0,
      "final hash 1 differs");
   CHECK(rc == expect, "peach_checkhash_cuda() = %d, expected %d", rc,
      expect);
   CHECK(emu_rt_live(0, NULL, 1) == 0, "memory leaked");
   section_end(NULL);
}  /* end scenario_checkhash() */

/**
 * (4) DEV_IDLE conditions: bnum solved by another device, tcount 0,
 * BRIDGEv3 expiry. Batches with solves are in flight when a condition
 * appears: nothing is reported or launched while it holds; solving
 * resumes when it clears. Requires a working device.
*/
static void scenario_idle(void)
{
   BTRAILER other;
   unsigned long launches;
   int i, rc, reported;

   /* every batch has a solve; make sure batches are in flight */
   rig_candidate(DIFF_ALWAYS);
   rig_solve(1);
   rig_poll();
   R.pause = 0;   /* call the solver regardless (another device, races) */

   /* (a) another device solved this block: btout holds its trailer */
   R.keep = 0;
   other = R.bt;
   memset(other.nonce, 0xAA, sizeof(other.nonce));
   R.btout = other;
   for (i = reported = 0; i < 20; i++) reported += rig_poll() == VEOK;
   CHECK(reported == 0, "(bnum solved) %d solve(s) reported", reported);
   CHECK(memcmp(&R.btout, &other, sizeof(other)) == 0,
      "(bnum solved) btout was overwritten");
   CHECK(R.dev.status == DEV_IDLE, "(bnum solved) status %d, not IDLE",
      R.dev.status);
   launches = emu_rt.launches;
   for (i = 0; i < 10; i++) rig_poll();
   CHECK(emu_rt.launches == launches, "(bnum solved) %lu launches while"
      " idle", emu_rt.launches - launches);
   memset(&R.btout, 0, sizeof(R.btout));
   R.keep = 1;
   rig_solve(1);
   CHECK(R.dev.status == DEV_WORK, "(bnum solved) no resume, status %d",
      R.dev.status);

   /* (b) no transactions (tcount 0) */
   put32(R.bt.tcount, 0);
   rig_present();
   rig_poll();
   for (i = reported = 0; i < 20; i++) reported += rig_poll() == VEOK;
   CHECK(reported == 0, "(tcount 0) %d solve(s) reported", reported);
   CHECK(R.dev.status == DEV_IDLE, "(tcount 0) status %d, not IDLE",
      R.dev.status);
   launches = emu_rt.launches;
   for (i = 0; i < 10; i++) rig_poll();
   CHECK(emu_rt.launches == launches, "(tcount 0) %lu launches while idle",
      emu_rt.launches - launches);
   put32(R.bt.tcount, 1);
   rig_candidate(DIFF_ALWAYS);
   rig_solve(1);
   CHECK(R.dev.status == DEV_WORK, "(tcount 0) no resume, status %d",
      R.dev.status);

   /* (c) expiry (BRIDGEv3 seconds after time0) */
   rig_poll();
   emu_rt_advance((BRIDGEv3 + 1) * 1000.0);
   for (i = reported = 0; i < 20; i++) reported += rig_poll() == VEOK;
   CHECK(reported == 0, "(expired) %d solve(s) reported", reported);
   CHECK(R.dev.status == DEV_IDLE, "(expired) status %d, not IDLE",
      R.dev.status);
   launches = emu_rt.launches;
   for (i = 0; i < 10; i++) rig_poll();
   CHECK(emu_rt.launches == launches, "(expired) %lu launches while idle",
      emu_rt.launches - launches);
   rig_candidate(DIFF_ALWAYS);
   rig_solve(1);
   CHECK(R.dev.status == DEV_WORK, "(expired) no resume, status %d",
      R.dev.status);

   /* (d) gpuminer's own pause (solver not called), then resume */
   R.pause = 1;
   emu_rt_advance((BRIDGEv3 + 1) * 1000.0);
   for (i = 0; i < 10; i++) {
      rc = rig_poll();
      CHECK(rc != VEOK, "(paused) solve reported");
   }
   rig_candidate(DIFF_SOLVE);
   rig_solve(2);
   printf("   solves %llu (%llu for an earlier candidate of the block)\n",
      (unsigned long long) R.solves, (unsigned long long) R.prev_candidate);
}  /* end scenario_idle() */

/**
 * (3) Block changes (new phash) while batches are in flight, after they
 * completed unharvested, while a stream is stuck, and during a map
 * build. Every batch has a solve for the old block; none may be
 * reported; the map is rebuilt. Then a stream stalls during a map build:
 * the build completes (and the transition table is built) only after
 * it finished. Requires a working device.
*/
static void scenario_blockchange(void)
{
   PEACH_CUDA_CTX *P;
   EMU_RT_KCFG *k;
   unsigned long builds, tl, chunks;
   word32 bnum = get32(R.bt.bnum);
   int i;

   R.pause = 1;
   R.keep = 1;

   /* (a) batches in flight (just launched) */
   rig_candidate(DIFF_ALWAYS);
   rig_solve(1);
   rig_poll();
   printf("   (a) in flight: block %u -> %u (phash B)\n", (unsigned) bnum,
      (unsigned) bnum + 1);
   rig_block(PHASH_B, ++bnum, DIFF_ALWAYS);
   rig_build();
   rig_solve(2);

   /* (b) batches completed (results landed) but not harvested */
   rig_poll();
   for (i = 0; i < 2 * BUSY_TICKS + 2; i++) emu_rt_tick();
   printf("   (b) completed, unharvested: block %u -> %u (phash A)\n",
      (unsigned) bnum, (unsigned) bnum + 1);
   rig_block(PHASH_A, ++bnum, DIFF_ALWAYS);
   rig_build();
   rig_solve(2);

   /* (c) stream 1 stuck with an old-block batch */
   rig_poll();
   P = PCTX;
   CHECK(P != NULL && emu_rt_stream_stuck(P->stream[1], 1) == 0,
      "cannot stall stream 1");
   printf("   (c) stream 1 stuck: block %u -> %u (phash B)\n",
      (unsigned) bnum, (unsigned) bnum + 1);
   rig_block(PHASH_B, ++bnum, DIFF_ALWAYS);
   builds = Bstat.launches;
   for (i = 0; i < 30; i++) rig_poll();
   CHECK(R.dev.status == DEV_INIT || R.dev.status == DEV_WORK,
      "(stuck) status %d", R.dev.status);
   if (P) emu_rt_stream_stuck(P->stream[1], 0);
   rig_build();
   rig_solve(2);
   CHECK(Bstat.launches > builds, "(stuck) map not rebuilt");

   /* (d) block change during a map build */
   printf("   (d) during the map build: block %u -> %u (phash A) -> %u"
      " (phash B)\n", (unsigned) bnum, (unsigned) bnum + 1,
      (unsigned) bnum + 2);
   rig_block(PHASH_A, ++bnum, DIFF_ALWAYS);
   builds = Bstat.launches;
   for (i = 0; i < 400 && Bstat.launches < builds + 3; i++) rig_poll();
   CHECK(R.dev.status == DEV_INIT, "(during build) status %d, not INIT",
      R.dev.status);
   rig_block(PHASH_B, ++bnum, DIFF_SOLVE);
   rig_build();
   rig_solve(2);

   /* (e) stream 1 stalls during the map build: stream 0 completes the
    * map; the build (and the transition table) must wait for stream 1 */
   printf("   (e) stream 1 stalls during the map build: block %u -> %u"
      " (phash A)\n", (unsigned) bnum, (unsigned) bnum + 1);
   rig_block(PHASH_A, ++bnum, DIFF_SOLVE);
   builds = Bstat.launches;
   for (i = 0; i < 400 && Bstat.launches < builds + 4; i++) rig_poll();
   P = PCTX;
   CHECK(R.dev.status == DEV_INIT && P != NULL &&
      emu_rt_stream_stuck(P->stream[1], 1) == 0, "(stall) cannot stall"
      " stream 1 during the map build (status %d)", R.dev.status);
   tl = Bstat.tlaunches;
   for (i = 0; i < 120; i++) rig_poll();
   /* chunks of the map build: see the kcu_peach_build configuration */
   k = emu_rt_kcfg("kcu_peach_build", 1);
   chunks = PEACHCACHELEN / ((unsigned long) k->min_grid * k->block);
   CHECK(Bstat.launches >= builds + chunks, "(stall) stream 0 did not"
      " complete the map (%lu of %lu chunks)", Bstat.launches - builds,
      chunks);
   CHECK(R.dev.status == DEV_INIT, "(stall) status %d while stream 1 is"
      " stalled, not INIT", R.dev.status);
   CHECK(Bstat.tlaunches == tl, "(stall) transition table launched while"
      " stream 1 is stalled");
   if (P) emu_rt_stream_stuck(P->stream[1], 0);
   rig_build();
   rig_solve(1);
}  /* end scenario_blockchange() */

/**
 * (6) Every allocation / stream / event creation of the initialization
 * fails once: a legacy (essential) one fails the device cleanly, a
 * pipeline-only one falls back to the legacy solver; nothing leaks; the
 * device works afterwards. @a legacy selects the configuration. In
 * pipeline mode, everything but the legacy solver's buffers and its two
 * streams counts as pipeline-only.
*/
static void scenario_allocfail(const char *legacy)
{
   static const char *api[] = {
      "cudaMalloc", "cudaMallocHost", "cudaHostAlloc", "cudaStreamCreate",
      "cudaEventCreate"
   };
   enum { NAPI = 5, MAXCALL = 64 };
   PEACH_CUDA_CTX *P;
   DEVICE_CTX d1;
   const EMU_RT_ALLOC *al;
   const void *essential[16];
   unsigned long before[NAPI], count[NAPI], n, worked = 0;
   word8 pipe_only[NAPI][MAXCALL];
   size_t i, k, ness = 0;
   int a, rc, pipeline;

   /* fault-free initialization: count and classify the calls */
   for (a = 0; a < NAPI; a++) before[a] = emu_rt_calls(api[a]);
   if (dev_start(legacy, NULL, NULL)) return;
   for (a = 0; a < NAPI; a++) count[a] = emu_rt_calls(api[a]) - before[a];
   pipeline = pipeline_mode();
   P = PCTX;
   essential[ness++] = P->d_map;
   essential[ness++] = P->d_phash;
   for (k = 0; k < 2; k++) {
      essential[ness++] = P->d_bt[k];
      essential[ness++] = P->d_solve[k];
      essential[ness++] = P->d_state[k];
      essential[ness++] = P->h_bt[k];
      essential[ness++] = P->h_solve[k];
   }
   memset(pipe_only, 0, sizeof(pipe_only));
   for (i = 0; i < emu_rt.nalloc; i++) {
      al = &emu_rt.alloc[i];
      for (a = 0; a < 3 && strcmp(al->api, api[a]) != 0; a++);
      if (a == 3 || al->call <= before[a]) continue;
      n = al->call - before[a];
      for (k = 0; k < ness && essential[k] != al->ptr; k++);
      if (n < MAXCALL) pipe_only[a][n] = pipeline && !(al->live && k < ness);
   }
   for (n = 3; n < MAXCALL; n++) pipe_only[3][n] = (word8) pipeline;
   for (n = 1; n < MAXCALL; n++) pipe_only[4][n] = (word8) pipeline;
   printf("   init (%s mode): %lu cudaMalloc, %lu cudaMallocHost, %lu"
      " cudaHostAlloc, %lu cudaStreamCreate, %lu cudaEventCreate\n",
      pipeline ? "pipeline" : "legacy", count[0], count[1], count[2],
      count[3], count[4]);
   dev_free();

   for (a = 0; a < NAPI; a++) {
      for (n = 1; n <= count[a] && n < MAXCALL; n++) {
         emu_rt_fault(api[a], n, cudaErrorMemoryAllocation, 0);
         env_config(legacy, NULL, NULL);
         rc = dev_init();
         CHECK(emu_rt_fault_clear() == 1, "%s #%lu: fault not reached",
            api[a], n);
         if (!pipe_only[a][n]) {
            CHECK(rc == VERROR && R.dev.status == DEV_FAIL, "%s #%lu"
               " failing: init = %d, status %d (expected a failed device)",
               api[a], n, rc, R.dev.status);
            if (R.dev.status < DEV_NULL) {
               emu_rt_thread(DEVICE_THREAD);
               rc = peach_solve_cuda(&R.dev, &R.bt, 0, &R.btout);
               emu_rt_thread(MAIN_THREAD);
               CHECK(rc == VETIMEOUT, "failed device: solve = %d, not"
                  " VETIMEOUT", rc);
            }
         } else {
            CHECK(rc == VEOK && !pipeline_mode(), "%s #%lu failing"
               " (pipeline only): init = %d, mode %s (expected legacy)",
               api[a], n, rc, pipeline_mode() ? "pipeline" : "legacy");
            if (rc == VEOK && worked++ == 0) {
               /* the fallback device works (once: the map is cached) */
               R.pause = R.keep = 1;
               rig_block(PHASH_A, get32(R.bt.bnum) + 1, DIFF_SOLVE);
               rig_build();
               rig_solve(1);
            }
         }
         dev_free();
      }
   }

   /* afterwards: a fault-free initialization works (gpuminer: solving
    * in another host thread than the failed initialization) */
   printf("   re-initialization after the failures\n");
   if (dev_start(legacy, NULL, NULL)) return;
   R.pause = R.keep = 1;
   rig_block(PHASH_A, get32(R.bt.bnum) + 1, DIFF_SOLVE);
   rig_build();
   rig_solve(1);
   dev_free();

   /* one host thread for everything (as the GPU A/B test): an error of
    * the failed initialization must not survive into the next one */
   printf("   re-initialization in the host thread of a failed one\n");
   emu_rt_fault("cudaMalloc", 1, cudaErrorMemoryAllocation, 0);
   env_config(legacy, NULL, NULL);
   dev_init();
   emu_rt_fault_clear();
   dev_free();
   if (dev_start(legacy, NULL, NULL)) return;
   CHECK(pipeline_mode() == pipeline, "(single host thread) re-initialized"
      " device uses the %s solver, expected the %s solver (an error of the"
      " failed initialization was left for cudaGetLastError())",
      pipeline_mode() ? "pipeline" : "legacy",
      pipeline ? "pipeline" : "legacy");
   R.thread = MAIN_THREAD;
   for (n = 0; n < 4 && R.dev.status >= DEV_NULL; n++) rig_poll();
   R.thread = DEVICE_THREAD;
   CHECK(R.dev.status >= DEV_NULL, "(single host thread) re-initialized"
      " device failed on an error left by the failed initialization: %s",
      LastAlert);
   dev_free();

   /* gpuminer: every device is initialized in the main thread; a device
    * whose initialization failed must not affect the next device */
   printf("   gpuminer: device 1 initialized after device 0 failed\n");
   emu_rt.dev.count = 2;
   emu_rt_fault("cudaMalloc", 1, cudaErrorMemoryAllocation, 0);
   env_config(legacy, NULL, NULL);
   rc = dev_init();
   CHECK(emu_rt_fault_clear() == 1 && rc == VERROR, "device 0 did not fail");
   memset(&d1, 0, sizeof(d1));
   d1.id = 1;
   d1.type = CUDA_DEVICE;
   d1.status = DEV_NULL;
   rc = peach_init_cuda_device(&d1);
   a = peach_pipeline_cuda_device(&d1);
   CHECK(rc == VEOK && a == pipeline, "(gpuminer) device 1 after a failed"
      " device 0: init = %d, %s solver, expected the %s solver", rc,
      a == 1 ? "pipeline" : "legacy", pipeline ? "pipeline" : "legacy");
   emu_rt_thread(DEVICE_THREAD);
   for (n = 0; n < 4 && d1.status >= DEV_NULL; n++) {
      emu_rt_sleep(DYNASLEEP);
      (void) peach_solve_cuda(&d1, &R.bt, 0, &R.btout);
   }
   emu_rt_thread(MAIN_THREAD);
   CHECK(d1.status >= DEV_NULL, "(gpuminer) device 1 failed: %s",
      LastAlert);
   rc = peach_free_cuda_device(&d1);
   CHECK(rc == VEOK, "(gpuminer) device 1: peach_free_cuda_device() = %d",
      rc);
   dev_free();
   emu_rt.dev.count = 1;
   /* errors left by this scenario must not affect later scenarios */
   emu_clear_errors();
}  /* end scenario_allocfail() */

/**
 * (10) gpuminer polls every device from one host thread: an error that
 * fails one device must not be left for the first cudaGetLastError() of
 * the next device polled. Device 0 fails on an injected (non-sticky)
 * error of @a api (@a err), in DEV_WORK if @a work is set, else during
 * its map build; device 1 (initialized as well) must keep working.
 * @a legacy selects the configuration.
*/
static void scenario_cascade(const char *legacy, const char *api,
   cudaError_t err, int work)
{
   DEVICE_CTX d1;
   int rc, n, garbage;

   printf("   device 0 fails on %s (%d) in %s\n", api, (int) err,
      work ? "DEV_WORK" : "DEV_INIT");
   emu_clear_errors();
   emu_rt.dev.count = 2;
   if (dev_start(legacy, NULL, NULL)) {
      emu_rt.dev.count = 1;
      return;
   }
   memset(&d1, 0, sizeof(d1));
   d1.id = 1;
   d1.type = CUDA_DEVICE;
   d1.status = DEV_NULL;
   emu_rt_thread(MAIN_THREAD);
   /* (no garbage fill: device 1 only starts its map build, which then
    * needs no second GiB of memory) */
   garbage = emu_rt.garbage;
   emu_rt.garbage = 0;
   rc = peach_init_cuda_device(&d1);
   emu_rt.garbage = garbage;
   CHECK(rc == VEOK && d1.status == DEV_INIT, "device 1: init = %d,"
      " status %d", rc, d1.status);
   R.pause = R.keep = 1;
   rig_block(PHASH_A, get32(R.bt.bnum) + 1, DIFF_SOLVE);
   if (work) rig_build();
   emu_rt_fault(api, 1, err, 0);
   for (n = 0; n < 200 && R.dev.status >= DEV_NULL; n++) rig_poll();
   CHECK(emu_rt_fault_clear() == 1 && R.dev.status == DEV_FAIL, "device 0"
      " did not fail on %s (status %d)", api, R.dev.status);
   /* the device thread polls the next device */
   emu_rt_thread(DEVICE_THREAD);
   for (n = 0; n < 4 && d1.status >= DEV_NULL; n++) {
      emu_rt_sleep(DYNASLEEP);
      (void) peach_solve_cuda(&d1, &R.bt, 0, &R.btout);
   }
   emu_rt_thread(MAIN_THREAD);
   CHECK(d1.status >= DEV_NULL, "device 1 failed after device 0 failed on"
      " %s: %s", api, LastAlert);
   rc = peach_free_cuda_device(&d1);
   CHECK(rc == VEOK, "device 1: peach_free_cuda_device() = %d", rc);
   dev_free();
   emu_rt.dev.count = 1;
   /* errors left by this scenario must not affect later scenarios */
   emu_clear_errors();
}  /* end scenario_cascade() */

/**
 * Legacy scenarios: (1), (4), (3), (7), (6), (10).
*/
static void scenarios_legacy(void)
{
   PEACH_CUDA_CTX *P;
   unsigned long alerts;
   int i;

   section("L1 legacy: init, map build, solves");
   alerts = Nlog[PLOG_ALERT];
   if (dev_start("1", NULL, NULL) == 0) {
      P = PCTX;
      CHECK(P->cfg_legacy == 1, "cfg_legacy %d", P->cfg_legacy);
      CHECK(P->mode == PEACH_CUDA_MODE_LEGACY, "mode %d", P->mode);
      R.pause = R.keep = 1;
      rig_block(PHASH_A, 2, DIFF_SOLVE);
      i = rig_build();
      printf("   map built in %d polls (%llu tiles computed, %llu copied)\n",
         i, (unsigned long long) Bstat.computed,
         (unsigned long long) Bstat.copied);
      rig_solve(3);
      CHECK(R.dev.work > 0, "ctx->work = 0");
      CHECK(P->bad_solves == 0, "%llu rejected solves",
         (unsigned long long) P->bad_solves);
      CHECK(Nlog[PLOG_ALERT] == alerts, "unexpected alert: %s", LastAlert);
   }
   section_end(R.dev.peach ? NULL : "FAIL");
   if (R.dev.peach == NULL) return;

   section("L4 legacy: bnum solved / tcount 0 / expiry -> DEV_IDLE");
   scenario_idle();
   CHECK(Nlog[PLOG_ALERT] == alerts, "unexpected alert: %s", LastAlert);
   section_end(NULL);

   section("L3 legacy: block changes with batches in flight");
   scenario_blockchange();
   CHECK(PCTX->bad_solves == 0, "%llu rejected solves",
      (unsigned long long) PCTX->bad_solves);
   CHECK(Nlog[PLOG_ALERT] == alerts, "unexpected alert: %s", LastAlert);
   section_end(NULL);

   section("L7 legacy: free + re-init with another configuration");
   dev_free();
   if (dev_start("1", "0x00", "4096") == 0) {
      P = PCTX;
      for (i = 0; i < 8; i++) {
         CHECK(P->cfg_skip[i] == 0, "cfg_skip[%d] = 0x%x", i,
            (unsigned) P->cfg_skip[i]);
      }
      CHECK(P->cfg_batch == 4096, "cfg_batch %u", (unsigned) P->cfg_batch);
      CHECK(P->mode == PEACH_CUDA_MODE_LEGACY, "mode %d", P->mode);
      /* free with batches in flight, then once more from the start */
      rig_block(PHASH_B, get32(R.bt.bnum) + 1, DIFF_SOLVE);
      rig_build();
      rig_solve(1);
      rig_poll();
      dev_free();
      if (dev_start("1", NULL, NULL) == 0) {
         rig_build();
         rig_solve(1);
         dev_free();
      }
   }
   section_end(NULL);

   section("L6 legacy: allocation failures during init");
   scenario_allocfail("1");
   section_end(NULL);

   section("L10 legacy: a failed device does not fail the next one");
   scenario_cascade("1", "cudaMemset", cudaErrorInvalidValue, 0);
   scenario_cascade("1", "cudaMemcpyAsync", cudaErrorInvalidValue, 1);
   section_end(NULL);
}  /* end scenarios_legacy() */

/**
 * Pipeline: work accounting over a steady stretch of polls: ctx->work
 * must grow by the completed nonces of the harvested batches.
*/
static void pipeline_work_check(void)
{
   word64 w0, dw, pre, post;
   int i, steady = 1;

   rig_candidate(DIFF_NEVER);
   for (i = 0; i < 4; i++) rig_poll();
   for (pre = 0, i = 0; i < PEACH_CUDA_NCTX_MAX; i++) pre += Fx.last[i];
   w0 = R.dev.work;
   Fx.fin_sum = 0;
   Fx.fin = 0;
   Fx.track = 1;
   for (i = 0; i < 40; i++) {
      rig_poll();
      if (R.dev.status != DEV_WORK) steady = 0;
   }
   Fx.track = 0;
   dw = R.dev.work - w0;
   CHECK(steady, "status left DEV_WORK");
   CHECK(Fx.fin > 0 && dw > 0, "no work (%lu batches, work +%llu)",
      Fx.fin, (unsigned long long) dw);
   CHECK(dw <= Fx.fin_sum + pre, "work +%llu exceeds the completed nonces"
      " of the batches (%llu + %llu in flight before)",
      (unsigned long long) dw, (unsigned long long) Fx.fin_sum,
      (unsigned long long) pre);
   for (post = 0, i = 0; i < PEACH_CUDA_NCTX_MAX; i++) post += Fx.last[i];
   CHECK(dw + post >= Fx.fin_sum, "work +%llu is short"
      " of the completed nonces %llu of %lu batches",
      (unsigned long long) dw, (unsigned long long) Fx.fin_sum, Fx.fin);
   printf("   work +%llu over %lu batches (completed %llu)\n",
      (unsigned long long) dw, Fx.fin, (unsigned long long) Fx.fin_sum);
}  /* end pipeline_work_check() */

/**
 * Predicate: every pipeline batch context has a batch of the current
 * trailer in flight.
*/
static int pred_inflight(void)
{
   PEACH_CUDA_CTX *P = PCTX;
   int id;

   if (P == NULL || R.dev.status != DEV_WORK) return 0;
   for (id = 0; id < P->nctx; id++) {
      if (!P->inflight[id] || memcmp(P->h_bt[id], &R.bt, 92) != 0) return 0;
   }
   return 1;
}  /* end pred_inflight() */

/**
 * Set the fake clock to 100 ms past its next second, so that the next
 * polls (DYNASLEEP apart) fall within one second of time(): the solver
 * sees no pause of the caller between them.
*/
static void clock_align(void)
{
   emu_rt_advance(1100.0 - fmod(emu_rt.now_ms, 1000.0));
}  /* end clock_align() */

/**
 * (9) Pipeline: polling intervals and outdated trailers. gpuminer sends
 * a solve only for its current or previous trailer (rig_solved() checks
 * every pipeline solve against both).
 * (a) polls 1, 3 and 5 s apart (gpuminer -d) on an unchanged trailer:
 *     batches are harvested and solves reported at any interval;
 * (b) batches of a trailer X in flight, then a pause of the caller (no
 *     calls) while the trailer changes twice, X -> Y -> Z: gpuminer's
 *     previous trailer is Y, never seen by the solver, so X's batches
 *     count as work but report no solve; solves resume for Z;
 * (c) batches of X in flight, then X -> Y seen at the next poll (no
 *     pause): a solve of X is still reported (gpuminer's BT_prev).
 * Requires a working pipeline device.
*/
static void scenario_polling(void)
{
   static const double interval[] = { 1000.0, 3000.0, 5000.0 };
   PEACH_CUDA_CTX *P = PCTX;
   word64 s0, w0, b0, bad0, p0;
   int i, j, n, rc;

   R.pause = R.keep = 1;
   /* (a) slow polling, unchanged trailer (40 polls < BRIDGEv3) */
   for (j = 0; j < (int) (sizeof(interval) / sizeof(interval[0])); j++) {
      rig_candidate(DIFF_SOLVE);
      s0 = R.solves;
      w0 = (word64) R.dev.work;
      b0 = P->batches_total;
      PollMs = interval[j];
      for (i = 0; i < 40; i++) rig_poll();
      PollMs = DYNASLEEP;
      printf("   (a) polls %.0f ms apart: +%llu batches, +%llu solves, work"
         " +%llu\n", interval[j], (unsigned long long) (P->batches_total -
         b0), (unsigned long long) (R.solves - s0),
         (unsigned long long) ((word64) R.dev.work - w0));
      CHECK(P->batches_total >= b0 + 10 && R.solves > s0 &&
         (word64) R.dev.work > w0, "(a) polls %.0f ms apart: +%llu"
         " batches, +%llu solves (expected batches and solves)",
         interval[j], (unsigned long long) (P->batches_total - b0),
         (unsigned long long) (R.solves - s0));
      CHECK(R.dev.status == DEV_WORK, "(a) status %d", R.dev.status);
   }

   /* (b) pause while the trailer changes twice (every batch solves) */
   rig_candidate(DIFF_ALWAYS);
   n = rig_until(pred_inflight, 40);
   CHECK(n >= 0, "(b) no batches of the new trailer in flight");
   for (i = 0; i < 2 * BUSY_TICKS + 2; i++) emu_rt_tick();
   emu_rt_advance(5000.0);
   rig_candidate(DIFF_ALWAYS);
   rig_candidate(DIFF_ALWAYS);
   b0 = P->batches_total;
   bad0 = P->bad_solves;
   rc = rig_poll();
   CHECK(rc != VEOK, "(b) solve reported for a trailer gpuminer cannot"
      " send");
   CHECK(P->batches_total == b0 + (word64) P->nctx, "(b) %llu batches"
      " harvested after the pause, expected %d (counted as work)",
      (unsigned long long) (P->batches_total - b0), P->nctx);
   CHECK(P->bad_solves == bad0, "(b) %llu rejected solves",
      (unsigned long long) (P->bad_solves - bad0));
   rig_solve(1);
   CHECK(memcmp(&R.last, &R.bt, 92) == 0, "(b) solve after the pause is"
      " not for the current trailer");

   /* (c) change without a pause: the previous trailer's solve counts */
   rig_candidate(DIFF_ALWAYS);
   clock_align();
   n = rig_until(pred_inflight, 40);
   CHECK(n >= 0, "(c) no batches of the new trailer in flight");
   for (i = 0; i < 2 * BUSY_TICKS + 2; i++) emu_rt_tick();
   rig_candidate(DIFF_ALWAYS);
   p0 = R.prev_candidate;
   rc = rig_poll();
   CHECK(rc == VEOK && R.prev_candidate == p0 + 1, "(c) no solve of the"
      " previous trailer reported (gpuminer sends it as BT_prev)");
   rig_solve(1);
   printf("   solves %llu (%llu for an earlier candidate of the block)\n",
      (unsigned long long) R.solves, (unsigned long long) R.prev_candidate);
}  /* end scenario_polling() */

/**
 * Build the map (cached) of a device started by dev_start(), expect
 * pipeline mode. @returns 0 on success
*/
static int pipeline_build(void)
{
   if (!pipeline_mode()) {
      CHECK(0, "pipeline mode expected");
      return (-1);
   }
   R.pause = R.keep = 1;
   rig_block(PHASH_A, get32(R.bt.bnum) + 1, DIFF_SOLVE);
   if (rig_build() < 0) return (-1);
   CHECK(pipeline_active(), "fell back to legacy: %s", LastAlert);
   return 0;
}  /* end pipeline_build() */

/**
 * Start a pipeline device and build its map (cached), expect pipeline
 * mode. @returns 0 on success
*/
static int pipeline_start(const char *skip, const char *batch)
{
   if (dev_start("0", skip, batch)) return (-1);
   return pipeline_build();
}  /* end pipeline_start() */

/**
 * Wait (poll) until the pipeline device has fallen back to the legacy
 * solver, then check the legacy solver works. No invalid solve may be
 * reported meanwhile (rig_solved()).
*/
static void expect_fallback(const char *what, unsigned long alerts)
{
   word64 bad0 = R.bad;
   int i;

   for (i = 0; i < 300 && pipeline_active(); i++) rig_poll();
   CHECK(PCTX != NULL && !pipeline_active(),
      "%s: no fallback to the legacy solver", what);
   CHECK(Nlog[PLOG_ALERT] > alerts, "%s: no alert", what);
   rig_candidate(DIFF_SOLVE);
   rig_build();
   rig_solve(1);
   CHECK(R.bad == bad0, "%s: invalid solve reported", what);
}  /* end expect_fallback() */

/**
 * Pipeline scenarios: (2), (4), (3), (8), (9), (5), (6), (7), (10).
*/
static void scenarios_pipeline(void)
{
   PEACH_CUDA_CTX *P;
   EMU_RT_KCFG *k;
   unsigned long alerts, nt, ns;
   word64 bad0;
   word32 n0, m;
   int i, n, avail;

   section("P2 pipeline: init, map + T build, self-test, solves");
   alerts = Nlog[PLOG_ALERT];
   if (dev_start("0", NULL, NULL)) {
      section_end(NULL);
      return;
   }
   avail = pipeline_mode();
   if (!avail) {
#ifdef PEACH_CUDA_SELFTEST_TILES
      /* this src/peach.cu has the pipeline solver: it must be used */
      CHECK(0, "MCM_PEACH_LEGACY=0 initialized the legacy solver: %s",
         LastWarn);
      dev_free();
      section_end(NULL);
#else
      printf("SKIP: pipeline mode not available in this build"
         " (MCM_PEACH_LEGACY=0 initialized the legacy solver)\n");
      dev_free();
      section_end("SKIP (pipeline mode not available)");
#endif
      return;
   }
   P = PCTX;
   printf("   N = %u slots (min %u, cap %u) x %d contexts\n",
      (unsigned) P->nslots, (unsigned) P->nslots_min, (unsigned) P->cap,
      P->nctx);
   CHECK(Streams != NULL || P->nctx == PEACH_CUDA_NCTX_AUTO, "%d batch"
      " contexts (automatic), expected %d", P->nctx, PEACH_CUDA_NCTX_AUTO);
   R.pause = R.keep = 1;
   k = emu_rt_kcfg("kcu_peach_pipe_transitions", 1);
   nt = k->launches;
   k = emu_rt_kcfg("kcu_peach_pipe_selftest", 1);
   ns = k->launches;
   rig_block(PHASH_A, get32(R.bt.bnum) + 1, DIFF_SOLVE);
   rig_build();
   CHECK(pipeline_active(), "fallback: %s", LastAlert);
   CHECK(emu_rt_kcfg("kcu_peach_pipe_transitions", 1)->launches > nt,
      "transition table kernel not launched");
   CHECK(emu_rt_kcfg("kcu_peach_pipe_selftest", 1)->launches > ns,
      "self-test kernel not launched");
   rig_solve(3);
   CHECK(Nlog[PLOG_ALERT] == alerts, "unexpected alert: %s", LastAlert);
   CHECK(P->bad_solves == 0, "%llu rejected solves",
      (unsigned long long) P->bad_solves);
   pipeline_work_check();
   CHECK(P->canary_checks > 0, "no canary verified in %llu batches",
      (unsigned long long) P->batches_total);
   CHECK(R.dev.hps > 0, "ctx->hps = 0 after %llu batches",
      (unsigned long long) P->batches_total);
   CHECK(Bchk.batches > 0, "no pipeline batch launch was checked");
   /* adaptive batch size, batches of two sizes in flight: two slow
    * batches halve the size once; the batches of the old size still in
    * flight after it, and the new ones they slow down (contend_batch()),
    * must not halve it again (no solves: the contexts stay in step) */
   rig_candidate(DIFF_NEVER);
   n0 = P->nslots;
   m = (n0 / 2) & ~((word32) PEACH_PIPE_BLOCK - 1);
   if (m < P->nslots_min) m = P->nslots_min;
   Fx.contend = n0;
   for (i = 0; i < 40; i++) rig_poll();
   Fx.contend = 0;
   k = emu_rt_kcfg("kcu_peach_pipe_final", 1);
   k->ms = -1.0;
   CHECK(n0 > P->nslots_min && P->nslots == m, "slow batches, then"
      " batches slowed down by the old size in flight (%d contexts):"
      " nslots %u -> %u, expected %u", P->nctx, (unsigned) n0,
      (unsigned) P->nslots, (unsigned) m);
   CHECK(pipeline_active(), "fallback: %s", LastAlert);
   rig_candidate(DIFF_SOLVE);
   /* adaptive batch size: batches slower than 250 ms */
   n0 = P->nslots;
   k->ms = 400.0;
   for (i = 0; i < 40; i++) rig_poll();
   k->ms = -1.0;
   CHECK(P->nslots < n0 && P->nslots >= P->nslots_min &&
      P->nslots % 128 == 0, "slow batches: nslots %u -> %u (min %u)",
      (unsigned) n0, (unsigned) P->nslots, (unsigned) P->nslots_min);
   CHECK(pipeline_active(), "fallback: %s", LastAlert);
   section_end(NULL);

   section("P4 pipeline: bnum solved / tcount 0 / expiry -> DEV_IDLE");
   scenario_idle();
   CHECK(pipeline_active(), "fallback: %s", LastAlert);
   section_end(NULL);

   section("P3 pipeline: block changes with batches in flight");
   scenario_blockchange();
   CHECK(pipeline_active(), "fallback: %s", LastAlert);
   CHECK(PCTX->bad_solves == 0, "%llu rejected solves",
      (unsigned long long) PCTX->bad_solves);
   CHECK(Nlog[PLOG_ALERT] == alerts, "unexpected alert: %s", LastAlert);
   section_end(NULL);

   section("P8 pipeline: trailer difficulty 0 -> every final hash solves");
   /* consensus (trigg_eval()) and the legacy solver accept every final
    * hash at difficulty 0: batches run, every solve is verified */
   rig_candidate(0);
   bad0 = PCTX->bad_solves;
   /* each batch context may still report one solve of the previous
    * candidate (gpuminer's previous trailer), possibly after solves of
    * the current one (a poll returns at its first solve, so the batches
    * of the first contexts, which all solve now, are harvested first) */
   for (i = 0; i < 4 * PCTX->nctx; i++) {
      if (rig_solve(1) < 0 || R.last.difficulty[0] == 0) break;
   }
   CHECK(R.last.difficulty[0] == 0, "difficulty 0: no solve of the"
      " difficulty 0 candidate (last solve has difficulty %u)",
      (unsigned) R.last.difficulty[0]);
   CHECK(PCTX->bad_solves == bad0, "difficulty 0: %llu rejected solves",
      (unsigned long long) (PCTX->bad_solves - bad0));
   CHECK(Nlog[PLOG_ALERT] == alerts, "unexpected alert: %s", LastAlert);
   CHECK(R.dev.status == DEV_WORK && pipeline_active(), "difficulty 0:"
      " status %d, %s", R.dev.status, pipeline_active() ? "pipeline" :
      "legacy fallback");
   rig_candidate(DIFF_SOLVE);
   rig_solve(1);
   section_end(NULL);

   section("P9 pipeline: slow polling, outdated trailers");
   scenario_polling();
   CHECK(pipeline_active(), "fallback: %s", LastAlert);
   CHECK(PCTX->bad_solves == 0, "%llu rejected solves",
      (unsigned long long) PCTX->bad_solves);
   CHECK(Nlog[PLOG_ALERT] == alerts, "unexpected alert: %s", LastAlert);
   section_end(NULL);
   dev_free();

   section("P5 pipeline: fault injection -> legacy fallback");
   /* (a) T corrupted before the self-test */
   if (pipeline_start(NULL, NULL) == 0) {
      alerts = Nlog[PLOG_ALERT];
      Fx.corrupt_T = 1;
      rig_block(PHASH_B, get32(R.bt.bnum) + 1, DIFF_SOLVE);
      expect_fallback("T corrupted before the self-test", alerts);
      Fx.corrupt_T = 0;
      dev_free();
   }
   /* (b) T corrupted after the self-test (canary / verification) */
   if (pipeline_start(NULL, NULL) == 0) {
      alerts = Nlog[PLOG_ALERT];
      corrupt_T();
      expect_fallback("T corrupted after the self-test", alerts);
      dev_free();
   }
   /* (c) forged device result (never solves at this difficulty) */
   if (pipeline_start(NULL, NULL) == 0) {
      alerts = Nlog[PLOG_ALERT];
      rig_candidate(DIFF_NEVER);
      Fx.forge = 1;
      expect_fallback("forged result", alerts);
      CHECK(PCTX->bad_solves >= 1, "forged result not rejected");
      Fx.forge = 0;
      dev_free();
   }
   /* (d) queue overflow reported by the device */
   if (pipeline_start(NULL, NULL) == 0) {
      alerts = Nlog[PLOG_ALERT];
      rig_candidate(DIFF_NEVER);
      Fx.overflow = 1;
      expect_fallback("overflow", alerts);
      Fx.overflow = 0;
      dev_free();
   }
   /* (e) stale result (an older batch's solve, wrong epoch): never
    * reported, not verified as a candidate; ignoring it or falling back
    * to legacy (with an alert) are both safe */
   if (pipeline_start(NULL, NULL) == 0) {
      Fx.capture = 1;
      Fx.have_captured = 0;
      rig_candidate(DIFF_ALWAYS);
      for (i = 0; i < 100 && !Fx.have_captured; i++) rig_poll();
      Fx.capture = 0;
      CHECK(Fx.have_captured, "no result captured");
      /* drain: batches launched at DIFF_ALWAYS report (valid) solves */
      rig_candidate(DIFF_NEVER);
      for (i = 0; i < 4 * BUSY_TICKS + 4; i++) rig_poll();
      alerts = Nlog[PLOG_ALERT];
      bad0 = PCTX->bad_solves;
      Fx.stale = 4;
      for (i = 0; i < 30; i++) {
         CHECK(rig_poll() != VEOK, "stale result reported");
      }
      Fx.stale = 0;
      CHECK(PCTX->bad_solves == bad0, "stale result verified as a candidate"
         " (%llu rejected)", (unsigned long long) (PCTX->bad_solves - bad0));
      CHECK(pipeline_active() || Nlog[PLOG_ALERT] > alerts,
         "fallback on a stale result without an alert");
      printf("   stale result: %s\n", pipeline_active() ? "ignored" :
         "legacy fallback");
      dev_free();
   }
   /* (f) a pipeline kernel launch fails: no invalid solve; the device
    * fails (as the legacy solver on launch errors) or falls back */
   if (pipeline_start(NULL, NULL) == 0) {
      rig_candidate(DIFF_SOLVE);
      emu_rt_fault("launch:kcu_peach_pipe_hash_sha256", 3,
         cudaErrorLaunchOutOfResources, 0);
      for (i = 0; i < 40 && R.dev.status >= DEV_NULL; i++) rig_poll();
      CHECK(emu_rt_fault_clear() == 1, "launch fault not reached");
      printf("   launch failure: %s\n", R.dev.status < DEV_NULL ?
         "device failed" : pipeline_active() ? "pipeline continues" :
         "legacy fallback");
      if (R.dev.status >= DEV_NULL) rig_solve(1);
      dev_free();
   }
   /* (g) a sticky device error (a kernel fault) with solving batches in
    * flight: the device fails (VETIMEOUT), nothing is reported, no
    * synchronization waits forever */
   if (pipeline_start(NULL, NULL) == 0) {
      rig_candidate(DIFF_ALWAYS);
      rig_poll();
      emu_rt_sticky(cudaErrorIllegalAddress);
      for (i = n = 0; i < 10; i++) n += rig_poll() == VEOK;
      CHECK(n == 0, "(sticky error) %d solve(s) reported", n);
      CHECK(R.dev.status == DEV_FAIL, "(sticky error) status %d, not"
         " DEV_FAIL", R.dev.status);
      CHECK(rig_poll() == VETIMEOUT, "(sticky error) failed device: solve"
         " is not VETIMEOUT");
      printf("   sticky device error: device failed\n");
      /* device reset (the test clears the error of every host thread) */
      emu_rt_sticky(cudaSuccess);
      emu_clear_errors();
      dev_free();
   }
   section_end(NULL);

   section("P6 pipeline: allocation failures during init");
   scenario_allocfail("0");
   section_end(NULL);

   section("P7 pipeline: free + re-init with other configurations");
   if (pipeline_start("0", "1024") == 0) {
      P = PCTX;
      for (i = 0; i < 8; i++) {
         CHECK(P->cfg_skip[i] == 0, "cfg_skip[%d] = 0x%x", i,
            (unsigned) P->cfg_skip[i]);
      }
      CHECK(P->nslots == 1024, "nslots %u (MCM_PEACH_BATCH=1024)",
         (unsigned) P->nslots);
      rig_solve(2);
      rig_poll();
      dev_free();
   }
   if (dev_start("1", NULL, NULL) == 0) {
      CHECK(!pipeline_mode(), "MCM_PEACH_LEGACY=1 in pipeline mode");
      rig_build();
      rig_solve(1);
      dev_free();
   }
   if (pipeline_start("0x40,0x40,0,0x40,0x40,0x40,0x40,0x04", NULL) == 0) {
      CHECK(PCTX->cfg_skip[2] == 0 && PCTX->cfg_skip[7] == 0x04,
         "per-round MCM_PEACH_SKIP not applied");
      CHECK(PCTX->cfg_skip_auto == 0, "explicit MCM_PEACH_SKIP is auto");
      rig_solve(2);
      dev_free();
   }
   /* default (automatic) skip masks per compute capability: 0x40 in
    * every round (the fake device: 8.6), MD2 evaluated in rounds 4..7
    * and SHA-256, SHA3, Keccak dropped in round 0 on 12.x; an explicit
    * MCM_PEACH_SKIP wins. Slots per batch: twice the default per thread
    * when the round 0 mask drops an algorithm of every pair (a, a + 4) */
   {
      static const struct {
         int cc;
         const char *skip;
         word8 mask[8];
         const char *log;
         word32 slots;     /* slots per batch per resident thread */
      } sk[] = {
         { 8, NULL, { 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40 },
            "MCM_PEACH_SKIP=auto(cc 8: 0x40)", PEACH_CUDA_SLOTS_PER_THREAD },
         { 12, NULL, { 0x78, 0x40, 0x40, 0x40, 0, 0, 0, 0 },
            "MCM_PEACH_SKIP=auto(cc 12: 0x78,0x40,0x40,0x40,0x00,0x00,"
            "0x00,0x00)", 2 * PEACH_CUDA_SLOTS_PER_THREAD },
         { 12, "0x40", { 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40 },
            "MCM_PEACH_SKIP=0x40 ", PEACH_CUDA_SLOTS_PER_THREAD },
         { 12, "0", { 0, 0, 0, 0, 0, 0, 0, 0 }, "MCM_PEACH_SKIP=0x00 ",
            PEACH_CUDA_SLOTS_PER_THREAD },
         { 8, "0x78,0x40,0x40,0x40,0,0,0,0",
            { 0x78, 0x40, 0x40, 0x40, 0, 0, 0, 0 },
            "MCM_PEACH_SKIP=0x78,0x40,0x40,0x40,0x00,0x00,0x00,0x00 ",
            2 * PEACH_CUDA_SLOTS_PER_THREAD },
         { 12, "0x70,0x40,0x40,0x40,0,0,0,0",
            { 0x70, 0x40, 0x40, 0x40, 0, 0, 0, 0 },
            "MCM_PEACH_SKIP=0x70,0x40,0x40,0x40,0x00,0x00,0x00,0x00 ",
            PEACH_CUDA_SLOTS_PER_THREAD },
         { 8, "0x87", { 0x87, 0x87, 0x87, 0x87, 0x87, 0x87, 0x87, 0x87 },
            "MCM_PEACH_SKIP=0x87 ", 2 * PEACH_CUDA_SLOTS_PER_THREAD }
      };
      const int cc = emu_rt.dev.cc_major;
      int j;

      for (i = 0; i < (int) (sizeof(sk) / sizeof(sk[0])); i++) {
         emu_rt.dev.cc_major = sk[i].cc;
         LastInfo[0] = '\0';
         n = dev_start("0", sk[i].skip, NULL);
         emu_rt.dev.cc_major = cc;
         if (n != 0) continue;
         /* the init log line names the masks in use */
         CHECK(strstr(LastInfo, sk[i].log) != NULL, "cc %d, MCM_PEACH_SKIP="
            "%s: init log line without \"%s\": %s", sk[i].cc,
            sk[i].skip ? sk[i].skip : "<unset>", sk[i].log, LastInfo);
         if (pipeline_build() != 0) continue;
         P = PCTX;
         for (j = 0; j < 8; j++) {
            CHECK(P->cfg_skip[j] == sk[i].mask[j], "cc %d, MCM_PEACH_SKIP=%s:"
               " cfg_skip[%d] = 0x%x, expected 0x%x", sk[i].cc,
               sk[i].skip ? sk[i].skip : "<unset>", j,
               (unsigned) P->cfg_skip[j], (unsigned) sk[i].mask[j]);
         }
         CHECK(P->cfg_skip_auto == (sk[i].skip == NULL), "cc %d,"
            " MCM_PEACH_SKIP=%s: cfg_skip_auto %d", sk[i].cc,
            sk[i].skip ? sk[i].skip : "<unset>", P->cfg_skip_auto);
         /* (the fake device fits full batches) */
         CHECK(P->cap == sk[i].slots * (word32) P->sms *
            (word32) P->max_threads_sm && P->nslots == P->cap, "cc %d,"
            " MCM_PEACH_SKIP=%s: %u slots per batch (cap %u), expected"
            " %u x %d SMs x %d threads", sk[i].cc,
            sk[i].skip ? sk[i].skip : "<unset>", (unsigned) P->nslots,
            (unsigned) P->cap, (unsigned) sk[i].slots, P->sms,
            P->max_threads_sm);
         rig_solve(2);
         printf("   cc %d, MCM_PEACH_SKIP=%s: masks %02x %02x %02x %02x"
            " %02x %02x %02x %02x, %u slots per batch\n", sk[i].cc,
            sk[i].skip ? sk[i].skip : "<unset>", P->cfg_skip[0],
            P->cfg_skip[1], P->cfg_skip[2], P->cfg_skip[3], P->cfg_skip[4],
            P->cfg_skip[5], P->cfg_skip[6], P->cfg_skip[7],
            (unsigned) P->cap);
         dev_free();
      }
   }
   /* the init log line names the masks in use */
   {
      const int cc = emu_rt.dev.cc_major;

      emu_rt.dev.cc_major = 12;
      LastInfo[0] = '\0';
      n = dev_start("0", NULL, NULL);
      emu_rt.dev.cc_major = cc;
      if (n == 0) {
         CHECK(strstr(LastInfo, "MCM_PEACH_SKIP=auto(cc 12: 0x78,0x40,0x40,"
            "0x40,0x00,0x00,0x00,0x00)") != NULL, "init log line: %s",
            LastInfo);
         dev_free();
      }
   }
   /* MD2 grid: with several batch contexts at most
    * PEACH_CUDA_MD2_BLOCKS resident blocks per SM, with one its
    * occupancy; never more than its occupancy */
   {
      EMU_RT_KCFG *k = emu_rt_kcfg("kcu_peach_pipe_hash_md2", 1);
      static const struct {
         int occ;
         const char *streams;    /* NULL: the scenarios' setting */
      } md[] = {
         { 6, NULL }, { PEACH_CUDA_MD2_BLOCKS, NULL }, { 3, NULL },
         { 1, NULL }, { 6, "1" }, { 6, "2" }
      };
      const int keep_occ = k->active_sm;
      const char *keep = Streams;
      int want;

      for (i = 0; i < (int) (sizeof(md) / sizeof(md[0])); i++) {
         k->active_sm = md[i].occ;
         if (md[i].streams != NULL) Streams = md[i].streams;
         n = dev_start("0", NULL, NULL);
         Streams = keep;
         if (n == 0) {
            P = PCTX;
            CHECK(pipeline_mode(), "MD2 occupancy %d: no pipeline mode",
               md[i].occ);
            want = P->nctx > 1 && md[i].occ > PEACH_CUDA_MD2_BLOCKS ?
               PEACH_CUDA_MD2_BLOCKS : md[i].occ;
            CHECK(P->launch.grid_hash[6] == P->sms * want, "MD2 occupancy"
               " %d, %d contexts: grid %d, expected %d x %d SMs",
               md[i].occ, P->nctx, P->launch.grid_hash[6], want, P->sms);
            CHECK(md[i].streams == NULL ||
               P->nctx == atoi(md[i].streams), "MCM_PEACH_STREAMS=%s:"
               " %d contexts", md[i].streams, P->nctx);
            CHECK(P->launch.grid_hash[0] == P->sms * keep_occ,
               "MD2 occupancy %d: blake2b grid %d", md[i].occ,
               P->launch.grid_hash[0]);
            dev_free();
         }
      }
      k->active_sm = keep_occ;
      printf("   MD2 grid: min(occupancy, %d) blocks per SM with several"
         " contexts, its occupancy with one\n", PEACH_CUDA_MD2_BLOCKS);
   }
   /* batch contexts: 1, 2, 3 with full evaluation (MD2 kernel), 4, then
    * invalid values (automatic) */
   {
      static const struct {
         const char *streams, *skip;
         int nctx;
      } sc[] = {
         { "1", NULL, 1 }, { "2", NULL, 2 }, { "3", "0", 3 },
         { "4", NULL, 4 }, { "5", NULL, PEACH_CUDA_NCTX_AUTO },
         { "two", NULL, PEACH_CUDA_NCTX_AUTO }
      };
      const char *keep = Streams;
      word64 b0;

      for (i = 0; i < (int) (sizeof(sc) / sizeof(sc[0])); i++) {
         Streams = sc[i].streams;
         n = pipeline_start(sc[i].skip, NULL);
         Streams = keep;
         if (n != 0) continue;
         P = PCTX;
         CHECK(P->nctx == sc[i].nctx, "MCM_PEACH_STREAMS=%s: %d contexts,"
            " expected %d", sc[i].streams, P->nctx, sc[i].nctx);
         /* (every kernel has the same occupancy on the fake device, not
          * above PEACH_CUDA_MD2_BLOCKS: the MD2 grid is the same with any
          * number of contexts) */
         CHECK(P->launch.grid_hash[6] == P->launch.grid_hash[0],
            "MCM_PEACH_STREAMS=%s: MD2 grid %d, other hash grids %d",
            sc[i].streams, P->launch.grid_hash[6], P->launch.grid_hash[0]);
         /* slots per batch do not depend on the contexts (the fake
          * device fits full batches for any number of them) */
         CHECK(P->cap == (word32) PEACH_CUDA_SLOTS_PER_THREAD *
            (word32) P->sms * (word32) P->max_threads_sm,
            "MCM_PEACH_STREAMS=%s: %u slots per batch, expected %d x %d"
            " SMs x %d threads", sc[i].streams, (unsigned) P->cap,
            PEACH_CUDA_SLOTS_PER_THREAD, P->sms, P->max_threads_sm);
         b0 = P->batches_total;
         n = rig_until(pred_inflight, 40);
         CHECK(n >= 0, "MCM_PEACH_STREAMS=%s: not every batch context has"
            " a batch in flight", sc[i].streams);
         rig_solve(2);
         printf("   MCM_PEACH_STREAMS=%s: %d contexts, md2 grid %d (of"
            " %d), +%llu batches\n", sc[i].streams, P->nctx,
            P->launch.grid_hash[6], P->launch.grid_hash[0],
            (unsigned long long) (P->batches_total - b0));
         dev_free();
      }
   }
   section_end(NULL);

   section("P10 pipeline: a failed device does not fail the next one");
   scenario_cascade("0", "cudaMemset", cudaErrorInvalidValue, 0);
   scenario_cascade("0", "cudaEventRecord", cudaErrorInvalidValue, 1);
   scenario_cascade("0", "cudaStreamQuery", cudaErrorLaunchFailure, 1);
   section_end(NULL);
}  /* end scenarios_pipeline() */

/**
 * Peak resident memory of this process in MiB (Linux), or 0.
*/
static unsigned long peak_mib(void)
{
   char line[128];
   unsigned long kib = 0;
   FILE *fp = fopen("/proc/self/status", "r");

   if (fp == NULL) return 0;
   while (fgets(line, sizeof(line), fp)) {
      if (sscanf(line, "VmHWM: %lu kB", &kib) == 1) break;
   }
   fclose(fp);
   return kib >> 10;
}  /* end peak_mib() */

int main(void)
{
   EMU_RT_KCFG *k;
   const char *only;
   unsigned long misuse;
   time_t t0;
   int i;

   if (env_long("PEACH_TEST_HOST", 0, 0, 1) != 1) {
      printf("SKIP: Peach CUDA host emulation test (minutes; set"
         " PEACH_TEST_HOST=1 to run it)\n");
      return 0;
   }
   t0 = time(NULL);
   Verbose = (int) env_long("PEACH_TEST_HOST_VERBOSE", 0, 0, 2);
   only = getenv("PEACH_TEST_HOST_ONLY");
   if (only != NULL && strcmp(only, "legacy") != 0 &&
         strcmp(only, "pipeline") != 0) only = NULL;
   Ncache = (int) env_long("PEACH_TEST_HOST_CACHE", MAXCACHE, 0, MAXCACHE);
   Streams = getenv("PEACH_TEST_HOST_STREAMS");
   if (Streams != NULL && *Streams == '\0') Streams = NULL;
   srand16(1, 2, 3);
   srand32(4);

   /* fake device and runtime */
   emu_rt_init();
   emu_rt.dev.sms = 2;
   emu_rt.dev.max_threads_sm = 32;
   emu_rt.busy_ticks = BUSY_TICKS;
   emu_rt.nproc = (int) env_long("PEACH_TEST_HOST_PROCS", 4, 1,
      EMU_RT_MAXPROC);
   emu_rt.verbose = Verbose;
   /* map chunks of 32768 tiles; legacy batches of 256 nonces */
   k = emu_rt_kcfg("kcu_peach_build", 1);
   k->min_grid = 256;
   k->block = 128;
   k = emu_rt_kcfg("kcu_peach_solve", 1);
   k->min_grid = 4;
   k->block = 64;
   /* pipeline kernels: __launch_bounds__(128) */
   k = emu_rt_kcfg("kcu_peach_pipe_*", 1);
   k->max_block = 128;
   emu_rt_on_launch(on_launch, NULL);
   printf("Peach host emulation: %d SMs x %d threads, %lu ticks busy, %d"
      " worker processes, %d cached maps\n", emu_rt.dev.sms,
      emu_rt.dev.max_threads_sm, emu_rt.busy_ticks, emu_rt.nproc, Ncache);

   memset(&R, 0, sizeof(R));
   R.thread = DEVICE_THREAD;
   scenario_checkhash();
   if (only == NULL || strcmp(only, "legacy") == 0) scenarios_legacy();
   if (R.dev.peach) dev_free();
   if (only == NULL || strcmp(only, "pipeline") == 0) scenarios_pipeline();
   if (R.dev.peach) dev_free();

   misuse = emu_rt.misuse;
   CHECK(misuse == 0, "%lu runtime API misuses (listed above)", misuse);
   CHECK(emu_rt.hangs == 0, "%lu synchronizations with a stuck stream",
      emu_rt.hangs);
   CHECK(emu_rt.crashes == 0, "%lu kernel crashes", emu_rt.crashes);
   CHECK(R.bad == 0, "%llu invalid solves reported",
      (unsigned long long) R.bad);

   printf("\n== summary (%ld s, peak %lu MiB; %llu polls, %llu solves"
      " verified, %lu kernel launches, map tiles: %llu computed, %llu"
      " copied)\n", (long) (time(NULL) - t0), peak_mib(),
      (unsigned long long) R.polls, (unsigned long long) R.solves,
      emu_rt.launches, (unsigned long long) Bstat.computed,
      (unsigned long long) Bstat.copied);
   for (i = 0; i < Nresults; i++) {
      printf("   %-62s %s\n", Results[i].name, Results[i].result);
   }
   printf("   %lu failed checks\n", Fails);

   return Fails ? EXIT_FAILURE : EXIT_SUCCESS;
}  /* end main() */
