/**
 * @file peach-gpuab-cu.c
 * @brief GPU A/B test of the Peach CUDA solvers: legacy vs pipeline.
 * @details OPT-IN (several minutes of GPU time): without
 * PEACH_TEST_GPUAB=1 the test prints a SKIP line and succeeds, so that
 * `make test` (which runs every *-cu test when nvcc is found) does not
 * turn into a benchmark. Run it with:
 * <br />`make test-peach-gpuab PEACH_TEST_GPUAB=1`
 * <br />GPU only; without a usable CUDA device the test is skipped
 * (exit 0). For each configuration, one CUDA device is initialized with
 * peach_init_cuda_device() under that configuration's environment
 * (MCM_PEACH_LEGACY, MCM_PEACH_SKIP, MCM_PEACH_BATCH, MCM_PEACH_STREAMS),
 * the Peach map is built (with the transition table and self-test of the
 * pipeline), and the solver runs for a fixed time on the same block
 * trailer at a moderate difficulty. Every reported solve must belong to
 * that trailer and pass peach_check(). The solver in use is checked with
 * peach_pipeline_cuda_device() after initialization, after the map build
 * and after solving: a pipeline configuration fails if the device uses
 * the legacy solver (failed pipeline setup, or a fallback after a self-
 * test, canary or verification failure). The device is released with
 * peach_free_cuda_device() between configurations. Reports completed
 * nonces per second (work counted by the solver over the solving time)
 * and the speedup over the legacy solver. Configurations: legacy,
 * pipeline (defaults), then pipeline variants: MD2 dropped in every
 * round (skip mask 0x40), full evaluation (skip mask 0), MD2 dropped in
 * rounds 0..3 only (the default masks on compute capability 12.x), a
 * wider skip mask, the smallest batch size, and two and four batch
 * contexts (streams). Time bounded: map build <= 300 s, solving as
 * below.
 * <br />
 * Optional environment:
 * - PEACH_GPUAB_SECONDS: solving time per configuration (default 20,
 *   at most 180 so the trailer cannot expire while solving)
 * - PEACH_GPUAB_DIFF: difficulty (default 24)
 * - PEACH_GPUAB_DEVICE: CUDA device index (default 0)
 * - PEACH_GPUAB_SWEEP=0: skip the pipeline variants (default 1)
 * - PEACH_GPUAB_DEBUG=1: debug logging (e.g. pipeline batch sizing)
 * - PEACH_GPUAB_POLL_MS: milliseconds between two solver calls (default
 *   1; gpuminer polls every 10 ms by default)
 * - PEACH_GPUAB_ONLY=<label>: run only the configuration with this label
 *   (e.g. "pipeline"); its default and automatic settings then come from
 *   the caller's MCM_PEACH_SKIP, MCM_PEACH_BATCH and MCM_PEACH_STREAMS
 *   instead of being unset, so one run can measure any setting (for
 *   tuning, and for sanitizer runs of one small configuration)
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <cuda_runtime_api.h>

#include "extint.h"
#include "extlib.h"
#include "exttime.h"

#include "error.h"
#include "peach.h"
#include "device.h"

#define GPUMAX          16
#define BUILD_TIMEOUT   300.0    /* seconds, map build limit */
#define DEF_SECONDS     20       /* seconds of solving per configuration */
#define MAX_SECONDS     180      /* stays below BRIDGEv3 (trailer expiry) */
#define DEF_DIFF        24       /* difficulty */
#define MIN_EXPECTED    30.0     /* expected solves for the "any" check */

/* Block 0x1 trailer data taken directly from the Mochimo Blockchain Tfile
 * (as src/test/peach-mining-cu.c); tcount = 1 */
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

/* solver configurations; the first is the baseline */
typedef struct {
   const char *name;    /* label */
   const char *legacy;  /* MCM_PEACH_LEGACY */
   const char *skip;    /* MCM_PEACH_SKIP, or NULL (unset: default) */
   const char *batch;   /* MCM_PEACH_BATCH, or NULL (unset: automatic) */
   const char *streams; /* MCM_PEACH_STREAMS, or NULL (unset: automatic) */
   int sweep;           /* pipeline variant (skipped with SWEEP=0) */
} GPUAB_CONFIG;

static const GPUAB_CONFIG Config[] = {
   { "legacy", "1", NULL, NULL, NULL, 0 },
   { "pipeline", "0", NULL, NULL, NULL, 0 },
   { "pipeline skip=0x40", "0", "0x40", NULL, NULL, 1 },
   { "pipeline skip=0x00", "0", "0", NULL, NULL, 1 },
   { "pipeline skip=0x40x4,0x00x4", "0",
      "0x40,0x40,0x40,0x40,0,0,0,0", NULL, NULL, 1 },
   { "pipeline skip=0x70", "0", "0x70", NULL, NULL, 1 },
   { "pipeline batch=min", "0", NULL, "1", NULL, 1 },
   { "pipeline streams=2", "0", NULL, NULL, "2", 1 },
   { "pipeline streams=4", "0", NULL, NULL, "4", 1 }
};

#define NCONFIG   ((int) (sizeof(Config) / sizeof(Config[0])))

/**
 * Monotonic time, in seconds.
*/
static double now_sec(void)
{
   struct timespec ts;

   clock_gettime(CLOCK_MONOTONIC, &ts);
   return (double) ts.tv_sec + (double) ts.tv_nsec * 1e-9;
}

/**
 * Read a small non-negative integer from the environment.
*/
static long env_long(const char *name, long def, long min, long max)
{
   const char *str = getenv(name);
   char *end;
   long value;

   if (str == NULL || *str == '\0') return def;
   value = strtol(str, &end, 10);
   if (*end != '\0' || value < min || value > max) {
      printf("ignoring invalid %s=\"%s\" (using %ld)\n", name, str, def);
      return def;
   }

   return value;
}

/**
 * Value of an environment variable, or @a def when unset.
*/
static const char *env_str(const char *name, const char *def)
{
   const char *str = getenv(name);

   return str != NULL ? str : def;
}

/**
 * Check that device @a dev uses the solver of configuration @a cfg.
 * @returns 1 if it does, else 0 (reported as a failure)
*/
static int check_solver(DEVICE_CTX *dev, const GPUAB_CONFIG *cfg,
   const char *when)
{
   int want = strcmp(cfg->legacy, "1") ? 1 : 0;
   int mode = peach_pipeline_cuda_device(dev);

   if (mode == want) return 1;
   if (mode < 0) printf("FAIL: %s: device has no Peach context\n", when);
   else {
      printf("FAIL: %s: device uses the %s solver, expected the %s"
         " solver\n", when, mode ? "pipeline" : "legacy",
         want ? "pipeline" : "legacy");
   }

   return 0;
}

/**
 * Run one solver configuration on device @a dev.
 * @returns completed nonces per second, or a negative value on failure
*/
static double run_config(DEVICE_CTX *dev, const GPUAB_CONFIG *cfg,
   const BTRAILER *bt_in, double seconds, word32 poll_ms, int keepenv)
{
   BTRAILER bt, btout;
   double t0, tstart, elapsed, build, rate, expect;
   unsigned long solves;
   int ecode, ok;

   /* configure (read once by peach_init_cuda_device()) */
   setenv("MCM_PEACH_LEGACY", cfg->legacy, 1);
   if (cfg->skip) setenv("MCM_PEACH_SKIP", cfg->skip, 1);
   else if (!keepenv) unsetenv("MCM_PEACH_SKIP");
   if (cfg->batch) setenv("MCM_PEACH_BATCH", cfg->batch, 1);
   else if (!keepenv) unsetenv("MCM_PEACH_BATCH");
   if (cfg->streams) setenv("MCM_PEACH_STREAMS", cfg->streams, 1);
   else if (!keepenv) unsetenv("MCM_PEACH_STREAMS");

   /* fresh trailer: not expired, not solved (btout bnum differs) */
   memcpy(&bt, bt_in, sizeof(bt));
   put32(bt.time0, (word32) time(NULL));
   memset(&btout, 0, sizeof(btout));

   printf("== %s (MCM_PEACH_LEGACY=%s MCM_PEACH_SKIP=%s"
      " MCM_PEACH_BATCH=%s MCM_PEACH_STREAMS=%s)\n", cfg->name,
      cfg->legacy, env_str("MCM_PEACH_SKIP", "<default>"),
      env_str("MCM_PEACH_BATCH", "<auto>"),
      env_str("MCM_PEACH_STREAMS", "<auto>"));
   fflush(stdout);
   if (peach_init_cuda_device(dev) != VEOK) {
      printf("FAIL: peach_init_cuda_device()\n");
      peach_free_cuda_device(dev);
      return -1.0;
   }

   /* build the Peach map (DEV_INIT -> DEV_IDLE -> DEV_WORK) */
   ok = check_solver(dev, cfg, "after initialization");
   t0 = now_sec();
   while (ok && dev->status != DEV_WORK) {
      /* keep time0 fresh until solving starts, so a slow map build cannot
       * expire the trailer (BRIDGEv3) before the IDLE -> WORK gate */
      put32(bt.time0, (word32) time(NULL));
      ecode = peach_solve_cuda(dev, &bt, 0, &btout);
      if (ecode == VETIMEOUT || dev->status < DEV_NULL) {
         printf("FAIL: device failed during map build (status %d)\n",
            dev->status);
         ok = 0;
         break;
      }
      if (ecode == VEOK) {
         printf("FAIL: solve reported during map build\n");
         ok = 0;
         break;
      }
      if (now_sec() - t0 > BUILD_TIMEOUT) {
         printf("FAIL: map build timeout (%.0f s)\n", BUILD_TIMEOUT);
         ok = 0;
         break;
      }
      millisleep(1);
   }
   build = now_sec() - t0;
   /* the pipeline self-test runs at the end of the map build */
   if (ok) ok = check_solver(dev, cfg, "after the map build");

   /* solve for a fixed time, verify every solve */
   solves = 0;
   tstart = now_sec();
   elapsed = 0.0;
   while (ok && (elapsed = now_sec() - tstart) < seconds) {
      ecode = peach_solve_cuda(dev, &bt, 0, &btout);
      if (ecode == VETIMEOUT || dev->status < DEV_NULL) {
         printf("FAIL: device failed while solving (status %d)\n",
            dev->status);
         ok = 0;
         break;
      }
      if (ecode == VEOK) {
         if (memcmp(&btout, &bt, 92) != 0) {
            printf("FAIL: solve for a different block trailer\n");
            ok = 0;
            break;
         }
         if (peach_check(&btout) != VEOK) {
            printf("FAIL: invalid solve (peach_check)\n");
            ok = 0;
            break;
         }
         solves++;
         /* keep solving: the block counts as "not solved yet" */
         memset(&btout, 0, sizeof(btout));
      }
      if (dev->status != DEV_WORK) {
         printf("FAIL: device left DEV_WORK (status %d)\n", dev->status);
         check_solver(dev, cfg, "after leaving DEV_WORK");
         ok = 0;
         break;
      }
      millisleep(poll_ms);
   }
   /* a pipeline defect would have switched to the legacy solver */
   if (ok) ok = check_solver(dev, cfg, "after solving");

   rate = elapsed > 0.0 ? (double) dev->work / elapsed : 0.0;
   expect = (double) dev->work / (double) (1ULL << bt.difficulty[0]);
   if (ok) {
      printf("   map %.1f s; %.3f M completed nonces/s over %.1f s;"
         " %lu solves verified (expected ~%.1f)\n", build, rate / 1e6,
         elapsed, solves, expect);
      if (solves == 0 && expect >= MIN_EXPECTED) {
         printf("FAIL: no solves, ~%.0f expected\n", expect);
         ok = 0;
      }
   }
   fflush(stdout);

   if (peach_free_cuda_device(dev) != VEOK) {
      printf("FAIL: peach_free_cuda_device()\n");
      ok = 0;
   }
   if (dev->peach != NULL || dev->status != DEV_NULL) {
      printf("FAIL: device context not released\n");
      ok = 0;
   }

   return ok ? rate : -1.0;
}

int main(void)
{
   DEVICE_CTX D[GPUMAX];
   BTRAILER bt;
   double rate[NCONFIG], seconds;
   cudaError_t err;
   word32 seed, poll_ms;
   const char *only;
   int count, devidx, i, sweep, fails, found;
   long diff;

   /* opt-in (see above) */
   if (env_long("PEACH_TEST_GPUAB", 0, 0, 1) != 1) {
      printf("SKIP: Peach GPU A/B test (minutes of GPU time; set"
         " PEACH_TEST_GPUAB=1 to run it)\n");
      return 0;
   }
   /* skip without a usable CUDA device (no driver, or no device) */
   count = 0;
   err = cudaGetDeviceCount(&count);
   if (err != cudaSuccess || count < 1) {
      printf("SKIP: no usable CUDA device (%s); GPU A/B test not run\n",
         err != cudaSuccess ? cudaGetErrorString(err) : "0 devices");
      return 0;
   }

   seconds = (double) env_long("PEACH_GPUAB_SECONDS", DEF_SECONDS, 1,
      MAX_SECONDS);
   diff = env_long("PEACH_GPUAB_DIFF", DEF_DIFF, 8, 48);
   devidx = (int) env_long("PEACH_GPUAB_DEVICE", 0, 0, GPUMAX - 1);
   sweep = (int) env_long("PEACH_GPUAB_SWEEP", 1, 0, 1);
   poll_ms = (word32) env_long("PEACH_GPUAB_POLL_MS", 1, 1, 1000);
   if (env_long("PEACH_GPUAB_DEBUG", 0, 0, 1)) setploglevel(PLOG_DEBUG);
   only = getenv("PEACH_GPUAB_ONLY");
   if (only != NULL && *only == '\0') only = NULL;
   for (found = 0, i = 0; only != NULL && i < NCONFIG; i++) {
      if (strcmp(only, Config[i].name) == 0) found = 1;
   }
   if (only != NULL && !found) {
      printf("FAIL: PEACH_GPUAB_ONLY=\"%s\" names no configuration\n", only);
      return EXIT_FAILURE;
   }

   memset(D, 0, sizeof(D));
   count = init_cuda_devices(D, GPUMAX);
   if (count < 1 || devidx >= count) {
      printf("SKIP: CUDA device %d not available (%d devices)\n",
         devidx, count);
      return 0;
   }
   printf("GPU A/B: device %d: %s; difficulty %ld; %.0f s per config;"
      " polls %lu ms apart\n", devidx, D[devidx].info, diff, seconds,
      (unsigned long) poll_ms);

   seed = (word32) time(NULL);
   srand16(seed, seed ^ 0x5a5a5a5a, seed ^ 0xa5a5a5a5);
   memcpy(&bt, Block1, sizeof(bt));
   bt.difficulty[0] = (word8) diff;

   fails = 0;
   for (i = 0; i < NCONFIG; i++) {
      rate[i] = 0.0;
      if (only != NULL ? strcmp(only, Config[i].name) != 0 :
            (Config[i].sweep && !sweep)) continue;
      rate[i] = run_config(&D[devidx], &Config[i], &bt, seconds, poll_ms,
         only != NULL);
      if (rate[i] < 0.0) fails++;
   }

   printf("== summary (completed nonces/s, speedup vs %s)\n", Config[0].name);
   for (i = 0; i < NCONFIG; i++) {
      if (only != NULL ? strcmp(only, Config[i].name) != 0 :
            (Config[i].sweep && !sweep)) continue;
      if (rate[i] < 0.0) printf("   %-28s FAILED\n", Config[i].name);
      else {
         printf("   %-28s %10.3f M/s  x%.2f\n", Config[i].name,
            rate[i] / 1e6, rate[0] > 0.0 ? rate[i] / rate[0] : 0.0);
      }
   }

   return fails ? EXIT_FAILURE : EXIT_SUCCESS;
}
