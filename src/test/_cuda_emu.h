/**
 * @file _cuda_emu.h
 * @brief Sequential CPU emulation of the CUDA device subset used by Peach.
 * @details Include BEFORE any Peach device header (peach_*.cuh,
 * peach_select.h) in a CPU test; the headers then compile as plain C.
 * Kernels launched with CUDA_KERNEL() run one thread at a time (block by
 * block, thread by thread) on the calling thread.
 * <br />
 * Warp/block collectives (__syncthreads, __syncwarp, __ballot_sync,
 * __shfl_sync, __match_any_sync, __activemask) and `__shared__` are
 * deliberately NOT defined: device headers MUST guard them with
 * `#ifdef __CUDA_ARCH__` and provide a per-thread fallback, so unguarded
 * use fails to compile here. Atomics use C11 _Generic over exactly the
 * CUDA overload set, so a call that would not compile under nvcc (e.g. a
 * `word64 *` on LP64) does not compile here either.
 * <br />
 * The emulated built-in variables (threadIdx, blockIdx, blockDim,
 * gridDim) are process globals: never launch emulated kernels from
 * inside an OpenMP parallel region.
 * @copyright Adequate Systems LLC, 2018-2025. All Rights Reserved.
 * <br />For license information, please refer to ../../LICENSE.md
*/

/* include guard */
#ifndef TEST_CUDA_EMU_H
#define TEST_CUDA_EMU_H


#if defined(__CUDACC__) || defined(__NVCC__)
   #error "_cuda_emu.h is for plain C CPU tests only"
#endif

#include <float.h>   /* for FLT_EVAL_METHOD */
#include <string.h>  /* for memcpy() */

/* plain C float ops are only equivalent to __f*_rn() without excess
 * precision (x87); FMA contraction cannot apply to single operations */
#if !defined(FLT_EVAL_METHOD) || FLT_EVAL_METHOD != 0
   #error "CUDA emulation requires FLT_EVAL_METHOD == 0"
#endif

/* emulation marker, checked by device headers */
#define PEACH_CUDA_EMU  1

/* function/variable space qualifiers */
#define __host__
#define __device__
#define __global__
#define __forceinline__    inline
#define __noinline__
#define __launch_bounds__(...)
#define __constant__       const
#define warpSize           32

/* vector types (layout compatible with CUDA) */
typedef struct { unsigned int x, y, z; } dim3;
typedef struct { unsigned int x, y; } uint2;
typedef struct { unsigned int x, y, z, w; } uint4;

/* runtime API placeholders (as used by shared host enqueue code) */
typedef void *cudaStream_t;
typedef enum { cudaSuccess = 0 } cudaError_t;

/**
 * @private
 * Emulated cudaGetLastError(): emulated launches cannot fail.
*/
static inline cudaError_t cudaGetLastError(void)
{
   return cudaSuccess;
}  /* end cudaGetLastError() */

/* emulated built-in variables (one "current" thread) */
static dim3 threadIdx, blockIdx, blockDim, gridDim;
static int emu__started;

/**
 * @private
 * Begin an emulated launch of (grid x block) threads. Additional launch
 * arguments (shared memory size, stream) are accepted and ignored.
*/
static inline void emu__launch(unsigned int grid, unsigned int block, ...)
{
   gridDim.x = grid; gridDim.y = gridDim.z = 1;
   blockDim.x = block; blockDim.y = blockDim.z = 1;
   threadIdx.y = threadIdx.z = blockIdx.y = blockIdx.z = 0;
   emu__started = 0;
}  /* end emu__launch() */

/**
 * @private
 * Advance the emulated launch to the next thread.
 * @returns non-zero while threads remain
*/
static inline int emu__next(void)
{
   if (!emu__started) {
      emu__started = 1;
      threadIdx.x = blockIdx.x = 0;
      return (gridDim.x > 0 && blockDim.x > 0);
   }
   if (++threadIdx.x < blockDim.x) return 1;
   threadIdx.x = 0;
   return (++blockIdx.x < gridDim.x);
}  /* end emu__next() */

/* CUDA_KERNEL(FN, grid, block[, shmem, stream])(args...);
 * same call shape as peach.cuh; (args...) binds to FN in the for() body */
#define CUDA_KERNEL(FN, ...) \
   for (emu__launch(__VA_ARGS__); emu__next(); ) FN

/* integer intrinsics (per-thread, safe to emulate) */
static inline unsigned int __byte_perm(unsigned int x, unsigned int y,
   unsigned int s)
{
   unsigned int r = 0, sel, b;
   int i;

   for (i = 0; i < 4; i++) {
      sel = (s >> (4 * i)) & 7;
      b = sel < 4 ? (x >> (8 * sel)) : (y >> (8 * (sel - 4)));
      r |= (b & 0xFF) << (8 * i);
   }
   return r;
}  /* end __byte_perm() */

static inline int __clz(unsigned int x)
   { return x ? __builtin_clz(x) : 32; }
static inline int __popc(unsigned int x)
   { return __builtin_popcount(x); }
static inline int __ffs(int x)
   { return __builtin_ffs(x); }

static inline unsigned int __funnelshift_l(unsigned int lo,
   unsigned int hi, unsigned int sh)
{
   sh &= 31;
   return sh ? (hi << sh) | (lo >> (32 - sh)) : hi;
}  /* end __funnelshift_l() */

static inline unsigned int __funnelshift_r(unsigned int lo,
   unsigned int hi, unsigned int sh)
{
   sh &= 31;
   return sh ? (lo >> sh) | (hi << (32 - sh)) : lo;
}  /* end __funnelshift_r() */

/* float intrinsics: IEEE single precision, round-to-nearest-even */
static inline float __fadd_rn(float a, float b) { return a + b; }
static inline float __fsub_rn(float a, float b) { return a - b; }
static inline float __fmul_rn(float a, float b) { return a * b; }
static inline float __fdiv_rn(float a, float b) { return a / b; }
static inline float __int2float_rn(int i) { return (float) i; }
static inline float __uint2float_rn(unsigned int u) { return (float) u; }
static inline unsigned int __float_as_uint(float f)
   { unsigned int u; memcpy(&u, &f, sizeof(u)); return u; }
static inline float __uint_as_float(unsigned int u)
   { float f; memcpy(&f, &u, sizeof(f)); return f; }

/* read-only cache load */
#define __ldg(ptr)  (*(ptr))

/* atomics: sequential emulation, exact CUDA overload set via _Generic */
static inline int emu__add_i32(int *p, int v)
   { int o = *p; *p = (int) ((unsigned int) o + (unsigned int) v); return o; }
static inline unsigned int emu__add_u32(unsigned int *p, unsigned int v)
   { unsigned int o = *p; *p = o + v; return o; }
static inline unsigned long long emu__add_u64(unsigned long long *p,
   unsigned long long v)
   { unsigned long long o = *p; *p = o + v; return o; }
static inline int emu__cas_i32(int *p, int c, int v)
   { int o = *p; if (o == c) *p = v; return o; }
static inline unsigned int emu__cas_u32(unsigned int *p, unsigned int c,
   unsigned int v)
   { unsigned int o = *p; if (o == c) *p = v; return o; }
static inline unsigned long long emu__cas_u64(unsigned long long *p,
   unsigned long long c, unsigned long long v)
   { unsigned long long o = *p; if (o == c) *p = v; return o; }

#define atomicAdd(p, v)  _Generic((p), \
   int *: emu__add_i32, \
   unsigned int *: emu__add_u32, \
   unsigned long long *: emu__add_u64)((p), (v))
#define atomicCAS(p, c, v)  _Generic((p), \
   int *: emu__cas_i32, \
   unsigned int *: emu__cas_u32, \
   unsigned long long *: emu__cas_u64)((p), (c), (v))

/* end include guard */
#endif
