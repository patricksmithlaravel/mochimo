/**
 * @file peach_compat.cuh
 * @brief Peach CUDA / CPU-emulation compatibility layer.
 * @details Qualifier, atomic, load and bit helper macros shared by the
 * Peach pipeline device headers. Every header built on this layer
 * compiles as CUDA C++ under nvcc (included by peach.cu), AND as plain C
 * under gcc (included by a CPU test after test/_cuda_emu.h).
 * <br />
 * C/CUDA common subset rules for code built on this layer:
 * - storage class first (`static __device__ ...`), every function via
 *   the PEACH_* qualifier macros (never a bare `inline`);
 * - no `= {}`, statement-expressions, binary literals, C++ features, or
 *   implicit `void *` conversions; no unused parameters or variables;
 * - warp/block collectives, `__shared__` and `__syncthreads()` ONLY inside
 *   `#ifdef __CUDA_ARCH__`, with a per-thread fallback in the `#else`;
 * - PEACH_LDG*() only in PEACH_DEV functions and kernels, never in
 *   PEACH_HD functions.
 * @copyright Adequate Systems LLC, 2018-2025. All Rights Reserved.
 * <br />For license information, please refer to ../LICENSE.md
*/

/* include guard */
#ifndef MOCHIMO_PEACH_COMPAT_CUH
#define MOCHIMO_PEACH_COMPAT_CUH


#include <stddef.h>  /* for size_t */
#include <string.h>  /* for memset() */
#include "extint.h"  /* for word types */

/* compilation mode qualifiers -- `static` ALWAYS first (C -Wextra) */
#if defined(__CUDACC__)
   #include <cuda_runtime.h>

   #define PEACH_HD        static __host__ __device__ __forceinline__
   #define PEACH_DEV       static __device__ __forceinline__
   #define PEACH_KERNEL    static __global__
   #define PEACH_HOST      static
   #define PEACH_CONST     __constant__
   #define PEACH_MEMSET_ASYNC(ptr, val, len, stream) \
      cudaMemsetAsync((ptr), (val), (len), (stream))

#elif defined(PEACH_CUDA_EMU)
   #define PEACH_HD        static inline
   #define PEACH_DEV       static inline
   #define PEACH_KERNEL    static inline
   #define PEACH_HOST      static inline
   #define PEACH_CONST     const
   #define PEACH_MEMSET_ASYNC(ptr, val, len, stream) \
      peach_emu_memset_async((ptr), (val), (len), (stream))

/**
 * @private
 * Synchronous memset() with the result type of cudaMemsetAsync().
 * Usable as a statement or as an expression, as under nvcc.
*/
static inline cudaError_t peach_emu_memset_async(void *ptr, int val,
   size_t len, cudaStream_t stream)
{
   (void) stream;
   memset(ptr, val, len);
   return cudaSuccess;
}  /* end peach_emu_memset_async() */

#else
   #error "peach_compat.cuh requires nvcc, or test/_cuda_emu.h first"

#endif

/* atomics -- explicit casts to the CUDA overload set (word64 is
 * `unsigned long` on LP64, word32 is `unsigned long` on LLP64) */
#define PEACH_ATOMIC_ADD32(ptr, val) \
   atomicAdd((unsigned int *) (ptr), (unsigned int) (val))
#define PEACH_ATOMIC_ADD64(ptr, val) \
   atomicAdd((unsigned long long *) (ptr), (unsigned long long) (val))
#define PEACH_ATOMIC_CAS32(ptr, cmp, val) \
   atomicCAS((unsigned int *) (ptr), (unsigned int) (cmp), \
      (unsigned int) (val))

/* read-only (non-coherent) loads -- device code ONLY (PEACH_DEV
 * functions and kernels), never inside PEACH_HD functions */
#define PEACH_LDG16(ptr)   __ldg((const unsigned short *) (ptr))
#define PEACH_LDG32(ptr)   __ldg((const unsigned int *) (ptr))
#define PEACH_LDG128(ptr)  __ldg((const uint4 *) (ptr))

/**
 * @private
 * Rotate a 32-bit word left by @a n bits (any @a n, incl. 0).
*/
PEACH_HD word32 peach_rotl32(word32 x, int n)
{
#ifdef __CUDA_ARCH__
   return __funnelshift_l(x, x, (unsigned int) n);
#else
   return (x << (n & 31)) | (x >> ((32 - n) & 31));
#endif
}  /* end peach_rotl32() */

/**
 * @private
 * Rotate a 32-bit word right by @a n bits (any @a n, incl. 0).
*/
PEACH_HD word32 peach_rotr32(word32 x, int n)
{
#ifdef __CUDA_ARCH__
   return __funnelshift_r(x, x, (unsigned int) n);
#else
   return (x >> (n & 31)) | (x << ((32 - n) & 31));
#endif
}  /* end peach_rotr32() */

/**
 * @private
 * Rotate a 64-bit word left by @a n bits (any @a n, incl. 0).
*/
PEACH_HD word64 peach_rotl64(word64 x, int n)
{
   return (x << (n & 63)) | (x >> ((64 - n) & 63));
}  /* end peach_rotl64() */

/**
 * @private
 * Rotate a 64-bit word right by @a n bits (any @a n, incl. 0).
*/
PEACH_HD word64 peach_rotr64(word64 x, int n)
{
   return (x >> (n & 63)) | (x << ((64 - n) & 63));
}  /* end peach_rotr64() */

/**
 * @private
 * Swap the byte order of a 32-bit word.
*/
PEACH_HD word32 peach_bswap32(word32 x)
{
#ifdef __CUDA_ARCH__
   return __byte_perm(x, 0, 0x0123);
#else
   return (x >> 24) | ((x >> 8) & WORD32_C(0xFF00)) |
      ((x << 8) & WORD32_C(0xFF0000)) | (x << 24);
#endif
}  /* end peach_bswap32() */

/* end include guard */
#endif
