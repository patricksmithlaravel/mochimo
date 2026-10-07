/**
 * @file peach_pipeline.cuh
 * @brief Peach CUDA "pipeline" solver: shared tables, types and kernels.
 * @details Compiles as CUDA C++ under nvcc (included by peach.cu) and as
 * plain C under gcc (included by CPU tests after test/_cuda_emu.h).
 * <br />
 * A batch of N nonce slots is processed as: init (thread per slot) ->
 * for round r = 0..7: one hash kernel per algorithm not skipped in that
 * round, consuming the per-algorithm queue of round r and appending to
 * the queues of round r + 1 -> final (sha256 with the last tile and the
 * difficulty check). The per-jump algorithm selection is described in
 * peach_select.h.
 * <br />
 * The legacy nonce frame tables, cuCONSTn860 and cu_rand64() also live
 * here (shared with the legacy kcu_peach_solve() in peach.cu); their
 * qualifiers are unchanged so the legacy kernel code is unchanged.
 * @copyright Adequate Systems LLC, 2018-2025. All Rights Reserved.
 * <br />For license information, please refer to ../LICENSE.md
*/

/* include guard */
#ifndef MOCHIMO_PEACH_PIPELINE_CUH
#define MOCHIMO_PEACH_PIPELINE_CUH


#include "extint.h"           /* for word types */
#include "peach.h"            /* for PEACH* constants */
#include "peach_compat.cuh"   /* for PEACH_* qualifiers, uint4 */
#include "peach_select.h"     /* for algorithm selection */

#ifdef __CUDACC__
   #include "peach.cuh"       /* for CUDA_KERNEL() */
#endif

/* sm_61 performs MUCH better with the __constant__ qualifier */
#if defined(PEACH_CUDA_EMU)
   #define cuCONSTn860 const
#elif __CUDA_ARCH__ == 610
   #define cuCONSTn860 __constant__
#else
   #define cuCONSTn860
#endif

/* Legacy nonce frame tables (second nonce half, kcu_peach_solve()) */
static __device__ cuCONSTn860 word64 Z_ING[32] = {
   18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
   34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 23, 24, 31, 32, 33, 34
};
static __device__ cuCONSTn860 word64 Z_NS[64] = {
   129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 145, 149, 154,
   155, 156, 157, 177, 178, 179, 180, 182, 183, 184, 185, 186, 187,
   188, 189, 190, 191, 192, 193, 194, 196, 197, 198, 199, 200, 201,
   202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 241,
   244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255
};
static __device__ cuCONSTn860 word64 Z_MASS[32] = {
   214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224,
   225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235,
   236, 237, 238, 239, 240, 242, 214, 215, 216, 219
};
static __device__ cuCONSTn860 word64 Z_PREP[8] = {
   12, 13, 14, 15, 16, 17, 12, 13
};
static __device__ cuCONSTn860 word64 Z_ADJ[64] = {
   61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75,
   76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90,
   91, 92, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104,
   105, 107, 108, 109, 110, 112, 114, 115, 116, 117, 118,
   119, 120, 121, 122, 123, 124, 125, 126, 127, 128
};

/**
 * CUDA device function to generate a 64-bit random number.
 * State generation based on SplitMix64 by Sebastiano Vigna.
 * @param d_state Pointer to location of state
 * @return (word64) value representing a 64-bit random number
 */
static __device__ __forceinline__ word64 cu_rand64(word64 *d_state)
{
   word64 index = (blockIdx.x * blockDim.x) + threadIdx.x;
   word64 z = (d_state[index] += WORD64_C(0x9e3779b97f4a7c15));
   z = (z ^ (z >> 30)) * WORD64_C(0xbf58476d1ce4e5b9);
   z = (z ^ (z >> 27)) * WORD64_C(0x94d049bb133111eb);
   return (d_state[index] = z ^ (z >> 31));
}  /* end cu_rand64() */

/* Pipeline layout constants */
#define PEACH_PIPE_BLOCK      128   /**< threads per block, every kernel */
#define PEACH_PIPE_NQUEUE     17    /**< queues: 2 parities x 8 + final */
#define PEACH_PIPE_CNTPAD     32    /**< words per (padded) counter */
#define PEACH_PIPE_NCNT       72    /**< counters: 9 rounds x 8 algos */
#define PEACH_PIPE_SKIP_MD2   0x40  /**< default per-round skip mask */

/**
 * Per-slot (nonce) state of a batch. 32 bytes.
*/
typedef struct {
   word32 nonce[4];     /**< nonce words 4..7 (GPU frame) */
   word32 mario;        /**< current tile index */
   word32 p;            /**< op after nonce words 0..7 (only p & 7 used) */
   word32 pad[2];       /**< padding to 32 bytes */
} PEACH_PIPE_SLOT;

/**
 * Per-batch constants, passed BY VALUE to every kernel.
*/
typedef struct {
   word32 mid[8];       /**< SHA256 state after compressing bt[0..63] */
   word32 tail[7];      /**< bt[64..91] as 7 raw LE words */
   word32 nonce_lo[4];  /**< nonce words 0..3 (host trigg_generate()) */
   word32 q;            /**< op after n0..n3 (index independent) */
   word32 diff;         /**< difficulty, clamped, > 0 */
   word32 nslots;       /**< slots in batch, <= cap, multiple of 128 */
   word32 epoch;        /**< batch epoch, copied to the result */
   word8 skip[8];       /**< per-round skip masks (bit a = drop algo a) */
} PEACH_PIPE_PARAMS;

/**
 * Per-batch result (device memory, copied back by the host).
*/
typedef struct {
   word32 found;              /**< solve flag (atomicCAS 0 -> 1) */
   word32 epoch;              /**< epoch of the batch */
   word64 completed;          /**< nonces that reached the final hash */
   word32 nonce_hi[4];        /**< solving nonce words 4..7 */
   word32 hash[8];            /**< solving final hash */
   word32 canary_valid;       /**< canary written (final queue entry 0) */
   word32 canary_nonce_hi[4]; /**< canary nonce words 4..7 */
   word32 canary_hash[8];     /**< canary final hash */
   word32 overflow;           /**< queue overflows (must stay 0) */
   word32 anomaly;            /**< device prefix NaN replacements (0) */
   word32 dropped;            /**< nonces dropped by skip masks */
   word32 pad;                /**< padding */
} PEACH_PIPE_RESULT;

/**
 * Optional per-slot trace for CPU tests (NULL in production).
*/
typedef struct {
   word32 mario[9];     /**< tile index entering round r, final in [8] */
   word8 algo[8];       /**< algorithm selected in round r */
   word8 drop_round;    /**< round the slot was dropped, 0xFF completed */
   word8 pad[3];        /**< padding */
   word32 final[8];     /**< final hash (as LE words) */
} PEACH_PIPE_TRACE;

/**
 * Device buffers of a batch context.
*/
typedef struct {
   const uint4 *d_map;        /**< Peach map, 1 GiB */
   const word16 *d_T;         /**< transition table, PEACHCACHELEN */
   word64 *d_rng;             /**< RNG state, cap */
   PEACH_PIPE_SLOT *d_slot;   /**< slot states, cap */
   word32 *d_hash;            /**< hash0 per slot, cap * 8 */
   word32 *d_q;               /**< queues, PEACH_PIPE_NQUEUE * cap:
                                 q[parity][algo] at (parity*8 + algo) * cap,
                                 final queue at 16 * cap */
   word32 *d_cnt;             /**< counters, PEACH_PIPE_NCNT * CNTPAD:
                                 cnt(r, a) = d_cnt[(r*8 + a) * 32] */
   PEACH_PIPE_RESULT *d_res;  /**< batch result */
   PEACH_PIPE_TRACE *d_trace; /**< optional trace, cap (or NULL) */
   word32 cap;                /**< queue capacity (= max slots) */
} PEACH_PIPE_BUFS;

/**
 * Launch configuration of a batch context.
*/
typedef struct {
   int block;           /**< threads per block (PEACH_PIPE_BLOCK) */
   int grid_init;       /**< blocks for the init kernel */
   int grid_final;      /**< blocks for the final kernel */
   int grid_hash[8];    /**< blocks for each hash kernel */
} PEACH_PIPE_LAUNCH;

/* compile time size checks */
typedef char peach_pipe_slot_size_check[
   sizeof(PEACH_PIPE_SLOT) == 32 ? 1 : -1];
typedef char peach_pipe_trace_size_check[
   sizeof(PEACH_PIPE_TRACE) == 80 ? 1 : -1];

/****************************************************************
 * PIPELINE KERNELS -- added by later stages, in this section:
 *    kcu_peach_pipe_transitions, kcu_peach_pipe_init,
 *    kcu_peach_pipe_hash_<algo> (x8), kcu_peach_pipe_final,
 *    kcu_peach_pipe_selftest and PEACH_HOST peach_pipe_enqueue().
 ****************************************************************/

/* end PIPELINE KERNELS */

/* end include guard */
#endif
