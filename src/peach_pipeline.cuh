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
#define PEACH_PIPE_NQUEUE     8     /**< slot queues: round 0, per algo */
#define PEACH_PIPE_CNTPAD     32    /**< words per (padded) counter */
#define PEACH_PIPE_NCNT       72    /**< counters: 9 rounds x 8 algos */
#define PEACH_PIPE_SKIP_MD2   0x40  /**< default per-round skip mask */
#define PEACH_PIPE_FINALCNT   64    /**< counter number of the final queue
                                       (round 8, algo 0) */
#define PEACH_PIPE_TILEVEC    64    /**< 16-byte vectors per map tile */
#define PEACH_PIPE_MAXCAP     WORD32_C(0x1000000)  /**< max. queue
   capacity (keeps all 8 x cap queue offsets in 32 bits) */
#define PEACH_PIPE_MAXGRID    65535  /**< max. blocks per launch; keeps
   the 32-bit stride of the block-uniform loops from wrapping */

/* Tile order of the queues of rounds 1..7 and of the final queue: a
 * counting sort of their entries by queue and tile bucket (tile >>
 * SORT_SHIFT), so the hash and final kernels read the map in ascending
 * tile order (and their entries sequentially) */
#ifndef PEACH_PIPE_SORT_SHIFT
#define PEACH_PIPE_SORT_SHIFT 8     /**< tiles per bucket: 1 << shift */
#endif
#define PEACH_PIPE_NBUCKET    (PEACHCACHELEN >> PEACH_PIPE_SORT_SHIFT)
   /**< tile buckets per histogram (4096) */
#define PEACH_PIPE_NHIST      57    /**< histograms: queue (r, a) of
   rounds r = 1..7 at (r - 1) * 8 + a, the final queue at 56 */
#define PEACH_PIPE_HISTOFF    (PEACH_PIPE_NCNT * PEACH_PIPE_CNTPAD)
   /**< word offset of the histograms in d_cnt */
#define PEACH_PIPE_CUROFF     (PEACH_PIPE_HISTOFF + \
   (PEACH_PIPE_NHIST * PEACH_PIPE_NBUCKET))
   /**< word offset of the bucket cursors (8 x NBUCKET) in d_cnt */
#define PEACH_PIPE_CNTZERO    PEACH_PIPE_CUROFF
   /**< leading words of d_cnt zeroed per batch (counters, histograms) */
#define PEACH_PIPE_CNTWORDS   (PEACH_PIPE_CUROFF + \
   (8 * PEACH_PIPE_NBUCKET))   /**< words of d_cnt */
#define PEACH_PIPE_KEYSHIFT   20    /**< key = queue << 20 | tile */
#define PEACH_PIPE_KEYDEAD    WORD32_C(0xFFFFFFFF)  /**< key of a slot
   that is not in the next round's queues */
#define PEACH_PIPE_PSHIFT     20    /**< position of P & 7 in the tile
                                       word of a slot state */
#define PEACH_PIPE_SCANGRID   8     /**< blocks of a scan launch (one per
   queue of the round) */

/* Trace values (PEACH_PIPE_TRACE) */
#define PEACH_PIPE_ALIVE      0xFF  /**< drop_round: never dropped */
#define PEACH_PIPE_LOST       0xFE  /**< drop_round: lost to a queue
                                       overflow (a bug signal) */
#define PEACH_PIPE_NOTILE     WORD32_C(0xFFFFFFFF)  /**< mario[] entry
                                       not reached */
#define PEACH_PIPE_NOALGO     0xFF  /**< algo[] entry not reached */

/* Self-test vector layout (kcu_peach_pipe_selftest()) */
#define PEACH_PIPE_SELFTEST_IN   3  /**< input words per entry: seed
                                       word, op, index */
#define PEACH_PIPE_SELFTEST_OUT  3  /**< output words per entry: op after
   peach_dflops_step(), its NaN flag (0/1), peach_dflops_incs() */

/**
 * State of a slot (nonce) in a round: of round 0 in the slot states
 * (d_slot, written by init), of rounds 1..7 and the final round in the
 * entries of the tile ordered queues (d_ent). 16 bytes, one uint4.
*/
typedef struct {
   word32 seed[2];      /**< the slot's frame random number (low, high
                           word): nonce words 4..7 are
                           peach_pipe_frame(seed) */
   word32 tile;         /**< tile index of the round | (P & 7) << 20,
                           P = op after nonce words 0..7 */
   word32 id;           /**< slot number */
} PEACH_PIPE_SLOT;

/**
 * Per-batch constants, passed BY VALUE to every kernel. 104 bytes.
 * Device code indexes the arrays with constants only, and reads the
 * skip mask of a round with a shift (never `p.skip[round]`): a runtime
 * index into a by-value kernel parameter copies the whole structure to
 * local memory.
*/
typedef struct {
   word32 mid[8];       /**< SHA256 state after compressing bt[0..63] */
   word32 tail[7];      /**< bt[64..91] as 7 raw LE words */
   word32 nonce_lo[4];  /**< nonce words 0..3 (host trigg_generate()) */
   word32 q;            /**< op after n0..n3 (index independent) */
   word32 diff;         /**< difficulty, clamped (0: every final hash
                           meets it, as in trigg_eval()) */
   word32 nslots;       /**< slots in batch, <= cap, multiple of 128 */
   word32 epoch;        /**< batch epoch, copied to the result (> 0) */
   word32 pad;          /**< padding (8-byte alignment of skip) */
   word64 skip;         /**< per-round skip masks: byte r = mask of
                           round r (bit a = drop a nonce whose jump in
                           round r uses algo a), see
                           peach_pipe_skip_pack() */
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
   word64 *d_rng;             /**< RNG state, cap (after init: the
                                 frame random number of the batch) */
   PEACH_PIPE_SLOT *d_slot;   /**< slot states of round 0, cap */
   word32 *d_hash;            /**< hash0 per slot, cap * 8 */
   word32 *d_q;               /**< round 0 queues (slot numbers),
                                 PEACH_PIPE_NQUEUE * cap: algo a at a * cap */
   PEACH_PIPE_SLOT *d_ent;    /**< entries of the tile ordered queues of
                                 the current round (1..7) or of the final
                                 queue, cap: queue a after the queues
                                 0..a-1 of its round */
   word32 *d_cnt;             /**< counters, PEACH_PIPE_CNTWORDS:
                                 cnt(r, a) = d_cnt[(r*8 + a) * 32], then
                                 the tile histograms and cursors */
   word32 *d_key;             /**< sort key per slot, cap: queue and tile
                                 of the slot's next queue entry
                                 (PEACH_PIPE_KEYDEAD: none) */
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
   sizeof(PEACH_PIPE_SLOT) == 16 ? 1 : -1];
typedef char peach_pipe_trace_size_check[
   sizeof(PEACH_PIPE_TRACE) == 80 ? 1 : -1];
typedef char peach_pipe_params_size_check[
   sizeof(PEACH_PIPE_PARAMS) == 104 ? 1 : -1];
typedef char peach_pipe_result_size_check[
   sizeof(PEACH_PIPE_RESULT) == 136 ? 1 : -1];
typedef char peach_pipe_bucket_check[(PEACH_PIPE_SORT_SHIFT >= 4 &&
   PEACH_PIPE_SORT_SHIFT <= 13) ? 1 : -1];

/**
 * @private
 * Skip mask of a round from the packed per-round masks.
 * @param skip Packed masks (PEACH_PIPE_PARAMS.skip)
 * @param round Round number, 0..7
 * @returns mask of @a round (bit a = drop jumps using algo a)
*/
PEACH_HD word32 peach_pipe_skip_mask(word64 skip, word32 round)
{
   return (word32) (skip >> (8 * (round & 7))) & 0xFF;
}  /* end peach_pipe_skip_mask() */

/**
 * Pack 8 per-round skip masks into PEACH_PIPE_PARAMS.skip.
 * @param masks Masks of rounds 0..7 (each 0..0xFE)
 * @returns packed masks, byte r = mask of round r
*/
PEACH_HD word64 peach_pipe_skip_pack(const word8 masks[8])
{
   return (word64) masks[0] | ((word64) masks[1] << 8) |
      ((word64) masks[2] << 16) | ((word64) masks[3] << 24) |
      ((word64) masks[4] << 32) | ((word64) masks[5] << 40) |
      ((word64) masks[6] << 48) | ((word64) masks[7] << 56);
}  /* end peach_pipe_skip_pack() */

/* end include guard (tables and types) */
#endif

/****************************************************************
 * PIPELINE KERNELS
 *    kcu_peach_pipe_transitions, kcu_peach_pipe_init,
 *    kcu_peach_pipe_hash_<algo> (x8), kcu_peach_pipe_scan,
 *    kcu_peach_pipe_scatter, kcu_peach_pipe_final,
 *    kcu_peach_pipe_selftest and PEACH_HOST peach_pipe_enqueue().
 * <br />
 * Compiled at the first inclusion of this header unless
 * PEACH_PIPE_DEFER_KERNELS is defined at that point; a later inclusion
 * without it compiles them (once, own guard). peach.cu defers them until
 * after the official kernels: `__constant__` data is laid out in
 * declaration order, so the pipeline's tables (Keccak round constants,
 * MD2 S-box) then follow the official kernels' sm_61 `__constant__`
 * tables, whose addresses -- and so the official kernels' SASS -- stay
 * unchanged.
 * <br />
 * Queues and counters of a batch (PEACH_PIPE_BUFS); queue (r, a) holds
 * the nonces whose jump in round r uses algo a, its length is the
 * counter cnt(r, a) (d_cnt[(r * 8 + a) * PEACH_PIPE_CNTPAD]), and the
 * final queue is queue (8, 0); counters are never reused within a batch.
 * - Round 0: init appends slot numbers to q[a] (d_q + a * cap), unordered
 *   (mario0 is tile 0 for most nonces); the hash kernels read the slot
 *   states (d_slot).
 * - Rounds 1..7 and the final queue: entries (the slot states of that
 *   round) in d_ent, queue (r, a) at offset off(r, a) = cnt(r, 0) + .. +
 *   cnt(r, a - 1), in ascending tile bucket order (tile >>
 *   PEACH_PIPE_SORT_SHIFT). A hash kernel of round r - 1 stores each
 *   surviving slot's next queue and tile as the slot's key (d_key) and
 *   counts it in that queue's tile bucket histogram; then
 *   kcu_peach_pipe_scan() turns the histograms of round r into bucket
 *   cursors and the counters cnt(r, a), and kcu_peach_pipe_scatter()
 *   sweeps the keys of all slots and stores the state of each live slot,
 *   with its next tile, at the next index of its bucket (a counting
 *   sort). The hash and final kernels so read their entries sequentially
 *   and the map in ascending tile order.
 * Each slot enters at most one queue per round, so with cap >= nslots no
 * queue can overflow; consumers still read only entries below cap, and an
 * entry at an index >= cap counts in PEACH_PIPE_RESULT.overflow.
 * <br />
 * Every kernel runs a block-uniform loop over its items:
 * `for (base = blockIdx.x * blockDim.x; base < n; base += gridDim.x *
 * blockDim.x)` with `valid = base + threadIdx.x < n`, so all threads of
 * a block iterate equally often and every warp collective is reached by
 * all 32 lanes (blockDim.x must be a multiple of 32, at most 128). Any
 * grid size >= 1 processes all items.
 ****************************************************************/

#if !defined(PEACH_PIPE_DEFER_KERNELS) && \
   !defined(MOCHIMO_PEACH_PIPELINE_KERNELS)
#define MOCHIMO_PEACH_PIPELINE_KERNELS

#include "peach_hash32.cuh"   /* for SHA-1/SHA-256/MD5, trailer, final */
#include "peach_hash64.cuh"   /* for Blake2b, SHA3, Keccak */
#include "peach_hashmd2.cuh"  /* for MD2, c_peach_md2_sbox */

/* loop to unroll fully on the device (constant trip count) */
#ifdef __CUDA_ARCH__
   #define PEACH_PIPE_UNROLL  _Pragma("unroll")
#else
   #define PEACH_PIPE_UNROLL
#endif

/**
 * @private
 * Difficulty check of a final hash, as trigg_eval() and the official
 * kcu_peach_solve(): the first @a diff bits (big-endian bit order of the
 * digest bytes) must be zero. Branch-free over constant word indices.
 * @param h Final hash as 8 little-endian words (digest bytes 4i..4i+3)
 * @param diff Difficulty, 0..255
 * @returns 1 if @a h meets @a diff, else 0
*/
PEACH_HD int peach_pipe_diff_ok(const word32 *h, word32 diff)
{
   word32 bad = 0, need, w;
   int i;

   PEACH_PIPE_UNROLL
   for (i = 0; i < 8; i++) {
      /* bits required to be zero in word i (most significant first) */
      need = diff > (word32) (32 * i) ? diff - (word32) (32 * i) : 0;
      w = peach_bswap32(h[i]);
      if (need >= 32) bad |= w;
      else if (need > 0) bad |= w >> (32 - need);
   }
   return bad == 0;
}  /* end peach_pipe_diff_ok() */

/**
 * @private
 * One self-test entry: a dflops step and the packed op increments of a
 * seed word, as used by the pipeline (see kcu_peach_pipe_selftest()).
 * @param w Seed word
 * @param op Operation code entering @a w
 * @param index Tile index (NaN replacement)
 * @param out PEACH_PIPE_SELFTEST_OUT words: peach_dflops_step() result,
 * its NaN replacement flag (0 or 1), peach_dflops_incs()
*/
PEACH_HD void peach_pipe_selftest_one(word32 w, word32 op, word32 index,
   word32 *out)
{
   int nanf = 0;

   out[0] = peach_dflops_step(w, op, index, &nanf);
   out[1] = (word32) nanf;
   out[2] = peach_dflops_incs(w, index);
}  /* end peach_pipe_selftest_one() */

/**
 * @private
 * Generate a 64-bit random number from a per-slot state, exactly as the
 * official cu_rand64() (SplitMix64 step, the output is the new state),
 * but indexed by slot instead of by thread.
 * @param state Pointer to the slot's RNG state
 * @returns 64-bit random number
*/
PEACH_DEV word64 peach_pipe_rand64(word64 *state)
{
   word64 z = (*state += WORD64_C(0x9e3779b97f4a7c15));

   z = (z ^ (z >> 30)) * WORD64_C(0xbf58476d1ce4e5b9);
   z = (z ^ (z >> 27)) * WORD64_C(0x94d049bb133111eb);
   return (*state = z ^ (z >> 31));
}  /* end peach_pipe_rand64() */

/* Nonce frame tables of peach_pipe_frame(), as bytes in one array (one
 * base address in the hash kernels): Z_ING, Z_PREP, Z_ADJ, Z_NS, Z_MASS
 * at the PEACH_PIPE_FRAME_* offsets, the same values (all below 256) */
#define PEACH_PIPE_FRAME_ING   0
#define PEACH_PIPE_FRAME_PREP  32
#define PEACH_PIPE_FRAME_ADJ   40
#define PEACH_PIPE_FRAME_NS    104
#define PEACH_PIPE_FRAME_MASS  168
#define PEACH_PIPE_FRAME_LEN   200
static __device__ const word8 c_peach_pipe_frame[PEACH_PIPE_FRAME_LEN] = {
   18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
   36, 37, 38, 39, 40, 41, 42, 43, 23, 24, 31, 32, 33, 34, 12, 13, 14, 15,
   16, 17, 12, 13, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74,
   75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92,
   94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 107, 108, 109, 110,
   112, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126,
   127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 145, 149,
   154, 155, 156, 157, 177, 178, 179, 180, 182, 183, 184, 185, 186, 187,
   188, 189, 190, 191, 192, 193, 194, 196, 197, 198, 199, 200, 201, 202,
   203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 241, 244, 245,
   246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 214, 215, 216, 217,
   218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231,
   232, 233, 234, 235, 236, 237, 238, 239, 240, 242, 214, 215, 216, 219
};

/**
 * @private
 * Nonce words 4..7 from a 64-bit random number: the official GPU nonce
 * frame of kcu_peach_solve() (always passes trigg_syntax(), and never
 * triggers a dflops NaN replacement).
 * @param s 64-bit random number
 * @param n Pointer to place nonce words 4..7 (4 words)
*/
PEACH_DEV void peach_pipe_frame(word64 s, word32 *n)
{
   const word8 *z = c_peach_pipe_frame;
   const word32 lo = (word32) s, hi = (word32) (s >> 32);

   /* the official 64-bit words, n2 = 0x10000050000 | ING | PREP << 8 |
    * ADJ << 24 | NS << 32 | MASS << 48 | ING << 56 and n3 = 0x50103 |
    * ADJ << 24 | NS << 32, as 32-bit halves (every table value is below
    * 256, so no term crosses a half) */
   n[0] = WORD32_C(0x50000) | (word32) z[PEACH_PIPE_FRAME_ING + (lo & 31)] |
      ((word32) z[PEACH_PIPE_FRAME_PREP + ((lo >> 5) & 7)] << 8) |
      ((word32) z[PEACH_PIPE_FRAME_ADJ + ((lo >> 8) & 63)] << 24);
   n[1] = WORD32_C(0x100) | (word32) z[PEACH_PIPE_FRAME_NS +
      ((lo >> 14) & 63)] |
      ((word32) z[PEACH_PIPE_FRAME_MASS + ((lo >> 20) & 31)] << 16) |
      ((word32) z[PEACH_PIPE_FRAME_ING + ((lo >> 25) & 31)] << 24);
   n[2] = WORD32_C(0x50103) | ((word32) z[PEACH_PIPE_FRAME_ADJ +
      (((lo >> 30) | (hi << 2)) & 63)] << 24);
   n[3] = (word32) z[PEACH_PIPE_FRAME_NS + ((hi >> 4) & 63)];
}  /* end peach_pipe_frame() */

/**
 * @private
 * Warp-aggregated queue append. Called by EVERY thread of the block in
 * every iteration of a block-uniform loop (all 32 lanes of each warp);
 * lanes without an item pass @a keep = @a drop = 0.
 * <br />
 * Device: 3 ballots on the key bits and one on @a keep give each lane
 * its peer group (same key, keeping); the lowest peer (leader) reserves
 * the group's entries with one atomic add and broadcasts the base with
 * __shfl_sync() from its lane; rank = peers below the lane. Dropped
 * items are counted with one more ballot (one atomic per warp).
 * CPU emulation: one atomic per item.
 * @param q Queue of key 0; the queue of key k is at q + k * cap
 * @param cnt Counter of key 0; the counter of key k is at
 * cnt + k * PEACH_PIPE_CNTPAD
 * @param cap Queue capacity
 * @param key Queue number, 0..7 (ignored unless @a keep)
 * @param keep Non-zero to append @a slot to queue @a key
 * @param drop Non-zero to count a nonce dropped by a skip mask
 * @param slot Slot number to append
 * @param res Batch result (dropped and overflow counters)
 * @returns 1 if @a slot was stored, else 0 (not kept, or overflow)
*/
PEACH_DEV int peach_pipe_append(word32 *q, word32 *cnt, word32 cap,
   word32 key, int keep, int drop, word32 slot, PEACH_PIPE_RESULT *res)
{
   word32 pos;

#ifdef __CUDA_ARCH__
   const word32 lane = threadIdx.x & 31;
   word32 b0, b1, b2, bk, bd, peers, base = 0;
   int leader;

   b0 = __ballot_sync(0xFFFFFFFFu, (int) (key & 1));
   b1 = __ballot_sync(0xFFFFFFFFu, (int) ((key >> 1) & 1));
   b2 = __ballot_sync(0xFFFFFFFFu, (int) ((key >> 2) & 1));
   bk = __ballot_sync(0xFFFFFFFFu, keep);
   bd = __ballot_sync(0xFFFFFFFFu, drop);
   /* keeping lanes with the same key (includes this lane if keep) */
   peers = bk & ((key & 1) ? b0 : ~b0) & ((key & 2) ? b1 : ~b1) &
      ((key & 4) ? b2 : ~b2);
   leader = __ffs((int) peers) - 1;
   if (keep && (int) lane == leader) {
      base = PEACH_ATOMIC_ADD32(&cnt[key * PEACH_PIPE_CNTPAD],
         __popc(peers));
   }
   /* every lane reads its own leader's base (unused unless keep) */
   base = __shfl_sync(0xFFFFFFFFu, base, leader & 31);
   pos = base + (word32) __popc(peers & ((1u << lane) - 1u));
   if (lane == 0 && bd != 0) {
      PEACH_ATOMIC_ADD32(&res->dropped, __popc(bd));
   }
#else
   pos = 0;
   if (keep) pos = PEACH_ATOMIC_ADD32(&cnt[key * PEACH_PIPE_CNTPAD], 1);
   if (drop) PEACH_ATOMIC_ADD32(&res->dropped, 1);
#endif

   if (!keep) return 0;
   if (pos < cap) {
      q[(key * cap) + pos] = slot;
      return 1;
   }
   PEACH_ATOMIC_ADD32(&res->overflow, 1);
   return 0;
}  /* end peach_pipe_append() */

/**
 * @private
 * Count completed nonces (final queue entries) of one loop iteration:
 * one ballot and one 64-bit atomic per warp on the device, one atomic
 * per entry in CPU emulation. Called by every thread (see append).
 * @param valid Non-zero if this thread processed an entry
 * @param res Batch result
*/
PEACH_DEV void peach_pipe_count(int valid, PEACH_PIPE_RESULT *res)
{
#ifdef __CUDA_ARCH__
   word32 c = __ballot_sync(0xFFFFFFFFu, valid);

   if ((threadIdx.x & 31) == 0 && c != 0) {
      PEACH_ATOMIC_ADD64(&res->completed, __popc(c));
   }
#else
   if (valid) PEACH_ATOMIC_ADD64(&res->completed, 1);
#endif
}  /* end peach_pipe_count() */

/**
 * @private
 * Count nonces dropped by a skip mask in one loop iteration: one ballot
 * and one atomic per warp on the device, one atomic per drop in CPU
 * emulation. Called by every thread (see append).
 * @param drop Non-zero if this thread dropped its nonce
 * @param res Batch result
*/
PEACH_DEV void peach_pipe_count_drop(int drop, PEACH_PIPE_RESULT *res)
{
#ifdef __CUDA_ARCH__
   word32 c = __ballot_sync(0xFFFFFFFFu, drop);

   if ((threadIdx.x & 31) == 0 && c != 0) {
      PEACH_ATOMIC_ADD32(&res->dropped, __popc(c));
   }
#else
   if (drop) PEACH_ATOMIC_ADD32(&res->dropped, 1);
#endif
}  /* end peach_pipe_count_drop() */

/**
 * @private
 * Offset of queue (r, @a a) in the entries of round r (rounds 1..8):
 * the sum of the counters of the queues before it.
 * @param cnt Counters of round r (cnt(r, 0))
 * @param a Queue (algorithm), 0..8 (8: the end of the round's queues)
 * @returns offset of the queue in d_ent
*/
PEACH_DEV word32 peach_pipe_qoff(const word32 *cnt, word32 a)
{
   word32 off = 0, i;

   PEACH_PIPE_UNROLL
   for (i = 0; i < 8; i++) {
      if (i < a) off += PEACH_LDG32(&cnt[i * PEACH_PIPE_CNTPAD]);
   }
   return off;
}  /* end peach_pipe_qoff() */

/**
 * @private
 * Entries of a queue that are stored (below the capacity).
 * @param cnt Queue counter
 * @param off Offset of the queue (0 for the queues of round 0)
 * @param cap Queue capacity
 * @returns number of stored entries
*/
PEACH_HD word32 peach_pipe_qlen(word32 cnt, word32 off, word32 cap)
{
   if (off >= cap) return 0;
   return cnt < cap - off ? cnt : cap - off;
}  /* end peach_pipe_qlen() */

/**
 * @private
 * Nighthash of a jump seed view with algorithm @a algo (a constant at
 * every call site, so only one hash survives inlining).
 * @param algo Algorithm 0..7 (peach_nighthash() numbering)
 * @param n Nonce, 8 words
 * @param m Tile index
 * @param tile Tile @a m (64 x uint4)
 * @param sbox MD2 S-box (algo 6 only; may be NULL otherwise)
 * @param out Digest as 8 little-endian words, as peach_nighthash()
*/
PEACH_DEV void peach_pipe_hash(word32 algo, const word32 *n, word32 m,
   const uint4 *tile, const word8 *sbox, word32 *out)
{
   switch (algo) {
      case 0: peach_sh_blake2b32(n, m, tile, out); break;
      case 1: peach_sh_blake2b64(n, m, tile, out); break;
      case 2: peach_sh_sha1(n, m, tile, out); break;
      case 3: peach_sh_sha256(n, m, tile, out); break;
      case 4: peach_sh_sha3(n, m, tile, out); break;
      case 5: peach_sh_keccak(n, m, tile, out); break;
      case 6: peach_sh_md2(n, m, tile, sbox, out); break;
      default: peach_sh_md5(n, m, tile, out); break;
   }
}  /* end peach_pipe_hash() */

/**
 * @private
 * Body of the hash kernel of algorithm @a algo in round @a round: for
 * each entry of queue (round, algo), hash the slot's jump seed view,
 * select the next algorithm (rounds 0..6) and, unless that round's skip
 * mask drops the slot, store its key (next queue, tile and P & 7) and
 * count it in the tile histogram of that queue; round 7 does so for the
 * final queue. A dropped slot gets the key PEACH_PIPE_KEYDEAD.
 * kcu_peach_pipe_scatter() then builds the queues of round + 1.
 * @param p Batch parameters
 * @param b Batch buffers
 * @param round Round number, 0..7
 * @param algo Algorithm of this kernel, 0..7 (constant)
 * @param sbox MD2 S-box (algo 6 only)
*/
PEACH_DEV void peach_pipe_round(PEACH_PIPE_PARAMS p, PEACH_PIPE_BUFS b,
   int round, word32 algo, const word8 *sbox)
{
   const word32 r = (word32) round & 7;
   const int last = (r == PEACHROUNDS - 1);
   const word32 *cnt = b.d_cnt + (r * 8 * PEACH_PIPE_CNTPAD);
   const word32 *qin = b.d_q + (algo * b.cap);
   const PEACH_PIPE_SLOT *ein;
   /* tile histograms of the round r + 1 queues (or the final queue) */
   word32 *hist = b.d_cnt + PEACH_PIPE_HISTOFF +
      (r * 8 * PEACH_PIPE_NBUCKET);
   word32 mask = last ? 0 : peach_pipe_skip_mask(p.skip, r + 1);
   word32 n8[8], dh[8], n, off, base, k, slot, m, pf, key;
   PEACH_PIPE_TRACE *t;
   uint4 w;
   int valid, keep, drop;

   /* round 0: slot queue q[algo]; else entries of queue (r, algo) */
   off = r == 0 ? 0 : peach_pipe_qoff(cnt, algo);
   n = peach_pipe_qlen(PEACH_LDG32(&cnt[algo * PEACH_PIPE_CNTPAD]), off,
      b.cap);
   ein = b.d_ent + off;
   for (base = blockIdx.x * blockDim.x; base < n;
         base += gridDim.x * blockDim.x) {
      k = base + threadIdx.x;
      valid = (k < n);
      slot = m = key = 0;
      keep = drop = 0;
      if (valid) {
         /* slot state: round 0 from the slot queue, else the entry */
         w = PEACH_LDG128(r == 0 ? &b.d_slot[PEACH_LDG32(&qin[k])] :
            &ein[k]);
         /* nonce: words 0..3 per batch, words 4..7 per slot */
         n8[0] = p.nonce_lo[0]; n8[1] = p.nonce_lo[1];
         n8[2] = p.nonce_lo[2]; n8[3] = p.nonce_lo[3];
         peach_pipe_frame(((word64) w.y << 32) | w.x, &n8[4]);
         m = w.z & PEACHCACHELEN_M1;
         pf = w.z >> PEACH_PIPE_PSHIFT;
         slot = w.w;
         /* jump: next tile = sum of the digest words */
         peach_pipe_hash(algo, n8, m,
            &b.d_map[(size_t) m * PEACH_PIPE_TILEVEC], sbox, dh);
         m = (dh[0] + dh[1] + dh[2] + dh[3] + dh[4] + dh[5] + dh[6] +
            dh[7]) & PEACHCACHELEN_M1;
         if (last) keep = 1;
         else {
            key = peach_select_algo(pf, PEACH_LDG16(&b.d_T[m]));
            keep = !((mask >> key) & 1);
            drop = !keep;
         }
         if (keep) {
            PEACH_ATOMIC_ADD32(&hist[(key * PEACH_PIPE_NBUCKET) +
               (m >> PEACH_PIPE_SORT_SHIFT)], 1);
            b.d_key[slot] = (key << PEACH_PIPE_KEYSHIFT) | m;
         } else b.d_key[slot] = PEACH_PIPE_KEYDEAD;
      }
      peach_pipe_count_drop(drop, b.d_res);
      if (valid && b.d_trace != NULL) {
         t = &b.d_trace[slot];
         t->mario[r + 1] = m;
         if (!last) t->algo[r + 1] = (word8) key;
         if (drop) t->drop_round = (word8) (r + 1);
      }
   }
}  /* end peach_pipe_round() */

/**
 * CUDA kernel computing transition table entries (peach_select.h) of
 * tiles offset .. offset + count - 1 (clamped to the map), one thread
 * per tile: the 4 op chains run together over the tile index word and
 * the 256 tile words (64 aligned 128-bit loads), each word computes its
 * 4 float results once and every chain selects branch-free.
 * Run after the map is complete; any grid size >= 1.
 * @param d_map Peach map (PEACHCACHELEN tiles of 64 x uint4)
 * @param d_T Transition table (PEACHCACHELEN entries)
 * @param offset First tile index
 * @param count Number of tiles
*/
PEACH_KERNEL void __launch_bounds__(PEACH_PIPE_BLOCK)
   kcu_peach_pipe_transitions(const uint4 *d_map, word16 *d_T,
   word32 offset, word32 count)
{
   const uint4 *tile;
   word32 n, k, t, ops;
   uint4 v;
   int i;

   n = offset < PEACHCACHELEN ? PEACHCACHELEN - offset : 0;
   if (count < n) n = count;
   for (k = (blockIdx.x * blockDim.x) + threadIdx.x; k < n;
         k += gridDim.x * blockDim.x) {
      t = offset + k;
      tile = &d_map[(size_t) t * PEACH_PIPE_TILEVEC];
      ops = peach_transition_step(PEACH_TRANSITION_INIT, t, t);
      for (i = 0; i < PEACH_PIPE_TILEVEC; i++) {
         v = PEACH_LDG128(&tile[i]);
         ops = peach_transition_step(ops, v.x, t);
         ops = peach_transition_step(ops, v.y, t);
         ops = peach_transition_step(ops, v.z, t);
         ops = peach_transition_step(ops, v.w, t);
      }
      d_T[t] = peach_transition_pack(ops);
   }
}  /* end kcu_peach_pipe_transitions() */

/**
 * CUDA kernel starting a batch, one slot per item (k < p.nslots): draw
 * nonce words 4..7 (official GPU frame) from the slot's RNG state,
 * hash0 = sha256(bt[0..123]) from the midstate, mario0 = product of the
 * hash0 bytes & 0xFFFFF, P = op after the nonce (from p.q over words
 * 4..7), algo0 = select(P, T[mario0]); append the slot to q[0][algo0]
 * unless skip mask 0 drops it. A NaN replacement in words 4..7 (never
 * expected) drops the slot and counts in the result's anomaly. Sets the
 * key of every slot to PEACH_PIPE_KEYDEAD (round 0 sets the keys of the
 * slots it processes).
 * Requires d_cnt and d_res zeroed (peach_pipe_enqueue()).
 * @param p Batch parameters
 * @param b Batch buffers
*/
PEACH_KERNEL void __launch_bounds__(PEACH_PIPE_BLOCK)
   kcu_peach_pipe_init(PEACH_PIPE_PARAMS p, PEACH_PIPE_BUFS b)
{
   const word32 mask = peach_pipe_skip_mask(p.skip, 0);
   word32 mid[8], tail[7], n8[8], h0[8], base, k, i, m, pf, algo;
   word64 seed;
   PEACH_PIPE_TRACE *t;
   uint4 st;
   int valid, keep, drop, stored, nanf;

   /* batch constants: constant indices only (kernel parameters) */
   PEACH_PIPE_UNROLL
   for (i = 0; i < 8; i++) mid[i] = p.mid[i];
   PEACH_PIPE_UNROLL
   for (i = 0; i < 7; i++) tail[i] = p.tail[i];
   for (base = blockIdx.x * blockDim.x; base < p.nslots;
         base += gridDim.x * blockDim.x) {
      k = base + threadIdx.x;
      valid = (k < p.nslots);
      m = algo = 0;
      keep = drop = nanf = 0;
      if (valid) {
         n8[0] = p.nonce_lo[0]; n8[1] = p.nonce_lo[1];
         n8[2] = p.nonce_lo[2]; n8[3] = p.nonce_lo[3];
         seed = peach_pipe_rand64(&b.d_rng[k]);
         peach_pipe_frame(seed, &n8[4]);
         peach_sha256_trailer(mid, tail, n8, h0);
         /* mario0 = product of the 32 hash0 bytes, mod 2^32 */
         m = 1;
         PEACH_PIPE_UNROLL
         for (i = 0; i < 8; i++) {
            m *= (h0[i] & 0xFF) * ((h0[i] >> 8) & 0xFF) *
               ((h0[i] >> 16) & 0xFF) * (h0[i] >> 24);
         }
         m &= PEACHCACHELEN_M1;
         /* P: continue the batch prefix q over nonce words 4..7 */
         pf = peach_dflops_step(n8[4], p.q, m, &nanf);
         pf = peach_dflops_step(n8[5], pf, m, &nanf);
         pf = peach_dflops_step(n8[6], pf, m, &nanf);
         pf = peach_dflops_step(n8[7], pf, m, &nanf);
         algo = peach_select_algo(pf, PEACH_LDG16(&b.d_T[m]));
         /* slot state and hash0 */
         st.x = (word32) seed;
         st.y = (word32) (seed >> 32);
         st.z = m | ((pf & 7) << PEACH_PIPE_PSHIFT);
         st.w = k;
         *((uint4 *) &b.d_slot[k]) = st;
         b.d_key[k] = PEACH_PIPE_KEYDEAD;
         PEACH_PIPE_UNROLL
         for (i = 0; i < 8; i++) b.d_hash[(k * 8) + i] = h0[i];
         if (nanf) PEACH_ATOMIC_ADD32(&b.d_res->anomaly, 1);
         else {
            keep = !((mask >> algo) & 1);
            drop = !keep;
         }
      }
      stored = peach_pipe_append(b.d_q, b.d_cnt, b.cap, algo, keep, drop,
         k, b.d_res);
      if (valid && b.d_trace != NULL) {
         t = &b.d_trace[k];
         t->mario[0] = m;
         PEACH_PIPE_UNROLL
         for (i = 1; i < 9; i++) t->mario[i] = PEACH_PIPE_NOTILE;
         t->algo[0] = (word8) algo;
         PEACH_PIPE_UNROLL
         for (i = 1; i < 8; i++) t->algo[i] = PEACH_PIPE_NOALGO;
         t->drop_round = (word8) (keep ? (stored ? PEACH_PIPE_ALIVE :
            PEACH_PIPE_LOST) : 0);
         t->pad[0] = t->pad[1] = t->pad[2] = 0;
         PEACH_PIPE_UNROLL
         for (i = 0; i < 8; i++) t->final[i] = 0;
      }
   }
}  /* end kcu_peach_pipe_init() */

/**
 * Generate a hash kernel of the pipeline (all but MD2).
 * Kernel arguments: (PEACH_PIPE_PARAMS p, PEACH_PIPE_BUFS b, int round),
 * round 0..7; consumes q[round & 1][ALGO], see peach_pipe_round().
*/
#define PEACH_PIPE_HASH_KERNEL(NAME, ALGO) \
   PEACH_KERNEL void __launch_bounds__(PEACH_PIPE_BLOCK) \
      NAME(PEACH_PIPE_PARAMS p, PEACH_PIPE_BUFS b, int round) \
   { \
      peach_pipe_round(p, b, round, (ALGO), NULL); \
   }

/** CUDA hash kernel, algo 0: Blake2b-256 keyed with 32 zero bytes. */
PEACH_PIPE_HASH_KERNEL(kcu_peach_pipe_hash_blake2b32, 0)
/** CUDA hash kernel, algo 1: Blake2b-256 keyed with 64 x 0x01. */
PEACH_PIPE_HASH_KERNEL(kcu_peach_pipe_hash_blake2b64, 1)
/** CUDA hash kernel, algo 2: SHA-1. */
PEACH_PIPE_HASH_KERNEL(kcu_peach_pipe_hash_sha1, 2)
/** CUDA hash kernel, algo 3: SHA-256. */
PEACH_PIPE_HASH_KERNEL(kcu_peach_pipe_hash_sha256, 3)
/** CUDA hash kernel, algo 4: SHA3-256. */
PEACH_PIPE_HASH_KERNEL(kcu_peach_pipe_hash_sha3, 4)
/** CUDA hash kernel, algo 5: Keccak-256. */
PEACH_PIPE_HASH_KERNEL(kcu_peach_pipe_hash_keccak, 5)
/** CUDA hash kernel, algo 7: MD5. */
PEACH_PIPE_HASH_KERNEL(kcu_peach_pipe_hash_md5, 7)

/**
 * CUDA hash kernel, algo 6: MD2. Stages the S-box in a 256-byte shared
 * table (data dependent lookups; constant memory would serialize them)
 * before the block-uniform loop of peach_pipe_round().
 * <br />
 * Its 64 state and checksum byte registers leave it needing 84..90
 * registers; without an occupancy target ptxas squeezes it into 80 (6
 * blocks of 128 threads per 64K-register SM) with stack spills on some
 * architectures. The target of 5 blocks per SM (at most 102 registers)
 * is a bound it meets without spills, not a register cap.
 * @param p Batch parameters
 * @param b Batch buffers
 * @param round Round number, 0..7
*/
PEACH_KERNEL void __launch_bounds__(PEACH_PIPE_BLOCK, 5)
   kcu_peach_pipe_hash_md2(PEACH_PIPE_PARAMS p, PEACH_PIPE_BUFS b,
   int round)
{
   const word8 *sbox;

#ifdef __CUDA_ARCH__
   __shared__ word8 s_sbox[256];
   word32 i;

   for (i = threadIdx.x; i < 256; i += blockDim.x) {
      s_sbox[i] = c_peach_md2_sbox[i];
   }
   __syncthreads();
   sbox = s_sbox;
#else
   sbox = c_peach_md2_sbox;
#endif

   peach_pipe_round(p, b, round, 6, sbox);
}  /* end kcu_peach_pipe_hash_md2() */

/**
 * CUDA kernel turning the tile histograms of the queues of round @a round
 * (1..7; 8: the final queue) into bucket cursors (exclusive prefix sums,
 * the start of each bucket in its queue) and the queue counters cnt(round,
 * a) (the totals). Block a handles queue a (round 8: block 0 only); grid
 * PEACH_PIPE_SCANGRID, any block that is a multiple of 32 (at most
 * PEACH_PIPE_BLOCK); each thread scans a contiguous run of buckets.
 * @param b Batch buffers
 * @param round Round of the queues, 1..8
*/
PEACH_KERNEL void __launch_bounds__(PEACH_PIPE_BLOCK)
   kcu_peach_pipe_scan(PEACH_PIPE_BUFS b, int round)
{
   const word32 rr = (word32) round;
   const word32 a = blockIdx.x;
   const word32 per = (PEACH_PIPE_NBUCKET + blockDim.x - 1) / blockDim.x;
   const word32 lo = threadIdx.x * per < PEACH_PIPE_NBUCKET ?
      threadIdx.x * per : PEACH_PIPE_NBUCKET;
   const word32 hi = lo + per < PEACH_PIPE_NBUCKET ?
      lo + per : PEACH_PIPE_NBUCKET;
   const word32 *h;
   word32 *cur, sum, pre, total, i;

   /* block-uniform exit: queues of the round */
   if (rr < 1 || rr > PEACHROUNDS || a >= (rr < PEACHROUNDS ? 8u : 1u)) {
      return;
   }
   h = b.d_cnt + PEACH_PIPE_HISTOFF +
      ((((rr - 1) * 8) + a) * PEACH_PIPE_NBUCKET);
   cur = b.d_cnt + PEACH_PIPE_CUROFF + (a * PEACH_PIPE_NBUCKET);
   for (sum = 0, i = lo; i < hi; i++) sum += h[i];

#ifdef __CUDA_ARCH__
   {
      __shared__ word32 s_part[PEACH_PIPE_BLOCK / 32];
      const word32 lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
      word32 x = sum, y, o;

      /* inclusive warp scan, then the totals of the warps before */
      for (o = 1; o < 32; o <<= 1) {
         y = __shfl_up_sync(0xFFFFFFFFu, x, o);
         if (lane >= o) x += y;
      }
      if (lane == 31) s_part[warp] = x;
      __syncthreads();
      pre = x - sum;
      for (total = 0, i = 0; i < (blockDim.x >> 5); i++) {
         if (i < warp) pre += s_part[i];
         total += s_part[i];
      }
   }
#else
   /* per-thread fallback: the buckets before this thread's run */
   for (pre = total = 0, i = 0; i < PEACH_PIPE_NBUCKET; i++) {
      if (i < lo) pre += h[i];
      total += h[i];
   }
#endif

   for (i = lo; i < hi; i++) {
      cur[i] = pre;
      pre += h[i];
   }
   if (threadIdx.x == 0) {
      b.d_cnt[((rr * 8) + a) * PEACH_PIPE_CNTPAD] = total;
   }
}  /* end kcu_peach_pipe_scan() */

/**
 * CUDA kernel building the queues of round @a round (1..7; 8: the final
 * queue) in tile bucket order: for each slot k < p.nslots with a live key
 * (queue a, tile m), take the next index of bucket m >> SORT_SHIFT of
 * queue a from its cursor (kcu_peach_pipe_scan()) and store there, at
 * offset off(round, a) in d_ent, the state of slot k with tile m. An
 * index >= cap (never with cap >= nslots) counts in the result's
 * overflow, kills the key and marks the slot lost. Any grid size >= 1.
 * @param p Batch parameters
 * @param b Batch buffers
 * @param round Round of the queues, 1..8
*/
PEACH_KERNEL void __launch_bounds__(PEACH_PIPE_BLOCK)
   kcu_peach_pipe_scatter(PEACH_PIPE_PARAMS p, PEACH_PIPE_BUFS b,
   int round)
{
   const word32 rr = (word32) round;
   const word32 *cnt;
   word32 *cur = b.d_cnt + PEACH_PIPE_CUROFF;
   word32 qoff[8], k, key, a, i, pos;
   uint4 e;

   if (rr < 1 || rr > PEACHROUNDS) return;
   /* queue offsets of the round (constant indices: registers) */
   cnt = b.d_cnt + (rr * 8 * PEACH_PIPE_CNTPAD);
   qoff[0] = 0;
   PEACH_PIPE_UNROLL
   for (i = 1; i < 8; i++) {
      qoff[i] = qoff[i - 1] + PEACH_LDG32(&cnt[(i - 1) * PEACH_PIPE_CNTPAD]);
   }
   for (k = (blockIdx.x * blockDim.x) + threadIdx.x; k < p.nslots;
         k += gridDim.x * blockDim.x) {
      key = b.d_key[k];
      if (key == PEACH_PIPE_KEYDEAD) continue;
      a = (key >> PEACH_PIPE_KEYSHIFT) & 7;
      pos = 0;
      PEACH_PIPE_UNROLL
      for (i = 1; i < 8; i++) {
         if (i == a) pos = qoff[i];
      }
      pos += PEACH_ATOMIC_ADD32(&cur[(a * PEACH_PIPE_NBUCKET) +
         ((key & PEACHCACHELEN_M1) >> PEACH_PIPE_SORT_SHIFT)], 1);
      if (pos < b.cap) {
         e = PEACH_LDG128(&b.d_slot[k]);
         e.z = (e.z & ~((word32) PEACHCACHELEN_M1)) |
            (key & PEACHCACHELEN_M1);
         *((uint4 *) &b.d_ent[pos]) = e;
      } else {
         PEACH_ATOMIC_ADD32(&b.d_res->overflow, 1);
         b.d_key[k] = PEACH_PIPE_KEYDEAD;
         if (b.d_trace != NULL) b.d_trace[k].drop_round = PEACH_PIPE_LOST;
      }
   }
}  /* end kcu_peach_pipe_scatter() */

/**
 * CUDA kernel finishing a batch: for each final queue entry, final =
 * sha256(hash0 || tile[mario]); counts completed nonces (per warp);
 * entry 0 writes the canary (nonce words 4..7 and final hash); the first
 * entry meeting p.diff wins atomicCAS(found, 0, 1) and writes its nonce
 * words 4..7 and final hash. Writes the batch epoch to the result.
 * @param p Batch parameters
 * @param b Batch buffers
*/
PEACH_KERNEL void __launch_bounds__(PEACH_PIPE_BLOCK)
   kcu_peach_pipe_final(PEACH_PIPE_PARAMS p, PEACH_PIPE_BUFS b)
{
   PEACH_PIPE_RESULT *res = b.d_res;
   word32 h0[8], out[8], n4[4], n, base, k, slot, m;
   uint4 v, w, nv;
   int valid, i;

   if (blockIdx.x == 0 && threadIdx.x == 0) res->epoch = p.epoch;
   n = peach_pipe_qlen(PEACH_LDG32(&b.d_cnt[PEACH_PIPE_FINALCNT *
      PEACH_PIPE_CNTPAD]), 0, b.cap);
   for (base = blockIdx.x * blockDim.x; base < n;
         base += gridDim.x * blockDim.x) {
      k = base + threadIdx.x;
      valid = (k < n);
      if (valid) {
         /* final queue entry: nonce words 4..7, final tile, slot */
         w = PEACH_LDG128(&b.d_ent[k]);
         peach_pipe_frame(((word64) w.y << 32) | w.x, n4);
         nv.x = n4[0]; nv.y = n4[1]; nv.z = n4[2]; nv.w = n4[3];
         m = w.z & PEACHCACHELEN_M1;
         slot = w.w;
         v = PEACH_LDG128(&b.d_hash[slot * 8]);
         h0[0] = v.x; h0[1] = v.y; h0[2] = v.z; h0[3] = v.w;
         v = PEACH_LDG128(&b.d_hash[(slot * 8) + 4]);
         h0[4] = v.x; h0[5] = v.y; h0[6] = v.z; h0[7] = v.w;
         peach_sha256_final(h0, &b.d_map[(size_t) m * PEACH_PIPE_TILEVEC],
            out);
         if (b.d_trace != NULL) {
            PEACH_PIPE_UNROLL
            for (i = 0; i < 8; i++) b.d_trace[slot].final[i] = out[i];
         }
         /* canary: final queue entry 0 */
         if (k == 0) {
            res->canary_nonce_hi[0] = nv.x;
            res->canary_nonce_hi[1] = nv.y;
            res->canary_nonce_hi[2] = nv.z;
            res->canary_nonce_hi[3] = nv.w;
            PEACH_PIPE_UNROLL
            for (i = 0; i < 8; i++) res->canary_hash[i] = out[i];
            res->canary_valid = 1;
         }
         /* solve: first entry meeting the difficulty */
         if (peach_pipe_diff_ok(out, p.diff) &&
               PEACH_ATOMIC_CAS32(&res->found, 0, 1) == 0) {
            res->nonce_hi[0] = nv.x;
            res->nonce_hi[1] = nv.y;
            res->nonce_hi[2] = nv.z;
            res->nonce_hi[3] = nv.w;
            PEACH_PIPE_UNROLL
            for (i = 0; i < 8; i++) res->hash[i] = out[i];
         }
      }
      peach_pipe_count(valid, res);
   }
}  /* end kcu_peach_pipe_final() */

/**
 * CUDA kernel for the device self-test of the dflops arithmetic used by
 * the pipeline (FTZ / fast-math detection): for each entry k < count,
 * peach_pipe_selftest_one() of d_in[3k] (seed word), d_in[3k + 1] (op),
 * d_in[3k + 2] (index) into d_out[3k .. 3k + 2]; the host compares with
 * peach_pipe_selftest_one() on the CPU. Any grid size >= 1.
 * @param d_in Input entries (PEACH_PIPE_SELFTEST_IN words each)
 * @param d_out Output entries (PEACH_PIPE_SELFTEST_OUT words each)
 * @param count Number of entries
*/
PEACH_KERNEL void __launch_bounds__(PEACH_PIPE_BLOCK)
   kcu_peach_pipe_selftest(const word32 *d_in, word32 *d_out, int count)
{
   word32 k;

   for (k = (blockIdx.x * blockDim.x) + threadIdx.x;
         count > 0 && k < (word32) count; k += gridDim.x * blockDim.x) {
      peach_pipe_selftest_one(d_in[(k * PEACH_PIPE_SELFTEST_IN)],
         d_in[(k * PEACH_PIPE_SELFTEST_IN) + 1],
         d_in[(k * PEACH_PIPE_SELFTEST_IN) + 2],
         &d_out[k * PEACH_PIPE_SELFTEST_OUT]);
   }
}  /* end kcu_peach_pipe_selftest() */

/**
 * @private
 * Launch the hash kernel of algorithm @a algo for round @a round.
 * @note Internal to peach_pipe_enqueue(), which validates the launch
 * shape: the kernels' full-warp collectives require a block that is a
 * multiple of 32 (at most PEACH_PIPE_BLOCK). Do not call it directly.
*/
PEACH_HOST void peach_pipe_launch_hash(int algo, int grid, int block,
   cudaStream_t s, const PEACH_PIPE_PARAMS *p, const PEACH_PIPE_BUFS *b,
   int round)
{
   switch (algo) {
      case 0:
         CUDA_KERNEL(kcu_peach_pipe_hash_blake2b32, grid, block, 0, s)
            (*p, *b, round);
         break;
      case 1:
         CUDA_KERNEL(kcu_peach_pipe_hash_blake2b64, grid, block, 0, s)
            (*p, *b, round);
         break;
      case 2:
         CUDA_KERNEL(kcu_peach_pipe_hash_sha1, grid, block, 0, s)
            (*p, *b, round);
         break;
      case 3:
         CUDA_KERNEL(kcu_peach_pipe_hash_sha256, grid, block, 0, s)
            (*p, *b, round);
         break;
      case 4:
         CUDA_KERNEL(kcu_peach_pipe_hash_sha3, grid, block, 0, s)
            (*p, *b, round);
         break;
      case 5:
         CUDA_KERNEL(kcu_peach_pipe_hash_keccak, grid, block, 0, s)
            (*p, *b, round);
         break;
      case 6:
         CUDA_KERNEL(kcu_peach_pipe_hash_md2, grid, block, 0, s)
            (*p, *b, round);
         break;
      default:
         CUDA_KERNEL(kcu_peach_pipe_hash_md5, grid, block, 0, s)
            (*p, *b, round);
         break;
   }
}  /* end peach_pipe_launch_hash() */

/**
 * @private
 * Launch the tile sort of the queues of round @a round (1..7; 8: the
 * final queue): kcu_peach_pipe_scan(), then kcu_peach_pipe_scatter()
 * (grid l->grid_init, as the init kernel: one item per slot).
 * @note Internal to peach_pipe_enqueue() (validated launch shape).
 * @returns cudaSuccess, or the error of a failed launch
*/
PEACH_HOST cudaError_t peach_pipe_launch_sort(const PEACH_PIPE_PARAMS *p,
   const PEACH_PIPE_BUFS *b, const PEACH_PIPE_LAUNCH *l, int round,
   cudaStream_t s)
{
   cudaError_t err;

   CUDA_KERNEL(kcu_peach_pipe_scan, PEACH_PIPE_SCANGRID, PEACH_PIPE_BLOCK,
      0, s)(*b, round);
   err = cudaGetLastError();
   if (err != cudaSuccess) return err;
   CUDA_KERNEL(kcu_peach_pipe_scatter, l->grid_init, l->block, 0, s)
      (*p, *b, round);
   return cudaGetLastError();
}  /* end peach_pipe_launch_sort() */

/**
 * Enqueue one pipeline batch on a stream: zero the counters, histograms
 * and the result, then init, for every round the tile sort of its queues
 * (rounds 1..7) and the hash kernels for each algorithm not in that
 * round's skip mask, then the tile sort of the final queue and final.
 * Shared by peach.cu and the CPU tests (emulated launches run
 * synchronously). The caller copies the result back after the stream
 * completes.
 * @param p Batch parameters (nslots <= b->cap; p->epoch > 0)
 * @param b Batch buffers (cap 1..PEACH_PIPE_MAXCAP)
 * @param l Launch configuration (block: a multiple of 32, at most
 * PEACH_PIPE_BLOCK; every grid 1..PEACH_PIPE_MAXGRID)
 * @param s Stream
 * @returns 0 on success, -1 on invalid arguments (nothing enqueued),
 * else the (non-zero) cudaError_t of the failed CUDA call
*/
PEACH_HOST int peach_pipe_enqueue(const PEACH_PIPE_PARAMS *p,
   const PEACH_PIPE_BUFS *b, const PEACH_PIPE_LAUNCH *l, cudaStream_t s)
{
   cudaError_t err;
   word32 mask;
   int r, a;

   /* validate (a bad launch shape would break the warp collectives) */
   if (p == NULL || b == NULL || l == NULL) return (-1);
   if (l->block < 32 || l->block > PEACH_PIPE_BLOCK || (l->block & 31)) {
      return (-1);
   }
   if (l->grid_init < 1 || l->grid_init > PEACH_PIPE_MAXGRID) return (-1);
   if (l->grid_final < 1 || l->grid_final > PEACH_PIPE_MAXGRID) return (-1);
   for (a = 0; a < 8; a++) {
      if (l->grid_hash[a] < 1 || l->grid_hash[a] > PEACH_PIPE_MAXGRID) {
         return (-1);
      }
   }
   if (b->cap < 1 || b->cap > PEACH_PIPE_MAXCAP || p->nslots > b->cap) {
      return (-1);
   }

   err = PEACH_MEMSET_ASYNC(b->d_cnt, 0,
      sizeof(word32) * PEACH_PIPE_CNTZERO, s);
   if (err != cudaSuccess) return (int) err;
   err = PEACH_MEMSET_ASYNC(b->d_res, 0, sizeof(PEACH_PIPE_RESULT), s);
   if (err != cudaSuccess) return (int) err;
   CUDA_KERNEL(kcu_peach_pipe_init, l->grid_init, l->block, 0, s)(*p, *b);
   err = cudaGetLastError();
   if (err != cudaSuccess) return (int) err;
   for (r = 0; r < PEACHROUNDS; r++) {
      if (r > 0) {
         err = peach_pipe_launch_sort(p, b, l, r, s);
         if (err != cudaSuccess) return (int) err;
      }
      mask = peach_pipe_skip_mask(p->skip, (word32) r);
      for (a = 0; a < 8; a++) {
         if ((mask >> a) & 1) continue;
         peach_pipe_launch_hash(a, l->grid_hash[a], l->block, s, p, b, r);
         err = cudaGetLastError();
         if (err != cudaSuccess) return (int) err;
      }
   }
   err = peach_pipe_launch_sort(p, b, l, PEACHROUNDS, s);
   if (err != cudaSuccess) return (int) err;
   CUDA_KERNEL(kcu_peach_pipe_final, l->grid_final, l->block, 0, s)
      (*p, *b);
   err = cudaGetLastError();

   return (int) err;
}  /* end peach_pipe_enqueue() */

/* end PIPELINE KERNELS */
#endif
