/**
 * @private
 * @headerfile peach.cuh <peach.cuh>
 * @copyright Adequate Systems LLC, 2018-2022. All Rights Reserved.
 * <br />For license information, please refer to ../LICENSE.md
 * @note A note on variable naming in this file...
 * [2] suffix implies dual stream variables; for maximum GPU utilisation
 * h_ prefix implies host memory; requires cudaMallocHost()
 * c_ prefix implies (constant) device memory
 * g_ prefix implies (global) device memory
 * d_ prefix implies device memory
 * *h_ pointers require cudaMallocHost();
 * *d_/*g_ pointers require cudaMalloc();
*/

/* include guard */
#ifndef MOCHIMO_PEACH_CU
#define MOCHIMO_PEACH_CU


#include "peach.cuh"

#include <stdlib.h>  /* for getenv(), calloc(), free() */

/* external support */
#include "extint.h"
#include "extmath.h"
#include "error.h"
/* external support -- Nighthash */
#include "blake2b.h"
#include "md2.h"
#include "md5.h"
#include "sha1.h"
#include "sha256.cu"
#include "sha3.h"

/* Peach pipeline support -- also provides cuCONSTn860, the nonce frame
 * tables (Z_ING, Z_NS, Z_MASS, Z_PREP, Z_ADJ) and cu_rand64() */
#include "peach_compat.cuh"
#include "peach_select.h"
#define PEACH_PIPE_DEFER_KERNELS  /* kernels: after the official ones */
#include "peach_pipeline.cuh"

/* Peach CUDA solver modes (PEACH_CUDA_CTX.mode) */
#define PEACH_CUDA_MODE_LEGACY      0  /**< official kcu_peach_solve() */
#define PEACH_CUDA_MODE_PIPELINE    1  /**< queued pipeline kernels */

/* Pipeline batch sizing (see peach_cuda_sizing()) */
#define PEACH_CUDA_NCTX_MAX         4  /**< max. batch contexts, one
   stream each; automatic: as many as memory allows for full batches */
#define PEACH_CUDA_NCTX_MIN         2  /**< min. batch contexts of the
   automatic choice */
#define PEACH_CUDA_SLOT_BYTES       ( 2 * sizeof(PEACH_PIPE_SLOT) + \
   8 * sizeof(word32) + sizeof(word64) + sizeof(word32) + \
   PEACH_PIPE_NQUEUE * sizeof(word32) )   /**< device bytes per slot and
   context: slot state 16 + queue entry 16 + hash0 32 + rng 8 + key 4 +
   8 round 0 queues x 4 = 108 */
#define PEACH_CUDA_SLOTS_PER_THREAD 32 /**< default slots in flight per
   resident thread, over all batch contexts (N = 32 x SMs x
   maxThreadsPerSM / contexts): the GPU work queued between two polls of
   the caller does not depend on the number of contexts */
#define PEACH_CUDA_MD2_SHARE        3  /**< with 3 or more contexts, the
   MD2 hash kernel (compute bound) runs 1/3 of its resident blocks per
   multiprocessor (at least one), so that the (memory bound) kernels of
   the other contexts share the multiprocessors with it */
#define PEACH_CUDA_MEM_PERCENT      80 /**< max. share of free memory used
   by the batch contexts, in percent */
#define PEACH_CUDA_BATCH_MAX        WORD32_C(0x1000000)  /**< hard upper
   bound of slots per batch (keeps 17 x N queue indices in 32 bits) */
#define PEACH_CUDA_T_BYTES   (sizeof(word16) * PEACHCACHELEN)  /**< T */

/* Pipeline solver safety and tuning */
#define PEACH_CUDA_REDRAW_MAX       64   /**< first nonce half draws per
   batch before the batch is skipped (host redraw rule) */
#define PEACH_CUDA_BATCH_MS         250  /**< batch time limit, in ms; two
   slower batches in a row halve the slots of later batches. Measured
   from enqueue to completion while the batches of the other contexts
   share the GPU, so about (contexts) times the time of the batch alone:
   it bounds the GPU time in flight, which a block change waits for */
#define PEACH_CUDA_BATCH_GROW       8    /**< consecutive batches faster
   than 2/5 of PEACH_CUDA_BATCH_MS (a doubled batch stays well below the
   limit) that double the slots again (up to cap) */
#define PEACH_CUDA_TRAILER_ID       92   /**< leading block trailer bytes
   that identify a solve target (phash .. mroot, before the nonce):
   gpuminer sends a solve only if they match its current or previous
   trailer */
#define PEACH_CUDA_POLL_GAP         1    /**< seconds (of time()) between
   two pipeline solver calls that count as a pause of the caller: a
   trailer change seen after a pause may hide trailers this solver never
   saw, so batches of the trailer before it report no solve */
#define PEACH_CUDA_CANARY_FIRST     16   /**< canary check of every batch
   up to this many batches after a map build ... */
#define PEACH_CUDA_CANARY_EVERY     64   /**< ... then of every 64th */
#define PEACH_CUDA_SELFTEST_TILES   6    /**< T entries checked by the
   self-test: tiles 0, 1, 0xFFFFF and 3 random tiles */
#define PEACH_CUDA_SELFTEST_RANDOM  32   /**< random seed words checked by
   the self-test, in addition to the special values */
#define PEACH_CUDA_SELFTEST_MAX     1024 /**< max. self-test entries */
#define PEACH_CUDA_SELFTEST_BYTES   ( sizeof(word32) * \
   PEACH_CUDA_SELFTEST_MAX * (PEACH_PIPE_SELFTEST_IN + \
   PEACH_PIPE_SELFTEST_OUT) )   /**< self-test buffer (in, then out) */

/* Block trailer snapshot attempts (see peach_cuda_snapshot()) */
#define PEACH_CUDA_SNAPSHOT_TRIES   64

/**
 * @private
 * Peach CUDA context. Managed internally by cross referencing parameters
 * of DEVICE_CTX passed to functions. Allocated zeroed, so a NULL pointer
 * (or stream/event handle) always means "not allocated".
*/
typedef struct {
   /* official (legacy) solver */
   cudaStream_t stream[PEACH_CUDA_NCTX_MAX];  /**< asynchronous streams:
                                          0 and 1 (legacy solver, map
                                          build), one per pipeline batch
                                          context */
   BTRAILER *h_bt[PEACH_CUDA_NCTX_MAX];  /**< BTRAILER (current), one per
                                          stream (pinned) */
   BTRAILER *d_bt[2];                  /**< BTRAILER (legacy, device) */
   word64 *h_solve[2], *d_solve[2];    /**< solve seeds */
   word64 *d_state[2];                 /**< PRNG state */
   word64 *d_map;                      /**< Peach Map */
   word32 *d_phash;                    /**< previous hash */
   word8 diff_inflight[2];             /**< clamped difficulty of each
                                          stream's in-flight launch */
   /* configuration, read once by peach_init_cuda_device() */
   int cfg_legacy;                     /**< MCM_PEACH_LEGACY (1 = legacy) */
   word8 cfg_skip[8];                  /**< MCM_PEACH_SKIP, per round */
   word32 cfg_batch;                   /**< MCM_PEACH_BATCH (0 = auto) */
   int cfg_nctx;                       /**< MCM_PEACH_STREAMS (0 = auto) */
   /* solver mode and safety */
   int mode;                           /**< PEACH_CUDA_MODE_* in use */
   int fallback;                       /**< pipeline abandoned for the
                                          legacy solver (until re-init) */
   word64 bad_solves;                  /**< candidates rejected by the
                                          CPU verification */
   word64 bad_canaries;                /**< canary results rejected by the
                                          CPU verification */
   /* device properties and pipeline sizing */
   int sms;                            /**< multiprocessor count */
   int max_threads_sm;                 /**< max threads per multiproc. */
   int nctx;                           /**< pipeline batch contexts, 1..
                                          PEACH_CUDA_NCTX_MAX (streams
                                          0 .. nctx - 1) */
   word32 cap;                         /**< slots per context (queue
                                          capacity N) */
   word32 nslots;                      /**< slots per batch (adaptive,
                                          nslots_min <= nslots <= cap) */
   word32 nslots_min;                  /**< lower bound of nslots */
   int slow_batches;                   /**< consecutive slow batches */
   int fast_batches;                   /**< consecutive fast batches */
   time_t last_poll;                   /**< time of the previous pipeline
                                          solver call (0 = none) */
   word8 trail_curr[PEACH_CUDA_TRAILER_ID];  /**< identifying bytes of
                                          the latest trailer seen */
   word8 trail_prev[PEACH_CUDA_TRAILER_ID];  /**< ... of the distinct
                                          trailer seen before it */
   int trail_prev_ok;                  /**< trail_prev is the caller's
                                          previous trailer too: the
                                          change was seen without a
                                          pause (PEACH_CUDA_POLL_GAP) */
   /* pipeline: per device */
   word16 *d_T;                        /**< transition table, T[tile] */
   word32 *d_selftest;                 /**< self-test vectors: input,
                                          then output entries */
   word8 map_phash[HASHLEN];           /**< phash of the map (and T) */
   word32 build_next;                  /**< map/T build cursor (next
                                          tile); ctx->work counts hashes */
   word32 epoch;                       /**< epoch, bumped on DEV_INIT and
                                          on every batch launch (never 0) */
   PEACH_PIPE_LAUNCH launch;           /**< launch configuration */
   /* pipeline: per batch context (one per stream) */
   PEACH_PIPE_PARAMS params[PEACH_CUDA_NCTX_MAX];  /**< in-flight batch
                                          parameters */
   PEACH_PIPE_BUFS bufs[PEACH_CUDA_NCTX_MAX];  /**< batch device buffers
                                          (d_map, d_T alias the per device
                                          ones) */
   PEACH_PIPE_RESULT *h_res[PEACH_CUDA_NCTX_MAX];  /**< pinned copy of the
                                          result */
   cudaEvent_t ev_start[PEACH_CUDA_NCTX_MAX];  /**< batch start events */
   cudaEvent_t ev_stop[PEACH_CUDA_NCTX_MAX];   /**< batch stop events */
   word32 epoch_inflight[PEACH_CUDA_NCTX_MAX];  /**< epoch of the
                                          in-flight batch */
   int inflight[PEACH_CUDA_NCTX_MAX];  /**< batch launched, unharvested */
   /* pipeline: counters */
   word64 batches;                     /**< batches since last DEV_INIT */
   word64 batches_total;               /**< batches since init */
   word64 canary_checks;               /**< canary verifications done */
} PEACH_CUDA_CTX;

/**
 * @private
 * 256-bit Blake2b (w/ key) computation optimized for the Peach algorithm.
 * Places the resulting hash in @a out.
 * @param in Pointer to data to hash
 * @param inlen Length of @a in data, in bytes
 * @param keylen Length of optional @a key input, in bytes
 * @param out Pointer to location to place the message digest
*/
__device__ void cu_peach_blake2b(const word64 *in, size_t inlen, int keylen,
   word64 *out)
{
   /* Blake2b compression constant */
   cuCONSTn860 static word8 c_sigma[12][16] = {
      { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 },
      { 14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3 },
      { 11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4 },
      { 7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8 },
      { 9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13 },
      { 2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9 },
      { 12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11 },
      { 13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10 },
      { 6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5 },
      { 10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0 },
      { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 },
      { 14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3 }
   };

   /* blake2b_init - outlen is always 256-bits in Peach */
   word64 v[16];
   word64 state[8];
   word64 final[16];
   word64 t[2] = { 128, 0 };

   /* FAST-FORWARD state to known keylen states */
   if (keylen == 64) {
      state[0] = WORD64_C(0x00B8AA23C261EF69);
      state[1] = WORD64_C(0xD38AE6ABCA237B9E);
      state[2] = WORD64_C(0x67FB881E5EE89069);
      state[3] = WORD64_C(0x3E5B8BD06B58D002);
      state[4] = WORD64_C(0x252D3F68395AAE91);
      state[5] = WORD64_C(0xD25465E23C6C1B27);
      state[6] = WORD64_C(0x852B4CC2E13303B5);
      state[7] = WORD64_C(0x3F38B9FF245BE7C1);
   } else {
      state[0] = WORD64_C(0x63320ACE264383EB);
      state[1] = WORD64_C(0x012AF5FD045A2737);
      state[2] = WORD64_C(0xF4F49C55E6BE39DF);
      state[3] = WORD64_C(0x791C5BC8AFFB11A7);
      state[4] = WORD64_C(0xC9BCACC002C0EA21);
      state[5] = WORD64_C(0x8295B8ABE2FDEDD6);
      state[6] = WORD64_C(0xB711490E5F9F41C8);
      state[7] = WORD64_C(0x3F8E4D1D9EBEAF1A);
   }

   /* blake2b_update */
   for(; inlen > 128; inlen -= 128, in = &in[16]) {
      t[0] += 128;
      blake2b_compress_init(v, state, t, 0);
      blake2b_compress_rounds(v, in, c_sigma);
      blake2b_compress_set(v, state);
   }

   /* blake2b_final - somewhat conveniently (and exclusive to Peach)...
    * the remaining datalen will always be 36... */
   final[0] = in[0];
   final[1] = in[1];
   final[2] = in[2];
   final[3] = in[3];
   final[4] = (word64) ((word32 *) in)[8];
   final[5] = 0;
   final[6] = 0;
   final[7] = 0;
   final[8] = 0;
   final[9] = 0;
   final[10] = 0;
   final[11] = 0;
   final[12] = 0;
   final[13] = 0;
   final[14] = 0;
   final[15] = 0;

   t[0] += 36;
   blake2b_compress_init(v, state, t, 1);
   blake2b_compress_rounds(v, final, c_sigma);

   /* blake2b_output */
   out[0] = state[0] ^ v[0] ^ v[8];
   out[1] = state[1] ^ v[1] ^ v[9];
   out[2] = state[2] ^ v[2] ^ v[10];
   out[3] = state[3] ^ v[3] ^ v[11];
}  /* end cu_peach_blake2b() */

/**
 * @private
 * 128-bit MD2 computation optimized for the Peach algorithm.
 * Places the resulting hash in @a out.
 * @param in Pointer to data to hash
 * @param inlen Length of @a in data, in bytes
 * @param out Pointer to location to place the message digest
*/
__device__ void cu_peach_md2(const word64 *in, size_t inlen, word64 *out)
{
   /* MD2 transformation constant */
   cuCONSTn860 static word8 s[256] = {
      41, 46, 67, 201, 162, 216, 124, 1, 61, 54, 84, 161, 236, 240, 6,
      19, 98, 167, 5, 243, 192, 199, 115, 140, 152, 147, 43, 217, 188,
      76, 130, 202, 30, 155, 87, 60, 253, 212, 224, 22, 103, 66, 111, 24,
      138, 23, 229, 18, 190, 78, 196, 214, 218, 158, 222, 73, 160, 251,
      245, 142, 187, 47, 238, 122, 169, 104, 121, 145, 21, 178, 7, 63,
      148, 194, 16, 137, 11, 34, 95, 33, 128, 127, 93, 154, 90, 144, 50,
      39, 53, 62, 204, 231, 191, 247, 151, 3, 255, 25, 48, 179, 72, 165,
      181, 209, 215, 94, 146, 42, 172, 86, 170, 198, 79, 184, 56, 210,
      150, 164, 125, 182, 118, 252, 107, 226, 156, 116, 4, 241, 69, 157,
      112, 89, 100, 113, 135, 32, 134, 91, 207, 101, 230, 45, 168, 2, 27,
      96, 37, 173, 174, 176, 185, 246, 28, 70, 97, 105, 52, 64, 126, 15,
      85, 71, 163, 35, 221, 81, 175, 58, 195, 92, 249, 206, 186, 197,
      234, 38, 44, 83, 13, 110, 133, 40, 132, 9, 211, 223, 205, 244, 65,
      129, 77, 82, 106, 220, 55, 200, 108, 193, 171, 250, 36, 225, 123,
      8, 12, 189, 177, 74, 120, 136, 149, 139, 227, 99, 232, 109, 233,
      203, 213, 254, 59, 0, 29, 57, 242, 239, 183, 14, 102, 88, 208, 228,
      166, 119, 114, 248, 235, 117, 75, 10, 49, 68, 80, 180, 143, 237,
      31, 26, 219, 153, 141, 51, 159, 17, 131, 20
   };

   /* md2_init */
   word64 state[6] = { 0 };
   word64 checksum[2] = { 0 };
   word64 pad64;
   word8 pad;

   /* prepare padding */
   pad = 16 - (inlen & 0xf);
   pad64 = pad | pad << 8;
   pad64 = pad64 | pad64 << 16;
   pad64 = pad64 | pad64 << 32;

   /* md2_update */
   for (; inlen >= 16; inlen -= 16, in = &in[2]) {
      md2_transform_init64(state, in);
      md2_transform_checksum(((word8 *) checksum), ((word8 *) in), s);
      md2_transform_state(((word8 *) state), s);
   }

   /* md2_final - only 4 bytes left, so 12 remaining bytes are pad */
   state[4] = (state[2] = *((word32 *) in) | (pad64 << 32)) ^ state[0];
   state[5] = (state[3] = pad64) ^ state[1];
   /* final transform part1 */
   md2_transform_checksum(((word8 *) checksum), ((word8 *) &state[2]), s);
   md2_transform_state(((word8 *) state), s);
   /* final transform part2 */
   md2_transform_init64(state, checksum);
   md2_transform_state(((word8 *) state), s);

   /* MD2 hash = 128 bits, zero fill remaining... */
   out[0] = state[0];
   out[1] = state[1];
   out[2] = 0;
   out[3] = 0;
}  /* end cu_peach_md2 */

/**
 * @private
 * 128-bit MD5 computation optimized for the Peach algorithm.
 * Places the resulting hash in @a out.
 * @param in Pointer to data to hash
 * @param inlen Length of @a in data, in bytes
 * @param out Pointer to location to place the message digest
*/
__device__ void cu_peach_md5(const word32 *in, size_t inlen, word64 *out)
{
   /* md5_init */
   word32 final[16];
   word32 state[4] = { 0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476 };

   /* prepare bitlen in final data */
   final[14] = inlen << 3;

   /* md5_update */
   for (; inlen >= 64; inlen -= 64, in = &in[16]) {
      md5_tranform_unrolled(state, in);
   }

   /* md5_final - somewhat conveniently (and exclusive to Peach)...
    * the remaining datalen will always be 36, so:
    * in[9] = 0x80; and in[10+] = 0; */
   final[0] = in[0];
   final[1] = in[1];
   final[2] = in[2];
   final[3] = in[3];
   final[4] = in[4];
   final[5] = in[5];
   final[6] = in[6];
   final[7] = in[7];
   final[8] = in[8];
   final[9] = 0x80;
   final[10] = 0;
   final[11] = 0;
   final[12] = 0;
   final[13] = 0;
   final[15] = 0;

   md5_tranform_unrolled(state, final);

   /* MD5 hash = 128 bits, zero fill remaining... */
   out[0] = ((word64 *) state)[0];
   out[1] = ((word64 *) state)[1];
   out[2] = 0;
   out[3] = 0;
}  /* end cuda_peach_md5 */

/**
 * @private
 * 160-bit Sha1 computation optimized for the Peach algorithm.
 * Places the resulting hash in @a out.
 * @param in Pointer to data to hash
 * @param inlen Length of @a in data, in bytes
 * @param out Pointer to location to place the message digest
*/
__device__ void cu_peach_sha1(const word32 *in, size_t inlen, word32 *out)
{
   /* SHA1 transformation constant */
   cuCONSTn860 static word32 c_k[4] =
      { 0x5a827999, 0x6ed9eba1, 0x8f1bbcdc, 0xca62c1d6 };
   /* sha1_init */
   word32 state[5] =
      { 0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0 };
   word32 final[16];

   final[15] = inlen << 3;
   final[15] = bswap32(final[15]);

   /* sha1_update */
   for(; inlen >= 64; inlen -= 64, in = &in[16]) {
      sha1_transform_unrolled(state, in, c_k);
   }

   /* sha1_final - somewhat conveniently (and exclusive to Peach)...
    * the remaining datalen will always be 36, so in[9] = 0x80. */
   final[0] = in[0];
   final[1] = in[1];
   final[2] = in[2];
   final[3] = in[3];
   final[4] = in[4];
   final[5] = in[5];
   final[6] = in[6];
   final[7] = in[7];
   final[8] = in[8];
   final[9] = 0x80;
   final[10] = 0;
   final[11] = 0;
   final[12] = 0;
   final[13] = 0;
   final[14] = 0;

   sha1_transform_unrolled(state, final, c_k);

   /* SHA1 hash = 160 bits, zero fill remaining... */
   out[0] = bswap32(state[0]);
   out[1] = bswap32(state[1]);
   out[2] = bswap32(state[2]);
   out[3] = bswap32(state[3]);
   out[4] = bswap32(state[4]);
   out[5] = 0;
   out[6] = 0;
   out[7] = 0;
}  /* end cu_peach_sha1() */

/**
 * @private
 * 256-bit SHA256 computation optimized for the Peach algorithm.
 * Places the resulting hash in @a out.
 * @param in Pointer to data to hash
 * @param inlen Length of @a in data, in bytes
 * @param out Pointer to location to place the message digest
*/
__device__ void cu_peach_sha256(const word32 *in, size_t inlen, word32 *out)
{
   /* SHA256 transformation constant */
   cuCONSTn860 static word32 c_k[64] = {
      0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
      0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
      0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
      0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
      0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
      0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
      0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
      0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
      0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
      0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
      0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
      0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
      0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
      0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
      0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
      0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
   };

   /* sha256_init */
   word32 final[16];
   word32 state[8] = {
      0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
      0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
   };

   /* prepare bitlen in final */
   final[15] = inlen << 3;
   final[15] = bswap32(final[15]);

   /* sha256_update */
   for(; inlen >= 64; inlen -= 64, in = &in[16]) {
      sha256_tranform_unrolled(state, in, c_k);
   }

   /* sha256_final - somewhat conveniently (and exclusive to Peach)...
    * the remaining datalen will always be 36, so in[9] = 0x80. */
   final[0] = in[0];
   final[1] = in[1];
   final[2] = in[2];
   final[3] = in[3];
   final[4] = in[4];
   final[5] = in[5];
   final[6] = in[6];
   final[7] = in[7];
   final[8] = in[8];
   final[9] = 0x80;
   final[10] = 0;
   final[11] = 0;
   final[12] = 0;
   final[13] = 0;
   final[14] = 0;
   sha256_tranform_unrolled(state, final, c_k);

   /* Since this implementation uses little endian byte ordering and
    * SHA uses big endian, reverse all the bytes when copying the
    * final state to the output hash. */
   out[0] = bswap32(state[0]);
   out[1] = bswap32(state[1]);
   out[2] = bswap32(state[2]);
   out[3] = bswap32(state[3]);
   out[4] = bswap32(state[4]);
   out[5] = bswap32(state[5]);
   out[6] = bswap32(state[6]);
   out[7] = bswap32(state[7]);
}  /* end cu_peach_sha256() */

/**
 * @private
 * 256-bit Sha3 (Keccak) computation optimized for the Peach algorithm.
 * Places the resulting hash in @a out.
 * @param in Pointer to data to hash
 * @param inlen Length of @a in data, in bytes
 * @param keccak_final Flag indicates hash should be finalized as Keccak
 * @param out Pointer to location to place the message digest
*/
__device__ void cu_peach_sha3(const word64 *in, size_t inlen,
   int keccak_final, word64 *out)
{
   /* Keccak permutation constant */
   cuCONSTn860 static word64 keccakf_rndc[24] = {
      WORD64_C(0x0000000000000001), WORD64_C(0x0000000000008082),
      WORD64_C(0x800000000000808a), WORD64_C(0x8000000080008000),
      WORD64_C(0x000000000000808b), WORD64_C(0x0000000080000001),
      WORD64_C(0x8000000080008081), WORD64_C(0x8000000000008009),
      WORD64_C(0x000000000000008a), WORD64_C(0x0000000000000088),
      WORD64_C(0x0000000080008009), WORD64_C(0x000000008000000a),
      WORD64_C(0x000000008000808b), WORD64_C(0x800000000000008b),
      WORD64_C(0x8000000000008089), WORD64_C(0x8000000000008003),
      WORD64_C(0x8000000000008002), WORD64_C(0x8000000000000080),
      WORD64_C(0x000000000000800a), WORD64_C(0x800000008000000a),
      WORD64_C(0x8000000080008081), WORD64_C(0x8000000000008080),
      WORD64_C(0x0000000080000001), WORD64_C(0x8000000080008008)
   };

   /* sha3_init */
   word8 state[200] = { 0 };
   word64 *st64 = (word64 *) state;
	int i;

   /* sha3_update - 136 is ctx->rsiz, fill only 17x 64-bit words in state */
   for(; inlen >= 136; inlen -= 136, in = &in[17]) {
      for (i = 0; i < 17; i++) st64[i] ^= in[i];
	   sha3_keccakf_unrolled(st64, keccakf_rndc);
   }

   /* sha3_final */
   st64[0] ^= in[0];
   st64[1] ^= in[1];
   st64[2] ^= in[2];
   st64[3] ^= in[3];
   if (inlen > PEACHGENLEN) {
      st64[4] ^= in[4];
      st64[5] ^= in[5];
      st64[6] ^= in[6];
      st64[7] ^= in[7];
      st64[8] ^= in[8];
      st64[9] ^= in[9];
      st64[10] ^= in[10];
      st64[11] ^= in[11];
      st64[12] ^= in[12];
      ((word32 *) st64)[26] ^= ((word32 *) in)[26];
      state[108] ^= keccak_final ? 0x01 : 0x06;
   } else {
      ((word32 *) st64)[8] ^= ((word32 *) in)[8];
      state[36] ^= keccak_final ? 0x01 : 0x06;
   }
   state[135] ^= 0x80;
	sha3_keccakf_unrolled(st64, keccakf_rndc);

   /* sha3_output */
   out[0] = st64[0];
   out[1] = st64[1];
   out[2] = st64[2];
   out[3] = st64[3];
}  /* end cu_peach_sha3 */

/**
 * @private
 * Perform deterministic (single precision) floating point operations on
 * @a len bytes of @a data (in 4 byte operations).
 * @param data Pointer to data to use in operations
 * @param len Length of @a data to use in operations
 * @param index Peach tile index number
 * @param txf Flag indicates @a data should be transformed by operations
 * @returns 32-bit unsigned operation code for subsequent Peach algo steps
 * @note Operations are guaranteed "deterministic" within the Peach
 * algorithm for all IEEE-754 compliant hardware on "round-to-nearest"
 * rounding mode. This is ensured by the use of CUDA intrinsics:
 * - __fdiv_rn(), __fmul_rn(), __fsub_rn(), __fadd_rn() operations, and
 * - __int2float_rn(), __uint2float_rn() conversions
*/
__device__ word32 cu_peach_dflops(void *data, size_t len,
   word32 index, int txf)
{
   cuCONSTn860 static word32 c_float[4] = {
      WORD32_C(0x26C34), WORD32_C(0x14198),
      WORD32_C(0x3D6EC), WORD32_C(0x80000000)
   };
   word8 *bp;
   float *flp, temp, flv;
   int32 operand;
   word32 op;
   unsigned i;
   word8 shift;

   /* process entire length of input data; limit to 4 byte multiples */
   /* len = len - (len & 3); // uncomment if (len % 4 != 0) is expected */
   for (op = i = 0; i < len; i += 4) {
      bp = &((word8 *) data)[i];
      if (txf) {
         /* input data is modified directly */
         flp = (float *) bp;
      } else {
         /* temp variable is modified, input data is unchanged */
         temp = *((float *) bp);
         flp = &temp;
      }
      /* first byte allocated to determine shift amount */
      shift = ((*bp & 7) + 1) << 1;
      /* remaining bytes are selected for 3 different operations based on
       * the first bytes resulting shift on precomputed contants to...
       * ... 1) determine the floating point operation type */
      op += bp[((c_float[0] >> shift) & 3)];
      /* ... 2) determine the value of the operand */
      operand = bp[((c_float[1] >> shift) & 3)];
      /* ... 3) determine the upper most bit of the operand
       *        NOTE: must be performed AFTER the allocation of the operand */
      if (bp[((c_float[2] >> shift) & 3)] & 1) operand ^= c_float[3];
      /* interpret operand as SIGNED integer and cast to float */
      flv = __int2float_rn(operand);
      /* Replace pre-operation NaN with index */
      if (isnan(*flp)) *flp = __uint2float_rn(index);
      /* Perform predetermined floating point operation */
      switch (op & 3) {
         case 3: *flp = __fdiv_rn(*flp, flv);  break;
         case 2: *flp = __fmul_rn(*flp, flv);  break;
         case 1: *flp = __fsub_rn(*flp, flv);  break;
         case 0: *flp = __fadd_rn(*flp, flv);  break;
      }
      /* Replace post-operation NaN with index */
      if (isnan(*flp)) *flp = __uint2float_rn(index);
      /* Add result of the operation to `op` as an array of bytes */
      bp = (word8 *) flp;
      op += bp[0];
      op += bp[1];
      op += bp[2];
      op += bp[3];
   }  /* end for(i = 0; ... */

   return op;
}  /* end cu_peach_dflops() */

/**
 * @private
 * Perform deterministic memory transformations on @a len bytes of @a data.
 * @param data Pointer to data to use in operations
 * @param len Length of @a data to use in operations
 * @param op Operating code from previous Peach algo steps
 * @returns 32-bit unsigned operation code for subsequent Peach algo steps
*/
__device__ word32 cu_peach_dmemtx(void *data, size_t len, word32 op)
{
   cuCONSTn860 static word64 c_flip64 = WORD64_C(0x8181818181818181);
   cuCONSTn860 static word32 c_flip32 = WORD64_C(0x81818181);
   word64 *qp = (word64 *) data;
   word32 *dp = (word32 *) data;
   word8 *bp = (word8 *) data;
   size_t len16, len32, len64, y;
   unsigned i, z;
   word8 temp;

   /* prepare memory pointers and lengths */
   len64 = (len32 = (len16 = len >> 1) >> 1) >> 1;
   /* perform memory transformations multiple times */
   for (i = 0; i < PEACHROUNDS; i++) {
      /* determine operation to use for this iteration */
      op += bp[i];
      /* select "random" transformation based on value of `op` */
      switch (op & 7) {
         case 0:  /* flip the first and last bit in every byte */
            for (z = 0; z < len64; z++) qp[z] ^= c_flip64;
            for (z <<= 1; z < len32; z++) dp[z] ^= c_flip32;
            break;
         case 1:  /* Swap bytes */
            for (y = len16, z = 0; z < len16; y++, z++) {
               temp = bp[z]; bp[z] = bp[y]; bp[y] = temp;
            }
            break;
         case 2:  /* 1's complement, all bytes */
            for (z = 0; z < len64; z++) qp[z] = ~qp[z];
            for (z <<= 1; z < len32; z++) dp[z] = ~dp[z];
            break;
         case 3:  /* Alternate +1 and -1 on all bytes */
            for (z = 0; z < len; z++) bp[z] += (z & 1) ? -1 : 1;
            break;
         case 4:  /* Alternate -i and +i on all bytes */
            for (z = 0; z < len; z++) bp[z] += (word8) ((z & 1) ? i : -i);
            break;
         case 5:  /* Replace every occurrence of 104 with 72 */ 
            for (z = 0; z < len; z++) if(bp[z] == 104) bp[z] = 72;
            break;
         case 6:  /* If byte a is > byte b, swap them. */
            for (y = len16, z = 0; z < len16; y++, z++) {
               if(bp[z] > bp[y]) {
                  temp = bp[z]; bp[z] = bp[y]; bp[y] = temp;
               }
            }
            break;
         case 7:  /* XOR all bytes */
            for (y = 0, z = 1; z < len; y++, z++) bp[z] ^= bp[y];
            break;
      } /* end switch(op & 7)... */
   } /* end for(i = 0; ... */

   return op;
}  /* end cu_peach_dmemtx() */

/**
 * @private
 * Perform Nighthash on @a inlen bytes of @a in and place result in @a out.
 * Utilizes deterministic float operations and memory transformations.
 * @param in Pointer to input data
 * @param inlen Length of data from @a in, used in non-transform steps
 * @param index Peach tile index number
 * @param txlen Length of data from @a in, used in transform steps
 * @param out Pointer to location to place resulting hash
*/
__device__ void cu_peach_nighthash(word64 *in, size_t inlen,
   word32 index, size_t txlen, word64 *out)
{
   /* Perform flops to determine initial algo type.
    * When txlen is non-zero the transformation of input data is enabled,
    * as well as the additional memory transformation process. */
   if (txlen) {
      index = cu_peach_dflops(in, txlen, index, 1);
      index = cu_peach_dmemtx(in, txlen, index);
   } else index = cu_peach_dflops(in, inlen, index, 0);

   /* reduce algorithm selection to 1 of 8 choices */
   switch (index & 7) {
      case 0: cu_peach_blake2b(in, inlen, 32, out); break;
      case 1: cu_peach_blake2b(in, inlen, 64, out); break;
      case 2: cu_peach_sha1((word32 *) in, inlen, (word32 *) out); break;
      case 3: cu_peach_sha256((word32 *) in, inlen, (word32 *) out); break;
      case 4: cu_peach_sha3(in, inlen, 0, out); break;
      case 5: cu_peach_sha3(in, inlen, 1, out); break;
      case 6: cu_peach_md2(in, inlen, out); break;
      case 7: cu_peach_md5((word32 *) in, inlen, out); break;
   }  /* end switch(algo_type)... */
}  /* end cu_peach_nighthash() */

/**
 * @private
 * Generate a tile of the Peach map.
 * @param index Index number of tile to generate
 * @param tilep Pointer to location to place generated tile
*/
__device__ void cu_peach_generate
   (word32 index, word64 *tilep, word32 *phash)
{
   int i;

   /* place initial data into seed */
   ((word32 *) tilep)[0] = index;
   ((word32 *) tilep)[1] = phash[0];
   ((word32 *) tilep)[2] = phash[1];
   ((word32 *) tilep)[3] = phash[2];
   ((word32 *) tilep)[4] = phash[3];
   ((word32 *) tilep)[5] = phash[4];
   ((word32 *) tilep)[6] = phash[5];
   ((word32 *) tilep)[7] = phash[6];
   ((word32 *) tilep)[8] = phash[7];
   /* perform initial nighthash into first row of tile */
   cu_peach_nighthash(tilep, PEACHGENLEN, index, PEACHGENLEN, tilep);
   /* fill the rest of the tile with the preceding Nighthash result */
   for (i = 0; i < (PEACHTILELEN64 - 4); i += 4) {
      tilep[i + 4] = index;
      cu_peach_nighthash(&tilep[i], PEACHGENLEN, index, SHA256LEN,
         &tilep[i + 4]);
   }
}  /* end cu_peach_generate() */

/**
 * @private
 * Perform an index jump using the hash result of the Nighthash function.
 * @param index Index number of (current) tile on Peach map
 * @param nonce Nonce for use as entropy in jump direction
 * @param tilep Pointer to tile data at @a index
 * @returns 32-bit unsigned index of next tile
*/
__device__ void cu_peach_jump(word32 *index, word64 *nonce, word64 *tilep)
{
   word64 seed[(PEACHJUMPLEN / 8) + 1];
   word32 *dp = (word32 *) seed;
   int i;

   /* construct seed for use as Nighthash input for this index on the map */
   seed[0] = nonce[0];
   seed[1] = nonce[1];
   seed[2] = nonce[2];
   seed[3] = nonce[3];
   dp[8] = *index;
#pragma unroll
   for (i = 0; i < PEACHTILELEN32; i++) {
      dp[i + 9] = ((word32 *) tilep)[i];
   }

   /* perform nighthash on PEACHJUMPLEN bytes of seed */
   cu_peach_nighthash(seed, PEACHJUMPLEN, *index, 0, seed);
   /* sum hash as 8x 32-bit unsigned integers */
   *index = dp[0] + dp[1] + dp[2] + dp[3] + dp[4] + dp[5] + dp[6] + dp[7];
   *index &= PEACHCACHELEN_M1;
}  /* end cu_peach_jump() */

/**
 * CUDA kernel for bulk generation of Peach Map tiles.
 * @param d_map Device pointer to location of Peach Map
 * @param offset Index number offset to generate tiles from
 */
__global__ void kcu_peach_build
   (word32 offset, word64 *d_map, word32 *d_phash)
{
   const word32 index = ((blockDim.x * blockIdx.x) + threadIdx.x) + offset;
   if (index < PEACHCACHELEN) {
      cu_peach_generate(index, &d_map[index * PEACHTILELEN64], d_phash);
   }
}  /* end kcu_peach_build() */

/**
 * CUDA kernel to expand a seed into a long state for parallel cu_rand64().
 * State generation based on SplitMix64 by Sebastiano Vigna.
 * @param d_seed Device pointer to location of state
 * @param seed 64-bit unsigned integer value of seed
*/
__global__ void kcu_srand64(word64 *d_state, word64 seed)
{
   word64 index = (blockDim.x * blockIdx.x) + threadIdx.x;
   d_state[index] = (seed ^ (index * WORD64_C(0x9e3779b97f4a7c15))) * WORD64_C(0xc6bc279692b5c323);
}  /* end kcu_srand64() */

/**
 * CUDA kernel for solving a tokenized haiku as nonce output for Peach proof
 * of work. Combine haiku protocols implemented in the Trigg Algorithm with
 * the memory intensive protocols of the Peach algorithm to generate haiku
 * output as proof of work.
 * @param d_map Device pointer to Peach Map
 * @param d_ictx Device pointer to incomplete hashing contexts
 * @param d_solve Device pointer to location to place nonce on solve
*/
__global__ void kcu_peach_solve
   (word64 *d_map, BTRAILER *d_bt, word64 *d_state, word8 diff, word64 *d_solve)
{
   SHA256_CTX ictx;
   word64 nonce[4], seed;
   word8 hash[SHA256LEN];
   word32 *x, mario, i;

   /* extract nonce from trailer and seed list*/
   for (i = 0; i < 4; i++) {
      ((word32 *) nonce)[i] = ((word32 *) d_bt->nonce)[i];
   }

   /* generate last half of nonce from seed (w/ largest known frame) */
   seed = cu_rand64(d_state);
   nonce[2] = WORD64_C(0x10000050000) | /* nonce8bit[2,5] */
       Z_ING[(seed     )  & 31]       | /* nonce8bit[0] */
      Z_PREP[(seed >> 5)  &  7] <<  8 | /* nonce8bit[1] */
       Z_ADJ[(seed >> 8)  & 63] << 24 | /* nonce8bit[3] */
        Z_NS[(seed >> 14) & 63] << 32 | /* nonce8bit[4] */
      Z_MASS[(seed >> 20) & 31] << 48 | /* nonce8bit[6] */
       Z_ING[(seed >> 25) & 31] << 56;  /* nonce8bit[7] */
   nonce[3] =       WORD64_C(0x50103) | /* nonce8bit[8:10] */
       Z_ADJ[(seed >> 30) & 63] << 24 | /* nonce8bit[11] */
        Z_NS[(seed >> 36) & 63] << 32;  /* nonce8bit[12] */

   /* sha256 hash trailer and nonce */
   cu_sha256_init(&ictx);
   cu_sha256_update(&ictx, d_bt, 92);
   cu_sha256_update(&ictx, nonce, 32);
   cu_sha256_final(&ictx, hash);
   /* initialize mario's starting index on the map, bound to PEACHCACHELEN */
   for (mario = hash[0], i = 1; i < SHA256LEN; i++) {
      mario *= hash[i];
   }
   mario &= PEACHCACHELEN_M1;
   /* perform tile jumps to find the final tile x8 */
   for (i = 0; i < PEACHROUNDS; i++) {
      cu_peach_jump(&mario, nonce, &d_map[mario * PEACHTILELEN64]);
   }
   /* hash block trailer with final tile */
   cu_sha256_init(&ictx);
   cu_sha256_update(&ictx, hash, SHA256LEN);
   cu_sha256_update(&ictx, &d_map[mario * PEACHTILELEN64], PEACHTILELEN);
   cu_sha256_final(&ictx, hash);
   /* Coarse/Fine evaluation checks */
   x = (word32 *) hash;
   for (i = diff >> 5; i; i--) if(*(x++) != 0) return;
   if (__clz(__byte_perm(*x, 0, 0x0123)) < (diff & 31)) return;

   /* check first to solve with atomic solve handling */
   if (!atomicCAS((int *) d_solve, 0, *((int *) nonce))) {
      d_solve[0] = nonce[0];
      d_solve[1] = nonce[1];
      d_solve[2] = nonce[2];
      d_solve[3] = nonce[3];
   }
}  /* end kcu_peach_solve() */

/**
 * CUDA kernel for checking Peach Proof-of-Work. The haiku must be
 * syntactically correct AND have the right vibe. Also, entropy MUST match
 * difficulty.
 * @param ictx Device pointer to incomplete hashing context
 * @param out Pointer to location to place final hash
 * @param eval Evaluation result: VEOK on success, else VERROR
*/
__global__ void kcu_peach_checkhash
   (BTRAILER *d_bt, word8 *d_out, word8 *d_eval)
{
   word64 data[(SHA256LEN + PEACHTILELEN) / 8] = { 0 };
   word64 nonce[4];
   BTRAILER *btp;
   word8 *hash = (word8 *) data;
   word64 *tile = (word64 *) &data[SHA256LEN / 8];
   word32 *x, mario;
   unsigned int tid;
   int i;

   /* init */
   tid = (blockDim.x * blockIdx.x) + threadIdx.x;
   btp = &d_bt[tid];

   /* copy nonce */
#pragma unroll
   for (i = 0; i < 8; i++) {
      ((word32 *) nonce)[i] = ((word32 *) btp->nonce)[i];
   }

   /* hash partial trailer */
   cu_sha256(btp, 124, hash);
   /* initialize mario's starting index on the map, bound to PEACHCACHELEN */
   for(mario = hash[0], i = 1; i < SHA256LEN; i++) mario *= hash[i];
   mario &= PEACHCACHELEN_M1;
   /* generate and perform tile jumps to find the final tile x8 */
   for (i = 0; i < PEACHROUNDS; i++) {
      cu_peach_generate(mario, tile, (word32 *) btp->phash);
      cu_peach_jump(&mario, nonce, tile);
   }
   /* generate the last tile */
   cu_peach_generate(mario, tile, (word32 *) btp->phash);
   /* hash bthash and final tile */
   cu_sha256(data, SHA256LEN + PEACHTILELEN, hash);
   /* pass final hash to out */
   memcpy(&d_out[SHA256LEN * tid], hash, SHA256LEN);
   /* Coarse/Fine evaluation checks */
   x = (word32 *) hash;
   i = btp->difficulty[0] >> 5;
   for (; i; i--) if(*(x++) != 0) { *d_eval = 1; return; }
   if (__clz(__byte_perm(*x, 0, 0x0123)) < (btp->difficulty[0] & 31)) {
      *d_eval = 1;
      return;
   }
}  /* end kcu_peach_checkhash() */

/* Peach pipeline kernels, compiled after the official kernels so that the
 * official sm_61 __constant__ tables keep their addresses (see
 * peach_pipeline.cuh) */
#undef PEACH_PIPE_DEFER_KERNELS
#include "peach_pipeline.cuh"

/**
 * @private
 * Take a stable snapshot of a block trailer that another thread may
 * update concurrently: copy it twice (volatile reads, so the two copies
 * are really taken from @a bt) until both copies are equal.
 * @note Best effort, not a consistency guarantee: without the writer's
 * cooperation (gpuminer updates its trailer with a plain memcpy()), this
 * detects a copy in progress during the reads, but not a writer that was
 * preempted partway through its copy (both reads then see the same mix
 * of old and new bytes). Every solve is verified on the CPU against the
 * snapshot it was searched for, so such a mixed trailer cannot yield an
 * invalid solve; at worst (very unlikely) a valid solve for a trailer
 * the caller never had, which it cannot send.
 * @param bt Pointer to (shared) block trailer
 * @param snap Pointer to location to place the snapshot
 * @returns VEOK on success, else VERROR (no stable copy was obtained)
*/
static int peach_cuda_snapshot(const BTRAILER *bt, BTRAILER *snap)
{
   const volatile word8 *src = (const volatile word8 *) bt;
   word8 check[sizeof(BTRAILER)];
   word8 *dst = (word8 *) snap;
   size_t n;
   int tries;

   for (tries = 0; tries < PEACH_CUDA_SNAPSHOT_TRIES; tries++) {
      for (n = 0; n < sizeof(BTRAILER); n++) dst[n] = src[n];
      for (n = 0; n < sizeof(BTRAILER); n++) check[n] = src[n];
      if (memcmp(dst, check, sizeof(BTRAILER)) == 0) return VEOK;
   }

   return VERROR;
}  /* end peach_cuda_snapshot() */

/**
 * @private
 * Verify a solve candidate on the CPU, with the consensus reference
 * peach_checkhash(), before it may be reported. A rejected candidate is
 * alerted and counted (P->bad_solves, or P->bad_canaries for a canary),
 * and must never be reported.
 * @param ctx Pointer to DEVICE_CTX the candidate was found by
 * @param P Pointer to the Peach CUDA context of @a ctx
 * @param cand Pointer to candidate block trailer (incl. complete nonce)
 * @param diff Difficulty the candidate was searched with (clamped)
 * @param hash Pointer to the final hash reported by the device (compared
 * with the reference final hash), or NULL if not available
 * @param canary Non-zero when @a cand is a pipeline canary (a completed
 * nonce checked at difficulty 0), not a solve; changes only the alert
 * and the counter
 * @returns VEOK if the candidate is a valid solve, else VERROR
*/
static int peach_cuda_verify(DEVICE_CTX *ctx, PEACH_CUDA_CTX *P,
   const BTRAILER *cand, word8 diff, const void *hash, int canary)
{
   word8 out[SHA256LEN];

   if (peach_checkhash(cand, diff, out) == VEOK) {
      if (hash == NULL || memcmp(out, hash, SHA256LEN) == 0) return VEOK;
   }

   if (canary) {
      P->bad_canaries++;
      palert("CUDA #%d: canary result does not match the CPU reference;"
         " %llu canary mismatch(es) so far", ctx->id,
         (unsigned long long) P->bad_canaries);
   } else {
      P->bad_solves++;
      palert("CUDA #%d: solve REJECTED by CPU verification (diff %u)%s;"
         " %llu rejected so far", ctx->id, (unsigned) diff,
         hash ? " or final hash mismatch" : "",
         (unsigned long long) P->bad_solves);
   }

   return VERROR;
}  /* end peach_cuda_verify() */

/**
 * @private
 * Strictly parse an unsigned integer from the characters [@a s, @a e):
 * decimal, or hexadecimal with a 0x/0X prefix. Surrounding blanks are
 * ignored; anything else (incl. an empty number) is invalid.
 * @param s Pointer to first character
 * @param e Pointer to the end of the characters (exclusive)
 * @param max Largest valid value
 * @param value Pointer to location to place the value (on success only)
 * @returns 0 on success, else (-1) if invalid or greater than @a max
*/
static int peach_cuda_parse_uint(const char *s, const char *e,
   word32 max, word32 *value)
{
   word32 base, digit, v;
   int digits;

   while (s < e && (*s == ' ' || *s == '\t')) s++;
   while (e > s && (e[-1] == ' ' || e[-1] == '\t')) e--;
   base = 10;
   if (e - s > 2 && s[0] == '0' && (s[1] == 'x' || s[1] == 'X')) {
      base = 16;
      s += 2;
   }
   for (v = 0, digits = 0; s < e; s++, digits++) {
      if (*s >= '0' && *s <= '9') digit = (word32) (*s - '0');
      else if (base == 16 && *s >= 'a' && *s <= 'f') {
         digit = (word32) (*s - 'a' + 10);
      } else if (base == 16 && *s >= 'A' && *s <= 'F') {
         digit = (word32) (*s - 'A' + 10);
      } else return (-1);
      /* reject v * base + digit > max (without overflow) */
      if (digit > max || v > (max - digit) / base) return (-1);
      v = v * base + digit;
   }
   if (digits == 0) return (-1);

   *value = v;
   return 0;
}  /* end peach_cuda_parse_uint() */

/**
 * @private
 * Read the Peach CUDA solver configuration from the environment into
 * @a P (once per device, by peach_init_cuda_device()). An invalid value
 * is ignored with a warning, and its default is used instead:
 * - MCM_PEACH_LEGACY=0|1 -- 1 selects the official (legacy) solver;
 *   default 0 (pipeline solver, where available);
 * - MCM_PEACH_SKIP=<mask> or 8 comma separated masks (round 0 first) --
 *   per-round skip masks of the pipeline solver: bit a drops a nonce
 *   whose jump in that round would use algorithm a (6 = MD2). One mask
 *   applies to all 8 rounds. Each mask 0..0xFE (0xFF would drop every
 *   nonce); default 0x40 in every round;
 * - MCM_PEACH_BATCH=<slots> -- slots per pipeline batch, 0 = automatic;
 *   default 0 (see peach_cuda_sizing() for clamping);
 * - MCM_PEACH_STREAMS=<n> -- pipeline batch contexts (one stream each),
 *   1..PEACH_CUDA_NCTX_MAX, 0 = automatic; default 0 (see
 *   peach_cuda_sizing()).
 * Numbers are decimal, or hexadecimal with a 0x prefix. An empty
 * variable counts as unset.
 * @param ctx Pointer to DEVICE_CTX (for log messages)
 * @param P Pointer to Peach CUDA context to configure
*/
static void peach_cuda_config(DEVICE_CTX *ctx, PEACH_CUDA_CTX *P)
{
   const char *str, *s, *e;
   word32 value, masks[8];
   int i, n;

   /* defaults */
   P->cfg_legacy = 0;
   for (i = 0; i < 8; i++) P->cfg_skip[i] = PEACH_PIPE_SKIP_MD2;
   P->cfg_batch = 0;
   P->cfg_nctx = 0;

   /* MCM_PEACH_LEGACY=0|1 */
   str = getenv("MCM_PEACH_LEGACY");
   if (str != NULL && *str != '\0') {
      if (peach_cuda_parse_uint(str, str + strlen(str), 1, &value) == 0) {
         P->cfg_legacy = (int) value;
      } else {
         pwarn("CUDA #%d: ignoring invalid MCM_PEACH_LEGACY=\"%.32s\""
            " (expected 0 or 1)", ctx->id, str);
      }
   }

   /* MCM_PEACH_SKIP=<mask>[,<mask> x 7] */
   str = getenv("MCM_PEACH_SKIP");
   if (str != NULL && *str != '\0') {
      for (n = 0, s = str; ; s = e + 1) {
         e = strchr(s, ',');
         if (e == NULL) e = s + strlen(s);
         if (n >= 8 || peach_cuda_parse_uint(s, e, 0xFE, &masks[n])) {
            n = -1;  /* too many masks, or an invalid mask */
            break;
         }
         n++;
         if (*e == '\0') break;
      }
      if (n == 1) for (i = 1; i < 8; i++) masks[i] = masks[0];
      if (n == 1 || n == 8) {
         for (i = 0; i < 8; i++) P->cfg_skip[i] = (word8) masks[i];
      } else {
         pwarn("CUDA #%d: ignoring invalid MCM_PEACH_SKIP=\"%.80s\""
            " (expected 1 or 8 comma separated masks, each 0..0xFE)",
            ctx->id, str);
      }
   }

   /* MCM_PEACH_BATCH=<slots> */
   str = getenv("MCM_PEACH_BATCH");
   if (str != NULL && *str != '\0') {
      if (peach_cuda_parse_uint(str, str + strlen(str),
            WORD32_C(0xFFFFFFFF), &value) == 0) {
         P->cfg_batch = value;
      } else {
         pwarn("CUDA #%d: ignoring invalid MCM_PEACH_BATCH=\"%.32s\""
            " (expected a number of slots, 0 = automatic)", ctx->id, str);
      }
   }

   /* MCM_PEACH_STREAMS=<n> */
   str = getenv("MCM_PEACH_STREAMS");
   if (str != NULL && *str != '\0') {
      if (peach_cuda_parse_uint(str, str + strlen(str), PEACH_CUDA_NCTX_MAX,
            &value) == 0) {
         P->cfg_nctx = (int) value;
      } else {
         pwarn("CUDA #%d: ignoring invalid MCM_PEACH_STREAMS=\"%.32s\""
            " (expected 0..%d, 0 = automatic)", ctx->id, str,
            PEACH_CUDA_NCTX_MAX);
      }
   }
}  /* end peach_cuda_config() */

/**
 * @private
 * Size the pipeline batch contexts of the current CUDA device. Slots per
 * batch N = PEACH_CUDA_SLOTS_PER_THREAD x SMs x maxThreadsPerSM / batch
 * contexts, or MCM_PEACH_BATCH when set, clamped to [128 x SMs, memory
 * limit] and
 * rounded to whole blocks of 128 slots; the memory limit wins when both
 * bounds conflict. The memory limit fits the batch contexts, of
 * PEACH_CUDA_SLOT_BYTES per slot each, into PEACH_CUDA_MEM_PERCENT
 * percent of the free device memory, minus @a reserve bytes still to be
 * allocated, and never exceeds PEACH_CUDA_BATCH_MAX slots. Batch
 * contexts: MCM_PEACH_STREAMS when set, else the most of
 * PEACH_CUDA_NCTX_MAX .. PEACH_CUDA_NCTX_MIN whose memory limit still
 * fits N (else PEACH_CUDA_NCTX_MIN, with a smaller N).
 * Sets P->sms, P->max_threads_sm, P->nctx, P->cap, P->nslots and
 * P->nslots_min.
 * @param ctx Pointer to DEVICE_CTX (current CUDA device)
 * @param P Pointer to Peach CUDA context
 * @param reserve Device memory, in bytes, to leave for later allocations
 * (e.g. the transition table, when it is not allocated yet)
 * @returns VEOK on success, else VERROR (CUDA error, or not enough
 * memory for one block of slots); a CUDA error is cleared, not sticky
*/
static int peach_cuda_sizing(DEVICE_CTX *ctx, PEACH_CUDA_CTX *P,
   size_t reserve)
{
   size_t mfree, mtotal, avail, limit;
   word64 want, n;
   word32 lo, hi;
   int nctx;
   cudaError_t err;

   mfree = mtotal = 0;
   err = cudaDeviceGetAttribute(&(P->sms),
      cudaDevAttrMultiProcessorCount, ctx->id);
   if (err == cudaSuccess) {
      err = cudaDeviceGetAttribute(&(P->max_threads_sm),
         cudaDevAttrMaxThreadsPerMultiProcessor, ctx->id);
   }
   if (err == cudaSuccess) err = cudaMemGetInfo(&mfree, &mtotal);
   if (err != cudaSuccess) {
      pwarn("CUDA #%d: pipeline sizing failed: (%d) %s", ctx->id,
         (int) err, cudaGetErrorString(err));
      /* clear the (non-sticky) error for later cudaGetLastError() */
      (void) cudaGetLastError();
      return VERROR;
   }
   if (P->sms < 1 || P->max_threads_sm < 1) {
      pwarn("CUDA #%d: pipeline sizing failed: %d SMs, %d threads/SM",
         ctx->id, P->sms, P->max_threads_sm);
      return VERROR;
   }

   /* batch contexts, slots per batch and memory limit, in whole blocks
    * of slots: the configured contexts, else the most whose memory limit
    * still fits the requested (or default) slots per batch, at least one
    * block per SM */
   lo = (word32) P->sms * PEACH_PIPE_BLOCK;
   avail = mfree > reserve ? mfree - reserve : 0;
   for (nctx = P->cfg_nctx ? P->cfg_nctx : PEACH_CUDA_NCTX_MAX; ; nctx--) {
      want = P->cfg_batch ? (word64) P->cfg_batch : (word64) P->sms *
         (word64) P->max_threads_sm * PEACH_CUDA_SLOTS_PER_THREAD /
         (word64) nctx;
      n = want < lo ? lo : want;
      n = (n + PEACH_PIPE_BLOCK - 1) & ~((word64) PEACH_PIPE_BLOCK - 1);
      if (n > PEACH_CUDA_BATCH_MAX) n = PEACH_CUDA_BATCH_MAX;
      limit = avail / 100 * PEACH_CUDA_MEM_PERCENT /
         ((size_t) nctx * PEACH_CUDA_SLOT_BYTES);
      if (limit > PEACH_CUDA_BATCH_MAX) limit = PEACH_CUDA_BATCH_MAX;
      if (P->cfg_nctx || nctx <= PEACH_CUDA_NCTX_MIN || limit >= n) break;
   }
   P->nctx = nctx;
   hi = (word32) (limit & ~((size_t) PEACH_PIPE_BLOCK - 1));
   if (hi < PEACH_PIPE_BLOCK) {
      pwarn("CUDA #%d: pipeline sizing failed: %llu MiB free memory",
         ctx->id, (unsigned long long) (mfree >> 20));
      return VERROR;
   }
   if (n > hi) n = hi;
   P->cap = P->nslots = (word32) n;
   P->nslots_min = lo < P->cap ? lo : P->cap;

   pdebug("CUDA #%d: pipeline sizing: %d SMs x %d threads/SM, %s %llu"
      " -> N = %u slots/batch (min %u, max %u); %u B/slot x %d contexts"
      " = %llu MiB of %llu MiB free (%llu MiB reserved)", ctx->id,
      P->sms, P->max_threads_sm, P->cfg_batch ? "MCM_PEACH_BATCH" :
      "auto", (unsigned long long) want, (unsigned) P->cap,
      (unsigned) P->nslots_min, (unsigned) hi,
      (unsigned) PEACH_CUDA_SLOT_BYTES, P->nctx,
      (unsigned long long) (((word64) P->cap * (word64) P->nctx *
      PEACH_CUDA_SLOT_BYTES) >> 20), (unsigned long long) (mfree >> 20),
      (unsigned long long) (reserve >> 20));

   return VEOK;
}  /* end peach_cuda_sizing() */

/**
 * @private
 * Release the pipeline solver resources of a Peach CUDA context: the
 * events, pinned results and device buffers of every batch context, the
 * streams and pinned trailers beyond the two legacy ones, the transition
 * table and the self-test buffer. Every pointer and handle is reset, so
 * a second call does nothing. Legacy resources (streams 0 and 1, map,
 * legacy buffers) are not touched.
 * @param ctx Pointer to DEVICE_CTX (current CUDA device; for messages)
 * @param P Pointer to Peach CUDA context
 * @returns cudaSuccess, or the first CUDA error (alerted; all resources
 * are released regardless)
*/
static cudaError_t peach_cuda_pipeline_release(DEVICE_CTX *ctx,
   PEACH_CUDA_CTX *P)
{
   cudaError_t err, first;
   int id;

#undef cuFREE
#define cuFREE(cuFN) \
   do { \
      err = (cuFN); \
      if (err != cudaSuccess) { \
         const char *str = cudaGetErrorString(err); \
         palert("CUDA ERROR on #(%d): (%d) %s", ctx->id, (int) err, str); \
         palert("... error returned by: %s", #cuFN); \
         if (first == cudaSuccess) first = err; \
      } \
   } while(0)

   first = cudaSuccess;
   for (id = 0; id < PEACH_CUDA_NCTX_MAX; id++) {
      if (id >= 2) {
         /* (streams and trailers 0 and 1 belong to the legacy solver) */
         if (P->stream[id]) cuFREE(cudaStreamDestroy(P->stream[id]));
         if (P->h_bt[id]) cuFREE(cudaFreeHost(P->h_bt[id]));
         P->stream[id] = NULL;
         P->h_bt[id] = NULL;
      }
      if (P->ev_start[id]) cuFREE(cudaEventDestroy(P->ev_start[id]));
      if (P->ev_stop[id]) cuFREE(cudaEventDestroy(P->ev_stop[id]));
      if (P->h_res[id]) cuFREE(cudaFreeHost(P->h_res[id]));
      if (P->bufs[id].d_rng) cuFREE(cudaFree(P->bufs[id].d_rng));
      if (P->bufs[id].d_slot) cuFREE(cudaFree(P->bufs[id].d_slot));
      if (P->bufs[id].d_hash) cuFREE(cudaFree(P->bufs[id].d_hash));
      if (P->bufs[id].d_q) cuFREE(cudaFree(P->bufs[id].d_q));
      if (P->bufs[id].d_ent) cuFREE(cudaFree(P->bufs[id].d_ent));
      if (P->bufs[id].d_cnt) cuFREE(cudaFree(P->bufs[id].d_cnt));
      if (P->bufs[id].d_key) cuFREE(cudaFree(P->bufs[id].d_key));
      if (P->bufs[id].d_res) cuFREE(cudaFree(P->bufs[id].d_res));
      if (P->bufs[id].d_trace) cuFREE(cudaFree(P->bufs[id].d_trace));
      /* (bufs[id].d_map and bufs[id].d_T alias P->d_map and P->d_T) */
      memset(&(P->bufs[id]), 0, sizeof(P->bufs[id]));
      P->ev_start[id] = P->ev_stop[id] = NULL;
      P->h_res[id] = NULL;
      P->inflight[id] = 0;
   }
   if (P->d_selftest) cuFREE(cudaFree(P->d_selftest));
   if (P->d_T) cuFREE(cudaFree(P->d_T));
   P->d_selftest = NULL;
   P->d_T = NULL;

   return first;
}  /* end peach_cuda_pipeline_release() */

/**
 * @private
 * Grid size of a pipeline kernel: SMs x resident blocks per SM (of
 * PEACH_PIPE_BLOCK threads), clamped to 1..PEACH_PIPE_MAXGRID. Every
 * pipeline kernel runs a block-uniform stride loop, so any grid >= 1
 * processes all items.
 * @param P Pointer to Peach CUDA context (sized)
 * @param blocks Resident blocks per SM (occupancy)
 * @returns grid size
*/
static int peach_cuda_grid(const PEACH_CUDA_CTX *P, int blocks)
{
   long grid = (long) P->sms * (long) blocks;

   if (grid < 1) grid = 1;
   if (grid > PEACH_PIPE_MAXGRID) grid = PEACH_PIPE_MAXGRID;

   return (int) grid;
}  /* end peach_cuda_grid() */

/**
 * @private
 * Set up the pipeline solver of a device (called by
 * peach_init_cuda_device() after the legacy buffers, streams and the map
 * are allocated, before any other kernel launch): size the batches,
 * allocate the transition table, the self-test buffer and every batch
 * context (stream and pinned trailer beyond the two legacy ones, RNG
 * states, slots, hash0, queues, counters, device result, pinned host
 * result, timing events), seed the RNG states of each context (distinct
 * seeds, exactly cap threads) and compute the grid of every pipeline
 * kernel from its occupancy: one wave of resident blocks, except the MD2
 * kernel with 3 or more contexts (see PEACH_CUDA_MD2_SHARE).
 * @param ctx Pointer to DEVICE_CTX (current CUDA device)
 * @param P Pointer to Peach CUDA context
 * @returns VEOK on success, else VERROR (warned); the caller then
 * releases the partial setup and uses the legacy solver
*/
static int peach_cuda_pipeline_setup(DEVICE_CTX *ctx, PEACH_CUDA_CTX *P)
{
   PEACH_PIPE_BUFS *b;
   word64 seed[PEACH_CUDA_NCTX_MAX];
   size_t cap;
   int nb[10], i, id;
   cudaError_t err;

#undef cuTRY
#define cuTRY(cuFN) \
   do { \
      err = (cuFN); \
      if (err != cudaSuccess) { \
         const char *str = cudaGetErrorString(err); \
         pwarn("CUDA #%d: pipeline setup failed: (%d) %s", ctx->id, \
            (int) err, str); \
         pwarn("... error returned by: %s", #cuFN); \
         return VERROR; \
      } \
   } while(0)

   /* batch size (sets P->sms, nctx, cap, nslots, nslots_min) */
   if (peach_cuda_sizing(ctx, P, PEACH_CUDA_T_BYTES +
         PEACH_CUDA_SELFTEST_BYTES + (PEACH_CUDA_NCTX_MAX *
         PEACH_PIPE_CNTWORDS * sizeof(word32))) != VEOK) {
      return VERROR;
   }
   if (P->nctx < 1 || P->nctx > PEACH_CUDA_NCTX_MAX ||
         P->cap < PEACH_PIPE_BLOCK || (P->cap % PEACH_PIPE_BLOCK) != 0 ||
         P->cap > PEACH_PIPE_MAXCAP) {
      pwarn("CUDA #%d: pipeline setup failed: invalid capacity %u",
         ctx->id, (unsigned) P->cap);
      return VERROR;
   }
   cap = (size_t) P->cap;

   /* per device: transition table and self-test vectors */
   cuTRY(cudaMalloc((void **) &(P->d_T), PEACH_CUDA_T_BYTES));
   cuTRY(cudaMalloc((void **) &(P->d_selftest), PEACH_CUDA_SELFTEST_BYTES));

   /* distinct RNG seeds per batch context */
   for (id = 0; id < P->nctx; id++) {
      do {
         seed[id] = ((word64) rand32() << 32) | (word64) rand32();
         for (i = 0; i < id && seed[i] != seed[id]; i++);
      } while (i < id);
   }

   /* per batch context (one per stream) */
   for (id = 0; id < P->nctx; id++) {
      b = &(P->bufs[id]);
      if (id >= 2) {
         /* stream and trailer snapshot beyond the two legacy ones */
         cuTRY(cudaStreamCreate(&(P->stream[id])));
         cuTRY(cudaMallocHost((void **) &(P->h_bt[id]), sizeof(BTRAILER)));
         memset(P->h_bt[id], 0, sizeof(BTRAILER));
      }
      cuTRY(cudaMalloc((void **) &(b->d_rng), cap * sizeof(word64)));
      cuTRY(cudaMalloc((void **) &(b->d_slot),
         cap * sizeof(PEACH_PIPE_SLOT)));
      cuTRY(cudaMalloc((void **) &(b->d_hash), cap * 8 * sizeof(word32)));
      cuTRY(cudaMalloc((void **) &(b->d_q),
         cap * PEACH_PIPE_NQUEUE * sizeof(word32)));
      cuTRY(cudaMalloc((void **) &(b->d_ent),
         cap * sizeof(PEACH_PIPE_SLOT)));
      cuTRY(cudaMalloc((void **) &(b->d_cnt),
         PEACH_PIPE_CNTWORDS * sizeof(word32)));
      cuTRY(cudaMalloc((void **) &(b->d_key), cap * sizeof(word32)));
      cuTRY(cudaMalloc((void **) &(b->d_res), sizeof(PEACH_PIPE_RESULT)));
      cuTRY(cudaMallocHost((void **) &(P->h_res[id]),
         sizeof(PEACH_PIPE_RESULT)));
      cuTRY(cudaEventCreate(&(P->ev_start[id])));
      cuTRY(cudaEventCreate(&(P->ev_stop[id])));
      b->d_map = (const uint4 *) P->d_map;
      b->d_T = P->d_T;
      b->d_trace = NULL;
      b->cap = P->cap;
      memset(P->h_res[id], 0, sizeof(PEACH_PIPE_RESULT));
      cuTRY(cudaMemsetAsync(b->d_res, 0, sizeof(PEACH_PIPE_RESULT),
         P->stream[id]));
      /* seed the RNG state of every slot: exactly cap threads, since
       * kcu_srand64() writes one state per thread without bounds */
      CUDA_KERNEL(kcu_srand64, (int) (P->cap / PEACH_PIPE_BLOCK),
         PEACH_PIPE_BLOCK, 0, P->stream[id])(b->d_rng, seed[id]);
      cuTRY(cudaGetLastError());
   }

   /* grids: SMs x resident blocks per SM, per kernel */
   cuTRY(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb[0],
      kcu_peach_pipe_hash_blake2b32, PEACH_PIPE_BLOCK, 0));
   cuTRY(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb[1],
      kcu_peach_pipe_hash_blake2b64, PEACH_PIPE_BLOCK, 0));
   cuTRY(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb[2],
      kcu_peach_pipe_hash_sha1, PEACH_PIPE_BLOCK, 0));
   cuTRY(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb[3],
      kcu_peach_pipe_hash_sha256, PEACH_PIPE_BLOCK, 0));
   cuTRY(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb[4],
      kcu_peach_pipe_hash_sha3, PEACH_PIPE_BLOCK, 0));
   cuTRY(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb[5],
      kcu_peach_pipe_hash_keccak, PEACH_PIPE_BLOCK, 0));
   cuTRY(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb[6],
      kcu_peach_pipe_hash_md2, PEACH_PIPE_BLOCK, 0));
   cuTRY(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb[7],
      kcu_peach_pipe_hash_md5, PEACH_PIPE_BLOCK, 0));
   cuTRY(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb[8],
      kcu_peach_pipe_init, PEACH_PIPE_BLOCK, 0));
   cuTRY(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb[9],
      kcu_peach_pipe_final, PEACH_PIPE_BLOCK, 0));
   for (i = 0; i < 10; i++) {
      if (nb[i] < 1) {
         pwarn("CUDA #%d: pipeline setup failed: kernel %d cannot run"
            " %d threads per block", ctx->id, i, PEACH_PIPE_BLOCK);
         return VERROR;
      }
   }
   P->launch.block = PEACH_PIPE_BLOCK;
   for (i = 0; i < 8; i++) P->launch.grid_hash[i] = peach_cuda_grid(P, nb[i]);
   if (P->nctx >= 3) {
      /* MD2 shares the multiprocessors with the other contexts */
      i = nb[6] / PEACH_CUDA_MD2_SHARE;
      P->launch.grid_hash[6] = peach_cuda_grid(P, i > 1 ? i : 1);
   }
   P->launch.grid_init = peach_cuda_grid(P, nb[8]);
   P->launch.grid_final = peach_cuda_grid(P, nb[9]);

   return VEOK;
}  /* end peach_cuda_pipeline_setup() */

/**
 * Check Peach proof of work with a CUDA device.
 * Uses the first available Cuda device to check multiple POW.
 * @param count Number of block trailers to check
 * @param bt Pointer to block trailer array
 * @param out Pointer to final hash array, if non-null
 * @returns (int) value representing the result of the operation
 * @retval (-1) Error occurred during operation
 * @retval 0 Evaluation successful
 * @retval 1 Evaluation failed
*/
int peach_checkhash_cuda(int count, BTRAILER bt[], void *out)
{
   size_t btsz = sizeof(BTRAILER) * count;
   size_t outsz = SHA256LEN * count;
   BTRAILER *d_bt;
   word8 *d_out, *d_eval;
   word8 eval = 0;
   int cuda_count;

#undef cuCHK
#define cuCHK(cuFN) \
   do { \
      cudaError_t err = (cuFN); \
      if (err != cudaSuccess) { \
         const char *str = cudaGetErrorString(err); \
         palert("CUDA ERROR: (%d) %s", (int) err, str); \
         palert("... error returned by: %s", #cuFN); \
         set_errno(EMCM_CUDA); \
         return (-1); \
      } \
   } while(0)

   cuCHK(cudaGetDeviceCount(&cuda_count));
   if (cuda_count < 1) {
      palert("No CUDA devices...");
      return -1;
   }
   cuCHK(cudaSetDevice(0));
   cuCHK(cudaMalloc(&d_bt, btsz));
   cuCHK(cudaMalloc(&d_out, outsz));
   cuCHK(cudaMalloc(&d_eval, 1));
   /* transfer data to device */
   cuCHK(cudaMemcpy(d_bt, bt, btsz, cudaMemcpyHostToDevice));
   cuCHK(cudaMemset(d_out, 0, outsz));
   cuCHK(cudaMemset(d_eval, 0, 1));
   /* launch kernel to check Peach */
   CUDA_KERNEL(kcu_peach_checkhash, 1, count)
      (d_bt, d_out, d_eval);
   cuCHK(cudaGetLastError());
   /* retrieve hash/eval data */
   cuCHK(cudaMemcpy(out, d_out, outsz, cudaMemcpyDeviceToHost));
   cuCHK(cudaMemcpy(&eval, d_eval, 1, cudaMemcpyDeviceToHost));
   /* wait for device to finish */
   cuCHK(cudaDeviceSynchronize());
   /* free memory */
   cuCHK(cudaFree(d_bt));
   cuCHK(cudaFree(d_out));
   cuCHK(cudaFree(d_eval));

   return (int) eval;
}  /* end peach_checkhash_cuda() */

/**
 * Release the Peach resources of a device context. Waits for all work on
 * the device, then frees all device and pinned host memory, events and
 * streams, and the Peach CUDA context itself. Afterwards `ctx->peach` is
 * NULL, the status is DEV_NULL and peach_init_cuda_device() may be used
 * again (e.g. with a different configuration). Safe on a context whose
 * initialization failed part way, and on one without a Peach context.
 * @param ctx Pointer to DEVICE_CTX to release
 * @returns VEOK on success, else VERROR if a CUDA call failed (all
 * resources are released regardless)
*/
int peach_free_cuda_device(DEVICE_CTX *ctx)
{
   PEACH_CUDA_CTX *P;
   cudaError_t err, first;
   int id;

#undef cuFREE
#define cuFREE(cuFN) \
   do { \
      err = (cuFN); \
      if (err != cudaSuccess) { \
         const char *str = cudaGetErrorString(err); \
         palert("CUDA ERROR on #(%d): (%d) %s", ctx->id, (int) err, str); \
         palert("... error returned by: %s", #cuFN); \
         if (first == cudaSuccess) first = err; \
      } \
   } while(0)

   P = (PEACH_CUDA_CTX *) ctx->peach;
   first = cudaSuccess;
   if (P != NULL) {
      cuFREE(cudaSetDevice(ctx->id));
      /* wait for outstanding work on all streams */
      cuFREE(cudaDeviceSynchronize());
      /* release pipeline resources (batch contexts, T, self-test) */
      err = peach_cuda_pipeline_release(ctx, P);
      if (first == cudaSuccess) first = err;
      /* release per stream (legacy) resources */
      for (id = 0; id < 2; id++) {
         if (P->h_solve[id]) cuFREE(cudaFreeHost(P->h_solve[id]));
         if (P->h_bt[id]) cuFREE(cudaFreeHost(P->h_bt[id]));
         if (P->d_solve[id]) cuFREE(cudaFree(P->d_solve[id]));
         if (P->d_state[id]) cuFREE(cudaFree(P->d_state[id]));
         if (P->d_bt[id]) cuFREE(cudaFree(P->d_bt[id]));
         if (P->stream[id]) cuFREE(cudaStreamDestroy(P->stream[id]));
      }
      /* release per device resources */
      if (P->d_phash) cuFREE(cudaFree(P->d_phash));
      if (P->d_map) cuFREE(cudaFree(P->d_map));
      if (P->bad_solves) {
         pwarn("CUDA #%d: %llu solve(s) were rejected by CPU verification",
            ctx->id, (unsigned long long) P->bad_solves);
      }
      if (P->bad_canaries) {
         pwarn("CUDA #%d: %llu canary result(s) did not match the CPU",
            ctx->id, (unsigned long long) P->bad_canaries);
      }
      free(P);
   }

   /* device context is uninitialized */
   ctx->peach = NULL;
   ctx->status = DEV_NULL;
   ctx->work = ctx->hps = 0;
   if (first != cudaSuccess) {
      set_errno(EMCM_CUDA);
      return VERROR;
   }

   return VEOK;
}  /* end peach_free_cuda_device() */

/**
 * (re)Initialize a device context with a CUDA device.
 * @param devp Pointer to DEVICE_CTX to initialize
 * @param id Index of CUDA device to initialize to DEVICE_CTX
 * @returns VEOK on success, else VERROR
 * @note The `id` parameter of the DEVICE_CTX must be set to an appropriate
 * CUDA device number. If not performing a re-initialization, recommend
 * using peach_init_cuda() first.
 * @note Reads the solver configuration from the environment, once per
 * initialization (see peach_cuda_config()), and logs it. To re-initialize
 * (e.g. after changing the configuration), first release the context
 * with peach_free_cuda_device(); this also releases a context whose
 * initialization failed.
 * @note Unless MCM_PEACH_LEGACY=1, also sets up the pipeline solver
 * (transition table and its batch contexts). If that setup fails, the
 * device is still initialized, with the official (legacy) solver, whose
 * resources are always allocated.
*/
int peach_init_cuda_device(DEVICE_CTX *ctx)
{
   PEACH_CUDA_CTX *p_ctx;
   const char *reason;
   size_t btsz, seedsz;
   int grid, block, i;
   char skip[48], batch[16], streams[16];

#undef cuCHK
#define cuCHK(cuFN) \
   do { \
      cudaError_t err = (cuFN); \
      if (err != cudaSuccess) { \
         const char *str = cudaGetErrorString(err); \
         palert("CUDA ERROR on #(%d): (%d) %s", ctx->id, (int) err, str); \
         palert("... error returned by: %s", #cuFN); \
         /* clear the (non-sticky) error: never left to another device */ \
         (void) cudaGetLastError(); \
         ctx->status = DEV_FAIL; \
         return VERROR; \
      } \
   } while(0)

   /* check for double init */
   if (ctx->peach) {
      set_errno(EINVAL);
      return VERROR;
   }
   /* drop a stale (non-sticky) error of this host thread, e.g. left by
    * another device's failed initialization */
   (void) cudaGetLastError();

   /* allocate peach context (zeroed: NULL means "not allocated") */
   p_ctx = (PEACH_CUDA_CTX *) calloc(1, sizeof(PEACH_CUDA_CTX));
   if (p_ctx == NULL) return VERROR;
   ctx->peach = p_ctx;
   /* reset progress; ctx->work is also the (legacy) map build cursor */
   ctx->work = ctx->hps = 0;

   /* read solver configuration (once per initialization) */
   peach_cuda_config(ctx, p_ctx);
   /* legacy solver until the pipeline solver is set up (below) */
   p_ctx->mode = PEACH_CUDA_MODE_LEGACY;

   /* set context to CUDA id */
   cuCHK(cudaSetDevice(ctx->id));
   /* determine CUDA occupancy for device */
   cuCHK(cudaOccupancyMaxPotentialBlockSize(&grid, &block, kcu_peach_solve, 0, 0));
   /* store grid/block and calculate threads and state sizes */
   ctx->grid = grid;
   ctx->block = block;
   ctx->threads = ctx->grid * ctx->block;
   seedsz = sizeof(word64) * ctx->threads;
   btsz = sizeof(BTRAILER);
   /* create streams for device */
   cuCHK(cudaStreamCreate(&(p_ctx->stream[0])));
   cuCHK(cudaStreamCreate(&(p_ctx->stream[1])));
   /* allocate pinned host memory for host/device transfers */
   cuCHK(cudaMallocHost(&(p_ctx->h_solve[0]), 32));
   cuCHK(cudaMallocHost(&(p_ctx->h_solve[1]), 32));
   cuCHK(cudaMallocHost(&(p_ctx->h_bt[0]), btsz));
   cuCHK(cudaMallocHost(&(p_ctx->h_bt[1]), btsz));
   /* allocate device memory for host/device transfers */
   cuCHK(cudaMalloc(&(p_ctx->d_solve[0]), 32));
   cuCHK(cudaMalloc(&(p_ctx->d_solve[1]), 32));
   cuCHK(cudaMalloc(&(p_ctx->d_state[0]), seedsz));
   cuCHK(cudaMalloc(&(p_ctx->d_state[1]), seedsz));
   cuCHK(cudaMalloc(&(p_ctx->d_bt[0]), btsz));
   cuCHK(cudaMalloc(&(p_ctx->d_bt[1]), btsz));
   /* allocate memory for Peach map on device */
   cuCHK(cudaMalloc(&(p_ctx->d_phash), 32));
   cuCHK(cudaMalloc(&(p_ctx->d_map), PEACHMAPLEN));
   /* clear device/host allocated memory */
   cuCHK(cudaMemsetAsync(p_ctx->d_bt[0], 0, btsz, p_ctx->stream[0]));
   cuCHK(cudaMemsetAsync(p_ctx->d_bt[1], 0, btsz, p_ctx->stream[1]));
   cuCHK(cudaMemsetAsync(p_ctx->d_solve[0], 0, 32, p_ctx->stream[0]));
   cuCHK(cudaMemsetAsync(p_ctx->d_solve[1], 0, 32, p_ctx->stream[1]));
   cuCHK(cudaMemsetAsync(p_ctx->d_phash, 0, 32, p_ctx->stream[0]));
   memset(p_ctx->h_bt[0], 0, btsz);
   memset(p_ctx->h_bt[1], 0, btsz);
   memset(p_ctx->h_solve[0], 0, 32);
   memset(p_ctx->h_solve[1], 0, 32);

   /* set up the pipeline solver (unless MCM_PEACH_LEGACY=1). Before any
    * other kernel launch, so that a failed setup clears only its own
    * (non-sticky) CUDA errors; the device then uses the legacy solver,
    * whose buffers above are always allocated (fallback) */
   reason = "MCM_PEACH_LEGACY=1";
   if (!p_ctx->cfg_legacy) {
      if (peach_cuda_pipeline_setup(ctx, p_ctx) == VEOK) {
         p_ctx->mode = PEACH_CUDA_MODE_PIPELINE;
      } else {
         peach_cuda_pipeline_release(ctx, p_ctx);
         (void) cudaGetLastError();
         pwarn("CUDA #%d: pipeline solver unavailable; using the legacy"
            " solver", ctx->id);
         reason = "pipeline setup failed";
      }
   }

   /* generate prng state */
   CUDA_KERNEL(kcu_srand64, grid, block, 0, p_ctx->stream[0])
      (p_ctx->d_state[0], rand32());
   CUDA_KERNEL(kcu_srand64, grid, block, 0, p_ctx->stream[1])
      (p_ctx->d_state[1], rand32());

   /* log configuration, one line per device */
   for (i = 1; i < 8; i++) {
      if (p_ctx->cfg_skip[i] != p_ctx->cfg_skip[0]) break;
   }
   if (i == 8) {
      snprintf(skip, sizeof(skip), "0x%02x", (unsigned) p_ctx->cfg_skip[0]);
   } else {
      snprintf(skip, sizeof(skip),
         "0x%02x,0x%02x,0x%02x,0x%02x,0x%02x,0x%02x,0x%02x,0x%02x",
         (unsigned) p_ctx->cfg_skip[0], (unsigned) p_ctx->cfg_skip[1],
         (unsigned) p_ctx->cfg_skip[2], (unsigned) p_ctx->cfg_skip[3],
         (unsigned) p_ctx->cfg_skip[4], (unsigned) p_ctx->cfg_skip[5],
         (unsigned) p_ctx->cfg_skip[6], (unsigned) p_ctx->cfg_skip[7]);
   }
   if (p_ctx->cfg_batch) {
      snprintf(batch, sizeof(batch), "%lu", (unsigned long) p_ctx->cfg_batch);
   } else snprintf(batch, sizeof(batch), "auto");
   if (p_ctx->cfg_nctx) {
      snprintf(streams, sizeof(streams), "%d", p_ctx->cfg_nctx);
   } else snprintf(streams, sizeof(streams), "auto");
   if (p_ctx->mode == PEACH_CUDA_MODE_PIPELINE) {
      plog("CUDA #%d: Peach solver: pipeline; MCM_PEACH_SKIP=%s"
         " MCM_PEACH_BATCH=%s MCM_PEACH_STREAMS=%s; N = %u slots/batch"
         " (min %u) x %d contexts; grids x%d: init %d,"
         " hash %d,%d,%d,%d,%d,%d,%d,%d, final %d",
         ctx->id, skip, batch, streams, (unsigned) p_ctx->nslots,
         (unsigned) p_ctx->nslots_min, p_ctx->nctx,
         p_ctx->launch.block, p_ctx->launch.grid_init,
         p_ctx->launch.grid_hash[0], p_ctx->launch.grid_hash[1],
         p_ctx->launch.grid_hash[2], p_ctx->launch.grid_hash[3],
         p_ctx->launch.grid_hash[4], p_ctx->launch.grid_hash[5],
         p_ctx->launch.grid_hash[6], p_ctx->launch.grid_hash[7],
         p_ctx->launch.grid_final);
   } else {
      plog("CUDA #%d: Peach solver: legacy kernel (%s); pipeline options"
         " (unused): MCM_PEACH_SKIP=%s MCM_PEACH_BATCH=%s"
         " MCM_PEACH_STREAMS=%s", ctx->id, reason, skip, batch, streams);
   }

   /* set device as initialized */
   ctx->status = DEV_INIT;

   return VEOK;
}  /* end peach_init_cuda_device() */

/**
 * @private
 * Official (legacy) Peach CUDA solver: kcu_peach_solve() on two streams.
 * Same state machine and solve semantics as the official solver, except
 * that a solve is only reported after CPU verification.
 * @param ctx Pointer to DEVICE_CTX to perform work with
 * @param bt Pointer to (snapshot of the) block trailer to solve for
 * @param diff Difficulty to test against entropy of final hash
 * @param btout Pointer to location to place solved block trailer
 * @returns VEOK on solve, VERROR on no solve, or VETIMEOUT if GPU is
 * either stopped or unrecoverable.
*/
static int peach_solve_cuda_legacy(DEVICE_CTX *ctx, const BTRAILER *bt,
   word8 diff, BTRAILER *btout)
{
   int id, grid, block, build;
   PEACH_CUDA_CTX *P;
   BTRAILER cand;
   cudaError_t err;

#undef cuCHK
#define cuCHK(cuFN) \
   do { \
      err = (cuFN); \
      if (err != cudaSuccess) { \
         const char *str = cudaGetErrorString(err); \
         palert("CUDA ERROR on #(%d): (%d) %s", ctx->id, (int) err, str); \
         palert("... error returned by: %s", #cuFN); \
         /* clear the (non-sticky) error: never left to another device */ \
         (void) cudaGetLastError(); \
         ctx->status = DEV_FAIL; \
         return VERROR; \
      } \
   } while(0)

   /* init */
   P = (PEACH_CUDA_CTX *) ctx->peach;
   /* report unuseable GPUs */
   if (ctx->status < DEV_NULL) return VETIMEOUT;

   /* set cuda device */
   cuCHK(cudaSetDevice(ctx->id));
   /* check for previous (async) execution errors */
   cuCHK(cudaGetLastError());

   /* build peach map */
   if (ctx->status == DEV_INIT) {
      /* build peach map -- init */
      for (build = id = 0; id < 2; id++) {
         /* check stream is ready */
         err = cudaStreamQuery(P->stream[id]);
         if (err == cudaErrorNotReady) continue;
         cuCHK(err);

         /* check pre-build state */
         if (ctx->work == 0 && build == 0) {
            /* ensure secondary stream is ready */
            err = cudaStreamQuery(P->stream[id ^ 1]);
            if (err == cudaErrorNotReady) break;
            cuCHK(err);
            /* clear late solves */
            cuCHK(cudaMemset(P->d_solve[0], 0, 32));
            cuCHK(cudaMemset(P->d_solve[1], 0, 32));
            memset(P->h_solve[0], 0, 32);
            memset(P->h_solve[1], 0, 32);
            /* update block trailer */
            memcpy(P->h_bt[0], bt, sizeof(BTRAILER));
            memcpy(P->h_bt[1], bt, sizeof(BTRAILER));
            /* record the map's phash; start a new epoch */
            memcpy(P->map_phash, bt->phash, HASHLEN);
            P->epoch++;
            /* update device phash */
            cuCHK(cudaMemcpy(P->d_phash, P->h_bt[0]->phash, 32, cudaMemcpyHostToDevice));
            /* synchronize memory transfers before building peach map */
            cuCHK(cudaDeviceSynchronize());
            /* flag build state */
            build = 1;
         }
         /* check build state */
         if (ctx->work > 0 || build) {
            if (ctx->work < PEACHCACHELEN) {
               /* prepare launch config and generate peach map */
               cuCHK(cudaOccupancyMaxPotentialBlockSize(&grid, &block, kcu_peach_build, 0, 0));
               CUDA_KERNEL(kcu_peach_build, grid, block, 0, P->stream[id])
                  ((word32) ctx->work, P->d_map, P->d_phash);
               cuCHK(cudaGetLastError());
               /* update build progress */
               ctx->work += grid * block;
            } else {
               /* ensure secondary stream is finished */
               err = cudaStreamQuery(P->stream[id ^ 1]);
               if (err == cudaErrorNotReady) break;
               cuCHK(err);
               /* build is complete */
               ctx->last = time(NULL);
               ctx->status = DEV_IDLE;
               ctx->work = 0;
               break;
            }
         }  /* end if (ctx->work > 0... */
      }  /* end for(build = id = 0... */
   }  /* end if (ctx->status == DEV_INIT)... */

   /* switch to WORK mode when all conditions are satisfied:
    * - transactions to solve
    * - block NOT already solved
    * - block NOT expired
    */
   while (ctx->status == DEV_IDLE) {
      if (get32(bt->tcount) == 0) break;
      if (cmp64(bt->bnum, btout->bnum) == 0) break;
      if (difftime(time(NULL), get32(bt->time0)) >= BRIDGEv3) break;
      ctx->last = time(NULL);
      ctx->status = DEV_WORK;
      ctx->work = 0;
      break;
   }

   /* solve work in block trailer */
   if (ctx->status == DEV_WORK) {
      for(id = 0; id < 2; id++) {
         err = cudaStreamQuery(P->stream[id]);
         if (err == cudaErrorNotReady) continue;
         cuCHK(err);
         /* check trailer for block update */
         if (memcmp(P->h_bt[id]->phash, bt->phash, HASHLEN)) {
            ctx->status = DEV_INIT;
            ctx->work = 0;
            break;
         }
         /* switch to IDLE mode when reasonable:
          * - no transaction to solve
          * - block already solved
          * - block expired
          */
         if (get32(bt->tcount) == 0 || cmp64(bt->bnum, btout->bnum) == 0 ||
               difftime(time(NULL), get32(bt->time0)) >= BRIDGEv3) {
            ctx->status = DEV_IDLE;
            ctx->work = 0;
            break;
         }
         /* check for solves */
         if (*(P->h_solve[id])) {
            /* combine solve with the trailer it was launched with */
            memcpy(&cand, P->h_bt[id], sizeof(BTRAILER));
            memcpy(cand.nonce, P->h_solve[id], 32);
            /* (async) clear solve */
            cuCHK(cudaMemsetAsync(P->d_solve[id], 0, 32, P->stream[id]));
            memset(P->h_solve[id], 0, 32);
            /* report (copy to output) ONLY a solve confirmed by the CPU */
            if (peach_cuda_verify(ctx, P, &cand, P->diff_inflight[id],
                  NULL, 0) == VEOK) {
               memcpy(btout, &cand, sizeof(BTRAILER));
               return VEOK;
            }
            /* rejected: btout untouched, continue with a new launch */
         }
         /* update block trailer (incl. half nonce) */
         memcpy(P->h_bt[id], bt, 92);
         trigg_generate(P->h_bt[id]->nonce);
         /* (async) update trailer data (incl. half nonce) */
         cuCHK(cudaMemcpyAsync(P->d_bt[id], P->h_bt[id],
            92 + 16, cudaMemcpyHostToDevice, P->stream[id]));
         /* (async) launch kernel to solve Peach (dynamic difficulty) */
         diff = diff && diff < bt->difficulty[0] ? diff : bt->difficulty[0];
         P->diff_inflight[id] = diff;  /* for verification of its solve */
         CUDA_KERNEL(kcu_peach_solve, ctx->grid, ctx->block, 0, P->stream[id])
            (P->d_map, P->d_bt[id], P->d_state[id], diff, P->d_solve[id]);
         /* check kernel launch errors */
         cuCHK(cudaGetLastError());
         /* (async) solve retrieval */
         cuCHK(cudaMemcpyAsync(P->h_solve[id], P->d_solve[id], 32,
            cudaMemcpyDeviceToHost, P->stream[id]));
         /* increment progress counters */
         ctx->work += ctx->threads;
         double delta = difftime(time(NULL), ctx->last);
         ctx->hps = ctx->work / (delta ? delta : 1);
      }  /* end for(id = 0; id < 2; id++)... */
   }  /* end if (ctx->status == DEV_WORK)... */

   return VERROR;
}  /* end peach_solve_cuda_legacy() */

/**
 * @private
 * Abandon the pipeline solver of a device for the legacy solver, until
 * the device is initialized again: alert once with the reason, discard
 * every batch in flight (no pipeline candidate is reported afterwards),
 * and restart at DEV_INIT with ctx->work = 0, so that the legacy solver
 * waits for its two streams and rebuilds its own state (map included).
 * Batches still running on the other pipeline streams only read the map
 * and write their own buffers; their results are never harvested.
 * @param ctx Pointer to DEVICE_CTX
 * @param P Pointer to its Peach CUDA context
 * @param reason Reason, for the alert
 * @returns (-1), for the convenience of callers
*/
static int peach_cuda_fallback(DEVICE_CTX *ctx, PEACH_CUDA_CTX *P,
   const char *reason)
{
   if (!P->fallback) {
      palert("CUDA #%d: pipeline solver disabled (%s); switching to the"
         " legacy solver", ctx->id, reason);
   }
   P->fallback = 1;
   memset(P->inflight, 0, sizeof(P->inflight));
   P->build_next = 0;
   ctx->status = DEV_INIT;
   ctx->work = 0;

   return (-1);
}  /* end peach_cuda_fallback() */

/**
 * @private
 * Pipeline self-test, after every map build (the map and the transition
 * table are complete and all streams are idle). Compares with the same
 * code on the CPU:
 * - the transition table entries of tiles 0, 1, 0xFFFFF and 3 random
 *   tiles with peach_transition_tile() over the device's own tiles;
 * - kcu_peach_pipe_selftest() over special float values (signed zeros,
 *   denormals, smallest normals, FLT_MAX, infinities, NaNs, a zero word
 *   for 0/0, a first nonce half whose third word is a NaN) and random
 *   words, each with ops 0..3 and indices 0, 1 and 0xFFFFF, with
 *   peach_pipe_selftest_one().
 * Detects device code built with flush-to-zero or fast math, and a
 * transition table that does not match the map.
 * @param ctx Pointer to DEVICE_CTX (current CUDA device)
 * @param P Pointer to its Peach CUDA context
 * @param s Stream for the self-test kernel
 * @returns 0 if passed, 1 on any mismatch (alerted), or (-1) on a CUDA
 * error (alerted; ctx->status = DEV_FAIL)
*/
static int peach_cuda_selftest(DEVICE_CTX *ctx, PEACH_CUDA_CTX *P,
   cudaStream_t s)
{
   static const word32 special[] = {
      /* +0, -0, denormals (0x000FFFFF: largest tile index as a word) */
      0x00000000, 0x80000000, 0x00000001, 0x80000001, 0x000FFFFF,
      0x007FFFFF, 0x807FFFFF,
      /* smallest normals, +-1, 2, +-2^31, +-FLT_MAX */
      0x00800000, 0x80800000, 0x3F800000, 0xBF800000, 0x40000000,
      0x4F000000, 0xCF000000, 0x7F7FFFFF, 0xFF7FFFFF,
      /* +-Inf, NaNs: quiet, negative quiet, signaling, all ones */
      0x7F800000, 0xFF800000, 0x7FC00000, 0xFFC00000, 0x7F800001,
      0xFF800001, 0x7FBFFFFF, 0xFFFFFFFF,
      /* first nonce half (haiku) whose third word is a NaN */
      0xD6540C12, 0x0312D601, 0xFF800501
   };
   static const word32 idx3[3] = { 0, 1, PEACHCACHELEN_M1 };
   word32 tiles[PEACH_CUDA_SELFTEST_TILES], tile[PEACHTILELEN32];
   word32 ref[PEACH_PIPE_SELFTEST_OUT], *in, *out, *d_in, *d_out, *e, w;
   word16 tdev, thost;
   int nspecial, nwords, n, i, j, k, op, bad;
   cudaError_t err;

#undef cuCHK
#define cuCHK(cuFN) \
   do { \
      err = (cuFN); \
      if (err != cudaSuccess) { \
         const char *str = cudaGetErrorString(err); \
         palert("CUDA ERROR on #(%d): (%d) %s", ctx->id, (int) err, str); \
         palert("... error returned by: %s", #cuFN); \
         /* clear the (non-sticky) error: never left to another device */ \
         (void) cudaGetLastError(); \
         ctx->status = DEV_FAIL; \
         free(in); \
         return (-1); \
      } \
   } while(0)

   /* host vectors: input entries, then output entries (as on device) */
   in = (word32 *) malloc(PEACH_CUDA_SELFTEST_BYTES);
   if (in == NULL) {
      palert("CUDA #%d: pipeline self-test: out of memory", ctx->id);
      return 1;
   }
   out = in + (PEACH_CUDA_SELFTEST_MAX * PEACH_PIPE_SELFTEST_IN);
   d_in = P->d_selftest;
   d_out = P->d_selftest + (PEACH_CUDA_SELFTEST_MAX * PEACH_PIPE_SELFTEST_IN);
   bad = 0;

   /* transition table entries vs the device's own tiles */
   tiles[0] = 0;
   tiles[1] = 1;
   tiles[2] = PEACHCACHELEN_M1;
   for (i = 3; i < PEACH_CUDA_SELFTEST_TILES; i++) {
      tiles[i] = rand32() & PEACHCACHELEN_M1;
   }
   for (i = 0; i < PEACH_CUDA_SELFTEST_TILES; i++) {
      cuCHK(cudaMemcpy(tile, &(P->d_map[(size_t) tiles[i] * PEACHTILELEN64]),
         PEACHTILELEN, cudaMemcpyDeviceToHost));
      cuCHK(cudaMemcpy(&tdev, &(P->d_T[tiles[i]]), sizeof(word16),
         cudaMemcpyDeviceToHost));
      thost = peach_transition_tile(tile, tiles[i]);
      if (tdev != thost) {
         palert("CUDA #%d: pipeline self-test: T[0x%05x] = 0x%04x, CPU"
            " 0x%04x", ctx->id, (unsigned) tiles[i], (unsigned) tdev,
            (unsigned) thost);
         bad++;
      }
   }

   /* dflops steps of special and random words, ops 0..3, 3 indices */
   nspecial = (int) (sizeof(special) / sizeof(special[0]));
   nwords = nspecial + PEACH_CUDA_SELFTEST_RANDOM;
   for (n = i = 0; i < nwords; i++) {
      w = i < nspecial ? special[i] : rand32();
      for (op = 0; op < 4; op++) {
         for (j = 0; j < 3 && n < PEACH_CUDA_SELFTEST_MAX; j++, n++) {
            e = &in[n * PEACH_PIPE_SELFTEST_IN];
            e[0] = w;
            e[1] = (word32) op;
            e[2] = idx3[j];
         }
      }
   }
   cuCHK(cudaMemcpy(d_in, in, sizeof(word32) * PEACH_PIPE_SELFTEST_IN * n,
      cudaMemcpyHostToDevice));
   CUDA_KERNEL(kcu_peach_pipe_selftest, (n + PEACH_PIPE_BLOCK - 1) /
      PEACH_PIPE_BLOCK, PEACH_PIPE_BLOCK, 0, s)((const word32 *) d_in,
      d_out, n);
   cuCHK(cudaGetLastError());
   cuCHK(cudaMemcpyAsync(out, d_out, sizeof(word32) *
      PEACH_PIPE_SELFTEST_OUT * n, cudaMemcpyDeviceToHost, s));
   cuCHK(cudaStreamSynchronize(s));
   for (k = 0; k < n; k++) {
      e = &in[k * PEACH_PIPE_SELFTEST_IN];
      peach_pipe_selftest_one(e[0], e[1], e[2], ref);
      if (memcmp(ref, &out[k * PEACH_PIPE_SELFTEST_OUT], sizeof(ref))) {
         if (bad < 8) {
            palert("CUDA #%d: pipeline self-test: dflops(0x%08x, op %u,"
               " index 0x%05x) = 0x%08x/%u/0x%08x, CPU 0x%08x/%u/0x%08x",
               ctx->id, (unsigned) e[0], (unsigned) e[1], (unsigned) e[2],
               (unsigned) out[(k * PEACH_PIPE_SELFTEST_OUT)],
               (unsigned) out[(k * PEACH_PIPE_SELFTEST_OUT) + 1],
               (unsigned) out[(k * PEACH_PIPE_SELFTEST_OUT) + 2],
               (unsigned) ref[0], (unsigned) ref[1], (unsigned) ref[2]);
         }
         bad++;
      }
   }
   free(in);

   if (bad) {
      palert("CUDA #%d: pipeline self-test FAILED: %d mismatch(es); was the"
         " device code built with flush-to-zero or fast math?", ctx->id,
         bad);
      return 1;
   }
   pdebug("CUDA #%d: pipeline self-test passed (%d T entries, %d dflops"
      " vectors)", ctx->id, PEACH_CUDA_SELFTEST_TILES, n);

   return 0;
}  /* end peach_cuda_selftest() */

/**
 * @private
 * Harvest the finished batch of pipeline context @a id: its stream is
 * idle, so the batch and the copy of its result to h_res[id] are
 * complete. The result is consumed (cleared) as it is read, and is used
 * only if its epoch is the epoch of the batch launched on this context.
 * Then, in this order: the completed nonces count in ctx->work; a queue
 * overflow or prefix anomaly abandons the pipeline solver; two batches in
 * a row slower than PEACH_CUDA_BATCH_MS halve the slots of later batches
 * (not below nslots_min), and PEACH_CUDA_BATCH_GROW fast batches in a row
 * double them again (up to cap), counting only batches launched with the
 * current slots (with several contexts, batches launched before a change
 * are still in flight after it); the canary (final queue entry 0) is
 * verified on the CPU for the first PEACH_CUDA_CANARY_FIRST batches after
 * a map build, then every PEACH_CUDA_CANARY_EVERY-th batch; a solve is
 * composed from the batch's own trailer snapshot h_bt[id] (incl. its
 * first nonce half) and the device's second nonce half, verified with
 * peach_checkhash() on the CPU (and its final hash) at the batch
 * difficulty, and copied to @a btout only if @a report is set. A canary
 * mismatch or a rejected solve abandons the pipeline solver (never
 * touching @a btout).
 * @param ctx Pointer to DEVICE_CTX
 * @param P Pointer to its Peach CUDA context
 * @param id Batch context (stream) number
 * @param btout Pointer to location to place solved block trailer
 * @param report Non-zero if a solve of this batch may be reported: its
 * trailer is one the caller can still send a solve for (see
 * peach_cuda_reportable()); else the batch only counts as work
 * @returns 1 if a verified solve was copied to @a btout, 0 if not, or
 * (-1) if the pipeline solver was abandoned (fallback)
*/
static int peach_cuda_harvest(DEVICE_CTX *ctx, PEACH_CUDA_CTX *P, int id,
   BTRAILER *btout, int report)
{
   PEACH_PIPE_RESULT res;
   BTRAILER cand;
   word32 n;
   float ms;

   /* consume the result */
   memcpy(&res, P->h_res[id], sizeof(res));
   memset(P->h_res[id], 0, sizeof(PEACH_PIPE_RESULT));
   P->inflight[id] = 0;
   if (res.epoch != P->epoch_inflight[id]) {
      palert("CUDA #%d: pipeline batch result has epoch %u, expected %u",
         ctx->id, (unsigned) res.epoch, (unsigned) P->epoch_inflight[id]);
      return peach_cuda_fallback(ctx, P, "batch result epoch mismatch");
   }
   P->batches++;
   P->batches_total++;
   ctx->work += (size_t) res.completed;
   if (res.overflow || res.anomaly) {
      palert("CUDA #%d: pipeline batch: %u queue overflow(s), %u prefix"
         " anomaly(ies)", ctx->id, (unsigned) res.overflow,
         (unsigned) res.anomaly);
      return peach_cuda_fallback(ctx, P, "queue overflow or prefix anomaly");
   }

   /* adaptive batch size: two slow batches in a row halve later batches
    * (a single outlier is ignored); a run of batches fast enough to stay
    * well below the limit when doubled (2/5 of it) doubles them again, up
    * to the capacity. A batch launched with other slots than the current
    * ones (before a change) does not count */
   if (P->params[id].nslots != P->nslots) {
      /* launched before the last change: says nothing about it */
   } else if (cudaEventElapsedTime(&ms, P->ev_start[id], P->ev_stop[id]) !=
         cudaSuccess) {
      (void) cudaGetLastError();  /* no timing: keep the batch size */
   } else if (ms > (float) PEACH_CUDA_BATCH_MS) {
      P->fast_batches = 0;
      if (++(P->slow_batches) >= 2 && P->nslots > P->nslots_min) {
         n = (P->nslots / 2) & ~((word32) PEACH_PIPE_BLOCK - 1);
         if (n < P->nslots_min) n = P->nslots_min;
         plog("CUDA #%d: pipeline batches of %u slots took %.0f ms (limit"
            " %d ms); %u slots per batch from now on", ctx->id,
            (unsigned) P->nslots, (double) ms, PEACH_CUDA_BATCH_MS,
            (unsigned) n);
         P->nslots = n;
         P->slow_batches = 0;
      }
   } else {
      P->slow_batches = 0;
      if (ms >= (float) PEACH_CUDA_BATCH_MS * 2 / 5 || P->nslots >= P->cap) {
         P->fast_batches = 0;  /* (in a row) */
      } else if (++(P->fast_batches) >= PEACH_CUDA_BATCH_GROW) {
         n = P->nslots * 2;
         if (n > P->cap || n < P->nslots) n = P->cap;
         pdebug("CUDA #%d: pipeline batches of %u slots took %.0f ms;"
            " %u slots per batch from now on", ctx->id,
            (unsigned) P->nslots, (double) ms, (unsigned) n);
         P->nslots = n;
         P->fast_batches = 0;
      }
   }

   /* canary: final queue entry 0 vs the CPU reference */
   if (res.canary_valid && (P->batches <= PEACH_CUDA_CANARY_FIRST ||
         (P->batches % PEACH_CUDA_CANARY_EVERY) == 0)) {
      P->canary_checks++;
      memcpy(&cand, P->h_bt[id], sizeof(BTRAILER));
      memcpy(cand.nonce + 16, res.canary_nonce_hi, 16);
      if (peach_cuda_verify(ctx, P, &cand, 0, res.canary_hash, 1) !=
            VEOK) {
         return peach_cuda_fallback(ctx, P, "canary mismatch");
      }
   }

   /* solve: report ONLY a candidate confirmed by the CPU */
   if (res.found) {
      memcpy(&cand, P->h_bt[id], sizeof(BTRAILER));
      memcpy(cand.nonce + 16, res.nonce_hi, 16);
      if (peach_cuda_verify(ctx, P, &cand, (word8) P->params[id].diff,
            res.hash, 0) != VEOK) {
         return peach_cuda_fallback(ctx, P, "solve rejected by the CPU");
      }
      if (!report) {
         pdebug("CUDA #%d: solve for an outdated block trailer (block"
            " 0x%x) not reported", ctx->id, (unsigned) get32(cand.bnum));
         return 0;
      }
      memcpy(btout, &cand, sizeof(BTRAILER));
      return 1;
   }

   return 0;
}  /* end peach_cuda_harvest() */

/**
 * @private
 * Launch the next batch on pipeline context @a id (its stream is idle,
 * its previous result harvested or discarded). The batch's trailer
 * snapshot h_bt[id] is @a bt with a first nonce half drawn by the host
 * redraw rule: trigg_generate() until nonce words 0..3 fire no NaN
 * replacement, so the prefix op q after them is index independent. The
 * parameters are the SHA-256 midstate of trailer bytes 0..63, bytes
 * 64..91, the first nonce half, q, the clamped difficulty, the slots per
 * batch, a new epoch and the packed skip masks. Then: start event,
 * peach_pipe_enqueue(), asynchronous copy of the result to h_res[id],
 * stop event. Nothing is launched when no first half qualifies
 * (alerted). A clamped difficulty of 0 (a trailer of difficulty 0)
 * accepts every final hash, as trigg_eval() and the official solver do.
 * @param ctx Pointer to DEVICE_CTX
 * @param P Pointer to its Peach CUDA context
 * @param id Batch context (stream) number
 * @param bt Pointer to (snapshot of the) block trailer to solve for
 * @param diff Difficulty to test against entropy of final hash
 * @returns 0 if launched or skipped, or (-1) if the pipeline solver was
 * abandoned (fallback) or a CUDA error occurred (alerted; ctx->status =
 * DEV_FAIL)
*/
static int peach_cuda_launch(DEVICE_CTX *ctx, PEACH_CUDA_CTX *P, int id,
   const BTRAILER *bt, word8 diff)
{
   PEACH_PIPE_PARAMS *p;
   SHA256_CTX sctx;
   word32 nlo[4], q;
   int draws, nanf, rc;
   cudaError_t err;

#undef cuCHK
#define cuCHK(cuFN) \
   do { \
      err = (cuFN); \
      if (err != cudaSuccess) { \
         const char *str = cudaGetErrorString(err); \
         palert("CUDA ERROR on #(%d): (%d) %s", ctx->id, (int) err, str); \
         palert("... error returned by: %s", #cuFN); \
         /* clear the (non-sticky) error: never left to another device */ \
         (void) cudaGetLastError(); \
         ctx->status = DEV_FAIL; \
         return (-1); \
      } \
   } while(0)

   /* clamped difficulty, as the official solver (0 only for a trailer
    * of difficulty 0, where every final hash is a valid solve) */
   diff = diff && diff < bt->difficulty[0] ? diff : bt->difficulty[0];

   /* host redraw rule: first nonce half without NaN replacement */
   q = 0;
   for (draws = 0; draws < PEACH_CUDA_REDRAW_MAX; draws++) {
      trigg_generate(nlo);
      nanf = 0;
      q = peach_prefix_words(nlo, 4, 0, 0, &nanf);
      if (nanf == 0) break;
   }
   if (draws >= PEACH_CUDA_REDRAW_MAX) {
      palert("CUDA #%d: no first nonce half without NaN replacement in %d"
         " draws; batch skipped", ctx->id, PEACH_CUDA_REDRAW_MAX);
      return 0;
   }

   /* trailer snapshot of this batch (unchanged while in flight) */
   memcpy(P->h_bt[id], bt, sizeof(BTRAILER));
   memcpy(P->h_bt[id]->nonce, nlo, 16);
   memset(P->h_bt[id]->nonce + 16, 0, 16);

   /* batch parameters */
   p = &(P->params[id]);
   memset(p, 0, sizeof(PEACH_PIPE_PARAMS));
   sha256_init(&sctx);
   sha256_update(&sctx, P->h_bt[id], 64);
   memcpy(p->mid, sctx.state, sizeof(p->mid));
   memcpy(p->tail, ((const word8 *) P->h_bt[id]) + 64, sizeof(p->tail));
   memcpy(p->nonce_lo, nlo, sizeof(p->nonce_lo));
   p->q = q;
   p->diff = diff;
   p->nslots = P->nslots;
   if (++(P->epoch) == 0) P->epoch = 1;
   p->epoch = P->epoch;
   p->skip = peach_pipe_skip_pack(P->cfg_skip);

   /* (async) start event, batch, result retrieval, stop event */
   cuCHK(cudaEventRecord(P->ev_start[id], P->stream[id]));
   rc = peach_pipe_enqueue(p, &(P->bufs[id]), &(P->launch), P->stream[id]);
   if (rc != 0) {
      if (rc > 0) {
         palert("CUDA #%d: pipeline batch launch failed: (%d) %s", ctx->id,
            rc, cudaGetErrorString((cudaError_t) rc));
         (void) cudaGetLastError();
      }
      return peach_cuda_fallback(ctx, P, rc > 0 ? "batch launch failed" :
         "invalid batch launch configuration");
   }
   cuCHK(cudaMemcpyAsync(P->h_res[id], P->bufs[id].d_res,
      sizeof(PEACH_PIPE_RESULT), cudaMemcpyDeviceToHost, P->stream[id]));
   cuCHK(cudaEventRecord(P->ev_stop[id], P->stream[id]));
   P->epoch_inflight[id] = p->epoch;
   P->inflight[id] = 1;

   return 0;
}  /* end peach_cuda_launch() */

/**
 * @private
 * Track the block trailers presented to the pipeline solver of a device
 * (once per call): the latest one, and the distinct one seen before it,
 * compared on their first PEACH_CUDA_TRAILER_ID bytes, as gpuminer
 * matches a solve with its current and previous trailer before sending
 * it. A change seen after a pause of the caller (at least
 * PEACH_CUDA_POLL_GAP seconds of time() since the previous call) may
 * hide trailers this solver never saw, so the trailer seen before it may
 * no longer be the caller's previous one (see peach_cuda_reportable()).
 * The caller's trailer changes at most once per network interval (at
 * least one second), so without a pause no trailer is missed.
 * @param P Pointer to Peach CUDA context
 * @param bt Pointer to (snapshot of the) block trailer of this call
*/
static void peach_cuda_track(PEACH_CUDA_CTX *P, const BTRAILER *bt)
{
   time_t now;
   int pause;

   now = time(NULL);
   pause = P->last_poll == 0 ||
      difftime(now, P->last_poll) >= PEACH_CUDA_POLL_GAP;
   P->last_poll = now;
   if (memcmp(P->trail_curr, bt, PEACH_CUDA_TRAILER_ID) != 0) {
      memcpy(P->trail_prev, P->trail_curr, PEACH_CUDA_TRAILER_ID);
      memcpy(P->trail_curr, bt, PEACH_CUDA_TRAILER_ID);
      P->trail_prev_ok = !pause;
   }
}  /* end peach_cuda_track() */

/**
 * @private
 * May a solve found by the batch of pipeline context @a id be reported?
 * Only if the caller can still send it: the batch's trailer (h_bt[id])
 * is the latest trailer seen, or the one before it when that change was
 * seen without a pause (see peach_cuda_track()). Otherwise the batch
 * still counts as work, but its solve is not reported (it would pause
 * the caller for the rest of the block without being sent).
 * @param P Pointer to Peach CUDA context
 * @param id Batch context (stream) number
 * @returns 1 if reportable, else 0
*/
static int peach_cuda_reportable(const PEACH_CUDA_CTX *P, int id)
{
   const void *b = P->h_bt[id];

   if (memcmp(b, P->trail_curr, PEACH_CUDA_TRAILER_ID) == 0) return 1;
   return P->trail_prev_ok &&
      memcmp(b, P->trail_prev, PEACH_CUDA_TRAILER_ID) == 0;
}  /* end peach_cuda_reportable() */

/**
 * @private
 * Check that every stream of a device other than @a id is idle: the two
 * legacy streams and the stream of every pipeline batch context.
 * @param P Pointer to Peach CUDA context
 * @param id Stream to leave out, or (-1)
 * @param errp Pointer to place a CUDA error (other than
 * cudaErrorNotReady), else cudaSuccess
 * @returns 1 if they are all idle, else 0 (busy, or a CUDA error)
*/
static int peach_cuda_idle(const PEACH_CUDA_CTX *P, int id,
   cudaError_t *errp)
{
   cudaError_t err;
   int i, n;

   *errp = cudaSuccess;
   n = P->nctx > 2 ? P->nctx : 2;
   for (i = 0; i < n; i++) {
      if (i == id) continue;
      err = cudaStreamQuery(P->stream[i]);
      if (err == cudaErrorNotReady) return 0;
      if (err != cudaSuccess) {
         *errp = err;
         return 0;
      }
   }

   return 1;
}  /* end peach_cuda_idle() */

/**
 * @private
 * Pipeline Peach CUDA solver: the official state machine, with one
 * pipeline batch context per stream (P->nctx streams; the map is built
 * on streams 0 and 1, as official), and these rules:
 * - DEV_INIT: the map build cursor is P->build_next (ctx->work counts
 *   hashes only); the pre-build (all streams idle) discards every batch
 *   result and records the phash of the map; when the map is complete and
 *   all streams are idle, the transition table is built and
 *   synchronized, then the self-test runs, then DEV_IDLE;
 * - DEV_IDLE -> DEV_WORK gate as official;
 * - DEV_WORK, per idle stream, in this order: a trailer phash other than
 *   the map's -> DEV_INIT; no transactions, block already solved or
 *   expired -> DEV_IDLE (results discarded, nothing reported); harvest
 *   the batch result (peach_cuda_harvest()), reporting its solve only if
 *   the caller can still send it (peach_cuda_reportable()); launch the
 *   next batch (peach_cuda_launch()).
 * Every call tracks the trailers seen (peach_cuda_track()), at any
 * polling interval of the caller.
 * Any pipeline defect switches the device to the legacy solver.
 * @param ctx Pointer to DEVICE_CTX to perform work with
 * @param bt Pointer to (snapshot of the) block trailer to solve for
 * @param diff Difficulty to test against entropy of final hash
 * @param btout Pointer to location to place solved block trailer
 * @returns VEOK on solve, VERROR on no solve, or VETIMEOUT if GPU is
 * either stopped or unrecoverable.
*/
static int peach_solve_cuda_pipeline(DEVICE_CTX *ctx, const BTRAILER *bt,
   word8 diff, BTRAILER *btout)
{
   PEACH_CUDA_CTX *P;
   double delta;
   int id, i, grid, block, build, rc;
   cudaError_t err;

#undef cuCHK
#define cuCHK(cuFN) \
   do { \
      err = (cuFN); \
      if (err != cudaSuccess) { \
         const char *str = cudaGetErrorString(err); \
         palert("CUDA ERROR on #(%d): (%d) %s", ctx->id, (int) err, str); \
         palert("... error returned by: %s", #cuFN); \
         /* clear the (non-sticky) error: never left to another device */ \
         (void) cudaGetLastError(); \
         ctx->status = DEV_FAIL; \
         return VERROR; \
      } \
   } while(0)

   /* init */
   P = (PEACH_CUDA_CTX *) ctx->peach;
   /* report unuseable GPUs */
   if (ctx->status < DEV_NULL) return VETIMEOUT;
   /* the trailers seen, for the solves the caller can send */
   peach_cuda_track(P, bt);

   /* set cuda device */
   cuCHK(cudaSetDevice(ctx->id));
   /* check for previous (async) execution errors */
   cuCHK(cudaGetLastError());

   /* build peach map and transition table */
   if (ctx->status == DEV_INIT) {
      for (build = id = 0; id < 2; id++) {
         /* check stream is ready */
         err = cudaStreamQuery(P->stream[id]);
         if (err == cudaErrorNotReady) continue;
         cuCHK(err);

         /* check pre-build state */
         if (P->build_next == 0 && build == 0) {
            /* ensure every other stream is ready */
            if (!peach_cuda_idle(P, id, &err)) {
               cuCHK(err);
               break;
            }
            /* discard every batch result and in-flight state */
            for (i = 0; i < P->nctx; i++) {
               cuCHK(cudaMemset(P->bufs[i].d_res, 0,
                  sizeof(PEACH_PIPE_RESULT)));
               memset(P->h_res[i], 0, sizeof(PEACH_PIPE_RESULT));
               P->inflight[i] = 0;
               P->epoch_inflight[i] = 0;
               memcpy(P->h_bt[i], bt, sizeof(BTRAILER));
            }
            /* record the map's phash; start a new epoch */
            memcpy(P->map_phash, bt->phash, HASHLEN);
            if (++(P->epoch) == 0) P->epoch = 1;
            P->batches = 0;
            /* full batches again: a slowdown during the previous block
             * must not shrink the batches of every later block */
            P->nslots = P->cap;
            P->slow_batches = P->fast_batches = 0;
            /* update device phash */
            cuCHK(cudaMemcpy(P->d_phash, P->map_phash, HASHLEN,
               cudaMemcpyHostToDevice));
            /* synchronize memory transfers before building peach map */
            cuCHK(cudaDeviceSynchronize());
            /* flag build state */
            build = 1;
         }
         /* check build state */
         if (P->build_next > 0 || build) {
            if (P->build_next < PEACHCACHELEN) {
               /* prepare launch config and generate peach map */
               cuCHK(cudaOccupancyMaxPotentialBlockSize(&grid, &block,
                  kcu_peach_build, 0, 0));
               CUDA_KERNEL(kcu_peach_build, grid, block, 0, P->stream[id])
                  (P->build_next, P->d_map, P->d_phash);
               cuCHK(cudaGetLastError());
               /* update build progress (map/T cursor, not ctx->work) */
               P->build_next += (word32) grid * (word32) block;
            } else {
               /* ensure every other stream is finished */
               if (!peach_cuda_idle(P, id, &err)) {
                  cuCHK(err);
                  break;
               }
               /* the map is complete and all streams are idle: build
                * the transition table and wait for it */
               CUDA_KERNEL(kcu_peach_pipe_transitions,
                  PEACHCACHELEN / PEACH_PIPE_BLOCK, PEACH_PIPE_BLOCK, 0,
                  P->stream[id])((const uint4 *) P->d_map, P->d_T,
                  (word32) 0, (word32) PEACHCACHELEN);
               cuCHK(cudaGetLastError());
               cuCHK(cudaStreamSynchronize(P->stream[id]));
               /* self-test (T and float arithmetic vs the CPU) */
               rc = peach_cuda_selftest(ctx, P, P->stream[id]);
               if (rc < 0) return VERROR;
               if (rc > 0) {
                  peach_cuda_fallback(ctx, P, "self-test mismatch");
                  return VERROR;
               }
               /* build is complete */
               ctx->last = time(NULL);
               ctx->status = DEV_IDLE;
               ctx->work = 0;
               P->build_next = 0;
               break;
            }
         }  /* end if (P->build_next > 0... */
      }  /* end for(build = id = 0... */
   }  /* end if (ctx->status == DEV_INIT)... */

   /* switch to WORK mode when all conditions are satisfied:
    * - transactions to solve
    * - block NOT already solved
    * - block NOT expired
    */
   while (ctx->status == DEV_IDLE) {
      if (get32(bt->tcount) == 0) break;
      if (cmp64(bt->bnum, btout->bnum) == 0) break;
      if (difftime(time(NULL), get32(bt->time0)) >= BRIDGEv3) break;
      ctx->last = time(NULL);
      ctx->status = DEV_WORK;
      ctx->work = 0;
      break;
   }

   /* solve work in block trailer */
   if (ctx->status == DEV_WORK) {
      for (id = 0; id < P->nctx; id++) {
         err = cudaStreamQuery(P->stream[id]);
         if (err == cudaErrorNotReady) continue;
         cuCHK(err);
         /* (1) block update: the map (and T) is for another phash;
          * results in flight are discarded */
         if (memcmp(P->map_phash, bt->phash, HASHLEN)) {
            memset(P->inflight, 0, sizeof(P->inflight));
            ctx->status = DEV_INIT;
            ctx->work = 0;
            P->build_next = 0;
            break;
         }
         /* (2) switch to IDLE mode when reasonable, without reporting
          * (results in flight are discarded):
          * - no transaction to solve
          * - block already solved
          * - block expired
          */
         if (get32(bt->tcount) == 0 || cmp64(bt->bnum, btout->bnum) == 0 ||
               difftime(time(NULL), get32(bt->time0)) >= BRIDGEv3) {
            memset(P->inflight, 0, sizeof(P->inflight));
            ctx->status = DEV_IDLE;
            ctx->work = 0;
            break;
         }
         /* (3) harvest the finished batch of this stream; report its
          * solve only if the caller can still send it */
         if (P->inflight[id]) {
            rc = peach_cuda_harvest(ctx, P, id, btout,
               peach_cuda_reportable(P, id));
            if (rc > 0) return VEOK;
            if (rc < 0) return VERROR;
         }
         /* (4) launch the next batch */
         if (peach_cuda_launch(ctx, P, id, bt, diff) < 0) return VERROR;
         /* update hashrate (completed nonces) */
         delta = difftime(time(NULL), ctx->last);
         ctx->hps = ctx->work / (delta ? delta : 1);
      }  /* end for(id = 0; id < P->nctx; id++)... */
   }  /* end if (ctx->status == DEV_WORK)... */

   return VERROR;
}  /* end peach_solve_cuda_pipeline() */

/**
 * Try solve for a tokenized haiku as nonce output for Peach proof of work
 * on CUDA devices. Combine haiku protocols implemented in the Trigg
 * Algorithm with the intensive protocols of the Peach algorithm to
 * generate haiku output as proof of work.
 * @param ctx Pointer to DEVICE_CTX to perform work with
 * @param bt Pointer to block trailer to solve for
 * @param diff Difficulty to test against entropy of final hash
 * @param btout Pointer to location to place solved block trailer
 * @returns VEOK on solve, VERROR on no solve, or VETIMEOUT if GPU is
 * either stopped or unrecoverable.
 * @note @a bt may be updated by another thread meanwhile: the solver only
 * uses a stable snapshot of it, taken on entry (best effort, see
 * peach_cuda_snapshot()). A solve is written to @a btout only after
 * peach_checkhash() confirmed it on the CPU.
 * @note Uses the pipeline solver when it was set up by
 * peach_init_cuda_device() and has not been abandoned after a defect
 * (see peach_pipeline_cuda_device()), else the official (legacy) solver.
*/
int peach_solve_cuda(DEVICE_CTX *ctx, BTRAILER *bt, word8 diff, BTRAILER *btout)
{
   PEACH_CUDA_CTX *P;
   BTRAILER snap;

   /* report unuseable (or uninitialized) GPUs */
   if (ctx->status < DEV_NULL) return VETIMEOUT;
   P = (PEACH_CUDA_CTX *) ctx->peach;
   if (P == NULL) return VETIMEOUT;
   /* use ONLY a (stable) snapshot of the (shared) block trailer */
   if (peach_cuda_snapshot(bt, &snap) != VEOK) {
      pdebug("CUDA #%d: block trailer is changing, retry later", ctx->id);
      return VERROR;
   }

   /* pipeline solver, unless configured legacy or abandoned (fallback) */
   if (P->mode == PEACH_CUDA_MODE_PIPELINE && !P->fallback) {
      return peach_solve_cuda_pipeline(ctx, &snap, diff, btout);
   }

   return peach_solve_cuda_legacy(ctx, &snap, diff, btout);
}  /* end peach_solve_cuda() */

/**
 * Check which Peach solver a CUDA device context uses.
 * @param ctx Pointer to DEVICE_CTX to check
 * @returns 1 if peach_solve_cuda() uses the pipeline solver on the
 * device, 0 if it uses the official (legacy) solver (MCM_PEACH_LEGACY=1,
 * failed pipeline setup, or a switch after a pipeline defect), or (-1)
 * if the device context is not initialized
*/
int peach_pipeline_cuda_device(const DEVICE_CTX *ctx)
{
   const PEACH_CUDA_CTX *P = (const PEACH_CUDA_CTX *) ctx->peach;

   if (P == NULL) return (-1);

   return (P->mode == PEACH_CUDA_MODE_PIPELINE && !P->fallback) ? 1 : 0;
}  /* end peach_pipeline_cuda_device() */

/* end include guard */
#endif
