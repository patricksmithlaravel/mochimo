/**
 * @file peach_hash64.cuh
 * @brief Peach jump hashes over the seed view: 64-bit lane family.
 * @details Blake2b-256 keyed with 32 zero bytes (algo 0) or with 64
 * bytes of 0x01 (algo 1), SHA3-256 (algo 4) and Keccak-256 (algo 5) of
 * the 1060-byte Peach jump seed
 * `nonce (32 B) || m (4 B, little-endian) || tile[m] (1024 B)`,
 * computed from its parts without materializing the seed: the 8 nonce
 * words, the tile index m and the 16-byte aligned tile (64 x uint4,
 * every vector loaded exactly once with PEACH_LDG128()). The 64-bit
 * message lanes are paired from those words; the 1..3 words of a load
 * that cross a block boundary are carried into the next block.
 * <br />
 * Outputs are the 8 digest words exactly as peach_nighthash() writes
 * them on a little-endian host (word i = digest bytes 4i..4i+3).
 * <br />
 * Device code (PEACH_DEV): CUDA C++ under nvcc, plain C after
 * test/_cuda_emu.h. All state lives in registers (0-byte stack frame).
 * Blake2b uses literal-sigma rounds over named words and one rolled
 * compression body for all 9 blocks (key block fast-forwarded). Keccak-f
 * is the 4-round unrolled loop of crypto-c sha3_keccakf_unrolled() (after
 * Marko Kreen's spongeshaker transform), one rolled permutation body for
 * all 8 blocks.
 * @copyright Adequate Systems LLC, 2018-2025. All Rights Reserved.
 * <br />For license information, please refer to ../LICENSE.md
 * <br /><br />
 * The Keccak-f permutation (peach_keccakf()) is derived from crypto-c
 * sha3.h, which carries the following notices:
 *
 * @copyright 2014 Marko Kreen <markokr@gmail.com>
 *
 * Permission to use, copy, modify, and/or distribute this software
 * for any purpose with or without fee is hereby granted, provided
 * that the above copyright notice and this permission notice appear
 * in all copies.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL
 * WARRANTIES WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL
 * THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT, INDIRECT, OR
 * CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
 * LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT,
 * NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN
 * CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
 *
 * @copyright 2015 Markku-Juhani O. Saarinen <mjos@iki.fi>
 * @copyright 2020-2022 Adequate Systems, LLC.
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use, copy,
 * modify, merge, publish, distribute, sublicense, and/or sell copies
 * of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * * The above copyright notice and this permission notice shall be
 *   included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
 * OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
*/

/* include guard */
#ifndef MOCHIMO_PEACH_HASH64_CUH
#define MOCHIMO_PEACH_HASH64_CUH


#include "extint.h"           /* for word types */
#include "peach_compat.cuh"   /* for PEACH_DEV, PEACH_LDG128(), uint4 */

/* keep the next loop rolled on device: one hash body per kernel
 * (the plain C build of the CPU emulation has nothing to unroll) */
#ifdef __CUDACC__
   #define PEACH_H64_ROLLED   _Pragma("unroll 1")
#else
   #define PEACH_H64_ROLLED
#endif

/* 64-bit little-endian lane from two 32-bit words */
#define PEACH_H64_LANE(lo, hi)   ((word64) (lo) | ((word64) (hi) << 32))

/* Blake2b initialization vectors */
#define PEACH_B2B_IV0   WORD64_C(0x6A09E667F3BCC908)
#define PEACH_B2B_IV1   WORD64_C(0xBB67AE8584CAA73B)
#define PEACH_B2B_IV2   WORD64_C(0x3C6EF372FE94F82B)
#define PEACH_B2B_IV3   WORD64_C(0xA54FF53A5F1D36F1)
#define PEACH_B2B_IV4   WORD64_C(0x510E527FADE682D1)
#define PEACH_B2B_IV5   WORD64_C(0x9B05688C2B3E6C1F)
#define PEACH_B2B_IV6   WORD64_C(0x1F83D9ABFB41BD6B)
#define PEACH_B2B_IV7   WORD64_C(0x5BE0CD19137E2179)

/* Blake2b-256 state after the key block of peach_nighthash()'s keys:
 * blake2b_init(key, keylen, 32), then the compression of the zero padded
 * key block (t = 128, not final). As cu_peach_blake2b() in peach.cu.
 * KEY32: 32 zero bytes (algo 0); KEY64: 64 bytes of 0x01 (algo 1). */
#define PEACH_B2B_KEY32_H0   WORD64_C(0x63320ACE264383EB)
#define PEACH_B2B_KEY32_H1   WORD64_C(0x012AF5FD045A2737)
#define PEACH_B2B_KEY32_H2   WORD64_C(0xF4F49C55E6BE39DF)
#define PEACH_B2B_KEY32_H3   WORD64_C(0x791C5BC8AFFB11A7)
#define PEACH_B2B_KEY32_H4   WORD64_C(0xC9BCACC002C0EA21)
#define PEACH_B2B_KEY32_H5   WORD64_C(0x8295B8ABE2FDEDD6)
#define PEACH_B2B_KEY32_H6   WORD64_C(0xB711490E5F9F41C8)
#define PEACH_B2B_KEY32_H7   WORD64_C(0x3F8E4D1D9EBEAF1A)
#define PEACH_B2B_KEY64_H0   WORD64_C(0x00B8AA23C261EF69)
#define PEACH_B2B_KEY64_H1   WORD64_C(0xD38AE6ABCA237B9E)
#define PEACH_B2B_KEY64_H2   WORD64_C(0x67FB881E5EE89069)
#define PEACH_B2B_KEY64_H3   WORD64_C(0x3E5B8BD06B58D002)
#define PEACH_B2B_KEY64_H4   WORD64_C(0x252D3F68395AAE91)
#define PEACH_B2B_KEY64_H5   WORD64_C(0xD25465E23C6C1B27)
#define PEACH_B2B_KEY64_H6   WORD64_C(0x852B4CC2E13303B5)
#define PEACH_B2B_KEY64_H7   WORD64_C(0x3F38B9FF245BE7C1)

/* Blake2b G mixing of the named state words a, b, c, d */
#define PEACH_B2B_G(a, b, c, d, x, y) \
   do { \
      a += b + (x); d = peach_rotr64(d ^ a, 32); \
      c += d; b = peach_rotr64(b ^ c, 24); \
      a += b + (y); d = peach_rotr64(d ^ a, 16); \
      c += d; b = peach_rotr64(b ^ c, 63); \
   } while (0)

/* one Blake2b round over the named state words v0..v15; the arguments
 * are the message words in the order of that round's sigma row */
#define PEACH_B2B_ROUND(s0, s1, s2, s3, s4, s5, s6, s7, \
   s8, s9, s10, s11, s12, s13, s14, s15) \
   do { \
      PEACH_B2B_G(v0, v4, v8, v12, s0, s1); \
      PEACH_B2B_G(v1, v5, v9, v13, s2, s3); \
      PEACH_B2B_G(v2, v6, v10, v14, s4, s5); \
      PEACH_B2B_G(v3, v7, v11, v15, s6, s7); \
      PEACH_B2B_G(v0, v5, v10, v15, s8, s9); \
      PEACH_B2B_G(v1, v6, v11, v12, s10, s11); \
      PEACH_B2B_G(v2, v7, v8, v13, s12, s13); \
      PEACH_B2B_G(v3, v4, v9, v14, s14, s15); \
   } while (0)

/* Keccak-f[1600] round constants */
static PEACH_CONST word64 c_peach_keccakf_rndc[24] = {
   WORD64_C(0x0000000000000001), WORD64_C(0x0000000000008082),
   WORD64_C(0x800000000000808A), WORD64_C(0x8000000080008000),
   WORD64_C(0x000000000000808B), WORD64_C(0x0000000080000001),
   WORD64_C(0x8000000080008081), WORD64_C(0x8000000000008009),
   WORD64_C(0x000000000000008A), WORD64_C(0x0000000000000088),
   WORD64_C(0x0000000080008009), WORD64_C(0x000000008000000A),
   WORD64_C(0x000000008000808B), WORD64_C(0x800000000000008B),
   WORD64_C(0x8000000000008089), WORD64_C(0x8000000000008003),
   WORD64_C(0x8000000000008002), WORD64_C(0x8000000000000080),
   WORD64_C(0x000000000000800A), WORD64_C(0x800000008000000A),
   WORD64_C(0x8000000080008081), WORD64_C(0x8000000000008080),
   WORD64_C(0x0000000080000001), WORD64_C(0x8000000080008008)
};

/**
 * @private
 * Keyed Blake2b-256 of a Peach jump seed (seed view), starting from the
 * state after the key block. Message block j = 0..7 holds seed bytes
 * 128j..128j+127 and is compressed with t = 128 (key block) + 128(j + 1);
 * the final block holds the last 36 bytes (zero padded) and is
 * compressed with t = 128 + 1060 and the final flag.
 * Loads (tile vectors, c = carried word, j = 1..7):
 * block 0 = n0..n7, index, tile[0..4], tile[5].xyz, c = tile[5].w;
 * block j = c, tile[8j-2..8j+4], tile[8j+5].xyz, c = tile[8j+5].w;
 * final = c, tile[62], tile[63].
 * @param n Pointer to the nonce, 8 words
 * @param index Tile index m (seed word 8)
 * @param tile Pointer to tile m, 64 x uint4 (16-byte aligned)
 * @param key64 Non-zero: 64 bytes of 0x01 key (algo 1), else 32 zero
 * bytes (algo 0)
 * @param out Pointer to 8 words for the digest (little-endian words)
*/
PEACH_DEV void peach_sh_blake2b(const word32 *n, word32 index,
   const uint4 *tile, int key64, word32 *out)
{
   word64 h0, h1, h2, h3, h4, h5, h6, h7, t, f;
   word64 v0, v1, v2, v3, v4, v5, v6, v7;
   word64 v8, v9, v10, v11, v12, v13, v14, v15;
   word64 m0, m1, m2, m3, m4, m5, m6, m7;
   word64 m8, m9, m10, m11, m12, m13, m14, m15;
   uint4 q0, q1, q2, q3, q4, q5, q6, q7;
   const uint4 *tp;
   word32 carry;
   int k;

   /* fast-forward: state after the key block */
   h0 = key64 ? PEACH_B2B_KEY64_H0 : PEACH_B2B_KEY32_H0;
   h1 = key64 ? PEACH_B2B_KEY64_H1 : PEACH_B2B_KEY32_H1;
   h2 = key64 ? PEACH_B2B_KEY64_H2 : PEACH_B2B_KEY32_H2;
   h3 = key64 ? PEACH_B2B_KEY64_H3 : PEACH_B2B_KEY32_H3;
   h4 = key64 ? PEACH_B2B_KEY64_H4 : PEACH_B2B_KEY32_H4;
   h5 = key64 ? PEACH_B2B_KEY64_H5 : PEACH_B2B_KEY32_H5;
   h6 = key64 ? PEACH_B2B_KEY64_H6 : PEACH_B2B_KEY32_H6;
   h7 = key64 ? PEACH_B2B_KEY64_H7 : PEACH_B2B_KEY32_H7;
   carry = 0;

   /* 8 message blocks + the final (36-byte) block, one body */
   PEACH_H64_ROLLED
   for (k = 0; k < 9; k++) {
      if (k == 0) {
         /* seed words 0..31: nonce, index, tile words 0..22 */
         q0 = PEACH_LDG128(&tile[0]);
         q1 = PEACH_LDG128(&tile[1]);
         q2 = PEACH_LDG128(&tile[2]);
         q3 = PEACH_LDG128(&tile[3]);
         q4 = PEACH_LDG128(&tile[4]);
         q5 = PEACH_LDG128(&tile[5]);
         m0 = PEACH_H64_LANE(n[0], n[1]);
         m1 = PEACH_H64_LANE(n[2], n[3]);
         m2 = PEACH_H64_LANE(n[4], n[5]);
         m3 = PEACH_H64_LANE(n[6], n[7]);
         m4 = PEACH_H64_LANE(index, q0.x);
         m5 = PEACH_H64_LANE(q0.y, q0.z);
         m6 = PEACH_H64_LANE(q0.w, q1.x);
         m7 = PEACH_H64_LANE(q1.y, q1.z);
         m8 = PEACH_H64_LANE(q1.w, q2.x);
         m9 = PEACH_H64_LANE(q2.y, q2.z);
         m10 = PEACH_H64_LANE(q2.w, q3.x);
         m11 = PEACH_H64_LANE(q3.y, q3.z);
         m12 = PEACH_H64_LANE(q3.w, q4.x);
         m13 = PEACH_H64_LANE(q4.y, q4.z);
         m14 = PEACH_H64_LANE(q4.w, q5.x);
         m15 = PEACH_H64_LANE(q5.y, q5.z);
         carry = q5.w;
         t = 256;
         f = 0;
      } else if (k < 8) {
         /* seed words 32k..32k+31: tile words 32k-9..32k+22 */
         tp = &tile[(8 * k) - 2];
         q0 = PEACH_LDG128(&tp[0]);
         q1 = PEACH_LDG128(&tp[1]);
         q2 = PEACH_LDG128(&tp[2]);
         q3 = PEACH_LDG128(&tp[3]);
         q4 = PEACH_LDG128(&tp[4]);
         q5 = PEACH_LDG128(&tp[5]);
         q6 = PEACH_LDG128(&tp[6]);
         q7 = PEACH_LDG128(&tp[7]);
         m0 = PEACH_H64_LANE(carry, q0.x);
         m1 = PEACH_H64_LANE(q0.y, q0.z);
         m2 = PEACH_H64_LANE(q0.w, q1.x);
         m3 = PEACH_H64_LANE(q1.y, q1.z);
         m4 = PEACH_H64_LANE(q1.w, q2.x);
         m5 = PEACH_H64_LANE(q2.y, q2.z);
         m6 = PEACH_H64_LANE(q2.w, q3.x);
         m7 = PEACH_H64_LANE(q3.y, q3.z);
         m8 = PEACH_H64_LANE(q3.w, q4.x);
         m9 = PEACH_H64_LANE(q4.y, q4.z);
         m10 = PEACH_H64_LANE(q4.w, q5.x);
         m11 = PEACH_H64_LANE(q5.y, q5.z);
         m12 = PEACH_H64_LANE(q5.w, q6.x);
         m13 = PEACH_H64_LANE(q6.y, q6.z);
         m14 = PEACH_H64_LANE(q6.w, q7.x);
         m15 = PEACH_H64_LANE(q7.y, q7.z);
         carry = q7.w;
         t = (word64) (128 * (k + 2));
         f = 0;
      } else {
         /* seed words 256..264: tile words 247..255, zero padded */
         q0 = PEACH_LDG128(&tile[62]);
         q1 = PEACH_LDG128(&tile[63]);
         m0 = PEACH_H64_LANE(carry, q0.x);
         m1 = PEACH_H64_LANE(q0.y, q0.z);
         m2 = PEACH_H64_LANE(q0.w, q1.x);
         m3 = PEACH_H64_LANE(q1.y, q1.z);
         m4 = (word64) q1.w;
         m5 = m6 = m7 = m8 = m9 = m10 = m11 = m12 = 0;
         m13 = m14 = m15 = 0;
         t = 128 + 1060;
         f = ~((word64) 0);
      }
      /* compression with literal sigma */
      v0 = h0; v1 = h1; v2 = h2; v3 = h3;
      v4 = h4; v5 = h5; v6 = h6; v7 = h7;
      v8 = PEACH_B2B_IV0; v9 = PEACH_B2B_IV1;
      v10 = PEACH_B2B_IV2; v11 = PEACH_B2B_IV3;
      v12 = PEACH_B2B_IV4 ^ t; v13 = PEACH_B2B_IV5;
      v14 = PEACH_B2B_IV6 ^ f; v15 = PEACH_B2B_IV7;
      PEACH_B2B_ROUND(m0, m1, m2, m3, m4, m5, m6, m7,
         m8, m9, m10, m11, m12, m13, m14, m15);
      PEACH_B2B_ROUND(m14, m10, m4, m8, m9, m15, m13, m6,
         m1, m12, m0, m2, m11, m7, m5, m3);
      PEACH_B2B_ROUND(m11, m8, m12, m0, m5, m2, m15, m13,
         m10, m14, m3, m6, m7, m1, m9, m4);
      PEACH_B2B_ROUND(m7, m9, m3, m1, m13, m12, m11, m14,
         m2, m6, m5, m10, m4, m0, m15, m8);
      PEACH_B2B_ROUND(m9, m0, m5, m7, m2, m4, m10, m15,
         m14, m1, m11, m12, m6, m8, m3, m13);
      PEACH_B2B_ROUND(m2, m12, m6, m10, m0, m11, m8, m3,
         m4, m13, m7, m5, m15, m14, m1, m9);
      PEACH_B2B_ROUND(m12, m5, m1, m15, m14, m13, m4, m10,
         m0, m7, m6, m3, m9, m2, m8, m11);
      PEACH_B2B_ROUND(m13, m11, m7, m14, m12, m1, m3, m9,
         m5, m0, m15, m4, m8, m6, m2, m10);
      PEACH_B2B_ROUND(m6, m15, m14, m9, m11, m3, m0, m8,
         m12, m2, m13, m7, m1, m4, m10, m5);
      PEACH_B2B_ROUND(m10, m2, m8, m4, m7, m6, m1, m5,
         m15, m11, m9, m14, m3, m12, m13, m0);
      PEACH_B2B_ROUND(m0, m1, m2, m3, m4, m5, m6, m7,
         m8, m9, m10, m11, m12, m13, m14, m15);
      PEACH_B2B_ROUND(m14, m10, m4, m8, m9, m15, m13, m6,
         m1, m12, m0, m2, m11, m7, m5, m3);
      h0 ^= v0 ^ v8; h1 ^= v1 ^ v9; h2 ^= v2 ^ v10; h3 ^= v3 ^ v11;
      h4 ^= v4 ^ v12; h5 ^= v5 ^ v13; h6 ^= v6 ^ v14; h7 ^= v7 ^ v15;
   }

   /* 256-bit digest = h0..h3 (little-endian) */
   out[0] = (word32) h0; out[1] = (word32) (h0 >> 32);
   out[2] = (word32) h1; out[3] = (word32) (h1 >> 32);
   out[4] = (word32) h2; out[5] = (word32) (h2 >> 32);
   out[6] = (word32) h3; out[7] = (word32) (h3 >> 32);
}  /* end peach_sh_blake2b() */

/**
 * @private
 * Keccak-f[1600] permutation of a state held in registers, 4 rounds per
 * loop iteration (lane positions repeat every 4 rounds), as crypto-c
 * sha3_keccakf_unrolled(). Only call with a local array (constant
 * indices only) that the forced inlining keeps in registers.
 * @param st Pointer to the state, 25 lanes
*/
PEACH_DEV void peach_keccakf(word64 *st)
{
   word64 Ba, Be, Bi, Bo, Bu, Ca, Ce, Ci, Co, Cu, Da, De, Di, Do, Du;
   int r;

   PEACH_H64_ROLLED
   for (r = 0; r < 24; r += 4) {
      /* round r + 0 */
      Ca = st[0] ^ st[5] ^ st[10] ^ st[15] ^ st[20];
      Ce = st[1] ^ st[6] ^ st[11] ^ st[16] ^ st[21];
      Ci = st[2] ^ st[7] ^ st[12] ^ st[17] ^ st[22];
      Co = st[3] ^ st[8] ^ st[13] ^ st[18] ^ st[23];
      Cu = st[4] ^ st[9] ^ st[14] ^ st[19] ^ st[24];
      Da = Cu ^ peach_rotl64(Ce, 1);
      De = Ca ^ peach_rotl64(Ci, 1);
      Di = Ce ^ peach_rotl64(Co, 1);
      Do = Ci ^ peach_rotl64(Cu, 1);
      Du = Co ^ peach_rotl64(Ca, 1);
      Ba = (st[0] ^ Da);
      Be = peach_rotl64((st[6] ^ De), 44);
      Bi = peach_rotl64((st[12] ^ Di), 43);
      Bo = peach_rotl64((st[18] ^ Do), 21);
      Bu = peach_rotl64((st[24] ^ Du), 14);
      st[0] = Ba ^ ((~Be) & Bi) ^ c_peach_keccakf_rndc[r];
      st[6]  = Be ^ ((~Bi) & Bo);
      st[12] = Bi ^ ((~Bo) & Bu);
      st[18] = Bo ^ ((~Bu) & Ba);
      st[24] = Bu ^ ((~Ba) & Be);
      Bi = peach_rotl64((st[10] ^ Da), 3);
      Bo = peach_rotl64((st[16] ^ De), 45);
      Bu = peach_rotl64((st[22] ^ Di), 61);
      Ba = peach_rotl64((st[3] ^ Do), 28);
      Be = peach_rotl64((st[9] ^ Du), 20);
      st[10] = Ba ^ ((~Be) & Bi);
      st[16] = Be ^ ((~Bi) & Bo);
      st[22] = Bi ^ ((~Bo) & Bu);
      st[3] = Bo ^ ((~Bu) & Ba);
      st[9] = Bu ^ ((~Ba) & Be);
      Bu = peach_rotl64((st[20] ^ Da), 18);
      Ba = peach_rotl64((st[1] ^ De), 1);
      Be = peach_rotl64((st[7] ^ Di), 6);
      Bi = peach_rotl64((st[13] ^ Do), 25);
      Bo = peach_rotl64((st[19] ^ Du), 8);
      st[20] = Ba ^ ((~Be) & Bi);
      st[1] = Be ^ ((~Bi) & Bo);
      st[7] = Bi ^ ((~Bo) & Bu);
      st[13] = Bo ^ ((~Bu) & Ba);
      st[19] = Bu ^ ((~Ba) & Be);
      Be = peach_rotl64((st[5] ^ Da), 36);
      Bi = peach_rotl64((st[11] ^ De), 10);
      Bo = peach_rotl64((st[17] ^ Di), 15);
      Bu = peach_rotl64((st[23] ^ Do), 56);
      Ba = peach_rotl64((st[4] ^ Du), 27);
      st[5] = Ba ^ ((~Be) & Bi);
      st[11] = Be ^ ((~Bi) & Bo);
      st[17] = Bi ^ ((~Bo) & Bu);
      st[23] = Bo ^ ((~Bu) & Ba);
      st[4] = Bu ^ ((~Ba) & Be);
      Bo = peach_rotl64((st[15] ^ Da), 41);
      Bu = peach_rotl64((st[21] ^ De), 2);
      Ba = peach_rotl64((st[2] ^ Di), 62);
      Be = peach_rotl64((st[8] ^ Do), 55);
      Bi = peach_rotl64((st[14] ^ Du), 39);
      st[15] = Ba ^ ((~Be) & Bi);
      st[21] = Be ^ ((~Bi) & Bo);
      st[2] = Bi ^ ((~Bo) & Bu);
      st[8] = Bo ^ ((~Bu) & Ba);
      st[14] = Bu ^ ((~Ba) & Be);

      /* round r + 1 */
      Ca = st[0] ^ st[10] ^ st[20] ^ st[5] ^ st[15];
      Ce = st[6] ^ st[16] ^ st[1] ^ st[11] ^ st[21];
      Ci = st[12] ^ st[22] ^ st[7] ^ st[17] ^ st[2];
      Co = st[18] ^ st[3] ^ st[13] ^ st[23] ^ st[8];
      Cu = st[24] ^ st[9] ^ st[19] ^ st[4] ^ st[14];
      Da = Cu ^ peach_rotl64(Ce, 1);
      De = Ca ^ peach_rotl64(Ci, 1);
      Di = Ce ^ peach_rotl64(Co, 1);
      Do = Ci ^ peach_rotl64(Cu, 1);
      Du = Co ^ peach_rotl64(Ca, 1);
      Ba = (st[0] ^ Da);
      Be = peach_rotl64((st[16] ^ De), 44);
      Bi = peach_rotl64((st[7] ^ Di), 43);
      Bo = peach_rotl64((st[23] ^ Do), 21);
      Bu = peach_rotl64((st[14] ^ Du), 14);
      st[0] = Ba ^ ((~Be) & Bi) ^ c_peach_keccakf_rndc[r + 1];
      st[16] = Be ^ ((~Bi) & Bo);
      st[7] = Bi ^ ((~Bo) & Bu);
      st[23] = Bo ^ ((~Bu) & Ba);
      st[14] = Bu ^ ((~Ba) & Be);
      Bi = peach_rotl64((st[20] ^ Da), 3);
      Bo = peach_rotl64((st[11] ^ De), 45);
      Bu = peach_rotl64((st[2] ^ Di), 61);
      Ba = peach_rotl64((st[18] ^ Do), 28);
      Be = peach_rotl64((st[9] ^ Du), 20);
      st[20] = Ba ^ ((~Be) & Bi);
      st[11] = Be ^ ((~Bi) & Bo);
      st[2] = Bi ^ ((~Bo) & Bu);
      st[18] = Bo ^ ((~Bu) & Ba);
      st[9] = Bu ^ ((~Ba) & Be);
      Bu = peach_rotl64((st[15] ^ Da), 18);
      Ba = peach_rotl64((st[6] ^ De), 1);
      Be = peach_rotl64((st[22] ^ Di), 6);
      Bi = peach_rotl64((st[13] ^ Do), 25);
      Bo = peach_rotl64((st[4] ^ Du), 8);
      st[15] = Ba ^ ((~Be) & Bi);
      st[6] = Be ^ ((~Bi) & Bo);
      st[22] = Bi ^ ((~Bo) & Bu);
      st[13] = Bo ^ ((~Bu) & Ba);
      st[4] = Bu ^ ((~Ba) & Be);
      Be = peach_rotl64((st[10] ^ Da), 36);
      Bi = peach_rotl64((st[1] ^ De), 10);
      Bo = peach_rotl64((st[17] ^ Di), 15);
      Bu = peach_rotl64((st[8] ^ Do), 56);
      Ba = peach_rotl64((st[24] ^ Du), 27);
      st[10] = Ba ^ ((~Be) & Bi);
      st[1] = Be ^ ((~Bi) & Bo);
      st[17] = Bi ^ ((~Bo) & Bu);
      st[8] = Bo ^ ((~Bu) & Ba);
      st[24] = Bu ^ ((~Ba) & Be);
      Bo = peach_rotl64((st[5] ^ Da), 41);
      Bu = peach_rotl64((st[21] ^ De), 2);
      Ba = peach_rotl64((st[12] ^ Di), 62);
      Be = peach_rotl64((st[3] ^ Do), 55);
      Bi = peach_rotl64((st[19] ^ Du), 39);
      st[5] = Ba ^ ((~Be) & Bi);
      st[21] = Be ^ ((~Bi) & Bo);
      st[12] = Bi ^ ((~Bo) & Bu);
      st[3] = Bo ^ ((~Bu) & Ba);
      st[19] = Bu ^ ((~Ba) & Be);

      /* round r + 2 */
      Ca = st[0] ^ st[20] ^ st[15] ^ st[10] ^ st[5];
      Ce = st[16] ^ st[11] ^ st[6] ^ st[1] ^ st[21];
      Ci = st[7] ^ st[2] ^ st[22] ^ st[17] ^ st[12];
      Co = st[23] ^ st[18] ^ st[13] ^ st[8] ^ st[3];
      Cu = st[14] ^ st[9] ^ st[4] ^ st[24] ^ st[19];
      Da = Cu ^ peach_rotl64(Ce, 1);
      De = Ca ^ peach_rotl64(Ci, 1);
      Di = Ce ^ peach_rotl64(Co, 1);
      Do = Ci ^ peach_rotl64(Cu, 1);
      Du = Co ^ peach_rotl64(Ca, 1);
      Ba = (st[0] ^ Da);
      Be = peach_rotl64((st[11] ^ De), 44);
      Bi = peach_rotl64((st[22] ^ Di), 43);
      Bo = peach_rotl64((st[8] ^ Do), 21);
      Bu = peach_rotl64((st[19] ^ Du), 14);
      st[0] = Ba ^ ((~Be) & Bi) ^ c_peach_keccakf_rndc[r + 2];
      st[11] = Be ^ ((~Bi) & Bo);
      st[22] = Bi ^ ((~Bo) & Bu);
      st[8] = Bo ^ ((~Bu) & Ba);
      st[19] = Bu ^ ((~Ba) & Be);
      Bi = peach_rotl64((st[15] ^ Da), 3);
      Bo = peach_rotl64((st[1] ^ De), 45);
      Bu = peach_rotl64((st[12] ^ Di), 61);
      Ba = peach_rotl64((st[23] ^ Do), 28);
      Be = peach_rotl64((st[9] ^ Du), 20);
      st[15] = Ba ^ ((~Be) & Bi);
      st[1] = Be ^ ((~Bi) & Bo);
      st[12] = Bi ^ ((~Bo) & Bu);
      st[23] = Bo ^ ((~Bu) & Ba);
      st[9] = Bu ^ ((~Ba) & Be);
      Bu = peach_rotl64((st[5] ^ Da), 18);
      Ba = peach_rotl64((st[16] ^ De), 1);
      Be = peach_rotl64((st[2] ^ Di), 6);
      Bi = peach_rotl64((st[13] ^ Do), 25);
      Bo = peach_rotl64((st[24] ^ Du), 8);
      st[5] = Ba ^ ((~Be) & Bi);
      st[16] = Be ^ ((~Bi) & Bo);
      st[2] = Bi ^ ((~Bo) & Bu);
      st[13] = Bo ^ ((~Bu) & Ba);
      st[24] = Bu ^ ((~Ba) & Be);
      Be = peach_rotl64((st[20] ^ Da), 36);
      Bi = peach_rotl64((st[6] ^ De), 10);
      Bo = peach_rotl64((st[17] ^ Di), 15);
      Bu = peach_rotl64((st[3] ^ Do), 56);
      Ba = peach_rotl64((st[14] ^ Du), 27);
      st[20] = Ba ^ ((~Be) & Bi);
      st[6] = Be ^ ((~Bi) & Bo);
      st[17] = Bi ^ ((~Bo) & Bu);
      st[3] = Bo ^ ((~Bu) & Ba);
      st[14] = Bu ^ ((~Ba) & Be);
      Bo = peach_rotl64((st[10] ^ Da), 41);
      Bu = peach_rotl64((st[21] ^ De), 2);
      Ba = peach_rotl64((st[7] ^ Di), 62);
      Be = peach_rotl64((st[18] ^ Do), 55);
      Bi = peach_rotl64((st[4] ^ Du), 39);
      st[10] = Ba ^ ((~Be) & Bi);
      st[21] = Be ^ ((~Bi) & Bo);
      st[7] = Bi ^ ((~Bo) & Bu);
      st[18] = Bo ^ ((~Bu) & Ba);
      st[4] = Bu ^ ((~Ba) & Be);

      /* round r + 3 */
      Ca = st[0] ^ st[15] ^ st[5] ^ st[20] ^ st[10];
      Ce = st[11] ^ st[1] ^ st[16] ^ st[6] ^ st[21];
      Ci = st[22] ^ st[12] ^ st[2] ^ st[17] ^ st[7];
      Co = st[8] ^ st[23] ^ st[13] ^ st[3] ^ st[18];
      Cu = st[19] ^ st[9] ^ st[24] ^ st[14] ^ st[4];
      Da = Cu ^ peach_rotl64(Ce, 1);
      De = Ca ^ peach_rotl64(Ci, 1);
      Di = Ce ^ peach_rotl64(Co, 1);
      Do = Ci ^ peach_rotl64(Cu, 1);
      Du = Co ^ peach_rotl64(Ca, 1);
      Ba = (st[0] ^ Da);
      Be = peach_rotl64((st[1] ^ De), 44);
      Bi = peach_rotl64((st[2] ^ Di), 43);
      Bo = peach_rotl64((st[3] ^ Do), 21);
      Bu = peach_rotl64((st[4] ^ Du), 14);
      st[0] = Ba ^ ((~Be) & Bi) ^ c_peach_keccakf_rndc[r + 3];
      st[1] = Be ^ ((~Bi) & Bo);
      st[2] = Bi ^ ((~Bo) & Bu);
      st[3] = Bo ^ ((~Bu) & Ba);
      st[4] = Bu ^ ((~Ba) & Be);
      Bi = peach_rotl64((st[5] ^ Da), 3);
      Bo = peach_rotl64((st[6] ^ De), 45);
      Bu = peach_rotl64((st[7] ^ Di), 61);
      Ba = peach_rotl64((st[8] ^ Do), 28);
      Be = peach_rotl64((st[9] ^ Du), 20);
      st[5] = Ba ^ ((~Be) & Bi);
      st[6] = Be ^ ((~Bi) & Bo);
      st[7] = Bi ^ ((~Bo) & Bu);
      st[8] = Bo ^ ((~Bu) & Ba);
      st[9] = Bu ^ ((~Ba) & Be);
      Bu = peach_rotl64((st[10] ^ Da), 18);
      Ba = peach_rotl64((st[11] ^ De), 1);
      Be = peach_rotl64((st[12] ^ Di), 6);
      Bi = peach_rotl64((st[13] ^ Do), 25);
      Bo = peach_rotl64((st[14] ^ Du), 8);
      st[10] = Ba ^ ((~Be) & Bi);
      st[11] = Be ^ ((~Bi) & Bo);
      st[12] = Bi ^ ((~Bo) & Bu);
      st[13] = Bo ^ ((~Bu) & Ba);
      st[14] = Bu ^ ((~Ba) & Be);
      Be = peach_rotl64((st[15] ^ Da), 36);
      Bi = peach_rotl64((st[16] ^ De), 10);
      Bo = peach_rotl64((st[17] ^ Di), 15);
      Bu = peach_rotl64((st[18] ^ Do), 56);
      Ba = peach_rotl64((st[19] ^ Du), 27);
      st[15] = Ba ^ ((~Be) & Bi);
      st[16] = Be ^ ((~Bi) & Bo);
      st[17] = Bi ^ ((~Bo) & Bu);
      st[18] = Bo ^ ((~Bu) & Ba);
      st[19] = Bu ^ ((~Ba) & Be);
      Bo = peach_rotl64((st[20] ^ Da), 41);
      Bu = peach_rotl64((st[21] ^ De), 2);
      Ba = peach_rotl64((st[22] ^ Di), 62);
      Be = peach_rotl64((st[23] ^ Do), 55);
      Bi = peach_rotl64((st[24] ^ Du), 39);
      st[20] = Ba ^ ((~Be) & Bi);
      st[21] = Be ^ ((~Bi) & Bo);
      st[22] = Bi ^ ((~Bo) & Bu);
      st[23] = Bo ^ ((~Bu) & Ba);
      st[24] = Bu ^ ((~Ba) & Be);
   }
}  /* end peach_keccakf() */

/**
 * @private
 * SHA3-256 / Keccak-256 of a Peach jump seed (seed view): rate 136 bytes
 * (17 lanes = 34 words), 7 full blocks + a final block of 108 bytes.
 * Loads (tile vectors, c = carried words, i = 0..2):
 * block 0 = n0..n7, index, tile[0..5], tile[6].x, c = tile[6].yzw;
 * block 2i+1 = c(3), tile[17i+7..17i+13], tile[17i+14].xyz,
 * c = tile[17i+14].w;
 * block 2i+2 = c(1), tile[17i+15..17i+22], tile[17i+23].x,
 * c = tile[17i+23].yzw;
 * block 7 = c(3), tile[58..63], then @a pad and the final 0x80 bit.
 * @param n Pointer to the nonce, 8 words
 * @param index Tile index m (seed word 8)
 * @param tile Pointer to tile m, 64 x uint4 (16-byte aligned)
 * @param pad Domain padding byte: 0x06 SHA3, 0x01 Keccak
 * @param out Pointer to 8 words for the digest (little-endian words)
*/
PEACH_DEV void peach_sh_keccak_pad(const word32 *n, word32 index,
   const uint4 *tile, word32 pad, word32 *out)
{
   word64 st[25];
   uint4 q0, q1, q2, q3, q4, q5, q6, q7, q8;
   const uint4 *tp;
   word32 c0, c1, c2;
   int k;

   st[0] = st[1] = st[2] = st[3] = st[4] = 0;
   st[5] = st[6] = st[7] = st[8] = st[9] = 0;
   st[10] = st[11] = st[12] = st[13] = st[14] = 0;
   st[15] = st[16] = st[17] = st[18] = st[19] = 0;
   st[20] = st[21] = st[22] = st[23] = st[24] = 0;
   c0 = c1 = c2 = 0;

   /* 7 full blocks + the final (108-byte) block, one permutation body */
   PEACH_H64_ROLLED
   for (k = 0; k < 8; k++) {
      if (k == 0) {
         /* seed words 0..33: nonce, index, tile words 0..24 */
         q0 = PEACH_LDG128(&tile[0]);
         q1 = PEACH_LDG128(&tile[1]);
         q2 = PEACH_LDG128(&tile[2]);
         q3 = PEACH_LDG128(&tile[3]);
         q4 = PEACH_LDG128(&tile[4]);
         q5 = PEACH_LDG128(&tile[5]);
         q6 = PEACH_LDG128(&tile[6]);
         st[0] ^= PEACH_H64_LANE(n[0], n[1]);
         st[1] ^= PEACH_H64_LANE(n[2], n[3]);
         st[2] ^= PEACH_H64_LANE(n[4], n[5]);
         st[3] ^= PEACH_H64_LANE(n[6], n[7]);
         st[4] ^= PEACH_H64_LANE(index, q0.x);
         st[5] ^= PEACH_H64_LANE(q0.y, q0.z);
         st[6] ^= PEACH_H64_LANE(q0.w, q1.x);
         st[7] ^= PEACH_H64_LANE(q1.y, q1.z);
         st[8] ^= PEACH_H64_LANE(q1.w, q2.x);
         st[9] ^= PEACH_H64_LANE(q2.y, q2.z);
         st[10] ^= PEACH_H64_LANE(q2.w, q3.x);
         st[11] ^= PEACH_H64_LANE(q3.y, q3.z);
         st[12] ^= PEACH_H64_LANE(q3.w, q4.x);
         st[13] ^= PEACH_H64_LANE(q4.y, q4.z);
         st[14] ^= PEACH_H64_LANE(q4.w, q5.x);
         st[15] ^= PEACH_H64_LANE(q5.y, q5.z);
         st[16] ^= PEACH_H64_LANE(q5.w, q6.x);
         c0 = q6.y; c1 = q6.z; c2 = q6.w;
      } else if (k == 7) {
         /* seed words 238..264: tile words 229..255, then padding */
         q0 = PEACH_LDG128(&tile[58]);
         q1 = PEACH_LDG128(&tile[59]);
         q2 = PEACH_LDG128(&tile[60]);
         q3 = PEACH_LDG128(&tile[61]);
         q4 = PEACH_LDG128(&tile[62]);
         q5 = PEACH_LDG128(&tile[63]);
         st[0] ^= PEACH_H64_LANE(c0, c1);
         st[1] ^= PEACH_H64_LANE(c2, q0.x);
         st[2] ^= PEACH_H64_LANE(q0.y, q0.z);
         st[3] ^= PEACH_H64_LANE(q0.w, q1.x);
         st[4] ^= PEACH_H64_LANE(q1.y, q1.z);
         st[5] ^= PEACH_H64_LANE(q1.w, q2.x);
         st[6] ^= PEACH_H64_LANE(q2.y, q2.z);
         st[7] ^= PEACH_H64_LANE(q2.w, q3.x);
         st[8] ^= PEACH_H64_LANE(q3.y, q3.z);
         st[9] ^= PEACH_H64_LANE(q3.w, q4.x);
         st[10] ^= PEACH_H64_LANE(q4.y, q4.z);
         st[11] ^= PEACH_H64_LANE(q4.w, q5.x);
         st[12] ^= PEACH_H64_LANE(q5.y, q5.z);
         /* pad byte at block byte 108, 0x80 at block byte 135 */
         st[13] ^= PEACH_H64_LANE(q5.w, pad);
         st[16] ^= WORD64_C(0x8000000000000000);
      } else if (k & 1) {
         /* seed words 34k..34k+33: tile words 34k-9..34k+24 */
         tp = &tile[(17 * (k >> 1)) + 7];
         q0 = PEACH_LDG128(&tp[0]);
         q1 = PEACH_LDG128(&tp[1]);
         q2 = PEACH_LDG128(&tp[2]);
         q3 = PEACH_LDG128(&tp[3]);
         q4 = PEACH_LDG128(&tp[4]);
         q5 = PEACH_LDG128(&tp[5]);
         q6 = PEACH_LDG128(&tp[6]);
         q7 = PEACH_LDG128(&tp[7]);
         st[0] ^= PEACH_H64_LANE(c0, c1);
         st[1] ^= PEACH_H64_LANE(c2, q0.x);
         st[2] ^= PEACH_H64_LANE(q0.y, q0.z);
         st[3] ^= PEACH_H64_LANE(q0.w, q1.x);
         st[4] ^= PEACH_H64_LANE(q1.y, q1.z);
         st[5] ^= PEACH_H64_LANE(q1.w, q2.x);
         st[6] ^= PEACH_H64_LANE(q2.y, q2.z);
         st[7] ^= PEACH_H64_LANE(q2.w, q3.x);
         st[8] ^= PEACH_H64_LANE(q3.y, q3.z);
         st[9] ^= PEACH_H64_LANE(q3.w, q4.x);
         st[10] ^= PEACH_H64_LANE(q4.y, q4.z);
         st[11] ^= PEACH_H64_LANE(q4.w, q5.x);
         st[12] ^= PEACH_H64_LANE(q5.y, q5.z);
         st[13] ^= PEACH_H64_LANE(q5.w, q6.x);
         st[14] ^= PEACH_H64_LANE(q6.y, q6.z);
         st[15] ^= PEACH_H64_LANE(q6.w, q7.x);
         st[16] ^= PEACH_H64_LANE(q7.y, q7.z);
         c0 = q7.w;
      } else {
         /* seed words 34k..34k+33: tile words 34k-9..34k+24 */
         tp = &tile[(17 * (k >> 1)) - 2];
         q0 = PEACH_LDG128(&tp[0]);
         q1 = PEACH_LDG128(&tp[1]);
         q2 = PEACH_LDG128(&tp[2]);
         q3 = PEACH_LDG128(&tp[3]);
         q4 = PEACH_LDG128(&tp[4]);
         q5 = PEACH_LDG128(&tp[5]);
         q6 = PEACH_LDG128(&tp[6]);
         q7 = PEACH_LDG128(&tp[7]);
         q8 = PEACH_LDG128(&tp[8]);
         st[0] ^= PEACH_H64_LANE(c0, q0.x);
         st[1] ^= PEACH_H64_LANE(q0.y, q0.z);
         st[2] ^= PEACH_H64_LANE(q0.w, q1.x);
         st[3] ^= PEACH_H64_LANE(q1.y, q1.z);
         st[4] ^= PEACH_H64_LANE(q1.w, q2.x);
         st[5] ^= PEACH_H64_LANE(q2.y, q2.z);
         st[6] ^= PEACH_H64_LANE(q2.w, q3.x);
         st[7] ^= PEACH_H64_LANE(q3.y, q3.z);
         st[8] ^= PEACH_H64_LANE(q3.w, q4.x);
         st[9] ^= PEACH_H64_LANE(q4.y, q4.z);
         st[10] ^= PEACH_H64_LANE(q4.w, q5.x);
         st[11] ^= PEACH_H64_LANE(q5.y, q5.z);
         st[12] ^= PEACH_H64_LANE(q5.w, q6.x);
         st[13] ^= PEACH_H64_LANE(q6.y, q6.z);
         st[14] ^= PEACH_H64_LANE(q6.w, q7.x);
         st[15] ^= PEACH_H64_LANE(q7.y, q7.z);
         st[16] ^= PEACH_H64_LANE(q7.w, q8.x);
         c0 = q8.y; c1 = q8.z; c2 = q8.w;
      }
      peach_keccakf(st);
   }

   /* 256-bit digest = lanes 0..3 (little-endian) */
   out[0] = (word32) st[0]; out[1] = (word32) (st[0] >> 32);
   out[2] = (word32) st[1]; out[3] = (word32) (st[1] >> 32);
   out[4] = (word32) st[2]; out[5] = (word32) (st[2] >> 32);
   out[6] = (word32) st[3]; out[7] = (word32) (st[3] >> 32);
}  /* end peach_sh_keccak_pad() */

/**
 * Peach jump hash, algo 0: Blake2b-256 keyed with 32 zero bytes.
 * @param n Pointer to the nonce, 8 words
 * @param m Tile index
 * @param tile Pointer to tile m, 64 x uint4 (16-byte aligned)
 * @param out Pointer to 8 words for the digest (little-endian words)
*/
PEACH_DEV void peach_sh_blake2b32(const word32 *n, word32 m,
   const uint4 *tile, word32 *out)
{
   peach_sh_blake2b(n, m, tile, 0, out);
}  /* end peach_sh_blake2b32() */

/**
 * Peach jump hash, algo 1: Blake2b-256 keyed with 64 bytes of 0x01.
 * @param n Pointer to the nonce, 8 words
 * @param m Tile index
 * @param tile Pointer to tile m, 64 x uint4 (16-byte aligned)
 * @param out Pointer to 8 words for the digest (little-endian words)
*/
PEACH_DEV void peach_sh_blake2b64(const word32 *n, word32 m,
   const uint4 *tile, word32 *out)
{
   peach_sh_blake2b(n, m, tile, 1, out);
}  /* end peach_sh_blake2b64() */

/**
 * Peach jump hash, algo 4: SHA3-256 (padding 0x06).
 * @param n Pointer to the nonce, 8 words
 * @param m Tile index
 * @param tile Pointer to tile m, 64 x uint4 (16-byte aligned)
 * @param out Pointer to 8 words for the digest (little-endian words)
*/
PEACH_DEV void peach_sh_sha3(const word32 *n, word32 m,
   const uint4 *tile, word32 *out)
{
   peach_sh_keccak_pad(n, m, tile, 0x06, out);
}  /* end peach_sh_sha3() */

/**
 * Peach jump hash, algo 5: Keccak-256 (padding 0x01).
 * @param n Pointer to the nonce, 8 words
 * @param m Tile index
 * @param tile Pointer to tile m, 64 x uint4 (16-byte aligned)
 * @param out Pointer to 8 words for the digest (little-endian words)
*/
PEACH_DEV void peach_sh_keccak(const word32 *n, word32 m,
   const uint4 *tile, word32 *out)
{
   peach_sh_keccak_pad(n, m, tile, 0x01, out);
}  /* end peach_sh_keccak() */

/* end include guard */
#endif
