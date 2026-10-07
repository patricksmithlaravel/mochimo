/**
 * @file peach_hash32.cuh
 * @brief Peach seed-view hashes of the 32-bit word family: SHA-1,
 * SHA-256 and MD5 (device), plus the SHA-256 of the block trailer and
 * the final SHA-256 of a Peach solve.
 * @details The jump seed of tile `m` is the 1060 byte "view"
 * `nonce (8 words) || m (1 word) || tile[m] (256 words)`. It is hashed
 * straight from registers and the Peach map, without a local seed copy:
 * the tile (`const uint4 *`, 64 x 16 bytes, 16-byte aligned) is read with
 * aligned 128-bit loads, each byte exactly once, and one word is carried
 * from block to block because the tile starts at seed word 9:
 * - block 0 = n0..n7, m, t[0].xyzw, t[1].xyz; carry t[1].w;
 * - block k = 1..15 = carry, t[4k-2], t[4k-1], t[4k], t[4k+1].xyz;
 *   carry t[4k+1].w;
 * - block 16 = carry, t[62], t[63] (36 bytes), then the padding of a
 *   1060 byte message (8480 bits).
 * <br />
 * Outputs are the 8 words peach_nighthash() writes on a little-endian
 * host, including the zero fill of the 20 byte SHA-1 (words 5..7) and the
 * 16 byte MD5 (words 4..7) digests; word i holds digest bytes 4i..4i+3.
 * <br />
 * Kernel shape: the round macros use literal constants, named state
 * registers and a 16-word message window indexed by constants only (no
 * tables, no dynamically indexed local arrays), so after inlining every
 * function compiles to a 0 byte stack frame. The 15 middle blocks run as
 * a non-unrolled loop on the device (instruction cache).
 * <br />
 * Compiles as CUDA C++ under nvcc and as plain C under gcc after
 * test/_cuda_emu.h (see peach_compat.cuh).
 * @copyright Adequate Systems LLC, 2018-2025. All Rights Reserved.
 * <br />For license information, please refer to ../LICENSE.md
*/

/* include guard */
#ifndef MOCHIMO_PEACH_HASH32_CUH
#define MOCHIMO_PEACH_HASH32_CUH


#include "extint.h"           /* for word types */
#include "peach.h"            /* for PEACHJUMPLEN, PEACHTILELEN */
#include "peach_compat.cuh"   /* for PEACH_DEV, PEACH_LDG128, rotates */

/* loop that must not be unrolled on the device (instruction cache) */
#ifdef __CUDA_ARCH__
   #define PEACH_HASH32_NOUNROLL  _Pragma("unroll 1")
#else
   #define PEACH_HASH32_NOUNROLL
#endif

/* message lengths in bits: jump seed, trailer (bt[0..123]), final */
#define PEACH_HASH32_SEEDBITS    ((word32) PEACHJUMPLEN * 8)
#define PEACH_HASH32_TRAILBITS   ((word32) 124 * 8)
#define PEACH_HASH32_FINALBITS   ((word32) (32 + PEACHTILELEN) * 8)

/****************************************************************
 * SHA-256 compression (FIPS 180-4), big-endian message words
 ****************************************************************/

#define PEACH_SHA256_BSIG0(x) \
   (peach_rotr32((x), 2) ^ peach_rotr32((x), 13) ^ peach_rotr32((x), 22))
#define PEACH_SHA256_BSIG1(x) \
   (peach_rotr32((x), 6) ^ peach_rotr32((x), 11) ^ peach_rotr32((x), 25))
#define PEACH_SHA256_SSIG0(x) \
   (peach_rotr32((x), 7) ^ peach_rotr32((x), 18) ^ ((x) >> 3))
#define PEACH_SHA256_SSIG1(x) \
   (peach_rotr32((x), 17) ^ peach_rotr32((x), 19) ^ ((x) >> 10))
#define PEACH_SHA256_CH(x, y, z)    ((z) ^ ((x) & ((y) ^ (z))))
#define PEACH_SHA256_MAJ(x, y, z)   (((x) & (y)) | ((z) & ((x) | (y))))

/* message word of round i: window word (rounds 0..15), or the expanded
 * schedule word, updated in place in the 16-word window w[] (16..63) */
#define PEACH_SHA256_W0(i)    (w[(i) & 15])
#define PEACH_SHA256_WS(i) \
   (w[(i) & 15] += PEACH_SHA256_SSIG1(w[((i) - 2) & 15]) + \
      w[((i) - 7) & 15] + PEACH_SHA256_SSIG0(w[((i) - 15) & 15]))

/* one round; k is a literal hex constant, wm selects the message word */
#define PEACH_SHA256_RND(a, b, c, d, e, f, g, h, i, k, wm) \
   h += PEACH_SHA256_BSIG1(e) + PEACH_SHA256_CH(e, f, g) + \
      WORD32_C(k) + wm(i); \
   d += h; h += PEACH_SHA256_BSIG0(a) + PEACH_SHA256_MAJ(a, b, c)

/* eight rounds i..i+7 (state rotates back to a..h) */
#define PEACH_SHA256_R8(i, k0, k1, k2, k3, k4, k5, k6, k7, wm) \
   PEACH_SHA256_RND(a, b, c, d, e, f, g, h, (i), k0, wm); \
   PEACH_SHA256_RND(h, a, b, c, d, e, f, g, (i) + 1, k1, wm); \
   PEACH_SHA256_RND(g, h, a, b, c, d, e, f, (i) + 2, k2, wm); \
   PEACH_SHA256_RND(f, g, h, a, b, c, d, e, (i) + 3, k3, wm); \
   PEACH_SHA256_RND(e, f, g, h, a, b, c, d, (i) + 4, k4, wm); \
   PEACH_SHA256_RND(d, e, f, g, h, a, b, c, (i) + 5, k5, wm); \
   PEACH_SHA256_RND(c, d, e, f, g, h, a, b, (i) + 6, k6, wm); \
   PEACH_SHA256_RND(b, c, d, e, f, g, h, a, (i) + 7, k7, wm)

/**
 * @private
 * SHA-256 compression of one 64 byte block.
 * @param st Chaining state (8 words), updated in place
 * @param w Message block as 16 big-endian words; overwritten (schedule)
*/
PEACH_DEV void peach_sha256_compress(word32 *st, word32 *w)
{
   word32 a, b, c, d, e, f, g, h;

   a = st[0]; b = st[1]; c = st[2]; d = st[3];
   e = st[4]; f = st[5]; g = st[6]; h = st[7];
   PEACH_SHA256_R8(0, 0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
      0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, PEACH_SHA256_W0);
   PEACH_SHA256_R8(8, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
      0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, PEACH_SHA256_W0);
   PEACH_SHA256_R8(16, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
      0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, PEACH_SHA256_WS);
   PEACH_SHA256_R8(24, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
      0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, PEACH_SHA256_WS);
   PEACH_SHA256_R8(32, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
      0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, PEACH_SHA256_WS);
   PEACH_SHA256_R8(40, 0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
      0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, PEACH_SHA256_WS);
   PEACH_SHA256_R8(48, 0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
      0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3, PEACH_SHA256_WS);
   PEACH_SHA256_R8(56, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
      0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2, PEACH_SHA256_WS);
   st[0] += a; st[1] += b; st[2] += c; st[3] += d;
   st[4] += e; st[5] += f; st[6] += g; st[7] += h;
}  /* end peach_sha256_compress() */

/**
 * @private
 * Set the SHA-256 initial hash value.
*/
PEACH_DEV void peach_sha256_iv(word32 *st)
{
   st[0] = WORD32_C(0x6a09e667); st[1] = WORD32_C(0xbb67ae85);
   st[2] = WORD32_C(0x3c6ef372); st[3] = WORD32_C(0xa54ff53a);
   st[4] = WORD32_C(0x510e527f); st[5] = WORD32_C(0x9b05688c);
   st[6] = WORD32_C(0x1f83d9ab); st[7] = WORD32_C(0x5be0cd19);
}  /* end peach_sha256_iv() */

/****************************************************************
 * SHA-1 compression (FIPS 180-4), big-endian message words
 ****************************************************************/

#define PEACH_SHA1_CH(x, y, z)    ((z) ^ ((x) & ((y) ^ (z))))
#define PEACH_SHA1_PAR(x, y, z)   ((x) ^ (y) ^ (z))
#define PEACH_SHA1_MAJ(x, y, z)   (((x) & (y)) | ((z) & ((x) | (y))))

/* message word of round i: window word (rounds 0..15), or the expanded
 * schedule word, updated in place in the 16-word window w[] (16..79) */
#define PEACH_SHA1_W0(i)   (w[(i) & 15])
#define PEACH_SHA1_WS(i) \
   (w[(i) & 15] = peach_rotl32(w[((i) + 13) & 15] ^ w[((i) + 8) & 15] ^ \
      w[((i) + 2) & 15] ^ w[(i) & 15], 1))

/* one round; fn is the round function, k a literal hex constant */
#define PEACH_SHA1_RND(a, b, c, d, e, i, fn, k, wm) \
   e += peach_rotl32(a, 5) + fn(b, c, d) + WORD32_C(k) + wm(i); \
   b = peach_rotl32(b, 30)

/* five rounds i..i+4 (state rotates back to a..e) */
#define PEACH_SHA1_R5(i, fn, k, wm) \
   PEACH_SHA1_RND(a, b, c, d, e, (i), fn, k, wm); \
   PEACH_SHA1_RND(e, a, b, c, d, (i) + 1, fn, k, wm); \
   PEACH_SHA1_RND(d, e, a, b, c, (i) + 2, fn, k, wm); \
   PEACH_SHA1_RND(c, d, e, a, b, (i) + 3, fn, k, wm); \
   PEACH_SHA1_RND(b, c, d, e, a, (i) + 4, fn, k, wm)

/**
 * @private
 * SHA-1 compression of one 64 byte block.
 * @param st Chaining state (5 words), updated in place
 * @param w Message block as 16 big-endian words; overwritten (schedule)
*/
PEACH_DEV void peach_sha1_compress(word32 *st, word32 *w)
{
   word32 a, b, c, d, e;

   a = st[0]; b = st[1]; c = st[2]; d = st[3]; e = st[4];
   /* rounds 0..19 (round 15 uses the last window word) */
   PEACH_SHA1_R5(0, PEACH_SHA1_CH, 0x5a827999, PEACH_SHA1_W0);
   PEACH_SHA1_R5(5, PEACH_SHA1_CH, 0x5a827999, PEACH_SHA1_W0);
   PEACH_SHA1_R5(10, PEACH_SHA1_CH, 0x5a827999, PEACH_SHA1_W0);
   PEACH_SHA1_RND(a, b, c, d, e, 15, PEACH_SHA1_CH, 0x5a827999,
      PEACH_SHA1_W0);
   PEACH_SHA1_RND(e, a, b, c, d, 16, PEACH_SHA1_CH, 0x5a827999,
      PEACH_SHA1_WS);
   PEACH_SHA1_RND(d, e, a, b, c, 17, PEACH_SHA1_CH, 0x5a827999,
      PEACH_SHA1_WS);
   PEACH_SHA1_RND(c, d, e, a, b, 18, PEACH_SHA1_CH, 0x5a827999,
      PEACH_SHA1_WS);
   PEACH_SHA1_RND(b, c, d, e, a, 19, PEACH_SHA1_CH, 0x5a827999,
      PEACH_SHA1_WS);
   /* rounds 20..39 */
   PEACH_SHA1_R5(20, PEACH_SHA1_PAR, 0x6ed9eba1, PEACH_SHA1_WS);
   PEACH_SHA1_R5(25, PEACH_SHA1_PAR, 0x6ed9eba1, PEACH_SHA1_WS);
   PEACH_SHA1_R5(30, PEACH_SHA1_PAR, 0x6ed9eba1, PEACH_SHA1_WS);
   PEACH_SHA1_R5(35, PEACH_SHA1_PAR, 0x6ed9eba1, PEACH_SHA1_WS);
   /* rounds 40..59 */
   PEACH_SHA1_R5(40, PEACH_SHA1_MAJ, 0x8f1bbcdc, PEACH_SHA1_WS);
   PEACH_SHA1_R5(45, PEACH_SHA1_MAJ, 0x8f1bbcdc, PEACH_SHA1_WS);
   PEACH_SHA1_R5(50, PEACH_SHA1_MAJ, 0x8f1bbcdc, PEACH_SHA1_WS);
   PEACH_SHA1_R5(55, PEACH_SHA1_MAJ, 0x8f1bbcdc, PEACH_SHA1_WS);
   /* rounds 60..79 */
   PEACH_SHA1_R5(60, PEACH_SHA1_PAR, 0xca62c1d6, PEACH_SHA1_WS);
   PEACH_SHA1_R5(65, PEACH_SHA1_PAR, 0xca62c1d6, PEACH_SHA1_WS);
   PEACH_SHA1_R5(70, PEACH_SHA1_PAR, 0xca62c1d6, PEACH_SHA1_WS);
   PEACH_SHA1_R5(75, PEACH_SHA1_PAR, 0xca62c1d6, PEACH_SHA1_WS);
   st[0] += a; st[1] += b; st[2] += c; st[3] += d; st[4] += e;
}  /* end peach_sha1_compress() */

/****************************************************************
 * MD5 compression (RFC 1321), little-endian message words
 ****************************************************************/

#define PEACH_MD5_F(x, y, z)   ((z) ^ ((x) & ((y) ^ (z))))
#define PEACH_MD5_G(x, y, z)   ((y) ^ ((z) & ((x) ^ (y))))
#define PEACH_MD5_H(x, y, z)   ((x) ^ (y) ^ (z))
#define PEACH_MD5_I(x, y, z)   ((y) ^ ((x) | ~(z)))

/* one step; fn is the round function, i the message word, s the
 * rotation, t a literal hex constant */
#define PEACH_MD5_STEP(fn, a, b, c, d, i, s, t) \
   a += fn(b, c, d) + w[i] + WORD32_C(t); \
   a = b + peach_rotl32(a, s)

/**
 * @private
 * MD5 compression of one 64 byte block.
 * @param st Chaining state (4 words), updated in place
 * @param w Message block as 16 little-endian words (unchanged)
*/
PEACH_DEV void peach_md5_compress(word32 *st, const word32 *w)
{
   word32 a, b, c, d;

   a = st[0]; b = st[1]; c = st[2]; d = st[3];
   /* round 1 */
   PEACH_MD5_STEP(PEACH_MD5_F, a, b, c, d, 0, 7, 0xd76aa478);
   PEACH_MD5_STEP(PEACH_MD5_F, d, a, b, c, 1, 12, 0xe8c7b756);
   PEACH_MD5_STEP(PEACH_MD5_F, c, d, a, b, 2, 17, 0x242070db);
   PEACH_MD5_STEP(PEACH_MD5_F, b, c, d, a, 3, 22, 0xc1bdceee);
   PEACH_MD5_STEP(PEACH_MD5_F, a, b, c, d, 4, 7, 0xf57c0faf);
   PEACH_MD5_STEP(PEACH_MD5_F, d, a, b, c, 5, 12, 0x4787c62a);
   PEACH_MD5_STEP(PEACH_MD5_F, c, d, a, b, 6, 17, 0xa8304613);
   PEACH_MD5_STEP(PEACH_MD5_F, b, c, d, a, 7, 22, 0xfd469501);
   PEACH_MD5_STEP(PEACH_MD5_F, a, b, c, d, 8, 7, 0x698098d8);
   PEACH_MD5_STEP(PEACH_MD5_F, d, a, b, c, 9, 12, 0x8b44f7af);
   PEACH_MD5_STEP(PEACH_MD5_F, c, d, a, b, 10, 17, 0xffff5bb1);
   PEACH_MD5_STEP(PEACH_MD5_F, b, c, d, a, 11, 22, 0x895cd7be);
   PEACH_MD5_STEP(PEACH_MD5_F, a, b, c, d, 12, 7, 0x6b901122);
   PEACH_MD5_STEP(PEACH_MD5_F, d, a, b, c, 13, 12, 0xfd987193);
   PEACH_MD5_STEP(PEACH_MD5_F, c, d, a, b, 14, 17, 0xa679438e);
   PEACH_MD5_STEP(PEACH_MD5_F, b, c, d, a, 15, 22, 0x49b40821);
   /* round 2 */
   PEACH_MD5_STEP(PEACH_MD5_G, a, b, c, d, 1, 5, 0xf61e2562);
   PEACH_MD5_STEP(PEACH_MD5_G, d, a, b, c, 6, 9, 0xc040b340);
   PEACH_MD5_STEP(PEACH_MD5_G, c, d, a, b, 11, 14, 0x265e5a51);
   PEACH_MD5_STEP(PEACH_MD5_G, b, c, d, a, 0, 20, 0xe9b6c7aa);
   PEACH_MD5_STEP(PEACH_MD5_G, a, b, c, d, 5, 5, 0xd62f105d);
   PEACH_MD5_STEP(PEACH_MD5_G, d, a, b, c, 10, 9, 0x02441453);
   PEACH_MD5_STEP(PEACH_MD5_G, c, d, a, b, 15, 14, 0xd8a1e681);
   PEACH_MD5_STEP(PEACH_MD5_G, b, c, d, a, 4, 20, 0xe7d3fbc8);
   PEACH_MD5_STEP(PEACH_MD5_G, a, b, c, d, 9, 5, 0x21e1cde6);
   PEACH_MD5_STEP(PEACH_MD5_G, d, a, b, c, 14, 9, 0xc33707d6);
   PEACH_MD5_STEP(PEACH_MD5_G, c, d, a, b, 3, 14, 0xf4d50d87);
   PEACH_MD5_STEP(PEACH_MD5_G, b, c, d, a, 8, 20, 0x455a14ed);
   PEACH_MD5_STEP(PEACH_MD5_G, a, b, c, d, 13, 5, 0xa9e3e905);
   PEACH_MD5_STEP(PEACH_MD5_G, d, a, b, c, 2, 9, 0xfcefa3f8);
   PEACH_MD5_STEP(PEACH_MD5_G, c, d, a, b, 7, 14, 0x676f02d9);
   PEACH_MD5_STEP(PEACH_MD5_G, b, c, d, a, 12, 20, 0x8d2a4c8a);
   /* round 3 */
   PEACH_MD5_STEP(PEACH_MD5_H, a, b, c, d, 5, 4, 0xfffa3942);
   PEACH_MD5_STEP(PEACH_MD5_H, d, a, b, c, 8, 11, 0x8771f681);
   PEACH_MD5_STEP(PEACH_MD5_H, c, d, a, b, 11, 16, 0x6d9d6122);
   PEACH_MD5_STEP(PEACH_MD5_H, b, c, d, a, 14, 23, 0xfde5380c);
   PEACH_MD5_STEP(PEACH_MD5_H, a, b, c, d, 1, 4, 0xa4beea44);
   PEACH_MD5_STEP(PEACH_MD5_H, d, a, b, c, 4, 11, 0x4bdecfa9);
   PEACH_MD5_STEP(PEACH_MD5_H, c, d, a, b, 7, 16, 0xf6bb4b60);
   PEACH_MD5_STEP(PEACH_MD5_H, b, c, d, a, 10, 23, 0xbebfbc70);
   PEACH_MD5_STEP(PEACH_MD5_H, a, b, c, d, 13, 4, 0x289b7ec6);
   PEACH_MD5_STEP(PEACH_MD5_H, d, a, b, c, 0, 11, 0xeaa127fa);
   PEACH_MD5_STEP(PEACH_MD5_H, c, d, a, b, 3, 16, 0xd4ef3085);
   PEACH_MD5_STEP(PEACH_MD5_H, b, c, d, a, 6, 23, 0x04881d05);
   PEACH_MD5_STEP(PEACH_MD5_H, a, b, c, d, 9, 4, 0xd9d4d039);
   PEACH_MD5_STEP(PEACH_MD5_H, d, a, b, c, 12, 11, 0xe6db99e5);
   PEACH_MD5_STEP(PEACH_MD5_H, c, d, a, b, 15, 16, 0x1fa27cf8);
   PEACH_MD5_STEP(PEACH_MD5_H, b, c, d, a, 2, 23, 0xc4ac5665);
   /* round 4 */
   PEACH_MD5_STEP(PEACH_MD5_I, a, b, c, d, 0, 6, 0xf4292244);
   PEACH_MD5_STEP(PEACH_MD5_I, d, a, b, c, 7, 10, 0x432aff97);
   PEACH_MD5_STEP(PEACH_MD5_I, c, d, a, b, 14, 15, 0xab9423a7);
   PEACH_MD5_STEP(PEACH_MD5_I, b, c, d, a, 5, 21, 0xfc93a039);
   PEACH_MD5_STEP(PEACH_MD5_I, a, b, c, d, 12, 6, 0x655b59c3);
   PEACH_MD5_STEP(PEACH_MD5_I, d, a, b, c, 3, 10, 0x8f0ccc92);
   PEACH_MD5_STEP(PEACH_MD5_I, c, d, a, b, 10, 15, 0xffeff47d);
   PEACH_MD5_STEP(PEACH_MD5_I, b, c, d, a, 1, 21, 0x85845dd1);
   PEACH_MD5_STEP(PEACH_MD5_I, a, b, c, d, 8, 6, 0x6fa87e4f);
   PEACH_MD5_STEP(PEACH_MD5_I, d, a, b, c, 15, 10, 0xfe2ce6e0);
   PEACH_MD5_STEP(PEACH_MD5_I, c, d, a, b, 6, 15, 0xa3014314);
   PEACH_MD5_STEP(PEACH_MD5_I, b, c, d, a, 13, 21, 0x4e0811a1);
   PEACH_MD5_STEP(PEACH_MD5_I, a, b, c, d, 4, 6, 0xf7537e82);
   PEACH_MD5_STEP(PEACH_MD5_I, d, a, b, c, 11, 10, 0xbd3af235);
   PEACH_MD5_STEP(PEACH_MD5_I, c, d, a, b, 2, 15, 0x2ad7d2bb);
   PEACH_MD5_STEP(PEACH_MD5_I, b, c, d, a, 9, 21, 0xeb86d391);
   st[0] += a; st[1] += b; st[2] += c; st[3] += d;
}  /* end peach_md5_compress() */

/****************************************************************
 * Seed view loads (shared by SHA-1, SHA-256 and MD5)
 ****************************************************************/

/**
 * @private
 * A seed word in message byte order: byte swapped when @a be is non-zero
 * (big-endian SHA-1/SHA-256), else as loaded (little-endian MD5).
 * @a be is a literal at every call site (folded after inlining).
*/
PEACH_DEV word32 peach_hash32_order(word32 x, int be)
{
   return be ? peach_bswap32(x) : x;
}  /* end peach_hash32_order() */

/**
 * @private
 * Load seed-view block 0: n0..n7, m, t[0].xyzw, t[1].xyz.
 * @param w Message block (16 words, in message byte order)
 * @param n Nonce (8 words)
 * @param m Tile index (seed word 8)
 * @param tile Tile m (64 x uint4)
 * @param be Non-zero for big-endian message words
 * @returns the carried word t[1].w, as loaded
*/
PEACH_DEV word32 peach_hash32_view0(word32 *w, const word32 *n, word32 m,
   const uint4 *tile, int be)
{
   uint4 v0, v1;

   v0 = PEACH_LDG128(&tile[0]);
   v1 = PEACH_LDG128(&tile[1]);
   w[0] = peach_hash32_order(n[0], be);
   w[1] = peach_hash32_order(n[1], be);
   w[2] = peach_hash32_order(n[2], be);
   w[3] = peach_hash32_order(n[3], be);
   w[4] = peach_hash32_order(n[4], be);
   w[5] = peach_hash32_order(n[5], be);
   w[6] = peach_hash32_order(n[6], be);
   w[7] = peach_hash32_order(n[7], be);
   w[8] = peach_hash32_order(m, be);
   w[9] = peach_hash32_order(v0.x, be);
   w[10] = peach_hash32_order(v0.y, be);
   w[11] = peach_hash32_order(v0.z, be);
   w[12] = peach_hash32_order(v0.w, be);
   w[13] = peach_hash32_order(v1.x, be);
   w[14] = peach_hash32_order(v1.y, be);
   w[15] = peach_hash32_order(v1.z, be);

   return v1.w;
}  /* end peach_hash32_view0() */

/**
 * @private
 * Load seed-view block k = 1..15: carry, t[4k-2], t[4k-1], t[4k],
 * t[4k+1].xyz.
 * @param w Message block (16 words, in message byte order)
 * @param carry Word carried from block k - 1 (t[4k-3].w), as loaded
 * @param tile Tile (64 x uint4)
 * @param k Block number, 1..15
 * @param be Non-zero for big-endian message words
 * @returns the carried word t[4k+1].w, as loaded
*/
PEACH_DEV word32 peach_hash32_viewk(word32 *w, word32 carry,
   const uint4 *tile, int k, int be)
{
   const uint4 *t = &tile[(4 * k) - 2];
   uint4 v0, v1, v2, v3;

   v0 = PEACH_LDG128(&t[0]);
   v1 = PEACH_LDG128(&t[1]);
   v2 = PEACH_LDG128(&t[2]);
   v3 = PEACH_LDG128(&t[3]);
   w[0] = peach_hash32_order(carry, be);
   w[1] = peach_hash32_order(v0.x, be);
   w[2] = peach_hash32_order(v0.y, be);
   w[3] = peach_hash32_order(v0.z, be);
   w[4] = peach_hash32_order(v0.w, be);
   w[5] = peach_hash32_order(v1.x, be);
   w[6] = peach_hash32_order(v1.y, be);
   w[7] = peach_hash32_order(v1.z, be);
   w[8] = peach_hash32_order(v1.w, be);
   w[9] = peach_hash32_order(v2.x, be);
   w[10] = peach_hash32_order(v2.y, be);
   w[11] = peach_hash32_order(v2.z, be);
   w[12] = peach_hash32_order(v2.w, be);
   w[13] = peach_hash32_order(v3.x, be);
   w[14] = peach_hash32_order(v3.y, be);
   w[15] = peach_hash32_order(v3.z, be);

   return v3.w;
}  /* end peach_hash32_viewk() */

/**
 * @private
 * Load seed-view block 16 (last): carry, t[62], t[63] (36 bytes), then
 * the padding of the 1060 byte seed: 0x80, zeros and the 64-bit message
 * length in bits (big-endian for SHA, little-endian for MD5).
 * @param w Message block (16 words, in message byte order)
 * @param carry Word carried from block 15 (t[61].w), as loaded
 * @param tile Tile (64 x uint4)
 * @param be Non-zero for big-endian message words
*/
PEACH_DEV void peach_hash32_view16(word32 *w, word32 carry,
   const uint4 *tile, int be)
{
   uint4 v0, v1;

   v0 = PEACH_LDG128(&tile[62]);
   v1 = PEACH_LDG128(&tile[63]);
   w[0] = peach_hash32_order(carry, be);
   w[1] = peach_hash32_order(v0.x, be);
   w[2] = peach_hash32_order(v0.y, be);
   w[3] = peach_hash32_order(v0.z, be);
   w[4] = peach_hash32_order(v0.w, be);
   w[5] = peach_hash32_order(v1.x, be);
   w[6] = peach_hash32_order(v1.y, be);
   w[7] = peach_hash32_order(v1.z, be);
   w[8] = peach_hash32_order(v1.w, be);
   w[9] = be ? WORD32_C(0x80000000) : WORD32_C(0x80);
   w[10] = 0;
   w[11] = 0;
   w[12] = 0;
   w[13] = 0;
   w[14] = be ? 0 : PEACH_HASH32_SEEDBITS;
   w[15] = be ? PEACH_HASH32_SEEDBITS : 0;
}  /* end peach_hash32_view16() */

/****************************************************************
 * Seed-view hashes (peach_nighthash() algorithms 2, 3 and 7)
 ****************************************************************/

/**
 * SHA-1 of the jump seed view (Nighthash algorithm 2).
 * @param n Nonce (8 words)
 * @param m Tile index (seed word 8)
 * @param tile Tile m (64 x uint4, 16-byte aligned, read with __ldg)
 * @param out 8 words as peach_nighthash() (words 5..7 are zero)
*/
PEACH_DEV void peach_sh_sha1(const word32 *n, word32 m, const uint4 *tile,
   word32 *out)
{
   word32 st[5], w[16], carry;
   int k;

   st[0] = WORD32_C(0x67452301);
   st[1] = WORD32_C(0xefcdab89);
   st[2] = WORD32_C(0x98badcfe);
   st[3] = WORD32_C(0x10325476);
   st[4] = WORD32_C(0xc3d2e1f0);
   carry = peach_hash32_view0(w, n, m, tile, 1);
   peach_sha1_compress(st, w);
   PEACH_HASH32_NOUNROLL
   for (k = 1; k < 16; k++) {
      carry = peach_hash32_viewk(w, carry, tile, k, 1);
      peach_sha1_compress(st, w);
   }
   peach_hash32_view16(w, carry, tile, 1);
   peach_sha1_compress(st, w);
   out[0] = peach_bswap32(st[0]);
   out[1] = peach_bswap32(st[1]);
   out[2] = peach_bswap32(st[2]);
   out[3] = peach_bswap32(st[3]);
   out[4] = peach_bswap32(st[4]);
   out[5] = 0;
   out[6] = 0;
   out[7] = 0;
}  /* end peach_sh_sha1() */

/**
 * SHA-256 of the jump seed view (Nighthash algorithm 3).
 * @param n Nonce (8 words)
 * @param m Tile index (seed word 8)
 * @param tile Tile m (64 x uint4, 16-byte aligned, read with __ldg)
 * @param out 8 words as peach_nighthash()
*/
PEACH_DEV void peach_sh_sha256(const word32 *n, word32 m, const uint4 *tile,
   word32 *out)
{
   word32 st[8], w[16], carry;
   int k;

   peach_sha256_iv(st);
   carry = peach_hash32_view0(w, n, m, tile, 1);
   peach_sha256_compress(st, w);
   PEACH_HASH32_NOUNROLL
   for (k = 1; k < 16; k++) {
      carry = peach_hash32_viewk(w, carry, tile, k, 1);
      peach_sha256_compress(st, w);
   }
   peach_hash32_view16(w, carry, tile, 1);
   peach_sha256_compress(st, w);
   out[0] = peach_bswap32(st[0]);
   out[1] = peach_bswap32(st[1]);
   out[2] = peach_bswap32(st[2]);
   out[3] = peach_bswap32(st[3]);
   out[4] = peach_bswap32(st[4]);
   out[5] = peach_bswap32(st[5]);
   out[6] = peach_bswap32(st[6]);
   out[7] = peach_bswap32(st[7]);
}  /* end peach_sh_sha256() */

/**
 * MD5 of the jump seed view (Nighthash algorithm 7).
 * @param n Nonce (8 words)
 * @param m Tile index (seed word 8)
 * @param tile Tile m (64 x uint4, 16-byte aligned, read with __ldg)
 * @param out 8 words as peach_nighthash() (words 4..7 are zero)
*/
PEACH_DEV void peach_sh_md5(const word32 *n, word32 m, const uint4 *tile,
   word32 *out)
{
   word32 st[4], w[16], carry;
   int k;

   st[0] = WORD32_C(0x67452301);
   st[1] = WORD32_C(0xefcdab89);
   st[2] = WORD32_C(0x98badcfe);
   st[3] = WORD32_C(0x10325476);
   carry = peach_hash32_view0(w, n, m, tile, 0);
   peach_md5_compress(st, w);
   PEACH_HASH32_NOUNROLL
   for (k = 1; k < 16; k++) {
      carry = peach_hash32_viewk(w, carry, tile, k, 0);
      peach_md5_compress(st, w);
   }
   peach_hash32_view16(w, carry, tile, 0);
   peach_md5_compress(st, w);
   out[0] = st[0];
   out[1] = st[1];
   out[2] = st[2];
   out[3] = st[3];
   out[4] = 0;
   out[5] = 0;
   out[6] = 0;
   out[7] = 0;
}  /* end peach_sh_md5() */

/****************************************************************
 * Block trailer and final SHA-256
 ****************************************************************/

/**
 * SHA-256 of the block trailer bytes 0..123 (hash0) from the midstate
 * of bytes 0..63. Block 2 = tail (28 bytes) || nonce (32 bytes) || 0x80
 * 00 00 00; block 3 = 56 zero bytes || be64(992).
 * @param mid SHA-256 state after compressing bt[0..63] (as crypto-c
 * SHA256_CTX.state)
 * @param tail bt[64..91] as 7 raw little-endian words
 * @param n Nonce, bt[92..123] as 8 raw little-endian words
 * @param hash0 Digest as 8 little-endian words (word i = bytes 4i..4i+3)
*/
PEACH_DEV void peach_sha256_trailer(const word32 mid[8],
   const word32 tail[7], const word32 n[8], word32 hash0[8])
{
   word32 st[8], w[16];

   st[0] = mid[0]; st[1] = mid[1]; st[2] = mid[2]; st[3] = mid[3];
   st[4] = mid[4]; st[5] = mid[5]; st[6] = mid[6]; st[7] = mid[7];
   w[0] = peach_bswap32(tail[0]);
   w[1] = peach_bswap32(tail[1]);
   w[2] = peach_bswap32(tail[2]);
   w[3] = peach_bswap32(tail[3]);
   w[4] = peach_bswap32(tail[4]);
   w[5] = peach_bswap32(tail[5]);
   w[6] = peach_bswap32(tail[6]);
   w[7] = peach_bswap32(n[0]);
   w[8] = peach_bswap32(n[1]);
   w[9] = peach_bswap32(n[2]);
   w[10] = peach_bswap32(n[3]);
   w[11] = peach_bswap32(n[4]);
   w[12] = peach_bswap32(n[5]);
   w[13] = peach_bswap32(n[6]);
   w[14] = peach_bswap32(n[7]);
   w[15] = WORD32_C(0x80000000);
   peach_sha256_compress(st, w);
   /* constant length block (its schedule folds at compile time) */
   w[0] = 0; w[1] = 0; w[2] = 0; w[3] = 0;
   w[4] = 0; w[5] = 0; w[6] = 0; w[7] = 0;
   w[8] = 0; w[9] = 0; w[10] = 0; w[11] = 0;
   w[12] = 0; w[13] = 0; w[14] = 0;
   w[15] = PEACH_HASH32_TRAILBITS;
   peach_sha256_compress(st, w);
   hash0[0] = peach_bswap32(st[0]);
   hash0[1] = peach_bswap32(st[1]);
   hash0[2] = peach_bswap32(st[2]);
   hash0[3] = peach_bswap32(st[3]);
   hash0[4] = peach_bswap32(st[4]);
   hash0[5] = peach_bswap32(st[5]);
   hash0[6] = peach_bswap32(st[6]);
   hash0[7] = peach_bswap32(st[7]);
}  /* end peach_sha256_trailer() */

/**
 * @private
 * Load 4 aligned tile vectors (64 bytes) as 16 big-endian words.
 * @param w Message block (16 words)
 * @param t First of the 4 vectors
*/
PEACH_DEV void peach_hash32_load64(word32 *w, const uint4 *t)
{
   uint4 v0, v1, v2, v3;

   v0 = PEACH_LDG128(&t[0]);
   v1 = PEACH_LDG128(&t[1]);
   v2 = PEACH_LDG128(&t[2]);
   v3 = PEACH_LDG128(&t[3]);
   w[0] = peach_bswap32(v0.x); w[1] = peach_bswap32(v0.y);
   w[2] = peach_bswap32(v0.z); w[3] = peach_bswap32(v0.w);
   w[4] = peach_bswap32(v1.x); w[5] = peach_bswap32(v1.y);
   w[6] = peach_bswap32(v1.z); w[7] = peach_bswap32(v1.w);
   w[8] = peach_bswap32(v2.x); w[9] = peach_bswap32(v2.y);
   w[10] = peach_bswap32(v2.z); w[11] = peach_bswap32(v2.w);
   w[12] = peach_bswap32(v3.x); w[13] = peach_bswap32(v3.y);
   w[14] = peach_bswap32(v3.z); w[15] = peach_bswap32(v3.w);
}  /* end peach_hash32_load64() */

/**
 * Final SHA-256 of a Peach solve: sha256(hash0 || tile), 1056 bytes.
 * Block 0 = hash0 || t[0], t[1]; block k = 1..15 = t[4k-2..4k+1];
 * block 16 = t[62], t[63] (tile bytes 992..1023) || 0x80 || zeros ||
 * be64(8448).
 * @param hash0 SHA-256 of bt[0..123] as 8 little-endian words
 * @param tile Final tile (64 x uint4, 16-byte aligned, read with __ldg)
 * @param out Digest as 8 little-endian words (word i = bytes 4i..4i+3)
*/
PEACH_DEV void peach_sha256_final(const word32 hash0[8], const uint4 *tile,
   word32 out[8])
{
   word32 st[8], w[16];
   uint4 v0, v1;
   int k;

   peach_sha256_iv(st);
   v0 = PEACH_LDG128(&tile[0]);
   v1 = PEACH_LDG128(&tile[1]);
   w[0] = peach_bswap32(hash0[0]);
   w[1] = peach_bswap32(hash0[1]);
   w[2] = peach_bswap32(hash0[2]);
   w[3] = peach_bswap32(hash0[3]);
   w[4] = peach_bswap32(hash0[4]);
   w[5] = peach_bswap32(hash0[5]);
   w[6] = peach_bswap32(hash0[6]);
   w[7] = peach_bswap32(hash0[7]);
   w[8] = peach_bswap32(v0.x); w[9] = peach_bswap32(v0.y);
   w[10] = peach_bswap32(v0.z); w[11] = peach_bswap32(v0.w);
   w[12] = peach_bswap32(v1.x); w[13] = peach_bswap32(v1.y);
   w[14] = peach_bswap32(v1.z); w[15] = peach_bswap32(v1.w);
   peach_sha256_compress(st, w);
   PEACH_HASH32_NOUNROLL
   for (k = 1; k < 16; k++) {
      peach_hash32_load64(w, &tile[(4 * k) - 2]);
      peach_sha256_compress(st, w);
   }
   v0 = PEACH_LDG128(&tile[62]);
   v1 = PEACH_LDG128(&tile[63]);
   w[0] = peach_bswap32(v0.x); w[1] = peach_bswap32(v0.y);
   w[2] = peach_bswap32(v0.z); w[3] = peach_bswap32(v0.w);
   w[4] = peach_bswap32(v1.x); w[5] = peach_bswap32(v1.y);
   w[6] = peach_bswap32(v1.z); w[7] = peach_bswap32(v1.w);
   w[8] = WORD32_C(0x80000000);
   w[9] = 0; w[10] = 0; w[11] = 0; w[12] = 0; w[13] = 0; w[14] = 0;
   w[15] = PEACH_HASH32_FINALBITS;
   peach_sha256_compress(st, w);
   out[0] = peach_bswap32(st[0]);
   out[1] = peach_bswap32(st[1]);
   out[2] = peach_bswap32(st[2]);
   out[3] = peach_bswap32(st[3]);
   out[4] = peach_bswap32(st[4]);
   out[5] = peach_bswap32(st[5]);
   out[6] = peach_bswap32(st[6]);
   out[7] = peach_bswap32(st[7]);
}  /* end peach_sha256_final() */

/* end include guard */
#endif
