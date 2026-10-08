/**
 * @file peach_hashmd2.cuh
 * @brief Peach jump MD2 (Nighthash algorithm 6) over the seed view.
 * @details MD2 of the 1060-byte Peach jump seed
 * `nonce (8 words) || m (1 word) || tile[m] (256 words)`, computed without
 * materializing the seed. The seed is streamed as 16-byte MD2 blocks:
 * block 0 = n0..n3, block 1 = n4..n7, block 2 = m, t[0].xyz, and block
 * k = 3..65 = carry, t[k-2].xyz (carry = previous t[].w), where t is the
 * tile as 64 aligned 16-byte loads (each tile byte is loaded once). Block
 * 66 holds the last 4 seed bytes and 12 padding bytes of value 12 (MD2
 * pads 1060 bytes to 1072), block 67 is the checksum: 68 transforms.
 * <br />
 * The 48-byte state and the 16-byte checksum are held one byte per
 * register; each of the 18 rounds of a transform (48 steps) is fully
 * unrolled with literal register indices, the round loop itself is not.
 * <br />
 * S-box lookups are data dependent and go through a caller supplied
 * pointer: device callers stage c_peach_md2_sbox into a 256-byte
 * `__shared__` table first (constant memory would serialize divergent
 * lookups), for example inside the hash kernel:
 * <pre>
 *    #ifdef __CUDA_ARCH__
 *       __shared__ word8 s_sbox[256];
 *       for (i = threadIdx.x; i < 256; i += blockDim.x) {
 *          s_sbox[i] = c_peach_md2_sbox[i];
 *       }
 *       __syncthreads();
 *       sbox = s_sbox;
 *    #else
 *       sbox = c_peach_md2_sbox;
 *    #endif
 * </pre>
 * CPU emulation (test/_cuda_emu.h) passes c_peach_md2_sbox directly.
 * The output is bit-identical to peach_nighthash() (peach.c) for
 * algorithm 6: MD2 digest in words 0..3, words 4..7 zero.
 * @copyright Adequate Systems LLC, 2018-2025. All Rights Reserved.
 * <br />For license information, please refer to ../LICENSE.md
*/

/* include guard */
#ifndef MOCHIMO_PEACH_HASHMD2_CUH
#define MOCHIMO_PEACH_HASHMD2_CUH


#include "extint.h"           /* for word types */
#include "peach_compat.cuh"   /* for PEACH_* qualifiers, uint4 */

/* MD2 padding of the 1060-byte seed: 12 bytes of value 12 */
#define PEACH_MD2_PAD   WORD32_C(0x0C0C0C0C)

/**
 * MD2 S-box (RFC 1319), a permutation of 0..255 derived from pi.
 * Device callers copy it to `__shared__` memory before hashing.
*/
static PEACH_CONST word8 c_peach_md2_sbox[256] = {
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

/* One MD2 round step on state byte register x[k]; t = previous byte.
 * Byte values stay in 0..255 (XOR of bytes), no masking needed. */
#define PEACH_MD2_STEP(x, k, t, s) \
   ((t) = ((x)[k] ^= (word32) (s)[t]))
#define PEACH_MD2_STEP4(x, k, t, s) \
   PEACH_MD2_STEP(x, (k), t, s); PEACH_MD2_STEP(x, (k) + 1, t, s); \
   PEACH_MD2_STEP(x, (k) + 2, t, s); PEACH_MD2_STEP(x, (k) + 3, t, s)
#define PEACH_MD2_STEP16(x, k, t, s) \
   PEACH_MD2_STEP4(x, (k), t, s); PEACH_MD2_STEP4(x, (k) + 4, t, s); \
   PEACH_MD2_STEP4(x, (k) + 8, t, s); PEACH_MD2_STEP4(x, (k) + 12, t, s)

/* Block byte j (value b, 0..255): state init x[16 + j] = b,
 * x[32 + j] = b ^ x[j], and checksum update c[j] ^= S[b ^ l], l = c[j] */
#define PEACH_MD2_BYTE(x, c, j, b, l, s) \
   (x)[16 + (j)] = (b); (x)[32 + (j)] = (x)[16 + (j)] ^ (x)[j]; \
   (l) = ((c)[j] ^= (word32) (s)[(x)[16 + (j)] ^ (l)])
#define PEACH_MD2_WORD(x, c, i, w, l, s) \
   PEACH_MD2_BYTE(x, c, 4 * (i), (w) & 0xFF, l, s); \
   PEACH_MD2_BYTE(x, c, (4 * (i)) + 1, ((w) >> 8) & 0xFF, l, s); \
   PEACH_MD2_BYTE(x, c, (4 * (i)) + 2, ((w) >> 16) & 0xFF, l, s); \
   PEACH_MD2_BYTE(x, c, (4 * (i)) + 3, ((w) >> 24) & 0xFF, l, s)

/* Checksum byte j as block byte (final block, no checksum update) */
#define PEACH_MD2_CBYTE(x, c, j) \
   (x)[16 + (j)] = (c)[j]; (x)[32 + (j)] = (c)[j] ^ (x)[j]
#define PEACH_MD2_CBYTE4(x, c, j) \
   PEACH_MD2_CBYTE(x, c, (j)); PEACH_MD2_CBYTE(x, c, (j) + 1); \
   PEACH_MD2_CBYTE(x, c, (j) + 2); PEACH_MD2_CBYTE(x, c, (j) + 3)

/* 4 state bytes x[k..k+3] as a little-endian word */
#define PEACH_MD2_LE32(x, k) \
   ((x)[k] | ((x)[(k) + 1] << 8) | ((x)[(k) + 2] << 16) | \
      ((x)[(k) + 3] << 24))

/**
 * @private
 * The 18 MD2 rounds over the 48 state byte registers.
 * @param x State, 48 byte registers (x[16..47] set by the caller)
 * @param sbox MD2 S-box (see peach_sh_md2())
*/
PEACH_DEV void peach_md2_rounds(word32 *x, const word8 *sbox)
{
   word32 t, j;

   t = 0;
#ifdef __CUDA_ARCH__
   #pragma unroll 1
#endif
   for (j = 0; j < 18; j++) {
      PEACH_MD2_STEP16(x, 0, t, sbox);
      PEACH_MD2_STEP16(x, 16, t, sbox);
      PEACH_MD2_STEP16(x, 32, t, sbox);
      t = (t + j) & 0xFF;
   }
}  /* end peach_md2_rounds() */

/**
 * @private
 * MD2 transform of one 16-byte message block, with checksum update.
 * @param x State, 48 byte registers
 * @param c Checksum, 16 byte registers
 * @param w0 Block bytes 0..3 as a little-endian word
 * @param w1 Block bytes 4..7 as a little-endian word
 * @param w2 Block bytes 8..11 as a little-endian word
 * @param w3 Block bytes 12..15 as a little-endian word
 * @param sbox MD2 S-box (see peach_sh_md2())
*/
PEACH_DEV void peach_md2_block(word32 *x, word32 *c, word32 w0, word32 w1,
   word32 w2, word32 w3, const word8 *sbox)
{
   word32 l = c[15];

   PEACH_MD2_WORD(x, c, 0, w0, l, sbox);
   PEACH_MD2_WORD(x, c, 1, w1, l, sbox);
   PEACH_MD2_WORD(x, c, 2, w2, l, sbox);
   PEACH_MD2_WORD(x, c, 3, w3, l, sbox);
   peach_md2_rounds(x, sbox);
}  /* end peach_md2_block() */

/**
 * @private
 * Final MD2 transform: the checksum as message block (the checksum
 * update of this block cannot affect the digest and is omitted).
 * @param x State, 48 byte registers
 * @param c Checksum, 16 byte registers
 * @param sbox MD2 S-box (see peach_sh_md2())
*/
PEACH_DEV void peach_md2_last(word32 *x, const word32 *c, const word8 *sbox)
{
   PEACH_MD2_CBYTE4(x, c, 0);
   PEACH_MD2_CBYTE4(x, c, 4);
   PEACH_MD2_CBYTE4(x, c, 8);
   PEACH_MD2_CBYTE4(x, c, 12);
   peach_md2_rounds(x, sbox);
}  /* end peach_md2_last() */

/**
 * MD2 of a Peach jump seed (Nighthash algorithm 6), seed view.
 * @param n Nonce, 8 words (seed words 0..7)
 * @param m Tile index (seed word 8)
 * @param tile Tile @a m as 64 x 16 bytes, 16-byte aligned (seed words
 * 9..264), read with 64 aligned 128-bit loads
 * @param sbox MD2 S-box: a `__shared__` copy of c_peach_md2_sbox on the
 * device, c_peach_md2_sbox itself in CPU emulation
 * @param out Digest as 8 little-endian words (as peach_nighthash()
 * writes it): MD2 in words 0..3, words 4..7 zero
*/
PEACH_DEV void peach_sh_md2(const word32 *n, word32 m, const uint4 *tile,
   const word8 *sbox, word32 *out)
{
   word32 x[48], c[16], carry, w0, w1, w2, w3;
   uint4 v;
   int i;

   /* zero state bytes 0..15 and checksum (x[16..47] set per block) */
   x[0] = x[1] = x[2] = x[3] = x[4] = x[5] = x[6] = x[7] = 0;
   x[8] = x[9] = x[10] = x[11] = x[12] = x[13] = x[14] = x[15] = 0;
   c[0] = c[1] = c[2] = c[3] = c[4] = c[5] = c[6] = c[7] = 0;
   c[8] = c[9] = c[10] = c[11] = c[12] = c[13] = c[14] = c[15] = 0;
   /* first tile load, latency hidden behind the 2 nonce blocks */
   v = PEACH_LDGTILE(&tile[0]);
   /* blocks 0, 1: nonce */
   peach_md2_block(x, c, n[0], n[1], n[2], n[3], sbox);
   peach_md2_block(x, c, n[4], n[5], n[6], n[7], sbox);
   /* blocks 2..65: carried word (m, then t[i - 1].w) || t[i].xyz;
    * the load of t[i + 1] overlaps the transform of block i + 2 */
   carry = m;
   for (i = 0; i < 64; i++) {
      w0 = carry;
      w1 = v.x;
      w2 = v.y;
      w3 = v.z;
      carry = v.w;
      if (i < 63) v = PEACH_LDGTILE(&tile[i + 1]);
      peach_md2_block(x, c, w0, w1, w2, w3, sbox);
   }
   /* block 66: last seed word (t[63].w) || 12 bytes of value 12 */
   peach_md2_block(x, c, carry, PEACH_MD2_PAD, PEACH_MD2_PAD,
      PEACH_MD2_PAD, sbox);
   /* block 67: checksum */
   peach_md2_last(x, c, sbox);
   /* digest = state bytes 0..15; words 4..7 zero */
   out[0] = PEACH_MD2_LE32(x, 0);
   out[1] = PEACH_MD2_LE32(x, 4);
   out[2] = PEACH_MD2_LE32(x, 8);
   out[3] = PEACH_MD2_LE32(x, 12);
   out[4] = out[5] = out[6] = out[7] = 0;
}  /* end peach_sh_md2() */

/* helper macros are local to this header */
#undef PEACH_MD2_STEP
#undef PEACH_MD2_STEP4
#undef PEACH_MD2_STEP16
#undef PEACH_MD2_BYTE
#undef PEACH_MD2_WORD
#undef PEACH_MD2_CBYTE
#undef PEACH_MD2_CBYTE4
#undef PEACH_MD2_LE32

/* end include guard */
#endif
