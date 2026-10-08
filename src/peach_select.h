/**
 * @file peach_select.h
 * @brief Peach jump algorithm selection, decomposed (host and device).
 * @details The algorithm of a Peach jump from tile `m` is
 * `peach_dflops(seed, 1060, m, 0) & 7` over the jump seed
 * `nonce (8 words) || m (1 word) || tile[m] (256 words)`. Control flow of a
 * single (non-transforming) dflops step depends only on the word and on
 * `(op + selected byte) & 3`, so the op increment of a fixed word
 * sequence modulo 8 depends only on `op & 3` when the sequence starts:
 * - P = op after nonce words 0..7 (peach_prefix_words());
 * - T[m] = 4 nibbles D[s] = (run(op = s, words 8..264) - s) & 7, s = 0..3
 *   (peach_transition_tile());
 * - algo = (P + ((T[m] >> ((P & 3) * 4)) & 0xF)) & 7 (peach_select_algo()).
 * P is independent of the tile index unless a NaN replacement fires in
 * the nonce words (reported via the @a nanf flag).
 * <br />
 * Every float operation goes through the wrappers below, keyed on
 * `__CUDA_ARCH__`: device code uses the round-to-nearest intrinsics,
 * host code (nvcc host pass and the CPU emulation) plain IEEE-754 C.
 * NaN tests are bitwise. Results are only exact without FTZ/DAZ and
 * fast-math (see the GNUmakefile NVCCARGS guard and the host guard below).
 * @copyright Adequate Systems LLC, 2018-2025. All Rights Reserved.
 * <br />For license information, please refer to ../LICENSE.md
*/

/* include guard */
#ifndef MOCHIMO_PEACH_SELECT_H
#define MOCHIMO_PEACH_SELECT_H


#include <string.h>           /* for memcpy() */
#include "extint.h"           /* for word types */
#include "peach.h"            /* for PEACHTILELEN32 */
#include "peach_compat.cuh"   /* for PEACH_HD, peach_rotr32() */

/* host float operations must be exact IEEE-754 (no -ffast-math) */
#if !defined(__CUDA_ARCH__)
   #if defined(__FAST_MATH__) || \
      (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__)
      #error "peach_select.h requires IEEE-754 float (no fast-math)"
   #endif
#endif

/**
 * @private
 * Bitwise NaN test of a 32-bit float word.
 * @param w Float bits
 * @returns 1 if @a w is any NaN (quiet or signaling, any sign), else 0
*/
PEACH_HD int peach_isnan_w(word32 w)
{
   return (w & WORD32_C(0x7FFFFFFF)) > WORD32_C(0x7F800000);
}  /* end peach_isnan_w() */

/**
 * @private
 * Reinterpret float bits as a 32-bit word.
*/
PEACH_HD word32 peach_f2w(float f)
{
#ifdef __CUDA_ARCH__
   return __float_as_uint(f);
#else
   word32 w;

   memcpy(&w, &f, sizeof(w));
   return w;
#endif
}  /* end peach_f2w() */

/**
 * @private
 * Reinterpret a 32-bit word as float bits.
*/
PEACH_HD float peach_w2f(word32 w)
{
#ifdef __CUDA_ARCH__
   return __uint_as_float((unsigned int) w);
#else
   float f;

   memcpy(&f, &w, sizeof(f));
   return f;
#endif
}  /* end peach_w2f() */

/**
 * @private
 * Round-to-nearest float operation selected by @a op & 3:
 * 0 add, 1 subtract, 2 multiply, 3 divide.
*/
PEACH_HD float peach_fop(float a, float b, word32 op)
{
#ifdef __CUDA_ARCH__
   switch (op & 3) {
      case 3: return __fdiv_rn(a, b);
      case 2: return __fmul_rn(a, b);
      case 1: return __fsub_rn(a, b);
      default: return __fadd_rn(a, b);
   }
#else
   switch (op & 3) {
      case 3: return a / b;
      case 2: return a * b;
      case 1: return a - b;
      default: return a + b;
   }
#endif
}  /* end peach_fop() */

/**
 * @private
 * Signed integer to float conversion (round-to-nearest).
*/
PEACH_HD float peach_i2f(int32 i)
{
#ifdef __CUDA_ARCH__
   return __int2float_rn((int) i);
#else
   return (float) i;
#endif
}  /* end peach_i2f() */

/**
 * @private
 * Unsigned integer to float conversion (round-to-nearest).
*/
PEACH_HD float peach_u2f(word32 u)
{
#ifdef __CUDA_ARCH__
   return __uint2float_rn((unsigned int) u);
#else
   return (float) u;
#endif
}  /* end peach_u2f() */

/**
 * @private
 * Sum of the 4 bytes of a 32-bit word.
*/
PEACH_HD word32 peach_bytesum(word32 r)
{
   return (r & 0xFF) + ((r >> 8) & 0xFF) + ((r >> 16) & 0xFF) + (r >> 24);
}  /* end peach_bytesum() */

/**
 * @private
 * Decode one dflops word: the op byte and the float operands.
 * @param w Seed word (little endian bytes)
 * @param index Tile index used for NaN replacement
 * @param x Pointer to place the first operand (NaN replaced by index)
 * @param y Pointer to place the second (integer derived) operand
 * @param nanf Set to 1 when the pre-operation NaN replacement occurs
 * @returns the byte added to the op code before the operation
*/
PEACH_HD word32 peach_dflops_decode(word32 w, word32 index, float *x,
   float *y, int *nanf)
{
   word32 shift, operand;

   /* first byte determines the byte selection for the 3 uses below */
   shift = ((w & 7) + 1) << 1;
   /* 2) value of the operand, 3) its sign bit (after allocation) */
   operand = (w >> (8 * ((WORD32_C(0x14198) >> shift) & 3))) & 0xFF;
   operand |= ((w >> (8 * ((WORD32_C(0x3D6EC) >> shift) & 3))) & 1) << 31;
   *y = peach_i2f((int32) operand);
   /* replace pre-operation NaN with index */
   if (peach_isnan_w(w)) {
      *x = peach_u2f(index);
      *nanf = 1;
   } else *x = peach_w2f(w);
   /* 1) byte determining the float operation type */
   return (w >> (8 * ((WORD32_C(0x26C34) >> shift) & 3))) & 0xFF;
}  /* end peach_dflops_decode() */

/**
 * @private
 * One non-transforming peach_dflops() step over a single 32-bit word.
 * @param w Seed word (little endian bytes)
 * @param op Operation code entering this word
 * @param index Tile index used for NaN replacement
 * @param nanf Set to 1 when a NaN replacement occurs (never cleared)
 * @returns Operation code after this word
*/
PEACH_HD word32 peach_dflops_step(word32 w, word32 op, word32 index,
   int *nanf)
{
   word32 r;
   float x, y;

   op += peach_dflops_decode(w, index, &x, &y, nanf);
   r = peach_f2w(peach_fop(x, y, op));
   /* replace post-operation NaN with index */
   if (peach_isnan_w(r)) {
      r = peach_f2w(peach_u2f(index));
      *nanf = 1;
   }
   return op + peach_bytesum(r);
}  /* end peach_dflops_step() */

/**
 * @private
 * Non-transforming peach_dflops() steps over @a count words.
 * @param w Pointer to seed words (little endian bytes)
 * @param count Number of words
 * @param op Operation code entering the first word
 * @param index Tile index used for NaN replacement
 * @param nanf Set to 1 when any NaN replacement occurs (never cleared)
 * @returns Operation code after the last word
 * @note With @a op = 0 and the 8 nonce words, the result is the prefix P
 * of a jump seed; P is index independent if @a nanf stays clear.
*/
PEACH_HD word32 peach_prefix_words(const word32 *w, int count, word32 op,
   word32 index, int *nanf)
{
   int i;

   for (i = 0; i < count; i++) op = peach_dflops_step(w[i], op, index, nanf);
   return op;
}  /* end peach_prefix_words() */

/**
 * @private
 * Increments (op & 7) of all four float operations of one dflops word,
 * packed per entering op & 3: byte j holds the op increment modulo 8 for
 * an op with (op & 3) == j entering this word.
*/
PEACH_HD word32 peach_dflops_incs(word32 w, word32 index)
{
   word32 b, r, idx, incs;
   float x, y;
   int nanf = 0;

   b = peach_dflops_decode(w, index, &x, &y, &nanf);
   idx = peach_f2w(peach_u2f(index));
   /* byte k = (b + bytesum(result of operation k)) & 7 */
   r = peach_f2w(peach_fop(x, y, 0));
   incs = (b + peach_bytesum(peach_isnan_w(r) ? idx : r)) & 7;
   r = peach_f2w(peach_fop(x, y, 1));
   incs |= ((b + peach_bytesum(peach_isnan_w(r) ? idx : r)) & 7) << 8;
   r = peach_f2w(peach_fop(x, y, 2));
   incs |= ((b + peach_bytesum(peach_isnan_w(r) ? idx : r)) & 7) << 16;
   r = peach_f2w(peach_fop(x, y, 3));
   incs |= ((b + peach_bytesum(peach_isnan_w(r) ? idx : r)) & 7) << 24;
   /* an entering op o performs operation (o + b) & 3: rotate so that
    * byte j holds the increment of operation (j + b) & 3 */
   return peach_rotr32(incs, (int) (8 * (b & 3)));
}  /* end peach_dflops_incs() */

/**
 * Initial packed state of the 4 transition chains: byte s holds the op
 * (mod 8) of the chain started with op = s.
*/
#define PEACH_TRANSITION_INIT  WORD32_C(0x03020100)

/**
 * @private
 * Advance the 4 packed transition chains over one seed word. Each chain
 * selects its increment from peach_dflops_incs() branch-free.
 * @param ops Packed chain state (PEACH_TRANSITION_INIT before word 8)
 * @param w Seed word (little endian bytes)
 * @param index Tile index used for NaN replacement
 * @returns packed chain state after @a w
*/
PEACH_HD word32 peach_transition_step(word32 ops, word32 w, word32 index)
{
   word32 incs = peach_dflops_incs(w, index);

   /* lanes stay < 8 and lane sums < 16: no carry between lanes */
   ops += ((incs >> (8 * (ops & 3))) & 7) |
      (((incs >> (8 * ((ops >> 8) & 3))) & 7) << 8) |
      (((incs >> (8 * ((ops >> 16) & 3))) & 7) << 16) |
      (((incs >> (8 * ((ops >> 24) & 3))) & 7) << 24);
   return ops & WORD32_C(0x07070707);
}  /* end peach_transition_step() */

/**
 * @private
 * Pack the final 4 chain states into a transition table entry.
 * @param ops Packed chain state after word 264
 * @returns T = D[0] | D[1] << 4 | D[2] << 8 | D[3] << 12
*/
PEACH_HD word16 peach_transition_pack(word32 ops)
{
   /* D[s] = (op_s - s) & 7 = (lane s + 8 - s) & 7, without lane carry */
   word32 d = (ops + WORD32_C(0x05060708)) & WORD32_C(0x07070707);

   return (word16) ((d & 0xF) | ((d >> 4) & 0xF0) | ((d >> 8) & 0xF00) |
      ((d >> 12) & 0xF000));
}  /* end peach_transition_pack() */

/**
 * @private
 * Transition table entry of a tile: the op increments (mod 8) of words
 * 8..264 of a jump seed (the tile index word, then the 256 tile words)
 * for each entering op & 3. The 4 chains are evaluated together; each
 * word computes the 4 float results once and every chain selects its
 * increment branch-free.
 * @param tile256 Pointer to the 256 words of tile @a index
 * @param index Tile index (word 8 of the seed, and NaN replacement)
 * @returns T = D[0] | D[1] << 4 | D[2] << 8 | D[3] << 12
*/
PEACH_HD word16 peach_transition_tile(const word32 *tile256, word32 index)
{
   word32 ops;
   int i;

   ops = peach_transition_step(PEACH_TRANSITION_INIT, index, index);
   for (i = 0; i < PEACHTILELEN32; i++) {
      ops = peach_transition_step(ops, tile256[i], index);
   }
   return peach_transition_pack(ops);
}  /* end peach_transition_tile() */

/**
 * @private
 * Select the jump algorithm from a prefix op and a transition entry.
 * @param p Operation code after the 8 nonce words (only p & 7 is used)
 * @param t Transition table entry of the tile (peach_transition_tile())
 * @returns algorithm number 0..7, equal to the reference
 * peach_dflops(seed, PEACHJUMPLEN, index, 0) & 7
*/
PEACH_HD word32 peach_select_algo(word32 p, word16 t)
{
   return (p + (((word32) t >> ((p & 3) * 4)) & 0xF)) & 7;
}  /* end peach_select_algo() */

/* end include guard */
#endif
