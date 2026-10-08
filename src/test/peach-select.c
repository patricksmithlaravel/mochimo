/**
 * @file peach-select.c
 * @brief Decomposed Peach algorithm selection (peach_select.h) vs the
 * consensus reference peach_dflops() / peach_checkhash() of peach.c.
 * @details CPU only (device headers via test/_cuda_emu.h). Checks:
 * (0) emulation shim and compat helper sanity (incl. cu_rand64());
 * (1) peach_dflops_step() vs a two-word reference peach_dflops() for
 *     words with exponent 0x00/0xFF (every 16th, or all of them with
 *     PEACH_TEST_EXHAUSTIVE=1) x op 0..3 x index {0, 1, 0xFFFFF};
 * (2) select(prefix, transition) vs reference algo on random and
 *     adversarial (nonce, index, tile) cases;
 * (3) the prefix NaN flag is index independent, and an unflagged
 *     (cached) prefix equals the reference;
 * (4) decomposed walk of the mainnet vectors vs reference walk/checkhash;
 * (5) the GPU nonce frame (words 4..7) never triggers NaN replacement;
 * (6) the host first-half redraw rule.
 * OpenMP is used for reference work only.
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/* reference access: public prototypes first, then the reference itself
 * with ALL its external definitions renamed (libmochimo.a has peach.o) */
#include "peach.h"
#define peach_checkhash ref_peach_checkhash
#define peach_init ref_peach_init
#define peach_solve ref_peach_solve
#include "peach.c"
#undef peach_checkhash
#undef peach_init
#undef peach_solve
#include "_assert.h"
#include "_cuda_emu.h"
#include "peach_select.h"
#include "peach_pipeline.cuh"

#define SWEEP_CHUNK  65536    /* words per reference chunk, check (1) */
#define SEL_TILES    20000    /* tiles, check (2) */
#define SEL_PER      10       /* nonces per tile, check (2) */
#define SEL_CHUNK    500      /* tiles per reference chunk, check (2) */
#define NREDRAW      2000     /* simulated batches, check (6) */
#define NVECTORS     5

/* Peach test vectors (as src/test/peach-vectors.c), from the Tfile */
static const word8 Pvector[NVECTORS][sizeof(BTRAILER)] = {
   {  /* Block 0x12852 (75858) - first Peach block, inevitably pseudo */
      0xca, 0x30, 0x56, 0x33, 0x1e, 0x3c, 0x48, 0x4d, 0xa7, 0xdd,
      0xa2, 0xdd, 0x36, 0x28, 0xaa, 0x12, 0x5d, 0x5d, 0xbb, 0xf5,
      0x1e, 0x02, 0x96, 0x94, 0x30, 0xdc, 0xcf, 0x59, 0x12, 0x8e,
      0x9c, 0x0c, 0x52, 0x28, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0xaa, 0xda, 0x1b, 0x5d, 0x2e, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x5f, 0xde, 0x1b, 0x5d, 0xf3, 0x4d,
      0x15, 0xe8, 0x86, 0x0d, 0xc5, 0x53, 0x7d, 0x40, 0xe6, 0x7c,
      0x93, 0x4d, 0x62, 0xb2, 0x66, 0x29, 0xc0, 0x9b, 0x0f, 0xb3,
      0xa8, 0x67, 0x23, 0x8d, 0xc5, 0x95, 0x48, 0x04, 0x65, 0x40
   },
   {  /* Block 0x1285f (75871) - low diff pseudo */
      0xb0, 0xdc, 0x58, 0xa1, 0x2e, 0x99, 0xdd, 0xd1, 0x01, 0xa9,
      0x5e, 0x4f, 0xf8, 0x20, 0xaf, 0x60, 0x6d, 0x0b, 0xe3, 0x99,
      0x1d, 0xe2, 0xb0, 0x15, 0xd8, 0xd7, 0x0b, 0xd2, 0xd6, 0x53,
      0x6a, 0x81, 0x5f, 0x28, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0xdb, 0x0a, 0x1c, 0x5d, 0x21, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x90, 0x0e, 0x1c, 0x5d, 0x41, 0x85,
      0xf2, 0x88, 0x09, 0x44, 0x32, 0x7f, 0xfb, 0x76, 0x1c, 0x32,
      0xc3, 0x12, 0x8e, 0xf1, 0xbf, 0xe2, 0xc0, 0x97, 0xfd, 0xc9,
      0xd3, 0x87, 0xc3, 0xf7, 0x0b, 0xe6, 0xe5, 0x66, 0x5e, 0xae
   },
   {  /* Block 0x128ff (76031) */
      0xc8, 0x7f, 0xdc, 0x08, 0xad, 0x6a, 0x53, 0xef, 0x5f, 0xd0,
      0xf9, 0x8b, 0xf2, 0xa6, 0x6d, 0xb6, 0xc5, 0x84, 0x26, 0x78,
      0x7c, 0xb7, 0x71, 0x24, 0x4e, 0xf7, 0xfc, 0x57, 0x4b, 0x45,
      0x46, 0x95, 0xff, 0x28, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00,
      0xf4, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x03, 0x00,
      0x00, 0x00, 0xbc, 0xc2, 0x1c, 0x5d, 0x20, 0x00, 0x00, 0x00,
      0xf0, 0xf9, 0x58, 0xeb, 0x58, 0xeb, 0xe5, 0x0b, 0x2c, 0xc8,
      0xbc, 0x84, 0xf4, 0xf0, 0x0b, 0x74, 0x80, 0xe9, 0xd2, 0xf6,
      0x10, 0xfe, 0x61, 0x12, 0x74, 0x38, 0xc8, 0xf7, 0xe8, 0x93,
      0x0a, 0x6f, 0x0f, 0x77, 0xe2, 0x01, 0xa5, 0x01, 0x12, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x55, 0xd6,
      0x01, 0x6b, 0xf2, 0x01, 0x2a, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x2b, 0xc4, 0x1c, 0x5d, 0xe4, 0x8a,
      0xca, 0x7b, 0xfa, 0xbd, 0xdb, 0x92, 0xaf, 0xbe, 0x08, 0x52,
      0x7b, 0xfd, 0x49, 0x71, 0x0d, 0xfc, 0x5f, 0xff, 0xe8, 0xed,
      0x15, 0xdf, 0x5b, 0x7c, 0x7a, 0x30, 0xe4, 0xb4, 0x0a, 0x51
   },
   {  /* Block 0x12fff (77823) */
      0xfa, 0x3c, 0x9f, 0x10, 0x8d, 0x12, 0x81, 0x56, 0xcc, 0x68,
      0x31, 0x82, 0x55, 0xc5, 0x14, 0xe7, 0x19, 0x9b, 0xdc, 0x6c,
      0x70, 0xe8, 0xdc, 0xf5, 0xb8, 0xa5, 0x12, 0x77, 0x34, 0xdf,
      0x60, 0x5e, 0xff, 0x2f, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00,
      0xf4, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0x00,
      0x00, 0x00, 0xb0, 0xeb, 0x24, 0x5d, 0x24, 0x00, 0x00, 0x00,
      0xd5, 0xcf, 0x68, 0x1d, 0x3f, 0x00, 0x6f, 0x3f, 0x0c, 0x49,
      0xee, 0x6f, 0x2c, 0xd9, 0x03, 0x08, 0xf5, 0x77, 0xd3, 0x90,
      0x63, 0x27, 0x44, 0xea, 0x31, 0x4b, 0x36, 0x88, 0x17, 0xd0,
      0x35, 0x94, 0xfd, 0xd9, 0x01, 0x68, 0xd8, 0x01, 0x18, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x1e, 0x0f,
      0x54, 0xd8, 0x01, 0xda, 0x15, 0x03, 0x01, 0x05, 0x3d, 0x81,
      0x00, 0x00, 0x00, 0x00, 0x7d, 0xed, 0x24, 0x5d, 0x69, 0xc5,
      0xde, 0xe1, 0x63, 0xbd, 0x2d, 0x72, 0xc2, 0x5b, 0xdc, 0xf7,
      0x3f, 0xc3, 0x61, 0x5a, 0x85, 0x34, 0x17, 0xef, 0x53, 0xc5,
      0x3f, 0x4f, 0x3b, 0xe5, 0x9a, 0x1d, 0x68, 0x88, 0x8c, 0xae
   },
   {  /* Block 0x1ffff (131071) */
      0x54, 0xb4, 0x14, 0x30, 0xe1, 0xaf, 0x0a, 0xe1, 0xfb, 0x4b,
      0x2a, 0xbf, 0x4b, 0x92, 0x33, 0x4e, 0x88, 0x66, 0x7c, 0xae,
      0xdc, 0x23, 0xdc, 0x45, 0x72, 0x3b, 0xb4, 0xdc, 0xbe, 0x37,
      0x2e, 0xa9, 0xff, 0xff, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00,
      0xf4, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0a, 0x00,
      0x00, 0x00, 0x6b, 0x8f, 0x15, 0x5e, 0x22, 0x00, 0x00, 0x00,
      0xed, 0x5a, 0xc4, 0xb3, 0xd4, 0xe1, 0x12, 0xb3, 0x2e, 0xe7,
      0xa1, 0xd7, 0xcc, 0xde, 0x55, 0xeb, 0xb5, 0x05, 0x6a, 0x08,
      0x1f, 0x0d, 0x0d, 0x12, 0xfd, 0x80, 0x7f, 0xa7, 0x9a, 0x60,
      0x5d, 0x9c, 0x0c, 0xff, 0x01, 0x05, 0xc0, 0x01, 0x1e, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xfb, 0xe1,
      0x01, 0x0f, 0x05, 0x60, 0xc7, 0x03, 0x01, 0x57, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0xf1, 0x8f, 0x15, 0x5e, 0x57, 0xa5,
      0xba, 0x1b, 0xb3, 0xed, 0x92, 0x6f, 0x99, 0xf4, 0xeb, 0xe1,
      0xe2, 0xf1, 0x8f, 0xa0, 0x85, 0xe9, 0x58, 0x00, 0x9c, 0x1b,
      0xab, 0x2e, 0x59, 0xf6, 0x12, 0xb5, 0x8c, 0x04, 0xae, 0x66
   }
};

/* Known peach_checkhash() results of Pvector (0 and 1 are pseudo) */
static const word8 Pexpect[NVECTORS][SHA256LEN] = {
   { 0 }, { 0 }, {
      0x00, 0x00, 0x00, 0x00, 0xf5, 0xc5, 0xa5, 0xce, 0xf0, 0xd9, 0x7e,
      0x71, 0x6b, 0xea, 0xe6, 0xe1, 0x37, 0x9d, 0x7d, 0x06, 0xb4, 0xe5,
      0xd9, 0x08, 0xe9, 0x8b, 0x0e, 0x4b, 0x8e, 0xca, 0xe2, 0xfc
   }, {
      0x00, 0x00, 0x00, 0x00, 0x01, 0xde, 0x9c, 0x4c, 0xd9, 0x6d, 0x9d,
      0xfe, 0xee, 0xc3, 0xe1, 0xc7, 0x58, 0x04, 0xfa, 0x24, 0xb6, 0x7e,
      0x80, 0x88, 0xe0, 0x1d, 0xe6, 0xf7, 0x18, 0xf2, 0x30, 0x1f
   }, {
      0x00, 0x00, 0x00, 0x00, 0x26, 0xd2, 0xfc, 0xb9, 0x4c, 0x59, 0x7f,
      0xd2, 0x32, 0x69, 0x98, 0x9c, 0xb1, 0x79, 0x83, 0x42, 0xe4, 0x6a,
      0xe8, 0x5d, 0x04, 0x14, 0xc4, 0x77, 0xa2, 0x24, 0x3a, 0x22
   }
};

/* special float words (NaNs, Infs, zeros, denormals, extremes) */
static const word32 Special[] = {
   0x7fc00000, 0xffffffff, 0x7f800001, 0xff800001, 0x7fffffff, 0xffc00000,
   0x7fbfffff, 0x7f800000, 0xff800000, 0x00000000, 0x80000000, 0x00000001,
   0x80000001, 0x007fffff, 0x807fffff, 0x00400000, 0x000fffff, 0x00000003,
   0x7f7fffff, 0xff7fffff, 0x00800000, 0x80800000, 0x3f800000, 0x4f000000,
   0xcf000000, 0x4f800000, 0x7f000000, 0x01000000, 0x34000000, 0x0d800000
};
#define NSPECIAL  ((int) (sizeof(Special) / sizeof(Special[0])))

/* index values checked exhaustively */
static const word32 Idx3[3] = { 0, 1, PEACHCACHELEN_M1 };

/* reference chunk buffers (shared by OpenMP reference loops) */
static word32 SweepRef[SWEEP_CHUNK * 12];
static word32 SelTile[SEL_CHUNK][PEACHTILELEN32];
static word32 SelIndex[SEL_CHUNK];
static int SelReal[SEL_CHUNK];
static word32 SelNonce[SEL_CHUNK][SEL_PER][8];
static word32 SelCidx[SEL_CHUNK][SEL_PER];
static word32 SelRefAlgo[SEL_CHUNK][SEL_PER];
static word32 SelRefP[SEL_CHUNK][SEL_PER];

/* deterministic test RNG (SplitMix64) */
static word64 Rs = WORD64_C(0x243F6A8885A308D3);

static word64 r64(void)
{
   word64 z = (Rs += WORD64_C(0x9E3779B97F4A7C15));
   z = (z ^ (z >> 30)) * WORD64_C(0xBF58476D1CE4E5B9);
   z = (z ^ (z >> 27)) * WORD64_C(0x94D049BB133111EB);
   return z ^ (z >> 31);
}

static word32 r32(void)
{
   return (word32) (r64() >> 32);
}

static double now_s(void)
{
   struct timespec ts;

   clock_gettime(CLOCK_MONOTONIC, &ts);
   return (double) ts.tv_sec + ((double) ts.tv_nsec * 1e-9);
}

/* random word of a float class: 0 zero, 1 denormal, 2 NaN, 3 Inf,
 * 4 exponent 0xFE, else any */
static word32 class_word(int c)
{
   word32 s = r32() & WORD32_C(0x80000000), m;

   switch (c) {
      case 0: return s;
      case 1:
         do { m = r32() & WORD32_C(0x7FFFFF); } while (m == 0);
         return s | m;
      case 2:
         do { m = r32() & WORD32_C(0x7FFFFF); } while (m == 0);
         return s | WORD32_C(0x7F800000) | m;
      case 3: return s | WORD32_C(0x7F800000);
      case 4: return s | WORD32_C(0x7F000000) | (r32() & 0x7FFFFF);
      default: return r32();
   }
}

/* 1 if the integer operand derived from w is +0 (x/0, x*0, x+-0) */
static int operand_is_zero(word32 w)
{
   word32 shift = ((w & 7) + 1) << 1;

   return ((w >> (8 * ((WORD32_C(0x14198) >> shift) & 3))) & 0xFF) == 0 &&
      ((w >> (8 * ((WORD32_C(0x3D6EC) >> shift) & 3))) & 1) == 0;
}

/* adversarial word: specials, NaN/Inf/0/denormal/huge, zero operands */
static word32 adv_word(void)
{
   static const int cls[5] = { 0, 1, 2, 4, 5 };
   word32 w;
   int i, c;

   switch (r32() % 12) {
      case 0: case 1: return Special[r32() % NSPECIAL];
      case 2: return class_word(2);
      case 3: return class_word(3);
      case 4: return class_word(0);
      case 5: return class_word(1);
      case 6: return class_word(4);
      case 7: case 8:
         c = cls[r32() % 5];
         for (i = 0; i < 256; i++) {
            w = class_word(c);
            if (operand_is_zero(w)) return w;
         }
         return 0;
      case 9: return peach_f2w((float) ((int) (r32() % 512) - 256));
      case 10: return WORD32_C(0x4F000000) + (r32() % 64) - 32;
      default:
         return (r32() & WORD32_C(0x80FFFFFF)) |
            ((r32() % 3) ? WORD32_C(0x00800000) : WORD32_C(0x7F000000));
   }
}

/* random tile index, biased to the edges 0, 1 and 0xFFFFF */
static word32 rand_index(void)
{
   switch (r32() % 16) {
      case 0: return 0;
      case 1: return 1;
      case 2: return PEACHCACHELEN_M1;
      default: return r32() & PEACHCACHELEN_M1;
   }
}

/* nonce words 4..7 from a 64-bit seed, exactly as kcu_peach_solve() */
static void frame_words(word64 seed, word32 *w)
{
   word64 n2, n3;

   n2 = WORD64_C(0x10000050000) |
       Z_ING[(seed     )  & 31]       |
      Z_PREP[(seed >> 5)  &  7] <<  8 |
       Z_ADJ[(seed >> 8)  & 63] << 24 |
        Z_NS[(seed >> 14) & 63] << 32 |
      Z_MASS[(seed >> 20) & 31] << 48 |
       Z_ING[(seed >> 25) & 31] << 56;
   n3 =       WORD64_C(0x50103) |
       Z_ADJ[(seed >> 30) & 63] << 24 |
        Z_NS[(seed >> 36) & 63] << 32;
   w[0] = (word32) n2;
   w[1] = (word32) (n2 >> 32);
   w[2] = (word32) n3;
   w[3] = (word32) (n3 >> 32);
}

/* nonce of a kind: 0 uniform, 1 realistic (host half + GPU frame),
 * 2 adversarial words */
static void make_nonce(word32 *n, int kind)
{
   int k;

   if (kind == 1) {
      trigg_generate(n);
      frame_words(r64(), n + 4);
   } else for (k = 0; k < 8; k++) n[k] = kind ? adv_word() : r32();
}

/* reference algo of a jump seed: nonce || index || tile */
static word32 ref_algo(const word32 *nonce, word32 index, const word32 *tile)
{
   word8 seed[PEACHJUMPLEN];

   memcpy(seed, nonce, 32);
   memcpy(seed + 32, &index, 4);
   memcpy(seed + 36, tile, PEACHTILELEN);
   return peach_dflops(seed, PEACHJUMPLEN, index, 0) & 7;
}

/* transition entry from 4 independent peach_dflops_step() chains */
static word16 naive_transition(const word32 *tile, word32 index)
{
   word32 s, op, t = 0;
   int nf = 0;

   for (s = 0; s < 4; s++) {
      op = peach_dflops_step(index, s, index, &nf);
      op = peach_prefix_words(tile, PEACHTILELEN32, op, index, &nf);
      t |= ((op - s) & 7) << (4 * s);
   }
   return (word16) t;
}

/* Nighthash digest with a given algorithm (as peach_nighthash()) */
static void nighthash_algo(word32 algo, const void *in, size_t len,
   word32 *out)
{
   static const word8 key32[32] = { 0 };
   word8 key64[64];

   memset(key64, 0x01, sizeof(key64));
   memset(out, 0, 32);
   switch (algo & 7) {
      case 0: blake2b(in, len, key32, 32, out, BLAKE2BLEN256); break;
      case 1: blake2b(in, len, key64, 64, out, BLAKE2BLEN256); break;
      case 2: sha1(in, len, out); break;
      case 3: sha256(in, len, out); break;
      case 4: sha3(in, len, out, SHA3LEN256); break;
      case 5: keccak(in, len, out, KECCAKLEN256); break;
      case 6: md2(in, len, out); break;
      default: md5(in, len, out); break;
   }
   /* sha1 (20 bytes), md2 and md5 (16 bytes) are zero filled */
   if ((algo & 7) == 2) out[5] = out[6] = out[7] = 0;
   if ((algo & 7) >= 6) out[4] = out[5] = out[6] = out[7] = 0;
}

/* reference walk: peach_checkhash() body without the syntax gate,
 * recording the reference algo of every jump */
static void ref_walk(const BTRAILER *bt, word8 *final, word32 *algos)
{
   SHA256_CTX ictx;
   word32 tile[PEACHTILELEN32], nonce[8], mario;
   word8 hash[SHA256LEN];
   int i;

   memcpy(nonce, bt->nonce, 32);
   sha256(bt, 124, hash);
   for (mario = hash[0], i = 1; i < SHA256LEN; i++) mario *= hash[i];
   mario &= PEACHCACHELEN_M1;
   for (i = 0; i < PEACHROUNDS; i++) {
      peach_generate(mario, bt->phash, (word8 *) tile);
      algos[i] = ref_algo(nonce, mario, tile);
      peach_jump(&mario, bt->nonce, (word8 *) tile);
   }
   peach_generate(mario, bt->phash, (word8 *) tile);
   sha256_init(&ictx);
   sha256_update(&ictx, hash, SHA256LEN);
   sha256_update(&ictx, tile, PEACHTILELEN);
   sha256_final(&ictx, final);
}

/* decomposed walk: prefix + transition selection, then the hash of the
 * selected algorithm; returns the prefix NaN flag */
static int dec_walk(const BTRAILER *bt, word8 *final, word32 *algos)
{
   SHA256_CTX ictx;
   word32 tile[PEACHTILELEN32], nonce[8], dh[8], mario, p0, p;
   word8 hash[SHA256LEN], seed[PEACHJUMPLEN];
   int i, flag = 0, f2;

   memcpy(nonce, bt->nonce, 32);
   sha256(bt, 124, hash);
   for (mario = hash[0], i = 1; i < SHA256LEN; i++) mario *= hash[i];
   mario &= PEACHCACHELEN_M1;
   p0 = peach_prefix_words(nonce, 8, 0, mario, &flag);
   for (i = 0; i < PEACHROUNDS; i++) {
      peach_generate(mario, bt->phash, (word8 *) tile);
      /* a flagged prefix is index dependent: recompute it per jump */
      f2 = 0;
      p = flag ? peach_prefix_words(nonce, 8, 0, mario, &f2) : p0;
      algos[i] = peach_select_algo(p, peach_transition_tile(tile, mario));
      memcpy(seed, nonce, 32);
      memcpy(seed + 32, &mario, 4);
      memcpy(seed + 36, tile, PEACHTILELEN);
      nighthash_algo(algos[i], seed, PEACHJUMPLEN, dh);
      mario = (dh[0] + dh[1] + dh[2] + dh[3] + dh[4] + dh[5] + dh[6] +
         dh[7]) & PEACHCACHELEN_M1;
   }
   peach_generate(mario, bt->phash, (word8 *) tile);
   sha256_init(&ictx);
   sha256_update(&ictx, hash, SHA256LEN);
   sha256_update(&ictx, tile, PEACHTILELEN);
   sha256_final(&ictx, final);
   return flag;
}

/* emulated kernel: one cu_rand64() per thread */
static void k_rand(word64 *d_state, word64 *d_out)
{
   d_out[(blockIdx.x * blockDim.x) + threadIdx.x] = cu_rand64(d_state);
}

/* (0) emulation shim and compat helpers */
static void check_compat(void)
{
   word64 state[8], seed[8], out[8], z, u64;
   word32 u32, x, buf[4];
   word16 u16 = 0xBEEF;
   uint4 v4;
   int i, n;

   /* CUDA_KERNEL() call shape + cu_rand64() (SplitMix64) */
   for (i = 0; i < 8; i++) seed[i] = state[i] = r64();
   CUDA_KERNEL(k_rand, 2, 4, 0, NULL)(state, out);
   ASSERT_EQ((cudaGetLastError()), (cudaSuccess));
   for (i = 0; i < 8; i++) {
      z = seed[i] + WORD64_C(0x9e3779b97f4a7c15);
      z = (z ^ (z >> 30)) * WORD64_C(0xbf58476d1ce4e5b9);
      z = (z ^ (z >> 27)) * WORD64_C(0x94d049bb133111eb);
      z ^= z >> 31;
      ASSERT_EQ((out[i]), (z));
      ASSERT_EQ((state[i]), (z));
   }
   /* memset, atomics and loads */
   memset(buf, 0xA5, sizeof(buf));
   ASSERT_EQ((PEACH_MEMSET_ASYNC(buf, 0, sizeof(buf), NULL)), (cudaSuccess));
   for (i = 0; i < 4; i++) ASSERT_EQ((buf[i]), (0));
   u32 = 5;
   ASSERT_EQ((PEACH_ATOMIC_ADD32(&u32, 3)), (5));
   ASSERT_EQ((u32), (8));
   ASSERT_EQ((PEACH_ATOMIC_CAS32(&u32, 7, 1)), (8));
   ASSERT_EQ((PEACH_ATOMIC_CAS32(&u32, 8, 1)), (8));
   ASSERT_EQ((u32), (1));
   u64 = WORD64_C(0xFFFFFFFF);
   ASSERT_EQ((PEACH_ATOMIC_ADD64(&u64, 1)), (WORD64_C(0xFFFFFFFF)));
   ASSERT_EQ((u64), (WORD64_C(0x100000000)));
   v4.x = 1; v4.y = 2; v4.z = 3; v4.w = 4;
   ASSERT_EQ((PEACH_LDG128(&v4).w), (4));
   ASSERT_EQ((PEACH_LDG32(&u32)), (1));
   ASSERT_EQ((PEACH_LDG16(&u16)), (0xBEEF));
   /* rotates and byte swap */
   x = WORD32_C(0x80000001) ^ r32();
   z = r64();
   for (n = 0; n < 64; n++) {
      if (n < 32) {
         ASSERT_EQ((peach_rotl32(x, n)), (n ? (x << n) | (x >> (32 - n)) : x));
         ASSERT_EQ((peach_rotr32(x, n)), (n ? (x >> n) | (x << (32 - n)) : x));
         ASSERT_EQ((peach_rotr32(peach_rotl32(x, n), n)), (x));
      }
      ASSERT_EQ((peach_rotl64(z, n)), (n ? (z << n) | (z >> (64 - n)) : z));
      ASSERT_EQ((peach_rotr64(z, n)), (n ? (z >> n) | (z << (64 - n)) : z));
   }
   ASSERT_EQ((peach_bswap32(WORD32_C(0x01020304))), (WORD32_C(0x04030201)));
   ASSERT_EQ((peach_bswap32(x)), (bswap32(x)));
   printf("(0) shim/compat: CUDA_KERNEL + cu_rand64, memset, atomics, "
      "loads, rotates, bswap OK\n");
}

/* word number k of the exponent 0x00/0xFF sweep */
static word32 sweep_word(word32 k, int exhaustive)
{
   word32 e = k;

   /* every 16th word, low nibble pseudo-random (all byte selections) */
   if (!exhaustive) e = (k << 4) | ((k * WORD32_C(2654435769)) >> 28);
   return (e & WORD32_C(0x7FFFFF)) | (((e >> 23) & 1) << 31) |
      (((e >> 24) & 1) ? WORD32_C(0x7F800000) : 0);
}

/* (1) single step vs two-word reference peach_dflops() */
static void check_step(void)
{
   const char *env = getenv("PEACH_TEST_EXHAUSTIVE");
   int exhaustive = (env != NULL && strcmp(env, "1") == 0);
   word32 nwords = exhaustive ? WORD32_C(1) << 25 : WORD32_C(1) << 21;
   word32 lead[12], op0[12], base, cand, o = 0, w, buf[2], mine;
   word64 steps = 0, nanhits = 0, mism = 0;
   double t0 = now_s();
   int c, j, n, nf;

   /* per (index, op & 3): a NaN-free lead word giving that op state */
   for (c = 0; c < 12; c++) {
      for (cand = WORD32_C(0x3F800000);; cand++) {
         nf = 0;
         o = peach_dflops_step(cand, 0, Idx3[c / 4], &nf);
         if (nf || (o & 3) != (word32) (c % 4)) continue;
         if (peach_dflops(&cand, 4, Idx3[c / 4], 0) == o) break;
      }
      lead[c] = cand;
      op0[c] = o;
   }
   /* explicit special words */
   for (j = 0; j < NSPECIAL; j++) {
      for (c = 0; c < 12; c++) {
         buf[0] = lead[c];
         buf[1] = Special[j];
         nf = 0;
         mine = peach_dflops_step(Special[j], op0[c], Idx3[c / 4], &nf);
         mism += (mine != peach_dflops(buf, 8, Idx3[c / 4], 0));
         steps++;
         nanhits += nf;
      }
   }
   /* exponent 0x00 / 0xFF sweep: reference in parallel, then serial */
   for (base = 0; base < nwords; base += SWEEP_CHUNK) {
      n = (int) (nwords - base < SWEEP_CHUNK ? nwords - base : SWEEP_CHUNK);
#pragma omp parallel for schedule(static)
      for (j = 0; j < n; j++) {
         word32 rbuf[2];
         int rc;

         rbuf[1] = sweep_word(base + (word32) j, exhaustive);
         for (rc = 0; rc < 12; rc++) {
            rbuf[0] = lead[rc];
            SweepRef[(j * 12) + rc] = peach_dflops(rbuf, 8, Idx3[rc / 4], 0);
         }
      }
      for (j = 0; j < n; j++) {
         w = sweep_word(base + (word32) j, exhaustive);
         for (c = 0; c < 12; c++) {
            nf = 0;
            mine = peach_dflops_step(w, op0[c], Idx3[c / 4], &nf);
            mism += (mine != SweepRef[(j * 12) + c]);
            steps++;
            nanhits += nf;
         }
      }
   }
   printf("(1) dflops step, exponent 0x00/0xFF %s (%u words) + %d specials"
      " x op 0..3 x index {0,1,0xFFFFF}: steps=%llu NaN-replacing=%llu"
      " MISMATCHES=%llu, %.2fs\n", exhaustive ? "full sweep" :
      "every 16th word", (unsigned) nwords, NSPECIAL,
      (unsigned long long) steps, (unsigned long long) nanhits,
      (unsigned long long) mism, now_s() - t0);
   ASSERT_EQ((mism), (0));
   ASSERT_GT((nanhits), (0));
}

/* (2) + (3) selection and prefix vs reference */
static void check_select(void)
{
   word8 phash[32];
   word32 p, pc, algo;
   word64 cases = 0, flagged = 0, idx0 = 0, idxmax = 0, real = 0;
   word64 bad_algo = 0, bad_t = 0, bad_p = 0, bad_flag = 0, bad_cached = 0;
   word16 t;
   double t0 = now_s();
   int base, i, j, k, kind, flag, flagc;

   for (i = 0; i < 32; i++) phash[i] = (word8) r32();
   for (base = 0; base < SEL_TILES; base += SEL_CHUNK) {
      /* inputs (serial: shared RNG and trigg_generate() state) */
      for (i = 0; i < SEL_CHUNK; i++) {
         kind = (base + i) % 5;
         SelIndex[i] = rand_index();
         SelReal[i] = (kind == 4);
         /* tile kinds: 0,1 uniform, 2 half adversarial, 3 adversarial,
          * 4 real (peach_generate(), below) */
         for (k = 0; k < PEACHTILELEN32 && kind < 4; k++) {
            if (kind <= 1) SelTile[i][k] = r32();
            else if (kind == 3 || (r32() & 1)) SelTile[i][k] = adv_word();
            else SelTile[i][k] = r32();
         }
         for (j = 0; j < SEL_PER; j++) {
            make_nonce(SelNonce[i][j], (i + j) % 3);
            SelCidx[i][j] = rand_index();
         }
      }
      /* reference work (parallel): real tiles, algo and prefix */
#pragma omp parallel for schedule(dynamic, 4)
      for (i = 0; i < SEL_CHUNK; i++) {
         int rj;

         if (SelReal[i]) {
            peach_generate(SelIndex[i], phash, (word8 *) SelTile[i]);
         }
         for (rj = 0; rj < SEL_PER; rj++) {
            SelRefAlgo[i][rj] = ref_algo(SelNonce[i][rj], SelIndex[i],
               SelTile[i]);
            SelRefP[i][rj] = peach_dflops(SelNonce[i][rj], 32, SelIndex[i], 0);
         }
      }
      /* decomposed (serial) */
      for (i = 0; i < SEL_CHUNK; i++) {
         t = peach_transition_tile(SelTile[i], SelIndex[i]);
         bad_t += (t != naive_transition(SelTile[i], SelIndex[i]));
         real += SelReal[i];
         for (j = 0; j < SEL_PER; j++) {
            flag = flagc = 0;
            p = peach_prefix_words(SelNonce[i][j], 8, 0, SelIndex[i], &flag);
            pc = peach_prefix_words(SelNonce[i][j], 8, 0, SelCidx[i][j],
               &flagc);
            algo = peach_select_algo(p, t);
            bad_algo += (algo != SelRefAlgo[i][j]);
            bad_p += (p != SelRefP[i][j]);
            /* (3) flag is index independent; unflagged P is cacheable */
            bad_flag += (flag != flagc);
            if (!flag) {
               bad_cached += (pc != SelRefP[i][j]);
               bad_cached += (peach_select_algo(pc, t) != SelRefAlgo[i][j]);
            }
            flagged += flag;
            idx0 += (SelIndex[i] == 0);
            idxmax += (SelIndex[i] == PEACHCACHELEN_M1);
            cases++;
         }
      }
   }
   printf("(2) select(prefix, T) vs ref peach_dflops(seed,1060,idx,0)&7: "
      "cases=%llu (%llu on real tiles, %llu idx 0, %llu idx 0xFFFFF) "
      "algo MISMATCHES=%llu, T vs 4 naive chains MISMATCHES=%llu, "
      "P vs ref MISMATCHES=%llu, %.2fs\n", (unsigned long long) cases,
      (unsigned long long) (real * SEL_PER), (unsigned long long) idx0,
      (unsigned long long) idxmax, (unsigned long long) bad_algo,
      (unsigned long long) bad_t, (unsigned long long) bad_p,
      now_s() - t0);
   printf("(3) prefix flag set=%llu, flag differs across indices=%llu, "
      "cached (unflagged) P or algo != ref=%llu\n",
      (unsigned long long) flagged, (unsigned long long) bad_flag,
      (unsigned long long) bad_cached);
   ASSERT_GE((cases), (200000));
   ASSERT_EQ((bad_algo), (0));
   ASSERT_EQ((bad_t), (0));
   ASSERT_EQ((bad_p), (0));
   ASSERT_EQ((bad_flag), (0));
   ASSERT_EQ((bad_cached), (0));
   ASSERT_GT((flagged), (0));
   ASSERT_GT((idx0), (0));
   ASSERT_GT((idxmax), (0));
}

/* (4) decomposed walk of the mainnet vectors */
static void check_vectors(void)
{
   BTRAILER bt;
   word8 dec[SHA256LEN], ref[SHA256LEN], out[SHA256LEN];
   word32 dalgo[PEACHROUNDS], ralgo[PEACHROUNDS];
   int v, r, flag, rc;

   for (v = 0; v < NVECTORS; v++) {
      memcpy(&bt, Pvector[v], sizeof(bt));
      ref_walk(&bt, ref, ralgo);
      flag = dec_walk(&bt, dec, dalgo);
      for (r = 0; r < PEACHROUNDS; r++) ASSERT_EQ((dalgo[r]), (ralgo[r]));
      ASSERT_CMP(dec, ref, SHA256LEN);
      memset(out, 0, sizeof(out));
      rc = ref_peach_checkhash(&bt, bt.difficulty[0], out);
      if (v < 2) ASSERT_EQ((rc), (VERROR));  /* pseudo-blocks */
      else {
         ASSERT_EQ((rc), (VEOK));
         ASSERT_CMP(out, Pexpect[v], SHA256LEN);
         ASSERT_CMP(dec, out, SHA256LEN);
      }
      /* vectors 2 and 3 have an index dependent (flagged) prefix */
      ASSERT_EQ((flag), ((v == 2 || v == 3) ? 1 : 0));
      printf("(4) vector %d: prefix-flag=%d algos=%u%u%u%u%u%u%u%u "
         "decomposed final == reference walk%s\n", v, flag,
         (unsigned) dalgo[0], (unsigned) dalgo[1], (unsigned) dalgo[2],
         (unsigned) dalgo[3], (unsigned) dalgo[4], (unsigned) dalgo[5],
         (unsigned) dalgo[6], (unsigned) dalgo[7],
         v < 2 ? " (pseudo-block)" : " == ref_peach_checkhash == expected");
   }
}

/* (5) the GPU nonce frame (words 4..7) never replaces a NaN */
static void check_frame(void)
{
   word32 w, n[8], s, idx;
   word64 checked = 0, fired = 0;
   int a, b, c, i, nf;

   for (s = 0; s < 4; s++) {
      for (i = 0; i < 2; i++) {
         idx = i ? PEACHCACHELEN_M1 : 0;
         /* word 4 = ING PREP 05 ADJ */
         for (a = 0; a < 32; a++) for (b = 0; b < 8; b++) {
            for (c = 0; c < 64; c++) {
               w = (word32) Z_ING[a] | (word32) Z_PREP[b] << 8 |
                  WORD32_C(0x05) << 16 | (word32) Z_ADJ[c] << 24;
               nf = 0;
               peach_dflops_step(w, s, idx, &nf);
               fired += nf;
               checked++;
            }
         }
         /* word 5 = NS 01 MASS ING */
         for (a = 0; a < 64; a++) for (b = 0; b < 32; b++) {
            for (c = 0; c < 32; c++) {
               w = (word32) Z_NS[a] | WORD32_C(0x01) << 8 |
                  (word32) Z_MASS[b] << 16 | (word32) Z_ING[c] << 24;
               nf = 0;
               peach_dflops_step(w, s, idx, &nf);
               fired += nf;
               checked++;
            }
         }
         /* word 6 = 03 01 05 ADJ, word 7 = NS 00 00 00 */
         for (a = 0; a < 64; a++) {
            nf = 0;
            peach_dflops_step(WORD32_C(0x00050103) | (word32) Z_ADJ[a] << 24,
               s, idx, &nf);
            peach_dflops_step((word32) Z_NS[a], s, idx, &nf);
            fired += nf;
            checked += 2;
         }
      }
   }
   /* the frame layout above is the one kcu_peach_solve() generates, and
    * every generated second half passes trigg_syntax() */
   for (i = 0; i < 4096; i++) {
      frame_words(r64(), n);
      ASSERT_EQ(((n[0] >> 16) & 0xFF), (0x05));
      ASSERT_EQ(((n[1] >> 8) & 0xFF), (0x01));
      ASSERT_EQ((n[2] & WORD32_C(0xFFFFFF)), (WORD32_C(0x050103)));
      ASSERT_EQ((n[3] >> 8), (0));
      ASSERT_EQ((trigg_syntax(n)), (VEOK));
   }
   printf("(5) GPU frame words 4..7 x op 0..3 x index {0,0xFFFFF}: "
      "values checked=%llu, NaN replacements=%llu; 4096 frames pass "
      "trigg_syntax\n", (unsigned long long) checked,
      (unsigned long long) fired);
   ASSERT_EQ((fired), (0));
}

/* (6) host redraw rule for the first nonce half */
static void check_redraw(void)
{
   word32 n[8], tile[PEACHTILELEN32], q, q2, p, idx;
   word64 draws = 0, flagged = 0, maxd = 0, d;
   double t0 = now_s();
   int b, k, nf;

   for (b = 0; b < NREDRAW; b++) {
      /* redraw until no NaN replacement in words 0..3 (any index) */
      for (d = 0; d < 64; d++) {
         trigg_generate(n);
         nf = 0;
         q = peach_prefix_words(n, 4, 0, r32() & PEACHCACHELEN_M1, &nf);
         if (!nf) break;
         flagged++;
      }
      ASSERT_LT((d), (64));
      draws += d + 1;
      if (d + 1 > maxd) maxd = d + 1;
      ASSERT_EQ((trigg_syntax(n)), (VEOK));
      /* Q is index independent */
      nf = 0;
      q2 = peach_prefix_words(n, 4, 0, 0, &nf);
      ASSERT_EQ((nf), (0));
      ASSERT_EQ((q2), (q));
      q2 = peach_prefix_words(n, 4, 0, PEACHCACHELEN_M1, &nf);
      ASSERT_EQ((nf), (0));
      ASSERT_EQ((q2), (q));
      /* device continuation over a GPU frame: never flagged, P == ref,
       * and the selected algo matches the reference */
      frame_words(r64(), n + 4);
      idx = rand_index();
      for (k = 0; k < PEACHTILELEN32; k++) {
         tile[k] = (b & 1) ? adv_word() : r32();
      }
      p = peach_prefix_words(n + 4, 4, q, idx, &nf);
      ASSERT_EQ((nf), (0));
      ASSERT_EQ((p), (peach_dflops(n, 32, idx, 0)));
      ASSERT_EQ((peach_select_algo(p, peach_transition_tile(tile, idx))),
         (ref_algo(n, idx, tile)));
   }
   printf("(6) host redraw: %d batches, %llu draws (%.3f per batch, max "
      "%llu), flagged draws %.1f%%; all first halves pass trigg_syntax, "
      "Q index independent, device continuation unflagged, %.2fs\n",
      NREDRAW, (unsigned long long) draws, (double) draws / NREDRAW,
      (unsigned long long) maxd, 100.0 * (double) flagged / (double) draws,
      now_s() - t0);
   /* about 30% of trigg_generate() first halves are flagged */
   ASSERT_GT((flagged * 100), (draws * 15));
   ASSERT_LT((flagged * 100), (draws * 45));
}

int main(void)
{
   double t0 = now_s();

   srand16(0x1234567, 0x89abcdef, 0x2468ace0);
   check_compat();
   check_step();
   check_select();
   check_vectors();
   check_frame();
   check_redraw();
   printf("peach-select: all checks passed, %.2fs\n", now_s() - t0);

   return 0;
}
