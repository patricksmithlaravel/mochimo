/**
 * @file peach-hashmd2.c
 * @brief Seed-view MD2 (peach_hashmd2.cuh) vs the consensus reference.
 * @details CPU only (device header via test/_cuda_emu.h). Every case
 * compares peach_sh_md2() with crypto-c md2() over the materialized
 * 1060-byte jump seed `nonce || index || tile`, post-processed as
 * peach_nighthash() does for algorithm 6 (words 4..7 zero). Checks:
 * (0) S-box sanity (permutation of 0..255) and crypto-c md2() known
 *     answers (RFC 1319);
 * (1) edge cases: tiles all 0x00, all 0xFF and real peach_generate()
 *     tiles 0, 1, 0xFFFFF x index {0, 1, 0xFFFFF, 0xFFFFFFFF} x nonces
 *     {zero, 0xFF, counting, random};
 * (2) random (nonce, index, tile) cases incl. real and low entropy
 *     tiles, run as an emulated kernel over a map of adjacent tiles;
 *     where the reference selection picks MD2, peach_nighthash() and
 *     peach_jump() of the reference agree with the seed-view result;
 * (3) the S-box is read through the pointer argument (a copy gives the
 *     same digest, swapping any two adjacent entries changes it), words
 *     4..7 are zeroed and nothing past out[7] is written.
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
#include "peach_hashmd2.cuh"

#define NRANDOM   4096                 /* random cases, check (2) */
#define TILE4     (PEACHTILELEN / 16)  /* uint4 per tile */
#define OUTGUARD  WORD32_C(0xA5A5A5A5)

/* deterministic test RNG (SplitMix64) */
static word64 Rs = WORD64_C(0x6A09E667F3BCC908);

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

/* materialized jump seed: nonce (32 B) || index (4 B) || tile (1024 B) */
static void make_seed(word8 *seed, const word32 *n, word32 m,
   const uint4 *tile)
{
   memcpy(seed, n, 32);
   memcpy(seed + 32, &m, 4);
   memcpy(seed + 36, tile, PEACHTILELEN);
}

/* reference digest: crypto-c md2() with peach_nighthash() zero fill */
static void ref_md2(const word8 *seed, word32 *out)
{
   memset(out, 0xA5, 32);
   md2(seed, PEACHJUMPLEN, out);
   out[4] = out[5] = out[6] = out[7] = 0;
}

/* tile of a kind: 0 all 0x00, 1 all 0xFF, 2 uniform random,
 * 3 bytes in {0, 1}, 4 one repeated random byte, 5 sparse (mostly 0x00),
 * 6 bytes in {0x00, 0x0C, 0x80, 0xFF}, 7 real peach_generate(m) tile */
static void make_tile(uint4 *tile, int kind, word32 m, const word8 *phash)
{
   static const word8 alpha[4] = { 0x00, 0x0C, 0x80, 0xFF };
   word8 *b = (word8 *) tile;
   word8 v;
   int i;

   switch (kind) {
      case 0: memset(b, 0x00, PEACHTILELEN); break;
      case 1: memset(b, 0xFF, PEACHTILELEN); break;
      case 2: for (i = 0; i < PEACHTILELEN; i++) b[i] = (word8) r32(); break;
      case 3: for (i = 0; i < PEACHTILELEN; i++) b[i] = r32() & 1; break;
      case 4:
         v = (word8) r32();
         memset(b, v, PEACHTILELEN);
         break;
      case 5:
         memset(b, 0x00, PEACHTILELEN);
         for (i = 0; i < 16; i++) b[r32() % PEACHTILELEN] = (word8) r32();
         break;
      case 6: for (i = 0; i < PEACHTILELEN; i++) b[i] = alpha[r32() & 3]; break;
      default: peach_generate(m, phash, b); break;
   }
}

/* random tile index, biased to the edges 0, 1, 0xFFFFF and 0xFFFFFFFF */
static word32 rand_index(void)
{
   switch (r32() % 16) {
      case 0: return 0;
      case 1: return 1;
      case 2: return PEACHCACHELEN_M1;
      case 3: return WORD32_C(0xFFFFFFFF);
      default: return r32() & PEACHCACHELEN_M1;
   }
}

/* emulated kernel: thread k hashes case k (seed view over the map) */
static void k_md2(const uint4 *d_map, const word32 *d_nonce,
   const word32 *d_index, word32 *d_out, int count)
{
   int k = (int) ((blockIdx.x * blockDim.x) + threadIdx.x);

   if (k >= count) return;
   peach_sh_md2(&d_nonce[k * 8], d_index[k], &d_map[k * TILE4],
      c_peach_md2_sbox, &d_out[k * 8]);
}

/* (0) S-box and reference sanity */
static void check_sbox(void)
{
   /* RFC 1319 test suite */
   static const char *kat_in[3] = { "", "abc", "message digest" };
   static const word8 kat_out[3][16] = {
      {  0x83, 0x50, 0xe5, 0xa3, 0xe2, 0x4c, 0x15, 0x3d,
         0xf2, 0x27, 0x5c, 0x9f, 0x80, 0x69, 0x27, 0x73 },
      {  0xda, 0x85, 0x3b, 0x0d, 0x3f, 0x88, 0xd9, 0x9b,
         0x30, 0x28, 0x3a, 0x69, 0xe6, 0xde, 0xd6, 0xbb },
      {  0xab, 0x4f, 0x49, 0x6b, 0xfb, 0x2a, 0x53, 0x0b,
         0x21, 0x9f, 0xf3, 0x30, 0x31, 0xfe, 0x06, 0xb0 }
   };
   word8 digest[16];
   int seen[256];
   int i;

   memset(seen, 0, sizeof(seen));
   for (i = 0; i < 256; i++) seen[c_peach_md2_sbox[i]]++;
   for (i = 0; i < 256; i++) ASSERT_EQ((seen[i]), (1));
   ASSERT_EQ((c_peach_md2_sbox[0]), (41));
   ASSERT_EQ((c_peach_md2_sbox[255]), (20));
   for (i = 0; i < 3; i++) {
      md2(kat_in[i], strlen(kat_in[i]), digest);
      ASSERT_CMP(digest, kat_out[i], 16);
   }
   ASSERT_EQ((sizeof(uint4)), (16));
   printf("(0) S-box is a permutation of 0..255; crypto-c md2() RFC 1319 "
      "known answers OK\n");
}

/* (1) edge tiles x edge indices x special nonces */
static void check_edges(void)
{
   static const word32 idx[4] = {
      0, 1, PEACHCACHELEN_M1, WORD32_C(0xFFFFFFFF)
   };
   static const word32 real[3] = { 0, 1, PEACHCACHELEN_M1 };
   uint4 tile[TILE4];
   word32 n[8], out[8], ref[8];
   word8 seed[PEACHJUMPLEN], phash[32];
   int tk, nk, ik, i, cases = 0;

   for (tk = 0; tk < 8; tk++) {
      /* 0 all 0x00, 1 all 0xFF, 2..4 real tiles 0, 1, 0xFFFFF with a
       * zero phash, 5..7 the same with a random phash */
      if (tk < 2) make_tile(tile, tk, 0, NULL);
      else {
         for (i = 0; i < 32; i++) phash[i] = tk < 5 ? 0 : (word8) r32();
         make_tile(tile, 7, real[(tk - 2) % 3], phash);
      }
      for (nk = 0; nk < 4; nk++) {
         for (i = 0; i < 8; i++) {
            switch (nk) {
               case 0: n[i] = 0; break;
               case 1: n[i] = WORD32_C(0xFFFFFFFF); break;
               case 2: n[i] = WORD32_C(0x03020100) +
                  (WORD32_C(0x04040404) * (word32) i); break;
               default: n[i] = r32(); break;
            }
         }
         for (ik = 0; ik < 4; ik++) {
            make_seed(seed, n, idx[ik], tile);
            ref_md2(seed, ref);
            memset(out, 0xA5, sizeof(out));
            peach_sh_md2(n, idx[ik], tile, c_peach_md2_sbox, out);
            ASSERT_CMP(out, ref, 32);
            cases++;
         }
      }
   }
   printf("(1) edge cases: tiles {0x00, 0xFF, real 0/1/0xFFFFF x 2 phash} "
      "x 4 nonces x index {0,1,0xFFFFF,0xFFFFFFFF}: %d cases match\n",
      cases);
}

/* (2) random cases through an emulated kernel */
static void check_random(void)
{
   uint4 *map;
   word32 *nonce, *index, *out, ref[8], nh[8], next;
   word8 seed[PEACHJUMPLEN], phash[32];
   word64 natural = 0, real = 0, low = 0, edge = 0;
   double t0 = now_s(), t1;
   int k, i, kind;

   map = (uint4 *) malloc((size_t) NRANDOM * TILE4 * sizeof(uint4));
   nonce = (word32 *) malloc((size_t) NRANDOM * 8 * sizeof(word32));
   index = (word32 *) malloc((size_t) NRANDOM * sizeof(word32));
   out = (word32 *) malloc((size_t) NRANDOM * 8 * sizeof(word32));
   ASSERT_NE((map), (NULL));
   ASSERT_NE((nonce), (NULL));
   ASSERT_NE((index), (NULL));
   ASSERT_NE((out), (NULL));
   for (i = 0; i < 32; i++) phash[i] = (word8) r32();
   /* inputs: tile kinds 2..7 (uniform twice as often), random nonces
    * (every 16th all zero), indices biased to the edges */
   for (k = 0; k < NRANDOM; k++) {
      kind = 2 + (k % 7);
      if (kind == 8) kind = 2;
      index[k] = rand_index();
      if (kind == 7) {
         /* a real tile belongs to its (20-bit) index */
         index[k] &= PEACHCACHELEN_M1;
         real++;
      }
      if (kind >= 3 && kind <= 6) low++;
      if (index[k] <= 1 || index[k] >= PEACHCACHELEN_M1) edge++;
      make_tile(&map[k * TILE4], kind, index[k], phash);
      for (i = 0; i < 8; i++) nonce[(k * 8) + i] = (k % 16) ? r32() : 0;
   }
   memset(out, 0xA5, (size_t) NRANDOM * 8 * sizeof(word32));
   CUDA_KERNEL(k_md2, (NRANDOM + 127) / 128, 128, 0, NULL)(map, nonce,
      index, out, NRANDOM);
   ASSERT_EQ((cudaGetLastError()), (cudaSuccess));
   t1 = now_s();
   /* reference */
   for (k = 0; k < NRANDOM; k++) {
      make_seed(seed, &nonce[k * 8], index[k], &map[k * TILE4]);
      ref_md2(seed, ref);
      ASSERT_CMP(&out[k * 8], ref, 32);
      /* reference algorithm selection of this seed picks MD2: the
       * consensus nighthash and jump agree with the seed view */
      if ((peach_dflops(seed, PEACHJUMPLEN, index[k], 0) & 7) == 6) {
         peach_nighthash(seed, PEACHJUMPLEN, index[k], 0, nh);
         ASSERT_CMP(&out[k * 8], nh, 32);
         next = index[k];
         peach_jump(&next, (const word8 *) &nonce[k * 8],
            (word8 *) &map[k * TILE4]);
         ASSERT_EQ((next), ((out[k * 8] + out[(k * 8) + 1] +
            out[(k * 8) + 2] + out[(k * 8) + 3]) & PEACHCACHELEN_M1));
         natural++;
      }
   }
   printf("(2) random cases: %d (%llu real tiles, %llu low entropy tiles, "
      "%llu edge indices) match crypto-c md2(); %llu select MD2 in the "
      "reference and match peach_nighthash() + peach_jump(); seed view "
      "%.2fs, reference %.2fs\n", NRANDOM, (unsigned long long) real,
      (unsigned long long) low, (unsigned long long) edge,
      (unsigned long long) natural, t1 - t0, now_s() - t1);
   ASSERT_GE((NRANDOM), (3000));
   ASSERT_GT((real), (0));
   ASSERT_GT((natural), (100));
   free(out);
   free(index);
   free(nonce);
   free(map);
}

/* (3) S-box pointer, output zeroing and bounds */
static void check_pointer(void)
{
   uint4 tile[TILE4];
   word32 n[8], out[16], ref[8], m;
   word8 seed[PEACHJUMPLEN], sb[256], tmp;
   int e, i, changed = 0;

   make_tile(tile, 2, 0, NULL);
   for (i = 0; i < 8; i++) n[i] = r32();
   m = r32() & PEACHCACHELEN_M1;
   make_seed(seed, n, m, tile);
   ref_md2(seed, ref);
   /* words 4..7 zeroed, out[8..15] untouched */
   for (i = 0; i < 16; i++) out[i] = OUTGUARD;
   peach_sh_md2(n, m, tile, c_peach_md2_sbox, out);
   ASSERT_CMP(out, ref, 32);
   for (i = 4; i < 8; i++) ASSERT_EQ((out[i]), (0));
   for (i = 8; i < 16; i++) ASSERT_EQ((out[i]), (OUTGUARD));
   /* a copied S-box (as the device __shared__ copy) gives the same digest */
   memcpy(sb, c_peach_md2_sbox, sizeof(sb));
   memset(out, 0, sizeof(out));
   peach_sh_md2(n, m, tile, sb, out);
   ASSERT_CMP(out, ref, 32);
   /* every entry is read through the pointer: swapping entries e and
    * e + 1 (mod 256) of the copy changes the digest */
   for (e = 0; e < 256; e++) {
      tmp = sb[e];
      sb[e] = sb[(e + 1) & 0xFF];
      sb[(e + 1) & 0xFF] = tmp;
      peach_sh_md2(n, m, tile, sb, out);
      changed += (memcmp(out, ref, 32) != 0);
      sb[(e + 1) & 0xFF] = sb[e];
      sb[e] = tmp;
   }
   ASSERT_EQ((memcmp(sb, c_peach_md2_sbox, sizeof(sb))), (0));
   printf("(3) S-box via pointer: copy matches, %d/256 adjacent swaps "
      "change the digest; words 4..7 zero, out[8..15] untouched\n",
      changed);
   ASSERT_EQ((changed), (256));
}

int main(void)
{
   double t0 = now_s();

   check_sbox();
   check_edges();
   check_random();
   check_pointer();
   printf("peach-hashmd2: all checks passed, %.2fs\n", now_s() - t0);

   return 0;
}
