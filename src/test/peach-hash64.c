/**
 * @file peach-hash64.c
 * @brief Seed-view jump hashes of the 64-bit lane family
 * (peach_hash64.cuh) vs the crypto-c digests used by peach_nighthash().
 * @details CPU only (device header via test/_cuda_emu.h). For algo 0
 * (Blake2b-256, key 32 zero bytes), 1 (Blake2b-256, key 64 x 0x01),
 * 4 (SHA3-256) and 5 (Keccak-256), peach_sh_*(nonce, m, tile) must equal
 * the crypto-c digest of the materialized 1060-byte jump seed, called
 * exactly as peach_nighthash() calls it. Checks:
 * (0) constants: Blake2b IVs and fast-forward (post key block) states vs
 *     crypto-c, Keccak-f round constants vs the reference LFSR;
 * (1) edge tiles (all 0x00, all 0xFF, real peach_generate() tiles incl.
 *     tiles 0, 1 and 0xFFFFF) x edge nonces x index {0, 1, 0xFFFFF,
 *     0xFFFFFFFF, own index};
 * (2) single-byte seed sweep: each of the 1060 seed bytes set alone
 *     (zero background) and cleared alone (0xFF background);
 * (3) random cases (>= 3000 per algo); when the reference algorithm
 *     selection picks the algo under test, also vs peach_nighthash();
 * (4) an emulated kernel launch over 128 threads.
 * Every call must load each of the 64 tile vectors exactly once, with an
 * aligned 128-bit load, and nothing else (instrumented __ldg()); nonce and
 * tile live in exact-size heap buffers (out-of-bounds reads show up under
 * AddressSanitizer).
*/

#include <stdint.h>
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

/* load tracking: the device header reads the tile ONLY through
 * PEACH_LDG128() -> __ldg(const uint4 *); count every such load */
static const uint4 *Ldg_base;    /* tracked range start */
static int *Ldg_count;           /* loads per tracked vector */
static int Ldg_nvec;             /* tracked vectors */
static long Ldg_other;           /* loads outside the tracked range */
static long Ldg_misaligned;      /* loads not 16-byte aligned */

static const uint4 *ldg_track(const uint4 *p)
{
   if ((((uintptr_t) p) & 15) != 0) Ldg_misaligned++;
   if (Ldg_base != NULL && p >= Ldg_base && p < Ldg_base + Ldg_nvec) {
      Ldg_count[p - Ldg_base]++;
   } else Ldg_other++;
   return p;
}

#undef __ldg
#define __ldg(ptr)  (*ldg_track(ptr))

#include "peach_hash64.cuh"

#define NRANDOM   4000     /* random cases per algo, check (3) */
#define NPOOL     12       /* real (generated) tiles in the pool */
#define NKTHREADS 128      /* emulated kernel threads, check (4) */
#define NALGO     4

/* algos under test */
static const int Algo[NALGO] = { 0, 1, 4, 5 };

/* deterministic test RNG (SplitMix64) */
static word64 Rs = WORD64_C(0x6A09E667BB67AE85);

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

/* digest of a materialized seed, exactly as peach_nighthash() calls the
 * crypto-c functions (no post-processing for these algos) */
static void ref_hash(int algo, const word8 *seed, word32 *out)
{
   static const word64 key32B[4] = { 0, 0, 0, 0 };
   static const word64 key64B[8] = {
      WORD64_C(0x0101010101010101), WORD64_C(0x0101010101010101),
      WORD64_C(0x0101010101010101), WORD64_C(0x0101010101010101),
      WORD64_C(0x0101010101010101), WORD64_C(0x0101010101010101),
      WORD64_C(0x0101010101010101), WORD64_C(0x0101010101010101),
   };

   switch (algo) {
      case 0:
         blake2b(seed, PEACHJUMPLEN, key32B, 32, out, BLAKE2BLEN256);
         break;
      case 1:
         blake2b(seed, PEACHJUMPLEN, key64B, 64, out, BLAKE2BLEN256);
         break;
      case 4: sha3(seed, PEACHJUMPLEN, out, SHA3LEN256); break;
      default: keccak(seed, PEACHJUMPLEN, out, KECCAKLEN256); break;
   }
}

/* seed-view hash under test */
static void dut_hash(int algo, const word32 *n, word32 index,
   const uint4 *tile, word32 *out)
{
   switch (algo) {
      case 0: peach_sh_blake2b32(n, index, tile, out); break;
      case 1: peach_sh_blake2b64(n, index, tile, out); break;
      case 4: peach_sh_sha3(n, index, tile, out); break;
      default: peach_sh_keccak(n, index, tile, out); break;
   }
}

/* materialize the jump seed: nonce || index || tile */
static void make_seed(word8 *seed, const word32 *n, word32 index,
   const uint4 *tile)
{
   memcpy(seed, n, 32);
   memcpy(seed + 32, &index, 4);
   memcpy(seed + 36, tile, PEACHTILELEN);
}

/* exact-size heap buffers (nonce, tile) for every single-call check */
static word32 *Nb;
static uint4 *Tb;

/* per-check counters */
static long Cases, Mism, Loadbad;

/* one case: nonce Nb, tile Tb, @a index, @a algo vs the reference */
static void check_case(int algo, word32 index, const char *what)
{
   word8 seed[PEACHJUMPLEN];
   word32 ref[8], out[8];
   int i, bad = 0;

   make_seed(seed, Nb, index, Tb);
   ref_hash(algo, seed, ref);
   memset(out, 0xA5, sizeof(out));
   Ldg_base = Tb;
   Ldg_nvec = PEACHTILELEN / 16;
   memset(Ldg_count, 0, sizeof(int) * (size_t) Ldg_nvec);
   Ldg_other = Ldg_misaligned = 0;
   dut_hash(algo, Nb, index, Tb, out);
   for (i = 0; i < Ldg_nvec; i++) bad |= (Ldg_count[i] != 1);
   if (bad || Ldg_other || Ldg_misaligned) Loadbad++;
   if (memcmp(out, ref, sizeof(out)) != 0) {
      if (Mism < 8) {
         printf("MISMATCH %s algo %d index 0x%08x:\n   dut", what, algo,
            (unsigned) index);
         for (i = 0; i < 8; i++) printf(" %08x", (unsigned) out[i]);
         printf("\n   ref");
         for (i = 0; i < 8; i++) printf(" %08x", (unsigned) ref[i]);
         printf("\n");
      }
      Mism++;
   }
   Cases++;
}

/* (0) constants */
static void check_constants(void)
{
   static const word64 key32B[4] = { 0, 0, 0, 0 };
   static const word64 iv[8] = {
      PEACH_B2B_IV0, PEACH_B2B_IV1, PEACH_B2B_IV2, PEACH_B2B_IV3,
      PEACH_B2B_IV4, PEACH_B2B_IV5, PEACH_B2B_IV6, PEACH_B2B_IV7
   };
   static const word64 civ[8] = {
      BLAKE2B_IV0, BLAKE2B_IV1, BLAKE2B_IV2, BLAKE2B_IV3,
      BLAKE2B_IV4, BLAKE2B_IV5, BLAKE2B_IV6, BLAKE2B_IV7
   };
   static const word64 ff32[8] = {
      PEACH_B2B_KEY32_H0, PEACH_B2B_KEY32_H1, PEACH_B2B_KEY32_H2,
      PEACH_B2B_KEY32_H3, PEACH_B2B_KEY32_H4, PEACH_B2B_KEY32_H5,
      PEACH_B2B_KEY32_H6, PEACH_B2B_KEY32_H7
   };
   static const word64 ff64[8] = {
      PEACH_B2B_KEY64_H0, PEACH_B2B_KEY64_H1, PEACH_B2B_KEY64_H2,
      PEACH_B2B_KEY64_H3, PEACH_B2B_KEY64_H4, PEACH_B2B_KEY64_H5,
      PEACH_B2B_KEY64_H6, PEACH_B2B_KEY64_H7
   };
   BLAKE2B_CTX ctx;
   word64 key64B[8], rc;
   word8 one = 0, lfsr = 1;
   int i, j, bit;

   for (i = 0; i < 8; i++) {
      ASSERT_EQ((iv[i]), (civ[i]));
      key64B[i] = WORD64_C(0x0101010101010101);
   }
   /* crypto-c compresses the key block when the first message byte
    * arrives: the state then is the fast-forward state */
   ASSERT_EQ((blake2b_init(&ctx, key32B, 32, BLAKE2BLEN256)), (0));
   blake2b_update(&ctx, &one, 1);
   for (i = 0; i < 8; i++) ASSERT_EQ((ctx.h[i]), (ff32[i]));
   ASSERT_EQ((ctx.t[0]), (128));
   ASSERT_EQ((blake2b_init(&ctx, key64B, 64, BLAKE2BLEN256)), (0));
   blake2b_update(&ctx, &one, 1);
   for (i = 0; i < 8; i++) ASSERT_EQ((ctx.h[i]), (ff64[i]));
   /* Keccak-f[1600] round constants: LFSR x^8 + x^6 + x^5 + x^4 + 1 */
   for (i = 0; i < 24; i++) {
      for (rc = 0, j = 0; j < 7; j++) {
         bit = lfsr & 1;
         lfsr = (word8) ((lfsr & 0x80) ? (lfsr << 1) ^ 0x71 : lfsr << 1);
         if (bit) rc ^= WORD64_C(1) << ((1 << j) - 1);
      }
      ASSERT_EQ((c_peach_keccakf_rndc[i]), (rc));
   }
   printf("(0) constants: Blake2b IVs, fast-forward states (32 zero byte"
      " and 64 x 0x01 keys), Keccak-f round constants OK\n");
}

/* (1) edge tiles x edge nonces x edge indices */
static void check_edges(void)
{
   word8 phash[32], gen[PEACHTILELEN];
   word32 gidx[6], idx[5];
   double t0 = now_s();
   int a, g, nk, x, i, ntile;

   for (i = 0; i < 32; i++) phash[i] = (word8) r32();
   gidx[0] = 0; gidx[1] = 1; gidx[2] = PEACHCACHELEN_M1;
   gidx[3] = r32() & PEACHCACHELEN_M1;
   gidx[4] = r32() & PEACHCACHELEN_M1;
   gidx[5] = r32() & PEACHCACHELEN_M1;
   ntile = 2 + 6;
   Cases = Mism = Loadbad = 0;
   for (g = 0; g < ntile; g++) {
      if (g == 0) memset(Tb, 0x00, PEACHTILELEN);
      else if (g == 1) memset(Tb, 0xFF, PEACHTILELEN);
      else {
         peach_generate(gidx[g - 2], phash, gen);
         memcpy(Tb, gen, PEACHTILELEN);
      }
      idx[0] = 0; idx[1] = 1; idx[2] = PEACHCACHELEN_M1;
      idx[3] = WORD32_C(0xFFFFFFFF);
      idx[4] = (g >= 2) ? gidx[g - 2] : r32();
      for (nk = 0; nk < 3; nk++) {
         for (i = 0; i < 8; i++) {
            Nb[i] = nk == 0 ? 0 : nk == 1 ? WORD32_C(0xFFFFFFFF) : r32();
         }
         for (x = 0; x < 5; x++) {
            for (a = 0; a < NALGO; a++) check_case(Algo[a], idx[x], "edge");
         }
      }
   }
   printf("(1) edge tiles (0x00, 0xFF, generated 0, 1, 0xFFFFF + 3) x 3"
      " nonces x 5 indices x 4 algos: cases=%ld MISMATCHES=%ld"
      " bad-load-calls=%ld, %.2fs\n", Cases, Mism, Loadbad, now_s() - t0);
   ASSERT_EQ((Mism), (0));
   ASSERT_EQ((Loadbad), (0));
}

/* (2) every seed byte alone */
static void check_sweep(void)
{
   word8 seed[PEACHJUMPLEN];
   word32 index;
   double t0 = now_s();
   int a, pos, bg;

   Cases = Mism = Loadbad = 0;
   for (bg = 0; bg < 2; bg++) {
      for (pos = 0; pos < PEACHJUMPLEN; pos++) {
         memset(seed, bg ? 0xFF : 0x00, sizeof(seed));
         seed[pos] = (word8) (bg ? 0x00 : ((r32() & 0xFF) | 1));
         memcpy(Nb, seed, 32);
         memcpy(&index, seed + 32, 4);
         memcpy(Tb, seed + 36, PEACHTILELEN);
         for (a = 0; a < NALGO; a++) check_case(Algo[a], index, "sweep");
      }
   }
   printf("(2) single-byte sweep, 1060 positions x {set on 0x00, clear on"
      " 0xFF} x 4 algos: cases=%ld MISMATCHES=%ld bad-load-calls=%ld,"
      " %.2fs\n", Cases, Mism, Loadbad, now_s() - t0);
   ASSERT_EQ((Mism), (0));
   ASSERT_EQ((Loadbad), (0));
}

/* random tile index, biased to the edges */
static word32 rand_index(void)
{
   switch (r32() % 16) {
      case 0: return 0;
      case 1: return 1;
      case 2: return PEACHCACHELEN_M1;
      case 3: return r32();
      default: return r32() & PEACHCACHELEN_M1;
   }
}

/* (3) random cases + direct peach_nighthash() where it selects algo */
static void check_random(void)
{
   static word8 pool[NPOOL][PEACHTILELEN];
   word8 phash[32], seed[PEACHJUMPLEN];
   word32 out[8], nh[8], index, w, sel;
   long nhcases[NALGO], nhmism = 0;
   double t0 = now_s();
   int a, c, i, kind;

   for (i = 0; i < 32; i++) phash[i] = (word8) r32();
   /* tiles 0, 1, 0xFFFFF and random tiles of the map */
   for (i = 0; i < NPOOL; i++) {
      index = (i < 2) ? (word32) i : (i == 2) ? PEACHCACHELEN_M1 :
         (r32() & PEACHCACHELEN_M1);
      peach_generate(index, phash, pool[i]);
   }
   for (a = 0; a < NALGO; a++) nhcases[a] = 0;
   Cases = Mism = Loadbad = 0;
   for (c = 0; c < NRANDOM; c++) {
      for (i = 0; i < 8; i++) Nb[i] = r32();
      index = rand_index();
      kind = (int) (r32() % 4);
      if (kind == 0) memcpy(Tb, pool[r32() % NPOOL], PEACHTILELEN);
      else for (i = 0; i < PEACHTILELEN32; i++) {
         w = r32();
         /* kind 1: runs of 0x00 / 0xFF bytes mixed in */
         if (kind == 1 && (r32() & 1)) {
            w = (r32() & 1) ? 0 : WORD32_C(0xFFFFFFFF);
         }
         memcpy((word8 *) Tb + (4 * i), &w, 4);
      }
      for (a = 0; a < NALGO; a++) check_case(Algo[a], index, "random");
      /* the consensus path itself, when it selects one of these algos */
      make_seed(seed, Nb, index, Tb);
      sel = peach_dflops(seed, PEACHJUMPLEN, index, 0) & 7;
      for (a = 0; a < NALGO; a++) {
         if ((word32) Algo[a] != sel) continue;
         memset(nh, 0x5A, sizeof(nh));
         peach_nighthash(seed, PEACHJUMPLEN, index, 0, nh);
         dut_hash(Algo[a], Nb, index, Tb, out);
         nhmism += (memcmp(out, nh, sizeof(out)) != 0);
         nhcases[a]++;
      }
   }
   printf("(3) random cases, %d per algo (generated/random/0x00-0xFF-run"
      " tiles, edge-biased index): cases=%ld MISMATCHES=%ld"
      " bad-load-calls=%ld; vs peach_nighthash() where selected:"
      " algo0=%ld algo1=%ld algo4=%ld algo5=%ld MISMATCHES=%ld, %.2fs\n",
      NRANDOM, Cases, Mism, Loadbad, nhcases[0], nhcases[1], nhcases[2],
      nhcases[3], nhmism, now_s() - t0);
   ASSERT_EQ((Mism), (0));
   ASSERT_EQ((Loadbad), (0));
   ASSERT_EQ((nhmism), (0));
   for (a = 0; a < NALGO; a++) ASSERT_GT((nhcases[a]), (0));
}

/* emulated kernel: one seed-view hash per thread */
static void k_hash64(int algo, const word32 *d_nonce, const word32 *d_index,
   const uint4 *d_tiles, word32 *d_out)
{
   word32 tid = (blockIdx.x * blockDim.x) + threadIdx.x;
   word32 n[8], out[8];
   int i;

   for (i = 0; i < 8; i++) n[i] = d_nonce[(tid * 8) + i];
   switch (algo) {
      case 0: peach_sh_blake2b32(n, d_index[tid], &d_tiles[tid * 64], out);
         break;
      case 1: peach_sh_blake2b64(n, d_index[tid], &d_tiles[tid * 64], out);
         break;
      case 4: peach_sh_sha3(n, d_index[tid], &d_tiles[tid * 64], out);
         break;
      default: peach_sh_keccak(n, d_index[tid], &d_tiles[tid * 64], out);
         break;
   }
   for (i = 0; i < 8; i++) d_out[(tid * 8) + i] = out[i];
}

/* (4) emulated kernel launch */
static void check_kernel(void)
{
   word8 seed[PEACHJUMPLEN];
   word32 *nonce, *index, *out, ref[8];
   uint4 *tiles;
   double t0 = now_s();
   int a, i, t, bad;

   nonce = (word32 *) malloc(sizeof(word32) * 8 * NKTHREADS);
   index = (word32 *) malloc(sizeof(word32) * NKTHREADS);
   out = (word32 *) malloc(sizeof(word32) * 8 * NKTHREADS);
   tiles = (uint4 *) aligned_alloc(16, (size_t) PEACHTILELEN * NKTHREADS);
   ASSERT_NE((nonce), (NULL));
   ASSERT_NE((index), (NULL));
   ASSERT_NE((out), (NULL));
   ASSERT_NE((tiles), (NULL));
   for (i = 0; i < 8 * NKTHREADS; i++) nonce[i] = r32();
   for (i = 0; i < NKTHREADS; i++) index[i] = rand_index();
   for (i = 0; i < PEACHTILELEN32 * NKTHREADS; i++) {
      ((word32 *) tiles)[i] = r32();
   }
   Cases = Mism = Loadbad = 0;
   free(Ldg_count);
   Ldg_count = (int *) malloc(sizeof(int) * 64 * NKTHREADS);
   ASSERT_NE((Ldg_count), (NULL));
   for (a = 0; a < NALGO; a++) {
      Ldg_base = tiles;
      Ldg_nvec = 64 * NKTHREADS;
      memset(Ldg_count, 0, sizeof(int) * (size_t) Ldg_nvec);
      Ldg_other = Ldg_misaligned = 0;
      memset(out, 0xA5, sizeof(word32) * 8 * NKTHREADS);
      CUDA_KERNEL(k_hash64, NKTHREADS / 32, 32, 0, NULL)(Algo[a], nonce,
         index, tiles, out);
      ASSERT_EQ((cudaGetLastError()), (cudaSuccess));
      for (bad = 0, i = 0; i < Ldg_nvec; i++) bad |= (Ldg_count[i] != 1);
      if (bad || Ldg_other || Ldg_misaligned) Loadbad++;
      for (t = 0; t < NKTHREADS; t++) {
         make_seed(seed, &nonce[t * 8], index[t], &tiles[t * 64]);
         ref_hash(Algo[a], seed, ref);
         Mism += (memcmp(&out[t * 8], ref, sizeof(ref)) != 0);
         Cases++;
      }
   }
   printf("(4) emulated kernel, %d threads x 4 algos: cases=%ld"
      " MISMATCHES=%ld bad-load-launches=%ld, %.2fs\n", NKTHREADS, Cases,
      Mism, Loadbad, now_s() - t0);
   ASSERT_EQ((Mism), (0));
   ASSERT_EQ((Loadbad), (0));
   free(tiles);
   free(out);
   free(index);
   free(nonce);
}

int main(void)
{
   double t0 = now_s();

   Nb = (word32 *) malloc(32);
   Tb = (uint4 *) aligned_alloc(16, PEACHTILELEN);
   Ldg_count = (int *) malloc(sizeof(int) * 64);
   ASSERT_NE((Nb), (NULL));
   ASSERT_NE((Tb), (NULL));
   ASSERT_NE((Ldg_count), (NULL));
   check_constants();
   check_edges();
   check_sweep();
   check_random();
   check_kernel();
   free(Ldg_count);
   free(Tb);
   free(Nb);
   printf("peach-hash64: all checks passed, %.2fs\n", now_s() - t0);

   return 0;
}
