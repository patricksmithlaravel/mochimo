/**
 * @file peach-hash32.c
 * @brief Peach seed-view SHA-1/SHA-256/MD5, trailer and final SHA-256
 * (peach_hash32.cuh) vs the crypto-c CPU hashes and peach.c.
 * @details CPU only (device header via test/_cuda_emu.h). Checks:
 * (1) peach_sh_sha1(), peach_sh_sha256() and peach_sh_md5() vs crypto-c
 *     sha1(), sha256() and md5() of the materialized 1060 byte jump seed
 *     with the peach_nighthash() output post-processing (zero fill), on
 *     random cases and edge cases (tiles all 0x00, all 0xFF and
 *     peach_generate() tiles incl. tile 0; index 0, 1 and 0xFFFFF), and
 *     vs peach_nighthash() itself whenever peach_dflops() selects that
 *     algorithm. Every tile sits against a PROT_NONE guard page (any load
 *     outside the 1024 tile bytes faults) and every output word is
 *     checked (outputs are pre-filled with garbage);
 * (2) peach_sha256_trailer() and the batch-prefix variant vs sha256(bt,
 *     124), midstate from crypto-c sha256_init() / sha256_update(bt, 64);
 *     one precomputed prefix reused across changing nonce suffixes;
 * (3) peach_sha256_final() vs sha256(hash0 || tile);
 * (4) Peach walks (mainnet vectors, then random nonces on their maps):
 *     hash0, every SHA-1/SHA-256/MD5 jump and the final hash vs the
 *     reference; the mainnet vectors reproduce ref_peach_checkhash().
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stddef.h>     /* for offsetof() */
#include <sys/mman.h>   /* for mmap(), mprotect() */
#include <unistd.h>     /* for sysconf() */

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
#include "peach_hash32.cuh"

#define NRANDOM   4000  /* random cases, checks (1) to (3) */
#define NREUSE    1000  /* nonce suffixes sharing one trailer prefix */
#define NWALKS    64    /* random nonce walks per mainnet vector, (4) */
#define NVECTORS  3     /* mainnet vectors (non-pseudo blocks) */
#define NALGO     3     /* algorithms under test */

/* Peach mainnet vectors 2..4 of src/test/peach-vectors.c (Tfile) */
static const word8 Pvector[NVECTORS][sizeof(BTRAILER)] = {
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

/* Known peach_checkhash() results of Pvector */
static const word8 Pexpect[NVECTORS][SHA256LEN] = {
   {
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

/* Nighthash algorithm numbers and names of the functions under test */
static const int Algo[NALGO] = { 2, 3, 7 };
static const char *Name[NALGO] = { "sha1", "sha256", "md5" };

/* deterministic test RNG (SplitMix64) */
static word64 Rs = WORD64_C(0x6A09E667F3BCC908);

/* failed checks (also reported by the exit status, even with NDEBUG) */
static word64 Fails;

/* guarded tile area: PROT_NONE page | GuardLen data bytes | PROT_NONE */
static word8 *Guard;
static size_t GuardLen;

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

static void rand_bytes(void *p, size_t len)
{
   word8 *b = (word8 *) p;
   size_t i;

   for (i = 0; i < len; i++) b[i] = (word8) (r64() >> 56);
}

/* map the guarded tile area (one or more data pages between two
 * inaccessible pages) */
static void guard_init(void)
{
   long pg = sysconf(_SC_PAGESIZE);
   size_t page = pg > 0 ? (size_t) pg : 4096;
   word8 *base;

   GuardLen = ((PEACHTILELEN + page - 1) / page) * page;
   base = (word8 *) mmap(NULL, GuardLen + (2 * page),
      PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
   ASSERT_NE((base), (MAP_FAILED));
   ASSERT_EQ((mprotect(base, page, PROT_NONE)), (0));
   ASSERT_EQ((mprotect(base + page + GuardLen, page, PROT_NONE)), (0));
   Guard = base + page;
}

/* copy a tile to the start (end == 0) or the end (end != 0) of the
 * guarded area, so a load before t[0] or after t[63] faults */
static const uint4 *place_tile(const word32 *tile, int end)
{
   word8 *p = end ? Guard + GuardLen - PEACHTILELEN : Guard;

   memcpy(p, tile, PEACHTILELEN);
   return (const uint4 *) p;
}

/* random tile index, biased to the edges 0, 1 and 0xFFFFF; sometimes
 * any 32-bit word (the functions hash any seed word 8) */
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

/* random nonce: uniform, haiku first half (trigg_generate()) + random
 * second half, all 0x00, all 0xFF */
static void rand_nonce(word32 *n)
{
   switch (r32() % 8) {
      case 0: memset(n, 0, 32); break;
      case 1: memset(n, 0xFF, 32); break;
      case 2: case 3: case 4:
         trigg_generate(n);
         rand_bytes(n + 4, 16);
         break;
      default: rand_bytes(n, 32); break;
   }
}

/* random tile: uniform, biased bytes (0x00/0xFF/0x80 runs), or a real
 * peach_generate() tile of a random index and previous hash */
static void rand_tile(word32 *tile)
{
   static const word8 bias[4] = { 0x00, 0xFF, 0x80, 0x7F };
   word8 phash[SHA256LEN], *b = (word8 *) tile;
   int i;

   switch (r32() % 4) {
      case 0:
         for (i = 0; i < PEACHTILELEN; i++) {
            b[i] = (r32() & 1) ? bias[r32() & 3] : (word8) r32();
         }
         break;
      case 1:
         rand_bytes(phash, sizeof(phash));
         peach_generate(rand_index() & PEACHCACHELEN_M1, phash, b);
         break;
      default: rand_bytes(tile, PEACHTILELEN); break;
   }
}

/* reference digest of algorithm 2 (sha1), 3 (sha256) or 7 (md5) with
 * the zero fill of peach_nighthash() (little-endian host) */
static void ref_digest(int algo, const void *in, size_t len, word32 *out)
{
   memset(out, 0xA5, 32);
   switch (algo) {
      case 2:
         sha1(in, len, out);
         memset((word8 *) out + 20, 0, 12);
         break;
      case 3: sha256(in, len, out); break;
      default:
         md5(in, len, out);
         memset((word8 *) out + 16, 0, 16);
         break;
   }
}

/* function under test number a (0 sha1, 1 sha256, 2 md5) */
static void dev_digest(int a, const word32 *n, word32 m, const uint4 *t4,
   word32 *out)
{
   memset(out, 0x5A, 32);  /* every output word must be written */
   switch (a) {
      case 0: peach_sh_sha1(n, m, t4, out); break;
      case 1: peach_sh_sha256(n, m, t4, out); break;
      default: peach_sh_md5(n, m, t4, out); break;
   }
}

static void print_words(const char *label, const word32 *w)
{
   int i;

   printf("   %s:", label);
   for (i = 0; i < 8; i++) printf(" %08x", (unsigned) w[i]);
   printf("\n");
}

/* one seed case, all algorithms: vs crypto-c and, where peach_dflops()
 * selects the algorithm, vs peach_nighthash(); counts into bad[] */
static void seed_case(const word32 *nonce, word32 m, const word32 *tile,
   int end, word64 *viaNH, word64 *bad)
{
   word8 seed[PEACHJUMPLEN];
   word32 ref[8], out[8], nh[8], sel;
   const uint4 *t4 = place_tile(tile, end);
   int a;

   memcpy(seed, nonce, 32);
   memcpy(seed + 32, &m, 4);
   memcpy(seed + 36, tile, PEACHTILELEN);
   sel = peach_dflops(seed, PEACHJUMPLEN, m, 0) & 7;
   for (a = 0; a < NALGO; a++) {
      ref_digest(Algo[a], seed, PEACHJUMPLEN, ref);
      dev_digest(a, nonce, m, t4, out);
      if (memcmp(out, ref, 32) != 0) {
         if (bad[a]++ == 0) {
            printf("(1) %s mismatch, index 0x%08x:\n", Name[a], (unsigned) m);
            print_words("ref", ref);
            print_words("got", out);
         }
      }
      if (sel == (word32) Algo[a]) {
         memset(nh, 0xC3, sizeof(nh));
         peach_nighthash(seed, PEACHJUMPLEN, m, 0, nh);
         if (memcmp(out, nh, 32) != 0) bad[a]++;
         viaNH[a]++;
      }
   }
}

/* (1) seed-view hashes vs crypto-c and peach_nighthash() */
static void check_seed(void)
{
   static const word32 idx[3] = { 0, 1, PEACHCACHELEN_M1 };
   word32 nonce[8], tile[PEACHTILELEN32];
   word8 phash[SHA256LEN];
   word64 bad[NALGO] = { 0 }, viaNH[NALGO] = { 0 }, cases = 0, edge = 0;
   double t0 = now_s();
   int i, j, k, e, nk, tk;

   /* edge cases: tiles x indices x nonces x placement */
   for (tk = 0; tk < 7; tk++) {
      switch (tk) {
         case 0: memset(tile, 0x00, sizeof(tile)); break;
         case 1: memset(tile, 0xFF, sizeof(tile)); break;
         case 2: case 3: case 4:  /* tile 0 of the mainnet maps */
            peach_generate(0, ((const BTRAILER *) Pvector[tk - 2])->phash,
               (word8 *) tile);
            break;
         case 5:
            peach_generate(PEACHCACHELEN_M1,
               ((const BTRAILER *) Pvector[0])->phash, (word8 *) tile);
            break;
         default:
            memset(phash, 0, sizeof(phash));
            peach_generate(1, phash, (word8 *) tile);
            break;
      }
      for (j = 0; j < 3; j++) {
         for (nk = 0; nk < 4; nk++) {
            switch (nk) {
               case 0: memset(nonce, 0x00, sizeof(nonce)); break;
               case 1: memset(nonce, 0xFF, sizeof(nonce)); break;
               case 2:
                  memcpy(nonce, ((const BTRAILER *) Pvector[2])->nonce, 32);
                  break;
               default: rand_bytes(nonce, sizeof(nonce)); break;
            }
            for (e = 0; e < 2; e++) {
               seed_case(nonce, idx[j], tile, e, viaNH, bad);
               edge++;
            }
         }
      }
   }
   /* random cases */
   for (i = 0; i < NRANDOM; i++) {
      rand_nonce(nonce);
      rand_tile(tile);
      seed_case(nonce, rand_index(), tile, i & 1, viaNH, bad);
      cases++;
   }
   /* single byte sensitivity at every seed byte position: flip one byte
    * of a fixed random seed (nonce, index or tile byte k) */
   rand_bytes(nonce, sizeof(nonce));
   rand_bytes(tile, sizeof(tile));
   for (k = 0; k < PEACHJUMPLEN; k++) {
      word32 n2[8], t2[PEACHTILELEN32], m2 = WORD32_C(0x000ABCDE);
      word8 x = (word8) ((r32() % 255) + 1);

      memcpy(n2, nonce, sizeof(n2));
      memcpy(t2, tile, sizeof(t2));
      if (k < 32) ((word8 *) n2)[k] ^= x;
      else if (k < 36) m2 ^= (word32) x << (8 * (k - 32));
      else ((word8 *) t2)[k - 36] ^= x;
      seed_case(n2, m2, t2, k & 1, viaNH, bad);
      cases++;
   }
   printf("(1) seed view: %llu random + %d byte-flip + %llu edge cases x "
      "%d algos; via peach_nighthash() sha1 %llu, sha256 %llu, md5 %llu; "
      "mismatches %llu/%llu/%llu, %.2fs\n",
      (unsigned long long) (cases - PEACHJUMPLEN), PEACHJUMPLEN,
      (unsigned long long) edge, NALGO, (unsigned long long) viaNH[0],
      (unsigned long long) viaNH[1], (unsigned long long) viaNH[2],
      (unsigned long long) bad[0], (unsigned long long) bad[1],
      (unsigned long long) bad[2], now_s() - t0);
   ASSERT_GE((cases), (NRANDOM));
   for (i = 0; i < NALGO; i++) {
      Fails += bad[i] + (viaNH[i] == 0);
      ASSERT_EQ((bad[i]), (0));
      ASSERT_GT((viaNH[i]), (0));
   }
}

/* reference inputs of peach_sha256_trailer() from a block trailer */
static void trailer_inputs(const BTRAILER *bt, word32 *mid, word32 *tail,
   word32 *n)
{
   SHA256_CTX ctx;

   sha256_init(&ctx);
   sha256_update(&ctx, bt, 64);
   memcpy(mid, ctx.state, 32);
   memcpy(tail, ((const word8 *) bt) + 64, 28);
   memcpy(n, ((const word8 *) bt) + 92, 32);
}

/* (2) trailer SHA-256 from midstate */
static void check_trailer(void)
{
   BTRAILER bt;
   word32 mid[8], tail[7], n[8], h[8], hp[8], pre[8], ref[8];
   word64 bad = 0, badpre = 0, badreuse = 0;
   int i;

   ASSERT_EQ((offsetof(BTRAILER, nonce)), (92));
   for (i = 0; i < NRANDOM + NVECTORS + 2; i++) {
      if (i < NVECTORS) memcpy(&bt, Pvector[i], sizeof(bt));
      else if (i == NVECTORS) memset(&bt, 0x00, sizeof(bt));
      else if (i == NVECTORS + 1) memset(&bt, 0xFF, sizeof(bt));
      else rand_bytes(&bt, sizeof(bt));
      trailer_inputs(&bt, mid, tail, n);
      memset(h, 0x5A, sizeof(h));
      memset(hp, 0x5A, sizeof(hp));
      peach_sha256_trailer(mid, tail, n, h);
      peach_sha256_trailer_prefix(mid, tail, n, pre);
      peach_sha256_trailer_pre(mid, tail, n, pre, hp);
      sha256(&bt, 124, ref);
      if (memcmp(h, ref, 32) != 0) {
         if (bad++ == 0) {
            printf("(2) trailer mismatch, case %d:\n", i);
            print_words("ref", ref);
            print_words("got", h);
         }
      }
      if (memcmp(hp, ref, 32) != 0) {
         if (badpre++ == 0) {
            printf("(2) trailer prefix mismatch, case %d:\n", i);
            print_words("ref", ref);
            print_words("got", hp);
         }
      }
   }
   /* The first 108 trailer bytes stay fixed; only nonce words 4..7 vary. */
   rand_bytes(&bt, sizeof(bt));
   trailer_inputs(&bt, mid, tail, n);
   peach_sha256_trailer_prefix(mid, tail, n, pre);
   for (i = 0; i < NREUSE; i++) {
      rand_bytes(bt.nonce + 16, 16);
      memcpy(n + 4, bt.nonce + 16, 16);
      memset(h, 0x5A, sizeof(h));
      memset(hp, 0x5A, sizeof(hp));
      peach_sha256_trailer(mid, tail, n, h);
      peach_sha256_trailer_pre(mid, tail, n, pre, hp);
      sha256(&bt, 124, ref);
      if (memcmp(h, ref, 32) != 0 || memcmp(hp, ref, 32) != 0) {
         if (badreuse++ == 0) {
            printf("(2) reused trailer prefix mismatch, case %d:\n", i);
            print_words("ref", ref);
            print_words("original", h);
            print_words("prefix", hp);
         }
      }
   }
   printf("(2) trailer: %d cases, mismatches %llu, prefix mismatches %llu; "
      "%d suffixes with one prefix, mismatches %llu\n", NRANDOM + NVECTORS + 2,
      (unsigned long long) bad, (unsigned long long) badpre, NREUSE,
      (unsigned long long) badreuse);
   Fails += bad + badpre + badreuse;
   ASSERT_EQ((bad), (0));
   ASSERT_EQ((badpre), (0));
   ASSERT_EQ((badreuse), (0));
}

/* (3) final SHA-256 of hash0 || tile */
static void check_final(void)
{
   SHA256_CTX ctx;
   word32 h0[8], tile[PEACHTILELEN32], out[8], ref[8];
   word64 bad = 0;
   int i;

   for (i = 0; i < NRANDOM + 4; i++) {
      rand_bytes(h0, sizeof(h0));
      if (i == 0 || i == 1) memset(h0, i ? 0xFF : 0x00, sizeof(h0));
      if (i == 0 || i == 2) memset(tile, 0x00, sizeof(tile));
      else if (i == 1 || i == 3) memset(tile, 0xFF, sizeof(tile));
      else rand_tile(tile);
      sha256_init(&ctx);
      sha256_update(&ctx, h0, 32);
      sha256_update(&ctx, tile, PEACHTILELEN);
      sha256_final(&ctx, ref);
      memset(out, 0x5A, sizeof(out));
      peach_sha256_final(h0, place_tile(tile, i & 1), out);
      if (memcmp(out, ref, 32) != 0) {
         if (bad++ == 0) {
            printf("(3) final mismatch, case %d:\n", i);
            print_words("ref", ref);
            print_words("got", out);
         }
      }
   }
   printf("(3) final: %d cases, mismatches %llu\n", NRANDOM + 4,
      (unsigned long long) bad);
   Fails += bad;
   ASSERT_EQ((bad), (0));
}

/* reference Peach walk (as peach_checkhash() without the syntax gate)
 * with hash0, every jump of a function under test and the final hash
 * recomputed by peach_hash32.cuh; returns mismatches */
static int walk(const BTRAILER *bt, word32 *final, word64 *jumps)
{
   SHA256_CTX ctx;
   word32 mid[8], tail[7], n[8], h0[8], tile[PEACHTILELEN32];
   word32 dh[8], out[8], ref[8], mario, sel;
   word8 hash[SHA256LEN], seed[PEACHJUMPLEN];
   int i, a, fails = 0;

   trailer_inputs(bt, mid, tail, n);
   memset(h0, 0x5A, sizeof(h0));
   peach_sha256_trailer(mid, tail, n, h0);
   sha256(bt, 124, hash);
   fails += (memcmp(h0, hash, SHA256LEN) != 0);
   for (mario = hash[0], i = 1; i < SHA256LEN; i++) mario *= hash[i];
   mario &= PEACHCACHELEN_M1;
   for (i = 0; i < PEACHROUNDS; i++) {
      peach_generate(mario, bt->phash, (word8 *) tile);
      memcpy(seed, bt->nonce, 32);
      memcpy(seed + 32, &mario, 4);
      memcpy(seed + 36, tile, PEACHTILELEN);
      sel = peach_dflops(seed, PEACHJUMPLEN, mario, 0) & 7;
      peach_nighthash(seed, PEACHJUMPLEN, mario, 0, dh);
      for (a = 0; a < NALGO; a++) {
         if (sel != (word32) Algo[a]) continue;
         dev_digest(a, n, mario, place_tile(tile, i & 1), out);
         fails += (memcmp(out, dh, 32) != 0);
         jumps[a]++;
      }
      mario = (dh[0] + dh[1] + dh[2] + dh[3] + dh[4] + dh[5] + dh[6] +
         dh[7]) & PEACHCACHELEN_M1;
   }
   peach_generate(mario, bt->phash, (word8 *) tile);
   memset(final, 0x5A, 32);
   peach_sha256_final(h0, place_tile(tile, 1), final);
   sha256_init(&ctx);
   sha256_update(&ctx, hash, SHA256LEN);
   sha256_update(&ctx, tile, PEACHTILELEN);
   sha256_final(&ctx, ref);
   fails += (memcmp(final, ref, 32) != 0);
   return fails;
}

/* (4) mainnet vectors and random nonce walks */
static void check_walks(void)
{
   BTRAILER bt;
   word32 final[8];
   word8 out[SHA256LEN];
   word64 jumps[NALGO] = { 0 }, bad = 0, walks = 0;
   double t0 = now_s();
   int v, i;

   for (v = 0; v < NVECTORS; v++) {
      memcpy(&bt, Pvector[v], sizeof(bt));
      bad += walk(&bt, final, jumps);
      walks++;
      bad += (memcmp(final, Pexpect[v], SHA256LEN) != 0);
      ASSERT_CMP(final, Pexpect[v], SHA256LEN);
      bad += (ref_peach_checkhash(&bt, bt.difficulty[0], out) != VEOK);
      bad += (memcmp(final, out, SHA256LEN) != 0);
      ASSERT_CMP(final, out, SHA256LEN);
      for (i = 0; i < NWALKS; i++) {
         trigg_generate(bt.nonce);
         rand_bytes(bt.nonce + 16, 16);
         bad += walk(&bt, final, jumps);
         walks++;
      }
   }
   printf("(4) walks: %llu (incl. %d mainnet vectors), jumps checked: "
      "sha1 %llu, sha256 %llu, md5 %llu; mismatches %llu, %.2fs\n",
      (unsigned long long) walks, NVECTORS, (unsigned long long) jumps[0],
      (unsigned long long) jumps[1], (unsigned long long) jumps[2],
      (unsigned long long) bad, now_s() - t0);
   Fails += bad;
   ASSERT_EQ((bad), (0));
   for (i = 0; i < NALGO; i++) {
      Fails += (jumps[i] == 0);
      ASSERT_GT((jumps[i]), (0));
   }
}

int main(void)
{
   double t0 = now_s();

   srand16(0x1234567, 0x89abcdef, 0x2468ace0);
   guard_init();
   check_seed();
   check_trailer();
   check_final();
   check_walks();
   if (Fails) {
      printf("peach-hash32: %llu checks FAILED\n", (unsigned long long) Fails);
      return 1;
   }
   printf("peach-hash32: all checks passed, %.2fs\n", now_s() - t0);

   return 0;
}
