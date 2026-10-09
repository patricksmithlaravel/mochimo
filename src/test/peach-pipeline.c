/**
 * @file peach-pipeline.c
 * @brief Peach pipeline kernels (peach_pipeline.cuh) end to end vs the
 * consensus reference walk of peach.c.
 * @details CPU only: the kernels run one thread at a time through
 * test/_cuda_emu.h, launched by peach_pipe_enqueue() itself (or, for the
 * synthetic and small capacity runs, by the same per-kernel launches).
 * The Peach map is a sparse 1 GiB anonymous mapping between two
 * PROT_NONE guard pages (every other device buffer also ends at a guard
 * page): only the tiles a reference walk visits are generated, together
 * with their transition table entries (computed by the transitions
 * kernel). Checks:
 * (0) helpers: peach_pipe_diff_ok() vs trigg_eval() for every difficulty
 *     and leading zero count, peach_pipe_frame() vs the official nonce
 *     frame for every table index, skip mask packing, enqueue argument
 *     validation (nothing enqueued), the self-test kernel vs reference
 *     dflops steps, the transitions kernel vs peach_transition_tile()
 *     incl. adversarial words and range clamping at the end of the map;
 * (1) realistic batch, pass 1 (init kernel only): nonce words 4..7 (RNG
 *     state and official frame), hash0, mario0 and P vs an independent
 *     computation; then a reference walk of every slot (peach_generate(),
 *     peach_dflops(), peach_jump() of peach.c) fills the visited tiles;
 * (2) realistic batch, pass 2 (peach_pipe_enqueue(), RNG reset: same
 *     nonces) for skip masks 0x40 x 8, all zero and mixed per round:
 *     every slot's trace (tiles, algorithms, drop round, final hash),
 *     packed states, queue counters, final queue, completed/dropped/overflow/
 *     anomaly counters, epoch, canary and solve vs the reference walk;
 *     solve, canary and a sample of slots vs ref_peach_checkhash(); the
 *     same batch with other launch shapes gives identical traces; round
 *     0 concentrates on two algorithms {x, x + 4} (> 90%);
 * (3) synthetic batch: slot states and round-0 queues written by the
 *     test with random nonzero tile indices (T and jumps off tile 0),
 *     then the hash and final kernels, vs the reference walk;
 * (4) small queue capacity (cap < nslots, full evaluation): appends past
 *     cap in a round-0 queue and entries past cap in the packed queues of
 *     a later round are counted, lost slots are marked, consumers stop at
 *     cap, everything else stays exact;
 * (5) tile sort kernels (scan, scatter) on original keys and packed input
 *     with hot/edge buckets, dead entries and several launch shapes: counters,
 *     queue (packed at its offset) a tile ordered set of its slots'
 *     entries, cursors at the bucket ends; with a small capacity, the
 *     first-sort overflow, bounded later input and original IDs marked lost
 *     by an injected cursor overflow; empty and all-dead packed queues.
 * Every batch check (2)-(4) also checks the final queue's entries (tile
 * order and states of the completed slots); the
 * per-kernel runs of (2), (3) and (4) check the tile ordered queues of
 * every round right after their sort.
 * OpenMP is used for reference work only (never around emulated kernels).
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
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
#include "peach_pipeline.cuh"

#define NSLOTS    4096              /* realistic batch slots */
#define CAP       (NSLOTS + 256)    /* queue capacity (> NSLOTS) */
#define NSYN      1536              /* synthetic batch slots */
#define SMALLCAP  1024              /* small capacity run */
#define NSAMPLE   8                 /* ref_peach_checkhash() every 8th */
#define NSELF     4096              /* self-test entries */
#define NCFG      3                 /* realistic skip configurations */

/* Peach mainnet vector 4 (block 0x1ffff) of src/test/peach-vectors.c:
 * its phash builds the map, its trailer the realistic batches */
static const word8 Pvector[sizeof(BTRAILER)] = {
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

/* reference walk of one slot */
typedef struct {
   word32 nonce[8];     /* full nonce */
   word32 hash0[8];     /* sha256(bt[0..123]) (or synthetic) */
   word32 mario[9];     /* tile entering round r, final tile in [8] */
   word32 algo[8];      /* reference algo of round r */
   word32 p;            /* reference op after the 8 nonce words */
   word32 final[8];     /* sha256(hash0 || tile[mario[8]]) */
} REFSLOT;

/* a batch configuration under test */
typedef struct {
   const char *name;
   word8 masks[8];      /* skip mask per round */
   word32 diff;         /* difficulty */
   int lost_ok;         /* small capacity: overflow expected */
} CFG;

/* deterministic test RNG (SplitMix64) */
static word64 Rs = WORD64_C(0x452821E638D01377);

/* failed checks (also reported by the exit status, even with NDEBUG) */
static word64 Fails;

/* map, tile generation flags and transition table */
static word8 *Map;
static word8 *Gen;
static word16 *T;
static word8 Phash[SHA256LEN];
static word32 TileList[CAP + NSYN];
static word64 TilesGenerated;

/* batch buffers (device memory in emulation) */
static PEACH_PIPE_BUFS Bufs;
static word64 *Rng0;          /* initial RNG states (realistic batch) */

/* reference data */
static REFSLOT RefR[NSLOTS];  /* realistic batch */
static REFSLOT RefS[NSYN];    /* synthetic batch */
static BTRAILER Bt;           /* realistic trailer (nonce first half) */
static PEACH_PIPE_PARAMS Par; /* realistic batch parameters */

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

/* count a failed check, print the first few */
static void fail(const char *what, long k, const char *detail)
{
   if (Fails++ < 24) printf("   FAIL: %s (item %ld) %s\n", what, k, detail);
}

/* zeroed allocation that ends at a PROT_NONE guard page (and starts
 * after one), so a write or read past the end faults */
static void *galloc(size_t len)
{
   long pg = sysconf(_SC_PAGESIZE);
   size_t page = pg > 0 ? (size_t) pg : 4096;
   size_t len16 = (len + 15) & ~(size_t) 15;
   size_t data = ((len16 + page - 1) / page) * page;
   word8 *base;

   base = (word8 *) mmap(NULL, data + (2 * page), PROT_READ | PROT_WRITE,
      MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
   ASSERT_NE((base), (MAP_FAILED));
   ASSERT_EQ((mprotect(base, page, PROT_NONE)), (0));
   ASSERT_EQ((mprotect(base + page + data, page, PROT_NONE)), (0));
   return base + page + data - len16;
}

/* the official RNG step (as cu_rand64()), independent of the header */
static word64 ref_rand64(word64 *state)
{
   word64 z = (*state += WORD64_C(0x9e3779b97f4a7c15));

   z = (z ^ (z >> 30)) * WORD64_C(0xbf58476d1ce4e5b9);
   z = (z ^ (z >> 27)) * WORD64_C(0x94d049bb133111eb);
   return (*state = z ^ (z >> 31));
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

/* host redraw rule: first nonce half without a NaN replacement */
static word32 redraw_first_half(word32 *n)
{
   word32 q;
   int d, nf;

   for (d = 0; d < 64; d++) {
      trigg_generate(n);
      nf = 0;
      q = peach_prefix_words(n, 4, 0, 0, &nf);
      if (!nf) return q;
   }
   ASSERT_LT((d), (64));
   return 0;
}

/* mario0 of a hash0 (product of the 32 bytes), as peach_checkhash() */
static word32 mario0_of(const word32 *h0)
{
   const word8 *b = (const word8 *) h0;
   word32 m = b[0];
   int i;

   for (i = 1; i < SHA256LEN; i++) m *= b[i];
   return m & PEACHCACHELEN_M1;
}

static word8 *tile_ptr(word32 t)
{
   return Map + ((size_t) t * PEACHTILELEN);
}

/* generate the listed tiles not generated yet (reference, in parallel),
 * then their transition entries with the transitions kernel (one launch
 * per tile, emulated serially), checked vs peach_transition_tile() */
static void ensure_tiles(const word32 *idx, int count)
{
   word16 tt;
   int i, n = 0;

   for (i = 0; i < count; i++) {
      if (Gen[idx[i]]) continue;
      Gen[idx[i]] = 1;
      TileList[n++] = idx[i];
   }
#pragma omp parallel for schedule(dynamic, 8)
   for (i = 0; i < n; i++) {
      peach_generate(TileList[i], Phash, tile_ptr(TileList[i]));
   }
   for (i = 0; i < n; i++) {
      CUDA_KERNEL(kcu_peach_pipe_transitions, 1, (i & 1) ? 32 : 128, 0,
         NULL)((const uint4 *) Map, T, TileList[i], 1);
      tt = peach_transition_tile((const word32 *) tile_ptr(TileList[i]),
         TileList[i]);
      if (T[TileList[i]] != tt) fail("T entry (transitions kernel)",
         (long) TileList[i], "!= peach_transition_tile()");
   }
   TilesGenerated += (word64) n;
}

/* reference walk of count slots (nonce, hash0 and mario[0] set): the
 * algo of every jump from peach_dflops(), the jump from peach_jump(),
 * the final hash from sha256(hash0 || tile) */
static void ref_walk(REFSLOT *R, int count)
{
   static word32 idx[CAP + NSYN];
   int k, r;

   for (r = 0; r <= PEACHROUNDS; r++) {
      for (k = 0; k < count; k++) idx[k] = R[k].mario[r];
      ensure_tiles(idx, count);
      if (r == PEACHROUNDS) break;
#pragma omp parallel for schedule(dynamic, 16)
      for (k = 0; k < count; k++) {
         word8 seed[PEACHJUMPLEN];
         word32 m = R[k].mario[r];

         memcpy(seed, R[k].nonce, 32);
         memcpy(seed + 32, &m, 4);
         memcpy(seed + 36, tile_ptr(m), PEACHTILELEN);
         R[k].algo[r] = peach_dflops(seed, PEACHJUMPLEN, m, 0) & 7;
         peach_jump(&m, (const word8 *) R[k].nonce, tile_ptr(m));
         R[k].mario[r + 1] = m;
      }
   }
#pragma omp parallel for schedule(static)
   for (k = 0; k < count; k++) {
      SHA256_CTX ctx;

      sha256_init(&ctx);
      sha256_update(&ctx, R[k].hash0, SHA256LEN);
      sha256_update(&ctx, tile_ptr(R[k].mario[PEACHROUNDS]), PEACHTILELEN);
      sha256_final(&ctx, R[k].final);
   }
}

/* the realistic trailer of a slot nonce */
static void slot_trailer(const word32 *nonce, BTRAILER *bt)
{
   *bt = Bt;
   memcpy(bt->nonce, nonce, 32);
}

/* round a slot is dropped at (8 = never) under the masks */
static int drop_round_of(const REFSLOT *R, const word8 *masks)
{
   int r;

   for (r = 0; r < PEACHROUNDS; r++) {
      if ((masks[r] >> R->algo[r]) & 1) return r;
   }
   return PEACHROUNDS;
}

static int words_eq(const word32 *a, const word32 *b, int n)
{
   return memcmp(a, b, (size_t) n * 4) == 0;
}

/* slot state (or queue entry) e == the reference state of slot R[k] in
 * round r: frame random number giving its nonce words 4..7, tile of
 * round r with P & 7, slot number k */
static int state_eq(const PEACH_PIPE_SLOT *e, const REFSLOT *R, word32 k,
   int r)
{
   word32 fr[4];

   frame_words(((word64) e->seed[1] << 32) | e->seed[0], fr);
   return words_eq(fr, &R[k].nonce[4], 4) && e->id == k &&
      e->tile == (R[k].mario[r] | ((R[k].p & 7) << PEACH_PIPE_PSHIFT));
}

/* Test the storage contract independently of the device selection helper. */
static PEACH_PIPE_SLOT *queue_entries(int r)
{
   return (r & 1) ? Bufs.d_ent : Bufs.d_slot;
}

/* Initial states must be checked before their storage becomes a packed queue. */
static void check_initial_slots(const CFG *c, const REFSLOT *R, int count)
{
   int k;

   for (k = 0; k < count; k++) {
      if (!state_eq(&Bufs.d_slot[k], R, (word32) k, 0)) {
         fail(c->name, k, "initial slot state (nonce words 4..7, mario0, P)");
      }
      if (!words_eq(&Bufs.d_hash[(word32) k * 8], R[k].hash0, 8)) {
         fail(c->name, k, "initial slot hash0");
      }
   }
}

/* fill the batch buffers with garbage (every read must be preceded by
 * a write of this batch) and the trace with a fill pattern */
static void scramble_buffers(word32 cap)
{
   memset(Bufs.d_slot, 0xA5, sizeof(PEACH_PIPE_SLOT) * CAP);
   memset(Bufs.d_hash, 0xA5, sizeof(word32) * 8 * CAP);
   memset(Bufs.d_q, 0xA5, sizeof(word32) * PEACH_PIPE_NQUEUE * CAP);
   memset(Bufs.d_ent, 0xA5, sizeof(PEACH_PIPE_SLOT) * CAP);
   memset(Bufs.d_cnt, 0xA5, sizeof(word32) * PEACH_PIPE_CNTWORDS);
   memset(Bufs.d_key, 0xA5, sizeof(word32) * CAP);
   memset(Bufs.d_res, 0xA5, sizeof(PEACH_PIPE_RESULT));
   memset(Bufs.d_trace, 0xEE, sizeof(PEACH_PIPE_TRACE) * CAP);
   Bufs.cap = cap;
}

/* the tile ordered queues of round r (1..7; 8: the final queue) right
 * after their sort (the final queue also after the batch): queue a at
 * off(r, a), its stored entries are the states (frame random number:
 * nonce words 4..7, tile of round r, P & 7) of distinct slots < count
 * that use algo a in round r (any algo for the final queue), in
 * ascending tile bucket order; and (unless lost_ok) exactly the slots
 * that reach round r */
static void check_round_queues(const CFG *c, const REFSLOT *R, int count,
   int r)
{
   static word8 seen[CAP];
   const PEACH_PIPE_SLOT *e;
   word32 want[8], cnt, off, n, i, slot, bk, prev, cap = Bufs.cap;
   char msg[96];
   int a, d, k, nq = r < PEACHROUNDS ? 8 : 1;

   memset(seen, 0, sizeof(seen));
   memset(want, 0, sizeof(want));
   for (k = 0; k < count; k++) {
      d = drop_round_of(&R[k], c->masks);
      if (d > r || d == PEACHROUNDS) want[r < PEACHROUNDS ? R[k].algo[r] : 0]++;
   }
   for (off = 0, a = 0; a < nq; a++, off += cnt) {
      cnt = Bufs.d_cnt[((r * 8) + a) * PEACH_PIPE_CNTPAD];
      if (!c->lost_ok && cnt != want[a]) {
         snprintf(msg, sizeof(msg), "round %d queue %d: cnt %u != %u", r, a,
            (unsigned) cnt, (unsigned) want[a]);
         fail(c->name, -1, msg);
      }
      n = peach_pipe_qlen(cnt, off, cap);
      for (prev = 0, i = 0; i < n; i++) {
         e = &queue_entries(r)[off + i];
         slot = e->id;
         snprintf(msg, sizeof(msg), "round %d queue %d entry %u", r, a,
            (unsigned) i);
         if (slot >= (word32) count || seen[slot]++) {
            fail(c->name, (long) slot, msg);
            continue;
         }
         d = drop_round_of(&R[slot], c->masks);
         if ((r < PEACHROUNDS && R[slot].algo[r] != (word32) a) ||
               (d <= r && d < PEACHROUNDS) || !state_eq(e, R, slot, r)) {
            fail(c->name, (long) slot, msg);
         }
         bk = (e->tile & PEACHCACHELEN_M1) >> PEACH_PIPE_SORT_SHIFT;
         if (i > 0 && bk < prev) {
            snprintf(msg, sizeof(msg), "round %d queue %d entry %u: tile"
               " bucket %u after %u", r, a, (unsigned) i, (unsigned) bk,
               (unsigned) prev);
            fail(c->name, (long) slot, msg);
         }
         prev = bk;
      }
   }
}

/* launch a batch with the per-kernel launches of peach_pipe_enqueue()
 * (used where the enqueue argument checks or its init do not apply);
 * with R != NULL, check the queues of every round after their sort */
static void launch_rounds(const PEACH_PIPE_PARAMS *p,
   const PEACH_PIPE_BUFS *b, const PEACH_PIPE_LAUNCH *l, int with_init,
   const CFG *c, const REFSLOT *R, int count)
{
   static word32 keys[CAP];
   word32 mask;
   int r, a;

   if (with_init) {
      PEACH_PIPE_PARAMS ready = *p;

      peach_sha256_trailer_prefix(p->mid, p->tail, p->nonce_lo,
         ready.sha_pre);
      memset(b->d_cnt, 0, sizeof(word32) * PEACH_PIPE_CNTZERO);
      memset(b->d_res, 0, sizeof(PEACH_PIPE_RESULT));
      CUDA_KERNEL(kcu_peach_pipe_init, l->grid_init, l->block, 0, NULL)
         (ready, *b);
   }
   if (R != NULL) check_initial_slots(c, R, count);
   for (r = 0; r < PEACHROUNDS; r++) {
      if (r > 0) {
         peach_pipe_launch_sort(p, b, l, r, NULL);
         if (R != NULL) check_round_queues(c, R, count, r);
         if (R != NULL && r == 1) {
            memcpy(keys, b->d_key, (size_t) count * sizeof(*keys));
         } else if (R != NULL &&
               memcmp(keys, b->d_key, (size_t) count * sizeof(*keys))) {
            fail(c->name, r, "round-0 keys changed by dense rounds");
         }
      }
      mask = peach_pipe_skip_mask(p->skip, (word32) r);
      for (a = 0; a < 8; a++) {
         if ((mask >> a) & 1) continue;
         peach_pipe_launch_hash(a, l->grid_hash[a], l->block, NULL, p, b,
            r);
      }
   }
   peach_pipe_launch_sort(p, b, l, PEACHROUNDS, NULL);
   if (R != NULL && memcmp(keys, b->d_key, (size_t) count * sizeof(*keys))) {
      fail(c->name, PEACHROUNDS, "round-0 keys changed by dense rounds");
   }
   if (R != NULL) check_round_queues(c, R, count, PEACHROUNDS);
   CUDA_KERNEL(kcu_peach_pipe_final, l->grid_final, l->block, 0, NULL)
      (*p, *b);
}

/* compare a finished batch with the reference walks R[0..count-1]
 * (slot k = R[k]); bt != NULL: realistic (trailer based) slots;
 * returns the number of mismatches */
static word64 check_batch(const CFG *c, const REFSLOT *R, int count,
   const BTRAILER *bt, word32 epoch)
{
   static word8 seen[CAP];
   const PEACH_PIPE_RESULT *res = Bufs.d_res;
   const PEACH_PIPE_TRACE *t;
   const PEACH_PIPE_SLOT *final = queue_entries(PEACHROUNDS);
   word32 expcnt[9][8], cnt, over, tot, m, e, i, slot, cap = Bufs.cap;
   word64 fails0 = Fails, alive = 0, dropped = 0, lost = 0, solvable = 0;
   BTRAILER cand;
   word8 out[SHA256LEN];
   char msg[96];
   int k, r, a, d, j;

   memset(expcnt, 0, sizeof(expcnt));
   memset(seen, 0, sizeof(seen));
   for (k = 0; k < count; k++) {
      t = &Bufs.d_trace[k];
      d = drop_round_of(&R[k], c->masks);
      if (!words_eq(&Bufs.d_hash[(word32) k * 8], R[k].hash0, 8)) {
         fail(c->name, k, "slot hash0");
      }
      if (c->lost_ok && t->drop_round == PEACH_PIPE_LOST) {
         /* lost to an overflow after entry j: prefix still exact */
         j = 0;
         while (j < 8 && t->mario[j + 1] != PEACH_PIPE_NOTILE) j++;
         lost++;
         /* kept (not dropped) in round j, so its drop round is later */
         if (j > d || (j == d && d < PEACHROUNDS)) {
            fail(c->name, k, "lost slot past its drop round");
         }
         for (i = 0; i < 8; i++) {
            if (t->final[i] != 0) fail(c->name, k, "lost slot final");
         }
         for (r = 0; r <= PEACHROUNDS; r++) {
            m = (word32) r <= (word32) j ? R[k].mario[r] : PEACH_PIPE_NOTILE;
            if (t->mario[r] != m) fail(c->name, k, "lost slot trace mario");
         }
         for (r = 0; r < PEACHROUNDS; r++) {
            m = r <= j ? R[k].algo[r] : PEACH_PIPE_NOALGO;
            if (t->algo[r] != m) fail(c->name, k, "lost slot trace algo");
         }
         continue;
      }
      /* expected trace */
      e = (word32) d;
      for (r = 0; r <= PEACHROUNDS; r++) {
         m = (word32) r <= e ? R[k].mario[r] : PEACH_PIPE_NOTILE;
         if (t->mario[r] != m) {
            snprintf(msg, sizeof(msg), "trace mario[%d] %08x != %08x", r,
               (unsigned) t->mario[r], (unsigned) m);
            fail(c->name, k, msg);
         }
      }
      for (r = 0; r < PEACHROUNDS; r++) {
         m = (word32) r <= e ? R[k].algo[r] : PEACH_PIPE_NOALGO;
         if (t->algo[r] != m) {
            snprintf(msg, sizeof(msg), "trace algo[%d] %u != %u", r,
               (unsigned) t->algo[r], (unsigned) m);
            fail(c->name, k, msg);
         }
      }
      if (t->pad[0] || t->pad[1] || t->pad[2]) {
         fail(c->name, k, "trace pad");
      }
      if (d < PEACHROUNDS) {
         dropped++;
         if (t->drop_round != (word8) d) fail(c->name, k, "drop round");
         for (i = 0; i < 8; i++) {
            if (t->final[i] != 0) fail(c->name, k, "dropped slot final");
         }
      } else {
         alive++;
         if (t->drop_round != PEACH_PIPE_ALIVE) {
            fail(c->name, k, "completed slot drop round");
         }
         if (!words_eq(t->final, R[k].final, 8)) {
            fail(c->name, k, "final hash != reference walk");
         }
         if (trigg_eval(R[k].final, (word8) c->diff) == VEOK) solvable++;
      }
      /* expected queue counters (appended in rounds < d) */
      for (r = 0; r < d; r++) expcnt[r][R[k].algo[r]]++;
      if (d == PEACHROUNDS) expcnt[8][0]++;
   }
   /* nothing written beyond the batch */
   for (k = count; k < count + 128 && k < CAP; k++) {
      t = &Bufs.d_trace[k];
      if (t->drop_round != 0xEE || t->mario[0] != WORD32_C(0xEEEEEEEE)) {
         fail(c->name, k, "trace written beyond nslots");
      }
   }
   /* counters; overflow: past cap in a round 0 queue, or past cap in the
    * entries of a later round */
   over = 0;
   for (r = 0; r <= PEACHROUNDS; r++) {
      for (tot = 0, a = 0; a < 8; a++) {
         cnt = Bufs.d_cnt[((r * 8) + a) * PEACH_PIPE_CNTPAD];
         if (r == 0 && cnt > cap) over += cnt - cap;
         tot += cnt;
         if (r > 0 && a == 7 && tot > cap) over += tot - cap;
         if (!c->lost_ok && cnt != expcnt[r][a]) {
            snprintf(msg, sizeof(msg), "cnt(%d,%d) %u != %u", r, a,
               (unsigned) cnt, (unsigned) expcnt[r][a]);
            fail(c->name, (long) ((r * 8) + a), msg);
         }
      }
   }
   if (res->completed != alive) fail(c->name, -1, "completed");
   if (res->dropped != dropped) fail(c->name, -1, "dropped");
   if (res->overflow != lost || over != lost) fail(c->name, -1, "overflow");
   if (c->lost_ok && lost == 0) fail(c->name, -1, "no overflow happened");
   if (res->anomaly != 0) fail(c->name, -1, "anomaly");
   if (res->epoch != epoch) fail(c->name, -1, "epoch");
   /* final queue: a permutation of completed original IDs and their states */
   cnt = Bufs.d_cnt[PEACH_PIPE_FINALCNT * PEACH_PIPE_CNTPAD];
   if ((cnt < cap ? cnt : cap) != alive) fail(c->name, -1, "final count");
   for (i = 0; i < alive && i < cap; i++) {
      slot = final[i].id;
      if (slot >= (word32) count || seen[slot]++ ||
            Bufs.d_trace[slot].drop_round != PEACH_PIPE_ALIVE) {
         fail(c->name, (long) i, "final queue entry");
      }
   }
   check_round_queues(c, R, count, PEACHROUNDS);
   /* canary: final queue entry peach_pipe_canary() */
   if (alive == 0) {
      if (res->canary_valid != 0) fail(c->name, -1, "canary without entry");
   } else {
      slot = final[peach_pipe_canary(epoch, cnt < cap ? cnt : cap)].id;
      if (res->canary_valid != 1 || slot >= (word32) count ||
            !words_eq(res->canary_nonce_hi, &R[slot].nonce[4], 4) ||
            !words_eq(res->canary_hash, R[slot].final, 8)) {
         fail(c->name, -1, "canary");
      } else if (bt != NULL) {
         slot_trailer(R[slot].nonce, &cand);
         if (ref_peach_checkhash(&cand, 0, out) != VEOK ||
               memcmp(out, res->canary_hash, SHA256LEN) != 0) {
            fail(c->name, -1, "canary vs ref_peach_checkhash()");
         }
      }
   }
   /* solve: found iff any completed slot meets the difficulty */
   if (res->found != (solvable ? 1u : 0u)) fail(c->name, -1, "found flag");
   if (res->found) {
      for (k = 0; k < count; k++) {
         if (Bufs.d_trace[k].drop_round == PEACH_PIPE_ALIVE &&
               words_eq(&R[k].nonce[4], res->nonce_hi, 4) &&
               words_eq(R[k].final, res->hash, 8)) break;
      }
      if (k == count) fail(c->name, -1, "solve nonce/hash of no slot");
      if (trigg_eval(res->hash, (word8) c->diff) != VEOK) {
         fail(c->name, -1, "solve hash does not meet diff");
      }
      if (bt != NULL) {
         cand = *bt;
         memcpy(cand.nonce, Par.nonce_lo, 16);
         memcpy(cand.nonce + 16, res->nonce_hi, 16);
         if (ref_peach_checkhash(&cand, (word8) c->diff, out) != VEOK ||
               memcmp(out, res->hash, SHA256LEN) != 0) {
            fail(c->name, -1, "solve vs ref_peach_checkhash()");
         }
      }
   }
   printf("   %-26s alive %4llu dropped %4llu lost %4llu, solvable %llu "
      "found %u, canary %u, mismatches %llu\n", c->name,
      (unsigned long long) alive, (unsigned long long) dropped,
      (unsigned long long) lost, (unsigned long long) solvable,
      (unsigned) res->found, (unsigned) res->canary_valid,
      (unsigned long long) (Fails - fails0));
   return Fails - fails0;
}

/* (0) helpers, argument validation, self-test and transitions kernels */
static void check_helpers(void)
{
   static word32 vin[NSELF * PEACH_PIPE_SELFTEST_IN];
   static word32 vout[NSELF * PEACH_PIPE_SELFTEST_OUT + 3];
   static word32 lead[NSELF];
   static const word32 idx3[3] = { 0, 1, PEACHCACHELEN_M1 };
   PEACH_PIPE_PARAMS p;
   PEACH_PIPE_LAUNCH l;
   PEACH_PIPE_BUFS b;
   word32 h[8], w, op, x, buf[2], tile[PEACHTILELEN32], list[20];
   word32 fr[4], fr2[4], quarter[4];
   word64 fails0 = Fails, sum0, sum1, sd;
   word8 masks[8], *hb = (word8 *) h;
   word16 tt;
   int diff, z, i, j, k, nf;

   /* difficulty check vs trigg_eval(): every leading zero count */
   for (z = 0; z <= 256; z++) {
      for (j = 0; j < 4; j++) {
         for (i = 0; i < 32; i++) hb[i] = (word8) r32();
         for (i = 0; i < z / 8; i++) hb[i] = 0;
         if (z < 256) {
            hb[z / 8] &= (word8) (0xFF >> (z & 7));
            hb[z / 8] |= (word8) (0x80 >> (z & 7));
         }
         for (diff = 0; diff < 256; diff++) {
            if (peach_pipe_diff_ok(h, (word32) diff) !=
                  (trigg_eval(h, (word8) diff) == VEOK)) {
               fail("diff_ok vs trigg_eval", (long) diff, "");
            }
         }
      }
   }
   /* nonce frame (packed tables, 32-bit halves) vs the official 64-bit
    * frame: every value of every table index field, then random seeds */
   for (i = 0; i < 64 + 4096; i++) {
      sd = i >= 64 ? r64() : (word64) (i & 31) | ((word64) (i & 7) << 5) |
         ((word64) (i & 63) << 8) | ((word64) (i & 63) << 14) |
         ((word64) (i & 31) << 20) | ((word64) (i & 31) << 25) |
         ((word64) (i & 63) << 30) | ((word64) (i & 63) << 36) |
         ((word64) r32() << 42);
      peach_pipe_frame(sd, fr);
      frame_words(sd, fr2);
      if (!words_eq(fr, fr2, 4)) fail("peach_pipe_frame()", i, "");
   }
   /* skip mask packing */
   for (i = 0; i < 64; i++) {
      for (j = 0; j < 8; j++) masks[j] = (word8) (r32() % 0xFF);
      for (j = 0; j < 8; j++) {
         if (peach_pipe_skip_mask(peach_pipe_skip_pack(masks), (word32) j)
               != masks[j]) fail("skip pack/mask", j, "");
      }
   }
   /* canary entry: below the queue length, and spread over the (tile
    * ordered) final queue from one epoch to the next */
   if (peach_pipe_canary(1, 0) != 0 || peach_pipe_canary(~0u, 1) != 0) {
      fail("peach_pipe_canary()", -1, "empty or single entry queue");
   }
   memset(quarter, 0, sizeof(quarter));
   for (i = 0; i < 64; i++) {
      x = peach_pipe_canary((word32) i + 1, 4096);
      if (x >= 4096) fail("peach_pipe_canary()", i, "out of range");
      else quarter[x / 1024]++;
      x = peach_pipe_canary(r32(), (word32) (i + 1));
      if (x > (word32) i) fail("peach_pipe_canary()", i, "out of range");
   }
   for (i = 0; i < 4; i++) {
      if (quarter[i] < 8) fail("peach_pipe_canary()", i, "quarter missed");
   }
   /* enqueue argument validation: nothing is enqueued */
   memset(&p, 0, sizeof(p));
   b = Bufs;
   b.cap = 256;
   p.nslots = 256;
   l.block = 128;
   l.grid_init = l.grid_final = 1;
   for (j = 0; j < 8; j++) l.grid_hash[j] = 1;
   memset(Bufs.d_res, 0x77, sizeof(PEACH_PIPE_RESULT));
   for (i = 0; i < 12; i++) {
      PEACH_PIPE_PARAMS p2 = p;
      PEACH_PIPE_BUFS b2 = b;
      PEACH_PIPE_LAUNCH l2 = l;

      switch (i) {
         case 0: l2.block = 0; break;
         case 1: l2.block = 96 + 1; break;
         case 2: l2.block = 256; break;
         case 3: l2.grid_init = 0; break;
         case 4: l2.grid_final = 0; break;
         case 5: l2.grid_hash[6] = 0; break;
         case 6: b2.cap = 0; break;
         case 7: p2.nslots = b2.cap + 128; break;
         case 8: l2.grid_init = PEACH_PIPE_MAXGRID + 1; break;
         case 9: l2.grid_final = PEACH_PIPE_MAXGRID + 1; break;
         case 10: l2.grid_hash[3] = PEACH_PIPE_MAXGRID + 1; break;
         default: b2.cap = PEACH_PIPE_MAXCAP + 128; break;
      }
      if (peach_pipe_enqueue(&p2, &b2, &l2, NULL) != -1) {
         fail("enqueue validation", i, "accepted");
      }
   }
   if (peach_pipe_enqueue(NULL, &b, &l, NULL) != -1) {
      fail("enqueue validation", -1, "NULL accepted");
   }
   if (((word8 *) Bufs.d_res)[0] != 0x77 ||
         ((word8 *) Bufs.d_res)[sizeof(PEACH_PIPE_RESULT) - 1] != 0x77) {
      fail("enqueue validation", -1, "something was enqueued");
   }
   /* self-test kernel: specials and random words x op x index; the op of
    * each entry is the op after a NaN-free lead word, so a two-word
    * reference peach_dflops() checks the step */
   for (k = 0; k < NSELF; k++) {
      w = k < NSPECIAL * 12 ? Special[k / 12] : (k & 1) ? r32() :
         Special[r32() % NSPECIAL] ^ (r32() & 0x8000000F);
      x = idx3[k % 3];
      do {
         buf[0] = r32();
         nf = 0;
         op = peach_dflops_step(buf[0], 0, x, &nf);
      } while (nf || peach_dflops(buf, 4, x, 0) != op);
      lead[k] = buf[0];
      vin[(k * 3)] = w;
      vin[(k * 3) + 1] = op;
      vin[(k * 3) + 2] = x;
   }
   memset(vout, 0xC3, sizeof(vout));
   CUDA_KERNEL(kcu_peach_pipe_selftest, 3, 64, 0, NULL)(vin, vout, NSELF);
   for (k = 0; k < NSELF; k++) {
      w = vin[k * 3];
      op = vin[(k * 3) + 1];
      x = vin[(k * 3) + 2];
      nf = 0;
      if (peach_dflops_step(w, op, x, &nf) != vout[k * 3] ||
            (word32) nf != vout[(k * 3) + 1]) {
         fail("self-test kernel step", k, "");
      }
      /* the step vs a two-word reference peach_dflops() */
      buf[0] = lead[k];
      buf[1] = w;
      if (peach_dflops(buf, 8, x, 0) != vout[k * 3]) {
         fail("self-test kernel vs ref peach_dflops", k, "");
      }
      /* increments: byte j = (step(op = j) - j) & 7 */
      for (j = 0; j < 4; j++) {
         nf = 0;
         if ((((peach_dflops_step(w, (word32) j, x, &nf) - (word32) j) & 7)
               != ((vout[(k * 3) + 2] >> (8 * j)) & 0xFF))) {
            fail("self-test kernel incs", k, "");
         }
      }
   }
   if (vout[NSELF * 3] != WORD32_C(0xC3C3C3C3)) {
      fail("self-test kernel", -1, "wrote past count");
   }
   /* transitions kernel: an adversarial tile (specials, NaN, Inf,
    * denormals) at tile 2, then real tiles at the end of the map with a
    * range reaching past it (clamped; the T array ends at a guard page) */
   for (i = 0; i < PEACHTILELEN32; i++) {
      tile[i] = (i & 1) ? Special[r32() % NSPECIAL] : r32();
   }
   memcpy(tile_ptr(2), tile, PEACHTILELEN);
   CUDA_KERNEL(kcu_peach_pipe_transitions, 2, 128, 0, NULL)
      ((const uint4 *) Map, T, 2, 1);
   tt = peach_transition_tile(tile, 2);
   if (T[2] != tt) fail("transitions kernel, adversarial tile", 2, "");
   memset(tile_ptr(2), 0, PEACHTILELEN);
   T[2] = 0;
   for (i = 0; i < 20; i++) list[i] = PEACHCACHELEN - 20 + (word32) i;
#pragma omp parallel for schedule(dynamic, 1)
   for (i = 0; i < 20; i++) {
      peach_generate(list[i], Phash, tile_ptr(list[i]));
   }
   for (i = 0; i < 20; i++) Gen[list[i]] = 1;
   CUDA_KERNEL(kcu_peach_pipe_transitions, 1, 128, 0, NULL)
      ((const uint4 *) Map, T, PEACHCACHELEN - 20, 1000);
   for (i = 0; i < 20; i++) {
      tt = peach_transition_tile((const word32 *) tile_ptr(list[i]),
         list[i]);
      if (T[list[i]] != tt) fail("transitions kernel, map end", i, "");
   }
   for (sum0 = 0, i = 0; i < PEACHCACHELEN; i++) {
      sum0 += (word64) T[i] * (word64) (i + 1);
   }
   CUDA_KERNEL(kcu_peach_pipe_transitions, 2, 64, 0, NULL)
      ((const uint4 *) Map, T, PEACHCACHELEN, 5);
   CUDA_KERNEL(kcu_peach_pipe_transitions, 2, 64, 0, NULL)
      ((const uint4 *) Map, T, WORD32_C(0xFFFFFFFF), 2);
   CUDA_KERNEL(kcu_peach_pipe_transitions, 2, 64, 0, NULL)
      ((const uint4 *) Map, T, 7, 0);
   for (sum1 = 0, i = 0; i < PEACHCACHELEN; i++) {
      sum1 += (word64) T[i] * (word64) (i + 1);
   }
   if (sum0 != sum1) fail("transitions kernel", -1, "empty range wrote T");
   printf("(0) diff_ok == trigg_eval (257 zero counts x 256 diffs), nonce "
      "frame, skip packing, enqueue validation, self-test kernel (%d "
      "entries), transitions kernel (adversarial tile, map end clamp): "
      "mismatches %llu\n", NSELF, (unsigned long long) (Fails - fails0));
   ASSERT_EQ((Fails - fails0), (0));
}

/* (1) realistic batch, pass 1 (init only) and the reference walks */
static void check_init(void)
{
   PEACH_PIPE_LAUNCH l;
   BTRAILER bt;
   SHA256_CTX ctx;
   word64 fails0 = Fails, st;
   word32 h0[8], fr[4], nzero = 0;
   double t0 = now_s(), t1;
   int k, nf;

   /* trailer: vector phash/bnum/... with a redrawn first nonce half */
   memcpy(&Bt, Pvector, sizeof(Bt));
   memset(&Par, 0, sizeof(Par));
   Par.q = redraw_first_half(Par.nonce_lo);
   memcpy(Bt.nonce, Par.nonce_lo, 16);
   sha256_init(&ctx);
   sha256_update(&ctx, &Bt, 64);
   memcpy(Par.mid, ctx.state, 32);
   memcpy(Par.tail, ((const word8 *) &Bt) + 64, 28);
   Par.nslots = NSLOTS;
   Par.diff = 1;
   Par.epoch = 1;
   Par.skip = 0;
   peach_sha256_trailer_prefix(Par.mid, Par.tail, Par.nonce_lo, Par.sha_pre);
   /* pass 1: init kernel only, to learn the nonces */
   for (k = 0; k < CAP; k++) Rng0[k] = r64();
   memcpy(Bufs.d_rng, Rng0, sizeof(word64) * CAP);
   scramble_buffers(CAP);
   l.block = 128;
   l.grid_init = 5;
   memset(Bufs.d_cnt, 0, sizeof(word32) * PEACH_PIPE_CNTZERO);
   memset(Bufs.d_res, 0, sizeof(PEACH_PIPE_RESULT));
   CUDA_KERNEL(kcu_peach_pipe_init, l.grid_init, l.block, 0, NULL)
      (Par, Bufs);
   for (k = 0; k < NSLOTS; k++) {
      /* RNG and frame, computed independently */
      st = Rng0[k];
      frame_words(ref_rand64(&st), fr);
      if (Bufs.d_rng[k] != st) fail("init RNG state", k, "");
      if (Bufs.d_slot[k].seed[0] != (word32) st ||
            Bufs.d_slot[k].seed[1] != (word32) (st >> 32)) {
         fail("init slot frame random number", k, "");
      }
      memcpy(RefR[k].nonce, Par.nonce_lo, 16);
      memcpy(&RefR[k].nonce[4], fr, 16);
      if (trigg_syntax(&RefR[k].nonce[4]) != VEOK ||
            trigg_syntax(RefR[k].nonce) != VEOK) {
         fail("nonce syntax", k, "");
      }
      slot_trailer(RefR[k].nonce, &bt);
      sha256(&bt, 124, h0);
      memcpy(RefR[k].hash0, h0, 32);
      RefR[k].mario[0] = mario0_of(h0);
      RefR[k].p = peach_dflops(RefR[k].nonce, 32, RefR[k].mario[0], 0);
      if (!words_eq(&Bufs.d_hash[k * 8], h0, 8)) fail("init hash0", k, "");
      if (!state_eq(&Bufs.d_slot[k], RefR, (word32) k, 0)) {
         fail("init slot state (nonce words 4..7, mario0, P)", k, "");
      }
      /* P is index independent (no NaN replacement in the nonce) */
      nf = 0;
      if (peach_prefix_words(RefR[k].nonce, 8, 0, (word32) k &
            PEACHCACHELEN_M1, &nf) != RefR[k].p || nf) {
         fail("P index independent", k, "");
      }
      nzero += (RefR[k].mario[0] == 0);
   }
   if (Bufs.d_res->anomaly != 0) fail("init anomaly", -1, "");
   t1 = now_s();
   /* reference walks; visited tiles + T entries */
   ref_walk(RefR, NSLOTS);
   printf("(1) realistic batch pass 1: %d slots, nonces/RNG/hash0/mario0/P "
      "vs independent computation; mario0 == 0 for %d slots; reference "
      "walks %.2fs (%llu tiles generated, T via transitions kernel); "
      "mismatches %llu, %.2fs\n", NSLOTS,
      (int) nzero, now_s() - t1,
      (unsigned long long) TilesGenerated,
      (unsigned long long) (Fails - fails0), now_s() - t0);
   ASSERT_EQ((Fails - fails0), (0));
}

/* reference sample: ref_peach_checkhash() == walk final (every 8th) */
static void check_sample(void)
{
   word64 bad = 0;
   int k;

#pragma omp parallel for schedule(dynamic, 4) reduction(+:bad)
   for (k = 0; k < NSLOTS; k += NSAMPLE) {
      BTRAILER bt;
      word8 out[SHA256LEN];

      slot_trailer(RefR[k].nonce, &bt);
      if (ref_peach_checkhash(&bt, 0, out) != VEOK ||
            memcmp(out, RefR[k].final, SHA256LEN) != 0) bad++;
   }
   printf("    reference walk final == ref_peach_checkhash() for %d "
      "sampled slots: mismatches %llu\n", NSLOTS / NSAMPLE,
      (unsigned long long) bad);
   Fails += bad;
   ASSERT_EQ((bad), (0));
}

/* (2) realistic batch, pass 2 (peach_pipe_enqueue()) */
static void check_realistic(void)
{
   static PEACH_PIPE_TRACE tr0[NSLOTS];
   CFG cfg[NCFG], empty;
   PEACH_PIPE_PARAMS p;
   PEACH_PIPE_LAUNCH l;
   word64 fails0 = Fails, hist[8], best;
   double t0 = now_s();
   int c, k, r, a, x;

   /* configurations: default, full evaluation, mixed per round (round 0
    * drops the less frequent of its two algorithms) */
   memset(cfg, 0, sizeof(cfg));
   cfg[0].name = "skip 0x40 x 8, diff 4";
   cfg[1].name = "skip all zero, diff 255";
   cfg[2].name = "skip mixed, diff 7";
   for (r = 0; r < 8; r++) cfg[0].masks[r] = PEACH_PIPE_SKIP_MD2;
   cfg[0].diff = 4;
   cfg[1].diff = 255;
   memset(hist, 0, sizeof(hist));
   for (k = 0; k < NSLOTS; k++) hist[RefR[k].algo[0]]++;
   for (best = 0, x = a = 0; a < 4; a++) {
      if (hist[a] + hist[a + 4] > best) { best = hist[a] + hist[a + 4]; x = a; }
   }
   cfg[2].masks[0] = (word8) (1 << (hist[x] < hist[x + 4] ? x : x + 4));
   cfg[2].masks[1] = 0x81;
   cfg[2].masks[2] = 0x06;
   cfg[2].masks[3] = 0x18;
   cfg[2].masks[4] = 0x60;
   cfg[2].masks[5] = 0x00;
   cfg[2].masks[6] = 0x22;
   cfg[2].masks[7] = 0x48;
   cfg[2].diff = 7;
   printf("(2) realistic batch pass 2 (peach_pipe_enqueue, same nonces):\n");
   for (c = 0; c < NCFG; c++) {
      p = Par;
      p.skip = peach_pipe_skip_pack(cfg[c].masks);
      p.diff = cfg[c].diff;
      p.epoch = 0x100 + (word32) c;
      /* launch shapes: stride loops, one item per thread, one block */
      l.block = c == 2 ? 64 : 128;
      l.grid_init = c == 0 ? 7 : c == 1 ? NSLOTS / 128 : 1;
      l.grid_final = c == 0 ? 3 : c == 1 ? NSLOTS / 128 : 1;
      for (a = 0; a < 8; a++) {
         l.grid_hash[a] = c == 0 ? 1 + (a * 5) : c == 1 ? NSLOTS / 128 : 1;
      }
      memcpy(Bufs.d_rng, Rng0, sizeof(word64) * CAP);
      scramble_buffers(CAP);
      ASSERT_EQ((peach_pipe_enqueue(&p, &Bufs, &l, NULL)), (0));
      check_batch(&cfg[c], RefR, NSLOTS, &Bt, p.epoch);
      if (c == 0) {
         memcpy(tr0, Bufs.d_trace, sizeof(tr0));
         /* round 0 concentrates on {x, x + 4} (mario0 == 0, T[0]) */
         memset(hist, 0, sizeof(hist));
         for (k = 0; k < NSLOTS; k++) hist[tr0[k].algo[0] & 7]++;
         for (best = 0, x = a = 0; a < 4; a++) {
            if (hist[a] + hist[a + 4] > best) {
               best = hist[a] + hist[a + 4];
               x = a;
            }
         }
         printf("   round 0 algos {%d, %d}: %.1f%% (%llu + %llu of %d)\n",
            x, x + 4, 100.0 * (double) best / NSLOTS,
            (unsigned long long) hist[x], (unsigned long long) hist[x + 4],
            NSLOTS);
         if (best * 10 <= (word64) NSLOTS * 9) {
            fail("round 0 concentration", -1, "<= 90%");
         }
      }
   }
   /* config 0 again with other launch shapes: identical traces */
   p = Par;
   p.skip = peach_pipe_skip_pack(cfg[0].masks);
   p.diff = cfg[0].diff;
   p.epoch = 0x1FF;
   l.block = 32;
   l.grid_init = 1;
   l.grid_final = 2 * NSLOTS / 32;
   for (a = 0; a < 8; a++) l.grid_hash[a] = 3 + a;
   memcpy(Bufs.d_rng, Rng0, sizeof(word64) * CAP);
   scramble_buffers(CAP);
   ASSERT_EQ((peach_pipe_enqueue(&p, &Bufs, &l, NULL)), (0));
   if (memcmp(tr0, Bufs.d_trace, sizeof(tr0)) != 0) {
      fail("launch shape independence", -1, "traces differ");
   }
   check_batch(&cfg[0], RefR, NSLOTS, &Bt, p.epoch);
   /* config 0 with nslots not a multiple of the block: the last
    * block-uniform iteration has idle lanes, and slots [nslots, cap)
    * must stay untouched */
   p.nslots = NSLOTS - 96;
   p.epoch = 0x2FF;
   l.block = 128;
   l.grid_init = l.grid_final = 3;
   for (a = 0; a < 8; a++) l.grid_hash[a] = 2 + a;
   memcpy(Bufs.d_rng, Rng0, sizeof(word64) * CAP);
   scramble_buffers(CAP);
   ASSERT_EQ((peach_pipe_enqueue(&p, &Bufs, &l, NULL)), (0));
   check_batch(&cfg[0], RefR, (int) p.nslots, &Bt, p.epoch);
   for (k = (int) p.nslots; k < CAP; k++) {
      const word8 *sb = (const word8 *) &Bufs.d_slot[k];

      if (Bufs.d_rng[k] != Rng0[k]) fail("idle slot rng", k, "touched");
      if (sb[0] != 0xA5 || sb[sizeof(PEACH_PIPE_SLOT) - 1] != 0xA5 ||
            Bufs.d_hash[k * 8] != WORD32_C(0xA5A5A5A5) ||
            Bufs.d_trace[k].drop_round != 0xEE) {
         fail("idle slot state", k, "touched");
      }
   }
   /* configs 0 and 2 kernel by kernel: the tile ordered queues of every
    * round right after their sort */
   for (c = 0; c < NCFG; c += 2) {
      p = Par;
      p.skip = peach_pipe_skip_pack(cfg[c].masks);
      p.diff = cfg[c].diff;
      p.epoch = 0x3FF + (word32) c;
      l.block = 96;
      l.grid_init = l.grid_final = 5;
      for (a = 0; a < 8; a++) l.grid_hash[a] = 1 + (2 * a);
      memcpy(Bufs.d_rng, Rng0, sizeof(word64) * CAP);
      scramble_buffers(CAP);
      launch_rounds(&p, &Bufs, &l, 1, &cfg[c], RefR, NSLOTS);
      check_batch(&cfg[c], RefR, NSLOTS, &Bt, p.epoch);
   }
   /* Empty packed queues, including a dense producer rejecting every
    * incoming entry; 97 slots also exercises partial warps. */
   for (c = 0; c < 2; c++) {
      memset(&empty, 0, sizeof(empty));
      empty.name = c == 0 ? "drop all at init" : "drop all after dense round 1";
      empty.masks[c == 0 ? 0 : 2] = 0xFF;
      empty.diff = 255;
      p = Par;
      p.nslots = 97;
      p.skip = peach_pipe_skip_pack(empty.masks);
      p.diff = empty.diff;
      p.epoch = 0x4FF + (word32) c;
      l.block = 128;
      l.grid_init = l.grid_final = 3;
      for (a = 0; a < 8; a++) l.grid_hash[a] = 1 + a;
      memcpy(Bufs.d_rng, Rng0, sizeof(word64) * CAP);
      scramble_buffers(CAP);
      ASSERT_EQ((peach_pipe_enqueue(&p, &Bufs, &l, NULL)), (0));
      check_batch(&empty, RefR, (int) p.nslots, &Bt, p.epoch);
   }
   check_sample();
   printf("    mismatches %llu, %.2fs\n",
      (unsigned long long) (Fails - fails0), now_s() - t0);
   ASSERT_EQ((Fails - fails0), (0));
}

/* (3) synthetic batch: random nonzero start tiles */
static void check_synthetic(void)
{
   static word32 order[NSYN];
   static word64 seed[NSYN];
   PEACH_PIPE_PARAMS p;
   PEACH_PIPE_LAUNCH l;
   PEACH_PIPE_TRACE *t;
   CFG cfg;
   word32 nlo[4], q, a, pos, tmp, cnt0;
   word64 fails0 = Fails, nonzero0 = 0;
   double t0 = now_s();
   int k, r, i, nf;

   memset(&cfg, 0, sizeof(cfg));
   cfg.name = "synthetic, md2 off r0/r5";
   cfg.masks[0] = PEACH_PIPE_SKIP_MD2;
   cfg.masks[5] = PEACH_PIPE_SKIP_MD2;
   cfg.diff = 3;
   q = redraw_first_half(nlo);
   for (k = 0; k < NSYN; k++) {
      memcpy(RefS[k].nonce, nlo, 16);
      seed[k] = r64();
      frame_words(seed[k], &RefS[k].nonce[4]);
      for (i = 0; i < 8; i++) RefS[k].hash0[i] = r32();
      switch (k) {
         case 0: RefS[k].mario[0] = 1; break;
         case 1: RefS[k].mario[0] = PEACHCACHELEN_M1; break;
         default:
            do {
               RefS[k].mario[0] = r32() & PEACHCACHELEN_M1;
            } while (RefS[k].mario[0] == 0);
      }
      nf = 0;
      RefS[k].p = peach_dflops(RefS[k].nonce, 32, RefS[k].mario[0], 0);
      if (peach_prefix_words(&RefS[k].nonce[4], 4, q, 0, &nf) != RefS[k].p
            || nf) fail("synthetic prefix", k, "");
      nonzero0 += (RefS[k].mario[0] != 0);
   }
   ref_walk(RefS, NSYN);
   /* slot states, frame random numbers, round-0 queues (random slot
    * order) and traces as the init kernel would write them */
   scramble_buffers(CAP);
   memset(Bufs.d_cnt, 0, sizeof(word32) * PEACH_PIPE_CNTZERO);
   memset(Bufs.d_res, 0, sizeof(PEACH_PIPE_RESULT));
   for (k = 0; k < NSYN; k++) order[k] = (word32) k;
   for (k = NSYN - 1; k > 0; k--) {
      i = (int) (r32() % (word32) (k + 1));
      tmp = order[k]; order[k] = order[i]; order[i] = tmp;
   }
   for (i = 0; i < NSYN; i++) {
      k = (int) order[i];
      Bufs.d_slot[k].seed[0] = (word32) seed[k];
      Bufs.d_slot[k].seed[1] = (word32) (seed[k] >> 32);
      Bufs.d_slot[k].tile = RefS[k].mario[0] | ((RefS[k].p & 7) <<
         PEACH_PIPE_PSHIFT);
      Bufs.d_slot[k].id = (word32) k;
      Bufs.d_rng[k] = seed[k];
      Bufs.d_key[k] = PEACH_PIPE_KEYDEAD;
      memcpy(&Bufs.d_hash[k * 8], RefS[k].hash0, 32);
      t = &Bufs.d_trace[k];
      memset(t, 0, sizeof(*t));
      for (r = 0; r <= PEACHROUNDS; r++) t->mario[r] = PEACH_PIPE_NOTILE;
      memset(t->algo, PEACH_PIPE_NOALGO, sizeof(t->algo));
      t->mario[0] = RefS[k].mario[0];
      a = RefS[k].algo[0];
      t->algo[0] = (word8) a;
      if ((cfg.masks[0] >> a) & 1) {
         t->drop_round = 0;
         Bufs.d_res->dropped++;
         continue;
      }
      t->drop_round = PEACH_PIPE_ALIVE;
      pos = Bufs.d_cnt[a * PEACH_PIPE_CNTPAD]++;
      Bufs.d_q[(a * CAP) + pos] = (word32) k;
   }
   for (cnt0 = 0, a = 0; a < 8; a++) {
      cnt0 += (Bufs.d_cnt[a * PEACH_PIPE_CNTPAD] != 0);
   }
   p = Par;
   memcpy(p.nonce_lo, nlo, 16);
   p.q = q;
   p.nslots = NSYN;
   p.skip = peach_pipe_skip_pack(cfg.masks);
   p.diff = cfg.diff;
   p.epoch = 0x5EED;
   l.block = 128;
   l.grid_init = l.grid_final = 2;
   for (a = 0; a < 8; a++) l.grid_hash[a] = 1 + (int) a;
   launch_rounds(&p, &Bufs, &l, 0, &cfg, RefS, NSYN);
   printf("(3) synthetic batch: %d slots, %llu nonzero start tiles, "
      "round 0 uses %u algorithms\n", NSYN, (unsigned long long) nonzero0,
      (unsigned) cnt0);
   check_batch(&cfg, RefS, NSYN, NULL, p.epoch);
   printf("    mismatches %llu, %.2fs\n",
      (unsigned long long) (Fails - fails0), now_s() - t0);
   if (cnt0 < 7) fail("synthetic round 0 algorithms", -1, "< 7 used");
   ASSERT_EQ((Fails - fails0), (0));
}

/* (4) small capacity: round-0 slot queues and the first packed queue
 * overflow; later packed queues contain at most cap entries. */
static void check_smallcap(void)
{
   PEACH_PIPE_PARAMS p;
   PEACH_PIPE_LAUNCH l;
   CFG cfg;
   word64 fails0 = Fails;
   word32 c0, cr, tot;
   double t0 = now_s();
   int a, r, rmax = 0;

   memset(&cfg, 0, sizeof(cfg));
   cfg.name = "cap 1024 < nslots 4096";
   cfg.diff = 5;
   cfg.lost_ok = 1;
   p = Par;
   p.skip = peach_pipe_skip_pack(cfg.masks);
   p.diff = cfg.diff;
   p.epoch = 0xCAB;
   l.block = 128;
   l.grid_init = 4;
   l.grid_final = 2;
   for (a = 0; a < 8; a++) l.grid_hash[a] = 2;
   memcpy(Bufs.d_rng, Rng0, sizeof(word64) * CAP);
   scramble_buffers(SMALLCAP);
   launch_rounds(&p, &Bufs, &l, 1, &cfg, RefR, NSLOTS);
   printf("(4) small capacity run (per-kernel launches):\n");
   check_batch(&cfg, RefR, NSLOTS, &Bt, p.epoch);
   /* both bounds exercised: a round-0 queue (slot queues) and the queues
    * of a later round (packed entries) */
   for (c0 = 0, a = 0; a < 8; a++) {
      if (Bufs.d_cnt[a * PEACH_PIPE_CNTPAD] > c0) {
         c0 = Bufs.d_cnt[a * PEACH_PIPE_CNTPAD];
      }
   }
   for (cr = 0, r = 1; r <= PEACHROUNDS; r++) {
      for (tot = 0, a = 0; a < 8; a++) {
         tot += Bufs.d_cnt[((r * 8) + a) * PEACH_PIPE_CNTPAD];
      }
      if (tot > cr) {
         cr = tot;
         rmax = r;
      }
   }
   if (c0 <= SMALLCAP || cr <= SMALLCAP) {
      fail(cfg.name, -1, "round-0 and later queues must overflow");
   }
   printf("    largest round-0 queue %u, largest later round %d: %u entries"
      " (cap %d), overflow counter %u; mismatches %llu, %.2fs\n",
      (unsigned) c0, rmax, (unsigned) cr, SMALLCAP,
      (unsigned) Bufs.d_res->overflow, (unsigned long long) (Fails - fails0),
      now_s() - t0);
   ASSERT_EQ((Fails - fails0), (0));
}

/* (5) original-key and packed-entry sorts, with independent snapshots. */
static void check_sort(void)
{
   static word32 want[8][CAP], nwant[8], hist[8][PEACH_PIPE_NBUCKET];
   static word32 expected_key[CAP], keys_before[CAP];
   static PEACH_PIPE_SLOT expected[CAP], input_before[CAP];
   static word8 seen[CAP];
   static const int rounds[5] = { 1, 2, 4, 7, PEACHROUNDS };
   PEACH_PIPE_SLOT *src, *dst, poison;
   const PEACH_PIPE_SLOT *e;
   PEACH_PIPE_PARAMS p;
   PEACH_PIPE_LAUNCH l;
   CFG cfg;
   word32 m, a, i, k, id, nq, cap, ns, raw, nin, pf, *cur;
   word32 lost, expover, kept, off, n, bk, prev, tot, filled, key;
   word32 prevcnt[8];
   word64 fails0 = Fails;
   double t0 = now_s();
   char msg[96];
   int t, rr, mode, j, dead;

   memset(&cfg, 0, sizeof(cfg));
   cfg.name = "tile sort kernels";
   memset(&poison, 0xA5, sizeof(poison));
   for (t = 0; t < 20; t++) {
      rr = rounds[t % 5];
      mode = t / 5;   /* ordinary, small capacity, empty, all dead */
      nq = rr < PEACHROUNDS ? 8 : 1;
      cap = mode == 1 ? 1500 : CAP;
      raw = mode == 0 ? 2983 : mode == 1 ? 3000 : mode == 2 ? 0 : 237;
      ns = rr == 1 ? raw : 3000;
      nin = rr == 1 ? raw : raw < cap ? raw : cap;
      scramble_buffers(cap);
      memset(Bufs.d_cnt, 0, sizeof(word32) * PEACH_PIPE_CNTZERO);
      memset(Bufs.d_res, 0, sizeof(PEACH_PIPE_RESULT));
      memset(hist, 0, sizeof(hist));
      memset(nwant, 0, sizeof(nwant));
      memset(expected_key, 0xFF, sizeof(expected_key));
      memset(prevcnt, 0, sizeof(prevcnt));
      src = queue_entries(rr - 1);
      dst = queue_entries(rr);
      if (src == dst) fail(cfg.name, t, "source aliases destination");
      for (k = 0; k < ns; k++) Bufs.d_trace[k].drop_round = PEACH_PIPE_ALIVE;
      for (k = 0; k < nin; k++) {
         /* Dense positions deliberately differ from original slot IDs.
          * The clamped cases retain IDs >= cap, within physical buffers. */
         id = rr == 1 ? k : ns - 1 - k;
         expected[id].seed[0] = r32();
         expected[id].seed[1] = r32();
         expected[id].id = id;
         pf = r32() & 7;
         i = r32() % 100;
         dead = mode == 3 || i < 15;
         m = i < 35 ? r32() & ((1u << PEACH_PIPE_SORT_SHIFT) - 1) :
            i < 45 ? PEACHCACHELEN_M1 :
            i < 60 ? (0x5A5A5u & ~((1u << PEACH_PIPE_SORT_SHIFT) - 1)) |
               (r32() & ((1u << PEACH_PIPE_SORT_SHIFT) - 1)) :
               r32() & PEACHCACHELEN_M1;
         a = nq == 1 ? 0 : (r32() % 3 == 0) ? 3 : r32() & 7;
         expected[id].tile = m | (pf << PEACH_PIPE_PSHIFT);
         src[k] = expected[id];
         if (rr == 1) {
            /* First scatter must take the new tile from the original key. */
            src[k].tile = (r32() & PEACHCACHELEN_M1) |
               (pf << PEACH_PIPE_PSHIFT);
            Bufs.d_key[k] = dead ? PEACH_PIPE_KEYDEAD :
               (a << PEACH_PIPE_KEYSHIFT) | m;
         } else {
            src[k].tile |= dead ? PEACH_PIPE_ENTRYDEAD :
               a << PEACH_PIPE_NEXTSHIFT;
         }
         if (dead) continue;
         expected_key[id] = (a << PEACH_PIPE_KEYSHIFT) | m;
         hist[a][m >> PEACH_PIPE_SORT_SHIFT]++;
         want[a][nwant[a]++] = id;
      }
      if (rr > 1) {
         /* Excluded entries look live: an overlong sweep must be observable. */
         for (k = nin; k < CAP; k++) {
            src[k].seed[0] = src[k].seed[1] = WORD32_C(0xDEADBEEF);
            src[k].tile = 0;
            src[k].id = 0;
         }
         prevcnt[0] = raw / 3;
         prevcnt[2] = raw / 3;
         prevcnt[7] = raw - prevcnt[0] - prevcnt[2];
         for (a = 0; a < 8; a++) {
            Bufs.d_cnt[(((rr - 1) * 8) + a) * PEACH_PIPE_CNTPAD] = prevcnt[a];
         }
      }
      memcpy(input_before, src, sizeof(input_before));
      memcpy(keys_before, Bufs.d_key, sizeof(keys_before));
      for (a = 0; a < nq; a++) {
         memcpy(Bufs.d_cnt + PEACH_PIPE_HISTOFF + (((rr - 1) * 8 + a) *
            PEACH_PIPE_NBUCKET), hist[a], sizeof(hist[a]));
      }
      p = Par;
      p.nslots = ns;
      l.block = 32 * (1 + (t % 4));
      l.grid_init = 1 + (t % 7);
      peach_pipe_launch_sort(&p, &Bufs, &l, rr, NULL);
      for (tot = 0, a = 0; a < 8; a++) {
         k = Bufs.d_cnt[((rr * 8) + a) * PEACH_PIPE_CNTPAD];
         if (k != (a < nq ? nwant[a] : 0)) {
            snprintf(msg, sizeof(msg), "round %d cnt(%u) %u != %u", rr,
               (unsigned) a, (unsigned) k, (unsigned) nwant[a]);
            fail(cfg.name, t, msg);
         }
         if (rr > 1 && Bufs.d_cnt[(((rr - 1) * 8) + a) *
               PEACH_PIPE_CNTPAD] != prevcnt[a]) {
            fail(cfg.name, t, "incoming raw counter changed");
         }
         if (a < nq) tot += nwant[a];
      }
      expover = tot > cap ? tot - cap : 0;
      if (rr > 1 && expover != 0) fail(cfg.name, t, "dense fixture grew");
      if (rr == 1 && mode == 1 && expover == 0) {
         fail(cfg.name, t, "first-sort overflow not exercised");
      }
      if (Bufs.d_res->overflow != expover) fail(cfg.name, t, "overflow counter");
      /* Valid dense input cannot grow; source entries remain unchanged. */
      if (memcmp(src, input_before, sizeof(input_before))) {
         fail(cfg.name, t, "source entries changed");
      }
      for (lost = 0, k = 0; k < CAP; k++) {
         key = keys_before[k];
         if (k < ns) {
            if (Bufs.d_trace[k].drop_round == PEACH_PIPE_LOST) {
               lost++;
               if (rr == 1) key = PEACH_PIPE_KEYDEAD;
            } else if (Bufs.d_trace[k].drop_round != PEACH_PIPE_ALIVE) {
               fail(cfg.name, (long) k, "trace drop round");
            }
         } else if (Bufs.d_trace[k].drop_round != 0xEE) {
            fail(cfg.name, (long) k, "trace beyond original slots touched");
         }
         if (Bufs.d_key[k] != key) fail(cfg.name, (long) k, "original key");
      }
      if (lost != expover) fail(cfg.name, t, "lost slots != overflow");
      memset(seen, 0, sizeof(seen));
      for (off = 0, a = 0; a < nq; off += nwant[a], a++) {
         n = off >= cap ? 0 : nwant[a] < cap - off ? nwant[a] : cap - off;
         for (prev = 0, i = 0; i < n; i++) {
            e = &dst[off + i];
            id = e->id;
            if (id >= ns || seen[id]++ || expected_key[id] == PEACH_PIPE_KEYDEAD
                  || (expected_key[id] >> PEACH_PIPE_KEYSHIFT) != a ||
                  memcmp(e, &expected[id], sizeof(*e)) ||
                  Bufs.d_trace[id].drop_round != PEACH_PIPE_ALIVE) {
               fail(cfg.name, (long) i, "queue entry versus independent snapshot");
               continue;
            }
            bk = (e->tile & PEACHCACHELEN_M1) >> PEACH_PIPE_SORT_SHIFT;
            if (i > 0 && bk < prev) fail(cfg.name, (long) i, "tile order");
            prev = bk;
         }
         for (kept = 0, j = 0; j < (int) nwant[a]; j++) {
            id = want[a][j];
            if (seen[id]) kept++;
            else if (Bufs.d_trace[id].drop_round != PEACH_PIPE_LOST) {
               fail(cfg.name, (long) id, "slot neither queued nor lost");
            }
         }
         if (kept != n) fail(cfg.name, (long) a, "queued slots");
         cur = Bufs.d_cnt + PEACH_PIPE_CUROFF + (a * PEACH_PIPE_NBUCKET);
         for (k = 0, i = 0; i < PEACH_PIPE_NBUCKET; i++) {
            k += hist[a][i];
            if (cur[i] != k) {
               fail(cfg.name, (long) i, "cursor != bucket end");
               break;
            }
         }
      }
      filled = tot < cap ? tot : cap;
      for (k = filled; k < CAP; k++) {
         if (memcmp(&dst[k], &poison, sizeof(poison))) {
            fail(cfg.name, (long) k, "destination past stored prefix touched");
         }
      }
   }
   /* A malformed cursor forces the defensive overflow path in both
    * dense directions. It must mark the original ID, not position zero. */
   for (t = 0; t < 2; t++) {
      rr = t == 0 ? 4 : 7;
      scramble_buffers(7);
      memset(Bufs.d_cnt, 0, sizeof(word32) * PEACH_PIPE_CNTZERO);
      memset(Bufs.d_res, 0, sizeof(PEACH_PIPE_RESULT));
      src = queue_entries(rr - 1);
      dst = queue_entries(rr);
      id = 37;
      src[0].seed[0] = WORD32_C(0x12345678);
      src[0].seed[1] = WORD32_C(0x9ABCDEF0);
      src[0].tile = 42 | (5u << PEACH_PIPE_PSHIFT) |
         (3u << PEACH_PIPE_NEXTSHIFT);
      src[0].id = id;
      input_before[0] = src[0];
      Bufs.d_trace[id].drop_round = PEACH_PIPE_ALIVE;
      Bufs.d_cnt[((rr - 1) * 8) * PEACH_PIPE_CNTPAD] = 1;
      Bufs.d_cnt[PEACH_PIPE_CUROFF + 3 * PEACH_PIPE_NBUCKET +
         (42 >> PEACH_PIPE_SORT_SHIFT)] = Bufs.cap;
      p = Par;
      p.nslots = 64;
      CUDA_KERNEL(kcu_peach_pipe_scatter, 3, 96, 0, NULL)(p, Bufs, rr);
      input_before[0].tile |= PEACH_PIPE_ENTRYDEAD;
      if (Bufs.d_res->overflow != 1 ||
            Bufs.d_trace[id].drop_round != PEACH_PIPE_LOST ||
            Bufs.d_trace[0].drop_round != 0xEE ||
            memcmp(&src[0], &input_before[0], sizeof(src[0]))) {
         fail(cfg.name, rr, "injected overflow did not preserve original ID/state");
      }
      for (k = 0; k < CAP; k++) {
         if (memcmp(&dst[k], &poison, sizeof(poison)) ||
               Bufs.d_key[k] != WORD32_C(0xA5A5A5A5) ||
               (k != id && Bufs.d_trace[k].drop_round != 0xEE) ||
               (k > 0 && memcmp(&src[k], &poison, sizeof(poison)))) {
            fail(cfg.name, (long) k, "injected overflow touched other storage");
         }
      }
   }
   printf("(5) tile sort kernels: 20 runs (first/dense, both parities, "
      "clamped/empty/all-dead input) and 2 injected overflows; mismatches "
      "%llu, %.2fs\n", (unsigned long long) (Fails - fails0), now_s() - t0);
   ASSERT_EQ((Fails - fails0), (0));
}

int main(void)
{
   double t0 = now_s();

   srand16(0x1234567, 0x89abcdef, 0x2468ace0);
   memcpy(Phash, ((const BTRAILER *) Pvector)->phash, SHA256LEN);
   /* sparse guarded map, T, generation flags and batch buffers */
   Map = (word8 *) galloc(PEACHMAPLEN);
   T = (word16 *) galloc(sizeof(word16) * PEACHCACHELEN);
   Gen = (word8 *) galloc(PEACHCACHELEN);
   Rng0 = (word64 *) galloc(sizeof(word64) * CAP);
   Bufs.d_map = (const uint4 *) Map;
   Bufs.d_T = T;
   Bufs.d_rng = (word64 *) galloc(sizeof(word64) * CAP);
   Bufs.d_slot = (PEACH_PIPE_SLOT *) galloc(sizeof(PEACH_PIPE_SLOT) * CAP);
   Bufs.d_hash = (word32 *) galloc(sizeof(word32) * 8 * CAP);
   Bufs.d_q = (word32 *) galloc(sizeof(word32) * PEACH_PIPE_NQUEUE * CAP);
   Bufs.d_ent = (PEACH_PIPE_SLOT *) galloc(sizeof(PEACH_PIPE_SLOT) *
      CAP);
   Bufs.d_cnt = (word32 *) galloc(sizeof(word32) * PEACH_PIPE_CNTWORDS);
   Bufs.d_key = (word32 *) galloc(sizeof(word32) * CAP);
   Bufs.d_res = (PEACH_PIPE_RESULT *) galloc(sizeof(PEACH_PIPE_RESULT));
   Bufs.d_trace = (PEACH_PIPE_TRACE *)
      galloc(sizeof(PEACH_PIPE_TRACE) * CAP);
   Bufs.cap = CAP;

   check_helpers();
   check_init();
   check_realistic();
   check_synthetic();
   check_smallcap();
   check_sort();
   if (Fails) {
      printf("peach-pipeline: %llu checks FAILED\n",
         (unsigned long long) Fails);
      return 1;
   }
   printf("peach-pipeline: all checks passed (%llu tiles generated), "
      "%.2fs\n", (unsigned long long) TilesGenerated, now_s() - t0);

   return 0;
}
