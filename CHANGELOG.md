# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

***

## [Unreleased]

Adds a new CUDA Peach solver, the "pipeline" solver, and makes it the default for CUDA devices. The official Peach kernel stays available (`MCM_PEACH_LEGACY=1`) and is used automatically whenever the pipeline solver cannot be set up or detects a defect. The pipeline kernels pass the ptxas gate for the architectures checked in CI, and their host-path code has been verified on the CPU under emulation, against the reference implementation. On GPU hardware, the solver has been measured and tested on one card, an RTX 5090 (sm_120, 170 SMs, 32 GB, 450 W power cap): with its defaults, `peach-gpuab-cu` measures 79.0 M completed nonces/s, against 6.0 M/s for the official kernel (x13.2); the closed-source lpminer measured 73-75 M/s on the same card. There, every solve of the GPU A/B test was verified on the CPU, and compute-sanitizer (memcheck, racecheck, synccheck) reported no errors in runs that use every pipeline kernel. Other GPUs have not been measured. The other changes make the existing solver safer and add configuration, build checks and tests.

### Added
- **Pipeline solver** for CUDA devices (default). Nonces are processed in batches of N slots: an init kernel, then 8 rounds where each round runs one kernel per Nighthash algorithm over a queue of the nonces whose next jump uses that algorithm, then a final kernel that hashes the last tile and checks the difficulty. By default three batches are in flight per device, one per batch context, each on its own stream (see `MCM_PEACH_STREAMS`). Jump algorithms come from a 2 MiB transition table, built after each Peach map build. This is an exact decomposition of the floating-point algorithm selection, checked bit for bit against the reference. Per-round skip masks drop nonces whose next jump would use a given algorithm; the defaults drop MD2 jumps, which cost several other jumps, in the early rounds or in every round, depending on the device (see `MCM_PEACH_SKIP`). Each completed nonce is an independent and fully valid attempt, so dropping some changes throughput only, never the validity of a solve. The reported hash rate counts completed nonces only.
  - Safeguards: every solve is verified with `peach_checkhash()` on the CPU (including the final hash) before it is reported. The final hash of one completed nonce per batch (a canary) is checked against the CPU reference for the first 16 batches after each map build, then every 64th batch. After each map build, a self-test compares transition table entries and the device's float arithmetic with the CPU, which catches device code built with flush-to-zero or fast math. Batch results carry an epoch, and results from before a block change, an idle period or a map rebuild are discarded. A solve is reported only for a block trailer the miner can still send it for: the current one, or the one before it when the solver saw that change without a pause of the miner (gpuminer sends nothing else, and an unsendable solve would pause it for the rest of the block). Batches of an older trailer still count as work, and results are harvested at any device polling interval (`-d`). A failed verification, canary or self-test, a queue overflow, or a launch failure switches the device to the official solver (with an alert) until it is initialized again, and an unverified solve is never reported.
  - The first half of each batch's nonce is drawn again until its prefix needs no NaN replacement, which makes the algorithm selection independent of the tile index.
  - Rounds 1 to 7 and the final kernel process their queues in tile order. The hash kernels of a round count each surviving nonce in a tile-bucket histogram of its next queue (256 tiles per bucket); a scan and a scatter kernel then build the next round's queues as a counting sort by queue and tile bucket. Queue entries carry the nonce's state (16 bytes: the random number of its nonce frame, its tile, the selection op bits and its slot), so the hash kernels read their entries sequentially and the map in ascending order instead of at random, and nonces that jump to the same tile read it from cache. Round 0 stays unordered, because most nonces start at tile 0. Device memory per slot and batch context drops from 140 to 108 bytes.
  - Map tile loads carry an L2 prefetch-size hint on sm_80 and newer (`ld.global.nc.L1::evict_last.L2::256B`), so an L2 miss fetches the whole 256-byte chunk from DRAM instead of 32-byte sectors (a jump reads every byte of its tile). The SHA3 and Keccak kernels are built for at least 4 resident blocks per SM (with the hinted loads, sm_120 otherwise allocated 162 registers per thread, 3 blocks).
  - The MD2 kernel declares an occupancy target of 5 blocks per SM. Without it, ptxas fitted the kernel into 80 registers with stack spills on some architectures (for example sm_120 with CUDA 12.8), which the ptxas gate rejects. Its grid is capped at 4 resident blocks per SM: MD2 is bound by the rate of shared-memory loads, the other hash kernels by memory, and the free room on every SM lets the kernels of the other batch contexts run next to it.
  - Slots per batch (N) start at 32 per resident GPU thread for each batch context, limited by free device memory (see `MCM_PEACH_BATCH`); large batches fill the tile-ordered queues densely. Two batches in a row that take longer than 250 ms (measured from launch to completion while the batches of the other contexts share the GPU, so this bounds the GPU work in flight, which a block change waits for) halve the batch size for later batches, down to 128 x SMs; 8 batches in a row under 100 ms double it again, up to the starting size, and every map build starts again at the starting size. Only batches launched with the current batch size count, so batches still in flight from before a change cannot halve it again.
  - `peach_init_cuda_device()` logs the solver, the skip masks in use (automatic ones as `auto(cc <major>: <masks>)`), N, the batch contexts and the grid of each pipeline kernel in one line per device. If the pipeline setup fails (for example, not enough device memory), the device is still initialized, with the official solver.
  - Throughput on an RTX 5090 (sm_120 build, 450 W power cap, `peach-gpuab-cu` with 20 s per configuration, two runs, every solve verified on the CPU; M completed nonces/s): official kernel 5.97-6.03; the pipeline solver before this tuning (2 contexts, unordered queues, MD2 dropped in every round) 47.2-47.8, full evaluation 53.3-53.9; now 79.0 with the defaults (3 contexts, MD2 dropped in rounds 0-3 only), 68.0 with MD2 dropped in every round, 72.1-72.3 with full evaluation, 78.4 with 2 contexts, 76.8-76.9 with 4. The closed-source lpminer measured 73-75 M/s on the same card. Contributions measured on this card: the tile-ordered queues took the old default from 47.9 to 65.9 (2 contexts); the L2 hint adds 2-5% with every skip mask (defaults 78.2-80.6 with it, 76.5-77.3 without; a default build, `make all`, ships sm_52 code and compute_52 PTX, which the driver compiles for sm_120 without the hint: 76.5-79.5); the late-MD2 default masks measured 16% more than MD2 dropped in every round and 9% more than full evaluation. On eight other Peach maps (`PEACH_GPUAB_PHASH=1..8`), the defaults measured 77-85 M/s, 12-16% more than MD2 dropped in every round (67-76) and 8-13% more than full evaluation (71-76), except on one map where about half of the nonces jump with MD2 in round 0: there full evaluation fell to 59 while the defaults measured 82. Four contexts sharing 32 slots per resident thread (2.09M slots per batch) lost 15-17% against two full batches once the queues were tile-ordered (58.1 vs 68.3-69.9 M/s, MD2 dropped in every round), and one MD2 block per SM was too few (full evaluation, 4 contexts: 65.2 vs 71.9 with 2 blocks). In a single stream, the defaults spend 37% of the batch time in MD2 (9.9 ns per MD2 job at 4 blocks per SM) and 0.84 ns per other jump job in rounds 1-7 (1.44 ns before the tile order and the hint); the final kernel takes 0.85 ns per completed nonce (1.9 before) and the tile sort 2% of the batch.
- `peach_pipeline_cuda_device()` reports whether a device uses the pipeline solver.
- `peach_free_cuda_device()` releases a device's Peach context (device and pinned memory, streams, events) so the device can be initialized again, for example with a different configuration. Contexts whose initialization failed partway are released too.
- Peach CUDA solver configuration through environment variables. `peach_init_cuda_device()` reads them once per device and logs the result in one line per device. An invalid value is ignored with a warning and its default is used.
  - `MCM_PEACH_LEGACY=1` selects the official solver (default `0`: the pipeline solver).
  - `MCM_PEACH_SKIP=<mask>` or 8 comma-separated masks (round 0 first) set the per-round skip masks of the pipeline solver. Bit `a` drops nonces whose jump in that round would use algorithm `a`. One mask applies to all rounds. Each mask is `0`..`0xFE` (`0xFF` would drop every nonce); `0` evaluates every nonce in full. The default is automatic, per device: on compute capability 12.x (measured on an RTX 5090), MD2 jumps are dropped in rounds 0 to 3 and evaluated in rounds 4 to 7 (`0x40,0x40,0x40,0x40,0,0,0,0`); on other devices, which have not been measured, `0x40` (MD2) in every round. Early in its walk, a nonce that would jump with MD2 is cheaper to drop and replace than to finish; late in its walk, it is cheaper to finish (a drop in round 0 costs almost nothing). Where MD2 is relatively cheaper or more expensive than on the RTX 5090, other masks may be faster; `peach-gpuab-cu` compares the main candidates.
  - `MCM_PEACH_BATCH=<slots>` sets the slots per pipeline batch. The default is automatic: 32 per resident GPU thread for each batch context, limited by free device memory. Values are clamped to [128 x SMs, memory limit] and rounded to multiples of 128.
  - `MCM_PEACH_STREAMS=<n>` sets the number of pipeline batch contexts (batches in flight, one stream each), `1`..`4`. The default (`0`) is automatic: 3, or 2 when free device memory would not fit full batches for 3. On an RTX 5090, 3 contexts beat 2 by up to 2% when the solver is polled every 10 ms, as gpuminer does by default.
  - Numbers may be decimal or hexadecimal with a `0x` prefix.
- Shared CUDA/C code of the pipeline solver: `src/peach_compat.cuh` (CUDA and CPU-emulation compatibility macros), `src/peach_select.h` (decomposed algorithm selection), `src/peach_hash32.cuh` (SHA-1, SHA-256, MD5), `src/peach_hash64.cuh` (Blake2b, SHA3, Keccak) and `src/peach_hashmd2.cuh` (MD2), which hash the jump seed in place, and `src/peach_pipeline.cuh` (queues, transition table, kernels). Test support: `src/test/_cuda_emu.h` runs device code in CPU tests, and `src/test/_cuda_rt_emu.h` is a fake CUDA runtime that runs CUDA host code on the CPU.
- Tests (CPU, run without CUDA): `peach-select` checks the decomposed algorithm selection against the reference `peach_dflops()`; `peach-hash32`, `peach-hash64` and `peach-hashmd2` check the in-place jump hashes against the crypto-c hashes used by the reference; `peach-pipeline` runs the pipeline kernels end to end against the reference Peach walk.
- Test `peach-host` (CPU, opt-in: `PEACH_TEST_HOST=1 make test-peach-host NO_CUDA=1`, about a minute and 3 GiB of memory) runs the host code of `src/peach.cu` on the fake CUDA runtime, driven like gpuminer: map builds, solves, block changes, idle periods, polling intervals, fault injection and fallback, allocation failures, release and re-initialization, and several devices; every reported solve must pass `peach_check()`.
- Test `peach-gpuab-cu` (GPU only, opt-in: `PEACH_TEST_GPUAB=1 make test-peach-gpuab`, several minutes) compares legacy and pipeline solver throughput in completed nonces per second, also for pipeline variants (MD2 dropped in every round, full evaluation, MD2 dropped in rounds 0 to 3, a wider skip mask, the smallest batch size, two and four batch contexts), optionally at gpuminer's polling interval (`PEACH_GPUAB_POLL_MS=10`), on another Peach map (`PEACH_GPUAB_PHASH=<n>`), or for one configuration with the caller's `MCM_PEACH_*` settings (`PEACH_GPUAB_ONLY=<label>`, for tuning and compute-sanitizer runs), verifies every solve with `peach_checkhash()`, fails when a pipeline configuration ends up on the official solver, and skips cleanly when no CUDA device is present.
- CI workflow `.github/workflows/peach-pipeline.yaml` runs the Peach CPU tests without CUDA (`peach-host` included), a CUDA build (the library and the CUDA-side tests, which are not run), and a ptxas gate (`.github/scripts/peach-ptxas-gate.sh`). The gate fails if any `kcu_peach_pipe_*` kernel has a stack frame or register spills on sm_52 (the code of a default build), sm_61, sm_75, sm_86, sm_89, sm_90 or sm_120, or when the compute_52 PTX of a default build is compiled for sm_86 or sm_120 (as the driver does on those GPUs).

### Changed
- CUDA devices use the pipeline solver by default instead of the official Peach kernel (see Added).
- `peach_solve_cuda()` now works on one snapshot of the block trailer per call, which another thread may update concurrently. Before, a trailer update in the middle of a call could pair a new trailer with the old Peach map for a whole block. The snapshot is taken twice until both copies are equal; this catches a copy in progress, but not a writer preempted in the middle of its copy (gpuminer updates its trailer without the solver's cooperation), so it is best effort, not a consistency guarantee.
- `peach_solve_cuda()` now writes a solve to the output trailer only after `peach_checkhash()` confirms it on the CPU at the difficulty it was searched with. A rejected candidate is alerted, counted and never reported: the official solver keeps working, and the pipeline solver switches the device to the official solver (see Added). Before, one invalid solve could pause every device for the rest of the block.

### Fixed
- Build: nvcc dependency tracking (`-MMD -MP`), so header changes rebuild `peach.cu`. `make` also refuses nvcc settings that would compile device code with flush-to-zero or approximate division (for example `--use_fast_math`), because Peach requires exact IEEE-754 single precision, denormals included.
- Tests: `make test-<name>` now exits non-zero when a matching test fails or fails to build. The removed `BTSIZE` macro in tests was replaced with `sizeof(BTRAILER)`.
- A CUDA error that fails one device is no longer left in the host thread's last error, where it made the next device polled by gpuminer fail as well.

## [3.1.0-beta] - April 18th, 2026

This pre-release consolidates the F-series audit remediations merged into `master` over the past weeks. The audit covered variability-induced failures, data races, error-handling gaps, and locale/platform determinism in consensus-critical and network-handling paths. No protocol or consensus rule changes are included; all changes are behavioral corrections on existing code paths. Operators running mainnet nodes are encouraged to test this release in a non-production environment and report any regressions prior to a stable `3.1.0` tag.

### Fixed

- **F-02** (#93): `mdst_val()` now validates multi-destination fee structure against the block-declared `mfee` instead of the node-local `Myfee`, preventing divergence between nodes configured with different fee policies.
- **F-07** (#95): Ledger credit-operation 64-bit balance overflow now triggers block rejection (`VEBAD2`) instead of silently zeroing the account.
- **F-09** (#100): `scan_quorum()` quorum membership is now deterministic under peer-list shuffling — only peers whose hash matches the highest observed are included; ties no longer depend on scan order.
- **F-10** (#104): `ecode` in `validate_tfile_pow_fp()` is now `volatile`, closing the data race between the OpenMP critical-section writers and the while-loop reader.
- **F-11**: 32-bit platforms now use `long long` with `ftell64`/`fseek64` in `b_val()` for file-offset consistency with 64-bit builds.
- **F-12**: `send_tf()` no longer shells out to `dd`; uses a native C copy loop instead.
- **F-13** (#90): `tx__init()` now rejects unsupported `TXDAT_TYPE` / `TXDSA_TYPE` values instead of leaving buffer offsets uninitialized, removing a remote-triggered memory-corruption vector.
- **F-14** (#104): Transaction reference validation (`mdst_val__reference`) replaces locale-dependent `isdigit()`/`isupper()` with explicit ASCII range checks, eliminating locale-induced inter-node disagreement.
- **F-15**: Unaligned `word32` casts in `scan_quorum()` and `refresh_ipl()` replaced with `get32()`, removing undefined behavior on strict-alignment architectures.
- **F-16**: Unaligned `word32` casts in `txmap()` and `mgc()` replaced with `get32()`/`put32()`.
- **F-18** (#92): `recv_file()` now caps cumulative bytes at 1 GiB (`MAX_RECV_FILE_BYTES`), preventing a malicious peer from exhausting local disk via unbounded `OP_SEND_FILE` streaming.
- **F-06**: `system()` calls in `sync`/`bup` replaced with native C file operations.
- **F-20**: Remaining `system()` calls in `bup`/`sync` now check return values, ensuring failed external-script steps do not silently pass.
- **F-21**: `syncup()` only updates `Weight` on successful `weigh_tfile()`, preventing corrupt weight promotion on tfile read errors.
- **F-23** (#99): `read_tfile()` now returns `0` on `fopen()` failure instead of `VERROR` (==1), which was previously indistinguishable from a legitimate one-trailer read. Pre-existing TODO in `send_found()` wired up to check `count != NTFTX`.
- **F-26**: Dead double-read of `OP_FOUND` proof in `refresh_ipl()` removed.
- **F-27**: Unaligned cast for `srand32()` seed in `main` replaced with `memcpy`.
- **F-29** (#92): `get_ipl()` rejects oversized peer-list responses from peers (DoS mitigation).
- **F-30** (#110): `scan_quorum()` OpenMP shared-variable data race on `result` resolved by splitting into per-thread-private weight-comparison and contribution-counting variables. Matching refactor applied to `catchup()` in `sync.c`.
- **F-31** (#117): `get_hash()` now closes the socket on every exit path via a single `CLEANUP:` label, fixing a file-descriptor leak on `send_op()`/`recv_tx()` failure.
- **F-32** (#112): `catchup()` shared `count` read race closed via a local `volatile` shadow, pairing with the OpenMP flush on critical-exit.

### Notes

- **F-01** (#86) and **F-04** (#94) were analyzed and closed as not-a-bug; see the respective GitHub issues for rationale.
- **F-08** (#98) was reclassified as a feature request (transaction rate-limiting) and tracked separately.

## [3.0.3] - March 6th, 2025

This release focuses on improving network synchronization, build system flexibility, and fixing various issues identified in previous versions. Key improvements include enhanced node synchronization procedures, better network scanning when network size grows, and several build system enhancements for greater flexibility and debugging capabilities.

### Changed
- Improved node (re)synchronization and catchup procedures
- Updated setup script to prune old branches during git updates
- Removed forced compiler selection (CC/GCC) for better build flexibility
- Enhanced fallback mechanism for NVCC in build system

### Fixed
- Issue with selecting specific (tag) version in setup script
- Network scan limitation when network size grows beyond RPLISTLEN
- Incorrect sorting on transaction destinations

## [3.0.2] - February 11th, 2025

This release focuses on improving the build system compatibility and CI/CD workflows. Key changes include build fixes for Ubuntu 20.04 systems and refined GitHub Actions workflows for better release management and code quality checks.

### Added
- Release workflow for attaching miner binary to GitHub releases
- Build system compatibility with Ubuntu 20.04
  - Added libdl and librt dependencies for cudart_static

### Changed
- Refined CI/CD workflow triggers
  - Build workflows now only trigger on version tags, pull requests, and manual triggers
- Updated CodeQL analysis
  - Upgraded to CodeQL Action v3
  - Configured to use security-and-quality suite only
  - Enhanced query suite options for database analysis

## [3.0.1] - February 10th, 2025

The latest update focuses on improving GPU mining efficiency and network performance. Key improvements include moving (last half) nonce generation to GPU with parallel PRNG implementation, which significantly reduces CPU usage on multi-GPU systems. Network scanning was enhanced with better thread utilization and increased peer sharing capabilities. The update also includes several CUDA-specific improvements, such as error handling, device counting fixes, and migration to static runtime library for the miner. Development workflows were streamlined with individual build targets and updated CI/CD runners. Non-essential features like NVML support and testnet troubleshooting code were removed to improve codebase maintainability.

### Added
- Device launch parameters header for CUDA
- Individual build target workflows in CI/CD
- Links to Wallet and API releases in README
- Manual execution instructions in README

### Changed
- Increased peer sharing capabilities beyond 32 peers
- Improved CPU efficiency of GPU miner
  - Moved nonce generation to GPU with parallel PRNG
  - Better device handling and status output
  - Stabilized hashrate display
- Enhanced network scanning with better thread utilization
- Updated CI/CD runners to latest versions
- Changed CUDA runtime library to static version

### Fixed
- CUDA-specific error handling and checking
- GPU miner (makefile) target dependencies
- Sudo handling in setup script
- CUDA device counting when no GPUs present
- Bridge time check for GPU IDLE->WORK mode

### Removed
- NVML support (wasn't providing useful data)
- Testnet troubleshooting from production code

## [3.0.0] - February 2nd, 2025

Major improvements to Mochimo Addresses including Hash-based Leadger formatting, Base58 error checking perpetual account tags, and UX for account management. Improvements to transactions with the standard transaction capable of 256 destinations, each with their own reference, and easy implementation of additional Digital Signature Algortihms. Improvements to Merkle Root hash allowing for development of Transaction Receipts to validate a transaction was part of the chain without having access to the block data. Improvements to chain linkage and Tfile validation procedures. Improvements to network bandwidth with Variable (sized) Protocol Data Units.

## [2.4.3] - June 1st, 2023

Service setup updates and improvements to miner for RTX4090 and future performance advancement of GPU architectures.

### Changed
- mochimo.service now waits for network to be available before starting
- setup script no longer install erroneous amounts of git software (installs `git` instead of `git-all`)
- moved a reasonable amount of CPU work to the GPU to improve performance of powerful GPUs on low tier hardware (specifically the RTX4090, but will also apply to future generations of GPU)

## [2.4.2] - July 12th, 2022

Implementation of the Adequate Systems Build-C repository for CI/CD processes and build utilities. Repository restructure in preparation for version 3.0 improvements. Some non-critical bug fixes related to the handling of transactions between block updates. Improvements to Peach POW Algorithm on all Cuda capable devices.

### Added
- introduced CI/CD and build utilities "merged" in from `build-c`
  - includes LICENSE.md (detectable repository license)
- added some basic unit tests for testing mochimo components
- added codebase modules `crypto-c` and `extended-c`
- added segfault tracing within UNIX
- added `--testnet` generation option to server binary
- added "trusted" peers (for post v3.0 Hi-Speed capabilities)
- added sleep time to mirror child while waiting for grand-children
- added improved miner binary with combined solo and pool capabilities
- added duplicate process detection mechanism (for detecting duplicate mochimo servers)
- added SO_REUSEADDR to listening socket options
- added setup.x script for easy provisioning and testing of mochimo nodes
- added system service registration to setup.x for easy restart capability
- enhanced splashcreen version identification:
  - `v<major>.<fork>.<minor>-<patch>-g<commit>-[dirty]`
### Changed
- moved/revised github specific templates and standards
- moved `LICENSE.PDF` to `.github/` (github detectable LICENSE.md remains)
  - NOTE: https://mochimo.org/license.pdf redirects appropriately
- updated network routines for compatibility with `extended-c` module
- replaced polymorphic shell sort in favor of Standard C's qsort()
- reimplemented sorting, validating and updating routines such that they may be executed within the calling process
- consolidated same type routines into compilation units
- updated print/logging functions to use the `extended-c` module
- updated server exit procedure
- improvements to the Peach POW algorithm
- reorganization of utility functions (util.c)
- moved most of the gomochi setup checks into main binary
- README with updates to repository
- `.gitignore` exclusions
### Fixed
- fixed corner-case where transactions may be missed during txclean() routines and subsequently caught during validation
- fixed illegal memory access errors typical on 30-series Cuda devices
### Removed
- removed code that was modularized in `crypto-c` and `extended-c`
- removed debilitating memcpy() calls in the Peach Algo for Cuda devices

## [2.4.1] Mochimo Patch Level 37

### Added
- byte Insyncup;  /* non-zero when syncup() runs */ to data.c
- Insyncup to syncup() in syncup.c
- addrlen to le_find() in ledger.c
- Tagidx[] to tag.c
- tag_free() to tag.c, update.c
- proper time_t vtime for vstart.lck check in server.c
- Insyncup to update() in update.c to help plog()'s
- char *solvestr to update() to reckon pushed, solved, and updated blocks
- Syncup Function, Removing Contention
- mochimo/bin/d/split directory
- send_found() to syncup.c
- Send found message to low weight peer in refresh_ipl()
- MTXTRIGGER to bval.c mtxval.c txclean.c txval.c config.h
- system call to init-external.sh on initial system sync
- system call to update-external.sh on successful block update
- advanced support for block-explorer export functionality
- support for third-party utility triggered system restart()
- directories mochimo/src/test and mochimo/src/old for testing this build
- weight checks to contention() and checkproof()
- functions sub_weight(), and past_weight() to proof.c
- return code to send_found() in update.c
- sftimer for send_found() in server.c
- vstart.lck restart trigger (Verisimility) to server.c

### Changed
- PATCHLEVEL to 37 in mochimo.c and minertest.c
- VERSIONSTR to "37" in mochimo.c and minertest.c
- tag_find() in tag.c, txclean.c, and mtxval.c
- pval.c server.c, and config.h to allow 0xff pseudo-blocks
- simplified parameter logic in tag_valid() and calls from bval and txval.c
- bx.c to indicate tags not found in MTX
- replaced 100 with MDST_NUM_DST in bval.c mtxval.c txclean.c
- proof.c comments
- syncup.c fprintf's to plog's
- server.c ipltime from 600 to 300
- checkproof() in proof.c
- comments and plog's in gettx.c
- put back V23TRIGGER check in checkproof() in proof.c
- tried to improve comments and plog() messages.
- swapped Bail(1) and Bail(2) to avoid recomputing past_weight() if first trailer doesn't match
- Improved Code readability / comments for checkproof() bail conditions 6 & 7

### Removed
- bigwait from server.c
- FILE *rlog from syncup()

### Fixed
- previous hash check in checkproof() in proof.c
- missing error return lines in bval2() in gettx.c
- Bugs in checkproof()
- bug in past_weight() to skip NG blocks
- bug in checkproof() to skip Difficulty check on init
- bug in syncup() w/first previous NG block
- bug in syncup() to skip NG blocks

## [2.4.0] Mochimo Patch Level 34

### Added
- FPGA-Resistant Algorithm
  - Extensible High-Memory Algo v24()
- Multi-Destination Transactions
  - Scales up high volume third-party payment systems

### Security

- fixes and networking tweaks

## [2.3.0] Mochimo Patch Level 33

### Added
- Pseudoblocks for mid-block difficulty adjustment
  - Impossible for blocks to exceed 15m49s after v2.3
- security fixes to MROOT creation
  - Removes certain spoofing attack vectors
- support for ZERO-tag OP_BAL Requests
  - Allows address lookup and balance query without TAG
  - Allows wallet recovery from seed phrase
- TXCLEAN Queue Re-validation following block updates
  - Prevents known cornercase block validation failures
  - Prevents all attack vectors involving poisoned TXs in the TXCLEAN queue
- low-balance pruning consensus mechanism (CAROUSEL)
  - Allows the community to clean low balances out of the ledger by consensus
  - Keeps the blockchain free of bloat, and recovers from spam attacks
- optional mining fee adjustment per miner
  - Allows future miners to create a mining fee market after mining rewards are gone
- TFILE PoPOW Chain in OP_FOUND
  - Allows nodes to definitively confirm an advertising node really solved a block
- Watchdog timer to restart and resync if no blocks solved or updated in 30 minutes +/- 10 minutes
  - Prevents Block of Death, Stuck at 0x0 events caused by temporary internet outage for nodes
  - Allows nodes to dynamically recover from any number of possible failure cases
- Upload Bandwidth limit (default = 5MB/s Upload, user configurable)
  - Prevents certain kinds of spam attacks against nodes
- NOMINER feature, initialized with -n at runtime
  - Allows a node to run in relay mode only without mining blocks
- command line compile options for CPU or GPU, merging both development branches
  - Involed with ./makeunx bin -DCPU  -or-  ./makeunx bin -DCUDA

### Changed
- inbound TX uniqueness test (CRC) with address-based validation
  - Allows the system to scale past 65,536 TXs per block at some indeterminate future date

## [2.2.0] Mochimo Patch Level 32

### Added
- support for diverse nVidia GPU Models
- OPCODE 15 & 16 (pull candidate / push solved blocks)
- support for Headless Miners and Mining Pools
- node capability bits to identify server capabilities during handshake
- new execute and dispatch functions for handling headless miner requests
- new reaping function for terminating stale child processes related to headless miners

### Changed
- CUDA Code optimizations for average HP/s +10-20%
- improved sanity checking in get_eon (prep for ring signatures)
- adjusted wait time up to 300 seconds from 180 if no quorum found

### Fixed
- random seed issue in rand.c
- various community requested patches


## [2.0.0] Mochimo Patch Level 31

October 27th, 2018

### Added
- added new open source license
- added trigger block for new weight calculation as 17185 (0x4321)
- added trigger block for new reward calculation as 17185 (0x4321)
- added trigger block for new difficulty calculation as 17185 (0x4321)
- added trigger block for tag system validation checks as 17185 (0x4321)
- added dynamic start nodes list download from mochimap.net
- added tag.c, tag related fixes throughout
- added wallet Build 31 with tag support

### Changed
- update system version number to 2
- updated default coreip.lst
- reorganized source code distro in prep for Github
- Adjusted TXVAL to insist src addresses must fully spent
- Enabled balance forwarding to change address
- bup.c
  - patch a bunch of stuff
  - balances debit '-' first, then credit 'A'
- bval.c
  - trancodes: '-' and 'A'
  - enforces no tag on bh.maddr
  - tag mods
  - permanent future time fix
- bupdata.c
  - new set_difficulty() with preset trigger
  - new set_difficulty() block trigger = 16383
- gomochi
  - sleep set to 1 second
  - added dynamic startnodes.lst download
- init.c
  - new add_weight() improved block weight fork on block trigger
  - new add_weight() -DNEWWEIGHT forks chain on block trigger
  - get_eon(): timeout set to 180
  - modified read_coreipl() and init_coreipl()
  - permanent future time fix
- data.c: #define CORELISTLEN 16
- gettx.c: added contention(), catchup(), and bval2()
- server.c: removed LULL timer
- txclean.c: fixed unlink(argv parameter 1) bug
- util.c: new get_mreward() on block trigger

### Removed
- removed default maddr.dat
- removed txq1.lck (process_tx() is now synchronous)

[Unreleased]: https://github.com/mochimodev/mochimo/compare/v3.0.3...HEAD
[3.0.3]: https://github.com/mochimodev/mochimo/compare/v3.0.2...v3.0.3
[3.0.2]: https://github.com/mochimodev/mochimo/compare/v3.0.1...v3.0.2
[3.0.1]: https://github.com/mochimodev/mochimo/compare/v3.0.0...v3.0.1
[3.0.0]: https://github.com/mochimodev/mochimo/compare/v2.4.3...v3.0.0
[2.4.3]: https://github.com/mochimodev/mochimo/compare/v2.4.2...v2.4.3
[2.4.2]: https://github.com/mochimodev/mochimo/compare/v2.4.1...v2.4.2
[2.4.1]: https://github.com/mochimodev/mochimo/compare/v2.4...v2.4.1
[2.4.0]: https://github.com/mochimodev/mochimo/compare/v2.3...v2.4
[2.3.0]: https://github.com/mochimodev/mochimo/compare/v2.2...v2.3
[2.2.0]: https://github.com/mochimodev/mochimo/compare/v2.1...v2.2
[2.1.0]: https://github.com/mochimodev/mochimo/compare/v2.0...v2.1
[2.0.0]: https://github.com/mochimodev/mochimo/releases/tag/v2.0
