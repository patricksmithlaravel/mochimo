#!/bin/bash
##
# peach-ptxas-gate.sh - ptxas resource gate for the Peach pipeline kernels.
# Copyright 2025 Adequate Systems, LLC. All Rights Reserved.
#
# Usage (from the repository root, submodules initialized):
#    .github/scripts/peach-ptxas-gate.sh [<target> ...]
#
# A target is an architecture <sm> (e.g. 86: src/peach.cu compiled for
# sm_<sm>), or <virt>:<sm> (e.g. 52:86: the compute_<virt> PTX of
# src/peach.cu compiled by ptxas for sm_<sm>, as the driver JIT does when
# a binary without sm_<sm> code runs on such a GPU; a build without
# -arch, like `make all`, ships sm_52 code and compute_52 PTX only).
# Default targets: 52 61 75 86 89 90 100 120 52:86 52:120.
# Each target is compiled with ptxas verbose output, and every
# kcu_peach_pipe_* kernel is checked: a non-zero stack frame, spill store
# or spill load fails the gate. Other kernels (the official kcu_peach_*
# kernels) are listed, not gated.
#
# Environment:
#    NVCC                    nvcc to use (default: nvcc on PATH, else
#                            /usr/local/cuda/bin/nvcc)
#    PTXAS                   ptxas to use for <virt>:<sm> targets (default:
#                            ptxas next to nvcc, else ptxas on PATH)
#    PEACH_GATE_MIN_KERNELS  minimum number of kcu_peach_pipe_* kernels
#                            required per target (default 0: none
#                            found is reported and passes)
#
# Exit status: 0 pass, 1 gate failure, 2 usage or compile error.
#

set -u

NVCC=${NVCC:-$(command -v nvcc || echo /usr/local/cuda/bin/nvcc)}
MIN=${PEACH_GATE_MIN_KERNELS:-0}
ARCHS=("$@")
if [ ${#ARCHS[@]} -eq 0 ]; then
   ARCHS=(52 61 75 86 89 90 100 120 52:86 52:120)
fi
PTXAS=${PTXAS:-$(dirname "$(command -v "$NVCC" || echo "$NVCC")")/ptxas}
if [ ! -x "$PTXAS" ]; then PTXAS=$(command -v ptxas || echo ptxas); fi

if [ ! -f src/peach.cu ]; then
   echo "error: run from the repository root (src/peach.cu not found)" >&2
   exit 2
fi
if [ ! -x "$NVCC" ] && ! command -v "$NVCC" >/dev/null 2>&1; then
   echo "error: nvcc not found ($NVCC)" >&2
   exit 2
fi
case "$MIN" in
   ''|*[!0-9]*) echo "error: invalid PEACH_GATE_MIN_KERNELS=$MIN" >&2
      exit 2 ;;
esac

# include directories as in GNUmakefile: src and submodule sources
INCS=(-Isrc)
for d in include/*/src; do [ -d "$d" ] && INCS+=("-I$d"); done

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

STATUS=0
for TARGET in "${ARCHS[@]}"; do
   case "$TARGET" in
      *:*) VIRT=${TARGET%%:*}; SM=${TARGET#*:}
         case "$VIRT" in
            ''|*[!0-9]*) echo "error: invalid target '$TARGET'" >&2; exit 2 ;;
         esac ;;
      *) VIRT=""; SM=$TARGET ;;
   esac
   case "$SM" in
      ''|*[!0-9]*) echo "error: invalid target '$TARGET'" >&2; exit 2 ;;
   esac
   if [ -z "$VIRT" ]; then
      LABEL="sm_$SM"
      LOG="$TMP/ptxas_sm$SM.log"
      echo "== $LABEL: nvcc -c src/peach.cu -Xptxas -v"
      if ! "$NVCC" -c src/peach.cu -o "$TMP/peach_sm$SM.o" "${INCS[@]}" \
            -Xptxas -Werror -Xptxas -v -Wno-deprecated-gpu-targets \
            -gencode "arch=compute_$SM,code=sm_$SM" > "$LOG" 2>&1; then
         cat "$LOG" >&2
         echo "error: compilation for $LABEL failed" >&2
         exit 2
      fi
   else
      LABEL="compute_${VIRT}->sm_$SM"
      LOG="$TMP/ptxas_c${VIRT}_sm$SM.log"
      PTX="$TMP/peach_compute$VIRT.ptx"
      echo "== $LABEL: nvcc -ptx src/peach.cu, then ptxas -v"
      if [ ! -f "$PTX" ] && ! "$NVCC" -ptx src/peach.cu -o "$PTX" \
            "${INCS[@]}" -Wno-deprecated-gpu-targets \
            -arch="compute_$VIRT" > "$LOG" 2>&1; then
         cat "$LOG" >&2
         echo "error: PTX compilation for compute_$VIRT failed" >&2
         exit 2
      fi
      if ! "$PTXAS" -arch="sm_$SM" -v -Werror "$PTX" \
            -o "$TMP/peach_c${VIRT}_sm$SM.cubin" > "$LOG" 2>&1; then
         cat "$LOG" >&2
         echo "error: ptxas of the compute_$VIRT PTX for sm_$SM failed" >&2
         exit 2
      fi
   fi
   # ptxas -v reports, per function:
   #    ptxas info    : Function properties for <name>
   #        <n> bytes stack frame, <n> bytes spill stores, <n> bytes spill loads
   #    ptxas info    : Used <n> registers, ...
   awk -v sm="$LABEL" -v min="$MIN" '
      function report() {
         if (fn == "") return
         gated = (fn ~ /kcu_peach_pipe_/)
         if (gated) {
            nk++
            bad = (stack != 0 || stores != 0 || loads != 0)
            if (bad) nbad++
         } else if (fn ~ /kcu_/) {
            bad = 0
         } else {
            fn = ""; return
         }
         printf("   %-6s %-60s stack %5d  spill st %5d  ld %5d  regs %s\n",
            gated ? (bad ? "FAIL" : "ok") : "info", fn, stack, stores,
            loads, regs)
         fn = ""
      }
      /Function properties for / {
         report()
         fn = $NF; stack = stores = loads = 0; regs = "?"; next
      }
      fn != "" && /bytes stack frame/ {
         n = split($0, f, ",")
         stack = f[1] + 0; stores = (n > 1) ? f[2] + 0 : 0
         loads = (n > 2) ? f[3] + 0 : 0; next
      }
      fn != "" && /Used [0-9]+ registers/ {
         match($0, /Used [0-9]+ registers/)
         regs = substr($0, RSTART + 5, RLENGTH - 15)
         report(); next
      }
      END {
         report()
         if (nk == 0) {
            printf("   %s: no kcu_peach_pipe_* kernels found", sm)
            if (min > 0) {
               printf(" (%d required) -> FAIL\n", min); exit 1
            }
            printf(" (pipeline kernels not built yet) -> nothing to gate\n")
            exit 0
         }
         printf("   %s: %d kcu_peach_pipe_* kernel(s), %d with %s\n",
            sm, nk, nbad, "stack frame or spills")
         if (nk < min) {
            printf("   %s: fewer than %d kernels -> FAIL\n", sm, min)
            exit 1
         }
         exit (nbad > 0 ? 1 : 0)
      }
   ' "$LOG" || STATUS=1
done

if [ $STATUS -ne 0 ]; then
   echo "ptxas gate: FAILED (see the FAIL lines above)"
else
   echo "ptxas gate: passed (${ARCHS[*]})"
fi
exit $STATUS
