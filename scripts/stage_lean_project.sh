#!/usr/bin/env bash
# Copy the pinned Lean project to node-local storage and print the staged path.
#
#   STAGED=$(scripts/stage_lean_project.sh ~/scratch/lean/mathlib_v4160) || STAGED=~/scratch/lean/mathlib_v4160
#   python scripts/prove_benchmark.py --lean-project "$STAGED" ...
#
# ## Why
#
# The project lives on CSF3's NFS home/scratch. `import Mathlib` reads ~5.4 GB of `.olean` files
# across ~14k files, and it is that read — not CPU — that dominates warm-up:
#
#   | where                        | `import Mathlib` |
#   |------------------------------|-----------------:|
#   | laptop, local SSD            |        90-160 s  |
#   | CSF3 `multicore`, 4 cores    |          439 s   |
#   | CSF3, 1 core                 |          691 s   |
#   | CSF3 node855, contended NFS  |     >900 s (died)|
#
# The tail is set by whatever else is hitting the shared filesystem, i.e. by other users' jobs.
# Every search worker pays this once, so a 16-worker job spends ~2.5 core-hours before proving
# anything. Copying once to node-local disk turns ~14k scattered random reads into one sequential
# stream, which NFS serves far better, and every worker on the node then reads from local disk.
#
# ## Design constraints
#
#   * **It must never make things worse.** Any failure — no local disk, not enough space, a copy
#     error — exits non-zero *without* having touched anything, so the caller falls back to the
#     NFS path. Staging is an optimisation, and an optimisation that can break a run is a bug.
#   * **Concurrent workers on one node must not race.** Workers share `$TMPDIR`, so the copy is
#     guarded by an flock and a completion marker: one worker copies, the rest wait and reuse.
#   * **A partial copy must never be reused.** The marker is written last and records the source's
#     size, so an interrupted copy is detected and redone rather than silently half-used.
#
# Print the staged path on stdout and nothing else — the caller captures it. Diagnostics go to
# stderr.

set -euo pipefail

SRC="${1:-${PROOFLENS_LEAN_PROJECT:-$HOME/scratch/lean/mathlib_v4160}}"
SRC="${SRC/#\~/$HOME}"

log() { printf '[stage-lean] %s\n' "$*" >&2; }
die() { log "NOT STAGING: $*"; log "caller should fall back to $SRC"; exit 1; }

[ -d "$SRC" ] || die "source project not found: $SRC"

# `$TMPDIR` is node-local on CSF3 and is cleaned up when the job ends. Fall back to /tmp for
# interactive testing, but never to anything under $HOME — that would be NFS again, i.e. all of the
# copy cost and none of the benefit.
LOCAL_ROOT="${TMPDIR:-/tmp}"
case "$LOCAL_ROOT" in
  "$HOME"/*) die "TMPDIR ($LOCAL_ROOT) is under \$HOME, so it is NFS — staging would gain nothing" ;;
esac
[ -d "$LOCAL_ROOT" ] && [ -w "$LOCAL_ROOT" ] || die "no writable local dir at $LOCAL_ROOT"

DEST="$LOCAL_ROOT/prooflens_lean/$(basename "$SRC")"
MARKER="$DEST/.prooflens_staged"

need_kb=$(du -sk "$SRC" 2>/dev/null | cut -f1) || die "cannot size $SRC"
# 15% headroom: Lean writes nothing here, but a full filesystem fails in confusing ways.
want_kb=$(( need_kb * 115 / 100 ))
free_kb=$(df -Pk "$LOCAL_ROOT" | awk 'NR==2 {print $4}')
log "source $(( need_kb / 1024 )) MB | free on $LOCAL_ROOT $(( free_kb / 1024 )) MB"
[ "$free_kb" -ge "$want_kb" ] || die "need $(( want_kb / 1024 )) MB, only $(( free_kb / 1024 )) MB free"

mkdir -p "$(dirname "$DEST")"

# One worker copies; the others block here and then reuse the result. flock is released when the
# subshell's fd 9 closes, including if the copier is killed.
LOCK="$LOCAL_ROOT/prooflens_lean/.lock"
exec 9>"$LOCK"
flock 9

if [ -f "$MARKER" ] && [ "$(cat "$MARKER" 2>/dev/null)" = "$need_kb" ]; then
    log "already staged at $DEST (marker matches source size)"
    echo "$DEST"
    exit 0
fi

# A stale or partial copy is worse than none: remove it rather than syncing on top.
if [ -e "$DEST" ]; then
    log "removing incomplete or outdated staging at $DEST"
    rm -rf "$DEST"
fi

log "copying $SRC -> $DEST (one sequential pass; several minutes)"
t0=$SECONDS
mkdir -p "$DEST"
# tar-to-tar rather than `cp -a`: it streams, so the read side stays sequential instead of
# interleaving stat/open per file, which is what NFS handles badly.
if ! tar -C "$SRC" -cf - . | tar -C "$DEST" -xf -; then
    rm -rf "$DEST"
    die "copy failed"
fi

# Written last, so its presence means the copy finished. Contents are the source size, so a source
# that changed since staging invalidates the cache.
echo "$need_kb" > "$MARKER"
took=$(( SECONDS - t0 ))
log "staged in ${took}s"

# Staging is an optimisation and it can lose. It buys ~500 s per `import Mathlib` (439-691 s from
# NFS against 158 s node-local), and a run pays that import a handful of times — once for the header
# plus once per REPL restart. Measured staging times on this cluster range from a few minutes to
# 2624 s, entirely depending on NFS contention, so the trade is decided by the day rather than by
# the design. Say so when it clearly went badly: a short run would have finished sooner without it.
if [ "$took" -gt 900 ]; then
    log "WARNING: staging cost ${took}s, more than the ~500s it saves per Mathlib import."
    log "         For a short run (a smoke test, or --limit under ~20) pass STAGE_LEAN=0 and read"
    log "         Mathlib from NFS instead. For a full benchmark it still pays off, because olean"
    log "         reads continue during elaboration, not only at import."
fi
echo "$DEST"
