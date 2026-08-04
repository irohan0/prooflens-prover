#!/usr/bin/env bash
# Create the pinned Lean 4 + Mathlib project that every prover run interacts with.
#
# Run this ONCE per machine (laptop / CSF3). It never compiles Mathlib from source:
# 'lake exe cache get' downloads pre-built .olean files, turning an hours-long build into a few
# minutes of download. Compiling Mathlib on a shared cluster node is the classic way to burn an
# allocation for nothing.
#
# Usage:
#   scripts/setup_lean_project.sh [TARGET_DIR] [LEAN_VERSION]
# Example:
#   scripts/setup_lean_project.sh ~/scratch/lean/prooflens_mathlib v4.31.0
#
# The project files are written directly rather than via 'lake new <name> math', because Lake's
# template names and lakefile format (.toml vs .lean) have shifted across releases; writing them
# ourselves keeps this script working across the Lean versions LeanInteract supports
# (v4.8.0 - v4.32.0-rc1).
#
# NOTE ON CSF3: compute nodes have outbound network (established in the predecessor project,
# results/phase_logs/phase20.md), so this *can* run in a batch job — but running it once on a
# login node and letting every later job reuse the built project from scratch storage is cheaper.

set -euo pipefail

# ---------------------------------------------------------------------------------------------
# SELF-SANITISING RE-EXEC. Do not remove.
#
# Mathlib's cache tool downloads ~8,500 files with curl. In a conda-touched shell, curl loads
# conda's libssl and every download fails with:
#   OpenSSL/3.0.8: error:16000069:STORE routines::unregistered scheme
#
# `conda deactivate` is NOT sufficient: it drops conda from PATH but leaves LD_LIBRARY_PATH
# pointing at conda's libs, so even /usr/bin/curl loads conda's OpenSSL at runtime.
#
# The earlier fix was "prefix the command with env -u LD_LIBRARY_PATH ...". Requiring an operator
# to prefix every invocation is not a fix: it failed in practice the first time the prefix was
# forgotten, after a 40-minute wait. The script now re-executes itself in a clean environment, so
# it is correct however it is invoked.
# ---------------------------------------------------------------------------------------------
# `env -i` (start from an EMPTY environment), not `env -u` (remove named variables). An earlier
# version listed the variables to unset; it failed anyway, because a blacklist only removes what its
# author thought of. Anything a site profile, module system or conda hook injects that is not on the
# list survives. Whitelisting is the only form of this that is actually a control.
if [ -z "${PROOFLENS_SANITISED_ENV:-}" ]; then
  exec env -i \
    PROOFLENS_SANITISED_ENV=1 \
    HOME="$HOME" \
    USER="${USER:-$(id -un)}" \
    TERM="${TERM:-dumb}" \
    LANG="${LANG:-C.UTF-8}" \
    PATH="/usr/bin:/bin:/usr/local/bin:$HOME/.elan/bin" \
    bash "$0" "$@"
fi

# Default to v4.16.0: that is REAL-Prover's version, and the Tier-1 benchmarks only fully
# elaborate there (141/141 FATE-M, 244/244 miniF2F vs 138 and 212 on v4.31.0).
TARGET_DIR="${1:-$HOME/scratch/lean/mathlib_v4160}"
LEAN_VERSION="${2:-v4.16.0}"

echo "=== ProofLens-Prover Lean project setup ==="
echo "target : $TARGET_DIR"
echo "lean   : $LEAN_VERSION"

# -- preflight: can curl actually do TLS? ------------------------------------------------------
# Mathlib's cache tool downloads ~8,500 files with curl. If curl cannot resolve a CA bundle,
# EVERY download fails — but only after it has tried all of them, so the failure surfaces ~40
# minutes in as "8542 download(s) failed" with an opaque OpenSSL message:
#
#   Transfer failed (error code: 0): OpenSSL/3.0.8:
#   error:16000069:STORE routines::unregistered scheme
#
# The usual cause is a conda environment earlier on PATH: its curl links against conda's OpenSSL,
# which does not see the system certificate store. Git clones still work (git uses its own TLS
# config), so the environment looks healthy right up until the cache step. Check it in 5 seconds
# instead, and repair it in place where we can.
CACHE_HOST="https://lakecache.blob.core.windows.net/mathlib4/"

tls_ok() { curl -sSI --max-time 20 "$CACHE_HOST" >/dev/null 2>&1; }

echo "--- preflight: curl TLS to the Mathlib cache ---"
if tls_ok; then
  echo "curl TLS OK ($(command -v curl))"
else
  echo "curl TLS FAILED with $(command -v curl) — attempting repair"
  # 1) drop CA overrides that point somewhere curl/OpenSSL cannot parse
  unset SSL_CERT_FILE SSL_CERT_DIR REQUESTS_CA_BUNDLE CURL_CA_BUNDLE || true
  # 2) drop LD_LIBRARY_PATH. THIS IS THE ONE THAT IS EASY TO MISS: `conda deactivate` removes
  #    conda from PATH but leaves LD_LIBRARY_PATH pointing at conda's libs, so even /usr/bin/curl
  #    still loads conda's libssl at runtime and keeps failing. A reported OpenSSL version that
  #    does not match the distro's is the tell.
  if [ -n "${LD_LIBRARY_PATH:-}" ]; then
    echo "  clearing LD_LIBRARY_PATH (was: $LD_LIBRARY_PATH)"
    unset LD_LIBRARY_PATH
  fi
  # 3) prefer the system curl over a conda/venv one
  if [ -x /usr/bin/curl ]; then export PATH="/usr/bin:$PATH"; fi
  # 3) point explicitly at whichever system CA bundle exists (RHEL vs Debian layouts)
  for ca in /etc/pki/tls/certs/ca-bundle.crt /etc/ssl/certs/ca-certificates.crt \
            /etc/ssl/cert.pem /etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem; do
    [ -r "$ca" ] && export CURL_CA_BUNDLE="$ca" && break
  done
  if tls_ok; then
    echo "repaired: curl=$(command -v curl) CURL_CA_BUNDLE=${CURL_CA_BUNDLE:-<default>}"
  else
    cat >&2 <<EOF

SETUP FAILED (preflight) — curl cannot establish TLS to the Mathlib cache.

Every one of the ~8,500 cache downloads would fail. Stopping now rather than after ~40 minutes.

  curl in use : $(command -v curl)
  version     : $(curl --version 2>&1 | head -1)
  linked ssl  : $(ldd "$(command -v curl)" 2>/dev/null | grep -i ssl | tr -s ' ' | head -2 | tr '\n' ' ')
  LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-<unset>}
  CURL_CA_BUNDLE=${CURL_CA_BUNDLE:-<unset>}  SSL_CERT_FILE=${SSL_CERT_FILE:-<unset>}

Most common cause: a conda environment. Note that 'conda deactivate' is NOT enough -- it leaves
LD_LIBRARY_PATH pointing at conda's libs, so even /usr/bin/curl keeps loading conda's libssl.
An OpenSSL version above that does not match your distro's is the tell.

Run the whole setup in a sanitised environment instead:

  env -u LD_LIBRARY_PATH -u SSL_CERT_FILE -u SSL_CERT_DIR -u CURL_CA_BUNDLE \\
      -u REQUESTS_CA_BUNDLE PATH=/usr/bin:/bin:\$HOME/.elan/bin \\
      bash scripts/setup_lean_project.sh "$TARGET_DIR" "$LEAN_VERSION"

Check it first with:

  env -u LD_LIBRARY_PATH -u SSL_CERT_FILE -u CURL_CA_BUNDLE PATH=/usr/bin:/bin \\
      curl -sS -o /dev/null -w 'HTTP %{http_code}\\n' $CACHE_HOST

Work already done is kept, so re-running is cheap.
EOF
    exit 1
  fi
fi

# -- elan (Lean toolchain manager) ------------------------------------------------------------
if ! command -v elan >/dev/null 2>&1 && [ ! -x "$HOME/.elan/bin/elan" ]; then
  echo "--- installing elan ---"
  curl -sSf https://elan.lean-lang.org/elan-init.sh | sh -s -- -y --default-toolchain none
fi
export PATH="$HOME/.elan/bin:$PATH"
elan --version

# `elan toolchain install` EXITS NON-ZERO when the toolchain is already present
# ("error: '...' is already installed"), which under `set -e` aborts the whole script on the
# second run. Check first so this stays genuinely idempotent.
if elan toolchain list 2>/dev/null | grep -q "^leanprover/lean4:$LEAN_VERSION"; then
  echo "--- toolchain leanprover/lean4:$LEAN_VERSION already installed ---"
else
  echo "--- installing toolchain leanprover/lean4:$LEAN_VERSION ---"
  elan toolchain install "leanprover/lean4:$LEAN_VERSION"
fi

# -- project scaffold (written directly; see header) -------------------------------------------
mkdir -p "$TARGET_DIR"
cd "$TARGET_DIR"

echo "leanprover/lean4:$LEAN_VERSION" > lean-toolchain

# Mathlib is pinned to the tag matching the toolchain, so the toolchain and the library can never
# drift apart — a mismatch is the single most common cause of an unbuildable Lean project.
cat > lakefile.toml <<EOF
name = "prooflens_mathlib"
version = "0.1.0"
defaultTargets = ["ProoflensMathlib"]

[[require]]
name = "mathlib"
scope = "leanprover-community"
rev = "$LEAN_VERSION"

[[lean_lib]]
name = "ProoflensMathlib"
EOF

mkdir -p ProoflensMathlib
cat > ProoflensMathlib.lean <<'EOF'
-- Import surface for the prover harness. The REPL elaborates `import Mathlib` against this
-- project, so nothing else is needed here.
import Mathlib
EOF

# -- dependencies + PREBUILT oleans (never compile) ---------------------------------------------
# Output is streamed, NOT piped to `tail`: these steps take tens of minutes on a cluster
# filesystem, and buffering them behind a pipe leaves the operator staring at a blank screen with
# no way to tell progress from a hang. Full logs also go to setup.log for after-the-fact reading.
LOG="$TARGET_DIR/setup.log"
echo "(full output also being written to $LOG)"

echo "--- lake update (resolving mathlib $LEAN_VERSION) — expect 5-20 min ---"
lake update -R 2>&1 | tee -a "$LOG"

echo "--- lake exe cache get (downloads ~5GB, then decompresses ~8500 files) — expect 20-45 min ---"
lake exe cache get 2>&1 | tee -a "$LOG"

echo "--- lake build (fast if the cache landed; if it is COMPILING files, stop and re-run cache get) ---"
lake build 2>&1 | tee -a "$LOG" | tail -20

# -- verify -------------------------------------------------------------------------------------
echo "--- verifying 'import Mathlib' elaborates and a theorem type-checks ---"
VERIFY_FILE="$(mktemp -t pl_verify_XXXXXX.lean)"
cat > "$VERIFY_FILE" <<'EOF'
import Mathlib
theorem pl_setup_check (a b : Nat) : a + b = b + a := Nat.add_comm a b
EOF

if lake env lean "$VERIFY_FILE"; then
  rm -f "$VERIFY_FILE"
  echo
  echo "SETUP OK — Mathlib imports and a theorem elaborates."
  echo "Point the harness at it with:"
  echo "    export PROOFLENS_LEAN_PROJECT=$TARGET_DIR"
else
  rm -f "$VERIFY_FILE"
  echo "SETUP FAILED — 'import Mathlib' did not elaborate. Check the cache download above." >&2
  exit 1
fi
