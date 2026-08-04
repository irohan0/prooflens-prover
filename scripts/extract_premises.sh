#!/usr/bin/env bash
# Run scripts/extract_premises.lean against a built Lean+Mathlib project and write the premise
# corpus that every retrieval arm indexes.
#
# Usage:
#   scripts/extract_premises.sh [LEAN_PROJECT] [OUT_JSONL]
#   PREMISE_LIMIT=2000 scripts/extract_premises.sh      # fast smoke run (~seconds after imports)
#
# Expect ~10-20 min for the full corpus: the cost is pretty-printing ~200k elaborated types, not
# reading them. Progress prints every 25k declarations.

set -euo pipefail

# Self-sanitising re-exec — same rationale as scripts/setup_lean_project.sh. A conda-touched
# LD_LIBRARY_PATH breaks dynamically-linked tools in ways that surface far from the cause, and a
# fix that depends on the operator remembering a prefix is not a fix.
if [ -z "${PROOFLENS_SANITISED_ENV:-}" ]; then
  exec env \
    -u LD_LIBRARY_PATH -u LD_PRELOAD \
    -u SSL_CERT_FILE -u SSL_CERT_DIR -u CURL_CA_BUNDLE -u REQUESTS_CA_BUNDLE \
    -u PYTHONHOME -u PYTHONPATH \
    PROOFLENS_SANITISED_ENV=1 \
    PREMISE_LIMIT="${PREMISE_LIMIT:-}" \
    PATH="/usr/bin:/bin:$HOME/.elan/bin" \
    bash "$0" "$@"
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LEAN_PROJECT="${1:-${PROOFLENS_LEAN_PROJECT:-$HOME/lean/mathlib_v4160}}"
OUT="${2:-$REPO_ROOT/data/premises/mathlib_v4160.jsonl}"

if [ ! -f "$LEAN_PROJECT/lean-toolchain" ]; then
  echo "No Lean project at $LEAN_PROJECT (run scripts/setup_lean_project.sh first)" >&2
  exit 1
fi

mkdir -p "$(dirname "$OUT")"

echo "=== premise extraction ==="
echo "project : $LEAN_PROJECT ($(cat "$LEAN_PROJECT/lean-toolchain"))"
echo "out     : $OUT"
echo "limit   : ${PREMISE_LIMIT:-0 (all)}"

# The extractor must live inside the project so `lake env lean` resolves `import Mathlib` against
# the project's own build. Copied rather than symlinked: WSL/Windows filesystem boundaries make
# symlinks unreliable here, and the file is 5 KB.
cp "$REPO_ROOT/scripts/extract_premises.lean" "$LEAN_PROJECT/ExtractPremises.lean"

cd "$LEAN_PROJECT"
START=$(date +%s)
PREMISE_OUT="$OUT" PREMISE_LIMIT="${PREMISE_LIMIT:-}" lake env lean ExtractPremises.lean
ELAPSED=$(( $(date +%s) - START ))

rm -f "$LEAN_PROJECT/ExtractPremises.lean"

LINES=$(wc -l < "$OUT")
BYTES=$(du -h "$OUT" | cut -f1)
echo
echo "EXTRACTION OK — $LINES premises, $BYTES, ${ELAPSED}s"
echo "  $OUT"
