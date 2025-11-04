#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./filter_euc2d.sh [DIR]
# If DIR is omitted, current directory is used.
DIR="${1:-.}"

# 1) Decompress all .tsp.gz (keep compressed files with -k)
#    Handles spaces in filenames safely.
echo "Decompressing *.tsp.gz under: $DIR"
find "$DIR" -type f -name '*.tsp.gz' -print0 | while IFS= read -r -d '' gz; do
  # Only unzip if no corresponding .tsp exists yet (avoids redoing work)
  tsp="${gz%.gz}"
  if [[ ! -f "$tsp" ]]; then
    gzip -dk "$gz"
  fi
done

# 2) Prepare output directory
OUT="$DIR/filtered_euc2d"
mkdir -p "$OUT"

# 3) Grep pattern and copy matches
#    Regex tolerates extra spaces and potential trailing chars/comments.
#    ^\s*EDGE_WEIGHT_TYPE\s*:\s*EUC_2D(\s|$)
echo "Scanning .tsp files and copying EUC_2D instances into: $OUT"
matches=0
total=0

find "$DIR" -type f -name '*.tsp' -print0 | while IFS= read -r -d '' tsp; do
  (( total++ )) || true
  if grep -Eq '^[[:space:]]*EDGE_WEIGHT_TYPE[[:space:]]*:[[:space:]]*EUC_2D([[:space:]]|$)' "$tsp"; then
    cp -n "$tsp" "$OUT/"
    (( matches++ )) || true
  fi
done

echo "Done."
echo "Total .tsp scanned: $total"
echo "Matched EUC_2D:     $matches"
echo "Output directory:   $OUT"
