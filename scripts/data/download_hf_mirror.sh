#!/bin/bash
# Download a HuggingFace repo from the hf-mirror.com mirror, one file at a time,
# using aria2c (multi-threaded, resumable). Handles subdirectories in the repo.
#
# Usage:
#   bash scripts/data/download_hf_mirror.sh GEAR-Dreams/DreamZero-AgiBot ./checkpoint_agibot
#
# Optional env overrides:
#   HF_MIRROR        mirror base URL            (default: https://hf-mirror.com)
#   ARIA2_THREADS    connections per file       (default: 8)
#
# Requirements: aria2, curl, jq
set -euo pipefail

REPO_ID="${1:-${REPO:-}}"
DEST="${2:-${DEST:-}}"
if [ -z "$REPO_ID" ] || [ -z "$DEST" ]; then
    echo "usage: $0 <repo_id> <local_dir>" >&2
    exit 1
fi

MIRROR="${HF_MIRROR:-https://hf-mirror.com}"
THREADS="${ARIA2_THREADS:-8}"
API="$MIRROR/api/models/$REPO_ID"

command -v aria2c >/dev/null || { echo "aria2c not found" >&2; exit 1; }
command -v jq >/dev/null || { echo "jq not found" >&2; exit 1; }

mkdir -p "$DEST"
cd "$DEST"

echo "== fetching file list: $API =="
mapfile -t FILES < <(curl -sf "$API" | jq -r '.siblings[].rfilename')
if [ "${#FILES[@]}" -eq 0 ]; then
    echo "no files found (repo private / gated / API failed)" >&2
    exit 1
fi
echo "== ${#FILES[@]} files to download =="

i=0
for f in "${FILES[@]}"; do
    i=$((i + 1))
    d="$(dirname "$f")"
    b="$(basename "$f")"
    mkdir -p "$d"
    echo ""
    echo "== [$i/${#FILES[@]}] $f =="
    aria2c -x "$THREADS" -s "$THREADS" -c -k 1M \
        "$MIRROR/$REPO_ID/resolve/main/$f" \
        -d "$d" -o "$b"
done

echo ""
echo "== all done. total size: =="
du -sh .
