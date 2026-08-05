"""
Inspect the checkpoints/ directory: show disk usage per subdir/file,
compare against expected weights, and flag anomalies (duplicates, temp files).

Usage:
    python scripts/check_checkpoints.py [--path ./checkpoints]
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path


def fmt_size(n: float) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"


def dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def main():
    parser = argparse.ArgumentParser(
        description="Show disk usage of the checkpoints dir and flag anomalies."
    )
    parser.add_argument("--path", type=str, default="checkpoints",
                        help="path to the checkpoints directory (default: ./checkpoints)")
    args = parser.parse_args()

    root = Path(args.path).resolve()
    if not root.exists():
        print(f"ERROR: {root} does not exist")
        sys.exit(1)

    print("=" * 60)
    print(f"checkpoints dir: {root}")
    print("=" * 60)

    # 1. Total + per top-level entry
    total = 0
    entries = []
    for entry in sorted(root.iterdir()):
        if entry.is_dir():
            size = dir_size(entry)
            entries.append((entry.name, size, "DIR"))
        else:
            size = entry.stat().st_size
            entries.append((entry.name, size, "FILE"))
        total += size
    print(f"\nTotal: {fmt_size(total)}")
    print("-" * 60)
    for name, size, kind in sorted(entries, key=lambda x: -x[1]):
        print(f"{fmt_size(size):>10}  [{kind}] {name}")

    # 2. Wan2.1 top files (largest 15)
    wan = root / "Wan2.1-I2V-14B-480P"
    if wan.exists():
        print(f"\nTop 15 files in {wan.name}:")
        files = []
        for f in wan.rglob("*"):
            if f.is_file():
                files.append((f, f.stat().st_size))
        for f, size in sorted(files, key=lambda x: -x[1])[:15]:
            print(f"  {fmt_size(size):>10}  {f.relative_to(wan)}")

        # Expected Wan2.1 inventory
        print(f"\nExpected Wan2.1 inventory check:")
        shards = sorted(wan.glob("diffusion_pytorch_model-*.safetensors"))
        print(f"  safetensors shards: {len(shards)} (expect 7)")
        for name in ["Wan2.1_VAE.pth",
                     "models_t5_umt5-xxl-enc-bf16.pth",
                     "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"]:
            p = wan / name
            print(f"  {name}: {'OK (' + fmt_size(p.stat().st_size) + ')' if p.exists() else 'MISSING!'}")

    # 3. Anomaly checks
    print("\nAnomaly checks:")
    name_map: dict[str, list[Path]] = defaultdict(list)
    for f in root.rglob("*"):
        if f.is_file():
            name_map[f.name].append(f)
    dups = {k: v for k, v in name_map.items() if len(v) > 1 and k.endswith((".pth", ".safetensors", ".json"))}
    if dups:
        print("  [WARN] Duplicate files (check if they are intentional copies):")
        for k, v in dups.items():
            print(f"    {k}: {len(v)} copies -> " + ", ".join(str(p) for p in v))
    else:
        print("  [OK] No duplicate model/config files")

    incompletes = list(root.rglob("*.incomplete"))
    if incompletes:
        print(f"  [WARN] {len(incompletes)} .incomplete temp file(s):")
        for f in incompletes:
            print(f"    {f} ({fmt_size(f.stat().st_size)})")
    else:
        print("  [OK] No .incomplete temp files")

    caches = list(root.rglob(".cache"))
    if caches:
        print(f"  [WARN] {len(caches)} .cache dir(s) (hf download leftovers):")
        for c in caches:
            print(f"    {c}")
    else:
        print("  [OK] No .cache dirs")

    print("\nDone. If a shard/encoder above shows MISSING or unexpected size, report it.")
    print("Remember: DreamZero-AgiBot (~45GB) is still needed for pretrained_model_path.")


if __name__ == "__main__":
    main()
