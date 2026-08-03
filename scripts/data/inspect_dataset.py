"""
Inspect a LeRobot v2 dataset and print the structure needed to create
DreamZero configs (modality.json, data YAML, training script).

Usage:
    python scripts/data/inspect_dataset.py --dataset-path /path/to/your_dataset

The output tells you:
  - meta/ file inventory (which GEAR metadata files exist)
  - info.json: total_episodes, fps, features (dtype/shape/video_info)
  - parquet columns and their vector dimensions (state/action)
  - video camera directory names and file naming
  - tasks.jsonl / episodes.jsonl content
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Print the structure of a LeRobot v2 dataset for DreamZero config creation."
    )
    parser.add_argument("--dataset-path", type=str, required=True, help="Path to the LeRobot v2 dataset")
    args = parser.parse_args()

    root = Path(args.dataset_path).resolve()
    if not root.exists():
        print(f"ERROR: dataset path does not exist: {root}")
        sys.exit(1)

    print("=" * 70)
    print(f"Dataset: {root}")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. meta/ file inventory
    # ------------------------------------------------------------------
    meta_dir = root / "meta"
    print("\n[1] meta/ files:")
    if meta_dir.exists():
        for f in sorted(meta_dir.iterdir()):
            size = f.stat().st_size if f.is_file() else -1
            marker = "DIR" if f.is_dir() else f"{size} bytes"
            print(f"    {f.name}  ({marker})")
    else:
        print("    MISSING meta/ directory!")

    # ------------------------------------------------------------------
    # 2. info.json
    # ------------------------------------------------------------------
    info_path = meta_dir / "info.json"
    print("\n[2] info.json:")
    if not info_path.exists():
        print("    MISSING meta/info.json!")
    else:
        with open(info_path) as f:
            info = json.load(f)
        print(f"    total_episodes: {info.get('total_episodes')}")
        print(f"    fps:            {info.get('fps')}")
        print(f"    chunks_size:    {info.get('chunks_size')}")
        print(f"    data_path:      {info.get('data_path')}")
        print(f"    codebase_version: {info.get('codebase_version')}")
        features = info.get("features", {})
        print(f"    features ({len(features)}):")
        for name, feat in features.items():
            dtype = feat.get("dtype")
            shape = feat.get("shape")
            video_info = feat.get("video_info") or feat.get("info")
            extra = f" video_info={video_info}" if video_info else ""
            print(f"        {name}: dtype={dtype} shape={shape}{extra}")

    # ------------------------------------------------------------------
    # 3. Sample parquet file: columns, shapes, values
    # ------------------------------------------------------------------
    print("\n[3] Sample parquet file:")
    parquet_files = sorted(root.glob("data/*/*.parquet"))
    if not parquet_files:
        print("    NO parquet files found under data/")
    else:
        sample = parquet_files[0]
        print(f"    sample: {sample}")
        df = pd.read_parquet(sample)
        print(f"    num_rows: {len(df)}")
        for col in df.columns:
            val = df[col].iloc[0]
            if hasattr(val, "shape"):
                shape = val.shape
                # print a few sample values for numeric arrays
                arr = val
                sample_vals = None
                try:
                    import numpy as np

                    arr_np = arr
                    if arr_np.ndim == 1 and arr_np.shape[0] <= 16:
                        sample_vals = (
                            str([round(float(x), 4) if not isinstance(x, str) else str(x) for x in arr_np])
                        )
                except Exception:
                    pass
                print(f"        {col}: shape={shape} {sample_vals or ''}")
            else:
                print(f"        {col}: scalar={val} (dtype={type(val).__name__})")

    # ------------------------------------------------------------------
    # 4. videos/ structure
    # ------------------------------------------------------------------
    print("\n[4] videos/ structure:")
    videos_dir = root / "videos"
    if not videos_dir.exists():
        print("    MISSING videos/ directory!")
    else:
        for chunk in sorted(videos_dir.iterdir()):
            if not chunk.is_dir():
                continue
            print(f"    {chunk.name}/")
            for cam in sorted(chunk.iterdir()):
                if not cam.is_dir():
                    continue
                mp4s = sorted(cam.glob("*.mp4"))
                print(f"        {cam.name}/  ({len(mp4s)} mp4 files)")
                if mp4s:
                    print(f"            first: {mp4s[0].name}")

    # ------------------------------------------------------------------
    # 5. tasks.jsonl
    # ------------------------------------------------------------------
    tasks_path = meta_dir / "tasks.jsonl"
    print("\n[5] tasks.jsonl (first 5 lines):")
    if not tasks_path.exists():
        print("    MISSING meta/tasks.jsonl")
    else:
        with open(tasks_path) as f:
            for i, line in enumerate(f):
                if i >= 5:
                    break
                print(f"    {line.strip()}")
        with open(tasks_path) as f:
            n_lines = sum(1 for _ in f)
        print(f"    (total {n_lines} lines)")

    # ------------------------------------------------------------------
    # 6. episodes.jsonl
    # ------------------------------------------------------------------
    episodes_path = meta_dir / "episodes.jsonl"
    print("\n[6] episodes.jsonl (first 3 lines):")
    if not episodes_path.exists():
        print("    MISSING meta/episodes.jsonl")
    else:
        with open(episodes_path) as f:
            for i, line in enumerate(f):
                if i >= 3:
                    break
                print(f"    {line.strip()}")

    # ------------------------------------------------------------------
    # 7. GEAR metadata presence
    # ------------------------------------------------------------------
    print("\n[7] GEAR metadata files:")
    for name in ["modality.json", "embodiment.json", "stats.json",
                 "relative_stats_dreamzero.json", "step_filter.jsonl"]:
        p = meta_dir / name
        print(f"    {name}: {'EXISTS' if p.exists() else 'missing'}")

    print("\n" + "=" * 70)
    print("Done. Paste this output into the chat to build the DreamZero configs.")
    print("=" * 70)


if __name__ == "__main__":
    main()
