"""Inspect raw ALOHA-style HDF5 episode files (e.g. youcheng_demo3) and print
their structure, so the DreamZero data-conversion code can be matched to the
actual on-disk format before any change is made.

Supports the same two layouts as convert_youcheng_hdf5_to_lerobot.py:
  - flat:  <root>/*.hdf5
  - split: <root>/success/*.hdf5 and <root>/fail/*.hdf5

Only metadata (keys/shapes/dtypes/attrs) is read -- no pixel data is loaded.

Usage:
    python scripts/data/inspect_raw_hdf5.py --hdf5-root /path/to/youcheng_demo3
    python scripts/data/inspect_raw_hdf5.py --hdf5-root /path/to/youcheng_demo3 \
        --max-files 10 --include-fail
"""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def list_files(root: Path, include_fail: bool) -> list[tuple[Path, str]]:
    flat = sorted(root.glob("*.hdf5")) + sorted(root.glob("*.h5"))
    if flat:
        return [(p, "flat") for p in flat]
    out: list[tuple[Path, str]] = []
    for outcome in ("success", "fail"):
        if outcome == "fail" and not include_fail:
            continue
        sub = root / outcome
        if not sub.is_dir():
            continue
        files = sorted(sub.glob("*.hdf5")) + sorted(sub.glob("*.h5"))
        out.extend((p, outcome) for p in files)
    return out


def _clean(v):
    """Make a raw HDF5 attribute JSON-serializable."""
    if isinstance(v, (np.str_, np.bytes_, bytes, bytearray)):
        return bytes(v).decode("utf-8", errors="replace")
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray):
        return f"array{list(v.shape)}:{v.dtype}"
    return v


def summarize(f: h5py.File, path: Path) -> dict:
    info = {"path": str(path)}
    info["attrs"] = {str(k): _clean(v) for k, v in f.attrs.items()}
    info["top_keys"] = sorted(f.keys())

    obs = f["observations"] if "observations" in f else None
    info["observations_keys"] = sorted(obs.keys()) if obs is not None else []

    images = {}
    if obs is not None and "images" in obs:
        for cam in sorted(obs["images"].keys()):
            ds = obs["images"][cam]
            images[cam] = {"shape": list(ds.shape), "dtype": str(ds.dtype)}
    info["images"] = images

    # low-dim signals: look in observations/* first, then top-level
    for key in ("qpos", "action", "timestamp"):
        node = None
        if obs is not None and key in obs:
            node = obs[key]
        elif key in f:
            node = f[key]
        info[key] = {"shape": list(node.shape), "dtype": str(node.dtype)} if node is not None else None
    return info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hdf5-root", type=Path, required=True,
                        help="Root dir of raw episodes (flat or success/fail layout).")
    parser.add_argument("--max-files", type=int, default=5,
                        help="Show per-file detail for the first N files (-1 = all).")
    parser.add_argument("--include-fail", action="store_true",
                        help="Also scan fail/ subdirectory.")
    args = parser.parse_args()

    if not args.hdf5_root.is_dir():
        raise FileNotFoundError(f"hdf5 root not found: {args.hdf5_root}")

    files = list_files(args.hdf5_root, args.include_fail)
    if not files:
        raise FileNotFoundError(
            f"No .hdf5 files found under {args.hdf5_root} "
            "(expected *.hdf5 directly or a success/ subdirectory)")

    n_success = sum(1 for _, o in files if o == "success")
    n_fail = sum(1 for _, o in files if o == "fail")
    print(f"root: {args.hdf5_root}")
    print(f"layout: {'flat' if any(o == 'flat' for _, o in files) else 'success/fail'}")
    print(f"found {len(files)} episodes "
          f"(success={n_success}, fail={n_fail}, other={len(files) - n_success - n_fail})")

    details = []
    for p, outcome in files:
        with h5py.File(p, "r") as f:
            info = summarize(f, p)
        info["outcome"] = outcome
        details.append(info)

    limit = len(details) if args.max_files < 0 else min(args.max_files, len(details))
    print(f"\n===== per-file detail (first {limit}) =====")
    for info in details[:limit]:
        print("---- " + info["path"] + f" [{info['outcome']}] ----")
        print(json.dumps(info, indent=2, ensure_ascii=False))

    print("\n===== aggregate across all files =====")
    cam_key_sets = {tuple(sorted(i["images"].keys())) for i in details}
    action_shapes = {tuple(i["action"]["shape"]) if i["action"] else None for i in details}
    qpos_shapes = {tuple(i["qpos"]["shape"]) if i["qpos"] else None for i in details}
    fps_vals = {i["attrs"].get("control_hz") for i in details if i["attrs"].get("control_hz") is not None}
    lens_ok = all(
        i["qpos"] and i["action"] and i["qpos"]["shape"][0] == i["action"]["shape"][0]
        for i in details)
    print(f"camera key sets (unique): {[list(s) for s in cam_key_sets]}")
    print(f"action shapes (unique):  {[list(s) if s else None for s in action_shapes]}")
    print(f"qpos shapes (unique):    {[list(s) if s else None for s in qpos_shapes]}")
    print(f"control_hz values:       {sorted(str(v) for v in fps_vals)}")
    print(f"timestamp present:       {all(i['timestamp'] is not None for i in details)}")
    print(f"qpos_len == action_len:  {lens_ok}")

    issues = []
    expected_cams = max(cam_key_sets, key=len) if cam_key_sets else set()
    for i in details:
        missing = set(expected_cams) - set(i["images"].keys())
        if missing:
            issues.append(f"{Path(i['path']).name}: missing cameras {sorted(missing)}")
        if i["qpos"] and i["qpos"]["shape"][1] != 14:
            issues.append(f"{Path(i['path']).name}: qpos dim {i['qpos']['shape']} (stack_cups expects 14)")
        if i["action"] and i["action"]["shape"][1] != 14:
            issues.append(f"{Path(i['path']).name}: action dim {i['action']['shape']} (stack_cups expects 14)")
    if issues:
        print("\n===== potential issues =====")
        for line in issues[:50]:
            print("  " + line)
    else:
        print("\n===== no obvious issues found =====")


if __name__ == "__main__":
    main()
