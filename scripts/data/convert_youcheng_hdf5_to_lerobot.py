"""Convert YAM/youcheng ALOHA-style HDF5 episodes into a LeRobot v2 dataset.

Raw data layout (one .hdf5 per episode). The root may either contain the
.hdf5 files directly or use success/ and fail/ subdirectories; by default
only success/ episodes are converted (failed demos are not used for
imitation learning), pass --include-fail to also convert fail/:

    youcheng_demo3/
    ├── success/0001.hdf5 ...   (converted)
    └── fail/0013.hdf5 ...      (converted only with --include-fail)

Each HDF5 file:
    /observations/images/{main,left,right}   (T, H, W, 3) uint8
    /observations/qpos                       (T, 14) float32
    /action                                  (T, 14) float32
    attrs: camera_names, control_hz, task_name, outcome, ...

Output layout (LeRobot v2, compatible with DreamZero's `stack_cups` embodiment):
    dataset/
    ├── data/chunk-000/episode_000000.parquet ...   # low-dim: state/action/task
    ├── videos/chunk-000/
    │   ├── cam_high/          episode_000000.mp4 ...   # was: main
    │   ├── cam_left_wrist/    episode_000000.mp4 ...   # was: left
    │   └── cam_right_wrist/   episode_000000.mp4 ...   # was: right
    └── meta/
        ├── info.json          # features, data_path, video_path, chunks_size, fps
        ├── tasks.jsonl        # task_index -> instruction text
        └── episodes.jsonl     # per-episode index/length

Why rename the cameras?  The `stack_cups` embodiment (14-dim state/action,
3 views in a 2x2 grid) is already fully wired in DreamZero.  By mapping
main/left/right -> cam_high/cam_left_wrist/cam_right_wrist we can reuse the
whole stack_cups pipeline (modality config, view layout, text template,
training script) without touching any model code.  After this script, run:

    python scripts/data/convert_lerobot_to_gear.py \
        --dataset-path $OUT \
        --embodiment-tag stack_cups \
        --state-keys '{"left_arm_joint_pos": [0, 6], "left_gripper_pos": [6, 7], \
                       "right_arm_joint_pos": [7, 13], "right_gripper_pos": [13, 14]}' \
        --action-keys '{"left_arm_joint_pos": [0, 6], "left_gripper_pos": [6, 7], \
                        "right_arm_joint_pos": [7, 13], "right_gripper_pos": [13, 14]}' \
        --relative-action-keys left_arm_joint_pos left_gripper_pos \
                               right_arm_joint_pos right_gripper_pos \
        --task-key annotation.task_index \
        --video-key-style short

Usage (defaults point to the shared H200 server's youcheng_demo3; success/ only,
add --include-fail to also convert fail/):
    python scripts/data/convert_youcheng_hdf5_to_lerobot.py
    python scripts/data/convert_youcheng_hdf5_to_lerobot.py \
        --hdf5-root /path/to/youcheng_demo3 \
        --output /path/to/youcheng_demo3_lerobot \
        --task "stack the cups"
"""
import argparse
import json
from pathlib import Path

import h5py
import imageio.v2 as imageio
import numpy as np
import pandas as pd

# Camera names in the raw HDF5 -> LeRobot video keys (must match stack_cups config).
DEFAULT_CAMERA_MAP = {
    "main": "cam_high",
    "left": "cam_left_wrist",
    "right": "cam_right_wrist",
}
DEFAULT_TASK = "stack the cups"
CHUNK_SIZE = 1000  # episodes per chunk directory (LeRobot convention)


def parse_camera_map(raw: str) -> dict[str, str]:
    if not raw:
        return dict(DEFAULT_CAMERA_MAP)
    return dict(item.split("=", 1) for item in raw.split(",") if "=" in item)


SUCCESS_DIR = "success"
FAIL_DIR = "fail"


def discover_episodes(root: Path, include_fail: bool) -> list[tuple[Path, str]]:
    """Find episode .hdf5/.h5 files under `root`.

    Two layouts are supported:
      - flat:  <root>/*.hdf5 (outcome "unknown")
      - split: <root>/success/*.hdf5, plus <root>/fail/*.hdf5 when include_fail

    Returns [(path, outcome)] sorted by (outcome, filename) so episode indices
    are stable and unique.
    """
    flat = sorted(root.glob("*.hdf5")) + sorted(root.glob("*.h5"))
    if flat:
        return [(p, "unknown") for p in flat]

    episodes: list[tuple[Path, str]] = []
    for outcome in (SUCCESS_DIR, FAIL_DIR):
        if outcome == FAIL_DIR and not include_fail:
            continue
        sub = root / outcome
        if not sub.is_dir():
            continue
        files = sorted(sub.glob("*.hdf5")) + sorted(sub.glob("*.h5"))
        episodes.extend((p, outcome) for p in files)
    return episodes


def read_fps(f: h5py.File, fallback: float) -> float:
    control_hz = f.attrs.get("control_hz")
    try:
        if control_hz is not None and float(control_hz) > 0:
            return float(control_hz)
    except (TypeError, ValueError):
        pass
    return fallback


def parse_scale(raw: str | None) -> tuple[int, int] | None:
    """Parse 'WxH' (e.g. '320x240') into (width, height); None if empty."""
    if not raw:
        return None
    w, h = raw.lower().split("x")
    return int(w), int(h)


def resize_frames(frames: np.ndarray, scale: tuple[int, int]) -> np.ndarray:
    """Resize a (T, H, W, 3) uint8 array to (T, new_h, new_w, 3)."""
    import cv2
    w, h = scale
    out = np.empty((frames.shape[0], h, w, 3), dtype=frames.dtype)
    for i in range(frames.shape[0]):
        out[i] = cv2.resize(frames[i], (w, h), interpolation=cv2.INTER_AREA)
    return out


def write_video(frames: np.ndarray, path: Path, fps: float) -> None:
    """Write a (T, H, W, 3) uint8 array as H.264 mp4."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(path, fps=fps, codec="libx264",
                            pixelformat="yuv420p", macro_block_size=1) as writer:
        for i in range(frames.shape[0]):
            writer.append_data(np.ascontiguousarray(frames[i]))


def write_episode(hdf5_path: Path, episode_index: int, out_root: Path,
                  camera_map: dict[str, str], fps: float,
                  task_index: int,
                  video_scale: tuple[int, int] | None = None) -> int:
    chunk_idx = episode_index // CHUNK_SIZE
    ep_str = f"episode_{episode_index:06d}"

    with h5py.File(hdf5_path, "r") as f:
        images_root = f["observations/images"]
        qpos = np.asarray(f["observations/qpos"][:], dtype=np.float32)
        action = np.asarray(f["action"][:], dtype=np.float32)
        try:
            timestamp = np.asarray(f["timestamp"][:], dtype=np.float64)
        except KeyError:
            timestamp = np.arange(len(action), dtype=np.float64) / max(fps, 1e-9)

        # --- videos ---
        for src, dst in camera_map.items():
            if src not in images_root:
                print(f"[warn] camera '{src}' missing in {hdf5_path.name}, skipping")
                continue
            frames = np.asarray(images_root[src][:])  # (T, H, W, 3) uint8
            if video_scale is not None:
                frames = resize_frames(frames, video_scale)
            video_path = out_root / "videos" / f"chunk-{chunk_idx:03d}" / dst / f"{ep_str}.mp4"
            write_video(frames, video_path, fps)
            print(f"  {hdf5_path.name}: {src} -> {video_path} ({frames.shape[0]} frames, {frames.shape[2]}x{frames.shape[1]})")

        # --- parquet (low-dim) ---
        n = len(qpos)
        df = pd.DataFrame({
            "episode_index": np.full(n, episode_index, dtype=np.int64),
            "frame_index": np.arange(n, dtype=np.int64),
            "timestamp": timestamp,
            "observation.state": [qpos[i] for i in range(n)],
            "action": [action[i] for i in range(n)],
            "annotation.task_index": np.full(n, task_index, dtype=np.int64),
        })
        parquet_path = out_root / "data" / f"chunk-{chunk_idx:03d}" / f"{ep_str}.parquet"
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(parquet_path, index=False)
        print(f"  {hdf5_path.name}: -> {parquet_path} ({n} steps)")
        return n

    return 0


def write_meta(out_root: Path, episodes: list[dict], task: str,
               camera_map: dict[str, str], fps: float) -> None:
    meta_dir = out_root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    features = {
        "observation.state": {"dtype": "float32", "shape": [14],
                              "names": ["left_arm_joint_pos", "left_gripper_pos",
                                        "right_arm_joint_pos", "right_gripper_pos"]},
        "action": {"dtype": "float32", "shape": [14],
                   "names": ["left_arm_joint_pos", "left_gripper_pos",
                             "right_arm_joint_pos", "right_gripper_pos"]},
        "annotation.task_index": {"dtype": "int64", "shape": [1]},
    }
    for cam in camera_map.values():
        features[f"observation.images.{cam}"] = {
            "dtype": "video",
            "video_info": {"video.fps": fps},
        }

    info = {
        "codebase_version": "v2.0",
        "robot_type": "youcheng",
        "fps": fps,
        "chunks_size": CHUNK_SIZE,
        "total_episodes": len(episodes),
        "total_frames": sum(ep["length"] for ep in episodes),
        "total_tasks": 1,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
    }
    with open(meta_dir / "info.json", "w") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    with open(meta_dir / "tasks.jsonl", "w") as f:
        f.write(json.dumps({"task_index": 0, "task": task}, ensure_ascii=False) + "\n")

    with open(meta_dir / "episodes.jsonl", "w") as f:
        for ep in episodes:
            f.write(json.dumps(ep, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hdf5-root", type=Path,
                        default=Path("/inspire/qb-ilm/project/robot-reasoning/public/data/lerobot/zyf/youcheng_demo3"),
                        help="Root directory of raw episodes. Accepts either a flat dir of "
                             "*.hdf5/*.h5 files or a dir with success/ (and optionally fail/) "
                             "subdirectories, e.g. .../youcheng_demo3. "
                             "(default: shared H200 server's youcheng_demo3)")
    parser.add_argument("--output", "-o", type=Path,
                        default=Path("/inspire/qb-ilm/project/robot-reasoning/public/data/lerobot/zyf/youcheng_demo3_lerobot"),
                        help="Output LeRobot v2 dataset directory "
                             "(default: .../zyf/youcheng_demo3_lerobot).")
    parser.add_argument("--task", type=str, default=DEFAULT_TASK,
                        help=f"Language instruction shared by all episodes (default: '{DEFAULT_TASK}').")
    parser.add_argument("--camera-map", type=str, default="",
                        help="Comma-separated source=dest camera mapping, e.g. "
                             "'main=cam_high,left=cam_left_wrist,right=cam_right_wrist' "
                             "(default maps main/left/right to cam_high/cam_left_wrist/cam_right_wrist).")
    parser.add_argument("--include-fail", action="store_true",
                        help="Also convert episodes under fail/ (default: success/ only, since "
                             "failed demos are not used for imitation learning).")
    parser.add_argument("--fps", type=float, default=None,
                        help="Override frame rate (default: read from hdf5 control_hz, else 30).")
    parser.add_argument("--video-scale", type=str, default="",
                        help="Resize videos to WxH before encoding (e.g. '320x240', half of the "
                             "native 640x480, ~4x faster decode at train time). "
                             "Default: keep native resolution.")
    args = parser.parse_args()

    if not args.hdf5_root.is_dir():
        raise FileNotFoundError(f"hdf5 root not found: {args.hdf5_root}")

    camera_map = parse_camera_map(args.camera_map)
    episode_files = discover_episodes(args.hdf5_root, args.include_fail)
    if not episode_files:
        raise FileNotFoundError(
            f"No .hdf5 files found under {args.hdf5_root} "
            "(expected *.hdf5 directly or a success/ subdirectory)")

    # Auto fps from the first file
    fps = args.fps
    if fps is None:
        with h5py.File(episode_files[0][0], "r") as f:
            fps = read_fps(f, 30.0)
    n_success = sum(1 for _, o in episode_files if o == SUCCESS_DIR)
    n_fail = sum(1 for _, o in episode_files if o == FAIL_DIR)
    print(f"Found {len(episode_files)} episodes "
          f"(success={n_success}, fail={n_fail}, other={len(episode_files) - n_success - n_fail}), "
          f"fps={fps}, task='{args.task}'")
    print(f"Camera mapping: {camera_map}")

    video_scale = parse_scale(args.video_scale)
    if video_scale is not None:
        print(f"Video resize: native -> {video_scale[0]}x{video_scale[1]}")

    episodes = []
    for idx, (h5, outcome) in enumerate(episode_files):
        length = write_episode(h5, idx, args.output, camera_map, fps, task_index=0,
                               video_scale=video_scale)
        episodes.append({"episode_index": idx, "tasks": [args.task], "length": length,
                         "outcome": outcome})

    write_meta(args.output, episodes, args.task, camera_map, fps)
    print(f"\nDone. LeRobot v2 dataset written to {args.output}")
    print("Next step: run convert_lerobot_to_gear.py (see docstring header).")


if __name__ == "__main__":
    main()
