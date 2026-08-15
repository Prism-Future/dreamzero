"""Convert image sequences stored in an HDF5 file into MP4 videos.

Supports the standard ALOHA-style HDF5 layout produced by robot
recording stacks, e.g.:

    /observations/images/<camera_name>  (T, H, W, 3) uint8
    /observations/qpos                  (T, dim)
    /action                             (T, dim)

The number of frames and cameras are auto-detected from the file. One MP4
per camera is written into the output directory. Encoding uses H.264
(libx264) with yuv420p pixel format for broad player compatibility.

Usage:
    python scripts/data/convert_hdf5_to_video.py --input 0009.hdf5 \
        --output-dir debug_image [--fps 30]
"""
import argparse
import json
import os
from pathlib import Path

import h5py
import imageio.v2 as imageio
import numpy as np


def read_attrs(f: h5py.File) -> dict:
    attrs = {}
    for k, v in f.attrs.items():
        try:
            attrs[k] = v
        except Exception:
            attrs[k] = None
    return attrs


def parse_names(value) -> list[str]:
    """camera_names may come back as a JSON string, bytes, or a numpy array."""
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8")
    if isinstance(value, str):
        return list(json.loads(value))
    if isinstance(value, np.ndarray):
        return [str(v) for v in value.tolist()]
    return [str(v) for v in value]


def get_camera_names(f: h5py.File, attrs: dict) -> list[str]:
    if "camera_names" in attrs:
        return parse_names(attrs["camera_names"])
    return sorted(f["observations/images"].keys())


def get_fps(attrs: dict, fallback: float) -> float:
    if "control_hz" in attrs and float(attrs["control_hz"]) > 0:
        return float(attrs["control_hz"])
    camera_metadata = attrs.get("camera_metadata")
    if camera_metadata:
        try:
            metadata = json.loads(camera_metadata) if isinstance(camera_metadata, str) else camera_metadata
            fps_values = [float(m["fps"]) for m in metadata.values() if m.get("fps")]
            if fps_values:
                return int(round(max(fps_values)))
        except Exception:
            pass
    return fallback


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert HDF5 image sequences to MP4 videos.")
    parser.add_argument("--input", "-i", type=Path, required=True, help="Path to the input .hdf5 file.")
    parser.add_argument("--output-dir", "-o", type=Path, default=Path("debug_image"),
                        help="Directory to write the MP4 files to (default: ./debug_image).")
    parser.add_argument("--fps", type=float, default=30.0,
                        help="Fallback frame rate if the file does not specify one (default: 30).")
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.input, "r") as f:
        attrs = read_attrs(f)
        camera_names = get_camera_names(f, attrs)
        fps = get_fps(attrs, args.fps)

        images_root = f["observations/images"]
        for camera in camera_names:
            if camera not in images_root:
                print(f"[skip] camera '{camera}' not found under observations/images")
                continue
            images = images_root[camera]
            if images.ndim != 4 or images.shape[3] != 3:
                print(f"[skip] camera '{camera}' has unexpected shape {images.shape}")
                continue

            output_path = args.output_dir / f"{args.input.stem}_{camera}.mp4"
            total = images.shape[0]
            print(f"[{camera}] {total} frames {images.shape[1]}x{images.shape[2]} @ {fps} fps -> {output_path}")

            with imageio.get_writer(output_path, fps=fps, codec="libx264",
                                    pixelformat="yuv420p", macro_block_size=1) as writer:
                for i in range(total):
                    frame = np.asarray(images[i])  # (H, W, 3) uint8
                    writer.append_data(frame)
                    if i == 0 or (i + 1) % 100 == 0 or i == total - 1:
                        print(f"    {i + 1}/{total}")
            print(f"[done] {camera} -> {output_path}")

    print("\nAll videos written.")


if __name__ == "__main__":
    main()
