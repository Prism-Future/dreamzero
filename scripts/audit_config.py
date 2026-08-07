#!/usr/bin/env python3
"""Audit Hydra config to expose all missing ${} variables before training."""
import sys
import os
import traceback

os.environ["HYDRA_FULL_ERROR"] = "1"

from hydra import initialize, compose
from omegaconf import OmegaConf


def audit_config():
    config_name = "experiment"
    overrides = [
        "data=dreamzero/stack_cups_relative_wan22",
        "model=dreamzero/vla",
        "model/dreamzero/action_head=wan_flow_matching_action_tf_wan22",
        "model/dreamzero/transform=dreamzero_cotrain",
        "train_architecture=lora",
        "num_frames=33",
        "action_horizon=24",
        "num_views=3",
        "num_frame_per_block=2",
        "num_action_per_block=24",
        "num_state_per_block=1",
        "max_chunk_size=4",
        "stack_cups_data_root=/tmp/dummy",
        "dit_version=/tmp/dummy",
        "text_encoder_pretrained_path=/tmp/dummy",
        "image_encoder_pretrained_path=/tmp/dummy",
        "vae_pretrained_path=/tmp/dummy",
        "tokenizer_path=/tmp/dummy",
    ]

    try:
        with initialize(version_base=None, config_path="../groot/vla/configs"):
            cfg = compose(config_name=config_name, overrides=overrides)

        print("=" * 60)
        print("CONFIG AUDIT - Checking all ${} variable resolutions")
        print("=" * 60)

        cfg_str = OmegaConf.to_yaml(cfg)

        unresolved = []
        for i, line in enumerate(cfg_str.split("\n"), 1):
            if "${" in line and "}" in line:
                unresolved.append((i, line.strip()))

        if unresolved:
            print("\n⚠️  POTENTIALLY UNRESOLVED ${} VARIABLES:")
            print("-" * 60)
            for line_no, content in unresolved:
                print(f"  Line {line_no}: {content[:120]}")
        else:
            print("\n✅  All ${} variables resolved successfully!")

        print("\n" + "=" * 60)
        print("KEY PARAMETER CHECK:")
        print("=" * 60)

        checks = {
            "max_state_dim": None,
            "max_action_dim": None,
            "hidden_size": None,
            "action_dim": None,
        }

        try:
            model_cfg = cfg.get("model", {})
            if isinstance(model_cfg, dict):
                for key in checks:
                    checks[key] = model_cfg.get(key, "NOT FOUND")
        except Exception as e:
            print(f"  Error accessing model config: {e}")

        for key, value in checks.items():
            status = "✅" if value != "NOT FOUND" else "❌"
            print(f"  {status} {key}: {value}")

        try:
            action_head = cfg.get("model", {}).get("action_head_cfg", {}).get("config", {})
            diffusion_cfg = action_head.get("diffusion_model_cfg", {})
            print("\n📦 diffusion_model_cfg parameters:")
            for key in ["max_state_dim", "hidden_size", "action_dim", "dim", "in_dim", "out_dim"]:
                val = diffusion_cfg.get(key, "NOT FOUND")
                status = "✅" if val != "NOT FOUND" else "❌"
                print(f"  {status} {key}: {val}")
        except Exception as e:
            print(f"  Error: {e}")

        print("\n" + "=" * 60)
        print("FULL CONFIG (first 100 lines):")
        print("=" * 60)
        for i, line in enumerate(cfg_str.split("\n")[:100]):
            print(f"{i+1:4d} | {line}")

        return 0

    except Exception as e:
        print(f"\n❌  CONFIG COMPOSITION FAILED:")
        print("=" * 60)
        traceback.print_exc()

        if "Error executing resolver" in str(e) or "InterpolationResolutionError" in str(type(e).__name__):
            print("\n💡  This means some ${} variables cannot be resolved!")
            print("    Check the error above for specific variable names.")

        return 1


if __name__ == "__main__":
    sys.exit(audit_config())
