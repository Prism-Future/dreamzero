#!/bin/bash
# DreamZero stack_cups Training Script with Wan2.2-TI2V-5B backbone (for 4090 GPUs)
#
# Usage:
#   NUM_GPUS=4 bash scripts/train/stack_cups_training_wan22.sh
#
# Prerequisites:
#   - stack_cups dataset in LeRobot v2 format at STACK_CUPS_DATA_ROOT
#     (state 14, action 14, 3 views: cam_high, cam_left_wrist, cam_right_wrist)
#     Must first generate GEAR metadata with:
#       python scripts/data/convert_lerobot_to_gear.py \
#           --dataset-path $STACK_CUPS_DATA_ROOT \
#           --embodiment-tag stack_cups \
#           --state-keys '{"left_arm_joint_pos": [0, 6], "left_gripper_pos": [6, 7], "right_arm_joint_pos": [7, 13], "right_gripper_pos": [13, 14]}' \
#           --action-keys '{"left_arm_joint_pos": [0, 6], "left_gripper_pos": [6, 7], "right_arm_joint_pos": [7, 13], "right_gripper_pos": [13, 14]}' \
#           --relative-action-keys left_arm_joint_pos left_gripper_pos right_arm_joint_pos right_gripper_pos \
#           --task-key task_index \
#           --video-key-style short
#   - Wan2.2-TI2V-5B weights (download from HuggingFace)
#     huggingface-cli download Wan-AI/Wan2.2-TI2V-5B --local-dir ./checkpoints/Wan2.2-TI2V-5B
#   - Image encoder (CLIP) from Wan2.1 - Wan2.2-TI2V-5B does not include it
#     CLIP file: models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth
#   - umt5-xxl tokenizer (already downloaded for 14B, shared)

export HYDRA_FULL_ERROR=1

# ============ USER CONFIGURATION ============
# Dataset path (stack_cups in LeRobot format)
STACK_CUPS_DATA_ROOT=${STACK_CUPS_DATA_ROOT:-"/inspire/qb-ilm/project/robot-reasoning/public/data/lerobot/zyf_dataset_have_mp4/stack_cups"}

# Output directory for training checkpoints
OUTPUT_DIR=${OUTPUT_DIR:-"./checkpoints/dreamzero_stack_cups_wan22_lora"}

# Number of GPUs to use (default: all visible GPUs)
if [ -z "${NUM_GPUS}" ]; then
  NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
fi
NUM_GPUS=${NUM_GPUS:-2}

# Base dir for backbone/tokenizer weights (kept alongside the dataset on the shared volume)
BASE_CKPT_DIR=${BASE_CKPT_DIR:-"/inspire/qb-ilm/project/robot-reasoning/public/data/lerobot/zyf_dataset_have_mp4/checkpoints"}

# Wan2.2-TI2V-5B checkpoint (contains: diffusion weights, T5, VAE)
WAN22_CKPT_DIR=${WAN22_CKPT_DIR:-"$BASE_CKPT_DIR/Wan2.2-TI2V-5B"}

# Image encoder: Wan2.2-TI2V-5B does NOT include CLIP - reuse Wan2.1's
IMAGE_ENCODER_DIR=${IMAGE_ENCODER_DIR:-"$BASE_CKPT_DIR/Wan2.1-I2V-14B-480P"}

TOKENIZER_DIR=${TOKENIZER_DIR:-"$BASE_CKPT_DIR/umt5-xxl"}
# =============================================

# ============ AUTO-DOWNLOAD WEIGHTS ============
if [ ! -d "$WAN22_CKPT_DIR" ] || [ -z "$(ls -A "$WAN22_CKPT_DIR" 2>/dev/null)" ]; then
    echo "Wan2.2-TI2V-5B not found at $WAN22_CKPT_DIR. Downloading from HuggingFace..."
    huggingface-cli download Wan-AI/Wan2.2-TI2V-5B --local-dir "$WAN22_CKPT_DIR"
fi

if [ ! -d "$TOKENIZER_DIR" ] || [ -z "$(ls -A "$TOKENIZER_DIR" 2>/dev/null)" ]; then
    echo "umt5-xxl tokenizer not found at $TOKENIZER_DIR. Downloading from HuggingFace..."
    huggingface-cli download google/umt5-xxl --local-dir "$TOKENIZER_DIR"
fi

# Validate image encoder exists (Wan2.1 CLIP file)
if [ ! -f "$IMAGE_ENCODER_DIR/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" ]; then
    echo "ERROR: CLIP image encoder not found at $IMAGE_ENCODER_DIR"
    echo "Download Wan2.1-I2V-14B-480P or set IMAGE_ENCODER_DIR"
    exit 1
fi
# ================================================

# Validate dataset exists
if [ ! -d "$STACK_CUPS_DATA_ROOT" ]; then
    echo "ERROR: stack_cups dataset not found at $STACK_CUPS_DATA_ROOT"
    echo "Set STACK_CUPS_DATA_ROOT to your LeRobot-format stack_cups dataset"
    exit 1
fi

# Validate GEAR metadata was generated
if [ ! -f "$STACK_CUPS_DATA_ROOT/meta/modality.json" ]; then
    echo "ERROR: meta/modality.json missing. Run convert_lerobot_to_gear.py first (see header)."
    exit 1
fi

torchrun --nproc_per_node $NUM_GPUS --standalone groot/vla/experiment/experiment.py \
    report_to=wandb \
    data=dreamzero/stack_cups_relative_wan22 \
    wandb_project=dreamzero \
    train_architecture=lora \
    num_frames=33 \
    action_horizon=24 \
    num_views=3 \
    model=dreamzero/vla \
    model/dreamzero/action_head=wan_flow_matching_action_tf_wan22 \
    model/dreamzero/transform=dreamzero_cotrain \
    num_frame_per_block=2 \
    num_action_per_block=24 \
    num_state_per_block=1 \
    seed=42 \
    training_args.learning_rate=1e-5 \
    training_args.deepspeed="groot/vla/configs/deepspeed/zero2.json" \
    save_steps=2000 \
    training_args.warmup_ratio=0.05 \
    output_dir=$OUTPUT_DIR \
    per_device_train_batch_size=4 \
    max_steps=50000 \
    weight_decay=1e-5 \
    save_total_limit=10 \
    upload_checkpoints=false \
    bf16=true \
    tf32=true \
    eval_bf16=true \
    dataloader_pin_memory=false \
    dataloader_num_workers=1 \
    save_lora_only=true \
    max_chunk_size=4 \
    save_strategy=steps \
    stack_cups_data_root=$STACK_CUPS_DATA_ROOT \
    dit_version=$WAN22_CKPT_DIR \
    text_encoder_pretrained_path=$WAN22_CKPT_DIR/models_t5_umt5-xxl-enc-bf16.pth \
    image_encoder_pretrained_path=$IMAGE_ENCODER_DIR/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth \
    vae_pretrained_path=$WAN22_CKPT_DIR/Wan2.2_VAE.pth \
    tokenizer_path=$TOKENIZER_DIR
