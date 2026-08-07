#!/bin/bash
# DreamZero stack_cups Training Script
#
# Usage:
#   # Set your dataset path and output directory, then run:
#   bash scripts/train/stack_cups_training.sh
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
#   - Wan2.1-I2V-14B-480P weights (auto-downloaded or pre-downloaded from HuggingFace)
#   - umt5-xxl tokenizer (auto-downloaded or pre-downloaded from HuggingFace)
#   - DreamZero-AgiBot pretrained checkpoint (for loading LoRA weights before fine-tuning)
#     Already downloaded to /inspire/qb-ilm/project/robot-reasoning/public/data/lerobot/zyf_dataset_have_mp4/checkpoint_agibot
#     (override via PRETRAINED_MODEL_PATH)

export HYDRA_FULL_ERROR=1

# ============ HARDCODED PATHS (H200 server, 2x H200) ============
# Dataset path (stack_cups in LeRobot format: state 14, action 14, videos cam_high/cam_left_wrist/cam_right_wrist)
STACK_CUPS_DATA_ROOT="/inspire/qb-ilm/project/robot-reasoning/public/data/lerobot/zyf_dataset_have_mp4/stack_cups"

# Output directory for training checkpoints
OUTPUT_DIR="./checkpoints/dreamzero_stack_cups_lora"

# Number of GPUs to use (auto-detect, default 2 for dual-H200)
if [ -z "${NUM_GPUS}" ]; then
  NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
fi
NUM_GPUS=${NUM_GPUS:-2}

# Model weight paths (Wan2.1-I2V-14B-480P + umt5-xxl)
BASE_CKPT_DIR="/inspire/qb-ilm/project/robot-reasoning/public/data/lerobot/zyf_dataset_have_mp4/checkpoints"
WAN_CKPT_DIR="$BASE_CKPT_DIR/Wan2.1-I2V-14B-480P"
TOKENIZER_DIR="$BASE_CKPT_DIR/umt5-xxl"

# Pretrained DreamZero-AgiBot checkpoint (for loading LoRA weights before fine-tuning)
PRETRAINED_MODEL_PATH="/inspire/qb-ilm/project/robot-reasoning/public/data/lerobot/zyf_dataset_have_mp4/checkpoint_agibot"
# =============================================

# ============ AUTO-DOWNLOAD WEIGHTS ============
if [ ! -d "$WAN_CKPT_DIR" ] || [ -z "$(ls -A "$WAN_CKPT_DIR" 2>/dev/null)" ]; then
    echo "Wan2.1-I2V-14B-480P not found at $WAN_CKPT_DIR. Downloading from HuggingFace..."
    huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P --local-dir "$WAN_CKPT_DIR"
fi

if [ ! -d "$TOKENIZER_DIR" ] || [ -z "$(ls -A "$TOKENIZER_DIR" 2>/dev/null)" ]; then
    echo "umt5-xxl tokenizer not found at $TOKENIZER_DIR. Downloading from HuggingFace..."
    huggingface-cli download google/umt5-xxl --local-dir "$TOKENIZER_DIR"
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

# Validate pretrained checkpoint exists
if [ ! -f "$PRETRAINED_MODEL_PATH/model.safetensors.index.json" ]; then
    echo "ERROR: pretrained checkpoint not found at $PRETRAINED_MODEL_PATH"
    echo "Set PRETRAINED_MODEL_PATH to your DreamZero-AgiBot checkpoint (GEAR-Dreams/DreamZero-AgiBot)"
    exit 1
fi

torchrun --nproc_per_node $NUM_GPUS --standalone groot/vla/experiment/experiment.py \
    report_to=wandb \
    data=dreamzero/stack_cups_relative \
    wandb_project=dreamzero \
    train_architecture=lora \
    num_frames=33 \
    action_horizon=24 \
    num_views=3 \
    model=dreamzero/vla \
    model/dreamzero/action_head=wan_flow_matching_action_tf \
    model/dreamzero/transform=dreamzero_cotrain \
    num_frame_per_block=2 \
    num_action_per_block=24 \
    num_state_per_block=1 \
    seed=42 \
    training_args.learning_rate=1e-5 \
    training_args.deepspeed="groot/vla/configs/deepspeed/zero2.json" \
    save_steps=5000 \
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
    image_resolution_width=320 \
    image_resolution_height=176 \
    save_lora_only=true \
    max_chunk_size=4 \
    frame_seqlen=880 \
    save_strategy=steps \
    stack_cups_data_root=$STACK_CUPS_DATA_ROOT \
    dit_version=$WAN_CKPT_DIR \
    text_encoder_pretrained_path=$WAN_CKPT_DIR/models_t5_umt5-xxl-enc-bf16.pth \
    image_encoder_pretrained_path=$WAN_CKPT_DIR/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth \
    vae_pretrained_path=$WAN_CKPT_DIR/Wan2.1_VAE.pth \
    tokenizer_path=$TOKENIZER_DIR \
    pretrained_model_path=$PRETRAINED_MODEL_PATH \
    ++action_head_cfg.config.skip_component_loading=true \
    ++action_head_cfg.config.defer_lora_injection=true
