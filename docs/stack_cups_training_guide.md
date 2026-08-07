# DreamZero 训练流程超详解

> 基于 DreamZero + Wan2.2-TI2V-5B + stack_cups 数据集  
> 训练脚本：`scripts/train/stack_cups_training_wan22.sh`  
> 核心代码：`groot/vla/model/dreamzero/action_head/wan_flow_matching_action_tf.py`

---

## 一、端到端数据流追踪（含张量形状）

### 第 1 步：数据加载

```
数据集: stack_cups/episodes/00001/
├── frames.mp4 (3 视角视频，H.264 编码)
│   原始: [T=33, V=3, H=480, W=640, C=3]  (uint8, 0-255)
│   ↓ decord 解码
│   ↓ resize 到 160×320
│   结果: [T=33, V=3, H=160, W=320, C=3]  (uint8)
│
├── states.jsonl (每帧一行 JSON)
│   内容: {"observation.state": [14], "action": [14]}
│   ↓ 读取并切分
│   state: [T=33, 14]  (float32, q99 归一化到 [-1,1])
│   action: [T=33, 14]  (float32, q99 归一化到 [-1,1])
│
└── meta/tasks.jsonl
    task_index=2 → "pick up the red cup"
```

### 第 2 步：多视角拼接（dreamzero_cotrain.py `_prepare_video`）

```python
# 输入: images [V=3, T=33, C=3, H=160, W=320]
# 拼接成 2×2 网格:
#   [cam_high    , cam_right_wrist]
#   [cam_left_wrist, black_screen  ]

concat_images = np.zeros((1, T=33, C=3, 2*160=320, 2*320=640))
concat_images[0, :, :, :160, :320]    = images[0]  # cam_high
concat_images[0, :, :, 160:, :320]     = images[1]  # cam_left_wrist
concat_images[0, :, :, :160, 320:]     = images[2]  # cam_right_wrist
# 右下角保持黑色 (零填充)

# 结果: [1, T=33, 3, 320, 640]  (单视角，2×2 拼接)
```

### 第 3 步：语言指令构造（dreamzero_cotrain.py `collate`）

```python
# 原始指令: "pick up the red cup" (来自 tasks.jsonl)
# 拼接成 DROID 格式 prompt:
prompt = f"""
A multi-view video shows that a robot pick up the red cup.
The video is split into three views:
The top view shows the camera view from the robot's wrist,
the bottom-left view shows the camera view from the left exterior camera,
and the bottom-right view shows the camera view from the right exterior camera.
During training, one of the two bottom exterior views may be a black screen (dropped view).
The robot pick up the red cup
"""

# umt5-xxl tokenizer 编码 (max_length=512)
input_ids: [512]  (int32, padding + truncation)
attention_mask: [512]  (bool)
```

### 第 4 步：T5 文本编码（forward 中 `encode_prompt`）

```python
prompt_embs = text_encoder(input_ids, attention_mask)
# 输入: input_ids [B=1, 512]
# T5 (umt5-xxl, 2.5B 参数，冻结)
#   ├── 32 层 Transformer Encoder
#   ├── 每个位置输出 dim=3072
# 输出: prompt_embs [B=1, 512, 3072]  (bfloat16)
# 注: 实际 token 长度可能远小于 512，多余位置 padding 为 0
```

### 第 5 步：CLIP 图像编码（forward 中 `encode_image`）

```python
# 取第一帧作为图像条件
image = videos[:, :, :1]  # [B=1, 3, 1, 320, 640]
image = image.transpose(1, 2)  # [B=1, 1, 3, 320, 640]

# CLIP ViT-Huge (冻结, 0.6B 参数)
#   输入 resize 到 224×224
#   ViT-Huge: patch=14, 深度=32, dim=1280
#   257 tokens = 1 CLS + 256 patch (16×16 网格)

clip_feas = image_encoder.encode_image(image)
# clip_feas: [B=1, 257, 1280]  (bfloat16)

# 同时用 VAE 编码第一帧（作为条件帧）
image_zeros = torch.zeros(B, 3, 32, 320, 640)  # 后续帧补零
y = vae.encode(concat_image)  # [B=1, 48, 1+(33-1)/4=9, 10, 20]

# 构造 mask: 只有第一帧有效
msk = zeros(B, 4, 9, 10, 20)
msk[:, :, 0:1, :, :] = 1  # 第一帧 mask=1

# 拼接: [B=1, 4+48=52, 9, 10, 20]
ys = concat([msk, y], dim=1)  # 52 通道条件 latent
```

### 第 6 步：VAE 视频编码（forward 中 `encode_video`）

```python
# 视频归一化到 [-1, 1]
videos = videos.float() / 255.0  # [B=1, 33, 3, 320, 640]
videos = videos * 2 - 1

# WanVideoVAE38 (冻结)
#   编码器: Conv3d(3→48, stride_t=1, stride_h=16, stride_w=16)
#   时序无下采样，空间 16× 下采样
#   48 通道 latent 空间

latents = vae.encode(videos, tiled=True, tile_size=(34,34), tile_stride=(18,16))
# latents: [B=1, 48, 33, 10, 20]
#   33 帧 → 33 帧 (时序不变)
#   320×640 → 10×20 (16× 下采样)
#   3 通道 → 48 通道
```

### 第 7 步：Flow Matching 加噪（forward 中）

```python
# 生成噪声
noise = torch.randn_like(latents)  # [B=1, 48, 33, 10, 20]
noise_action = torch.randn_like(actions)  # [B=1, 24, 14]

# === 视频 timestep 采样 ===
# 使用 Beta(3, 1) 分布 → 偏向高噪声 (均值 0.75)
video_noise_ratio ~ Beta(3, 1)  # [B=1, 33]
timestep_id = ((1.0 - video_noise_ratio) * 1000).long()  # [B=1, 33]

# === 动作 timestep 采样 ===
# 独立均匀分布 [0, 1000]
timestep_action_id = torch.randint(0, 1000, (B=1, 24))  # [B=1, 24]

# === Flow Matching 公式 ===
# add_noise: sample = (1 - σ) * original + σ * noise
# σ = scheduler.sigmas[timestep_id]
# sigmas 通过 shift=5 调整，使 timestep 分布更均匀

noisy_latents = scheduler.add_noise(latents, noise, timestep)
# noisy_latents: [B=1, 48, 33, 10, 20]

noisy_actions = scheduler.add_noise(actions, noise_action, timestep_action)
# noisy_actions: [B=1, 24, 14]

# === 训练目标 ===
# training_target = noise - sample (即 "要去噪的方向")
training_target = scheduler.training_target(latents, noise, timestep)
# training_target: [B=1, 48, 33, 10, 20]

training_target_action = scheduler.training_target(actions, noise_action, timestep_action)
# training_target_action: [B=1, 24, 14]
```

### 第 8 步：DiT 前向推理

```python
# === DiT 输入 ===
# noisy_latents: [B=1, 33, 48, 10, 20]  (注意：转置为 [B, T, C, H, W])
# timestep: [B=1, 33]  (float, 0-1000)
# clip_feature: [B=1, 33, 257, 1280]  (视觉条件)
# y (state): [B=1, 1, 14]  (关节状态)
# context (prompt_embs): [B=1, 512, 3072]  (文本条件)
# seq_len: 50 = (10//2) * (20//2) = 5*10 (DiT patch_embedding 输出)
# action: [B=1, 24, 14]  (加噪后的动作)
# timestep_action: [B=1, 24]  (动作 timestep)

# === DiT 内部结构 ===
# patch_embedding: Conv3d(48 → 3072, kernel=(1,2,2), stride=(1,2,2))
#   [B=1, 33, 48, 10, 20] → [B=1, 33, 3072, 5, 10]
#   每个空间位置 (5×10=50) 变成 3072 维 token
#   token 序列: 33 * 50 = 1650 tokens

# 30 × CausalWanAttentionBlock:
#   ├── causal self-attention (时序因果掩码)
#   │   num_heads=24, head_dim=128
#   │   每个 token 只能 attend 到当前及之前帧的 token
#   ├── cross-attention (对 prompt_embs)
#   │   注入文本条件
#   ├── cross-attention (对 clip_feas)
#   │   注入视觉条件 (每帧独立 attend)
#   ├── MLP: Linear(3072→14336) → GELU → Linear(14336→3072)
#   ├── state_encoder: Linear(14→3072) (注入关节状态)
#   ├── action_encoder: Linear(14→3072) (注入动作)
#   └── action_decoder: Linear(3072→14) (解码动作预测)

# 输出:
video_noise_pred: [B=1, 33, 48, 10, 20]  (视频噪声预测)
action_noise_pred: [B=1, 24, 14]  (动作噪声预测)
```

### 第 9 步：Loss 计算

```python
# === 视频 Loss ===
# 逐样本 MSE (mean over H, W, C)
dynamics_loss_per_sample = F.mse_loss(
    video_noise_pred.float(),    # [B=1, 33, 48, 10, 20]
    training_target.float(),      # [B=1, 33, 48, 10, 20]
    reduction='none'
).mean(dim=(1, 3, 4))  # → [B=1, 33]

# 加权: 高噪声 timestep 权重更大
# training_weight: 基于 BSM-NTW 加权 (timestep 权重分布)
weight_dynamics = dynamics_loss_per_sample * scheduler.training_weight(timestep)
weighted_dynamics_loss = weight_dynamics.mean()

# === 动作 Loss ===
action_loss_per_sample = F.mse_loss(
    action_noise_pred.float(),   # [B=1, 24, 14]
    training_target_action.float(), # [B=1, 24, 14]
    reduction='none'
) * action_mask  # padding 位置 mask=0

# 乘 has_real_action (有些样本可能是 dream/lapa 实例)
action_loss_per_sample = has_real_action * action_loss_per_sample

weight_action = action_loss_per_sample.mean(dim=2) * scheduler.training_weight(timestep_action)
weighted_action_loss = weight_action.mean()

# === 总 Loss ===
loss = weighted_dynamics_loss + weighted_action_loss
```

---

## 二、Flow Matching 数学原理

### 2.1 从扩散模型到 Flow Matching

```
传统 DDPM/DDIM:
  正向过程 (固定):   q(x_t | x_{t-1}) = N(x_t; sqrt(1-β_t) x_{t-1}, β_t I)
  反向过程 (学习):   p_θ(x_{t-1} | x_t) ≈ N(x_{t-1}; μ_θ(x_t, t), Σ_t)
  训练目标:         L = ||μ_θ(x_t, t) - (1/√α_t)(x_t - √(1-α_t)ε)||²
  推理:             需要 50-1000 步迭代

Flow Matching:
  构造路径:         x_t = (1 - σ_t) x_0 + σ_t ε,  ε ~ N(0, I)
  学习速度场:       v_θ(x_t, t) ≈ dx/dt = ε - x_0
  训练目标:         L = ||v_θ(x_t, t) - (ε - x_0)||²
                   = ||model_output - (noise - sample)||²
  推理:             ODE 积分, 仅需 4-16 步
```

### 2.2 噪声调度详解

```python
# FlowMatchScheduler 初始化
scheduler = FlowMatchScheduler(shift=5, sigma_min=0.003, sigma_max=1.0)
# shift=5 是关键超参，控制 timestep 分布

# set_timesteps(1000, training=True)
# 生成 1000 个 sigma 值 (从 sigma_max 到 sigma_min)
sigmas = linspace(1.0, 0.003, 1000)
# shift 调整: sigmas = 5 * sigmas / (1 + 4 * sigmas)
# 效果: timestep 分布更集中，低 sigma 区域分辨率更高

# 转换为 timestep: timesteps = sigmas * 1000
# timesteps ≈ [1000, 995, ..., 3, 0]

# 训练时 timestep 采样:
# 视频: Beta(3, 1) → 偏向高噪声 (timestep ≈ 250-1000)
# 动作: Uniform(0, 1000) → 全范围覆盖

# training_weight (BSM-NTW 加权):
# 根据 timestep 位置计算权重
# 中间 timestep 权重更大，两端权重更小
y = exp(-2 * ((t - 500) / 500)²)
weights = y / sum(y) * 1000
```

### 2.3 加噪公式

```python
def add_noise(self, original_samples, noise, timestep):
    # 1. 找到 timestep_id 对应的 sigma
    #    timestep_id = argmin(|self.timesteps - timestep|)
    sigma = self.sigmas[timestep_id]

    # 2. Flow Matching 线性插值
    #    sample = (1 - σ) * clean_data + σ * noise
    sample = (1 - sigma) * original_samples + sigma * noise

    # 3. 这就是 Flow Matching 的核心路径
    #    当 σ=0: sample = clean_data
    #    当 σ=1: sample = noise
    #    模型学习从 sample 到 noise 的"速度"
    return sample

def training_target(self, sample, noise, timestep):
    # 目标 = noise - sample = ε - (1-σ)x_0 - σ ε
    #       = (1-σ)(ε - x_0)
    # 这正是 dx/dt 的估计
    target = noise - sample
    return target
```

---

## 三、DiT 架构详解（Wan2.2-TI2V-5B）

### 3.1 整体结构

```
Wan2.2-TI2V-5B DiT (5B 参数):
├── patch_embedding: Conv3d(48→3072, kernel=(1,2,2), stride=(1,2,2))
│   作用: 将 VAE latent 切分为 2×2 patch, 每个 patch → 3072 维 token
│   输入: [B, T=33, 48, H=10, W=20]
│   输出: [B, T=33, 3072, H'=5, W'=10]
│   token 数: 33 × 5 × 10 = 1650
│
├── 30 × CausalWanAttentionBlock
│   ├── Self-Attention (Causal)
│   │   ├── 24 heads, head_dim=128
│   │   ├── 因果掩码: token[i] 只能 attend token[j] where j ≤ i
│   │   └── 时序因果: 帧 t 的 token 只能 attend 帧 ≤ t 的 token
│   │
│   ├── Cross-Attention (文本)
│   │   ├── Q = 3072, K/V = 3072
│   │   ├── 对 prompt_embs [B, 512, 3072] 做 cross-attention
│   │   └── 注入语言条件
│   │
│   ├── Cross-Attention (视觉)
│   │   ├── 对 clip_feas [B, T, 257, 1280] 做 cross-attention
│   │   ├── 注入视觉条件
│   │   └── 每帧独立 attend 对应帧的 clip 特征
│   │
│   ├── MLP (FFN)
│   │   ├── Linear(3072 → 14336)  # 4.7× 扩展
│   │   ├── SiLU 激活
│   │   └── Linear(14336 → 3072)  # 投影回
│   │
│   ├── State Encoder
│   │   ├── Linear(14 → 3072)
│   │   └── 注入关节位置信息
│   │
│   └── Action Encoder/Decoder
│       ├── Encoder: Linear(14 → 3072)
│       ├── Decoder: Linear(3072 → 14)
│       └── 注入并预测动作
│
└── Output Head
    ├── Linear(3072 → 48*2=96)
    │   拆分: [48, 48] = [latent_pred, velocity_pred]
    └── 用于计算 video_noise_pred
```

### 3.2 因果注意力详解

```
Causal Attention Mask (33 帧示例):

帧 0: [✓, ✓, ✓, ..., ✓]  ← 只能 attend 自己
帧 1: [✗, ✓, ✓, ..., ✓]  ← 只能 attend 帧 0, 1
帧 2: [✗, ✗, ✓, ..., ✓]  ← 只能 attend 帧 0, 1, 2
...
帧 32:[✗, ✗, ✗, ..., ✓]  ← 只能 attend 帧 0-32

实现方式:
  mask[i, j] = 0 if frame[j] ≤ frame[i] else -inf
  其中 frame[i] = i // (H' * W') = i // 50
```

### 3.3 条件注入方式

```
文本条件注入 (Cross-Attention):
  Q = h (3072)  ← 来自 token 特征
  K = prompt_embs (3072)  ← 来自 T5 编码器
  V = prompt_embs (3072)
  输出 = softmax(QK^T / √d) × V
  → 每个视频 token 都能"看到"整个文本描述

视觉条件注入 (Per-Frame Cross-Attention):
  Q = h[:, frame_i] (3072)  ← 当前帧 token
  K = clip_feas[:, frame_i] (1280)  ← 对应帧的 CLIP 特征
  V = clip_feas[:, frame_i] (1280)
  先投影到相同维度: K, V: 1280 → 3072
  → 每帧独立 attend 自己的视觉特征 (无因果掩码)

状态条件注入 (State Encoder):
  state [B, 1, 14] → Linear(14, 3072) → state_emb [B, 1, 3072]
  → 加到第一个 token 上 (作为额外条件)

动作条件注入 (Action Encoder):
  action [B, 24, 14] → Linear(14, 3072) → action_emb [B, 24, 3072]
  → 拼接到 video token 序列后面: [video_tokens, action_tokens]
  → 通过 self-attention 交互
```

---

## 四、LoRA 微调机制

### 4.1 为什么用 LoRA

```
全量微调:
  - DiT 5B 参数全部更新
  - 需要 ~20GB+ 显存
  - 训练慢 (每步 ~数秒)
  - 可能遗忘预训练知识

LoRA (Low-Rank Adaptation):
  - 只训练低秩矩阵 A, B
  - 原始权重 W 冻结不变
  - 增量: ΔW = (α/r) × B × A
  - 新输出: y = Wx + (α/r) × B × A × x
```

### 4.2 具体实现

```python
# 配置
LoraConfig(
    r=4,              # rank=4
    lora_alpha=4,     # alpha=4, 缩放因子 = alpha/r = 1
    target_modules=[  # 注入哪些层
        "q",          # Query 投影
        "k",          # Key 投影
        "v",          # Value 投影
        "o",          # Output 投影
        "ffn.0",      # FFN 第一层 (Linear 3072→14336)
        "ffn.2",      # FFN 第三层 (Linear 14336→3072)
    ],
    init_lora_weights="kaiming",  # Kaiming 初始化
)

# 每个目标层的参数增量:
# 原始 q_proj: Linear(3072, 3072) → 3072×3072 = 9,437,184 参数
# LoRA A: Linear(3072, 4) → 3072×4 = 12,288 参数
# LoRA B: Linear(4, 3072) → 4×3072 = 12,288 参数
# 增量: 24,576 / 9,437,184 = 0.26%

# 30 层 × 6 目标层 × 24,576 = 4,423,680 ≈ 4.4M 参数
# 加上 state_encoder, action_encoder, action_decoder 的全量微调
# 总可训练参数 ≈ 5-10M (相比 5B 减少 99.8%)
```

### 4.3 LoRA 参数保存与加载

```python
# 训练时保存 (save_lora_only=True)
# 只保存 LoRA 的 A 和 B 矩阵 (几 MB)
adapter_model.safetensors  ← LoRA 权重
adapter_config.json        ← LoRA 配置

# 推理时加载
# 1. 加载原始 Wan2.2-TI2V-5B 权重
# 2. 注入 LoRA 结构
# 3. 加载 adapter_model.safetensors
# 4. W_eff = W_pretrained + (α/r) × B_saved × A_saved
```

---

## 五、分布式训练详解

### 5.1 DeepSpeed ZeRO-2 配置

```json
// zero2.json
{
  "zero_optimization": {
    "stage": 2,
    "overlap_comm": false,
    "contiguous_gradients": true,
    "sub_group_size": 1000000000,
    "reduce_bucket_size": 100000000
  }
}
```

```
ZeRO-2 分片策略 (4 GPU):
┌─────────────────────────────────────────────────────┐
│ 优化器状态 (Adam):  fp32, 每卡 1/4                  │
│   total: 5B × 4bytes = 20GB → 每卡 5GB              │
│                                                      │
│ 梯度:  fp32, 每卡 1/4                                │
│   total: 5B × 4bytes = 20GB → 每卡 5GB              │
│                                                      │
│ 模型参数:  bf16, 每卡 1 份 (LoRA 可负担)              │
│   每个模型副本: 5B × 2bytes = 10GB                   │
│                                                      │
│ 总显存估算 (每卡):                                   │
│   模型: 10GB (bf16)                                  │
│   + 优化器: 5GB (fp32)                               │
│   + 梯度: 5GB (fp32)                                 │
│   + 激活值: ~5GB (bf16, gradient checkpointing)      │
│   + KV Cache: ~2GB                                   │
│   = ~27GB → 超出 4090 24GB → 需要 CPU offload        │
└─────────────────────────────────────────────────────┘
```

### 5.2 VRAM 管理（CPU Offload）

```python
# 当训练时显存不足，启用 CPU offload
# enable_vram_management() 会将冻结模块的参数 offload 到 CPU

# 训练流程中的显存管理:
# 1. 文本编码阶段: text_encoder on GPU, 其余 offload
# 2. 图像编码阶段: image_encoder + VAE on GPU
# 3. DiT 前向: DiT on GPU, 编码模块 offload
# 4. 反向传播: 同上
# 5. 优化器更新: 优化器 on GPU

# 通过 AutoWrappedModule 实现按需加载/卸载
# 计算时: 参数从 CPU → GPU
# 计算后: 参数从 GPU → CPU
# 代价: 增加 PCIe 传输开销 (~10-20% 训练速度)
```

### 5.3 DDP 梯度同步

```
4 GPU DDP 通信流程:
┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐
│  GPU 0   │     │  GPU 1   │     │  GPU 2   │     │  GPU 3   │
│ batch_0  │     │ batch_1  │     │ batch_2  │     │ batch_3  │
└────┬─────┘     └────┬─────┘     └────┬─────┘     └────┬─────┘
     │                │                │                │
     ▼                ▼                ▼                ▼
  Forward          Forward          Forward          Forward
     │                │                │                │
     ▼                ▼                ▼                ▼
  Loss_0           Loss_1           Loss_2           Loss_3
     │                │                │                │
     ▼                ▼                ▼                ▼
  Backward         Backward         Backward         Backward
     │                │                │                │
     └────────────────┼────────────────┼────────────────┘
                      │                │
                      ▼                ▼
              All-Reduce (平均梯度)
                      │
                      ▼
              Optimizer Step (每卡独立更新)
```

---

## 六、完整训练循环伪代码

```python
def train():
    # 1. 初始化模型
    model = WANPolicyHead(config)
    # 加载预训练权重 (T5, CLIP, VAE, DiT)
    # 注入 LoRA (rank=4)
    # 冻结 text_encoder, image_encoder, vae
    # 设置 optimizer (AdamW, lr=1e-5, weight_decay=1e-5)

    for step in range(max_steps):  # 100000 步
        # 2. 获取 batch
        batch = next(dataloader)
        # batch 包含:
        #   images: [B=1, 33, 3, 320, 640]  (uint8)
        #   text: [B=1, 512]  (int32)
        #   text_attention_mask: [B=1, 512]
        #   state: [B=1, 1, 14]  (float32)
        #   action: [B=1, 24, 14]  (float32)

        # 3. 冻结模块 → eval 模式
        model.set_frozen_modules_to_eval_mode()

        # 4. 视频归一化
        videos = batch["images"].float() / 255.0  # [0, 1]
        videos = videos * 2 - 1  # [-1, 1]

        # 5. T5 文本编码 (torch.no_grad)
        prompt_embs = model.encode_prompt(batch["text"], batch["text_attention_mask"])
        # prompt_embs: [B=1, 512, 3072]

        # 6. 视频 resize
        videos = interpolate(videos, size=(160, 320))
        # videos: [B=1, 33, 3, 160, 320]

        # 7. VAE 视频编码 (torch.no_grad)
        latents = model.encode_video(videos)
        # latents: [B=1, 48, 33, 10, 20]

        # 8. CLIP 图像编码 (torch.no_grad)
        clip_feas, ys, _ = model.encode_image(videos[:, :, :1], 33, 160, 320)
        # clip_feas: [B=1, 33, 257, 1280]
        # ys: [B=1, 52, 9, 10, 20]

        # 9. Flow Matching 加噪
        noise_latents = randn_like(latents)
        noise_action = randn_like(actions)

        video_ratio ~ Beta(3, 1)  # 高噪声偏向
        timestep_id = ((1 - video_ratio) * 1000).long()
        timestep_action_id = randint(0, 1000, (B, 24))

        noisy_latents = scheduler.add_noise(latents, noise_latents, timestep)
        noisy_actions = scheduler.add_noise(actions, noise_action, timestep_action)

        # 10. DiT 前向 (LoRA 参数参与)
        with autocast(dtype=bfloat16):
            video_pred, action_pred = model.dit(
                noisy_latents, timestep=timestep,
                clip_feature=clip_feas, y=ys,
                context=prompt_embs, state=state,
                action=noisy_actions, timestep_action=timestep_action,
            )

            # 11. Loss 计算
            dynamics_loss = mse(video_pred, noise_latents - latents)
            action_loss = mse(action_pred, noise_action - actions)
            loss = dynamics_loss + action_loss

        # 12. 反向传播
        loss.backward()

        # 13. 优化器步进
        optimizer.step()
        optimizer.zero_grad()

        # 14. 日志和检查点
        log_wandb(loss, dynamics_loss, action_loss)
        if step % 500 == 0:
            save_lora_checkpoint()  # 只保存 LoRA 权重
```

---

## 七、配置文件关系图

```
stack_cups_training_wan22.sh
├── data=dreamzero/stack_cups_relative_wan22.yaml
│   ├── defaults: dreamzero/base_48_wan_fine_aug_relative.yaml
│   │   ├── modality_configs → 定义 video/state/action/language 模态
│   │   ├── transforms → dreamzero_cotrain.py (数据处理)
│   │   └── fps=...
│   ├── image_resolution: 320×160  (Wan2.2 适配)
│   ├── max_state_dim: 14
│   ├── mixture_dataset_cls: ShardedLeRobotMixtureDataset
│   └── stack_cups_data_root: /path/to/stack_cups
│
├── model=dreamzero/vla
│   └── model/dreamzero/action_head=wan_flow_matching_action_tf_wan22.yaml
│       ├── defaults: wan_flow_matching_action_tf.yaml
│       ├── frame_seqlen: 50
│       ├── target_video: 160×320
│       └── diffusion_model_cfg:
│           ├── dim: 3072, in_dim: 48, out_dim: 48
│           ├── num_heads: 24, num_layers: 30
│           └── ffn_dim: 14336
│
├── model/dreamzero/transform=dreamzero_cotrain
│   └── DreamTransform 类 (dreamzero_cotrain.py)
│
├── training_args.deepspeed=zero2.json
│   └── ZeRO-2 配置
│
└── 权重路径:
    ├── dit_version → Wan2.2-TI2V-5B/diffusion_pytorch_model.safetensors
    ├── text_encoder → Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth
    ├── image_encoder → Wan2.1/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth
    ├── vae → Wan2.2-TI2V-5B/Wan2.2_VAE.pth
    └── tokenizer → umt5-xxl/
```

---

## 八、常见问题与排查

### 8.1 KeyError: 'cam_high'

```
原因:
  modality.json 的 original_key 与实际特征名不匹配
  实际特征: observation.images.cam_high
  配置中: cam_high

修复:
  修改 stack_cups/meta/modality.json:
  {
    "video": {
      "cam_high": {
        "original_key": "observation.images.cam_high"  // 加前缀
      }
    }
  }
```

### 8.2 libGL.so.1 缺失

```
原因: 无头服务器缺少 OpenGL 库

解决方案 (Ubuntu 24.04):
  apt-get install -y libgl1-mesa-dev
  # 或
  pip install opencv-python-headless==4.9.0.41

  注意: opencv-python-headless >= 4.10 会缺少 CV_8U 常量
  需要 patch albucore/utils.py:
    cv2.CV_8U → getattr(cv2, 'CV_8U', 0)
```

### 8.3 transformers ↔ deepspeed 循环导入

```
原因: deepspeed hybrid_engine.py 引用 OPTLearnedPositionalEmbedding
     但 transformers 尚未完成初始化

修复: 注释掉 hybrid_engine.py 第 25 行
  # OPTLearnedPositionalEmbedding = transformers.models.opt.modeling_opt.OPTLearnedPositionalEmbedding
```

### 8.4 显存不足 (OOM)

```
原因: 4090 (24GB) 无法容纳完整 5B 模型 + ZeRO-2

解决方案:
  1. 启用 CPU offload (enable_vram_management)
  2. 使用 gradient checkpointing (已默认开启)
  3. 减小 batch_size (per_device_train_batch_size=1)
  4. 使用 LoRA 微调 (减少可训练参数)
  5. 切换到 ZeRO-3 (分片参数到多卡)
  6. 使用更小模型 (如 1.3B 或 0.5B)
```

### 8.5 Loss 为 NaN

```
原因:
  1. 输入数据有 NaN/Inf (检查 states.jsonl)
  2. 学习率过大 (当前 1e-5, 不宜超过 1e-4)
  3. bfloat16 精度溢出

排查:
  1. 在 forward 中加 assert: assert not torch.isnan(loss)
  2. 打印各中间张量的 max/min
  3. 减小学习率试
```

---

## 九、训练速度与资源估算

### 9.1 单步耗时估算 (4 × 4090)

```
单步计算分解 (batch_size=1, 33 帧):
├── 数据加载:          ~200ms (decord 解码 + 预处理)
├── T5 文本编码:       ~100ms (冻结, bfloat16)
├── CLIP 图像编码:     ~150ms (冻结, ViT-Huge)
├── VAE 视频编码:      ~300ms (冻结, 33 帧)
├── DiT 前向 (LoRA):  ~2000ms (30 层, 1650 tokens)
├── Loss 计算:         ~50ms
├── 反向传播:          ~2500ms (只更新 LoRA)
├── Optimizer:         ~100ms
└── 总计:              ~5400ms ≈ 5.4s/step

考虑 CPU offload 开销: ~+20%
实际: ~6.5s/step → ~4.6 步/秒
```

### 9.2 完整训练时间

```
总步数: 100,000 步
每步时间: ~6.5s
总时间: 100,000 × 6.5s = 650,000s ≈ 180.6 小时 ≈ 7.5 天

4 × H200 估算:
  H200 比 4090 快 ~2-3× (FP8 + HBM3e)
  预计: ~2-3 天

4 × H100 估算:
  H100 比 4090 快 ~1.5-2×
  预计: ~4-5 天
```

---

## 十、快速上手命令清单

```bash
# 1. 环境准备
source .venv/bin/activate
export BASE_CKPT_DIR="/inspire/.../checkpoints"
export STACK_CUPS_DATA_ROOT="/inspire/.../stack_cups"

# 2. 冒烟测试 (验证流程, 5 步)
NUM_GPUS=4 max_steps=5 bash scripts/train/stack_cups_training_wan22.sh
# 预期输出: loss 从 ~1.0 下降到 ~0.1, 无报错

# 3. 快速训练 (1000 步, ~2 小时)
NUM_GPUS=4 max_steps=1000 bash scripts/train/stack_cups_training_wan22.sh

# 4. 完整训练 (100k 步, 后台运行)
nohup bash scripts/train/stack_cups_training_wan22.sh > train.log 2>&1 &
tail -f train.log

# 5. 检查点恢复
# 自动从 output_dir 中最新的 checkpoint 恢复
# 或指定: training_args.resume_from_checkpoint=./checkpoints/dreamzero_stack_cups_wan22_lora/checkpoint-5000

# 6. 仅跑验证 (不训练)
NUM_GPUS=1 max_steps=0 bash scripts/train/stack_cups_training_wan22.sh
```

---

## 附录 A: 关键文件路径索引

| 文件 | 作用 |
|------|------|
| [stack_cups_training_wan22.sh](file:///z:/CodeLibraryI/Lab/dreamzero/scripts/train/stack_cups_training_wan22.sh) | 训练启动脚本 |
| [wan_flow_matching_action_tf.py](file:///z:/CodeLibraryI/Lab/dreamzero/groot/vla/model/dreamzero/action_head/wan_flow_matching_action_tf.py) | VLA 模型核心 (forward/Loss) |
| [dreamzero_cotrain.py](file:///z:/CodeLibraryI/Lab/dreamzero/groot/vla/model/dreamzero/transform/dreamzero_cotrain.py) | 数据预处理 (视频拼接/语言构造) |
| [flow_match_scheduler.py](file:///z:/CodeLibraryI/Lab/dreamzero/groot/vla/model/dreamzero/modules/flow_match_scheduler.py) | Flow Matching 调度器 |
| [wan_flow_matching_action_tf_wan22.yaml](file:///z:/CodeLibraryI/Lab/dreamzero/groot/vla/configs/model/dreamzero/action_head/wan_flow_matching_action_tf_wan22.yaml) | 5B 模型配置 |
| [stack_cups_relative_wan22.yaml](file:///z:/CodeLibraryI/Lab/dreamzero/groot/vla/configs/data/dreamzero/stack_cups_relative_wan22.yaml) | stack_cups 数据配置 |
| [zero2.json](file:///z:/CodeLibraryI/Lab/dreamzero/groot/vla/configs/deepspeed/zero2.json) | DeepSpeed ZeRO-2 配置 |

## 附录 B: 术语表

| 缩写 | 全称 | 说明 |
|------|------|------|
| VLA | Vision-Language-Action | 视觉-语言-动作模型 |
| DiT | Diffusion Transformer | 扩散 Transformer |
| Flow Matching | 一种扩散训练范式 | 学习速度场而非反向过程 |
| LoRA | Low-Rank Adaptation | 低秩适应微调 |
| VAE | Variational Autoencoder | 变分自编码器 (视频压缩) |
| CLIP | Contrastive Language-Image Pre-training | 对比学习视觉编码器 |
| T5 | Text-to-Text Transfer Transformer | 文本编码器 |
| PEFT | Parameter-Efficient Fine-Tuning | 参数高效微调 |
| ZeRO | Zero Redundancy Optimizer | DeepSpeed 优化器 |
| DDP | DistributedDataParallel | 分布式数据并行 |
| LeRobot | 机器人学习数据集格式 | HuggingFace 的机器人数据标准 |
| GEAR | 机器人数据格式 | DreamZero 项目使用的格式 |
