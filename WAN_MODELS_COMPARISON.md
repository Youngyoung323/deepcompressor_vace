# Wan2.1 模型家族对比 — 面向量化的上下文反思

> 本文件聚焦于 `diffsynth/models/` 目录下 Wan 系列模型的代码结构对比，
> 为量化 Wan2.1-1.3B（非 VACE）提供决策依据。

---

## 一、Wan 模型文件全景

`diffsynth/models/` 下的 wan 相关文件：

| 文件 | 模型类 | 用途 |
|------|--------|------|
| `wan_video_dit.py` | `WanModel`, `DiTBlock` | **核心主干**，T2V / I2V / VACE 共享 |
| `wan_video_vace.py` | `VaceWanModel`, `VaceWanAttentionBlock` | VACE 条件旁路 |
| `wan_video_dit_s2v.py` | `WanS2VModel`, `WanS2VDiTBlock` | Speech-to-Video 变体 |
| `wan_video_mot.py` | `MotWanModel`, `MotWanAttentionBlock` | Video-As-Prompt (VAP) motion 旁路 |
| `wan_video_vae.py` | `WanVideoVAE` | VAE 编解码器 |
| `wan_video_text_encoder.py` | | T5 文本编码器 |
| `wan_video_image_encoder.py` | | CLIP 图像编码器 |
| `wan_video_camera_controller.py` | `SimpleAdapter` | 相机控制适配器 |
| `wan_video_motion_controller.py` | | 运动控制器 |
| `wan_video_animate_adapter.py` | | Animate 适配器 |

---

## 二、核心发现：Wan2.1-T2V-1.3B 与 VACE-1.3B 使用完全相同的主干类

从 `model_configs.py` 中的配置可以明确看到：

### Wan2.1-T2V-1.3B（纯 T2V，无 VACE）

```python
{
    "model_hash": "9269f8db9040a9d860eaca435be61814",
    "model_name": "wan_video_dit",
    "model_class": "diffsynth.models.wan_video_dit.WanModel",
    "extra_kwargs": {
        'has_image_input': False,
        'patch_size': [1, 2, 2],
        'in_dim': 16,
        'dim': 1536,
        'ffn_dim': 8960,
        'freq_dim': 256,
        'text_dim': 4096,
        'out_dim': 16,
        'num_heads': 12,
        'num_layers': 30,
        'eps': 1e-06,
    }
    # 无 state_dict_converter（原始权重格式直接加载）
}
```

### VACE-Wan2.1-1.3B（同一权重文件加载两次）

**主干部分**（与上面完全相同的 WanModel 类和参数）：
```python
{
    "model_hash": "a61453409b67cd3246cf0c3bebad47ba",
    "model_name": "wan_video_dit",
    "model_class": "diffsynth.models.wan_video_dit.WanModel",
    "extra_kwargs": {  # 与 T2V-1.3B 完全一致
        'has_image_input': False, 'dim': 1536, 'ffn_dim': 8960,
        'num_heads': 12, 'num_layers': 30, ...
    },
    "state_dict_converter": "WanVideoDiTStateDictConverter"  # 过滤掉 vace.* 前缀
}
```

**VACE 旁路部分**（VaceWanModel，同一文件的另一半权重）：
```python
{
    "model_hash": "a61453409b67cd3246cf0c3bebad47ba",  # 同一文件
    "model_name": "wan_video_vace",
    "model_class": "diffsynth.models.wan_video_vace.VaceWanModel",
    # 无 extra_kwargs → 使用默认值：
    # vace_layers=(0,2,4,6,8,10,12,14,16,18,20,22,24,26,28), dim=1536, num_heads=12, ffn_dim=8960
    "state_dict_converter": "VaceWanModelDictConverter"  # 只保留 vace.* 前缀
}
```

**关键结论**：量化 Wan2.1-T2V-1.3B 的 `WanModel` 主干，与量化 VACE-1.3B 的主干部分，
**代码路径完全一致**——同一个 `WanModel` 类、同样的参数。区别仅在于：
- T2V-1.3B：只有主干，无 VACE 旁路
- VACE-1.3B：主干 + VACE 旁路（额外 15 个 VaceWanAttentionBlock）

---

## 三、1.3B vs 14B 参数对比

| 参数 | Wan2.1-T2V-1.3B | Wan2.1-T2V-14B | VACE-14B |
|------|-----------------|----------------|----------|
| `dim` | 1536 | 5120 | 5120 |
| `ffn_dim` | 8960 | 13824 | 13824 |
| `num_heads` | 12 | 40 | 40 |
| `num_layers` | 30 | 40 | 40 |
| `in_dim` | 16 | 16 | 16 |
| `out_dim` | 16 | 16 | 16 |
| `text_dim` | 4096 | 4096 | 4096 |
| `freq_dim` | 256 | 256 | 256 |
| `patch_size` | (1,2,2) | (1,2,2) | (1,2,2) |
| `has_image_input` | False | False | False |
| head_dim | 128 | 128 | 128 |
| 模型类 | WanModel | WanModel | WanModel |

**结构完全一致**，仅尺寸不同。这意味着：
- 已有的 `WanDiTStruct` 直接适用于 1.3B
- 已有的 `WanCalibCacheLoader` 直接适用于 1.3B
- 量化配置只需调整路径和可能的超参

---

## 四、1.3B 量化的序列长度优势

对于同样的 480×832、81帧视频：

```
tokens = 21 × 30 × 52 = 32,760  （与 14B 相同，取决于输入分辨率而非模型大小）
```

但每个 token 的处理代价小得多：
- 14B: Linear(5120, 5120) → 26.2M 参数/层
- 1.3B: Linear(1536, 1536) → 2.36M 参数/层 → **11× 更小**

Smooth grid search 每次评估的 matmul：
- 14B: (1, 32760, 5120) × (5120, 5120)
- 1.3B: (1, 32760, 1536) × (1536, 1536) → **约 11× 更快**

预估 smooth 耗时（100 samples, 与 14B 同配置）：
- 14B: ~128 小时
- 1.3B: ~**12 小时**（11× 加速）

---

## 五、VACE 旁路与主干的关键差异

### 5.1 VaceWanAttentionBlock vs DiTBlock

`VaceWanAttentionBlock` 继承自 `DiTBlock`，内部结构完全一致：

```
DiTBlock:                        VaceWanAttentionBlock (继承 DiTBlock):
├── modulation (Parameter)       ├── modulation (继承)
├── norm1 (LayerNorm)            ├── norm1 (继承)
├── self_attn (SelfAttention)    ├── self_attn (继承)
├── norm3 (LayerNorm)            ├── norm3 (继承)
├── cross_attn (CrossAttention)  ├── cross_attn (继承)
├── norm2 (LayerNorm)            ├── norm2 (继承)
├── ffn (Sequential)             ├── ffn (继承)
├── gate (GateModule)            ├── gate (继承)
                                 ├── before_proj (Linear) ← 仅 block_id==0
                                 └── after_proj (Linear)  ← 所有 block
```

**forward 差异**：
```python
# DiTBlock.forward(x, context, t_mod, freqs):
#   标准 DiT 处理，输入输出都是 (B, L, D)

# VaceWanAttentionBlock.forward(c, x, context, t_mod, freqs):
#   block_id==0: c = before_proj(c) + x; 然后调用 super().forward(c, ...)
#   block_id>0:  从 c 栈中取出最后一个; 然后调用 super().forward(c, ...)
#   每个 block 末尾: c_skip = after_proj(c); 将 c_skip 和 c 入栈
#   输出是一个 stack tensor，维度随 block 增长
```

### 5.2 VaceWanModel vs WanModel

```
WanModel:                        VaceWanModel:
├── patch_embedding (Conv3d)     ├── vace_patch_embedding (Conv3d)
├── text_embedding (Sequential)  │   (无，共享主干)
├── time_embedding (Sequential)  │   (无，共享主干)
├── time_projection (Sequential) │   (无，共享主干)
├── blocks[0..29] (DiTBlock)     ├── vace_blocks[0..14] (VaceWanAttentionBlock)
├── head (Head)                  │   (无)
└── freqs (预计算 RoPE)          └── vace_layers_mapping (dict)
```

### 5.3 VACE-1.3B vs VACE-14B 旁路的差异

| 参数 | VACE-1.3B | VACE-14B |
|------|-----------|----------|
| vace_layers | (0,2,4,...,28) — 15层 | (0,5,10,...,35) — 8层 |
| dim | 1536 | 5120 |
| num_heads | 12 | 40 |
| ffn_dim | 8960 | 13824 |
| vace_in_dim | 96 | 96 |

1.3B 旁路块数更多（15 vs 8）但每块更小。

---

## 六、其他 Wan 变体模型分析

### 6.1 WanS2VModel (`wan_video_dit_s2v.py`) — Speech-to-Video

- 继承思路类似，`WanS2VDiTBlock` 继承自 `DiTBlock`
- 额外组件：`CausalAudioEncoder`, `AudioInjector_WAN`, `FramePackMotioner`
- `WanS2VDiTBlock.forward()` 多了 `seq_len_x` 参数，用于区分生成帧和参考帧的 t_mod
- 使用自定义的 `rope_precompute()` 函数（支持多段序列拼接 + 可学习频率）
- **与量化的关系**：S2V 的核心 DiTBlock 结构同样复用了 `DiTBlock`，
  但 forward 签名不同，且附加了大量音频/运动相关模块

### 6.2 MotWanModel (`wan_video_mot.py`) — Video-As-Prompt

- `MotWanAttentionBlock` 继承 `DiTBlock`，但 forward 完全重写
- 核心区别：self-attention 时将**主干的 QKV 与 motion 的 QKV 拼接后一起做 attention**，再拆分
- 有自己的 `patch_embedding`, `text_embedding`, `time_embedding` 等
- **与量化的关系**：forward 中直接操作了 `wan_block`（主干 block）的子模块，
  是"寄生式"设计，不能独立量化

---

## 七、权重文件加载机制

### 7.1 state_dict_converter 机制

同一个 `.safetensors` 文件通过不同的 converter 加载为不同的模型：

| Converter | 逻辑 | 用于 |
|-----------|------|------|
| `WanVideoDiTStateDictConverter` | 过滤掉 `vace.*` / `pose_*` / `face_*` / `motion_*` 前缀 | VACE 模型的主干部分 |
| `VaceWanModelDictConverter` | 只保留 `vace.*` 前缀 | VACE 旁路部分 |
| `WanVideoDiTFromDiffusers` | diffusers 格式重命名 (attn1→self_attn, attn2→cross_attn 等) | diffusers 格式权重 |
| (无 converter) | 直接加载 | 原始格式的 T2V-1.3B / T2V-14B |

### 7.2 Wan2.1-T2V-1.3B 的权重加载

T2V-1.3B 配置中**没有 `state_dict_converter`**，意味着权重 key 直接匹配 `WanModel` 的参数名。
权重文件中的 key 格式举例：
```
blocks.0.self_attn.q.weight
blocks.0.self_attn.q.bias
blocks.0.cross_attn.k.weight
blocks.0.ffn.0.weight
blocks.0.ffn.2.weight
blocks.0.modulation
...
patch_embedding.weight
text_embedding.0.weight
time_embedding.0.weight
head.head.weight
head.modulation
```

---

## 八、量化 Wan2.1-T2V-1.3B 的可行性分析

### 8.1 可直接复用的组件

| 组件 | 文件 | 状态 |
|------|------|------|
| `WanDiTStruct` | `deepcompressor/app/diffusion/nn/wan_struct.py` | ✅ 直接可用 |
| `WanTransformerBlockStruct` | 同上 | ✅ 直接可用 |
| `WanAttentionStruct` | 同上 | ✅ 直接可用 |
| `WanFeedForwardStruct` | 同上 | ✅ 直接可用 |
| `WanCalibCacheLoader` | `deepcompressor/app/diffusion/dataset/calib_wan_loader.py` | ✅ 直接可用 |
| `ModelFnCollectHook` | `deepcompressor/app/diffusion/dataset/collect/utils.py` | ✅ 直接可用 |

### 8.2 需要修改/新建的组件

| 组件 | 改动 |
|------|------|
| `ptq_vace.py` | 修改 `_build_wan_vace_pipeline()` 中的模型路径，指向 1.3B |
| YAML 配置 | 新建 `wan2.1-t2v-1.3b.yaml`，调整路径和超参 |
| 校准数据收集 | `calib_wan.py` 调整路径，T2V 模式无需 VACE 相关输入 |
| Pipeline 工厂 | 注册 `wan2.1-t2v-1.3b` 工厂（或复用，仅改路径）|

### 8.3 T2V-1.3B vs VACE 场景的简化

量化 T2V-1.3B 相比 VACE-14B **大幅简化**：
1. **无 VACE 旁路**：不需要创建 `VaceDiTStruct`，不需要旁路校准
2. **模型更小**：smooth/lowrank 计算量降低 ~11×
3. **标准 forward**：`WanModel.forward()` 被直接调用（非 VACE 的 model_fn 绕过模式），
   校准数据收集更简单
4. **无 vace_context**：校准数据只需 `(latent, timestep, context)`，
   不涉及 VACE 条件处理

### 8.4 注意事项

1. **model_fn 绕过问题依然存在**：即使是 T2V-1.3B，`model_fn_wan_video` 仍然直接
   操作 `dit` 子模块而不调用 `dit.forward()`。校准数据收集仍需使用 `ModelFnCollectHook`。
2. **T2V 无 CLIP/VAE 图像输入**：`has_image_input=False`，无 `img_emb`，
   `CrossAttention` 中无 `k_img/v_img`。struct 中需确保对 `None` 的处理正确。
3. **序列长度仍然大**：32,760 tokens（480×832@81帧），但 hidden_dim 小（1536 vs 5120），
   所以整体计算量可控。

---

## 九、总结

| 维度 | Wan2.1-T2V-1.3B | VACE-1.3B | VACE-14B |
|------|-----------------|-----------|----------|
| 主干类 | WanModel | WanModel | WanModel |
| 主干参数 | dim=1536, 30层 | dim=1536, 30层 | dim=5120, 40层 |
| VACE 旁路 | **无** | 15块 VaceWanAttentionBlock | 8块 VaceWanAttentionBlock |
| 权重文件 | 独立 .safetensors | 主干+旁路合一 | 主干+旁路合一 |
| 量化复杂度 | **最低** — 仅主干 | 中等 — 主干+旁路 | 最高 — 大模型+主干+旁路 |
| 预估 smooth 耗时 | ~12h (100 samples) | ~12h 主干 + ~5h 旁路 | ~128h 主干 + ~25h 旁路 |
| 现有代码可复用度 | **几乎 100%** | ~70%（需旁路 struct） | ~70%（需旁路 struct） |

**结论**：量化 Wan2.1-T2V-1.3B 是当前最务实的选择。现有的 `WanDiTStruct` + `WanCalibCacheLoader` 
可以直接使用，只需：
1. 收集 T2V-1.3B 的校准数据
2. 新建一个 pipeline 工厂和 YAML 配置
3. 运行 PTQ

---

## 十、现有 VACE-14B 量化框架代码深度分析

> 以下对 `deepcompressor/app/diffusion/` 目录下的 VACE-14B 量化实现进行逐文件分析，
> 重点关注：**在现有框架下量化 Wan2.1-T2V-1.3B 需要做哪些修改**。

### 10.1 代码架构总览

```
deepcompressor/app/diffusion/
├── ptq_vace.py          ★ VACE 入口：Pipeline 构建 + Struct 注册 + 猴子补丁
├── ptq.py               ★ 通用 PTQ 流程：smooth → rotate → weight → activation
├── config.py              顶层配置 DiffusionPtqRunConfig
├── utils.py               控制图/mask 工具
├── nn/
│   ├── struct.py          通用 DiffusionModelStruct（UNet/DiT/Flux/PixArt...）
│   └── wan_struct.py    ★ Wan 专用 Struct（WanDiTStruct, WanTransformerBlockStruct 等）
├── dataset/
│   ├── calib_wan_loader.py ★ Wan 校准数据加载器
│   └── collect/
│       ├── calib_wan.py   ★ VACE 校准数据收集脚本
│       └── utils.py       ★ ModelFnCollectHook（拦截 model_fn）
├── pipeline/
│   └── config.py          Pipeline 工厂注册机制
└── quant/                   smooth/weight/activation 量化算法（通用）
```

### 10.2 ptq_vace.py — 入口文件详细分析

该文件完成四件事：

**① 猴子补丁 build_loader → WanCalibCacheLoader**
```python
DiffusionCalibCacheLoaderConfig.build_loader = _wan_build_loader
```
将通用的校准缓存加载器替换为 Wan 专用的 `WanCalibCacheLoader`。这是**全局替换**，
一旦 import ptq_vace，所有后续调用都会使用 Wan 加载器。

**② 注册 Pipeline 工厂 `wan2.1-vace-14b`**
```python
DiffusionPipelineConfig.register_pipeline_factory(
    "wan2.1-vace-14b", _build_wan_vace_pipeline
)
```
工厂函数 `_build_wan_vace_pipeline` 硬编码了：
- VACE-14B 权重路径：`Wan-AI/Wan2.1-VACE-14B/diffusion_pytorch_model*.safetensors`
- T5 编码器路径：`models_t5_umt5-xxl-enc-bf16.safetensors`
- VAE 路径：`Wan2.1_VAE.safetensors`
- LoRA 加载：`wan2.1_t2v_14b_lora_rank64_lightx2v_4step.safetensors`
- Tokenizer 路径：`Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl`

**③ 覆盖 DiffusionModelStruct._default_construct**
```python
def _extended_default_construct(module, /, ...):
    if isinstance(module, WanVideoPipeline):
        module = module.dit  # 从 Pipeline 中提取 dit
    if isinstance(module, WanModel):
        return WanDiTStruct.construct(module, ...)  # 构建 Wan 专用 Struct
    return _orig_default_construct(module, ...)
```
关键：`pipeline.dit` 被提取出来构建为 `WanDiTStruct`。VACE 旁路 `pipeline.vace` **不在此处处理**。

**④ generate_vace / main_vace — 推理与量化入口**
- `main_vace()` 调用 `config.pipeline.build()` 构建 pipeline
- 如果开启量化，构建 `DiffusionModelStruct`（→ 提取 dit → `WanDiTStruct`）
- 调用 `ptq(model, config.quant, ...)` 进行量化
- 可选地调用 `generate_vace()` 生成视频

### 10.3 wan_struct.py — Struct 层次结构

这是整个量化框架的核心。Struct 定义了模型的"抽象结构图"，使通用量化算法能遍历模型。

```
WanDiTStruct (DiffusionModelStruct)
├── pre_module_structs:
│   ├── patch_embedding → DiffusionModuleStruct (rkey="input_embed")
│   ├── time_embedding  → DiffusionModuleStruct (rkey="time_embed")
│   └── text_embedding  → DiffusionModuleStruct (rkey="text_embed")
├── block_structs_list[0..N-1]:
│   └── WanTransformerBlockStruct (DiffusionTransformerBlockStruct)
│       ├── pre_attn_norms: [norm1, norm3]
│       ├── attns:
│       │   ├── [0] self_attn → WanAttentionStruct
│       │   │       q_proj=q, k_proj=k, v_proj=v, o_proj=o
│       │   │       add_k_proj=k_img (if has_image_input), add_v_proj=v_img
│       │   └── [1] cross_attn → WanAttentionStruct
│       │           q_proj=q, add_k_proj=k, add_v_proj=v, o_proj=o
│       ├── pre_ffn_norm: norm2
│       └── ffn → WanFeedForwardStruct
│               up_projs=[ffn.0], down_projs=[ffn.2]
└── post_module_structs:
    └── head → DiffusionModuleStruct (rkey="output_embed")
```

**工厂注册**（模块导入时自动执行）：
```python
WanAttentionStruct.register_factory((SelfAttention, CrossAttention), ...)
WanFeedForwardStruct.register_factory(nn.Sequential, ...)
WanTransformerBlockStruct.register_factory(DiTBlock, ...)
WanDiTStruct.register_factory(WanModel, ...)
```

**关键细节**：
- `has_image_input=False` 时，`SelfAttention` 的 `add_k_proj` / `add_v_proj` 为 `None`（无 k_img/v_img）
- `CrossAttention` 始终将 k/v 视为 add_k_proj/add_v_proj（来自 context）
- FFN 结构固定为 `Sequential(Linear, GELU, Linear)`
- 1.3B 和 14B **使用完全相同的 Struct 类**，区别仅在于 dim/ffn_dim/num_heads/num_layers

### 10.4 calib_wan_loader.py — 校准数据加载

`WanCalibCacheLoader` 继承自 `BaseCalibCacheLoader`，实现了：

**数据加载**：从 `.pt` 文件加载预收集的校准数据，格式为：
```python
{
    "input_args": [latent_tensor],          # (1, C, F, H, W)
    "input_kwargs": {
        "timestep": tensor,
        "context": tensor,
        "clip_feature": tensor_or_None,
        "y": tensor_or_None,
    },
    "outputs": [noise_pred_tensor],
    "sample_id": str,
    "step": int,
    "guidance": int,
}
```

**iter_samples()**：将 `.pt` 文件转为 `ModuleForwardInput` 供 ptq 使用。

**_init_cache()**：为不同模块类型创建不同的缓存结构：
- `nn.Linear` → 标准 channels_dim=-1
- `SelfAttention` → inputs={x, freqs}, outputs={tensor}
- `CrossAttention` → inputs={x, y}, outputs={tensor}

**WanConcatCacheAction**：处理 Wan 特有的 freqs 维度对齐问题
（freqs 是 (L, 1, D)，需要添加 batch 维成为 (1, L, 1, D)）。

**iter_layer_activations()**：逐层遍历 DiTBlock，收集校准数据。
对 self_attn，q/k/v 共享同一个 cache（因为输入相同）。

### 10.5 collect/calib_wan.py — 校准数据收集

该脚本专为 VACE-14B 设计：
- 从 VACE-Benchmark JSONL 文件加载样本
- 构建 `WanVideoPipeline`（硬编码 VACE-14B 路径 + LoRA）
- 使用 `ModelFnCollectHook` 拦截 `pipeline.model_fn` 调用
- 收集 `(latents, timestep, context, clip_feature, y)` 保存为 `.pt` 文件

**ModelFnCollectHook**（utils.py）的关键行为：
```python
KWARGS_TENSOR_KEYS = ("timestep", "context", "clip_feature", "y")
```
仅收集这 4 个 kwargs。**未收集 `vace_context`**，因为当前只量化主干。

### 10.6 YAML 配置 wan2.1-vace-14b.yaml

```yaml
pipeline:
  name: wan2.1-vace-14b
  dtype: torch.bfloat16
  path: /data1/lyf/Lab/DiffSynth-Studio/models
quant:
  calib:
    path: datasets/{dtype}/Wan2.1-VACE-14B-lora-20steps/VACE-benchmark-real/s120/caches
    num_samples: 100
  wgts:
    skips: [embed, transformer_norm, transformer_add_norm, attn_add, ffn_add]
  ipts:
    skips: [embed, transformer_norm, transformer_add_norm, attn_add, ffn_add]
```

skips 含义（由 `WanDiTStruct._get_default_key_map()` 定义）：
- `embed` → patch_embedding + time_embedding + text_embedding + head
- `transformer_norm` → norm1, norm2, norm3
- `transformer_add_norm` → cross_attn 的额外 norm（不存在时无害）
- `attn_add` → k_img, v_img（`has_image_input=False` 时无模块，无害）
- `ffn_add` → 无额外 FFN（无害）

---

## 十一、量化 Wan2.1-T2V-1.3B 需要的具体修改

### 11.1 需要修改的文件

| 文件 | 修改类型 | 修改内容 |
|------|----------|----------|
| `ptq_vace.py` | **修改或新建** | 注册新的 pipeline 工厂 |
| `collect/calib_wan.py` | **修改或新建** | 适配 T2V 场景的校准数据收集 |
| YAML 配置 | **新建** | `wan2.1-t2v-1.3b.yaml` |

### 11.2 不需要修改的文件

| 文件 | 原因 |
|------|------|
| `nn/wan_struct.py` | WanModel 结构完全一致，Struct 直接复用 |
| `calib_wan_loader.py` | `.pt` 数据格式一致，加载器直接复用 |
| `collect/utils.py` | `ModelFnCollectHook` 通用，直接复用 |
| `ptq.py` | 通用 PTQ 流程，无需修改 |
| `quant/*` | 通用量化算法，无需修改 |

### 11.3 详细修改方案

#### 方案 A：最小修改 — 修改 ptq_vace.py

直接在 `ptq_vace.py` 中添加一个新的 pipeline 工厂：

```python
def _build_wan_t2v_1_3b_pipeline(
    name: str, path: str, dtype, device, shift_activations: bool,
) -> WanVideoPipeline:
    model_configs = [
        ModelConfig(
            path=sorted(glob.glob(os.path.join(
                path, "Wan-AI/Wan2.1-T2V-1.3B/diffusion_pytorch_model*.safetensors"
            ))),
        ),
        ModelConfig(path=os.path.join(path, "T5编码器路径")),
        ModelConfig(path=os.path.join(path, "VAE路径")),
    ]
    tokenizer_config = ModelConfig(path=os.path.join(path, "tokenizer路径"))
    pipeline = WanVideoPipeline.from_pretrained(
        torch_dtype=dtype, device=device,
        model_configs=model_configs, tokenizer_config=tokenizer_config,
    )
    # 注意：T2V-1.3B 可能不需要 LoRA
    return pipeline

DiffusionPipelineConfig.register_pipeline_factory(
    "wan2.1-t2v-1.3b", _build_wan_t2v_1_3b_pipeline
)
```

**关键注意点**：
1. `_extended_default_construct` 已经处理了 `WanModel`，因此 `WanDiTStruct` 构建无需修改。
2. `_wan_build_loader` 猴子补丁也已经在模块级别生效，无需额外处理。
3. 如果不使用 LoRA，可以删除 LoRA 加载行。

#### 方案 B（推荐）：新建 ptq_wan_t2v.py

新建一个独立入口文件，避免与 VACE 逻辑耦合：

```python
# ptq_wan_t2v.py — Wan2.1-T2V-1.3B 量化入口
#
# 与 ptq_vace.py 的区别：
# 1. Pipeline 工厂指向 T2V-1.3B 权重
# 2. 不加载 LoRA
# 3. 使用通用 T2V 提示词而非 VACE 数据集
# 4. generate 函数使用简单的 T2V pipeline 调用
```

#### 校准数据收集

**方案 A：复用 VACE calib_wan.py 的框架**

关键修改点：
```python
# 原始（VACE-14B）：
model_configs = [
    ModelConfig(path=...VACE-14B...),
    ...
]
# 加载 LoRA
pipeline.load_lora(...)
# VACE 数据集（含 vace_video, vace_video_mask, vace_reference_image）
build_pipeline_kwargs_for_sample(sample, height, width)

# 修改为（T2V-1.3B）：
model_configs = [
    ModelConfig(path=...T2V-1.3B...),
    ...
]
# 无 LoRA（除非有 1.3B 专用 LoRA）
# T2V 数据集（仅需 prompt，无 VACE 条件输入）
# pipeline(prompt=prompt, seed=seed, height=height, width=width, ...)
```

**方案 B（推荐）：新建 collect/calib_wan_t2v.py**

更干净的做法，使用简单的文本提示词列表（如 COCO captions）替代 VACE 数据集。

#### YAML 配置

新建 `configs/model/wan2.1-t2v-1.3b.yaml`：

```yaml
pipeline:
  name: wan2.1-t2v-1.3b
  dtype: torch.bfloat16
  path: /data1/lyf/Lab/DiffSynth-Studio/models
eval:
  num_steps: 20
  guidance_scale: 5.0
  protocol: fmeuler{num_steps}-g{guidance_scale}
  height: 480
  width: 832
  num_samples: -1
  benchmarks: []
quant:
  calib:
    data: t2v-benchmark  # 或其他数据集名
    path: datasets/{dtype}/Wan2.1-T2V-1.3B/caches  # 校准数据路径
    batch_size: 1
    num_samples: 100
  wgts:
    calib_range:
      element_batch_size: 64
      sample_batch_size: 16
      element_size: 512
      sample_size: -1
    low_rank:
      sample_batch_size: 1
      sample_size: -1
    skips:
    - embed
    - transformer_norm
    - transformer_add_norm
    - attn_add
    - ffn_add
  ipts:
    calib_range:
      element_batch_size: 64
      sample_batch_size: 1
      element_size: 512
      sample_size: -1
    skips:
    - embed
    - transformer_norm
    - transformer_add_norm
    - attn_add
    - ffn_add
```

### 11.4 model_fn 绕过问题

**这是最需要注意的技术细节。**

`model_fn_wan_video()` 不调用 `dit.forward()`，而是直接操作 dit 的子模块：
```python
# model_fn_wan_video 内部：
t = dit.time_embedding(...)
t_mod = dit.time_projection(t).unflatten(...)
context = dit.text_embedding(context)
x = dit.patchify(x, ...)
# ...
for block in dit.blocks:
    x = block(x, context, t_mod, freqs)
x = dit.head(x, t)
x = dit.unpatchify(x, ...)
```

这意味着：
1. **校准数据收集**：不能用 `dit.register_forward_hook()`，因为 `dit.forward()` 不被调用。
   必须使用 `ModelFnCollectHook` 包装 `pipeline.model_fn`。
2. **校准数据格式**：`.pt` 文件存的是 `model_fn` 的输入 `(latents, timestep, context, ...)`，
   而非 `dit.forward()` 的输入。`WanCalibCacheLoader` 在 `iter_layer_activations` 中
   会对 `WanDiTStruct` 执行 `_iter_layer_activations`，该方法会先对 model 做一次完整 forward
   来收集每层的实际输入。
3. **T2V-1.3B 中 `model_fn` 的行为与 VACE-14B 完全一致**（只是没有 vace 分支），
   因此 `ModelFnCollectHook` 的 `KWARGS_TENSOR_KEYS` 不需要修改。

### 11.5 T2V-1.3B 特殊考虑

| 项目 | VACE-14B | T2V-1.3B | 影响 |
|------|----------|----------|------|
| `has_image_input` | False | False | 无差异 |
| `clip_feature` | None | None | 无差异，收集 hook 中会跳过 |
| `y` (VAE embedding) | None | None | 无差异 |
| LoRA | 14B 4-step LoRA | 无（或有独立 LoRA） | pipeline 构建时处理 |
| VACE 旁路 | 有 vace 模型 | **无** | model_fn 中 vace 分支不执行 |
| num_layers | 40 | 30 | Struct 自动适应 |
| dim | 5120 | 1536 | Struct 自动适应 |
| 序列长度 | 32,760 | 32,760（同分辨率时） | 校准数据大小相同 |
| smooth 耗时 | ~128h (100 samples) | **~12h** (100 samples) | 主要优势 |

### 11.6 工作量估计

| 任务 | 预估时间 | 复杂度 |
|------|----------|--------|
| 编写 pipeline 工厂（ptq_wan_t2v.py 或修改 ptq_vace.py） | 30 min | 低 |
| 编写校准数据收集脚本 | 1-2 h | 低 |
| 编写 YAML 配置 | 15 min | 低 |
| 准备 T2V 校准数据集（提示词列表） | 30 min | 低 |
| 收集校准数据（GPU 运行时间） | ~4-8 h | - |
| 运行 smooth 量化 | ~12 h | - |
| 运行 weight + activation 量化 | ~2-4 h | - |
| 调试与验证 | 2-4 h | 中 |
| **总计（人工时间）** | **~5 h** | **低** |

### 11.7 结论

在当前 VACE-14B 框架下量化 Wan2.1-T2V-1.3B，**核心代码修改量极小**：

1. **Struct 层（wan_struct.py）**：**零修改**。`WanDiTStruct` 及其子 Struct 完全适用于 1.3B。
2. **校准加载器（calib_wan_loader.py）**：**零修改**。数据格式一致。
3. **量化算法（quant/*）**：**零修改**。通用算法不感知模型大小。
4. **入口文件（ptq_vace.py）**：需添加 pipeline 工厂注册（~20 行），或新建独立入口。
5. **校准收集（calib_wan.py）**：需适配 T2V 数据集和 1.3B 权重路径。
6. **YAML 配置**：需新建，但可基于 VACE-14B 的配置简单修改。

**最大风险**：校准数据的质量和数量。T2V-1.3B 的校准数据应该使用 T2V 任务的提示词
（如 COCO captions），而非 VACE 特定的视频编辑任务数据。

---

## 十二、已完成的修改

### 12.1 新建文件列表

| 文件 | 作用 | 状态 |
|------|------|------|
| `deepcompressor/app/diffusion/ptq_wan_t2v.py` | T2V-1.3B 量化入口 | 已完成 |
| `deepcompressor/app/diffusion/dataset/collect/calib_wan_t2v_1_3b.py` | T2V-1.3B 校准数据收集 | 已完成（正在运行） |
| `examples/diffusion/configs/model/wan2.1-t2v-1.3b.yaml` | T2V-1.3B YAML 配置 | 已完成 |

### 12.2 未修改的文件（验证可复用）

| 文件 | 原因 |
|------|------|
| `nn/wan_struct.py` | WanDiTStruct 直接适用于 1.3B，零修改 |
| `calib_wan_loader.py` | WanCalibCacheLoader 数据格式一致，零修改 |
| `collect/utils.py` | ModelFnCollectHook 通用，零修改 |
| `ptq.py` | 通用 PTQ 流程，零修改 |
| `quant/*` | 通用量化算法，零修改 |
| `ptq_vace.py` | VACE-14B 入口，独立不受影响 |

### 12.3 ptq_wan_t2v.py 关键设计

**与 ptq_vace.py 的对比**：

| 项目 | ptq_vace.py | ptq_wan_t2v.py |
|------|-------------|----------------|
| Pipeline 工厂名 | `wan2.1-vace-14b` | `wan2.1-t2v-1.3b` |
| 模型权重 | Wan2.1-VACE-14B | Wan2.1-T2V-1.3B |
| LoRA | 加载 14B 4-step LoRA | 不加载 |
| 生成函数 | `generate_vace()` (VACE 条件输入) | `generate_t2v()` (仅 prompt) |
| 评估数据集 | VACE-Benchmark (JSONL) | Ditto-1M captions (JSON) |
| 猴子补丁 | 同 | 同 (WanCalibCacheLoader + WanDiTStruct) |

**三个猴子补丁（与 ptq_vace.py 完全一致）**：
1. `DiffusionCalibCacheLoaderConfig.build_loader` → `_wan_build_loader` (使用 WanCalibCacheLoader)
2. `DiffusionModelStruct._default_construct` → `_extended_default_construct` (WanModel→WanDiTStruct)
3. `DiffusionModelStruct.register_factory(WanVideoPipeline, ...)` (Pipeline→dit 提取)

### 12.4 wan2.1-t2v-1.3b.yaml 关键配置

```yaml
pipeline:
  name: wan2.1-t2v-1.3b          # 关联 pipeline 工厂
  dtype: torch.bfloat16
  path: /data1/lyf/Lab/DiffSynth-Studio/models
quant:
  calib:
    path: datasets/{dtype}/Wan2.1-T2V-1.3B-40steps/Ditto-1M/s64/caches
    num_samples: 50               # 从 64 收集中选 50
  wgts/ipts/opts:
    skips: [embed, transformer_norm, transformer_add_norm, attn_add, ffn_add]
```

校准数据路径解析：`{dtype}` → `torch.bfloat16`，最终绝对路径由 `DiffusionPtqRunConfig.__post_init__` 拼接。

### 12.5 校准数据收集进度

- 脚本：`calib_wan_t2v_1_3b.py`
- 参数：40 steps, cfg=5.0, 480x832, 81 frames, 64 samples
- 数据源：Ditto-1M captions（按可用视频目录过滤）
- 输出路径：`datasets/torch.bfloat16/Wan2.1-T2V-1.3B-40steps/Ditto-1M/s64/`
- 每个样本产生 40 steps × 2 guidances = 80 个 `.pt` 文件

### 12.6 量化运行方法

校准数据收集完成后，在 `examples/diffusion/` 目录下执行：

```bash
# 运行量化（示例使用 nvfp4 量化配置）
python -m deepcompressor.app.diffusion.ptq_wan_t2v \
    configs/model/wan2.1-t2v-1.3b.yaml \
    configs/svdquant/nvfp4.yaml \
    --skip-eval true

# 如需生成参考视频（不量化）
python -m deepcompressor.app.diffusion.ptq_wan_t2v \
    configs/model/wan2.1-t2v-1.3b.yaml \
    --output-dirname reference \
    --skip-eval true
```
