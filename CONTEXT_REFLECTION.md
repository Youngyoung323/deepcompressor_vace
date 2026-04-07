# DeepCompressor VACE 量化项目 — 上下文反思

> 本文件用于在上下文窗口有限时，作为对 deepcompressor_vace 项目代码结构、核心逻辑和 VACE 适配现状的快速回顾。

---

## 一、项目定位

DeepCompressor 是面向 **大语言模型(LLM)** 和 **扩散模型(Diffusion)** 的后训练量化(PTQ)框架。核心算法是 **SVDQuant**：将权重分解为 `W = W_q + A·B`，其中 `W_q` 为低精度量化权重，`A·B` 为全精度低秩补偿分支。

本仓库 `deepcompressor_vace` 是在原版 DeepCompressor 基础上，**适配 Wan2.1-VACE-14B 视频扩散模型**的分支。

---

## 二、目录结构速览

```
deepcompressor/
├── app/
│   ├── diffusion/          # ★ Diffusion 量化应用层
│   │   ├── ptq.py          # PTQ 主入口 (main函数)
│   │   ├── config.py       # DiffusionPtqRunConfig 顶层配置
│   │   ├── pipeline/config.py  # Pipeline 构建配置
│   │   ├── quant/          # 量化实现
│   │   │   ├── config.py   # DiffusionQuantConfig
│   │   │   ├── weight.py   # 权重量化 (SVDQuant低秩+RTN/GPTQ)
│   │   │   ├── activation.py # 激活量化 (forward hook)
│   │   │   ├── smooth.py   # SmoothQuant
│   │   │   └── rotate.py   # Hadamard/旋转变换
│   │   ├── nn/
│   │   │   ├── struct.py   # ★ 模型结构抽象 (DiTStruct/FluxStruct等)
│   │   │   └── patch.py    # 模块替换 (ConcatLinear/ShiftedLinear)
│   │   ├── dataset/
│   │   │   └── collect/
│   │   │       ├── calib.py      # 校准数据收集 (原版diffusers)
│   │   │       ├── calib_wan.py  # ★ VACE校准数据收集 (已完成)
│   │   │       └── utils.py      # CollectHook + ModelFnCollectHook
│   │   └── eval/config.py  # 评估配置
│   └── llm/                # LLM 量化 (参考用)
├── quantizer/              # 量化器核心
│   ├── impl/base.py        # QuantizerImpl 多步量化编排
│   ├── impl/simple.py      # simple_quantize 数学原子操作
│   └── kernel/
│       ├── rtn.py          # RTN (Round-to-Nearest)
│       └── gptq.py         # GPTQ (二阶Hessian优化)
├── calib/                  # 校准算法
│   ├── smooth.py           # SmoothQuant 校准器
│   ├── lowrank.py          # ★ SVDQuant 低秩校准器
│   └── search.py           # 搜索式校准基类
├── dataset/cache.py        # 逐层激活缓存 (BaseCalibCacheLoader)
├── data/                   # 量化数据结构 (scale/zero/codebook/dtype)
├── nn/                     # 通用patch (LowRankBranch等)
├── backend/nunchaku/       # Nunchaku推理引擎转换
└── utils/                  # 工具 (config/hooks/math/logging)
```

---

## 三、PTQ 主流程 (`ptq.py`)

```
命令行/YAML → DiffusionPtqRunConfig 解析
  ↓
main(config):
  1. pipeline = config.pipeline.build()        # 构建 diffusion pipeline
  2. model = DiffusionModelStruct.construct()   # 解析模型结构
  3. ptq(model, config.quant):
     a. rotate_diffusion()      → Hadamard/旋转变换 (可选)
     b. smooth_diffusion()      → SmoothQuant 平滑 (可选)
     c. quantize_weights()      → SVDQuant低秩分解 + RTN/GPTQ量化
     d. quantize_activations()  → forward hook 动态量化激活
  4. load_lora()                               # 加载 LoRA (可选)
  5. llm_ptq() for text encoders               # T5量化 (可选)
  6. eval.evaluate()                           # 生成&评估
```

---

## 四、关键量化算法

### 4.1 SVDQuant (lowrank.py + weight.py)

核心思想：`W = W_q + A·B`
- 交替优化：固定L优化W_q，固定W_q优化L
- `A` 为降维矩阵(共享)，`B` 为升维矩阵(按模块切分)
- 低秩分支通过 `LowRankBranch` 注册为 forward hook
- 典型 rank=16~64，在推理时额外计算 `output += A·B·input`

### 4.2 SmoothQuant (smooth.py)

数学变换：`Y = (X / s) · (s · W)`
- `s` 为 per-channel scale，通过 `s = α_base^α / β_base^β` 搜索
- `α_base` = 激活 span，`β_base` = 权重 span
- 可融合到前置 LayerNorm 或通过 `ActivationSmoother` hook 实现

### 4.3 旋转 (rotate.py)

- 对 QKV 投影输入通道应用 **Hadamard 变换**
- 对 V→O 投影应用 **随机正交旋转矩阵**
- 使权重分布更均匀，降低量化误差

### 4.4 量化执行 (quantizer/)

- **QuantizerImpl**：多步量化编排器 (per-tensor → per-channel → per-group)
- **RTN (rtn.py)**：`q = round(x/s + z), x' = (q-z)*s`，默认kernel
- **GPTQ (gptq.py)**：基于 Hessian 逆矩阵逐列量化 + 误差补偿
- **simple_quantize**：支持 int/float/exponent 量化 + STE

---

## 五、模型结构抽象 (`struct.py`)

`DiffusionModelStruct` 是统一接口，通过工厂模式自动识别：
- `UNetStruct` → SD/SDXL
- `DiTStruct` → PixArt/Sana/SD3
- `FluxStruct` → Flux (含 single_transformer_blocks)

每个 struct 提供：
- `iter_transformer_block_structs()` → 遍历所有 block
- `get_named_layers()` → 有序层字典
- `_get_default_key_map()` → 抽象名→实际层名映射

**当前问题**：`DiffusionModelStruct` 没有 `WanModel` 的对应结构体。VACE适配的核心工作之一就是为 WanModel 创建类似 `WanDiTStruct` 的结构体。

---

## 六、校准数据收集 — 已完成的工作

### 6.1 calib_wan.py

- 使用 `WanVideoPipeline.from_pretrained()` 显式创建 pipeline
- 用 `ModelConfig(path=...)` 指定本地模型权重路径
- 从 VACE-Benchmark `real.txt` (JSONL) 加载数据集
- 根据 task 类型 (T2V/MV2V/V2V/R2V) 构建不同的 VACE kwargs
- 通过 `sys.path.insert` 解决跨项目导入

### 6.2 utils.py — CollectHook 适配

**关键发现**：`model_fn_wan_video` 直接操作 `dit` 的子模块（blocks/head/patch_embedding），**不调用 `dit.forward()`**。因此 `register_forward_hook` 在 `dit` 上永远不会触发。

**解决方案**：新增 `ModelFnCollectHook` 类，包装 `pipeline.model_fn`：
- 拦截 `model_fn(**kwargs)` 调用
- 提取 `latents` → `input_args[0]`
- 提取 `timestep, context, clip_feature, y` → `input_kwargs`
- 输出包装为 `[output]` (list)
- Cache 格式与原 CollectHook 一致

---

## 七、VACE 量化下一步需要做的工作

### 7.1 模型结构适配 (最核心)

需要在 `struct.py` 中为 WanModel 创建结构体，或者在 `pipeline/config.py` 中注册 WanVideo pipeline 工厂。

WanModel 的结构：
- `patch_embedding`: Conv3d (in_dim → dim)
- `text_embedding`: MLP (text_dim → dim)
- `time_embedding`: MLP (freq_dim → dim)
- `time_projection`: Linear (dim → dim*6)
- `blocks`: N × DiTBlock
  - `self_attn`: SelfAttention (q/k/v/o + RoPE)
  - `cross_attn`: CrossAttention (q/k/v/o, 可选 img_input)
  - `norm1, norm2, norm3`: LayerNorm
  - `ffn`: Sequential(Linear→GELU→Linear)
  - `modulation`: Parameter (1, 6, dim)
- `head`: Head (norm + linear + modulation)
- `img_emb`: MLP (用于 CLIP feature)
- 附带 VACE 模型: `VaceWanModel`

关键差异 vs FluxStruct/DiTStruct：
- 3D 视频模型 (Conv3d patch embedding)
- 使用 RoPE 3D (f/h/w 三维频率)
- modulation 机制 (类似 AdaLN-Zero)
- VACE 分支在 block 之间插入 hint
- `model_fn_wan_video` 绕过 `dit.forward()` 直接编排

### 7.2 Pipeline 注册

在 `DiffusionPipelineConfig` 中注册 `WanVideoPipeline` 的构建方式：
```python
DiffusionPipelineConfig.register_pipeline_factory("wan2.1-vace-14b", build_wan_pipeline)
```

### 7.3 逐层激活缓存适配

`BaseCalibCacheLoader._iter_layer_activations()` 需要能够处理 WanModel 的 block 结构（DiTBlock 而非 diffusers 的 BasicTransformerBlock）。

### 7.4 量化配置

创建 VACE 专用的 YAML 配置文件，指定：
- 模型路径/dtype
- 量化精度 (W4A4/W4A8等)
- 跳过的层 (patch_embedding/head 等)
- SmoothQuant 参数
- SVDQuant 低秩 rank
- 校准数据路径

---

## 八、配置系统要点

- 使用 `omniconfig` 的 `@configclass` 装饰器
- 多 YAML 叠加 + 命令行覆盖
- `_key_map` 将抽象名映射到实际层名 (如 `"q_proj"` → `"attn1.to_q"`)
- 缓存路径自动按量化配置生成层级目录
- 输出目录自动包含量化配置的描述性名称

---

## 九、关键类型和数据结构

| 类型 | 说明 |
|------|------|
| `DiffusionModelStruct` | 模型结构抽象，提供统一遍历接口 |
| `DiffusionTransformerBlockStruct` | 单个 transformer block 的结构 |
| `DiffusionAttentionStruct` | Attention 模块封装 (q/k/v/o_proj) |
| `DiffusionFeedForwardStruct` | FFN 模块封装 (up/down_proj) |
| `LowRankBranch` | SVDQuant 低秩分支 (A·B，作为 hook 注册) |
| `ActivationSmoother` | SmoothQuant 激活缩放 hook |
| `QuantizerImpl` | 量化器实现 (管理 scale/zero/kernel) |
| `IOTensorsCache` | 模块 I/O 缓存 (用于校准) |
| `CollectHook` | 标准 forward hook 数据收集器 |
| `ModelFnCollectHook` | ★ model_fn 包装器 (VACE适配) |

---

## 十、已知限制和注意事项

1. **struct.py 不支持 WanModel**：当前只支持 diffusers 的 UNet/DiT/Flux 结构
2. **model_fn 绕过 forward**：VACE pipeline 的 `model_fn_wan_video` 不调用 `dit()`
3. **3D 视频模型**：原框架只处理 2D 图像模型，Conv3d/RoPE 3D 等需要适配
4. **VACE 分支**：VaceWanModel 在 block 间插入 hint，影响量化精度评估
5. **batch_size=1**：WanVideoPipeline 不支持批量处理
6. **显存**：14B 参数 + 视频生成，显存需求极大

---

# DiffSynth-Studio 框架 — 上下文反思

> 本项目使用 DiffSynth-Studio 作为模型部署框架（而非 diffusers）。校准阶段已通过 `WanVideoPipeline.from_pretrained()` 使用。

---

## 十一、DiffSynth-Studio 目录结构

```
diffsynth/
├── __init__.py               # 导出 core/*
├── configs/
│   ├── model_configs.py      # ★ MODEL_CONFIGS: hash→模型类映射注册表
│   └── vram_management_module_maps.py  # VRAM管理模块包装规则
├── core/
│   ├── attention/attention.py   # flash_attention 统一分发
│   ├── data/unified_dataset.py  # 训练用数据集
│   ├── device/                  # NPU 兼容
│   ├── gradient/                # 梯度检查点
│   ├── loader/
│   │   ├── config.py           # ★ ModelConfig 数据类
│   │   ├── file.py             # load_state_dict / hash_model_file
│   │   └── model.py            # ★ load_model / enable_vram_management
│   └── vram/
│       ├── disk_map.py         # DiskMap 惰性磁盘加载
│       ├── initialization.py   # skip_model_initialization
│       └── layers.py           # ★ 4级VRAM状态机 (AutoWrappedModule等)
├── diffusion/
│   ├── base_pipeline.py        # ★ BasePipeline + PipelineUnit 系统
│   ├── flow_match.py           # ★ FlowMatchScheduler
│   ├── training_module.py      # 训练模块
│   └── runner.py               # 训练启动器
├── models/
│   ├── wan_video_dit.py        # ★★ WanModel (DiTBlock/SelfAttn/CrossAttn/Head)
│   ├── wan_video_vace.py       # ★★ VaceWanModel (VACE条件编码器)
│   ├── wan_video_vae.py        # WanVideoVAE
│   ├── wan_video_text_encoder.py  # T5 文本编码器
│   ├── wan_video_image_encoder.py # CLIP 图像编码器
│   ├── model_loader.py         # ★ ModelPool (自动模型识别)
│   └── ...                     # Flux/Qwen/Z-Image 等其他模型
├── pipelines/
│   └── wan_video.py            # ★★ WanVideoPipeline 推理管线
└── utils/
    ├── state_dict_converters/
    │   ├── wan_video_dit.py    # WanModel 权重转换器
    │   └── wan_video_vace.py   # VACE 权重转换器
    ├── data/__init__.py        # VideoData/LowMemoryVideo/save_video
    └── lora/                   # LoRA 工具
```

---

## 十二、WanVideoPipeline 推理流程 (`pipelines/wan_video.py`)

### 12.1 Pipeline 架构

`WanVideoPipeline` 继承 `BasePipeline(torch.nn.Module)`，核心设计：

**模型槽位：**
- `tokenizer`: HuggingfaceTokenizer
- `text_encoder`: WanTextEncoder (T5)
- `image_encoder`: WanImageEncoder (CLIP)
- `dit` / `dit2`: WanModel (支持去噪后期切换第二个 DiT)
- `vae`: WanVideoVAE
- `vace` / `vace2`: VaceWanModel (VACE 条件模型)
- `motion_controller`, `vap`, `animate_adapter`, `audio_encoder`: 可选控制模型

**PipelineUnit 预处理链（21个 Unit 顺序执行）：**
1. `ShapeChecker` → 对齐 H/W/T 到整除因子
2. `NoiseInitializer` → 生成 `[1, z_dim, T', H', W']` 初始噪声
3. `PromptEmbedder` → T5 编码正/负 prompt → `context`
4. `InputVideoEmbedder` → V2V: VAE 编码输入视频 + 加噪
5. `ImageEmbedderCLIP` → CLIP 编码参考图 → `clip_feature`
6. `ImageEmbedderVAE` → I2V: 首帧 VAE latent → `y`
7. `VACE` → ★ VACE 条件处理（详见 12.3）
8. 其他：SpeedControl, FunControl, Camera, Animate, VAP, TeaCache, CfgMerger...

### 12.2 去噪循环

```python
for timestep in scheduler.timesteps:
    # 可选: 切换 dit→dit2 (在 switch_DiT_boundary 处)
    noise_pred_posi = self.model_fn(**models, **inputs_shared, **inputs_posi, timestep=timestep)
    # CFG: noise_pred = nega + cfg_scale * (posi - nega)
    latents = scheduler.step(noise_pred, timestep, latents)
```

### 12.3 VACE Unit 处理流程

```
vace_video + vace_video_mask
  ↓
inactive = video * (1 - mask)     # 未遮挡区域
reactive = video * mask            # 遮挡区域
  ↓
VAE encode → concat(inactive_latents, reactive_latents)  # channel dim
  ↓
mask 下采样 (空间8x, 时间4x)
  ↓
vace_context = concat(vace_video_latents, vace_mask_latents)  # channel=96
  ↓ (可选: reference_image latents 拼接到时间维度前端)
输出 vace_context 供 model_fn 使用
```

---

## 十三、model_fn_wan_video — 核心推理函数

这是 `WanVideoPipeline.model_fn` 指向的函数，**直接操作 dit 的子模块**而不调用 `dit.forward()`。

```
model_fn_wan_video(**models, **inputs, timestep):

  1. 时间嵌入:
     t = dit.time_embedding(sinusoidal(timestep))     → (B, dim)
     t_mod = dit.time_projection(t).unflatten(1,(6,dim)) → (B, 6, dim)

  2. 文本嵌入:
     context = dit.text_embedding(context)             → (B, S_text, dim)

  3. 构建输入 x:
     x = latents                                        → (B, C, F, H, W)
     if y:  x = cat([x, y], dim=1)                     → (B, C+C_y, F, H, W)
     if clip: context = cat([dit.img_emb(clip), context])

  4. Patchify + 展平:
     x = dit.patch_embedding(x)                         → (B, dim, f, h, w)
     x = rearrange('b c f h w -> b (f h w) c')         → (B, S, dim)

  5. 3D RoPE:
     freqs = cat([f_freqs, h_freqs, w_freqs])           → (S, 1, head_dim/2)

  6. VACE 前向 (一次性):
     vace_hints = vace(x, vace_context, context, t_mod, freqs)  → 8个(B,S,dim)

  7. ★ DiT Blocks 循环:
     for block_id, block in enumerate(dit.blocks):       # 40层 (14B)
         x = block(x, context, t_mod, freqs)
         if block_id in vace.vace_layers_mapping:
             x = x + vace_hints[i] * vace_scale          # 加性注入

  8. Head + Unpatchify:
     x = dit.head(x, t)                                 → (B, S, out_dim*patch_prod)
     x = dit.unpatchify(x, (f,h,w))                     → (B, out_dim, F, H, W)
```

**关键：model_fn 绕过 dit.forward()，直接调用子模块**。这对量化框架有重大影响——传统的 `register_forward_hook(dit)` 不会触发。

---

## 十四、WanModel 架构详解 (`models/wan_video_dit.py`)

### 14.1 整体结构

```
WanModel:
  patch_embedding: Conv3d(in_dim, dim, patch_size, stride=patch_size)
  text_embedding:  Sequential(Linear(text_dim, dim), GELU, Linear(dim, dim))
  time_embedding:  Sequential(Linear(freq_dim, dim), SiLU, Linear(dim, dim))
  time_projection: Sequential(SiLU, Linear(dim, dim*6))
  img_emb:         MLP(1280, dim)  [可选, has_image_input=True]
  blocks:          ModuleList([DiTBlock × num_layers])
  head:            Head(dim, out_dim, patch_size)
  freqs:           预计算的3D RoPE频率 (不可训练)
```

**VACE-14B 参数**: `dim=5120, num_layers=40, in_dim=16, out_dim=16, ffn_dim=8960*scale, num_heads=40, patch_size=(1,2,2), freq_dim=256, text_dim=4096`

### 14.2 DiTBlock 结构

```
DiTBlock:
  modulation: Parameter(1, 6, dim)  # 可学习调制基底
  norm1: LayerNorm(dim, affine=False)  # 无仿射参数 (由AdaLN控制)
  self_attn: SelfAttention
    q: Linear(dim, dim)
    k: Linear(dim, dim)
    v: Linear(dim, dim)
    o: Linear(dim, dim)
    norm_q: RMSNorm(dim)  # QK Normalization
    norm_k: RMSNorm(dim)
    attn: AttentionModule(num_heads)  # flash_attention 分发
  norm3: LayerNorm(dim, affine=True)  # cross-attn 前的标准 LN
  cross_attn: CrossAttention
    q, k, v, o: Linear(dim, dim)
    norm_q, norm_k: RMSNorm(dim)
    [可选] k_img, v_img, norm_k_img: 图像条件的额外 KV
  norm2: LayerNorm(dim, affine=False)  # FFN 前的无仿射 LN
  ffn: Sequential(Linear(dim, ffn_dim), GELU('tanh'), Linear(ffn_dim, dim))
```

**DiTBlock forward 流程：**
```
1. t_mod + self.modulation → chunk 为 6 个调制量:
   shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp

2. Self-Attention:
   x = x + gate_msa * self_attn(modulate(LN(x), shift_msa, scale_msa), freqs)

3. Cross-Attention (无调制):
   x = x + cross_attn(LN(x), context)

4. FFN:
   x = x + gate_mlp * ffn(modulate(LN(x), shift_mlp, scale_mlp))
```

### 14.3 Attention 实现

**SelfAttention**: Q/K 经 RMSNorm → 3D RoPE 旋转 → flash_attention
**CrossAttention**: Q/K 经 RMSNorm → 无 RoPE → 如有 `has_image_input`，对图像 token 做第二次 attention 并加性融合

**flash_attention 优先级**: Flash3 > Flash2 > SageAttn > PyTorch SDPA

### 14.4 Head 输出

```python
Head:
  modulation: Parameter(1, 2, dim)  # 2个调制量
  norm: LayerNorm(dim, affine=False)
  head: Linear(dim, out_dim * prod(patch_size))  # 如 dim→16*1*2*2=64

forward(x, t):  # t 是 time_embedding 的输出 (非 t_mod)
  shift, scale = chunk(t + self.modulation, 2)
  x = modulate(LN(x), shift, scale)
  x = head(x)
```

### 14.5 3D RoPE

维度分配：`head_dim = dim // num_heads`
- Frame: `head_dim - 2*(head_dim//3)` 维（吸收余数，略多）
- Height: `head_dim//3` 维
- Width: `head_dim//3` 维

每个维度独立预计算频率，在 forward 时按 `(f, h, w)` 展开并拼接。

---

## 十五、VaceWanModel 架构 (`models/wan_video_vace.py`)

### 15.1 整体结构

```
VaceWanModel:
  vace_patch_embedding: Conv3d(vace_in_dim=96, dim, patch_size, stride=patch_size)
  vace_blocks: ModuleList([VaceWanAttentionBlock × len(vace_layers)])
  vace_layers_mapping: dict {主模型block_id → VACE hint索引}
```

**VACE-14B**: `vace_layers=(0,5,10,15,20,25,30,35)`, 即 8 层 VACE block
**VACE-1.3B**: `vace_layers=(0,2,4,6,8,10,12,14,16,18,20,22,24,26,28)`, 即 15 层

### 15.2 VaceWanAttentionBlock (继承 DiTBlock)

在 DiTBlock 基础上增加：
- `before_proj`: Linear(dim, dim) — 仅第一个 block 有
- `after_proj`: Linear(dim, dim) — 每个 block 都有（生成 skip/hint）

```
forward(c, x, context, t_mod, freqs):
  if block_id == 0:
    c = before_proj(c) + x         # 与主模型 latent 融合
  else:
    all_c = unbind(c); c = pop last

  c = DiTBlock.forward(c, context, t_mod, freqs)  # 标准 DiT 处理
  c_skip = after_proj(c)                            # 生成 hint

  all_c += [c_skip, c]
  return stack(all_c)
```

### 15.3 VACE 与主模型的协作

```
VACE (一次性前向):
  vace_context → patchify → [VaceBlock_0, VaceBlock_5, ..., VaceBlock_35]
                              ↓ hint[0]    ↓ hint[1]  ...  ↓ hint[7]

主模型 (逐步注入):
  x → [Block_0 + hint[0]*s, Block_1, ..., Block_5 + hint[1]*s, ..., Block_35 + hint[7]*s, ...]
```

注入方式：`x = x + vace_hints[i] * vace_scale`（简单残差加法）

---

## 十六、模型加载与 VRAM 管理

### 16.1 ModelConfig → 模型加载流程

```
ModelConfig(path=...) → download_if_necessary()
  ↓
hash_model_file(path) → MD5(key_names + shapes)
  ↓
MODEL_CONFIGS 匹配 → 得到 model_class + converter + kwargs
  ↓
load_model():
  skip_model_initialization() → 跳过随机初始化
  load_state_dict() → safetensors 加载
  state_dict_converter() → 权重名转换
  model.load_state_dict(assign=True) → 就位
  enable_vram_management() → 包装为 AutoWrapped 层
```

**VACE 权重文件的双重解析**：同一个 safetensors 文件通过 hash 匹配两次：
- `WanVideoDiTStateDictConverter`: 过滤掉 `vace.*` 前缀 → 加载为 `WanModel`
- `VaceWanModelDictConverter`: 只保留 `vace.*` 前缀 → 加载为 `VaceWanModel`

### 16.2 四级 VRAM 状态机

| 状态 | 编号 | 说明 |
|------|------|------|
| offload | 0 | 参数在 CPU/磁盘，最低精度 |
| onload | 1 | 参数加载到中间设备 |
| preparing | 2 | 参数就绪，接近计算精度 |
| computation | 临时 | 计算时临时转换 (不改变 state) |

**关键包装类：**
- `AutoWrappedModule`: 包装任意 Module，支持 offload/onload/preparing/computation
- `AutoWrappedNonRecurseModule`: 仅管理直接参数（WanModel.DiTBlock 使用此类）
- `AutoWrappedLinear`: Linear 层专用，支持 FP8 计算 + LoRA

**Pipeline 级别调度**：`load_models_to_device(["dit", "vace"])` → 自动 onload 需要的模型 + offload 不需要的模型

### 16.3 WanModel 的 VRAM 规则

```python
# WanModel: DiTBlock 用 NonRecurse (内部不递归)
WanModel: { DiTBlock → AutoWrappedNonRecurseModule, MLP/Head → AutoWrappedModule, ... }
# VaceWanModel: DiTBlock 用普通递归包装
VaceWanModel: { DiTBlock → AutoWrappedModule, ... }
```

---

## 十七、FlowMatchScheduler

DiffSynth 唯一调度器，Wan 模板的关键参数：

```python
# Wan 模板
sigma_shift = 5.0  # 默认
sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)  # shift 变换
timesteps = sigmas * num_train_timesteps  # 映射到整数时间步
# 截断: timesteps = timesteps[:-1] (不含末尾0)

# 去噪步进 (velocity prediction):
prev_sample = sample + model_output * (sigma_next - sigma_current)

# 加噪:
noisy = (1 - sigma) * original + sigma * noise
```

---

## 十八、量化适配中的关键对应关系

### DeepCompressor struct → WanModel 的映射

| DeepCompressor 抽象概念 | WanModel 对应 |
|---|---|
| `DiffusionModelStruct` | 需新建 `WanDiTStruct` |
| `DiffusionTransformerBlockStruct` | `DiTBlock` |
| `DiffusionAttentionStruct.q_proj` | `block.self_attn.q` / `block.cross_attn.q` |
| `DiffusionAttentionStruct.k_proj` | `block.self_attn.k` / `block.cross_attn.k` |
| `DiffusionAttentionStruct.v_proj` | `block.self_attn.v` / `block.cross_attn.v` |
| `DiffusionAttentionStruct.o_proj` | `block.self_attn.o` / `block.cross_attn.o` |
| `DiffusionFeedForwardStruct.up_proj` | `block.ffn.0` (Linear(dim→ffn_dim)) |
| `DiffusionFeedForwardStruct.down_proj` | `block.ffn.2` (Linear(ffn_dim→dim)) |
| `pre_modules` | `patch_embedding, text_embedding, time_embedding, time_projection, img_emb` |
| `post_modules` | `head` |
| `named_layers` | `blocks` (每个 DiTBlock 为一层) |

### 量化目标模块

| 模块 | 类型 | 是否量化 | 说明 |
|---|---|---|---|
| `self_attn.q/k/v/o` | Linear | ✅ 权重+激活 | 核心量化目标 |
| `cross_attn.q/k/v/o` | Linear | ✅ 权重+激活 | |
| `cross_attn.k_img/v_img` | Linear | ✅ 权重 | 图像条件KV (可选跳过) |
| `ffn.0` (up) | Linear | ✅ 权重+激活 | |
| `ffn.2` (down) | Linear | ✅ 权重+激活 | GELU后激活为非负 |
| `patch_embedding` | Conv3d | ❌ 跳过 | 首层 |
| `head.head` | Linear | ❌ 跳过 | 末层 |
| `text_embedding` | Linear | ❌ 跳过 | 嵌入层 |
| `time_embedding/projection` | Linear | ❌ 跳过 | 条件嵌入 |
| `img_emb` | Linear | ❌ 跳过 | CLIP投影 |
| `norm_q/norm_k` (RMSNorm) | — | ❌ 跳过 | 归一化 |
| VACE 模型 | — | 待定 | VACE 是否需要量化取决于精度需求 |

### SmoothQuant 适配要点

- **Self-Attn QKV 共享输入**：`norm1(x)` 经 AdaLN 调制后同时送入 q/k/v → smooth 需要联合处理
- **Cross-Attn QKV 分离**：Q 来自 x，K/V 来自 context → 不同源的 smooth
- **FFN up_proj**：输入经 AdaLN 调制 → smooth 可融合到 norm2 (但 norm2 无仿射参数，需特殊处理)
- **FFN down_proj**：输入经 GELU → 非负激活，可用无符号量化

### 旋转适配要点

- **Self-Attn**: Q/K 已有 RoPE → 不适合对输入通道做 Hadamard（会与 RoPE 冲突）
- **V→O 投影对**: 可以安全应用旋转矩阵（V 无 RoPE）
- **Cross-Attn**: Q 有 RMSNorm 但无 RoPE → 需要评估是否适合旋转

---

## 十九、完整数据路径总结

### 校准阶段（已完成）

```
real.txt (JSONL) → load_dataset() → samples
  ↓
WanVideoPipeline.from_pretrained(local_paths) → pipeline
  ↓
ModelFnCollectHook 包装 pipeline.model_fn
  ↓
pipeline(prompt, vace_video, vace_mask, ...) × N samples
  ↓
caches: [{input_args: [latent], input_kwargs: {timestep, context, clip_feature, y}, outputs: [noise_pred]}]
  ↓
保存为 .pt 文件 → 校准数据集
```

### 量化阶段（待实现）

```
校准数据 .pt → BaseCalibCacheLoader
  ↓
WanDiTStruct.construct(pipeline.dit) → 结构抽象
  ↓
ptq():
  rotate_diffusion() → 旋转 (需评估与RoPE的兼容性)
  smooth_diffusion() → 平滑 (需适配AdaLN-Zero的norm结构)
  quantize_weights() → SVDQuant + RTN/GPTQ
  quantize_activations() → forward hook
  ↓
量化后的 WanModel → 保存/部署
```

---

## 二十、VACE 完整量化方案（主干 + 旁路）

### 20.1 现状

当前量化流程**只覆盖主干 `WanModel`**（通过 `WanDiTStruct` → `ptq()`），
旁路 `VaceWanModel` 完全未被量化。原因：

1. `wan_struct.py` 中只有 `WanDiTStruct` 包装 `WanModel`，无 `VaceWanModel` 的 Struct
2. `ptq_vace.py` 中 `_extended_default_construct` 只处理 `WanModel`
3. 校准数据只捕获了主干输入，未捕获 `vace_context`

### 20.2 核心思路

`VaceWanAttentionBlock` 继承自 `DiTBlock`，内部结构完全一致（self_attn, cross_attn, ffn, modulation, gate），
仅多了 `before_proj`（block 0）和 `after_proj`（所有 block）。
因此可以复用现有的 `WanTransformerBlockStruct` 思路，只需注册旁路的 Struct 结构。

```
主干 (WanModel)                      旁路 (VaceWanModel)
├── patch_embedding (Conv3d)         ├── vace_patch_embedding (Conv3d)
├── time_embedding (Sequential)      │   (无，共享主干)
├── text_embedding (Sequential)      │   (无，共享主干)
├── time_projection (Sequential)     │   (无，共享主干)
├── blocks[0..39] (DiTBlock)         ├── vace_blocks[0..7] (VaceWanAttentionBlock)
│   ├── self_attn                    │   ├── before_proj (Linear, 仅block 0)
│   ├── cross_attn                   │   ├── DiTBlock 内部结构 (同主干)
│   ├── norm1/2/3, ffn               │   │   ├── self_attn, cross_attn
│   ├── modulation, gate             │   │   ├── norm1/2/3, ffn, modulation, gate
│                                    │   └── after_proj (Linear)
└── head (Head)                      └── (无)
```

### 20.3 具体改动步骤

#### 步骤 1：扩展校准数据收集

修改 `ModelFnCollectHook`（`dataset/collect/utils.py`），额外捕获 `vace_context`。
或者直接 hook `VaceWanModel.forward()` 来捕获旁路完整输入 `(x, vace_context, context, t_mod, freqs)`。

#### 步骤 2：创建旁路 Struct（`wan_struct.py`）

- **`VaceTransformerBlockStruct`**：继承 `WanTransformerBlockStruct`，
  额外将 `before_proj` / `after_proj` 注册为 pre/post module structs
- **`VaceDiTStruct`**：类似 `WanDiTStruct` 但简化 —
  `input_embed` → `vace_patch_embedding`；`time_embed` / `text_embed` / `head` → `None`；`blocks` → `vace_blocks`
- 注册工厂：`VaceTransformerBlockStruct.register_factory(VaceWanAttentionBlock, ...)`

#### 步骤 3：创建旁路 CalibCacheLoader

`VaceCalibCacheLoader` 继承 `WanCalibCacheLoader`，处理旁路校准数据。
旁路输入 `(x, context, t_mod, freqs)` 来自主干 embedding 层（当前配置中 embedding 层在 `skips` 中不被量化），
可从主干校准数据重新计算。

#### 步骤 4：修改 PTQ 入口（`ptq_vace.py`）

在 `_extended_default_construct` 中增加 `VaceWanModel` 的处理。
在 `main_vace` 中先量化主干，再量化旁路：
```python
backbone_struct = DiffusionModelStruct.construct(pipeline.dit)
ptq(backbone_struct, config.quant, ...)

vace_struct = DiffusionModelStruct.construct(pipeline.vace)
ptq(vace_struct, config.quant_vace, ...)
```

#### 步骤 5：YAML 配置扩展

增加 `quant_vace` 配置项，可与主干共享或独立调参。

### 20.4 量化顺序的影响

当前配置 `skips: [embed]`（embedding 层不量化），旁路接收的 `x, context, t_mod, freqs` 始终全精度，
先后顺序不影响结果。两者量化互相独立，甚至可以并行在不同 GPU 上执行。

若未来取消 `embed` 的 skip，则需要用量化后主干的输出来校准旁路。

### 20.5 需要修改的文件

| 文件 | 改动 |
|------|------|
| `deepcompressor/app/diffusion/nn/wan_struct.py` | 新增 VaceTransformerBlockStruct, VaceDiTStruct |
| `deepcompressor/app/diffusion/ptq_vace.py` | 注册 VaceWanModel 工厂，增加旁路量化流程 |
| `deepcompressor/app/diffusion/dataset/collect/utils.py` | 增加 vace_context 捕获 |
| `deepcompressor/app/diffusion/dataset/collect/calib_wan.py` | 增加旁路校准数据收集 |
| `deepcompressor/app/diffusion/dataset/calib_wan_loader.py` | 新增 VaceCalibCacheLoader |
| `examples/diffusion/configs/model/wan2.1-vace-14b.yaml` | 增加旁路量化配置 |

### 20.6 注意事项

1. `before_proj` / `after_proj` 对 VACE 精度影响大，初期可跳过
2. 旁路块数少（14B 只有 8 块 vs 主干 40 块），校准样本可相应减少
3. 校准数据需来自同一推理过程，保证配对一致性

---

## 二十一、Smooth 量化性能分析

### 21.1 运行对比澄清

两次运行的**配置不同**，不仅仅是样本数不同：

| | 100 samples 运行 | 1 sample 运行 |
|---|---|---|
| 目录关键字 | `smooth.proj-w.static.lowrank` | `w.static.lowrank` |
| Smooth 阶段 | **有** (`smooth.proj`) | **无** |
| Low-rank 阶段 | 有（但还没开始） | 有（这是主要耗时）|
| 预估时间 | ~120h（仅 smooth 阶段） | ~50h（仅 lowrank 阶段）|

120h vs 50h 的"小差距"是因为比较的是**不同阶段**，而非同一操作不同样本数。

### 21.2 从日志精确计算 smooth 耗时

100 样本 smooth 运行中每个 block 的耗时：

| Block | 开始 | 结束 | 耗时 |
|-------|------|------|------|
| blocks.0 | 02:10:56 | 05:14:09 | 3h 03min |
| blocks.1 | 05:14:09 | 08:39:03 | 3h 25min |
| blocks.2 | 08:39:03 | 11:38:40 | 3h 00min |
| blocks.3 | 11:38:41 | 15:05:43 | 3h 27min |

**平均 ~3.2 小时/block × 40 blocks = 128 小时 ≈ 120 小时**

每个 block 内 6 个投影层的 smooth 耗时（以 block 0 为例）：

| 投影 | 耗时 |
|------|------|
| self_attn.qkv_proj | 29 min |
| self_attn.out_proj | 19 min |
| cross_attn.qkv_proj | 39 min |
| cross_attn.out_proj | 21 min |
| ffn.up_proj | 29 min |
| ffn.down_proj | 35 min |

### 21.3 根本原因：视频模型序列长度远超图像模型

Wan2.1-VACE-14B（480×832, 81帧）的序列长度：

```
latent: f = 21,  h = 60,  w = 104  (VAE 8x空间, 4x时间下采样)
patchify (1,2,2): tokens = 21 × 30 × 52 = 32,760
```

对比 PixArt-Sigma（512×512 图像）：tokens = 32 × 32 = 1,024

**视频模型序列长度是图像模型的 32 倍。**

Grid search 计算量（每个投影层）：
```
38 候选(num_grids=20, alpha-only + alpha+beta) × 100 样本 × forward(32760 × 5120)
= 3,800 次量化评估
```

每次评估：量化权重 + 量化激活 + matmul + GPU→CPU 传输 + 误差计算

### 21.4 优化建议

| 参数 | 当前值 | 建议值 | 预估加速 |
|------|--------|--------|---------|
| `sample_size` | -1 (全部100) | 20 | 5× |
| `num_grids` | 20 | 10 | ~2× |
| `sample_batch_size` | 1 | 4 (如显存允许) | ~2× |
| 校准分辨率 | 480×832 | 320×576 | ~3× |

综合优化后：120h → ~**6-13h**

### 21.5 完整 PTQ 流程预估时间

| 阶段 | 100 samples (原配置) | 优化后 |
|------|---------------------|--------|
| Smooth | ~128h | ~13h |
| Low-rank | 未开始，预估 >100h | ~50h (与样本数弱相关) |
| Weight calib | ~数小时 | ~数小时 |
| **总计** | ~230h+ | ~65h |
