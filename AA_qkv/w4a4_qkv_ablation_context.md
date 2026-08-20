# Wan2.1 W4A4 QKV 量化归因实验方案

## 1. 实验目标

验证 Wan2.1 使用 SVDQuant 量化到 W4A4 后出现的时序抖动和视频伪影，是否主要来自
self-attention 中 Q、K 或 V 投影层的量化误差。

本实验只做 W4A4 下的模块归因，不引入原论文中的 Jensen bias correction，也不修改
推理算法。

核心问题：

```text
Full W4A4 的视频退化，能否由 Q、K 或 V 中某一个分支单独复现？
```

## 2. 固定实验条件

所有实验必须保持以下条件一致：

- Wan2.1 模型版本；
- SVDQuant 的 W4A4 配置；
- prompt；
- random seed 和初始 noise；
- sampler、采样步数和 guidance；
- 分辨率、帧数和 batch size；
- VAE、文本编码器以及其他非目标模块。

建议先选择 2～3 个已知容易出现时序抖动的 prompt，减少实验成本。每个配置至少运行
一次完全相同的推理流程；重要结论最好使用多个 seed 复现。

## 3. Q/K/V 消融配置

所有候选配置都沿用同一个 W4A4 SVDQuant 流程。表中 `W4A4` 表示启用量化，
`FP` 表示仅对该分支临时跳过量化，用于归因。

| 配置 | Q | K | V | 用途 |
|---|---|---|---|---|
| FP baseline | FP | FP | FP | 全精度参考 |
| Full W4A4 | W4A4 | W4A4 | W4A4 | 复现当前问题 |
| Q-only | W4A4 | FP | FP | 判断 Q 量化影响 |
| K-only | FP | W4A4 | FP | 判断 K 量化影响 |
| V-only | FP | FP | W4A4 | 判断 V 量化影响 |
| QK-W4A4 | W4A4 | W4A4 | FP | 判断 QK attention score 误差 |
| KV-W4A4 | FP | W4A4 | W4A4 | 判断 K/V 联合影响 |

如果希望先做最小实验，只运行前五组：

```text
FP baseline
Full W4A4
Q-only
K-only
V-only
```

这里的 `FP` 不是重新训练的模型，而是在同一个推理程序和同一组权重下，临时绕过
对应 Q/K/V 分支的量化模块。

## 4. 需要保存的数据

每个实验配置保存：

1. 最终生成的视频；
2. 相同 seed 下的逐帧 latent；
3. self-attention 中 Q、K、V 的输出；
4. attention score 或 attention probability；
5. 每一层、每个 denoising timestep 的统计结果。

不需要一开始保存所有层。建议先选择：

- 前、中、后三个 DiT block；
- 早期、中期、后期三个 denoising timestep；
- 2～3 个有明显伪影的 prompt。

如果当前使用 FlashAttention、SageAttention 等 fused kernel，调试阶段建议临时使用
PyTorch eager/SDPA attention，以便保存 attention score 和 probability。

## 5. 评价指标

### 5.1 Q/K/V 输出误差

以 FP baseline 为参考，计算：

```text
err_Q = mean(abs(Q_w4a4 - Q_fp))
err_K = mean(abs(K_w4a4 - K_fp))
err_V = mean(abs(V_w4a4 - V_fp))
```

同时按 video frame 统计：

```text
err_X_frame[t] = mean(abs(X_w4a4[t] - X_fp[t]))
```

其中 `X` 分别为 Q、K、V。重点观察误差是否在时间维度上剧烈变化。

### 5.2 Attention map 差异

在相同输入下计算：

```text
S_fp = Q_fp K_fpᵀ / √d
S_q  = Q_q  K_qᵀ / √d

P_fp = softmax(S_fp)
P_q  = softmax(S_q)
```

然后统计：

```text
attention_MSE = mean((P_q - P_fp)^2)
attention_JSD = JSD(P_q, P_fp)
```

需要按以下维度分析：

- layer；
- attention head；
- denoising timestep；
- query token；
- video frame。

重点检查 attention 差异是否集中在相邻 video frame，或是否只集中在少数 head。

### 5.3 时序抖动指标

优先在 latent 空间计算相邻帧差异：

```text
temporal_diff = mean(abs(latent[t+1] - latent[t]))
```

再沿时间维做 FFT，计算高频能量比例：

```text
temporal_HF_ratio
```

该指标越高，通常意味着闪烁、局部抖动和细碎伪影越严重。

如果有参考视频，也可以额外记录 PSNR、SSIM、LPIPS；但本实验的主要判断依据是
时序指标和 attention 指标。

## 6. 可选的 score 扰动分析

为了判断 Q/K 量化是否直接改变 attention score，可以计算：

```text
ΔS = S_q - S_fp
```

统计：

```text
mean(ΔS)
mean(abs(ΔS))
std(ΔS)
max(abs(ΔS))
```

解释方式：

- `mean(ΔS)` 明显偏离 0：存在系统性 score 偏移；
- `mean(ΔS)` 接近 0，但 `std(ΔS)` 很大：更可能是 score 排序随机扰动；
- 某些 frame 的 `ΔS` 明显更大：可能造成时序 attention 不稳定。

这一部分只用于诊断，不做 score correction。

## 7. 结果判定规则

### 情况 A：K 是主要原因

如果同时满足：

```text
K-only 已出现明显视频抖动；
K-only 的 attention MSE/JSD 较大；
K-only 的 temporal_HF_ratio 明显升高；
```

则 K 量化很可能是主要问题。此时应进一步检查 K 的量化误差、scale、outlier 和
时间维度分布。

### 情况 B：Q/K 联合量化是主要原因

如果：

```text
QK-W4A4 接近 Full W4A4；
K-only 的退化明显较轻；
```

则更可能是 Q/K 联合量化造成 QK score 排序变化，而不是单独的 K 误差。

如果 `mean(ΔS)` 接近 0、但 `std(ΔS)` 很大，说明主要是随机 score 扰动。

### 情况 C：V 是主要原因

如果：

```text
V-only 已明显恶化；
但 V-only 的 attention map 与 FP 接近；
```

则问题主要来自 V 的 value 重建误差，而非 attention 分配变化。

### 情况 D：组合误差或其他模块问题

如果 Q-only、K-only、V-only 都不严重，但 Full W4A4 很严重，说明可能存在：

- QKV 与 output projection 的组合误差；
- 多层 attention 误差累积；
- FFN、norm 或 modulation 的量化误差；
- SVDQuant 的 scale 或低秩补偿问题。

此时再增加两个控制实验：

```text
QKV W4A4 + attention output projection FP
QKV W4A4 + FFN FP
```

## 8. 推荐执行顺序

### 阶段一：最小归因

运行：

```text
FP baseline
Full W4A4
Q-only
K-only
V-only
```

比较视频、latent temporal difference、temporal HF ratio、attention MSE/JSD。

### 阶段二：确认 QK 联合效应

如果 Q-only 和 K-only 都较轻，但 Full W4A4 较严重，运行：

```text
QK-W4A4
KV-W4A4
```

判断是否存在 Q/K 或 K/V 的组合误差。

### 阶段三：定位具体层和时间步

对退化最严重的配置，按 layer、head 和 timestep 排序：

```text
attention_MSE
attention_JSD
err_Q
err_K
err_V
temporal_HF_ratio
```

优先检查同时满足“中间误差大”和“最终时序指标恶化”的层。

## 9. 最终输出格式

建议每个配置生成一条记录：

```text
config:
  q: FP/W4A4
  k: FP/W4A4
  v: FP/W4A4

quality:
  video_path: ...
  temporal_diff: ...
  temporal_HF_ratio: ...

attention:
  attention_MSE: ...
  attention_JSD: ...
  score_mean_delta: ...
  score_std_delta: ...

projection_error:
  err_Q: ...
  err_K: ...
  err_V: ...
```

最后只需要根据以下关系判断根因：

```text
Full W4A4 是否可以由 Q-only、K-only 或 V-only 单独复现？
```

本方案不涉及论文中的 correction，也不改变 Wan2.1 的原始推理流程；它的目标只是
确定 W4A4 下时序抖动究竟来自 Q、K、V，还是来自它们的组合误差。

