# Quantized Keys Steal Attention：KV-Cache 压缩中的偏差校正

> 来源：`/data1/lyf/220110805-算法作业/pdf/qkv.pdf`  
> 论文题目：*Quantized Keys Steal Attention: Bias Correction for KV-Cache Compression in Video Diffusion*  
> 作者：Tuna Tuncer、Felix Becker、Thomas Pfeil  
> 文档用途：为 `deepcompressor_vace` 中的 Q/K/V 量化、KV-cache 压缩和注意力实现提供上下文。

## 1. 一句话总结

对历史 chunk 的 K（以及通常同时缓存的 V）做低比特量化时，K 的零均值量化噪声经过
`exp` 非线性会产生正的 Jensen 偏差，使 cached keys 在 softmax 中获得过多注意力；
论文在 softmax 前从每个 cached-key score 中减去一个可解析的 bias，几乎不增加显存，
即可恢复 BF16 级别的视频质量。

## 2. 问题背景

Chunk-wise autoregressive video diffusion 会把已经生成的 video chunks 的 K/V 写入
KV cache，并在后续 chunk 的每个 denoising step 中重复读取。缓存的长度持续增长，
因此 KV-cache 的显存、带宽和访问延迟成为瓶颈。将缓存的 K/V 从 BF16 压缩为 INT2/INT4
可以显著降低存储，但会导致：

- 画面质量下降、跨帧一致性变差；
- cached token 的 attention mass 增大，current chunk 获得的注意力减少；
- INT2 下这种现象尤其严重；INT4 下通常较小。

关键点是：即使量化误差在 score 空间中近似零均值，softmax 使用的是指数函数，
所以 `E[exp(δ)] > exp(E[δ])`。因此“零均值噪声不会改变期望”的直觉不适用于
attention partition sum。

## 3. 注意力和符号约定

对 query `q ∈ R^d`、key `k_i ∈ R^d`：

```text
s_i = qᵀ k_i / √d
p_i = exp(s_i) / Σ_j exp(s_j)
```

把 key 按 token 分为：

- `S`：cached keys，已经量化；
- `R`：current-chunk keys，保持全精度。

定义：

```text
Z_S = Σ_{i∈S} exp(s_i)
Z_R = Σ_{i∈R} exp(s_i)
Z = Z_S + Z_R
P_S = Z_S / Z
```

`P_S` 是 cached block 获得的总 attention mass。

对 cached key 的逐通道量化误差，设：

```text
k̂_i = k_i + ε_i
ε_{i,c} ~ Uniform(-Δ_{i,c}/2, Δ_{i,c}/2)
δ_i = qᵀ ε_i / √d
ŝ_i = s_i + δ_i
```

其中 `Δ_{i,c}` 是对应通道的量化 step。未量化的 current keys 不参与校正。

## 4. Jensen bias 的推导

量化后的 cached partition sum 为：

```text
Ẑ_S = Σ_{i∈S} exp(s_i + δ_i)
E[Ẑ_S] = Σ_{i∈S} exp(s_i) E[exp(δ_i)]
```

由于 `exp` 是凸函数，且 `E[δ_i]=0`：

```text
E[exp(δ_i)] ≥ exp(E[δ_i]) = 1
```

因此 `E[Ẑ_S] ≥ Z_S`，cached block 的 attention mass 被系统性放大。这就是论文
命名的 **Jensen bias** 或 **attention stealing**。可用以下量衡量 cached attention
mass 的偏移：

```text
ΔP_S = P̂_S - P_S
P̂_S = Ẑ_S / (Ẑ_S + Z_R)
```

实验中 `ΔP_S` 在未校正 INT2 下明显为正，校正后分布回到接近 0。

## 5. 核心校正公式

### 5.1 通用形式

只修改 cached scores：

```text
s̃_i = ŝ_i - b_i,  i ∈ S
s̃_i = s_i,       i ∈ R
```

要求校正后的每个 cached token 在期望意义下恢复原始贡献：

```text
exp(s_i - b_i) E[exp(δ_i)] = exp(s_i)
```

所以：

```text
b_i = log E[exp(δ_i)]
```

这保证了校正后 `E[Z̃_S] = Z_S`，而不需要重新训练模型或修改量化后的 K/V 数据。

### 5.2 均匀逐通道量化下的精确公式

因为各通道误差独立：

```text
E[exp(δ_i)]
= Π_c sinh(q_c Δ_{i,c}/(2√d)) / (q_c Δ_{i,c}/(2√d))
```

因此：

```text
b_i = Σ_c log(
  sinh(q_c Δ_{i,c}/(2√d))
  /
  (q_c Δ_{i,c}/(2√d))
)
```

实现精确公式时需要稳定处理 `sinh(x)/x` 在大 `|x|` 下的溢出，且每个 score
都要做 `O(d)` 运算，实际代价接近完整 attention。因此论文采用二阶近似。

### 5.3 推荐的 Taylor 近似

利用：

```text
log(sinh(α)/α) = α²/6 + O(α⁴)
α = q_c Δ_{i,c}/(2√d)
```

得到简单、稳定的校正：

```text
b_i ≈ (1 / (24d)) Σ_c q_c² Δ_{i,c}²
```

它等于 score-space noise variance 的一半：

```text
σ_i² = Var(δ_i) = (1 / (12d)) Σ_c q_c² Δ_{i,c}²
b_i ≈ σ_i² / 2
```

该近似在小量化 step 时非常准确；INT2 的极端 bitwidth 下可能过度估计 bias，
但论文实验显示仍能稳定改善端到端质量。

## 6. 分组量化和 QuaRot

### 6.1 group-wise per-token quantization

若每个 token 的 `d` 个通道分成 `G=d/g` 个 group，第 `j` 个 group 共享 step
`Δ_{i,j}`，定义 query 在该 group 上的平方范数：

```text
||q_j||² = Σ_{c∈group j} q_c²
```

则不必逐通道计算：

```text
b_i ≈ (1 / (24d)) Σ_{j=1}^G Δ_{i,j}² ||q_j||²
```

这正是论文实验使用的形式。若量化 scheme 是 per-channel，则可退化为逐通道公式；
若一个 step 在所有 token 间共享，则 bias `b` 对 cached tokens 相同，可一次计算并
广播；若是 per-token，则必须得到 token-dependent 的 `b_i`。

### 6.2 QuaRot / Hadamard rotation

若 K、Q 先经过正交 Hadamard 矩阵 `H`：

```text
k' = Hk, q' = Hq
```

由于正交变换保持完整向量范数，但一般会改变每个 group 的范数，校正应使用旋转后
query 的 group norm：

```text
b_i^(H) ≈ (1 / (24d)) Σ_j Δ_{i,j}² ||(Hq)_j||²
```

工程上等价于把公式中的 `q` 换成 `Hq`。论文实验表明该校正也适用于 QuaRot+RTN。

## 7. 推荐实现流程

对每个 attention layer、query block 或 query token：

1. 解量化 cached keys `K̂_S`；current keys `K_R` 保持原精度。
2. 计算 `S_S = Q K̂_Sᵀ / √d`，`S_R = Q K_Rᵀ / √d`。
3. 根据量化 metadata（每 token/group 的 step）和 Q 计算 `b_i`。
4. 只对 `S_S` 做 `S_S -= b`，不要修改 `S_R`。
5. 拼接 cached/current scores，执行原有 softmax 和 `P V`。

等价伪代码：

```python
K_cached = dequantize(K_cached_q)
S_cached = Q @ K_cached.transpose(-1, -2) / sqrt(d)
S_current = Q @ K_current.transpose(-1, -2) / sqrt(d)

# delta: [cached_tokens, groups] or broadcastable equivalent
# q_group_sq: [query_tokens, groups]
bias = (delta.square() * q_group_sq).sum(dim=-1) / (24 * d)
S_cached = S_cached - bias

S = concat([S_cached, S_current], dim=-1)
P = softmax(S, dim=-1)
O = P @ concat([V_cached, V_current], dim=-2)
```

当使用 FlexAttention 或类似 fused attention kernel 时，论文建议通过 score modifier
在 softmax 前在线减去 bias，避免物化与 attention score 同尺寸的 dense bias tensor。

## 8. 复杂度、显存与有效 bitwidth

对 group size `g`、维度 `d`、query 数 `Q`、cached key 数 `K`：

- 每个 query 计算 group-wise `||q_j||²`：`O(d)`；
- 每个 cached key 处理 quantization step：`O(KG)`；
- score-entry correction：`O(QKG)`；
- 相比标准 attention 的 `O(QKd)`，校正的主项小约 `g=d/G` 倍。

论文给出的总修正代价：

```text
O(Qd + K·G + Q·K·G)
```

存储方面，每个 group 还需存储一个 step。若量化值为每元素 `B` bit，metadata
约为每 group 24 bit，则有效 bitwidth：

```text
B_eff = B + 24/g
```

默认 `d=128, g=32, B=2` 时，`B_eff = 2.75` bit。校正本身不增加额外存储，
因为直接复用已有 quantization scale/step。

## 9. 实验结论与关键数字

论文在 MAGI-1、SkyReels-V2、HY-WorldPlay 三个 chunk-wise autoregressive video
diffusion 模型上评估，主要使用 group-wise per-token INT2 KV-cache，比较 RTN、
QuaRot+RTN、QVG，并报告 PSNR、SSIM、LPIPS 和 VBench。

代表性结果（`无校正 → 有校正`）：

- MAGI-1 + RTN：PSNR `23.17 → 24.08`，SSIM `0.799 → 0.828`，
  LPIPS `0.195 → 0.131`，VBench `76.67 → 77.86`。
- MAGI-1 + QuaRot+RTN：PSNR `17.10 → 22.97`，SSIM `0.630 → 0.801`，
  LPIPS `0.453 → 0.165`，VBench `70.24 → 78.02`。
- MAGI-1 + QVG：PSNR `23.01 → 25.29`，SSIM `0.826 → 0.856`，
  LPIPS `0.132 → 0.107`，VBench `77.81 → 78.23`。
- SkyReels-V2 + QuaRot+RTN：PSNR `19.20 → 20.42`，SSIM `0.708 → 0.784`，
  LPIPS `0.319 → 0.202`，VBench `71.44 → 78.58`。

总体规律：

- INT2 下 attention mass shift、attention JSD 和 attention-output MSE 都明显降低；
- PSNR、SSIM、LPIPS 以及 VBench 均稳定改善；
- 校正保持原有 group-size 控制的存储—质量 trade-off，只把曲线整体推向更好的质量；
- 在 MAGI-1 上，带校正的有效 bitwidth `2.19` 可超过无校正的 `4.38` bit 质量，
  对应约 50% 的 memory cost；
- INT4 的原始 bias 已较小，校正收益相应较温和。

作者还在 Llama-3.1-8B、Mistral-7B-Instruct-v0.3、Qwen2.5-32B-Instruct 的
partial-prefill 实验中观察到类似现象：INT2 KV-cache 会提高 teacher-forced NLL，
Taylor correction 通常降低该退化。该结果主要作为跨领域诊断，不等同于完整 LLM
benchmark。

## 10. 工程落地注意事项

1. **校正位置**：必须发生在 softmax 前的 attention score 空间；不能在 V 上做同样
   的减法，也不能把 bias 加到 current chunk。
2. **量化误差模型**：公式假设 round-to-nearest 误差近似独立、零均值、均匀分布。
   对 FP、MXFP、NVFP 或非均匀网格，需要按实际误差分布重新估计
   `log E[exp(δ)]`。
3. **Q 的来源**：校正依赖当前 full-precision query；QuaRot 场景应使用旋转后的 Q。
4. **metadata 对齐**：`Δ` 必须与 cached key 的 token/group 布局一致，尤其要确认
   `[batch, head, token, group]` 和 attention score 的广播维度。
5. **数值稳定性**：推荐二阶 Taylor 公式；若使用精确式，应使用稳定的 `log(sinh(x)/x)`
   实现，并处理 `x≈0` 的极限值 0。
6. **稀疏或滑动窗口 attention**：只对实际进入 softmax 的 cached keys 计算 bias；
   不要把已被 mask 的位置计入。
7. **缓存策略**：论文聚焦“历史 cached chunk 量化、当前 chunk 全精度”的结构。
   若 current chunk 也量化，需分别建模其 bias，不能直接套用本文的单侧校正。
8. **验证顺序**：建议先检查 `ΔP_S` 是否从正值回到 0 附近，再检查 attention JSD、
   output MSE，最后检查视频质量指标和显存收益。

## 11. 适合在本项目中优先检查的接口

实现或集成时，优先定位以下逻辑而不是修改模型权重：

- KV-cache 写入时保存的 quantization step/scale/zero-point；
- cached K 的 dequantize 和 QKᵀ score 计算；
- attention softmax 前的 score modifier / mask 融合点；
- group size、head dimension、token layout 与广播规则；
- fused attention kernel 是否允许对 cached/current 两段 score 使用不同 modifier。

最小可行验证可固定一个 layer 和一个 query，比较：

```text
mean(exp(ŝ_cached)) / mean(exp(s_cached))
ΔP_S before correction
ΔP_S after correction
```

若实现正确，INT2 下第一项应大于 1，校正后 `ΔP_S` 应显著接近 0；同时不应改变
current-token score 的数值。

## 12. 局限性

- 论文只系统研究 chunk-wise autoregressive video diffusion；标准单 token decoding
  中 cached/current 结构不同，校正可用空间更小。
- 误差独立、零均值、均匀分布是近似假设；当 attention 高度集中在少量 cached token
  或量化 step 很小时，实际偏差估计可能有噪声。
- Taylor 近似在极端 INT2 量化下会过估计，但经验上仍有效；精确公式计算成本更高。
- 论文只校正 attention score，未讨论不同非均匀量化器对误差分布的系统影响。

